"""Two separately budgeted arms, matched training tasks, checkpoint-based evaluation."""

import argparse
from dataclasses import asdict
import fcntl
import getpass
import json
import os
from pathlib import Path
import random
import time

from hayekmas.adapters.researchworld.runtime import (
    ResearchTrainer,
    create_research_agents,
    create_research_env,
    load_research_runtime_config,
    make_research_agent_deserializer,
)
from hayekmas.adapters.researchworld.env import load_research_tasks
from hayekmas.base.mas import HayekMAS
from hayekmas.utils.logger import logger
from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ResearchTaskEnv, load_tasks
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.policy import INSTRUCTIONS
from hayekmas.adapters.teams.replay import write_replay
from .campaign_client import CampaignClient, CampaignStop, atomic_json, append
from .team_checkpoint import dump_team, load_team


def repetition_gate(root, arm, repeat, deadline, now=None):
    """Check fixed protocol gates before an optional repeat; no model calls."""
    now = time.time() if now is None else now
    if repeat not in (1, 2):
        return "Only two additional repeats were declared"
    for candidate in ("original", "teams"):
        folder = root / candidate
        training = list(folder.glob("training/epoch-1/*/result.json"))
        evaluation = list(folder.glob("evaluation/epoch-1/*/result.json"))
        if len(training) != 40 or len(evaluation) != 19 or not (folder / "epoch-1.json").exists():
            return f"{candidate}: primary comparison incomplete"
    # A started repeat can resume in the final two hours; a new one cannot start then.
    started = (root / arm / "replications" / f"repeat-{repeat}" / "started.json").exists()
    if now >= deadline or (not started and deadline - now < 7200):
        return "Insufficient time to start a new repeat"
    return None


def second_epoch_started(out, state):
    """A start-time gate must not discard an epoch already begun before a restart."""
    return (
        state.get("training_completed", 0) > 40
        or (out / "epoch-2.json").exists()
        or any((out / "training/epoch-2").glob("*/attempt-*"))
    )


def run(arm, root, key, repetitions_only=False):
    plan = json.loads((root / "plan.json").read_text())
    out = root / arm
    out.mkdir(exist_ok=True)
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = out / "status.json"
    state = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {
            "arm": arm,
            "training_completed": 0,
            "eval_completed": {},
            "status": "starting",
            "phase": "training",
            "started_at": time.time(),
            "episodes": [],
            "recoveries": 0,
        }
    )
    state.update(pid=os.getpid(), status="running")
    state.pop("error", None)

    def tick():
        state["usage"] = client.usage()
        state["updated_at"] = time.time()
        atomic_json(status_path, state)

    client = CampaignClient(key, out, plan["deadline_unix"], cap=25, tick=tick)
    tick()
    logger.configure(verbose=False, log_dir=str(out), profile=arm)
    train_path = "third_party/benchmarks/frontier-science-research/data/research_train.jsonl"
    test_path = "third_party/benchmarks/frontier-science-research/data/research_test.jsonl"
    native_train = load_research_tasks([Path(train_path)], limit=40)
    native_test = load_research_tasks([Path(test_path)])
    assert len(native_train) == 40 and len(native_test) == 19
    assert not ({t.id for t in native_train} & {t.id for t in native_test})
    raw = json.loads(Path("global_configs/train_research.json").read_text())
    raw["mas"]["wakeup"].update(wakeup_model=None, wakeup_parallel_enabled=False)
    raw["mas"]["evaluation"]["periodic_test_enabled"] = False
    cfg = load_research_runtime_config(raw)
    trainer = ResearchTrainer(client, cfg)
    tcfg = TeamConfig(
        num_agents=12,
        rounds=80,
        max_steps=10,
        action_tokens=1024,
        solution_tokens=8192,
        judge_tokens=8192,
        inspect_tokens=1024,
        update_tokens=1024,
        summary_tokens=512,
        evidence_tokens=8192,
        environment_tokens=16384,
        context_tokens=65536,
        max_calls=100000,
    )

    class Policy:
        label = "llm"

        def __init__(self):
            self.client = client

        def respond(self, phase, agent, observation, max_tokens):
            return client.generate(
                json.dumps({"instruction": INSTRUCTIONS[phase], "observation": observation}, ensure_ascii=False),
                system_prompt=agent.get_system_prompt(),
                max_tokens=max_tokens,
                reasoning_effort="medium" if phase in {"discuss", "finalize"} else "none",
                kind=phase,
                json_mode=True,
            )

    policy = Policy()
    teams_train = load_tasks(train_path, "train")
    teams_test = load_tasks(test_path, "test")
    atomic_json(
        out / "protocol.json",
        {
            "arm": arm,
            "original_config": asdict(cfg.mas) if arm == "original" else None,
            "team_config": asdict(tcfg) if arm == "teams" else None,
            "model": "openai/gpt-6-luna",
            "solve_reasoning": "medium",
            "control_reasoning": "none",
            "output_cap": 8192,
            "visible_control_cap": 1024,
            "test_policy": "Frozen original economy"
            if arm == "original"
            else "Fresh trained copy per test, no formation/reflection; within-task negotiated contributions and transfers retained, then discarded.",
        },
    )

    def original_engine(checkpoint, training):
        if checkpoint.exists():
            engine = HayekMAS.load(str(checkpoint), agent_deserializer=make_research_agent_deserializer(client))
        else:
            engine = HayekMAS(cfg.mas)
            agents = create_research_agents(client)
            for a in agents:
                a.initialize(initial_wealth=cfg.mas.engine.initial_wealth)
                engine.population.add_agent(a)
            engine._initial_agents = list(agents)
        engine.set_agent_factory(trainer.create_agent_factory_good_birth(), trainer.create_agent_factory_bad_birth())
        engine.wakeup_llm_override = client.as_callable(max_tokens=1024, reasoning_effort="none", kind="wakeup")
        return engine.train() if training else engine.eval()

    def save_original(engine, path):
        tmp = path.with_suffix(".tmp.json")
        engine.save(str(tmp))
        tmp.replace(path)

    def episode(training, index, epoch, repeat=None):
        task = (native_train if training else native_test)[index]
        state.update(
            phase="replication" if repeat is not None else "training" if training else "evaluation",
            task_id=task.id,
            epoch=epoch,
            task_number=index + 1,
            active_since=time.time(),
        )
        if repeat is not None:
            state["repeat"] = repeat
        client.context = {
            "protocol_version": 3,
            "phase": state["phase"],
            "task_id": task.id,
            "epoch": epoch,
            "task_number": index + 1,
        }
        if repeat is not None:
            client.context["repeat"] = repeat
        client.phase_cap = 18 if training else 25
        # Reserve final hours for evaluation; never silently call partial training a full pass.
        if training and time.time() > plan["training_deadline_unix"]:
            raise CampaignStop("Training time window ended")
        directory = (
            out / "replications" / f"repeat-{repeat}" if repeat is not None else out / state["phase"] / f"epoch-{epoch}"
        ) / f"{index + 1:02d}-{task.id}"
        directory.mkdir(parents=True, exist_ok=True)
        done = directory / "result.json"
        if done.exists():
            return json.loads(done.read_text())
        checkpoint = out / ("training_checkpoint.json" if training else f"epoch-{epoch}.json")
        for retry in range(5):
            attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1}"
            attempt.mkdir()
            before = client.usage()
            random.seed(7 + (repeat * 100000 if repeat is not None else epoch * 1000) + index)
            tick()
            try:
                with logger.scoped_task_log(attempt / "trajectory.log"):
                    if arm == "original":
                        engine = original_engine(checkpoint, training)
                        initial = [a.serialize() for a in engine.population.get_all()]
                        env = create_research_env(
                            task,
                            llm_client=client,
                            reward_config=engine.config.reward,
                            use_judge=True,
                            judge_threshold=0.5,
                            max_steps=10,
                        )
                        env.llm_fn = client.as_callable(max_tokens=8192, reasoning_effort="medium", kind="judge")
                        engine.run_one_episode(env, step_ckpt_save_path=str(attempt / "steps"))
                        final = [a.serialize() for a in engine.population.get_all()]
                        if not training and final != initial:
                            raise CampaignStop("Frozen population mutation detected")
                        if training:
                            save_original(engine, directory / "checkpoint.json")
                        score = env.get_terminal_score()
                        answer = env.final_answer
                        result = {
                            "score": score,
                            "has_final_answer": bool(answer),
                            "passed": bool(env.is_successful()),
                            "steps": env.step_count,
                            "judge": env._last_judge_reason,
                            "termination": engine.last_termination_reason.value,
                            "replays": engine.last_episode_metrics.get("replays", 0),
                            "population_before": initial,
                            "population_after": final,
                            "actions": env.action_history,
                        }
                    else:
                        engine = (
                            load_team(json.loads(checkpoint.read_text()), tcfg, policy)
                            if checkpoint.exists()
                            else TeamMAS(tcfg, policy)
                        )
                        if repeat is not None:
                            engine.rng.seed(7 + repeat * 100000 + index)
                        if not training:
                            engine.invitations.clear()
                            # Fresh copy has the learned membership, wealth and strategies; no test feedback persists.
                        engine.events = []
                        engine.event_sink = lambda event: append(attempt / "events.jsonl", event)
                        engine.emit(
                            "initialized",
                            condition="dynamic",
                            backend="llm",
                            config=asdict(tcfg),
                            roster=engine.roster(),
                            agents=engine.state()["agents"],
                        )
                        initial = engine.state()["agents"]
                        env = ResearchTaskEnv(
                            (teams_train if training else teams_test)[index],
                            tcfg.reward,
                            lambda prompt: client.generate(prompt, max_tokens=8192, kind="judge"),
                        )
                        metric = engine.run_one_episode(env, formation=training, reflection=training)
                        write_replay(engine, attempt, "complete")
                        if training:
                            atomic_json(directory / "checkpoint.json", dump_team(engine))
                        submissions = [e for e in engine.events if e["event"] == "submission"]
                        finals = [e for e in submissions if e.get("final")]
                        answer = finals[-1]["answer"] if finals else None
                        score = env.get_terminal_score()
                        result = {
                            "score": score,
                            "has_final_answer": bool(finals),
                            "passed": bool(finals) and score >= 0.5,
                            "steps": metric["step_count"],
                            "invalid_actions": metric["invalid_actions"],
                            "population_before": initial,
                            "population_after": engine.state()["agents"],
                            "actions": submissions,
                            "metrics": metric,
                            "replay": str((attempt / "replay.html").relative_to(root)),
                        }
                    (attempt / "answer.md").write_text(
                        str(answer) if answer is not None else "No final answer produced.\n"
                    )
                    after = client.usage()
                    result.update(
                        protocol_version=3,
                        transport_revision=2,
                        request_start=before["requests"] + 1,
                        request_end=after["requests"],
                        arm=arm,
                        phase=state["phase"],
                        epoch=epoch,
                        index=index + 1,
                        task_id=task.id,
                        subject=task.subject,
                        usage={
                            k: after[k] - before[k] for k in ("cost_usd", "requests", "input_tokens", "output_tokens")
                        },
                        cumulative_cost=after["cost_usd"],
                        time=time.time(),
                        artifact=str(attempt.relative_to(root)),
                    )
                    if repeat is not None:
                        result.update(repeat=repeat, scheduling_seed=7 + repeat * 100000 + index)
                    atomic_json(done, result)
                    append(out / ("replication_episodes.jsonl" if repeat is not None else "episodes.jsonl"), result)
                    return result
            except CampaignStop as exc:
                state["last_error"] = str(exc)
                state["recoveries"] += 1
                atomic_json(
                    attempt / "interruption.json", {"error": str(exc), "time": time.time(), "usage": client.usage()}
                )
                tick()
                if not str(exc).startswith("Retries exhausted:") or retry == 4:
                    raise
                time.sleep(min(45, max(0, plan["deadline_unix"] - time.time())))
        raise CampaignStop("Episode retry limit exhausted")

    def primary():
        for epoch in (1, 2):
            # One complete matched pass is primary. A second pass requires both primary evaluations finished.
            if epoch == 2:
                if not second_epoch_started(out, state):
                    while time.time() < plan["second_epoch_start_deadline_unix"]:
                        if (root / "STOP").exists():
                            raise CampaignStop("STOP requested while waiting for the other arm")
                        other = root / ("teams" if arm == "original" else "original") / "status.json"
                        other_state = json.loads(other.read_text()) if other.exists() else {}
                        if other_state.get("eval_completed", {}).get("1") == 19:
                            break
                        state.update(status="waiting_for_other_arm", phase="between_epochs")
                        tick()
                        time.sleep(30)
                    else:
                        break
                    if client.usage()["cost_usd"] > 15:
                        break
                state["status"] = "running"
            for index in range(40):
                if state["training_completed"] >= (epoch - 1) * 40 + index + 1:
                    continue
                episode(True, index, epoch)
                task_id = native_train[index].id
                committed = out / "training" / f"epoch-{epoch}" / f"{index + 1:02d}-{task_id}" / "checkpoint.json"
                temporary = out / "training_checkpoint.tmp.json"
                temporary.write_bytes(committed.read_bytes())
                temporary.replace(out / "training_checkpoint.json")
                state["training_completed"] = (epoch - 1) * 40 + index + 1
                tick()
            epoch_path = out / f"epoch-{epoch}.json"
            if not epoch_path.exists():
                epoch_path.write_bytes((out / "training_checkpoint.json").read_bytes())
            completed = state["eval_completed"].get(str(epoch), 0)
            for index in range(completed, 19):
                episode(False, index, epoch)
                state["eval_completed"][str(epoch)] = index + 1
                tick()
            state["primary_complete"] = True
            tick()

    def repetitions():
        for repeat in (1, 2):
            reason = repetition_gate(root, arm, repeat, plan["deadline_unix"])
            if reason:
                state["replication_stop_reason"] = reason
                return
            directory = out / "replications" / f"repeat-{repeat}"
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / "started.json"
            if not marker.exists():
                atomic_json(
                    marker,
                    {
                        "time": time.time(),
                        "repeat": repeat,
                        "checkpoint": "epoch-1.json",
                        "seed_rule": "7 + repeat * 100000 + zero_based_task_index",
                    },
                )
            for index in range(19):
                episode(False, index, 1, repeat=repeat)
                state.setdefault("replication_completed", {})[str(repeat)] = index + 1
                tick()

    try:
        if repetitions_only:
            repetitions()
        else:
            primary()
        state.update(status="complete", phase="finished")
    except (CampaignStop, Exception) as exc:
        state.update(status="stopped", error=str(exc))
    finally:
        tick()
        logger.close()
        print(json.dumps({k: v for k, v in state.items() if k != "episodes"}, indent=2), flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--arm", choices=["original", "teams"], required=True)
    args = p.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter key (hidden): ")
    root = Path(args.root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("worker_mode") == "grader_recheck":
        from .campaign_grader import run_recheck

        run_recheck(root, args.arm, key)
        return
    if plan.get("worker_mode") == "diagnostic":
        if args.arm != "teams":
            raise ValueError("The training-only interface diagnostic uses only the team arm")
        from .campaign_diagnostic import run_diagnostic

        run_diagnostic(root, key)
        return
    run(args.arm, root, key, repetitions_only=plan.get("worker_mode") == "replication")


if __name__ == "__main__":
    main()
