"""Training-only interface diagnostic, separately gated and capped at $1.

This is never invoked by primary workers. It reuses the team arm's lock and
durable request ledger only after both primary studies have completed.
"""

from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

from hayekmas.adapters.teams.config import TeamConfig, TokenBudget
from hayekmas.adapters.teams.env import ResearchTaskEnv, load_tasks
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.policy import INSTRUCTIONS
from hayekmas.adapters.teams.replay import write_replay
from hayekmas.utils.logger import logger
from .campaign_client import CampaignClient, CampaignStop, atomic_json, append
from .campaign_report import episode_rows
from .team_checkpoint import dump_team, load_team


def interface_prompt(phase, observation, config, variant):
    """Clarify existing mechanics only; retain objectives, actions, and prices."""
    instruction = INSTRUCTIONS[phase]
    visible = dict(observation)
    if variant == "baseline":
        return instruction, visible
    if variant != "clarified":
        raise ValueError("Unknown diagnostic variant")
    if phase == "discuss":
        visible["discussion_window"] = {
            "total_turns_per_member": config.discussion_turns,
            "current_turn_one_based": observation["turn"] + 1,
            "your_turns_remaining_after_this": max(0, config.discussion_turns - observation["turn"] - 1),
            **observation.get("discussion_window", {}),
        }
        consequence = (
            "a separate final-answer window opens"
            if config.finalization_enabled
            else "the episode ends without a submitted action"
        )
        instruction += (
            " The discussion window is finite. If no member supplies an eligible nonempty candidate "
            f"before it closes, {consequence}. Text in message alone "
            "is discussion, not a candidate. A candidate may be a final answer or, when allowed, "
            "an intermediate public step. Choosing what to propose remains up to you."
        )
    elif phase in ("contribute", "negotiate"):
        instruction += (
            " A group is eligible to win the auction only if its total pledged contribution is positive. "
            "A zero personal pledge is permitted. Only the winning group's pledges are charged."
        )
    elif phase == "reflect":
        instruction += (
            " Paying the displayed reflection cost buys inspection of bounded own experience, "
            "teammate summaries, and public summaries from other teams, followed by an opportunity "
            "to edit your strategy for future tasks. The fee is deducted from your wealth and remains "
            "spent if the inspection or update is invalid. Reflection is optional."
        )
    return instruction, visible


def diagnostic_gate(root, deadline, now=None):
    now = time.time() if now is None else now
    for arm in ("original", "teams"):
        rows = episode_rows(root / arm)
        train = [r for r in rows if r["phase"] == "training" and r["epoch"] == 1]
        test = [r for r in rows if r["phase"] == "evaluation" and r["epoch"] == 1]
        if len(train) != 40 or len(test) != 19:
            return f"{arm}: primary study incomplete"
    started = (root / "teams/interface-diagnostic/started.json").exists()
    if now >= deadline or (not started and deadline - now < 2700):
        return "Insufficient time to start the diagnostic"
    return None


def study_budget(path, usage):
    """Persist the original incremental cap so a restart cannot replenish it."""
    if path.exists():
        return json.loads(path.read_text())
    data = {
        "time": time.time(),
        "starting_billed_usd": usage["cost_usd"],
        "maximum_increment_usd": 1.0,
        "absolute_cap_usd": min(25.0, usage["cost_usd"] + 1.0),
        "variants": ["baseline", "clarified"],
        "tasks": "First three bundled training tasks, in the original order",
        "design": "Fresh population per variant; three tasks sequentially train that population. Same configuration and initial RNG seed; branching decisions may change later random draws. Sequential execution is exploratory, not an independent replication.",
    }
    atomic_json(path, data)
    return data


def run_diagnostic(root, key):
    plan = json.loads((root / "plan.json").read_text())
    out = root / "teams"
    study = out / "interface-diagnostic"
    study.mkdir(exist_ok=True)
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((out / "status.json").read_text())
    state.update(pid=os.getpid(), status="running", phase="diagnostic")
    state.pop("error", None)

    def tick():
        state.update(usage=client.usage(), updated_at=time.time())
        atomic_json(out / "status.json", state)
        atomic_json(
            study / "status.json",
            {
                k: state.get(k)
                for k in (
                    "pid",
                    "status",
                    "phase",
                    "diagnostic_variant",
                    "task_number",
                    "task_id",
                    "usage",
                    "error",
                    "updated_at",
                )
            },
        )

    client = CampaignClient(key, out, plan["deadline_unix"], cap=25, tick=tick)
    logger.configure(verbose=False, log_dir=str(study), profile="interface-diagnostic")
    try:
        reason = diagnostic_gate(root, plan["deadline_unix"])
        if reason:
            raise CampaignStop(reason)
        budget = study_budget(study / "started.json", client.usage())
        client.phase_cap = budget["absolute_cap_usd"]
        if not (study / "primary-manifest.json").exists():
            protected = []
            for arm in ("original", "teams"):
                folder = root / arm
                protected.extend(folder.glob("training/epoch-*/*/result.json"))
                protected.extend(folder.glob("evaluation/epoch-*/*/result.json"))
                protected.extend(folder.glob("epoch-*.json"))
                protected.extend(folder.glob("training_checkpoint.json"))
            atomic_json(
                study / "primary-manifest.json",
                {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected},
            )
        config_values = json.loads((out / "protocol.json").read_text())["team_config"]
        # Historical protocols predate the final-answer stage. Do not silently
        # change their diagnostic intervention when reading a saved protocol.
        config_values.setdefault("finalization_enabled", False)
        config = TeamConfig(**config_values)
        tokens = TokenBudget()
        task_path = (
            Path(__file__).resolve().parents[2]
            / "third_party/benchmarks/frontier-science-research/data/research_train.jsonl"
        )
        tasks = load_tasks(str(task_path), "train")[:3]
        atomic_json(
            study / "protocol.json",
            {
                "config": asdict(config),
                "task_ids": [t.id for t in tasks],
                "intervention": "Describe existing finite windows, submission consequences, auction eligibility, and optional reflection access. No role assignment or forced behavior.",
                "seed_rule": "7 + 1000 + zero_based_task_index for global scheduling; engine retains configured seed and its within-variant checkpoint RNG",
            },
        )

        class Policy:
            label = "llm"

            def __init__(self, variant):
                self.variant, self.client = variant, client
                self.emit = None

            def respond(self, phase, agent, observation, max_tokens):
                instruction, visible = interface_prompt(phase, observation, config, self.variant)
                prompt = json.dumps({"instruction": instruction, "observation": visible}, ensure_ascii=False)
                if tokens.count(agent.get_system_prompt()) + tokens.count(prompt) > config.context_tokens:
                    raise CampaignStop("Clarified diagnostic observation exceeds context limit")
                if self.variant == "clarified" and instruction != INSTRUCTIONS[phase]:
                    self.emit(
                        "interface_clarification",
                        phase=phase,
                        agent=agent.name,
                        instruction=instruction,
                        discussion_window=visible.get("discussion_window"),
                    )
                return client.generate(
                    prompt,
                    system_prompt=agent.get_system_prompt(),
                    max_tokens=max_tokens,
                    reasoning_effort="medium" if phase in {"discuss", "finalize"} else "none",
                    kind=phase,
                    json_mode=True,
                )

        for variant in ("baseline", "clarified"):
            policy = Policy(variant)
            previous = None
            for index, task in enumerate(tasks):
                directory = study / variant / f"{index + 1:02d}-{task.id}"
                directory.mkdir(parents=True, exist_ok=True)
                state.update(diagnostic_variant=variant, task_number=index + 1, task_id=task.id)
                client.context = {
                    "protocol_version": 3,
                    "phase": "diagnostic_training",
                    "study": "interface_diagnostic",
                    "variant": variant,
                    "task_id": task.id,
                    "task_number": index + 1,
                    "epoch": 1,
                }
                tick()
                if (directory / "result.json").exists():
                    previous = directory / "checkpoint.json"
                    continue
                for retry in range(3):
                    attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1}"
                    attempt.mkdir()
                    before = client.usage()
                    random.seed(1007 + index)
                    try:
                        with logger.scoped_task_log(attempt / "trajectory.log"):
                            engine = (
                                load_team(json.loads(previous.read_text()), config, policy)
                                if previous
                                else TeamMAS(config, policy)
                            )
                            engine.events = []
                            engine.event_sink = lambda event: append(attempt / "events.jsonl", event)
                            policy.emit = engine.emit
                            initial = engine.state()["agents"]
                            engine.emit(
                                "initialized",
                                condition="dynamic",
                                backend="llm",
                                config=asdict(config),
                                roster=engine.roster(),
                                agents=initial,
                            )
                            env = ResearchTaskEnv(
                                task,
                                config.reward,
                                lambda prompt: client.generate(prompt, max_tokens=8192, kind="judge"),
                            )
                            metric = engine.run_one_episode(env, formation=True, reflection=True)
                            write_replay(engine, attempt, "complete")
                            atomic_json(directory / "checkpoint.json", dump_team(engine))
                            final = [e for e in engine.events if e["event"] == "submission" and e.get("final")]
                            score = env.get_terminal_score()
                            (attempt / "answer.md").write_text(
                                str(final[-1]["answer"]) if final else "No final answer produced.\n"
                            )
                            after = client.usage()
                            atomic_json(
                                directory / "result.json",
                                {
                                    "variant": variant,
                                    "task_id": task.id,
                                    "index": index + 1,
                                    "subject": task.subject,
                                    "time": time.time(),
                                    "score": score,
                                    "passed": bool(final) and score >= 0.5,
                                    "has_final_answer": bool(final),
                                    "steps": metric["step_count"],
                                    "metrics": metric,
                                    "population_before": initial,
                                    "population_after": engine.state()["agents"],
                                    "request_start": before["requests"] + 1,
                                    "request_end": after["requests"],
                                    "artifact": str(attempt.relative_to(root)),
                                    "replay": str((attempt / "replay.html").relative_to(root)),
                                    "protocol_version": 3,
                                    "transport_revision": 2,
                                },
                            )
                            previous = directory / "checkpoint.json"
                            break
                    except CampaignStop as exc:
                        atomic_json(
                            attempt / "interruption.json",
                            {"error": str(exc), "time": time.time(), "usage": client.usage()},
                        )
                        if not str(exc).startswith("Retries exhausted:") or retry == 2:
                            raise
                        time.sleep(min(30, max(0, plan["deadline_unix"] - time.time())))
        state.update(status="complete", phase="finished")
    except (CampaignStop, Exception) as exc:
        state.update(status="stopped", error=str(exc))
    finally:
        tick()
        logger.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        print(json.dumps({"status": state["status"], "error": state.get("error"), "usage": client.usage()}), flush=True)
