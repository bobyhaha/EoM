"""Isolated, resumable evaluation of the answer-delivery fix on 19 saved tasks."""

import argparse
from dataclasses import asdict
import fcntl
import getpass
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ResearchTaskEnv, load_tasks
from hayekmas.adapters.teams.policy import INSTRUCTIONS
from hayekmas.adapters.teams.replay import write_replay
from hayekmas.utils.logger import logger
from .campaign_client import CampaignClient, CampaignStop, append, atomic_json
from .team_checkpoint import load_team


REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, source, *, protocol_variant="finalization", cap_usd=5.0):
    root, source = Path(root).resolve(), Path(source).resolve()
    if protocol_variant not in {"finalization", "reviewed", "sealed_only", "review_only"}:
        raise ValueError("Unknown protocol variant")
    if type(cap_usd) not in (int, float) or not math.isfinite(cap_usd) or not 0 < cap_usd <= 5:
        raise ValueError("cap_usd must be positive and at most the authorized $5 ceiling")
    protocol = json.loads((source / "teams/protocol.json").read_text())
    overrides = {"finalization_enabled": True,
                 "bidding_mode": "sealed" if protocol_variant in {"reviewed", "sealed_only"} else "negotiated",
                 "collaboration_mode": "reviewed" if protocol_variant in {"reviewed", "review_only"} else "discussion"}
    config = TeamConfig(**{**protocol["team_config"], **overrides})
    dataset = REPO / "third_party/benchmarks/frontier-science-research/data/research_test.jsonl"
    tasks = load_tasks(dataset, "test")
    if len(tasks) != 19:
        raise ValueError("Expected all 19 bundled test tasks")
    checkpoint = source / "teams/epoch-1.json"
    population = json.loads(checkpoint.read_text())
    if len(population["agents"]) != config.num_agents or population["counters"]["round"] != 40:
        raise ValueError("Expected the completed 40-task team checkpoint")
    baseline = [json.loads(p.read_text()) for p in sorted((source / "teams/evaluation/epoch-1").glob("*/result.json"))]
    if [r["task_id"] for r in baseline] != [t.id for t in tasks]:
        raise ValueError("Saved primary results do not match the prescribed task order")
    root.mkdir(parents=True, exist_ok=False)
    (root / "teams").mkdir()
    shutil.copy2(checkpoint, root / "checkpoint.json")
    shutil.copy2(dataset, root / "tasks.jsonl")
    sources = {}
    for path in sorted((REPO / "hayekmas").rglob("*.py")):
        relative = str(path.relative_to(REPO))
        copy = root / "source" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        sources[relative] = sha(path)
    plan = {
        "study": f"teams-{protocol_variant}-19", "protocol_variant": protocol_variant,
        "prepared_at": time.time(),
        "source_campaign": str(source), "model": protocol["model"],
        "team_config": asdict(config), "cap_usd": cap_usd, "time_limit_seconds": 6 * 3600,
        "task_ids": [t.id for t in tasks], "checkpoint_sha256": sha(root / "checkpoint.json"),
        "dataset_sha256": sha(root / "tasks.jsonl"), "source_sha256": sources,
        "test_policy": "Fresh saved epoch-1 population per task; no formation, reflection or retraining. "
        "Within-task money flows remain active. Saved checkpoint RNG matches primary evaluation; "
        "global scheduling seed is 1007 + zero-based task index. Finalization is enabled. "
        f"Bidding: {config.bidding_mode}; collaboration: {config.collaboration_mode}.",
        "interpretation": "Development evaluation on the 19 previously inspected test tasks. "
        "Not a fresh held-out efficacy estimate. No original-arm rerun. "
        "Equal spending ceilings do not imply equal actual inference compute.",
        "python": sys.version,
    }
    atomic_json(root / "plan.json", plan)
    atomic_json(root / "baseline.json", baseline)
    report(root, {"status": "awaiting_credential", "completed": 0, "usage": {}})
    return plan


def report(root, state):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    title = "Teams: " + plan.get("protocol_variant", "finalization").replace("_", " ") + " evaluation"
    rows = [json.loads(p.read_text()) for p in sorted((root / "teams/evaluation").glob("*/result.json"))]
    baseline = {r["task_id"]: r for r in json.loads((root / "baseline.json").read_text())}
    state = {**state, "completed": len(rows), "updated_at": time.time()}
    atomic_json(root / "status.json", state)
    complete = len(rows) == 19
    summary = {
        **state, "planned_tasks": 19,
        "submitted": sum(r["has_final_answer"] for r in rows),
        "passes": sum(r["passed"] for r in rows),
        "mean_score": sum(r["score"] for r in rows) / 19 if complete else None,
        "completed_task_mean": sum(r["score"] for r in rows) / len(rows) if rows else None,
        "baseline_mean_on_completed_tasks": sum(baseline[r["task_id"]]["score"] for r in rows) / len(rows) if rows else None,
        "interpretation": plan["interpretation"],
    }
    atomic_json(root / "summary.json", summary)
    body = []
    for row in rows:
        old = baseline[row["task_id"]]
        body.append(
            f'<tr><td>{row["index"]}</td><td>{escape(row["subject"])}</td>'
            f'<td>{old["score"]:.3f}</td><td>{row["score"]:.3f}</td>'
            f'<td>{"Yes" if row["has_final_answer"] else "No"}</td>'
            f'<td>{row["finalization_windows"]}</td><td><a href="{escape(row["replay"], quote=True)}">Replay</a> · '
            f'<a href="{escape(row["answer_path"], quote=True)}">Answer</a></td></tr>'
        )
    usage = state.get("usage", {})
    mean = f'{summary["mean_score"]:.3f}' if complete else "Pending until all 19 finish"
    html = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        f'<title>{escape(title)}</title><style>'
        'body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}'
        'table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}'
        f'a{{color:#1759aa}}</style><h1>{escape(title)}</h1>'
        f'<p>Status: <strong>{escape(state["status"])}</strong> · Completed: {len(rows)}/19 · '
        f'Submitted: {summary["submitted"]} · Threshold passes: {summary["passes"]}</p>'
        f'<p>Full-run mean score: <strong>{mean}</strong>. '
        f'Billed: ${usage.get("cost_usd", 0):.4f}; reserved: ${usage.get("unconfirmed_usd", 0):.4f}; '
        f'ceiling: ${plan["cap_usd"]:.4f}.</p>'
        f'<p>{escape(state.get("error", ""))}</p><p>{escape(plan["interpretation"])}</p>'
        f'<p>{escape(plan.get("startup_diagnostic", {}).get("note", ""))}</p>'
        f'<p>Same trained population and model. Bidding: {escape(plan["team_config"].get("bidding_mode", "negotiated"))}; '
        f'collaboration: {escape(plan["team_config"].get("collaboration_mode", "discussion"))}. '
        'Missing submissions count as zero.</p>'
        '<table><thead><tr><th>Task</th><th>Subject</th><th>Old primary</th><th>Updated</th>'
        '<th>Submitted</th><th>Recovery windows</th><th>Artifacts</th></tr></thead><tbody>'
        + ''.join(body) + '</tbody></table><p><a href="summary.json">Summary data</a> · '
        '<a href="plan.json">Protocol</a></p>'
    )
    temporary = root / "index.html.tmp"
    temporary.write_text(html)
    temporary.replace(root / "index.html")
    return summary


def run(root, key):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    if sha(root / "checkpoint.json") != plan["checkpoint_sha256"] or sha(root / "tasks.jsonl") != plan["dataset_sha256"]:
        raise ValueError("Frozen checkpoint or dataset changed")
    if any(sha(REPO / name) != digest for name, digest in plan["source_sha256"].items()):
        raise ValueError("Source changed since preparation; use a new evaluation directory")
    out = root / "teams"
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((root / "status.json").read_text())
    if state["status"] == "complete":
        lock.close()
        return state
    started_path = root / "started.json"
    if not started_path.exists():
        atomic_json(started_path, {"started_at": time.time(), "deadline": time.time() + plan["time_limit_seconds"]})
    deadline = json.loads(started_path.read_text())["deadline"]
    state.update(status="running", pid=os.getpid())
    state.pop("error", None)

    def tick():
        state["usage"] = client.usage()
        report(root, state)

    client = CampaignClient(key, out, deadline, cap=plan["cap_usd"], tick=tick)
    if client.model != plan["model"]:
        lock.close()
        raise ValueError("Client model does not match frozen protocol")
    config = TeamConfig(**plan["team_config"])
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    tasks = load_tasks(root / "tasks.jsonl", "test")
    logger.configure(verbose=False, log_dir=str(out), profile="teams-finalization")

    class Policy:
        label = "llm"

        def __init__(self):
            self.client = client

        def respond(self, phase, agent, observation, max_tokens):
            return client.generate(
                json.dumps({"instruction": INSTRUCTIONS[phase], "observation": observation}, ensure_ascii=False),
                system_prompt=agent.get_system_prompt(), max_tokens=max_tokens,
                reasoning_effort="medium" if phase in {"discuss", "finalize", "draft", "independent_check", "review", "revise"} else "none",
                kind=phase, json_mode=True,
            )

    try:
        tick()
        for index, task in enumerate(tasks):
            directory = out / "evaluation" / f"{index + 1:02d}-{task.id}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "result.json").exists():
                continue
            state.update(task_number=index + 1, task_id=task.id)
            client.context = {"study": plan["study"], "phase": "evaluation", "task_number": index + 1, "task_id": task.id}
            prior_attempts = len(list(directory.glob("attempt-*")))
            if prior_attempts >= 3:
                raise CampaignStop("Task attempt limit exhausted")
            for number in range(prior_attempts + 1, 4):
                attempt = directory / f"attempt-{number}"
                attempt.mkdir()
                before = client.usage()
                try:
                    random.seed(1007 + index)
                    engine = load_team(json.loads(json.dumps(checkpoint)), config, Policy())
                    engine.invitations.clear()
                    engine.events = []
                    engine.event_sink = lambda e: append(attempt / "events.jsonl", e)
                    engine.emit("initialized", condition=config.condition, backend="llm", config=asdict(config),
                                roster=engine.roster(), agents=engine.state()["agents"])
                    initial = engine.state()["agents"]
                    env = ResearchTaskEnv(task, config.reward, engine.judge)
                    with logger.scoped_task_log(attempt / "trajectory.log"):
                        metric = engine.run_one_episode(env, formation=False, reflection=False)
                    write_replay(engine, attempt, "complete")
                    answer = env.native.final_answer
                    answer_path = attempt / "answer.txt"
                    answer_path.write_text(answer or "No final answer produced.\n")
                    after = client.usage()
                    result = {
                        "index": index + 1, "task_id": task.id, "subject": task.subject,
                        "score": env.get_terminal_score(), "has_final_answer": bool(answer),
                        "passed": bool(answer) and env.get_terminal_score() >= 0.5,
                        "judge": env.native._last_judge_reason, "metrics": metric,
                        "population_before": initial, "population_after": engine.state()["agents"],
                        "finalization_windows": sum(e["event"] == "finalization_started" for e in engine.events),
                        "review_windows": sum(e["event"] == "review_started" for e in engine.events),
                        "review_fallbacks": sum(e["event"] == "review_fallback" for e in engine.events),
                        "funding_repairs": sum(e["event"] == "funding_repair_started" for e in engine.events),
                        "protocol_variant": plan.get("protocol_variant", "finalization"),
                        "replay": str((attempt / "replay.html").relative_to(root)),
                        "answer_path": str(answer_path.relative_to(root)),
                        "usage": {k: after[k] - before[k] for k in ("cost_usd", "requests", "input_tokens", "output_tokens")},
                        "completed_at": time.time(),
                    }
                    atomic_json(directory / "result.json", result)
                    tick()
                    print(json.dumps({"task": index + 1, "score": result["score"], "submitted": bool(answer),
                                      "billed_usd": after["cost_usd"]}), flush=True)
                    break
                except CampaignStop as exc:
                    atomic_json(attempt / "interruption.json", {"error": str(exc), "usage": client.usage()})
                    if not str(exc).startswith("Retries exhausted:") or number == 3:
                        raise
                    time.sleep(min(20, max(0, deadline - time.time())))
        if sha(root / "checkpoint.json") != plan["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during evaluation")
        state.update(status="complete", finished_at=time.time())
        (root / "STOP").write_text("Completed all 19 tasks; further model calls stopped.\n")
    except (CampaignStop, Exception) as exc:
        state.update(status="stopped", error=str(exc).replace(key, "[redacted]"))
    finally:
        tick()
        logger.close()
        lock.close()
    return json.loads((root / "summary.json").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--prepare-from")
    p.add_argument("--protocol", choices=("finalization", "reviewed", "sealed_only", "review_only"),
                   default="finalization", help="Protocol to freeze when preparing a new run")
    p.add_argument("--key-file")
    args = p.parse_args()
    if args.prepare_from:
        prepare(args.root, args.prepare_from, protocol_variant=args.protocol)
        print("Prepared", Path(args.root).resolve())
        return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and args.key_file:
        key = Path(args.key_file).expanduser().read_text().strip()
    if not key:
        key = getpass.getpass("OpenRouter key (hidden; kept only in memory): ")
    if not key.strip():
        raise ValueError("A nonempty OpenRouter key is required")
    summary = run(args.root, key.strip())
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
