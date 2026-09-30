"""Read-only live report and paired comparisons; no inference calls."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time
from .campaign_client import atomic_json
from .campaign_statistics import paired_intervals
from .campaign_native_trace import native_trace
from .campaign_negotiation import negotiation_trace
from .campaign_grader_report import recheck_summary

VISIBLE_EVENTS = {
    "message",
    "team_message",
    "joined",
    "left",
    "team_dissolved",
    "pledge",
    "bid_transfer",
    "candidate",
    "submission",
    "strategy_updated",
    "auction",
    "no_submission",
    "finalization_started",
    "finalization_abstained",
    "discussion_shortened",
    "phase_skipped",
    "invalid_action",
    "settlement",
}


def visible_events(events):
    return [
        e
        for e in events
        if e["event"] in VISIBLE_EVENTS and (e["event"] != "message" or e.get("channel") in ("formation", "invitation"))
    ]


def live_team_episode(root, folder, status, records):
    if status.get("status") != "running" or status.get("phase") not in ("training", "evaluation", "replication"):
        return None
    base = (
        folder / "replications" / f"repeat-{status['repeat']}"
        if status["phase"] == "replication"
        else folder / status["phase"] / f"epoch-{status['epoch']}"
    )
    directory = base / f"{status['task_number']:02d}-{status['task_id']}"
    if (directory / "result.json").exists():
        return None
    attempts = sorted(directory.glob("attempt-*/events.jsonl"), key=lambda p: int(p.parent.name.split("-")[-1]))
    if not attempts:
        return None
    events = rows(attempts[-1])
    initial = next((e.get("agents", []) for e in events if e["event"] == "initialized"), [])
    agents = {a["name"]: dict(a) for a in initial}
    for event in events:
        if event["event"] == "joined":
            for name in (event["agent"], event.get("inviter")):
                if name in agents:
                    agents[name]["team"] = event["team"]
        if event["event"] == "left" and event["agent"] in agents:
            agents[event["agent"]]["team"] = None
        if event["event"] == "team_dissolved":
            remaining = event.get("remaining", [])
            for name in [remaining] if isinstance(remaining, str) else remaining:
                if name in agents:
                    agents[name]["team"] = None
        if event["event"] == "settlement":
            for name, wealth in event["wealth"].items():
                if name in agents:
                    agents[name]["wealth"] = wealth
    return {
        "live": True,
        "phase": status["phase"],
        "epoch": status["epoch"],
        "repeat": status.get("repeat") if status["phase"] == "replication" else None,
        "index": status["task_number"],
        "task_id": status["task_id"],
        "subject": "in progress",
        "score": None,
        "steps": max((e.get("step", -1) for e in events if e.get("step") is not None), default=-1) + 1,
        "has_final_answer": any(e["event"] == "submission" and e.get("final") for e in events),
        "population_before": initial,
        "population_after": list(agents.values()),
        "events": visible_events(events),
        "actions": [e for e in events if e["event"] == "submission"],
        "artifact": str(attempts[-1].parent.relative_to(root)),
        "full_episode_cost": math.fsum(
            r.get("cost_usd", 0)
            for r in records.values()
            if r.get("task_id") == status["task_id"]
            and r.get("phase") == status["phase"]
            and r.get("epoch") == status["epoch"]
            and r.get("repeat") == (status.get("repeat") if status["phase"] == "replication" else None)
            and r.get("protocol_version") == 3
        ),
        "full_episode_unconfirmed_usd": math.fsum(
            r.get("reserved_usd", 0)
            for r in records.values()
            if "cost_usd" not in r
            and r.get("task_id") == status["task_id"]
            and r.get("phase") == status["phase"]
            and r.get("epoch") == status["epoch"]
            and r.get("repeat") == (status.get("repeat") if status["phase"] == "replication" else None)
            and r.get("protocol_version") == 3
        ),
    }


def rows(path):
    if not path.exists():
        return []
    output = []
    for line in path.read_text().splitlines():
        try:
            output.append(json.loads(line))
        except ValueError:
            pass  # An append may be observed before its newline.
    return output


def episode_rows(folder, include_replications=False):
    """Committed task files are authoritative, even after an interrupted index append."""
    indexed = {}
    indices = ["episodes.jsonl"] + (["replication_episodes.jsonl"] if include_replications else [])
    patterns = ["training/epoch-*/*/result.json", "evaluation/epoch-*/*/result.json"]
    if include_replications:
        patterns.append("replications/repeat-*/*/result.json")
    for name in indices:
        for row in rows(folder / name):
            indexed[(row["phase"], row["epoch"], row.get("repeat"), row["task_id"])] = row
    for pattern in patterns:
        for path in folder.glob(pattern):
            row = json.loads(path.read_text())
            indexed[(row["phase"], row["epoch"], row.get("repeat"), row["task_id"])] = row
    return sorted(
        indexed.values(),
        key=lambda r: (
            r["phase"] == "replication",
            r.get("repeat", 0),
            r["epoch"],
            r["phase"] == "evaluation",
            r["index"],
        ),
    )


def diagnostics(row, events):
    counts = Counter(e["event"] for e in events)
    invalid = Counter(e["reason"] for e in events if e["event"] == "invalid_action")
    before = {a["id"] if a.get("id") is not None else a["name"] for a in row.get("population_before", [])}
    after = {a["id"] if a.get("id") is not None else a["name"] for a in row.get("population_after", [])}
    if row.get("has_final_answer"):
        outcome = "graded_pass" if row["passed"] else "graded_below_threshold"
    else:
        reasons = [e["reason"] for e in events if e["event"] == "no_submission"]
        zero_bid = any(e["event"] == "auction" and e.get("winner") is None for e in events)
        outcome = (
            reasons[-1] if reasons else "no positive bid" if zero_bid else row.get("termination", "no final answer")
        )
    bids = [value for e in events if e["event"] == "auction" for value in e["contributions"].values()]
    result = {
        "outcome": outcome,
        "events": dict(counts),
        "invalid_reasons": dict(invalid),
        "invalid_actions_this_episode": sum(invalid.values()),
        "births": len(after - before),
        "removed_agents": len(before - after),
        "zero_contribution_fraction": sum(b == 0 for b in bids) / len(bids) if bids else None,
        "agent_bid_count": len(bids),
        "messages": counts["team_message"],
        "messages_by_channel": dict(Counter(e["channel"] for e in events if e["event"] == "team_message")),
        "joins": counts["joined"],
        "leaves": counts["left"],
        "total_bid_paid": math.fsum(e.get("paid", 0) for e in events if e["event"] == "auction"),
        "total_bid_transferred": math.fsum(e.get("total", 0) for e in events if e["event"] == "bid_transfer"),
        "total_bid_burned": math.fsum(e.get("burned", 0) for e in events if e["event"] == "bid_transfer"),
    }
    if row.get("arm") == "original":
        # The native engine has no team event log. Missing telemetry is not zero activity.
        for key in (
            "invalid_actions_this_episode",
            "zero_contribution_fraction",
            "agent_bid_count",
            "messages",
            "messages_by_channel",
            "joins",
            "leaves",
            "total_bid_paid",
            "total_bid_transferred",
            "total_bid_burned",
        ):
            result[key] = None
    return result


def summarize(episodes):
    groups = []
    for epoch, phase, repeat in sorted({(r["epoch"], r["phase"], r.get("repeat", 0)) for r in episodes}):
        subset = [r for r in episodes if r["epoch"] == epoch and r["phase"] == phase and r.get("repeat", 0) == repeat]
        n = len(subset)
        groups.append(
            {
                "epoch": epoch,
                "phase": phase,
                "repeat": repeat,
                "n": n,
                "mean_score": math.fsum(r["score"] or 0 for r in subset) / n,
                "passed": sum(bool(r["passed"]) for r in subset),
                "final_answers": sum(bool(r["has_final_answer"]) for r in subset),
                "cost_usd": math.fsum(r["full_episode_cost"] for r in subset),
                "unconfirmed_usd": math.fsum(r.get("full_episode_unconfirmed_usd", 0) for r in subset),
                "outcomes": dict(Counter(r["diagnostics"]["outcome"] for r in subset)),
                "messages": sum(r["diagnostics"]["messages"] or 0 for r in subset)
                if subset[0].get("arm") != "original"
                else None,
                "invalid_actions": sum(r["diagnostics"]["invalid_actions_this_episode"] or 0 for r in subset)
                if subset[0].get("arm") != "original"
                else None,
            }
        )
    return groups


def write_csv(root, result):
    fields = [
        "arm",
        "phase",
        "epoch",
        "repeat",
        "index",
        "task_id",
        "subject",
        "score",
        "passed",
        "has_final_answer",
        "steps",
        "full_episode_cost",
        "full_episode_unconfirmed_usd",
        "outcome",
        "messages",
        "joins",
        "leaves",
        "births",
        "removed_agents",
        "invalid_actions_this_episode",
        "zero_contribution_fraction",
        "total_bid_paid",
        "total_bid_transferred",
        "total_bid_burned",
        "artifact",
    ]
    temporary = root / "episodes.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for arm in result["arms"].values():
            for row in arm["episodes"]:
                writer.writerow({**row, **row["diagnostics"]})
    temporary.replace(root / "episodes.csv")


def write_snapshot(root, result):
    template = Path(__file__).with_name("campaign_dashboard.html").read_text()
    # Escaping '<' prevents model-authored text from closing this data element.
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    embedded = f'<script id="snapshot-data" type="application/json">{payload}</script>\n'
    page = template.replace("<script>\n", embedded + "<script>\n", 1)
    temporary = root / "snapshot.html.tmp"
    temporary.write_text(page)
    temporary.replace(root / "snapshot.html")


def diagnostic_summary(root, records):
    folder = root / "teams/interface-diagnostic"
    calls = [r for r in records.values() if r.get("study") == "interface_diagnostic"]
    variants = []
    for variant in ("baseline", "clarified"):
        tasks = []
        for path in sorted((folder / variant).glob("*/result.json")):
            row = json.loads(path.read_text())
            events = rows(root / row["artifact"] / "events.jsonl")
            tasks.append(
                {
                    "index": row["index"],
                    "task_id": row["task_id"],
                    "score": row["score"],
                    "has_final_answer": row["has_final_answer"],
                    "passed": row["passed"],
                    "replay": row["replay"],
                    "candidate_count": sum(e["event"] == "candidate" for e in events),
                    "paid_reflections": sum(e["event"] == "reflection_paid" for e in events),
                    "strategy_updates": sum(e["event"] == "strategy_updated" for e in events),
                    "outcome": diagnostics({**row, "arm": "teams"}, events)["outcome"],
                    "cost_usd": math.fsum(
                        r.get("cost_usd", 0)
                        for r in calls
                        if r.get("variant") == variant and r.get("task_id") == row["task_id"]
                    ),
                }
            )
        variants.append(
            {
                "variant": variant,
                "n": len(tasks),
                "tasks": tasks,
                "final_answers": sum(t["has_final_answer"] for t in tasks),
                "paid_reflections": sum(t["paid_reflections"] for t in tasks),
                "strategy_updates": sum(t["strategy_updates"] for t in tasks),
                "mean_score": math.fsum(t["score"] or 0 for t in tasks) / len(tasks) if tasks else None,
                "cost_usd": math.fsum(r.get("cost_usd", 0) for r in calls if r.get("variant") == variant),
            }
        )
    return {
        "started": (folder / "started.json").exists(),
        "variants": variants,
        "status": json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else None,
        "cost_usd": math.fsum(r.get("cost_usd", 0) for r in calls),
        "maximum_increment_usd": 1.0,
    }


def build(root):
    plan = json.loads((root / "plan.json").read_text())
    result = {"plan": plan, "updated_at": time.time(), "arms": {}, "paired": [], "repeated": []}
    closure = root / "paid-experiments-closed.json"
    result["paid_execution"] = json.loads(closure.read_text()) if closure.exists() else None
    benchmark = Path(__file__).resolve().parents[2] / "third_party/benchmarks/frontier-science-research/data"
    result["tasks"] = {
        str(row.get("id", row.get("task_group_id"))): {"problem": row["problem"], "subject": row["subject"]}
        for filename in ("research_train.jsonl", "research_test.jsonl")
        for row in rows(benchmark / filename)
    }
    for arm in ("original", "teams"):
        folder = root / arm
        status = (
            json.loads((folder / "status.json").read_text())
            if (folder / "status.json").exists()
            else {"status": "starting"}
        )
        records = {r["request"]: r for r in rows(folder / "api_usage.jsonl")}
        if arm == "teams":
            result["interface_diagnostic"] = diagnostic_summary(root, records)
        episodes = episode_rows(folder, include_replications=True)
        view = []
        for row in episodes:
            row = dict(row)
            for key in ("population_before", "population_after"):
                row[key] = [
                    {
                        k: a.get(k)
                        for k in (
                            "id",
                            "name",
                            "wealth",
                            "team",
                            "team_tag",
                            "status",
                            "bid",
                            "type",
                            "strategy",
                            "summary",
                        )
                    }
                    for a in row.get(key, [])
                ]
            row["full_episode_cost"] = sum(
                r.get("cost_usd", 0)
                for r in records.values()
                if r.get("task_id") == row["task_id"]
                and r.get("phase") == row["phase"]
                and r.get("epoch") == row["epoch"]
                and r.get("repeat") == row.get("repeat")
                and r.get("protocol_version", 1) == row.get("protocol_version", 1)
            )
            row["full_episode_unconfirmed_usd"] = math.fsum(
                r.get("reserved_usd", 0)
                for r in records.values()
                if "cost_usd" not in r
                and r.get("task_id") == row["task_id"]
                and r.get("phase") == row["phase"]
                and r.get("epoch") == row["epoch"]
                and r.get("repeat") == row.get("repeat")
                and r.get("protocol_version", 1) == row.get("protocol_version", 1)
            )
            events = rows(root / row["artifact"] / "events.jsonl")
            row["diagnostics"] = diagnostics(row, events)
            row["events"] = visible_events(events)
            if arm == "original":
                row["native_trace"] = native_trace(root / row["artifact"] / "trajectory.log")
            else:
                row["negotiation"] = negotiation_trace(events)
            view.append(row)
        result["arms"][arm] = {
            "status": status,
            "grader_recheck": recheck_summary(root, arm, records),
            "episodes": view,
            "summary": summarize(view),
            "live": live_team_episode(root, folder, status, records) if arm == "teams" else None,
            "accounting": {
                "preflight_cost_usd": math.fsum(
                    r.get("cost_usd", 0)
                    for r in records.values()
                    if r.get("protocol_version", 1) < plan["protocol_version"]
                ),
                "main_cost_usd": math.fsum(
                    r.get("cost_usd", 0)
                    for r in records.values()
                    if r.get("protocol_version", 1) == plan["protocol_version"] and not r.get("study")
                ),
                "diagnostic_cost_usd": math.fsum(
                    r.get("cost_usd", 0) for r in records.values() if r.get("study") == "interface_diagnostic"
                ),
                "grading_check_cost_usd": math.fsum(
                    r.get("cost_usd", 0) for r in records.values() if r.get("study") == "grader_recheck"
                ),
                "charged_empty_responses": sum(r["status"] == "charged_empty" for r in records.values()),
                "unconfirmed_requests": sum("cost_usd" not in r for r in records.values()),
            },
        }
    for phase, epoch, repeat in [
        ("evaluation", 1, None),
        ("evaluation", 2, None),
        ("replication", 1, 1),
        ("replication", 1, 2),
    ]:
        maps = {
            a: {
                r["task_id"]: r
                for r in v["episodes"]
                if r["phase"] == phase and r["epoch"] == epoch and r.get("repeat") == repeat
            }
            for a, v in result["arms"].items()
        }
        paired_ids = sorted(maps["original"].keys() & maps["teams"].keys())
        if paired_ids:
            paired = [
                {
                    "task_id": i,
                    "subject": maps["original"][i]["subject"],
                    "original": maps["original"][i]["score"] or 0,
                    "teams": maps["teams"][i]["score"] or 0,
                    "original_cost": maps["original"][i]["full_episode_cost"],
                    "teams_cost": maps["teams"][i]["full_episode_cost"],
                    "original_reserved": maps["original"][i]["full_episode_unconfirmed_usd"],
                    "teams_reserved": maps["teams"][i]["full_episode_unconfirmed_usd"],
                    "original_pass": maps["original"][i]["passed"],
                    "teams_pass": maps["teams"][i]["passed"],
                }
                for i in paired_ids
            ]
            result["repeated" if repeat is not None else "paired"].append(
                {
                    "epoch": epoch,
                    "repeat": repeat,
                    "n": len(paired),
                    "complete": len(paired) == 19,
                    "tasks": paired,
                    "mean_score": {a: sum(r[a] for r in paired) / len(paired) for a in maps},
                    "pass_rate": {a: sum(r[a + "_pass"] for r in paired) / len(paired) for a in maps},
                    "paired_score_difference": math.fsum(r["teams"] - r["original"] for r in paired) / len(paired),
                    "wins_ties_losses": {
                        "teams_higher": sum(r["teams"] > r["original"] for r in paired),
                        "tied": sum(r["teams"] == r["original"] for r in paired),
                        "original_higher": sum(r["teams"] < r["original"] for r in paired),
                    },
                    "intervals": paired_intervals(paired),
                }
            )
    atomic_json(root / "dashboard-data.json", result)
    write_csv(root, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--figures", action="store_true", help="Refresh PNG/SVG/PDF exports after each completed episode")
    args = p.parse_args()
    root = Path(args.root)
    previous = None
    while True:
        result = build(root)
        signature = tuple(
            (arm, r["phase"], r["epoch"], r.get("repeat"), r["index"])
            for arm, values in result["arms"].items()
            for r in values["episodes"]
        )
        signature += tuple(
            (arm, values["status"].get("status"), values["status"].get("phase"))
            for arm, values in result["arms"].items()
        )
        signature += (time.time() >= result["plan"]["deadline_unix"],)
        signature += tuple(("diagnostic", v["variant"], v["n"]) for v in result["interface_diagnostic"]["variants"])
        signature += tuple(
            ("grader_recheck", arm, values["grader_recheck"].get("completed_grades", 0))
            for arm, values in result["arms"].items()
        )
        if args.figures and signature != previous:
            from .campaign_figures import export
            from .campaign_audit import audit
            from .campaign_behavior import build_behavior
            from .campaign_findings import export_findings

            export(root)
            audit(root, Path(__file__).resolve().parents[2])
            build_behavior(root)
            export_findings(root)
            write_snapshot(root, result)
            previous = signature
        if not args.watch or time.time() > result["plan"]["deadline_unix"] + 120:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
