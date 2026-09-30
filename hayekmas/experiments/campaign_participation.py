"""Describe team participation from completed primary training event logs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .campaign_client import atomic_json


def participation(events, names):
    agents = {name: Counter() for name in names}
    candidates = {}
    for event in events:
        kind = event["event"]
        if kind == "team_message":
            person = agents[event["agent"]]
            person[event["channel"] + "_messages"] += 1
            if event["channel"] == "team" and len(event["members"]) > 1:
                person["discussion_messages_with_peers"] += 1
        elif kind == "candidate":
            key = event["step"], event["id"]
            if key in candidates:
                raise ValueError("Duplicate candidate within one auction")
            candidates[key] = event
            agents[event["author"]]["candidates"] += 1
        elif kind == "submission":
            candidate = candidates[(event["step"], event["candidate_id"])]
            if event["answer"] != candidate["answer"] or event["final"] != candidate["final"]:
                raise ValueError("Selected answer does not match its candidate")
            field = "selected_final" if event["final"] else "selected_intermediate"
            agents[candidate["author"]][field] += 1
    return agents


def export(root):
    paths = sorted((root / "teams/training/epoch-1").glob("*/result.json"))
    results = [json.loads(path.read_text()) for path in paths]
    if len(results) != 40:
        raise ValueError("This fixed-campaign analysis requires all 40 primary training tasks")
    names = [agent["name"] for agent in results[0]["population_before"]]
    totals = {name: Counter() for name in names}
    tasks, hashes = [], {}
    for path, result in zip(paths, results, strict=True):
        event_path = root / result["artifact"] / "events.jsonl"
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        counts = participation(events, names)
        tasks.append({"index": result["index"], "task_id": result["task_id"], "agents": counts})
        for name in names:
            totals[name].update(counts[name])
            for field in ("paid", "bid_income", "reward_income"):
                totals[name][field] += result["metrics"][field].get(name, 0)
        for source in (path, event_path):
            hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sum(a["selected_final"] for a in totals.values()) == sum(r["has_final_answer"] for r in results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Completed primary team training only; each event is counted once. Selected proposals are attributed to their recorded author using both auction step and candidate ID.",
        "interpretation": "Counts describe recorded participation, not intellectual quality, causal credit, free riding, or stable functional roles. A selected answer can incorporate teammates' discussion. Money is simulation wealth, not API dollars.",
        "agents": totals,
        "tasks": tasks,
        "source_hashes": hashes,
    }
    atomic_json(root / "team-participation.json", report)
    lines = [
        "TEAM PARTICIPATION DURING PRIMARY TRAINING",
        report["scope"],
        report["interpretation"],
        "",
        "Agent | Bidding messages | Discussion messages | Candidates | Selected intermediate | Selected final | Paid bids | Bid income | Reward income",
    ]
    for name, row in totals.items():
        fields = (
            "bidding_messages",
            "team_messages",
            "candidates",
            "selected_intermediate",
            "selected_final",
            "paid",
            "bid_income",
            "reward_income",
        )
        lines.append(name + " | " + " | ".join(f"{row[field]:g}" for field in fields))
    (root / "TEAM_PARTICIPATION.txt").write_text("\n".join(lines) + "\n")
    plot(report, root / "figures")
    return report


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(report["agents"])
    positions = np.arange(len(names))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=(1, 1.6), layout="constrained")
    for offset, field, label, color in [
        (-0.25, "candidates", "Proposed candidates", "#95b8b2"),
        (0, "selected_intermediate", "Selected intermediate steps", "#168174"),
        (0.25, "selected_final", "Selected final answers", "#b87333"),
    ]:
        top.bar(
            positions + offset,
            [report["agents"][name].get(field, 0) for name in names],
            width=0.25,
            label=label,
            color=color,
        )
    top.set(xticks=positions, xticklabels=names, ylabel="Recorded events", title="Candidate authorship and selection")
    top.legend(frameon=False, ncols=3, fontsize=9)
    counts = np.array([[task["agents"][name].get("candidates", 0) for task in report["tasks"]] for name in names])
    mesh = bottom.imshow(counts, cmap="YlGnBu", vmin=0, aspect="auto")
    for y, name in enumerate(names):
        for x, task in enumerate(report["tasks"]):
            if task["agents"][name].get("selected_final", 0):
                bottom.scatter(x, y, marker="*", s=50, color="#ffb551", edgecolors="#57361b", linewidths=0.5)
    bottom.set(
        yticks=positions,
        yticklabels=names,
        xticks=range(0, 40, 5),
        xticklabels=range(1, 41, 5),
        xlabel="Primary training task in fixed order",
        title="Candidates proposed over time · star = selected final-answer author",
    )
    fig.colorbar(mesh, ax=bottom, label="Candidates proposed", shrink=0.8)
    fig.suptitle("Observed team participation · 40 training tasks", fontsize=15)
    fig.supxlabel(
        "Authorship is not exclusive intellectual credit: selected proposals may incorporate teammates' discussion.\nCounts do not establish quality, free riding, or stable functional roles.",
        fontsize=9,
    )
    output.mkdir(exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"team-participation.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    export(parser.parse_args().root)
