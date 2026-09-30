"""Describe submission failures and paired score gaps without changing scores."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .campaign_client import atomic_json


def score(row):
    if not row["has_final_answer"]:
        assert row["score"] in (None, 0)
        return 0.0
    value = row["score"]
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("Submitted answer lacks a valid completed grade")
    return value


def export(root):
    summaries, hashes = [], {}
    for label, location in [
        ("primary", "evaluation/epoch-1"),
        ("repeat-1", "replications/repeat-1"),
        ("repeat-2", "replications/repeat-2"),
    ]:
        records = {}
        for arm in ("original", "teams"):
            records[arm] = {}
            for path in sorted((root / arm / location).glob("*/result.json")):
                row = json.loads(path.read_text())
                records[arm][row["task_id"]] = row
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(records["original"]) != 19 or set(records["original"]) != set(records["teams"]):
            raise ValueError("Submission review requires all 19 paired tasks in each run")
        partition = {name: [] for name in ("both", "original_only", "teams_only", "neither")}
        failures = Counter()
        for task_id, team in records["teams"].items():
            original = records["original"][task_id]
            flags = original["has_final_answer"], team["has_final_answer"]
            name = {(True, True): "both", (True, False): "original_only", (False, True): "teams_only", (False, False): "neither"}[flags]
            partition[name].append(task_id)
            if not team["has_final_answer"]:
                path = root / team["artifact"] / "events.jsonl"
                events = [json.loads(line) for line in path.read_text().splitlines()]
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                submissions = [e for e in events if e["event"] == "submission"]
                assert not any(e["final"] for e in submissions)
                reasons = [e["reason"] for e in events if e["event"] == "no_submission"]
                category = "after intermediate submissions" if submissions else "before any submission"
                failures[(reasons[-1] if reasons else "unspecified") + " / " + category] += 1
        overall = math.fsum(score(records["teams"][i]) - score(records["original"][i]) for i in records["teams"]) / 19
        groups = {}
        for group, ids in partition.items():
            groups[group] = {
                "n": len(ids),
                "task_ids": ids,
                "conditional_mean": {arm: math.fsum(score(records[arm][i]) for i in ids) / len(ids) if ids else None for arm in records},
                "contribution_to_overall_teams_minus_original": math.fsum(score(records["teams"][i]) - score(records["original"][i]) for i in ids) / 19,
            }
        assert math.isclose(sum(g["contribution_to_overall_teams_minus_original"] for g in groups.values()), overall, abs_tol=1e-12)
        summaries.append({"run": label, "overall_teams_minus_original": overall, "submission_groups": groups, "team_failure_counts": dict(failures)})
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Post hoc descriptive analysis of all primary and repeat held-out records. Original scores and failure zeros are unchanged.",
        "interpretation": "Submission groups are selected by observed outcomes, not randomized treatments. Conditional means describe different task subsets and cannot establish answer quality or communication effects. The contribution columns are an exact arithmetic partition of the full 19-task score gap, not causal attribution. Repeats use the same tasks and trained populations.",
        "runs": summaries,
        "source_hashes": hashes,
    }
    atomic_json(root / "submission-review.json", report)
    lines = ["SUBMISSION FAILURES AND PAIRED SCORE GAPS", report["scope"], report["interpretation"], ""]
    for run in summaries:
        lines += [run["run"].upper(), "Submission group | Tasks | Original mean within group | Teams mean within group | Contribution to overall teams-minus-original gap"]
        for name, group in run["submission_groups"].items():
            values = ["n/a" if group["conditional_mean"][a] is None else f"{group['conditional_mean'][a]:.6f}" for a in ("original", "teams")]
            lines.append(f"{name} | {group['n']} | {' | '.join(values)} | {group['contribution_to_overall_teams_minus_original']:+.6f}")
        lines.append(f"Overall teams-minus-original gap: {run['overall_teams_minus_original']:+.6f}")
        lines.extend(f"Team failure: {cause}: {n}" for cause, n in run["team_failure_counts"].items())
        lines.append("")
    both = [run["submission_groups"]["both"] for run in summaries]
    if all(group["n"] and group["conditional_mean"]["original"] > group["conditional_mean"]["teams"] for group in both):
        lines.append("Original EoM has the higher mean within each run's subset where both systems submitted. These subsets are selected by outcomes.")
    lines.append("Submission failures and conditional score differences do not identify the effect of repairing the team interface; that requires a separate prospective evaluation.")
    (root / "SUBMISSION_REVIEW.txt").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    export(parser.parse_args().root)
