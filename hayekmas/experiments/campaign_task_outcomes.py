"""Task-level reading guide from saved evaluations; never calls a model."""

import json


def export_task_outcomes(root, data):
    design_path = root / "grader-recheck-design.json"
    design = json.loads(design_path.read_text()) if design_path.exists() else {"arms": {}}
    sources = {(arm, row["task_id"]): row for arm, records in design["arms"].items() for row in records}
    lookup = {
        (arm, row["task_id"], row.get("repeat")): row
        for arm, values in data["arms"].items()
        for row in values["episodes"]
        if row["epoch"] == 1 and row["phase"] in ("evaluation", "replication")
    }
    ordered = sorted({row["index"]: row["task_id"] for row in lookup.values()}.items())
    lines = [
        "TASK-BY-TASK RESEARCH OUTCOMES",
        "Primary scores stay fixed; repeats evaluate the same trained populations afresh on the same tasks.",
        "A threshold pass (score >= 0.5) is not a fully solved task. Missing submissions score zero.",
        "Grader explanations below are the exact primary model-grader responses, not independent expert findings.",
        "Problem excerpts are truncated; use the viewer for the complete task and each saved answer.",
        "",
    ]
    for index, task_id in ordered:
        task = data["tasks"][task_id]
        problem = " ".join(task["problem"].split())
        lines.extend(
            [
                f"{index:02d}. {task_id} [{task['subject']}]",
                problem[:600] + (" … [excerpt]" if len(problem) > 600 else ""),
                "System       Run         Score    Final answer   Outcome",
            ]
        )
        for arm in ("original", "teams"):
            for repeat in (None, 1, 2):
                row = lookup.get((arm, task_id, repeat))
                label = "Primary" if repeat is None else f"Repeat {repeat}"
                if row is None:
                    lines.append(f"{arm:<12} {label:<11} Incomplete")
                    continue
                score = float(row["score"] or 0)
                lines.append(
                    f"{arm:<12} {label:<11} {score:<8.3f} "
                    f"{'yes' if row['has_final_answer'] else 'no':<14} "
                    f"{row['diagnostics']['outcome']}"
                )
            primary = lookup.get((arm, task_id, None))
            source = sources.get((arm, task_id), {})
            if primary:
                lines.append(f"  {arm} primary answer: {primary['artifact']}/answer.md")
                if primary["has_final_answer"]:
                    response = source.get("source_judge_response")
                    lines.append(
                        f"  Saved {arm} primary grader response: {response or 'Unavailable in frozen grading design.'}"
                    )
                else:
                    lines.append(f"  {arm}: no final answer; no final-answer grading response.")
        lines.append("")
    path = root / "TASK_OUTCOMES.tmp.txt"
    path.write_text("\n".join(lines))
    path.replace(root / "TASK_OUTCOMES.txt")
