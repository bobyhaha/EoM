"""Read-only reporting and provenance checks for fixed-answer regrading."""

import json
import math
from .campaign_grader import score_from_text, sha256


def recheck_summary(root, arm, records):
    path = root / "grader-recheck-design.json"
    if not path.exists():
        return {"declared": False, "started": False, "replicates": []}
    sources = json.loads(path.read_text())["arms"].get(arm, [])
    study = root / arm / "grader-recheck"
    started = (study / "started.json").exists()
    calls = [row for row in records.values() if row.get("study") == "grader_recheck"]
    replicates = []
    for replicate in range(1, 4):
        tasks = []
        for source in sources:
            grade_path = study / f"replicate-{replicate}/{source['index']:02d}-{source['task_id']}.json"
            grade = json.loads(grade_path.read_text()) if grade_path.exists() else None
            score = (grade["score"] if grade and grade.get("valid") else None) if source["has_final_answer"] else 0
            tasks.append(
                {
                    "index": source["index"],
                    "task_id": source["task_id"],
                    "has_final_answer": source["has_final_answer"],
                    "primary_score": source["primary_score"],
                    "score": score,
                    "grade_attempted": grade is not None,
                    "grade_valid": grade.get("valid", False) if grade else False,
                    "artifact": str(grade_path.relative_to(root)) if grade else None,
                }
            )
        complete = started and len(tasks) == 19 and all(row["score"] is not None for row in tasks)
        replicates.append(
            {
                "replicate": replicate,
                "complete": complete,
                "tasks": tasks,
                "valid_grades": sum(row["grade_valid"] for row in tasks),
                "attempted_grades": sum(row["grade_attempted"] for row in tasks),
                "expected_grades": sum(row["has_final_answer"] for row in tasks),
                "mean_score": math.fsum(row["score"] for row in tasks) / 19 if complete else None,
                "passes": sum(row["score"] >= 0.5 for row in tasks) if complete else None,
            }
        )
    return {
        "declared": True,
        "started": started,
        "status": json.loads((study / "status.json").read_text()) if (study / "status.json").exists() else None,
        "billed_usd": math.fsum(row.get("cost_usd", 0) for row in calls),
        "reserved_usd": math.fsum(row["reserved_usd"] for row in calls if "cost_usd" not in row),
        "incremental_cap_usd": 0.5,
        "replicates": replicates,
        "completed_grades": sum(rep["attempted_grades"] for rep in replicates),
        "expected_grades": 3 * sum(source["has_final_answer"] for source in sources),
    }


def recheck_audit(root, arm, ledger, requests, responses, as_of=float("inf")):
    study = root / arm / "grader-recheck"
    if not (study / "started.json").exists():
        return []
    sources = {
        source["task_id"]: source
        for source in json.loads((root / "grader-recheck-design.json").read_text())["arms"][arm]
    }
    calls = [row for row in ledger.values() if row.get("study") == "grader_recheck"]
    cost = math.fsum(row.get("cost_usd", row.get("reserved_usd", 0)) for row in calls)
    budget = json.loads((study / "started.json").read_text())
    scope_errors, prompt_errors, grade_errors, source_errors = [], [], [], []
    pending_request_logs, newer_grade_records = [], []
    for source in sources.values():
        for kind in ("result", "answer"):
            path = root / source[kind]
            if not path.exists() or sha256(path) != source[kind + "_sha256"]:
                source_errors.append(source[kind])
        if source["has_final_answer"]:
            original = requests.get(source["source_request_number"], {})
            if {name: original.get(name) for name in source["judge_request"]} != source["judge_request"]:
                source_errors.append(source["source_request_number"])
    for call in calls:
        source = sources.get(call.get("task_id"), {})
        if (
            not source.get("has_final_answer")
            or call.get("grading_replicate") not in (1, 2, 3)
            or call.get("kind") != "judge"
            or call.get("phase") != "grader_recheck"
            or call.get("source_request_number") != source.get("source_request_number")
        ):
            scope_errors.append(call["request"])
            continue
        request = requests.get(call["request"], {})
        if not request and call.get("status") == "reserved":
            pending_request_logs.append(call["request"])
            continue
        expected = source["judge_request"]
        matches = all(request.get(name) == expected[name] for name in ("prompt", "system", "reasoning_effort"))
        matches = matches and request.get("max_tokens") == expected["max_tokens"] * 2 ** (call["attempt"] - 1)
        if not matches:
            prompt_errors.append(call["request"])
    for path in study.glob("replicate-*/*.json"):
        row = json.loads(path.read_text())
        if row["time"] > as_of:
            newer_grade_records.append(str(path.relative_to(root)))
            continue
        source = sources.get(row.get("task_id"), {})
        response = responses.get(row["request_end"], {})
        call = ledger.get(row["request_end"], {})
        expected_score = score_from_text(response.get("text"))
        if (
            not source.get("has_final_answer")
            or row.get("grading_replicate") not in (1, 2, 3)
            or row.get("source_request_number") != source.get("source_request_number")
            or row.get("response") != response.get("text")
            or call.get("status") != "charged"
            or call.get("study") != "grader_recheck"
            or call.get("task_id") != row.get("task_id")
            or call.get("grading_replicate") != row.get("grading_replicate")
            or call.get("source_request_number") != row.get("source_request_number")
            or row.get("score") != expected_score
            or row.get("valid") != (expected_score is not None)
        ):
            grade_errors.append(str(path.relative_to(root)))
    protected = (
        json.loads((study / "protected-files.json").read_text()) if (study / "protected-files.json").exists() else {}
    )
    expected_files = set(json.loads((root / "primary-results-manifest.json").read_text())["files"])
    expected_files.update(
        str(path.relative_to(root))
        for name in ("original", "teams")
        for path in (root / name / "replications").glob("repeat-*/*/result.json")
    )
    changed = [
        name for name, digest in protected.items() if not (root / name).exists() or sha256(root / name) != digest
    ]
    return [
        (
            f"{arm}: grading check $0.50 ceiling",
            cost <= 0.5 + 1e-9 and budget["maximum_increment_usd"] == 0.5,
            {"billed_plus_reserved_usd": cost, "maximum_usd": 0.5},
        ),
        (
            f"{arm}: grading check uses exact primary sources",
            not source_errors and len(sources) == 19,
            {"primary_tasks": len(sources), "changed_sources": source_errors},
        ),
        (
            f"{arm}: grading check call scope and prompt fidelity",
            not scope_errors and not prompt_errors,
            {
                "requests_checked": len(calls) - len(pending_request_logs),
                "scope_errors": scope_errors,
                "prompt_errors": prompt_errors,
                "reserved_requests_waiting_for_payload_log": pending_request_logs,
            },
        ),
        (
            f"{arm}: grading check outputs trace to charged responses",
            not grade_errors,
            {
                "grade_records_checked": len(list(study.glob("replicate-*/*.json"))) - len(newer_grade_records),
                "mismatches": grade_errors,
                "new_records_deferred_until_next_snapshot": newer_grade_records,
            },
        ),
        (
            f"{arm}: grading check preserves primary and repeats",
            bool(protected) and set(protected) == expected_files and not changed,
            {
                "protected_files": len(protected),
                "changed_files": changed,
                "missing_manifest_entries": sorted(expected_files - set(protected)),
            },
        ),
    ]
