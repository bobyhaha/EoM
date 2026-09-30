"""Read-only provenance and completeness audit; never issues model requests."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from .campaign_client import atomic_json
from .campaign_report import episode_rows, rows
from .campaign_negotiation import negotiation_trace
from .campaign_grader_report import recheck_audit


def checkpoint_population(checkpoint, arm):
    if arm == "original":
        return checkpoint["population"]
    return [
        {
            "name": a["name"],
            "wealth": a["wealth"],
            "team": a["team_tag"],
            "strategy": a["trainable_system_prompt"],
            "summary": a["summary"],
        }
        for a in checkpoint["agents"]
    ]


def request_range(episode, ledger):
    end = episode.get("request_end")
    if end is None:
        end = max((n for n, r in ledger.items() if r["time"] <= episode["time"]), default=0)
    start = episode.get("request_start")
    if start is None:
        start = end - episode["usage"]["requests"] + 1
    return [ledger[n] for n in range(start, end + 1) if n in ledger]


def primary_preservation(root):
    path = root / "primary-results-manifest.json"
    protected = json.loads(path.read_text()).get("files", {}) if path.exists() else {}
    expected = {
        str(p.relative_to(root))
        for arm in ("original", "teams")
        for pattern in (
            "training/epoch-1/*/result.json",
            "training/epoch-1/*/checkpoint.json",
            "evaluation/epoch-1/*/result.json",
            "epoch-1.json",
            "training_checkpoint.json",
        )
        for p in (root / arm).glob(pattern)
    }
    changed = [
        name
        for name, digest in protected.items()
        if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    ]
    return {
        "preserved": bool(protected) and set(protected) == expected and not changed,
        "files_checked": len(protected),
        "expected_files": len(expected),
        "changed_or_missing_files": sorted(changed),
        "missing_manifest_entries": sorted(expected - set(protected)),
        "unexpected_manifest_entries": sorted(set(protected) - expected),
    }


def closed_ledger_preservation(root):
    manifest = json.loads((root / "closed-ledger-manifest.json").read_text())
    protected = manifest.get("files", {})
    expected = {
        f"{arm}/{name}.jsonl"
        for arm in ("original", "teams")
        for name in ("api_usage", "requests", "responses")
    }
    changed = []
    for name in sorted(expected & set(protected)):
        path = root / name
        if not path.is_file() or path.stat().st_size != protected[name]["bytes"]:
            changed.append(name)
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != protected[name]["sha256"]:
            changed.append(name)
    restart_markers = sorted(p.name for p in root.glob("RESTART-*"))
    stop_present = (root / "STOP").is_file()
    return {
        "preserved": set(protected) == expected and not changed and stop_present and not restart_markers,
        "recorded_at": manifest.get("recorded_at"),
        "files_checked": len(expected & set(protected)),
        "changed_or_missing_files": changed,
        "missing_manifest_entries": sorted(expected - set(protected)),
        "unexpected_manifest_entries": sorted(set(protected) - expected),
        "stop_present": stop_present,
        "restart_markers": restart_markers,
    }


def audit(root, repository):
    plan = json.loads((root / "plan.json").read_text())
    checks = []

    def add(name, condition, detail, pending=False, informational=False):
        checks.append(
            {
                "requirement": name,
                "status": "info" if informational else "pass" if condition else "pending" if pending else "fail",
                "evidence": detail,
            }
        )

    sources = {}
    task_data = {}
    benchmark = repository / "third_party/benchmarks/frontier-science-research/data"
    for phase, filename in [("training", "research_train.jsonl"), ("evaluation", "research_test.jsonl")]:
        path = benchmark / filename
        task_rows = rows(path)
        task_data.update({str(r.get("id", r.get("task_group_id"))): r for r in task_rows})
        sources[phase] = {
            "ids": [r.get("id", r.get("task_group_id")) for r in task_rows],
            "normalized_problem_sha256": [
                hashlib.sha256(re.sub(r"\s+", " ", r["problem"]).strip().encode()).hexdigest() for r in task_rows
            ],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    add(
        "Training/test split",
        len(sources["training"]["ids"]) == 40
        and len(sources["evaluation"]["ids"]) == 19
        and not set(sources["training"]["ids"]) & set(sources["evaluation"]["ids"])
        and not set(sources["training"]["normalized_problem_sha256"])
        & set(sources["evaluation"]["normalized_problem_sha256"]),
        sources,
    )
    native_paths = ["hayekmas/base", "hayekmas/adapters/researchworld", "hayekmas/utils"]
    diff = subprocess.run(
        ["git", "diff", "--name-only", plan["upstream_commit"], "--", *native_paths],
        cwd=repository,
        text=True,
        capture_output=True,
    )
    add(
        "Original EoM source unchanged",
        diff.returncode == 0 and not diff.stdout.strip(),
        {"upstream_commit": plan["upstream_commit"], "changed_paths": diff.stdout.splitlines()},
    )
    if (root / "primary-results-manifest.json").exists() or plan.get("worker_mode") in ("replication", "diagnostic"):
        preservation = primary_preservation(root)
        add("Primary results and trained checkpoints preserved", preservation["preserved"], preservation)
    if (root / "closed-ledger-manifest.json").exists():
        preservation = closed_ledger_preservation(root)
        add("Closed API logs unchanged; paid execution remains stopped", preservation["preserved"], preservation)
    total = 0.0
    pending_cost = 0.0
    arms = {}
    for arm in ("original", "teams"):
        folder = root / arm
        ledger_observed_at = time.time()
        ledger = {r["request"]: r for r in rows(folder / "api_usage.jsonl")}
        responses = {r["request"]: r for r in rows(folder / "responses.jsonl")}
        requests = {r["request"]: r for r in rows(folder / "requests.jsonl")}
        for name, condition, evidence in recheck_audit(root, arm, ledger, requests, responses, as_of=ledger_observed_at):
            add(name, condition, evidence)
        test_solver_requests = []
        test_rubric_matches = []
        training_mutation_rubrics = []
        unexpected_training_matches = []
        for request in requests.values():
            if request.get("kind") == "judge" or request.get("protocol_version") != plan["protocol_version"]:
                continue
            task_id = request.get("task_id")
            if task_id not in task_data:
                continue
            held_out = task_id in sources["evaluation"]["ids"]
            if held_out:
                test_solver_requests.append(request["request"])
            rubric = task_data[task_id]["answer"]
            prompt = request.get("prompt", "") + "\n" + (request.get("system") or "")
            copied = any(
                value and value in prompt for value in (rubric.strip(), json.dumps(rubric, ensure_ascii=False)[1:-1])
            )
            if copied:
                if held_out:
                    test_rubric_matches.append(request["request"])
                elif arm == "original" and request["prompt"].startswith("You are an expert prompt engineer."):
                    training_mutation_rubrics.append(request["request"])
                else:
                    unexpected_training_matches.append(request["request"])
        add(
            f"{arm}: full held-out rubrics absent from non-judge requests",
            bool(test_solver_requests) and not test_rubric_matches,
            {
                "requests_checked": len(test_solver_requests),
                "matches": test_rubric_matches,
                "method": "Literal full-rubric check, including JSON-escaped form; not a semantic leakage test.",
            },
            pending=not test_solver_requests,
        )
        add(
            f"{arm}: training rubric access documented",
            True,
            {
                "native_original_training_mutation_requests": training_mutation_rubrics,
                "unexpected_requests": unexpected_training_matches,
                "interpretation": "Native original prompt mutation uses training targets. Other training matches require interpretation, not automatic classification as held-out leakage. Team reflection has different supervision.",
            },
            informational=True,
        )
        billed = math.fsum(r.get("cost_usd", 0) for r in ledger.values())
        uncertain = math.fsum(r["reserved_usd"] for r in ledger.values() if "cost_usd" not in r)
        total += billed
        pending_cost += uncertain
        add(
            f"{arm}: spending ceiling",
            billed + uncertain <= plan["arm_max_usd"] + 1e-9,
            {"billed_usd": billed, "unconfirmed_reserved_usd": uncertain, "ceiling_usd": plan["arm_max_usd"]},
        )
        late = [r["request"] for r in ledger.values() if r["time"] >= plan["deadline_unix"]]
        add(f"{arm}: request deadline", not late, {"requests_started_after_deadline": late})
        models = sorted({r["model"] for r in ledger.values() if r.get("model")})
        add(f"{arm}: model identity", models == [plan["model"]], models)
        episodes = episode_rows(folder, include_replications=True)
        train = [r for r in episodes if r["phase"] == "training"]
        test = [r for r in episodes if r["phase"] in ("evaluation", "replication")]
        continuity = [
            f"{a['epoch']}:{a['index']}->{b['epoch']}:{b['index']}"
            for a, b in zip(train, train[1:])
            if a["population_after"] != b["population_before"]
        ]
        add(
            f"{arm}: training population continuity",
            not continuity and bool(train),
            {"links_verified": max(0, len(train) - 1), "mismatches": continuity},
            pending=not train,
        )
        schedule_errors = [
            f"{r['phase']}:{r['epoch']}:{r['index']}"
            for r in episodes
            if sources["evaluation" if r["phase"] == "replication" else r["phase"]]["ids"][r["index"] - 1]
            != r["task_id"]
        ]
        add(
            f"{arm}: prescribed task order",
            not schedule_errors,
            {"committed_episodes": len(episodes), "mismatches": schedule_errors},
        )
        isolation_errors = []
        frozen_errors = []
        for r in test:
            checkpoint = json.loads((folder / f"epoch-{r['epoch']}.json").read_text())
            if checkpoint_population(checkpoint, arm) != r["population_before"]:
                isolation_errors.append(r["task_id"])
            if arm == "original" and r["population_after"] != r["population_before"]:
                frozen_errors.append(r["task_id"])
        add(
            f"{arm}: test starts from fixed trained snapshot",
            bool(test) and not isolation_errors,
            {"evaluations_checked": len(test), "mismatches": isolation_errors},
            pending=not test,
        )
        if arm == "original":
            add(
                "original: test population frozen",
                bool(test) and not frozen_errors,
                {"evaluations_checked": len(test), "mutations": frozen_errors},
                pending=not test,
            )
        provider_errors = []
        judge_errors = []
        missing_artifacts = []
        diagnostic_episodes = (
            [json.loads(p.read_text()) for p in (folder / "interface-diagnostic").glob("*/*/result.json")]
            if arm == "teams"
            else []
        )
        for r in episodes + diagnostic_episodes:
            calls = request_range(r, ledger)
            provider_errors.extend(
                c["request"] for c in calls if c.get("finish_reason") == "error" and c.get("transport_revision", 1) < 2
            )
            for call in calls:
                if call.get("kind") != "judge" or call.get("status") != "charged":
                    continue
                response = responses.get(call["request"], {}).get("text", "")
                values = re.findall(r"(?im)^\s*SCORE:\s*([^\s,]+)", response)
                try:
                    valid = len(values) == 1 and math.isfinite(float(values[0])) and 0 <= float(values[0]) <= 1
                except ValueError:
                    valid = False
                if not valid:
                    judge_errors.append(call["request"])
            for filename in ("trajectory.log", "answer.md"):
                if not (root / r["artifact"] / filename).exists():
                    missing_artifacts.append(f"{r['artifact']}/{filename}")
            if arm == "teams" and not (root / r["replay"]).exists():
                missing_artifacts.append(r["replay"])
        add(
            f"{arm}: no accepted provider-error text in committed attempts",
            not provider_errors,
            {"legacy_error_requests": provider_errors, "transport_revision": plan.get("transport_revision", 1)},
        )
        add(f"{arm}: valid rubric score syntax", not judge_errors, {"invalid_judge_requests": judge_errors})
        add(f"{arm}: saved answers and trajectories", not missing_artifacts, {"missing_files": missing_artifacts})
        if arm == "teams":
            negotiation_errors = []
            verified = 0
            for episode in episodes + diagnostic_episodes:
                trace = negotiation_trace(rows(root / episode["artifact"] / "events.jsonl"))
                verified += trace["counts"].get("wealth_values_checked", 0)
                negotiation_errors.extend({"artifact": episode["artifact"], **e} for e in trace["errors"])
            add(
                "teams: binding pledges, previous-winner transfers and personal wealth",
                verified > 0 and not negotiation_errors,
                {"personal_wealth_values_checked": verified, "mismatches": negotiation_errors},
                pending=verified == 0,
            )
        primary_train = [r for r in train if r["epoch"] == 1]
        primary_test = [r for r in test if r["epoch"] == 1 and r["phase"] == "evaluation"]
        complete = len(primary_train) == 40 and len(primary_test) == 19
        add(
            f"{arm}: full primary training and held-out evaluation",
            complete,
            {
                "training": len(primary_train),
                "required_training": 40,
                "evaluation": len(primary_test),
                "required_evaluation": 19,
            },
            pending=not complete,
        )
        arms[arm] = {"billed_usd": billed, "reserved_usd": uncertain, "training": len(train), "evaluation": len(test)}
        study = folder / "interface-diagnostic"
        if arm == "teams" and (study / "started.json").exists():
            study_calls = [r for r in ledger.values() if r.get("study") == "interface_diagnostic"]
            study_cost = math.fsum(r.get("cost_usd", r.get("reserved_usd", 0)) for r in study_calls)
            add(
                "Training-only diagnostic: $1 incremental ceiling",
                study_cost <= 1.0 + 1e-9,
                {"billed_plus_reserved_usd": study_cost, "maximum_usd": 1.0},
            )
            scope_errors = [
                r["request"]
                for r in study_calls
                if r.get("task_id") not in sources["training"]["ids"][:3]
                or r.get("variant") not in ("baseline", "clarified")
                or r.get("phase") != "diagnostic_training"
            ]
            add(
                "Training-only diagnostic: prescribed training tasks and variants",
                not scope_errors,
                {"requests_checked": len(study_calls), "mismatches": scope_errors},
            )
            protected = (
                json.loads((study / "primary-manifest.json").read_text())
                if (study / "primary-manifest.json").exists()
                else {}
            )
            changed = [
                name
                for name, digest in protected.items()
                if not (root / name).exists() or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
            ]
            add(
                "Training-only diagnostic: primary results and checkpoints preserved",
                bool(protected) and not changed,
                {"files_checked": len(protected), "changed_files": changed},
            )
    add(
        "Combined spending ceiling",
        total + pending_cost <= plan["max_usd"] + 1e-9,
        {"billed_usd": total, "unconfirmed_reserved_usd": pending_cost, "ceiling_usd": plan["max_usd"]},
    )
    add(
        "Research window elapsed",
        time.time() >= plan["deadline_unix"],
        {"deadline_utc": plan["deadline_utc"]},
        pending=True,
    )
    result = {
        "checked_at": time.time(),
        "checks": checks,
        "arms": arms,
        "failures": sum(c["status"] == "fail" for c in checks),
        "pending": sum(c["status"] == "pending" for c in checks),
    }
    atomic_json(root / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = audit(args.root, args.repository)
    print(json.dumps({"failures": result["failures"], "pending": result["pending"], "arms": result["arms"]}, indent=2))
    for check in result["checks"]:
        if check["status"] == "fail":
            print(json.dumps(check))
