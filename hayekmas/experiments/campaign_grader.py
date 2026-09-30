"""Post hoc grading reliability: replay saved judge prompts, never new answers."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import re
import time

from .campaign_client import CampaignClient, CampaignStop, atomic_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def score_from_text(text):
    matches = re.findall(r"(?im)^\s*SCORE:\s*([^\s,]+)", text or "")
    if len(matches) != 1:
        return None
    try:
        score = float(matches[0])
    except ValueError:
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_design(root):
    """Freeze every primary task and exact successful grading request offline."""
    destination = root / "grader-recheck-design.json"
    if destination.exists():
        return json.loads(destination.read_text())
    design = {
        "declared_at": datetime.now(timezone.utc).isoformat(),
        "status": "Post hoc reliability analysis, declared after primary results and some repeat results were known; no regrading calls yet.",
        "replicates": 3,
        "incremental_cap_per_arm_usd": 0.5,
        "minimum_minutes_to_start": 30,
        "priority": "After all planned repeat evaluations and the training-only interface diagnostic, if it runs.",
        "selection": "Every primary held-out task in prescribed order. Regrade all submitted primary answers regardless of original score; retain zero for missing submissions without a model call.",
        "request_rule": "Exact saved successful primary judge prompt, system message, reasoning effort, and output cap. Same CampaignClient model/endpoint/routing and bounded transport retries. No new agent reasoning or answer revision.",
        "reporting": "Separate from primary and solver repeats. Invalid grading syntax remains missing, not zero; no complete mean until every submitted answer has a valid grade in that replicate. Never replace original scores, choose a preferred grade, or treat grades as independent tasks.",
        "limits": "Measures response variability of the same model judge on fixed answers, not expert validation, training-seed variation, or solver variation. API decoding/provider variation is not separately controlled.",
        "arms": {},
    }
    for arm in ("original", "teams"):
        folder = root / arm
        ledger = {row["request"]: row for row in read_rows(folder / "api_usage.jsonl")}
        requests = {row["request"]: row for row in read_rows(folder / "requests.jsonl")}
        responses = {row["request"]: row for row in read_rows(folder / "responses.jsonl")}
        paths = sorted((folder / "evaluation/epoch-1").glob("*/result.json"))
        if len(paths) != 19:
            raise ValueError(f"{arm}: primary coverage incomplete")
        sources = []
        for path in paths:
            row = json.loads(path.read_text())
            source = {
                "index": row["index"],
                "task_id": row["task_id"],
                "subject": row["subject"],
                "has_final_answer": row["has_final_answer"],
                "primary_score": row["score"] or 0,
                "result": str(path.relative_to(root)),
                "result_sha256": sha256(path),
                "answer": row["artifact"] + "/answer.md",
                "answer_sha256": sha256(root / row["artifact"] / "answer.md"),
            }
            judges = [
                call
                for number, call in ledger.items()
                if row["request_start"] <= number <= row["request_end"]
                and call.get("kind") == "judge"
                and call.get("status") == "charged"
            ]
            if row["has_final_answer"]:
                if len(judges) != 1:
                    raise ValueError(f"{arm}/{row['task_id']}: expected one successful primary judge call")
                number = judges[0]["request"]
                request = requests[number]
                grade = score_from_text(responses[number].get("text"))
                if grade is None or not math.isclose(grade, source["primary_score"]):
                    raise ValueError("Primary judge response does not match saved score")
                source["judge_request"] = {
                    key: request.get(key) for key in ("prompt", "system", "reasoning_effort", "max_tokens")
                }
                source["source_request_number"] = number
                source["source_judge_response"] = responses[number]["text"]
            elif judges:
                raise ValueError("Unexpected judge call for a missing final answer")
            sources.append(source)
        design["arms"][arm] = sources
    atomic_json(destination, design)
    return design


def recheck_gate(root, arm, deadline, now=None):
    now = time.time() if now is None else now
    for name in ("original", "teams"):
        for repeat in (1, 2):
            if len(list((root / name / f"replications/repeat-{repeat}").glob("*/result.json"))) != 19:
                return "Both full solver repeats must finish first"
    started = (root / arm / "grader-recheck/started.json").exists()
    if now >= deadline or (not started and deadline - now < 1800):
        return "Insufficient time to start the grading check"
    diagnostic = root / "teams/interface-diagnostic/status.json"
    analysis_path = root / "analysis-plan.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    skipped = analysis.get("optional_training_interface_diagnostic", {}).get("skipped_reason")
    if not skipped and (
        not diagnostic.exists() or json.loads(diagnostic.read_text()).get("status") not in ("complete", "stopped")
    ):
        return "The interface diagnostic must finish or be explicitly skipped first"
    return None


def recheck_budget(path, usage, cap):
    if path.exists():
        return json.loads(path.read_text())
    budget = {
        "started_at": time.time(),
        "starting_billed_usd": usage["cost_usd"],
        "maximum_increment_usd": 0.5,
        "absolute_cap_usd": min(cap, usage["cost_usd"] + 0.5),
    }
    atomic_json(path, budget)
    return budget


def run_recheck(root, arm, key):
    plan = json.loads((root / "plan.json").read_text())
    design = json.loads((root / "grader-recheck-design.json").read_text())
    sources = design["arms"][arm]
    out = root / arm
    study = out / "grader-recheck"
    study.mkdir(exist_ok=True)
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((out / "status.json").read_text())
    state.update(pid=os.getpid(), status="running", phase="grader_recheck")
    state.pop("error", None)
    state["grades_expected"] = 3 * sum(source["has_final_answer"] for source in sources)

    def tick():
        state.update(usage=client.usage(), updated_at=time.time())
        atomic_json(out / "status.json", state)
        atomic_json(
            study / "status.json",
            {
                name: state.get(name)
                for name in (
                    "pid",
                    "status",
                    "phase",
                    "task_number",
                    "task_id",
                    "grading_replicate",
                    "grades_completed",
                    "grades_expected",
                    "usage",
                    "error",
                    "updated_at",
                )
            },
        )

    client = CampaignClient(key, out, plan["deadline_unix"], cap=plan["arm_max_usd"], tick=tick)
    try:
        reason = recheck_gate(root, arm, plan["deadline_unix"])
        if reason:
            raise CampaignStop(reason)
        original_requests = {row["request"]: row for row in read_rows(out / "requests.jsonl")}
        for source in sources:
            for name in ("result", "answer"):
                if sha256(root / source[name]) != source[name + "_sha256"]:
                    raise CampaignStop("A saved primary grading source changed")
            if source["has_final_answer"]:
                original = original_requests.get(source["source_request_number"], {})
                if {name: original.get(name) for name in source["judge_request"]} != source["judge_request"]:
                    raise CampaignStop("Frozen grading parameters differ from the original saved request")
        budget = recheck_budget(study / "started.json", client.usage(), plan["arm_max_usd"])
        client.phase_cap = budget["absolute_cap_usd"]
        if not (study / "protected-files.json").exists():
            files = json.loads((root / "primary-results-manifest.json").read_text())["files"]
            files.update(
                {
                    str(path.relative_to(root)): sha256(path)
                    for name in ("original", "teams")
                    for path in (root / name / "replications").glob("repeat-*/*/result.json")
                }
            )
            atomic_json(study / "protected-files.json", files)
        for replicate in range(1, 4):
            for source in sources:
                if not source["has_final_answer"]:
                    continue
                directory = study / f"replicate-{replicate}"
                directory.mkdir(exist_ok=True)
                path = directory / f"{source['index']:02d}-{source['task_id']}.json"
                state.update(grading_replicate=replicate, task_number=source["index"], task_id=source["task_id"])
                if path.exists():
                    continue
                client.context = {
                    "protocol_version": plan["protocol_version"],
                    "study": "grader_recheck",
                    "phase": "grader_recheck",
                    "epoch": 1,
                    "task_id": source["task_id"],
                    "task_number": source["index"],
                    "grading_replicate": replicate,
                    "source_request_number": source["source_request_number"],
                }
                request = source["judge_request"]
                tick()
                first = max(client.records, default=0) + 1
                response = client.generate(
                    request["prompt"],
                    system_prompt=request["system"],
                    reasoning_effort=request["reasoning_effort"],
                    max_tokens=request["max_tokens"],
                    kind="judge",
                )
                score = score_from_text(response)
                atomic_json(
                    path,
                    {
                        "arm": arm,
                        "task_id": source["task_id"],
                        "index": source["index"],
                        "grading_replicate": replicate,
                        "source_request_number": source["source_request_number"],
                        "primary_score": source["primary_score"],
                        "score": score,
                        "valid": score is not None,
                        "response": response,
                        "request_start": first,
                        "request_end": max(client.records),
                        "time": time.time(),
                    },
                )
                state["grades_completed"] = len(list(study.glob("replicate-*/*.json")))
                tick()
        state.update(status="complete", phase="finished")
    except (CampaignStop, Exception) as exc:
        state.update(status="stopped", error=str(exc).replace(key, "[redacted]")[:500])
    finally:
        state["grades_completed"] = len(list(study.glob("replicate-*/*.json")))
        tick()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        print(
            json.dumps({name: state.get(name) for name in ("status", "error", "grades_completed", "usage")}), flush=True
        )
