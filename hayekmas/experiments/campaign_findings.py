"""Export a readable research summary from saved reports, without API calls."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .campaign_client import atomic_json
from .campaign_task_outcomes import export_task_outcomes


def collect_findings(data, audit, behavior, analysis, now=None):
    now = time.time() if now is None else now
    plan = data["plan"]
    required = plan["test_tasks"]
    maximum = analysis.get("optional_repeated_evaluation", {}).get("maximum_additional_repeats", 2)
    runs = []
    for repeat in [None, *range(1, maximum + 1)]:
        phase = "evaluation" if repeat is None else "replication"
        arms = {}
        for arm, values in data["arms"].items():
            rows = [
                row
                for row in values["episodes"]
                if row["phase"] == phase and row["epoch"] == 1 and row.get("repeat") == repeat
            ]
            arms[arm] = {
                "completed": len(rows),
                "required": required,
                "complete": len(rows) == required,
                "mean_score": sum(row["score"] or 0 for row in rows) / len(rows) if rows else None,
                "passes": sum(bool(row["passed"]) for row in rows),
                "final_answers": sum(bool(row["has_final_answer"]) for row in rows),
                "completed_episode_billed_usd": sum(row["full_episode_cost"] for row in rows),
                "completed_episode_reserved_usd": sum(row["full_episode_unconfirmed_usd"] for row in rows),
                "outcomes": {},
            }
            for row in rows:
                outcome = row.get("diagnostics", {}).get("outcome", "unknown")
                arms[arm]["outcomes"][outcome] = arms[arm]["outcomes"].get(outcome, 0) + 1
        complete = all(arm["complete"] for arm in arms.values())
        pair = next(
            (
                pair
                for pair in data["paired" if repeat is None else "repeated"]
                if pair["epoch"] == 1 and pair.get("repeat") == repeat and pair["complete"]
            ),
            None,
        )
        runs.append(
            {
                "label": "Primary" if repeat is None else f"Repeat {repeat}",
                "repeat": repeat,
                "complete": complete and pair is not None,
                "arms": arms,
                "paired_comparison": {key: value for key, value in pair.items() if key != "tasks"}
                if complete and pair is not None
                else None,
            }
        )
    return {
        "generated_at": now,
        "results_as_of": data["updated_at"],
        "costs_and_audit_as_of": audit["checked_at"],
        "research_window": "ended" if now >= plan["deadline_unix"] else "in progress",
        "paid_execution": data.get("paid_execution"),
        "deadline_utc": plan["deadline_utc"],
        "model": plan["model"],
        "training_tasks_per_arm": plan["primary_train_tasks"],
        "test_tasks_per_run": required,
        "costs": audit["arms"],
        "arm_ceiling_usd": plan["arm_max_usd"],
        "total_ceiling_usd": plan["max_usd"],
        "audit": {key: audit[key] for key in ("failures", "pending")},
        "runs": runs,
        "training_behavior": behavior,
        "diagnostic": data.get("interface_diagnostic", {}),
        "grading_recheck": {arm: values.get("grader_recheck", {}) for arm, values in data["arms"].items()},
        "limits": plan["comparison_limits"],
        "scope": "Each repeat is a separate evaluation of the same trained populations on the same tasks. No pooling as independent tasks, checkpoint selection, or substitution for primary results.",
    }


def render_findings(findings):
    stamp = datetime.fromtimestamp(findings["generated_at"], timezone.utc).isoformat()
    lines = [
        "ORIGINAL EOM / VOLUNTARY TEAMS — RESEARCH FINDINGS",
        f"Generated: {stamp}",
        f"Research window: {findings['research_window']}; deadline {findings['deadline_utc']}.",
        "Window timing alone does not certify experiment completion; coverage and audit are reported below.",
        "",
        f"Model: {findings['model']}. Each arm trains on {findings['training_tasks_per_arm']} bundled scientific tasks; each evaluation uses the same {findings['test_tasks_per_run']} held-out tasks.",
        "Scores use the benchmark's model-judged rubric. Missing final answers count as zero. A score >= 0.5 is a threshold pass, not proof of a fully solved scientific task.",
        "",
    ]
    if findings.get("paid_execution"):
        lines += [
            f"Paid experiments closed at {findings['paid_execution']['closed_at']}. Further research review uses saved artifacts only; STOP remains in place.",
            "",
        ]
    for run in findings["runs"]:
        lines += [
            run["label"].upper() + (" — COMPLETE" if run["complete"] else " — INCOMPLETE"),
            "System       Coverage   Mean score   Passes   Final answers   Completed-episode API cost",
        ]
        for arm, values in run["arms"].items():
            mean = f"{values['mean_score']:.6f}" if values["mean_score"] is not None else "—"
            coverage = f"{values['completed']}/{values['required']}"
            lines.append(
                f"{arm:<12} {coverage:<10} {mean:<12} {values['passes']:<8} {values['final_answers']:<15} ${values['completed_episode_billed_usd']:.6f}"
            )
            lines.append(
                "  Outcomes: "
                + (
                    ", ".join(f"{name}: {count}" for name, count in values["outcomes"].items())
                    or "no committed results"
                )
            )
        pair = run["paired_comparison"]
        if pair:
            counts = pair["wins_ties_losses"]
            lines.append(
                f"Paired tasks: original higher {counts['original_higher']}, tied {counts['tied']}, teams higher {counts['teams_higher']}. Teams minus original mean score: {pair['paired_score_difference']:+.6f}."
            )
            interval = pair.get("intervals")
            if interval:
                low, high = interval["score_difference"]
                lines.append(
                    f"95% paired task-bootstrap interval: [{low:+.6f}, {high:+.6f}]. {interval['interpretation']}"
                )
        else:
            lines.append(
                "Incomplete coverage: displayed means describe completed tasks only; no full-run paired conclusion is reported."
            )
        lines.append("")
    lines += [findings["scope"], "", "CUMULATIVE SPENDING"]
    for arm, values in findings["costs"].items():
        lines.append(
            f"{arm}: ${values['billed_usd']:.6f} billed + ${values['reserved_usd']:.6f} unconfirmed reserved; ${findings['arm_ceiling_usd']:.2f} ceiling."
        )
    lines += [
        "Includes all preflights, interruptions, recoveries, solver repeats, diagnostic calls, and grading rechecks. Unconfirmed reservations count against the ceiling and are not claimed as confirmed bills. Live reservations may settle later. Completed-episode costs above exclude unfinished tasks and the separate grading check; cumulative costs include them.",
        "Equal ceilings do not imply equal realized spending or token use. Agent wealth is a simulation unit, not API dollars.",
        "",
        "TRAINING DYNAMICS",
        "Training here adapts the agent population, wealth, membership, and editable prompts through task experience. It does not fine-tune the underlying model weights. A permitted prompt update is not evidence that an update occurred; actual changes are counted below.",
    ]
    behavior = findings["training_behavior"].get("arms", {})
    original, teams = behavior.get("original", {}), behavior.get("teams", {})
    if original:
        lines.append(
            f"Original EoM: {original['initial_population_size']} initial agents to {original['current_population_size']} after {original['completed_training_tasks']} tasks; {len(original['retained_initial_agent_ids'])} initial IDs retained; {original['distinct_trainable_prompt_texts']} distinct final trainable prompts."
        )
        lines.append(
            "Final role counts: "
            + ", ".join(f"{name}: {count}" for name, count in original["role_counts"].items())
            + "."
        )
    if teams:
        counts = teams["negotiation_counts"]
        lines.append(
            f"Teams: {teams['current_population_size']} agents; {teams['joins']} joins and {teams['leaves']} leaves over {teams['completed_training_tasks']} tasks. {teams['paid_reflections']} paid reflections; {teams['strategy_updates']} strategy edits. Declined reflection decisions: {teams['reflection_decisions'].get('declined', 0)}."
        )
        lines.append(
            f"Negotiations: {counts.get('revised_personal_pledges', 0)} of {counts.get('revisable_personal_pledges', 0)} personal pledges changed between first and final turn; {counts.get('unequal_winning_teams', 0)} winning groups had unequal contributions; {counts.get('winning_teams_with_zero_money_contributor', 0)} included a zero-money contributor."
        )
        lines.append(
            "These describe observed behavior, not motives, stable emergent roles, or a causal performance benefit. Zero money does not imply zero intellectual contribution; initial pledges may already respond to teammates."
        )
    lines += [
        "",
        "TEAM MECHANISM",
        "Agents communicate through a shared scratchpad and negotiate individual pledges. The group bid is their sum. Each member of the winning group pays their own final pledge. The whole paid bid goes equally to members of the previous winning group; the first winning bid burns. Task rewards split equally among the acting team's members. No functional roles or mandatory critique are imposed.",
        "Held-out team tests start from a fresh copy of the trained checkpoint, with formation and reflection disabled. Within-task money flows remain active, then reset. Native original evaluation freezes the serialized population.",
        "",
        "TRAINING-ONLY INTERFACE DIAGNOSTIC",
    ]
    diagnostic = findings["diagnostic"]
    if diagnostic.get("started"):
        lines.append(
            f"Separately reported exploratory study: ${diagnostic['cost_usd']:.6f} billed within a $1 incremental team allowance."
        )
        for variant in diagnostic["variants"]:
            mean = f"{variant['mean_score']:.6f}" if variant["mean_score"] is not None else "—"
            lines.append(
                f"{variant['variant']}: {variant['n']}/3 training tasks, {variant['final_answers']} final answers, mean score {mean}."
            )
            if "strategy_updates" in variant:
                lines.append(
                    f"  {variant['paid_reflections']} paid reflections; {variant['strategy_updates']} saved strategy edits."
                )
        lines.append(
            "Fresh population per variant; three sequential training tasks. Existing mechanics are described more explicitly; no assigned roles, forced contribution, or changed payoffs. This small sequential diagnostic is not a held-out efficacy estimate."
        )
    else:
        lines.append("Not run in this snapshot. Primary and repeated held-out evaluations take priority.")
    lines += [
        "",
        "FIXED-ANSWER GRADING RELIABILITY",
        "Post hoc check of unchanged primary answers with exact saved judge prompts. Primary scores are not replaced. Three additional grades per submitted answer; missing submissions stay zero without a model call. This measures same-model grading response variation, not expert validation or independent training/solver variation.",
    ]
    for arm, check in findings.get("grading_recheck", {}).items():
        if not check.get("started"):
            lines.append(f"{arm}: not run in this snapshot.")
            continue
        lines.append(
            f"{arm}: ${check['billed_usd']:.6f} billed + ${check['reserved_usd']:.6f} reserved, within its $0.50 incremental allowance and existing arm ceiling."
        )
        for row in check["replicates"]:
            outcome = (
                f"mean {row['mean_score']:.6f}, passes {row['passes']}/19"
                if row["complete"]
                else "incomplete; no full-task mean"
            )
            lines.append(
                f"  Grading repetition {row['replicate']}: {row['valid_grades']}/{row['expected_grades']} valid submitted-answer grades; {outcome}."
            )
    lines += [
        "",
        "LIMITS",
        *["- " + limit for limit in findings["limits"]],
        "- Text-only task execution: neither system browses, retrieves papers, or executes scientific code. Promises to verify something are not evidence of external verification.",
        "",
        f"AUDIT: {findings['audit']['failures']} failed checks; {findings['audit']['pending']} pending checks. See audit.json for each requirement and its evidence.",
        "",
        "VIEW AND REGENERATE",
        "Open snapshot.html for the saved interactive viewer. Keep this result folder intact for the raw artifacts, conversations, and detailed round replay links.",
        "Static exports: figures/comparison.pdf, figures/repeat-stability.pdf, figures/evolution-overview.pdf, figures/team-dynamics.pdf, figures/negotiation-dynamics.pdf, figures/team-participation.pdf, figures/grading-variability.pdf.",
        "Detailed reviews: TASK_OUTCOMES.txt, SUBMISSION_REVIEW.txt, TEAM_PARTICIPATION.txt, ARITHMETIC_SPOT_CHECK.txt and RUBRIC_NOTATION_REVIEW.txt.",
        "Additional checks: SYMMETRY_ALGEBRA_REVIEW.txt, PHOTOCHEMISTRY_UNITS_REVIEW.txt and SOURCE_CONSISTENCY_REVIEW.txt. FOLLOWUP_STUDY_DESIGN.txt is a proposal, not another completed experiment.",
        "From the repository, regenerate without API calls (only after an existing report watcher has stopped):",
        "  .venv/bin/python -m hayekmas.experiments.campaign_report --root runs/eom-vs-teams-12h --figures",
        "Serve the saved viewer:",
        "  .venv/bin/python -m http.server 8769 --bind 127.0.0.1 --directory runs/eom-vs-teams-12h",
        "Then open http://127.0.0.1:8769/ and select a system, episode, and detailed team replay.",
        "",
    ]
    return "\n".join(lines)


def export_findings(root):
    read = lambda name: json.loads((root / name).read_text())
    findings = collect_findings(
        read("dashboard-data.json"), read("audit.json"), read("behavior-summary.json"), read("analysis-plan.json")
    )
    atomic_json(root / "research-findings.json", findings)
    temporary = root / "RESEARCH_FINDINGS.tmp.txt"
    temporary.write_text(render_findings(findings))
    temporary.replace(root / "RESEARCH_FINDINGS.txt")
    export_task_outcomes(root, read("dashboard-data.json"))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    export_findings(Path(parser.parse_args().root))


if __name__ == "__main__":
    main()
