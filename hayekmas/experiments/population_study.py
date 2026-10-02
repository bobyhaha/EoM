"""Prepare an offline population-scaling study and transparent API cost scenarios.

This module has no model client and no paid execution entry point. Cell files
are study specifications, not configs accepted by the legacy campaign launcher.
"""

import argparse
from dataclasses import asdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import load_tasks


REPO = Path(__file__).resolve().parents[2]
KS = (10, 20, 50, 100)
PRICE = {"input_per_million": 0.10, "output_per_million": 0.50,
         "verified_date": "2026-10-01", "model": "openai/gpt-6-luna",
         "source": "https://openrouter.ai/openai/gpt-6-luna/providers"}
# Mean input/output tokens are explicit planning assumptions, not measurements
# of the new protocol. Output includes hidden reasoning billed by the provider.
SCENARIOS = {
    "low": dict(formation_base=3000, roster_per_agent=40, recap_base=4000,
                chat_input=4000, bid_input=5000, control_output=40, chat_output=128,
                bid_output=80, work_input=8000, work_output=1500,
                final_input=10000, final_output=2500, judge_input=5000, judge_output=1000,
                reflection_fraction=0, invitation_reply_fraction=.2, wake_fraction=.35, trials=1, birth_fraction=.1),
    "central": dict(formation_base=6500, roster_per_agent=80, recap_base=6500,
                    chat_input=8000, bid_input=8000, control_output=80, chat_output=256,
                    bid_output=160, work_input=12000, work_output=2500,
                    final_input=16000, final_output=5000, judge_input=8000, judge_output=1500,
                    reflection_fraction=.1, invitation_reply_fraction=.5, wake_fraction=.6, trials=1.5, birth_fraction=.35),
    "high": dict(formation_base=12000, roster_per_agent=200, recap_base=12000,
                 chat_input=14000, bid_input=14000, control_output=256, chat_output=512,
                 bid_output=320, work_input=20000, work_output=4000,
                 final_input=28000, final_output=8000, judge_input=12000, judge_output=3000,
                 reflection_fraction=.5, invitation_reply_fraction=1, wake_fraction=1, trials=2, birth_fraction=1),
}

# Separate team-only assumptions; upstream and reference-baseline budgets stay
# unchanged. These are planned mean tokens, not claims that larger caps are used.
SOLVE_FIRST_SCENARIOS = {
    "low": dict(formation_base=1500, roster_per_agent=12, control_output=40, bid_input=768, bid_output=32,
                chat_input=4000, chat_output=512, work_input=12000, work_output=3000,
                final_input=16000, final_output=6000),
    "central": dict(formation_base=2200, roster_per_agent=16, control_output=64, bid_input=1280, bid_output=48,
                    chat_input=8000, chat_output=1024, work_input=16000, work_output=6000,
                    final_input=24000, final_output=10000),
    "high": dict(formation_base=4000, roster_per_agent=16, control_output=128, bid_input=2900, bid_output=80,
                 chat_input=14000, chat_output=1536, work_input=24000, work_output=10000,
                 final_input=32000, final_output=14000),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def priced(calls, inputs, outputs):
    return {"calls": calls, "input_tokens": calls * inputs, "output_tokens": calls * outputs,
            "usd": calls * (inputs * PRICE["input_per_million"] + outputs * PRICE["output_per_million"]) / 1e6}


def estimate(k, scenario="central", *, steps=10, train=40, test=19, schedule="compact", token_profile=None):
    """One seed; all steps used, average winning team size three (cap four)."""
    s, episodes, m = SCENARIOS[scenario], train + test, 3
    if schedule not in {"compact", "full"}:
        raise ValueError("Unknown round schedule")
    token_profile = token_profile or ("solve_first" if schedule == "compact" else "standard")
    if token_profile not in {"standard", "solve_first"} or (token_profile == "solve_first" and schedule != "compact"):
        raise ValueError("Invalid token profile for schedule")
    roster = s["roster_per_agent"] * k
    team = {
        "membership_actions": priced(episodes * steps * 3 * k, s["formation_base"] + roster, s["control_output"]),
        "opening_discussion": priced(episodes * steps * k, s["recap_base"] + roster, s["chat_output"]),
        "private_discussion": priced(episodes * steps * 2 * k, s["chat_input"], s["chat_output"]),
        "bid_negotiation": priced(episodes * steps * 2 * k, s["bid_input"], s["bid_output"]),
        "winning_work": priced(episodes * (steps - 1) * 2 * m, s["work_input"], s["work_output"]),
        "votes": priced(episodes * steps * m, s["work_input"] + 2000, 20),
        "final_proposals": priced(episodes * m, s["final_input"], s["final_output"]),
        "judge": priced(episodes, s["judge_input"], s["judge_output"]),
        "reflection_choices": priced(train * k, 1000, 16),
        "reflection_edits": priced(train * k * s["reflection_fraction"] * 2, 4000, 1000),
    }
    if schedule == "compact":
        team.pop("membership_actions")
        team.pop("opening_discussion")
        team.pop("bid_negotiation")
        team.update({
            "membership_proposals": priced(episodes * steps * k, s["formation_base"] + roster, s["control_output"]),
            "invitation_replies": priced(episodes * steps * k * s["invitation_reply_fraction"],
                                         s["formation_base"] + roster, s["control_output"]),
            "private_discussion": priced(episodes * steps * k, s["chat_input"], s["chat_output"]),
            "act_and_pledge": priced(episodes * steps * k, s["bid_input"], s["bid_output"]),
        })
        if token_profile == "solve_first":
            p = SOLVE_FIRST_SCENARIOS[scenario]
            membership_input = min(6144, p["formation_base"] + p["roster_per_agent"] * k)
            team.update({
                "membership_proposals": priced(episodes * steps * k, membership_input, p["control_output"]),
                "invitation_replies": priced(episodes * steps * k * s["invitation_reply_fraction"],
                                             membership_input, p["control_output"]),
                "act_and_pledge": priced(episodes * steps * k, p["bid_input"], p["bid_output"]),
                "private_discussion": priced(episodes * steps * k, p["chat_input"], p["chat_output"]),
                "winning_work": priced(episodes * (steps - 1) * 2 * m, p["work_input"], p["work_output"]),
                "votes": priced(episodes * steps * m, p["work_input"] + 2000, 20),
                "final_proposals": priced(episodes * m, p["final_input"], p["final_output"]),
            })
    trials = train * s["trials"] + test
    original = {
        "wakeups": priced(trials * steps * k * s["wake_fraction"], s["chat_input"] / 2, 60),
        "actions": priced(trials * steps, s["work_input"] / 2, s["work_output"]),
        "judge": priced(trials, s["judge_input"], s["judge_output"]),
        "mutations": priced(train * k * s["birth_fraction"], s["work_input"], s["work_output"]),
    }
    result = {arm: {"phases": phases, **{key: math.fsum(p[key] for p in phases.values())
                                      for key in ("calls", "input_tokens", "output_tokens", "usd")}}
            for arm, phases in (("original", original), ("teams", team))}
    solving = math.fsum(team[p]["usd"] for p in ("private_discussion", "winning_work", "final_proposals"))
    result["teams"]["solving_usd"] = solving
    result["teams"]["solving_share"] = solving / result["teams"]["usd"]
    return result


def estimate_growing(k, scenario="central", *, steps=10, train=40, test=19, at_cap=False):
    """Budget scenario: +2 births every 5 tasks, capped at 2k, plus replacement-call allowance.

    Population path assumes successful births and no net attrition. Replacement
    rates remain assumptions; this is not an accuracy or spending guarantee.
    at_cap is a conservative population sensitivity, not a hard dollar ceiling.
    """
    phases = {"original": {}, "teams": {}}
    population = k
    trajectory = []
    periodic_births = 0
    for task in range(train + test):
        training = task < train
        size = 2 * k if at_cap else population
        trajectory.append(size)
        row = estimate(size, scenario, steps=steps, train=int(training), test=int(not training))
        for arm in phases:
            for phase, amounts in row[arm]["phases"].items():
                target = phases[arm].setdefault(phase, dict.fromkeys(amounts, 0.0))
                for key, value in amounts.items():
                    target[key] += value
        if training and (task + 1) % 5 == 0:
            count = min(2, 2 * k - population)
            population += count
            periodic_births += count
    s = SCENARIOS[scenario]
    training_agent_tasks = sum(trajectory[:train])
    phases["original"]["periodic_mutations"] = priced(periodic_births, s["work_input"], s["work_output"])
    phases["teams"]["birth_mutations"] = priced(
        periodic_births + training_agent_tasks * s["birth_fraction"], 8000, 1024)
    result = {arm: {"phases": ps, **{key: math.fsum(p[key] for p in ps.values())
                                   for key in ("calls", "input_tokens", "output_tokens", "usd")}}
              for arm, ps in phases.items()}
    solving = math.fsum(phases["teams"][p]["usd"] for p in ("private_discussion", "winning_work", "final_proposals"))
    result["teams"].update(solving_usd=solving, solving_share=solving / result["teams"]["usd"])
    result["population_assumption"] = {"initial": k, "cap": 2*k, "counts_before_tasks": trajectory,
                                       "periodic_births": periodic_births, "at_cap_sensitivity": at_cap,
                                       "replacement_calls_per_agent_task": s["birth_fraction"]}
    return result


def reference_baselines(scenario="central", test=19):
    s = SCENARIOS[scenario]
    # One pool of 100 independent samples/task supplies single-agent and every
    # pass@k estimate; do not pay for separate 10+20+50+100 pools.
    solve = priced(test * 100, 2500, s["final_output"])
    judge = priced(test * 100, s["final_output"] + s["judge_input"], s["judge_output"])
    return {"calls": solve["calls"] + judge["calls"], "usd": solve["usd"] + judge["usd"]}


def prior_spend(repo=REPO):
    """Deduplicate copied receipts and retain all unresolved reservations."""
    sources = ["eom-vs-teams-12h", "teams-finalization-19-20260930",
               "teams-reviewed-19-20260930", "teams-reviewed-19-v2-20260930"]
    charged, uncertain, manifests = {}, {}, []
    for source in sources:
        for path in sorted((repo / "runs" / source).rglob("api_usage.jsonl")):
            rows = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                rows[row["request"]] = row
            manifests.append({"path": str(path.relative_to(repo)), "sha256": sha(path)})
            for row in rows.values():
                if "cost_usd" in row:
                    identity = row.get("generation_id") or (str(path), row["request"])
                    charged[identity] = max(charged.get(identity, 0), row["cost_usd"])
                else:
                    identity = (row.get("time"), row.get("reserved_usd"), row.get("kind"))
                    uncertain[identity] = max(uncertain.get(identity, 0), row.get("reserved_usd", 0))
    billed, reserved = math.fsum(charged.values()), math.fsum(uncertain.values())
    return {"billed_usd": billed, "reserved_usd": reserved, "existing_limit_usd": 50,
            "remaining_usd": max(0, 50 - billed - reserved), "receipt_sources": manifests,
            "scope": "Recorded $50 comparison campaign and subsequent team follow-ups; earlier separate $5 exploration excluded."}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "cells").mkdir()
    dataset = REPO / "third_party/benchmarks/frontier-science-research/data"
    tasks = {split: load_tasks(dataset / f"research_{split}.jsonl", split) for split in ("train", "test")}
    if len(tasks["train"]) != 40 or len(tasks["test"]) != 19:
        raise ValueError("Expected the bundled 40/19 split")
    if {t.id for t in tasks["train"]} & {t.id for t in tasks["test"]}:
        raise ValueError("Training/test task overlap")
    dataset_hashes = {split: sha(dataset / f"research_{split}.jsonl") for split in tasks}
    estimates = {s: {str(k): estimate_growing(k, s) for k in KS} for s in SCENARIOS}
    totals = {}
    for s, rows in estimates.items():
        core = math.fsum(v[arm]["usd"] for v in rows.values() for arm in ("original", "teams"))
        reference = reference_baselines(s)
        totals[s] = {"core_usd": core, "reference_usd": reference["usd"],
                     "one_seed_with_20_percent_allowance": (core + reference["usd"]) * 1.2,
                     "three_seeds_with_20_percent_allowance": (core + reference["usd"]) * 3 * 1.2,
                     "calls": math.fsum(v[arm]["calls"] for v in rows.values()
                                        for arm in ("original", "teams")) + reference["calls"]}
    cells = []
    for k in KS:
        team_config = TeamConfig(interaction_protocol="rounds", round_schedule="compact", token_profile="solve_first",
                                 coordination_fee_lambda=0.0, evolution_enabled=True, population_cap_multiplier=2,
                                 num_agents=k, rounds=40, max_steps=10, formation_context_tokens=max(6144, 128*k),
                                 formation_turns=3, discussion_turns=2, bidding_turns=2, max_team_size=4,
                                 action_tokens=192, bid_tokens=96, discussion_tokens=2048,
                                 solution_tokens=16384, judge_tokens=8192,
                                 inspect_tokens=1024, update_tokens=1024, summary_tokens=512,
                                 environment_tokens=16384, evidence_tokens=8192, context_tokens=262144,
                                 max_calls=2_000_000)
        for arm in ("original", "teams"):
            spec = {"schema": "population-study-cell-v1", "status": "prepared_not_authorized_to_run",
                    "arm": arm, "k": k, "seeds": [7, 17, 29], "initial_review_scope": "one seed (7)",
                    "training_tasks": 40, "test_tasks": 19, "max_steps": 10,
                    "model": PRICE["model"], "solver_reasoning": "high" if arm == "teams" else "medium",
                    "control_reasoning": "none", "team_discussion_reasoning": "medium" if arm == "teams" else None,
                    "dataset_sha256": dataset_hashes,
                    "budget": {"execution_authorized": False, "allocated_usd": 0},
                    "team_config": asdict(team_config) if arm == "teams" else None,
                    "original_initialization": {"agents_per_role": k // 5,
                                                "method": "Fresh instances of each of the five upstream research roles; unique IDs and names."}
                    if arm == "original" else None,
                    "original_overrides": {"engine": {"min_num_agents": 0, "max_num_agents": 2*k,
                                                       "birth_interval": 5, "num_births_per_interval": 2,
                                                       "rent": 0, "max_trials_per_episode": 2,
                                                       "max_steps_per_episode": 10},
                                           "evolution": {"p_a": 0.0, "p_b": 1.0, "periodical_good_p": 0.5},
                                           "wakeup": {"wakeup_model": None, "wakeup_parallel_enabled": False},
                                           "evaluation": {"periodic_test_enabled": False}}
                    if arm == "original" else None,
                    "evaluation": "Fresh copy of final 40-task checkpoint for every test. Original eval freezes wealth and prompts; "
                    "teams retain within-episode auctions/membership/rewards but discard all test state afterward. No test reflection or population evolution (training=False).",
                    "required_prelaunch_checks": ["Construct exactly k agents; verify unique identities and upstream role coverage.",
                                                  "Honor team PHASE_REASONING settings, phase output caps and control input limits; "
                                                  "record provider-confirmed cost and reasoning tokens per phase.",
                                                  "Start with k agents, cap both arms at 2k; record births/deaths/counts. Stop if population becomes empty.",
                                                  "Reserve worst-case next-request cost in restart-safe per-cell and global ledgers.",
                                                  "Verify source and dataset hashes; do not reuse checkpoints trained under old team rules.",
                                                  "Use a new launcher with this schema; the legacy campaign hard-codes population/budget settings."]}
            filename = f"{arm}-k{k}.json"
            write_json(destination / "cells" / filename, spec)
            cells.append({"arm": arm, "k": k, "file": f"cells/{filename}"})
    plan = {"schema": "population-study-plan-v1", "prepared_at": datetime.now(timezone.utc).isoformat(),
            "status": "prepared_only_no_paid_calls", "execution_authorized": False,
            "population_sizes": list(KS), "seeds": [7, 17, 29], "initial_review_seed": 7,
            "tasks": {split: [{"id": t.id, "subject": t.subject} for t in ts] for split, ts in tasks.items()},
            "dataset_sha256": dataset_hashes, "cells": cells,
            "core_episodes_per_seed": 8 * 59, "core_episodes_three_seeds": 8 * 59 * 3,
            "round_sequence": ["One public membership proposal per agent", "Targeted replies to valid invitations",
                               "One private discussion turn per member", "One act vote + personal pledge; strict majority activates team",
                               "Select and charge winning team", "Winning team publishes one action"],
            "coordination_fee": {"lambda": 0.0, "rule": "Per member per discussion round: lambda * (team size - 1).",
                                 "first_study": "Disabled: no charge and no extra consent calls.",
                                 "future_ablation": "Positive lambda requires consent before payment. Declining or unaffordable "
                                 "members leave; recompute fees for remaining groups. Burn fees even if the group loses. "
                                 "Future positive-fee studies need extra cost estimates for consent calls."},
            "reward": "Grade the final answer once. N counts accepted public actions including finalization. "
                      "Every saved member receives R/N for each contributed round; no team-size division.",
            "population_interpretation": "k is the initial population; both arms cap living agents at 2k. Both attempt two births every five training "
                "tasks, with p_a=0, p_b=1, periodic good probability=0.5 and rent=0. Team removals occur only after "
                "settlement; newborns start solo. No minimum-population replenishment is requested. Native original "
                "role preservation and bankruptcy-triggered trial replay remain; teams have neither. "
                "Counts can differ across arms and need reporting at every task. Evaluation disables evolution.",
            "baselines": "In addition to eight trained cells, one pool of 100 independent zero-shot answers per test "
                "supplies single-agent and pass@10/20/50/100 estimates. Report success using 1-C(100-c,k)/C(100,k). "
                "Same rubric threshold 0.5; report full continuous scores too. This is an oracle success metric, "
                "not an implemented best-answer selector, and has different training exposure.",
            "comparison_limits": ["Same tasks/model/step ceiling and initial k, cap 2k; final populations, tokens and dollars may differ.",
                                  "Team solve-first uses high reasoning and 16,384 output caps for public work/final proposals; "
                                  "the original baseline remains unchanged. This is not a matched-compute comparison.",
                                  "Both arms evolve editable strategies; native roles and bankruptcy replay differ from role-free episode-boundary team evolution. Team nonnegative rewards and affordable pledges with rent=0 may produce no bankruptcies.",
                                  "Training means economic/strategy adaptation, not model-weight fine tuning.",
                                  "These 19 test tasks have been inspected before; label this a development comparison.",
                                  "No causal claim about communication alone; reward, selection and finalization also differ.",
                                  "Team-size cap four is an engineering constraint, not evidence of an optimal team size.",
                                  "The first study sets coordination lambda to zero; positive fee ablations are not scheduled.",
                                  "Compact activation requires a strict majority. Ties abstain; no forced unfunded finalization. "
                                  "Measure no-bid rounds and final-answer failures; lower spend alone is not an improvement.",
                                  "Track environment score separately from amplified wealth; larger payouts are not better answers."],
            "metrics": ["paired task score and completion rate", "pass rate at fixed threshold", "billed and reserved cost",
                        "request/token counts by phase", "team activation votes and abstentions", "team size and membership switches per round", "membership reasons",
                        "pledges, payments and path credits per agent", "actual strategy edits and population changes"],
            "pricing": PRICE, "cost_scenarios": SCENARIOS, "solve_first_scenarios": SOLVE_FIRST_SCENARIOS,
            "estimates": estimates, "totals": totals,
            "growth_budget_assumption": "Successful +2 periodic births every 5 tasks until 2k, no net attrition; "
                "additional mutation-call allowance uses assumed replacement rates. Costs are scenarios, not hard caps.",
            "at_cap_cost_sensitivity": {str(k): estimate_growing(k, at_cap=True) for k in KS},
            "token_allocation": {"profile": "solve_first", "solving_phases": ["private_discussion", "winning_work", "final_proposals"],
                                 "goal": "Most planned model cost funds substantive problem-solving opportunities.",
                                 "limits": "Caps are ceilings, not spending targets. Phase labels measure opportunities, not usefulness. "
                                 "Actual shares require billed usage; no accuracy improvement is established.",
                                 "act_pledge": "96 output tokens, reasoning none, input at most 3072 reference tokens, "
                                 "using latest team messages and balances instead of the full task/history.",
                                 "membership": "192 output tokens; reasoning none; input at most 6144 reference tokens with "
                                 "bounded public-work/profile excerpts and complete roster identities; input cap scales to max(6144, 128*k) to accommodate 2k agents.",
                                 "private_discussion": "2048 output tokens; medium reasoning; full task and shared solution.",
                                 "public_work_and_final": "16384 output tokens; high reasoning; no assigned roles.",
                                 "selection": "64 output tokens; reasoning none; candidate text retained."},
            "prior_spending": prior_spend(),
            "schedule_comparison": {str(k): {"full": estimate(k, schedule="full")["teams"],
                                             "compact": estimate(k, token_profile="standard")["teams"],
                                             "solve_first": estimate(k)["teams"]} for k in KS},
            "value_ablation_plan": [
                "Measure final environment score and completion per dollar, not amplified agent wealth.",
                "Compare compact and full schedules on paired training tasks/seeds; the comparison bundles call schedule, "
                "activation rule and final recovery changes, so it does not isolate one cause.",
                "In separate controlled ablations hold activation/finalization/reward fixed and vary membership windows, "
                "pre-bid discussion length or repeated pledges one at a time. Preserve winning-work budgets.",
                "To estimate an accepted round's marginal value, omit its public work and rerun the suffix on matched seeds. "
                "Equal R/N payments and a final grade alone do not provide causal attribution.",
                "Ablations are proposals only; not included in the eight-cell budget and not authorized to run."],
            "estimate_limits": "Scenario calculations, not confidence intervals or spending guarantees. Assume ten rounds "
                "with an active winning team every round. Compact uses one membership proposal, one private message and "
                "one act/pledge ballot per agent; invitation reply fractions are 0.2/0.5/1.0 in low/central/high scenarios "
                "and are unmeasured assumptions. Solve-first token means are listed separately; output means include "
                "provider-billed reasoning. No abstention or cache discount. Winning work remains two turns. A 20% planning "
                "allowance covers some retries/corrections, not an absolute bound. Calibrate on training tasks before launch.",
            "runtime": f"At the old team's median 1.85 seconds between request starts, the k=100 compact team cell alone "
                f"projects roughly {estimates['central']['100']['teams']['calls'] * 1.85 / 86400:.1f} days per seed if kept sequential. "
                "This is not a 12-hour study. Parallel independent "
                "cells do not remove sequential dependencies within membership and shared discussions.",
            "next_execution_work": "Build a launcher that consumes these cell specs and enforces aggregate authorization. "
                "Preparation deliberately has no API client or launch operation.",
            "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / "hayekmas").rglob("*.py"))}}
    write_json(destination / "plan.json", plan)
    write_json(destination / "cost-estimate.json", {"pricing": PRICE, "assumptions": SCENARIOS,
                                                  "solve_first_assumptions": SOLVE_FIRST_SCENARIOS,
                                                  "cells": estimates, "totals": totals})
    shutil.copy2(__file__, destination / "estimator-source.py")
    render(destination, plan)
    return plan


def render(root, plan):
    totals = plan["totals"]
    rows = []
    for k in KS:
        scenarios = plan["estimates"]
        cells = [scenarios[s][str(k)] for s in ("low", "central", "high")]
        cost = lambda arm: [c[arm]["usd"] for c in cells]
        orig, team = cost("original"), cost("teams")
        rows.append(f'<tr><td>{k}</td><td>${orig[1]:.0f} <small>(${orig[0]:.0f}–${orig[2]:.0f})</small></td>'
                    f'<td>${team[1]:.0f} <small>(${team[0]:.0f}–${team[2]:.0f})</small></td>'
                    f'<td>{cells[1]["teams"]["calls"]:,.0f}</td>'
                    f'<td><a href="cells/original-k{k}.json">Original</a> · '
                    f'<a href="cells/teams-k{k}.json">Teams</a></td></tr>')
    safe = json.dumps(totals).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EoM population study · proposed budget</title><style>
body{background:#101720;color:#e5edf7;font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 22px}
h1{line-height:1.2;font-size:32px}h2{font-size:21px}a{color:#8bdccc}small,.muted{color:#aabbcf}
.badge{color:#f0c279}.card{padding:20px;background:#192330;border:1px solid #344354;border-radius:12px;margin:20px 0}
.flow{display:flex;flex-wrap:wrap;gap:10px}.flow span{padding:10px;border:1px solid #46586d;border-radius:8px}
table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #354454}
.scroll{overflow:auto}select{font:inherit;padding:5px;background:#263649;color:white}#total{font-size:30px;color:#8bdccc}
li{margin-bottom:8px}.bar{height:18px;background:#8bdccc;border-radius:3px;min-width:2px}figure{margin:12px 0}
</style><body><p class="badge">PREPARED ONLY · NO PAID RUNS LAUNCHED</p>
<h1>Initial population: 10, 20, 50, 100 · cap 2k</h1><p>Both arms start at k and may grow to 2k. Two births are attempted every five training tasks. Same 40 training tasks and 19 test tasks for each trained system.
One round ends after the winning team's public action. One final answer receives one grade.
Coordination fee λ = 0 for this first study; fee ablations are deferred.</p>
<div class="flow">'''
    html += ''.join(f'<span>{i+1}. {escape(text)}</span>' for i, text in enumerate(plan["round_sequence"]))
    html += '''</div><div class="card"><label>Planning scenario <select id="scenario"><option value="low">Low</option>
<option value="central" selected>Central</option><option value="high">High</option></select></label> ·
<label>Independent seeds <select id="seeds"><option value="1">1</option><option value="3">3</option></select></label>
<p id="total"></p><p id="scope"></p><small>Includes independent single-agent/pass@k reference samples and a 20% planning allowance.
These are scenario estimates, not measured costs or spending authorization.</small></div>
<h2>Per population, one seed</h2><p class="muted">Central estimate (low–high); raw model cost before the shared allowance and reference baselines.</p>
<div class="scroll"><table><thead><tr><th>k</th><th>Original EoM variant</th><th>Round teams</th><th>Team requests</th><th>Cell specs</th></tr></thead><tbody>'''
    html += ''.join(rows) + '</tbody></table></div><h2>Why the team estimate grows</h2>'
    html += '<p>The compact schedule uses 3n + I calls before the winning action, where n is the CURRENT living population: n membership proposals, I valid invitation '
    html += 'replies (0 ≤ I ≤ n), n private discussion messages, and n combined act/pledge ballots. No assigned leader. '
    html += 'A strict majority activates each team; only members can authorize their own money. '
    html += 'Starting at k=10 this is 30–40 calls per round; at the 20-agent cap it is 60–80. The solve-first allocation keeps the same number of calls, '
    html += 'bounds control context and doubles public-work/final-proposal output caps from 8,192 to 16,384 tokens.</p>'
    before = plan['schedule_comparison']['10']['compact']['usd']
    after = plan['schedule_comparison']['10']['solve_first']['usd']
    html += f'<p>Historical fixed-10-agent team allocation, central estimate: <strong>${before:.2f} → ${after:.2f}</strong> '
    html += '(previous compact → solve-first; 40 training + 19 test tasks; one seed; before allowance). These are projections.</p>'
    allocation = plan['estimates']['central']['10']['teams']
    html += f'<div class="card"><strong>{allocation["solving_share"]:.0%} of planned k=10 team cost funds problem-solving calls</strong>'
    html += f'<p>${allocation["solving_usd"]:.2f} for substantive private discussion, winning public work and complete final proposals. '
    html += 'Selection votes are counted separately. Actual billed shares and usefulness still need measurement.</p></div>'
    html += '<p>' + escape(plan['growth_budget_assumption']) + '</p>'
    html += '<p>Every compact team agent is told: “Spend most of your reasoning and response-token budget on useful contributions '
    html += 'to solving the problem. Keep membership, act/abstain, and pledge decisions brief.” This concerns computation, not money pledged.</p>'
    html += '<ul>' + ''.join(f'<li>{escape(plan["token_allocation"][key])}</li>'
                           for key in ('act_pledge', 'membership', 'private_discussion', 'public_work_and_final', 'selection')) + '</ul>'
    full = plan['schedule_comparison']['10']['compact']['phases']
    compact = plan['schedule_comparison']['10']['solve_first']['phases']
    cost_rows = [
        ('Formation and public messages', ['membership_proposals', 'invitation_replies'],
         ['membership_proposals', 'invitation_replies']),
        ('Private pre-bid discussion', ['private_discussion'], ['private_discussion']),
        ('Bidding / act decision', ['act_and_pledge'], ['act_and_pledge']),
        ('Winning work + answer proposals + selection', ['winning_work', 'final_proposals', 'votes'],
         ['winning_work', 'final_proposals', 'votes']),
        ('Grading + reflection', ['judge', 'reflection_choices', 'reflection_edits'],
         ['judge', 'reflection_choices', 'reflection_edits']),
    ]
    html += '<div class="scroll"><table><thead><tr><th>k=10 phase</th><th>Previous compact</th><th>Solve-first</th></tr></thead><tbody>'
    for label, old_keys, new_keys in cost_rows:
        old_cost = math.fsum(full[p]['usd'] for p in old_keys)
        new_cost = math.fsum(compact[p]['usd'] for p in new_keys)
        html += f'<tr><td>{label}</td><td>${old_cost:.2f}</td><td>${new_cost:.2f}</td></tr>'
    html += '</tbody></table></div>'
    html += '<h2>What is worth spending on?</h2><p>Working hypothesis: prioritize substantive reasoning and '
    html += 'answer checking. An equal R/N payout is an incentive rule, '
    html += 'not evidence that each accepted round was equally useful. No paid accuracy result is available for this schedule.</p><ul>'
    html += ''.join(f'<li>{escape(item)}</li>' for item in plan['value_ablation_plan']) + '</ul>'
    for k in KS:
        value = plan["estimates"]["central"][str(k)]["teams"]["usd"]
        maximum = plan["estimates"]["central"]["100"]["teams"]["usd"]
        html += f'<figure><figcaption>k={k}: ${value:.0f}</figcaption><div class="bar" style="width:{value/maximum*100:.1f}%"></div></figure>'
    spent = plan["prior_spending"]
    html += f'<div class="card">Recorded prior spend: ${spent["billed_usd"]:.2f}; unresolved reservations: ${spent["reserved_usd"]:.2f}. '
    html += f'Remaining under the existing $50 limit: <strong>${spent["remaining_usd"]:.2f}</strong>. No new spending is authorized by this plan.</div>'
    html += '<h2>How to interpret this study</h2><ul>'
    html += ''.join(f'<li>{escape(t)}</li>' for t in [plan["population_interpretation"], plan["baselines"], *plan["comparison_limits"], plan["runtime"]])
    html += '</ul><p>For each test, restore a fresh trained checkpoint. Save every membership decision, message, bid, action, '
    html += 'final grade and per-agent payment. Show completion failures separately and never count API interruptions as wrong answers.</p>'
    html += '<p><a href="plan.json">Full study plan</a> · <a href="cost-estimate.json">All arithmetic and assumptions</a> · '
    html += f'<a href="{PRICE["source"]}">OpenRouter pricing verified October 1</a> · '
    html += '<a href="http://127.0.0.1:8786/replay.html">Current scripted mechanism replay</a></p>'
    html += f'<script id="cost-data" type="application/json">{safe}</script>'
    html += '''<script>const costs=JSON.parse(document.getElementById('cost-data').textContent);
function update(){const seed=Number(document.getElementById('seeds').value),s=document.getElementById('scenario').value;
document.getElementById('total').textContent='$'+Math.round(costs[s].one_seed_with_20_percent_allowance*seed).toLocaleString();
document.getElementById('scope').textContent=(472*seed).toLocaleString()+' trained-system episodes · approximately '+Math.round(costs[s].calls*seed).toLocaleString()+' API requests';}
document.getElementById('scenario').onchange=update;document.getElementById('seeds').onchange=update;update();</script></body></html>'''
    (root / "index.html").write_text(html)
    lines = ["POPULATION STUDY — PREPARED ONLY, NO PAID EXECUTION", "",
             "k = 10, 20, 50, 100. Each cell: 40 training tasks + 19 test tasks; 10 decision rounds.",
             "Two trained arms per k; eight cells and 472 episodes per seed.",
             "One-seed estimates include a 100-sample/task single-agent/pass@k pool and 20% allowance:"]
    lines += [f"  {s}: ${t['one_seed_with_20_percent_allowance']:.0f}; three seeds ${t['three_seeds_with_20_percent_allowance']:.0f}"
              for s, t in totals.items()]
    lines += ["", plan["estimate_limits"], "", plan["runtime"], "", plan["population_interpretation"],
              "", plan["reward"], "", *plan["comparison_limits"], "", plan["next_execution_work"],
              "Open index.html for the interactive budget and all cell specifications."]
    (root / "STUDY_PLAN.txt").write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = prepare(args.out)
    print(json.dumps({"out": str(Path(args.out).resolve()), "status": plan["status"], "totals": plan["totals"]}, indent=2))


if __name__ == "__main__":
    main()
