"""Describe observed adaptation and verify each team member's wealth accounting."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

from .campaign_client import atomic_json
from .campaign_report import episode_rows, rows
from .campaign_native_trace import native_trace
from .campaign_negotiation import negotiation_trace


def build_behavior(root):
    report = {
        "updated_at": time.time(),
        "scope": "Completed training tasks only. Descriptive evidence, not held-out performance or a causal explanation.",
        "arms": {},
    }
    for arm in ("original", "teams"):
        folder = root / arm
        episodes = [r for r in episode_rows(folder) if r["phase"] == "training"]
        ledger = {r["request"]: r for r in rows(folder / "api_usage.jsonl")}
        costs, requests = defaultdict(float), Counter()
        for record in ledger.values():
            costs[record["kind"]] += record.get("cost_usd", 0)
            requests[record["kind"]] += 1
        summary = {
            "completed_training_tasks": len(episodes),
            "all_campaign_cost_by_call_kind_usd": dict(costs),
            "all_campaign_requests_by_kind": dict(requests),
            "cost_scope": "Includes preflight, interruptions and recovery; does not treat simulation wealth as API dollars.",
            "call_kind_note": "The original client's default solve category includes agent actions and prompt-generation/mutation calls.",
        }
        report["arms"][arm] = summary
        if not episodes:
            continue
        first, last = episodes[0], episodes[-1]
        summary["initial_population_size"] = len(first["population_before"])
        summary["current_population_size"] = len(last["population_after"])
        summary["latest_committed_result"] = str((root / last["artifact"]).parent.relative_to(root) / "result.json")
        if arm == "original":
            initial = {a["id"] for a in first["population_before"]}
            population = last["population_after"]
            lifecycle = Counter()
            for episode in episodes:
                lifecycle.update(native_trace(root / episode["artifact"] / "trajectory.log")["counts"])
            summary.update(
                retained_initial_agent_ids=sorted(initial & {a["id"] for a in population}),
                role_counts=dict(Counter(a["type"] for a in population)),
                spawn_methods=dict(Counter(a.get("spawn_method") for a in population)),
                distinct_trainable_prompt_texts=len({a.get("trainable_system_prompt", "") for a in population}),
                native_logged_lifecycle_counts=dict(lifecycle),
                lifecycle_scope="All native replay trials within committed API attempts. Text-log payment amounts are rounded and are not used as an exact wealth ledger.",
                interpretation="Different prompt texts and population turnover do not themselves establish improved reasoning.",
            )
            continue
        events = [e for r in episodes for e in rows(root / r["artifact"] / "events.jsonl")]
        negotiation_counts = Counter()
        negotiation_errors = []
        first_revision = None
        for episode in episodes:
            trace = negotiation_trace(rows(root / episode["artifact"] / "events.jsonl"))
            negotiation_counts.update(trace["counts"])
            negotiation_errors.extend({"artifact": episode["artifact"], **error} for error in trace["errors"])
            if first_revision is None:
                group = next((g for g in trace["groups"] if g["revised_agents"]), None)
                if group is not None:
                    first_revision = {
                        "task_index": episode["index"],
                        "epoch": episode["epoch"],
                        "artifact": episode["artifact"],
                        **group,
                    }
        reflection = Counter()
        fees = Counter()
        event_counts = Counter(e["event"] for e in events)
        for event in events:
            if event["event"] == "reflection_paid":
                fees[event["agent"]] += event["cost"]
            if event["event"] == "decision_response" and event["phase"] == "reflect":
                try:
                    choice = json.loads(event["response"]).get("reflect")
                except (ValueError, AttributeError):
                    choice = None
                reflection["chosen" if choice is True else "declined" if choice is False else "invalid"] += 1
        initial = {a["name"]: a for a in first["population_before"]}
        agents = []
        for agent in last["population_after"]:
            name = agent["name"]
            paid = math.fsum(r["metrics"]["paid"].get(name, 0) for r in episodes)
            income = math.fsum(r["metrics"]["bid_income"].get(name, 0) for r in episodes)
            rewards = math.fsum(r["metrics"]["reward_income"].get(name, 0) for r in episodes)
            expected = initial[name]["wealth"] - paid + income + rewards - fees[name]
            agents.append(
                {
                    "name": name,
                    "team": agent["team"],
                    "initial_wealth": initial[name]["wealth"],
                    "wealth": agent["wealth"],
                    "bid_paid": paid,
                    "bid_income": income,
                    "reward_income": rewards,
                    "reflection_fees": fees[name],
                    "expected_wealth": expected,
                    "wealth_equation_holds": math.isclose(agent["wealth"], expected, rel_tol=1e-10, abs_tol=1e-8),
                    "strategy_changed_from_initial": agent["strategy"] != initial[name]["strategy"],
                }
            )
        metrics = last["metrics"]
        summary.update(
            negotiation_counts=dict(negotiation_counts),
            negotiation_errors=negotiation_errors,
            first_observed_pledge_revision=first_revision,
            reflection_decisions=dict(reflection),
            paid_reflections=event_counts["reflection_paid"],
            strategy_updates=event_counts["strategy_updated"],
            distinct_strategy_texts=len({a["strategy"] for a in last["population_after"]}),
            message_counts=dict(Counter(e["channel"] for e in events if e["event"] == "team_message")),
            joins=event_counts["joined"],
            leaves=event_counts["left"],
            simulation_totals={
                k: metrics[k]
                for k in (
                    "reward_total",
                    "bid_paid_total",
                    "bid_transfer_total",
                    "bid_burn_total",
                    "reflection_burn_total",
                )
            },
            agents=agents,
            all_personal_wealth_equations_hold=all(a["wealth_equation_holds"] for a in agents),
            interpretation="Membership, wealth, and memory can change even when agents decline paid strategy editing. Motives cannot be inferred from a boolean reflection response.",
        )
    atomic_json(root / "behavior-summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    summary = build_behavior(parser.parse_args().root)
    print(
        json.dumps(
            {
                a: {
                    k: v
                    for k, v in s.items()
                    if k
                    in (
                        "completed_training_tasks",
                        "reflection_decisions",
                        "distinct_trainable_prompt_texts",
                        "all_personal_wealth_equations_hold",
                    )
                }
                for a, s in summary["arms"].items()
            },
            indent=2,
        )
    )
