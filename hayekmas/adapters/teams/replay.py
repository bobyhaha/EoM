"""A self-contained replay, refreshed atomically after each completed episode."""

import json
from collections import defaultdict
from pathlib import Path


VISIBLE = {
    "decision_round_started", "membership_window", "membership_committed", "membership_decision", "activation",
    "path_reward", "finalization_recovery", "selection_fallback",
    "coordination_fee", "coordination_decision",
    "invited",
    "joined",
    "left",
    "team_dissolved",
    "team_message",
    "pledge",
    "contributions_committed",
    "auction",
    "bid_transfer",
    "candidate",
    "vote",
    "submission",
    "no_submission",
    "finalization_started",
    "finalization_abstained",
    "discussion_shortened",
    "phase_skipped",
    "settlement",
    "reflection_paid",
    "strategy_updated",
    "invalid_action",
    "review_started",
    "review_draft",
    "review_feedback",
    "review_revision",
    "review_fallback",
    "review_abstained",
    "funding_repair_started",
}


def replay_data(engine, status):
    visible_by_round = defaultdict(list)
    for item in engine.events:
        kind = item["event"]
        if (
            kind in VISIBLE
            or (kind == "message" and item.get("channel") in {"formation", "invitation"})
            or (kind == "decision_response" and item.get("phase") == "inspect")
        ):
            visible_by_round[item["round"]].append(item)
    rounds = []
    for event in engine.events:
        if event["event"] != "round_complete":
            continue
        episode = event["round"]
        rounds.append(
            {
                "round": episode,
                "metrics": event["metrics"],
                "agents": event["agents"],
                "teams": event["teams"],
                "events": visible_by_round[episode],
            }
        )
    return {
        "schema": 1,
        "status": status,
        "backend": engine.policy.label,
        "condition": engine.config.condition,
        "interaction_protocol": engine.config.interaction_protocol,
        "coordination_fee_lambda": engine.config.coordination_fee_lambda,
        "bidding_mode": engine.config.bidding_mode,
        "collaboration_mode": engine.config.collaboration_mode,
        "planned_rounds": engine.config.rounds,
        "initial_wealth": engine.config.initial_wealth,
        "initial_roster": engine.events[0]["roster"],
        "initial_agents": engine.events[0].get("agents"),
        "rounds": rounds,
    }


def write_replay(engine, out, status="running"):
    out = Path(out)
    payload = json.dumps(replay_data(engine, status), ensure_ascii=False, allow_nan=False)
    temporary = out / "replay.json.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(out / "replay.json")
    # Prevent a model message from terminating the data element or executing HTML.
    safe = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("replay.html").read_text(encoding="utf-8")
    temporary = out / "replay.html.tmp"
    temporary.write_text(template.replace("__REPLAY_DATA__", safe), encoding="utf-8")
    temporary.replace(out / "replay.html")
