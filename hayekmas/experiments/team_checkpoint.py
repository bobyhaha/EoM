"""Plain JSON checkpoints: no client, credentials, or executable pickle state."""

from dataclasses import asdict
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.team import Team

COUNTERS = (
    "round",
    "calls",
    "requested_output_tokens",
    "reference_input_tokens",
    "invalid_actions",
    "reward_total",
    "bid_burn_total",
    "bid_paid_total",
    "bid_transfer_total",
    "reflection_burn_total",
    "coordination_burn_total",
    "initial_total",
)
FIELDS = ("name", "wealth", "team_tag", "summary", "public_summary", "trajectory", "trainable_system_prompt")


def dump_team(engine):
    return {
        "agents": [{**{k: getattr(a, k) for k in FIELDS}, "bid": a.get_bid()} for a in engine.agents],
        "teams": {k: asdict(t) for k, t in engine.team_manager.teams.items()},
        "team_counter": engine.team_manager.team_counter,
        "invitation_counter": engine.team_manager.invitation_counter,
        "invitations": engine.invitations,
        "public_summaries": engine.public_summaries,
        "inboxes": engine.inboxes,
        "counters": {k: getattr(engine, k) for k in COUNTERS},
        "rng": engine.rng.getstate(),
    }


def load_team(data, config, policy):
    engine = TeamMAS(config, policy)
    for agent, row in zip(engine.agents, data["agents"], strict=True):
        for key in FIELDS:
            setattr(agent, key, row[key])
        agent.set_bid(row["bid"])
    engine.team_manager.teams = {k: Team(**v) for k, v in data["teams"].items()}
    engine.team_manager.team_counter = data["team_counter"]
    engine.team_manager.invitation_counter = data["invitation_counter"]
    engine.team_manager.invitations = data["invitations"]
    engine.invitations = engine.team_manager.invitations
    engine.public_summaries = data["public_summaries"]
    engine.inboxes = data["inboxes"]
    for key, value in data["counters"].items():
        setattr(engine, key, value)

    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value

    engine.rng.setstate(tuples(data["rng"]))
    return engine
