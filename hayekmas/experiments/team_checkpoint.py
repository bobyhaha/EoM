"""Plain JSON checkpoints: no client, credentials, or executable pickle state."""

from dataclasses import asdict
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.team import Team
from hayekmas.adapters.teams.agent import TeamAgent
from hayekmas.base.agent import BaseAgent
from hayekmas.base.population import Population

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
    "birth_endowment_total", "removed_wealth_total", "rent_burn_total", "next_agent_number",
)
FIELDS = ("name", "wealth", "team_tag", "summary", "public_summary", "trajectory", "trainable_system_prompt",
          "id", "capability_score", "parent_agent_id", "parent_agent_name", "father_agent_id", "father_agent_name",
          "root_ancestor_class", "spawn_method", "tasks_lived", "bankruptcy_episode",
          "recent_failure_trace", "recent_failure_task", "recent_failure_answer")


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
        "retired_agents": engine.retired_agents,
        "training": engine.training,
        "rng": engine.rng.getstate(),
    }


def load_team(data, config, policy):
    engine = TeamMAS(config, policy)
    engine.population = Population()
    engine.team_manager.population = engine.population
    for row in data["agents"]:
        agent = TeamAgent(row["name"], row["wealth"])
        if config.interaction_protocol == "rounds":
            agent.frozen_system_prompt = (agent.COMPACT_ROUND_SYSTEM_PROMPT if config.round_schedule == "compact"
                                         else agent.ROUND_SYSTEM_PROMPT)
        agent.configure_objective(config)
        if config.token_profile == "solve_first":
            from hayekmas.adapters.teams.token_allocation import PHASE_REASONING
            agent.phase_reasoning_efforts = dict(PHASE_REASONING)
        if config.evolution_enabled:
            from hayekmas.adapters.teams.evolution import configure_agent
            configure_agent(engine, agent)
        for key in FIELDS:
            if key in row:
                setattr(agent, key, row[key])
        BaseAgent._id_counter = max(BaseAgent._id_counter, agent.id)
        agent.set_bid(row["bid"])
        engine.population.add_agent(agent)
    engine.retired_agents = data.get("retired_agents", [])
    engine.training = data.get("training", True)
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
    if config.evolution_enabled:
        engine.events.clear()
        engine.emit("initialized", condition=config.condition, backend=policy.label,
                    config=asdict(config), roster=engine.roster(), agents=engine.state()["agents"])
    return engine
