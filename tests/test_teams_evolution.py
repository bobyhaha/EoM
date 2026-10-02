"""Population lifecycle invariants, using scripted policies only (no API calls)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ExactTaskEnv, demo_tasks
from hayekmas.adapters.teams.evaluation import evaluation_copy
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.policy import DemoPolicy
from hayekmas.adapters.teams.replay import replay_data, write_replay
from hayekmas.adapters.teams.report import analyze
from hayekmas.experiments.team_checkpoint import dump_team, load_team
from hayekmas.experiments.population_study import estimate_growing


def engine(**kw):
    return TeamMAS(TeamConfig(**dict(interaction_protocol="rounds", token_profile="solve_first",
                                   evolution_enabled=True, num_agents=4, max_steps=1,
                                   context_tokens=32768, update_tokens=512, **kw)), DemoPolicy(7))


def run(mas, **kw):
    return mas.run_one_episode(ExactTaskEnv(demo_tasks(7, 1)[0], 12), reflection=False, **kw)


class EvolutionTests(unittest.TestCase):
    def test_periodic_growth_cap_and_no_calls_at_cap(self):
        mas = engine(birth_interval=1, num_births_per_interval=2)
        for expected in (6, 8, 8):
            row = run(mas)
            self.assertEqual(row["population_size"], expected)
            mas.assert_accounting()
        born = [e for e in mas.events if e["event"] == "agent_born"]
        self.assertEqual(len(born), 4)
        self.assertEqual(mas.birth_endowment_total, 80)
        self.assertTrue(all(e["state"]["team"] is None for e in born))
        self.assertEqual(len({a.name for a in mas.agents}), 8)
        self.assertEqual(sum(e["event"] == "decision_request" and e["phase"].startswith("birth_") for e in mas.events), 4)

    def test_settlement_precedes_death_and_replacement_keeps_no_credit(self):
        mas = engine(birth_interval=0, rent=100, rent_interval=1)
        mas.team_manager.create_team(mas.agents[:2], 0)
        names = {a.name for a in mas.agents}
        row = run(mas)
        self.assertEqual(len(mas.agents), 4)
        self.assertTrue(names.isdisjoint(a.name for a in mas.agents))
        self.assertEqual(len(mas.retired_agents), 4)
        self.assertTrue(all(a.team_tag is None and a.wealth == 20 for a in mas.agents))
        self.assertTrue(all(a.parent_agent_name in names and a.spawn_method == "bad_birth" for a in mas.agents))
        self.assertTrue(set(row["reward_income"]) <= names)
        kinds = [e["event"] for e in mas.events]
        self.assertLess(kinds.index("path_reward"), kinds.index("agent_removed"))
        self.assertEqual(mas.invitations, {})
        self.assertEqual(set(mas.inboxes), {a.name for a in mas.agents})
        mas.assert_accounting()
        with tempfile.TemporaryDirectory() as tmp:
            analyze(mas, tmp, plots=True)
            write_replay(mas, tmp)
            self.assertTrue((Path(tmp)/"replay.html").exists())
            self.assertEqual(len(replay_data(mas, "complete")["all_roster"]), 8)

    def test_zero_wealth_is_not_bankruptcy(self):
        mas = engine(initial_wealth=0, birth_interval=0)
        run(mas)
        self.assertEqual(len(mas.agents), 4)
        self.assertEqual(mas.retired_agents, [])

    def test_no_birth_probability_and_empty_population_stops(self):
        mas = engine(birth_interval=0, rent=100, rent_interval=1, p_a=0, p_b=0)
        run(mas)
        self.assertEqual(mas.agents, [])
        mas.assert_accounting()
        with self.assertRaisesRegex(RuntimeError, "Population is empty"):
            run(mas)

    def test_good_bankruptcy_birth_uses_rich_survivor(self):
        mas = engine(birth_interval=0, rent=10, rent_interval=1, p_a=1, p_b=0)
        # Transfer initial wealth to one survivor, preserving accounting.
        for a in mas.agents:
            a.wealth = 0
        mas.agents[0].wealth = 80
        run(mas)
        born = [e for e in mas.events if e["event"] == "agent_born"]
        self.assertTrue(born)
        self.assertTrue(all(e["birth_kind"] == "good" and e["parent"] == "agent-0" for e in born))
        mas.assert_accounting()

    def test_eval_never_evolves_or_charges_rent(self):
        mas = engine(birth_interval=1, rent=100, rent_interval=1)
        before = dump_team(mas)
        clone = evaluation_copy(mas, 8, DemoPolicy(8))
        run(clone)
        self.assertEqual(len(clone.agents), 4)
        self.assertFalse(any(e["event"] in {"agent_born", "agent_removed", "population_rent"} for e in clone.events))
        self.assertEqual(dump_team(mas), before)
        run(mas, training=False)
        self.assertEqual(mas.rent_burn_total, 0)

    def test_variable_population_checkpoint_continues_deterministically(self):
        mas = engine(birth_interval=1, num_births_per_interval=1, rent=0)
        run(mas)
        state = json.loads(json.dumps(dump_team(mas)))
        other = load_team(state, mas.config, deepcopy(mas.policy))
        self.assertEqual(dump_team(other), state | {"rng": other.rng.getstate()})
        self.assertEqual(len(other.agents), 5)
        self.assertIs(other.team_manager.population, other.population)
        left, right = run(mas), run(other)
        self.assertEqual(left, right)
        self.assertEqual([(a.name, a.parent_agent_name, a.wealth) for a in mas.agents],
                         [(a.name, a.parent_agent_name, a.wealth) for a in other.agents])
        other.assert_accounting()

    def test_invalid_strategy_falls_back_without_mutating_parent(self):
        from hayekmas.adapters.teams.evolution import birth
        mas = engine(birth_interval=0)
        parent = mas.agents[0]
        old = parent.trainable_system_prompt
        mas.policy.respond = lambda *args: '{}'
        self.assertTrue(birth(mas, parent, "good", "test"))
        self.assertEqual(mas.agents[-1].trainable_system_prompt, old)
        self.assertEqual(parent.trainable_system_prompt, old)
        self.assertEqual(mas.agents[-1].frozen_system_prompt, parent.frozen_system_prompt)

    def test_budget_and_failed_birth_do_not_mint_money(self):
        from hayekmas.adapters.teams.evolution import birth
        mas = engine(birth_interval=0)
        def fail(*args):
            raise RuntimeError("synthetic provider failure")
        mas.policy.respond = fail
        self.assertFalse(birth(mas, mas.agents[0], "bad", "test"))
        mas.calls = mas.config.max_calls
        self.assertFalse(birth(mas, mas.agents[0], "bad", "test"))
        self.assertEqual(mas.birth_endowment_total, 0)
        mas.assert_accounting()

    def test_largest_study_cap_keeps_complete_membership_roster(self):
        from hayekmas.adapters.teams.token_allocation import control_observation
        from hayekmas.adapters.teams.policy import INSTRUCTIONS
        cfg = TeamConfig(interaction_protocol="rounds", token_profile="solve_first", evolution_enabled=True,
                         num_agents=200, formation_context_tokens=12800, context_tokens=262144)
        mas = TeamMAS(cfg, DemoPolicy(7))
        agent = mas.agents[0]
        obs = {"roster": mas.roster(), "invitations": [], "task": {"id": "test", "problem": "A task"},
               "you": {"name": agent.name, "wealth": agent.wealth, "team": None}}
        trimmed = control_observation(mas, "round_join", agent, obs)
        self.assertEqual(len(trimmed["roster"]), 200)
        text = json.dumps({"instruction": INSTRUCTIONS["round_join"], "observation": trimmed}, sort_keys=True)
        self.assertLessEqual(mas.tokens.count(agent.get_system_prompt()) + mas.tokens.count(text), 12800)

    def test_cost_growth_uses_scheduled_counts_and_no_test_births(self):
        row = estimate_growing(10)
        pop = row["population_assumption"]
        self.assertEqual(pop["counts_before_tasks"][:6], [10]*5+[12])
        self.assertEqual(pop["periodic_births"], 10)
        self.assertEqual(pop["counts_before_tasks"][40:], [20]*19)
        self.assertGreater(estimate_growing(10, at_cap=True)["teams"]["usd"], row["teams"]["usd"])
