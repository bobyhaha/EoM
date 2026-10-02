import json
from unittest import TestCase
from unittest.mock import patch

from test_teams_compact import compact
from test_teams_openrouter import BUDGET, response
from test_teams_round_protocol import run
from hayekmas.adapters.teams.mas import BudgetExceeded
from hayekmas.adapters.teams.policy import ModelPolicy
from hayekmas.adapters.teams.token_allocation import phase_costs
from hayekmas.experiments.team_checkpoint import dump_team, load_team


class TokenAllocationTests(TestCase):
    def test_every_agent_and_restored_agent_gets_the_same_budget_guidance(self):
        mas = compact(token_profile="solve_first")
        restored = load_team(dump_team(mas), mas.config, mas.policy)
        for agent in mas.agents + restored.agents:
            prompt = agent.get_system_prompt()
            self.assertIn("Spend most of your reasoning and response-token budget", prompt)
            self.assertIn("not how much personal wealth", prompt)
            self.assertEqual(agent.phase_reasoning_efforts["round_commit"], "none")
            self.assertEqual(agent.phase_reasoning_efforts["round_work"], "high")

    def test_large_membership_input_keeps_all_k100_identities_and_money(self):
        mas = compact(num_agents=100, token_profile="solve_first", context_tokens=262144)
        for agent in mas.agents:
            agent.public_summary = "previous public contribution " * 1000
            agent.trainable_system_prompt = "Remember what worked before. " * 120
        obs = {"task": {"id": "x", "problem": "scientific problem " * 8000},
               "environment_state": "public contribution " * 10000,
               "roster": mas.roster(), "discussion": "public membership talk " * 2000,
               "summary": "my memory " * 1000, "episode": 0, "step": 0}
        mas.ask("round_membership", mas.agents[0], obs)
        request = [e for e in mas.events if e["event"] == "decision_request"][-1]
        self.assertLessEqual(request["input_tokens"], mas.config.formation_context_tokens)
        self.assertEqual(len(request["observation"]["roster"]), 100)
        for actual, original in zip(request["observation"]["roster"], obs["roster"]):
            self.assertEqual({k: actual[k] for k in ("name", "team", "wealth")},
                             {k: original[k] for k in ("name", "team", "wealth")})
        self.assertTrue(request["observation"]["context_truncated"])
        self.assertEqual(obs["task"]["problem"], "scientific problem " * 8000)
        self.assertEqual(mas.calls, 1)

    def test_ballot_uses_each_members_latest_current_message_without_full_problem(self):
        mas = compact(token_profile="solve_first", context_tokens=262144)
        mas.step = 1
        agent = mas.agents[0]
        group = agent.team_tag
        mas.scratchpads[group] = [
            {"author": "agent-0", "step": 0, "text": "OLD PRIVATE WORK"},
            {"author": "agent-0", "step": 1, "text": "derivation " * 500 + " FIRST CURRENT PLEDGE"},
            {"author": "agent-1", "step": 1, "text": "check " * 500 + " SECOND CURRENT PLEDGE"},
        ]
        obs = mas.scratchpad({"id": "x", "problem": "LONG PROBLEM " * 10000}, mas.agents[:2], [],
                            wealth={a.name: a.wealth for a in mas.agents[:2]}, activation_rule="strict majority", bid_cost_rate=1)
        mas.ask("round_commit", agent, obs)
        request = [e for e in mas.events if e["event"] == "decision_request"][-1]
        self.assertLessEqual(request["input_tokens"], mas.config.pledge_context_tokens)
        self.assertEqual(request["output_limit"], 96)
        self.assertEqual(request["reasoning_effort"], "none")
        text = json.dumps(request["observation"])
        self.assertIn("FIRST CURRENT PLEDGE", text)
        self.assertIn("SECOND CURRENT PLEDGE", text)
        self.assertNotIn("OLD PRIVATE WORK", text)
        self.assertNotIn("LONG PROBLEM", text)

    def test_oversize_required_context_fails_before_a_paid_request(self):
        mas = compact(num_agents=100, token_profile="solve_first", formation_context_tokens=1024)
        with self.assertRaises(BudgetExceeded):
            mas.ask("round_membership", mas.agents[0], {"roster": mas.roster()})
        self.assertEqual(mas.calls, 0)

    def test_larger_solving_caps_preserve_control_caps_and_reward_mechanism(self):
        mas = compact(token_profile="solve_first", action_tokens=192, bid_tokens=96,
                      solution_tokens=16384, discussion_tokens=2048, context_tokens=262144)
        metric = run(mas)
        self.assertEqual(metric["reward_income"], {"agent-0": 12, "agent-1": 12, "agent-2": 0, "agent-3": 0})
        expected = {"round_membership": 192, "round_chat": 2048, "round_commit": 96,
                    "round_work": 16384, "finalize": 16384, "vote": 64}
        for event in mas.events:
            if event["event"] == "decision_request" and event["phase"] in expected:
                self.assertEqual(event["output_limit"], expected[event["phase"]])
        mas.assert_accounting()

    @patch("hayekmas.adapters.teams.openrouter.requests.post")
    def test_reasoning_options_reach_the_provider_and_costs_are_recorded_by_phase(self, post):
        post.return_value = response()
        mas = compact(token_profile="solve_first", context_tokens=262144, solution_tokens=16384)
        mas.policy = ModelPolicy({"api": "openrouter", "name": "openai/gpt-6-luna", "api_key": "test",
                                  "extra_kwargs": {"temperature": None}}, BUDGET)
        mas.step = 0
        obs = mas.scratchpad({"id": "x", "problem": "q"}, mas.agents[:2], [],
                            wealth={a.name: a.wealth for a in mas.agents[:2]}, activation_rule="majority", bid_cost_rate=1)
        mas.ask("round_commit", mas.agents[0], obs)
        self.assertEqual(post.call_args.kwargs["json"]["reasoning"], {"effort": "none"})
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 96)
        mas.ask("round_work", mas.agents[0], obs, 16384)
        self.assertEqual(post.call_args.kwargs["json"]["reasoning"], {"effort": "high"})
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 16384)
        costs = mas.policy.client.usage()["phase_costs"]
        self.assertEqual(costs["phases"]["round_commit"]["calls"], 1)
        self.assertAlmostEqual(costs["solving_cost_usd"], .0001)
        self.assertAlmostEqual(costs["solving_share_of_confirmed_cost"], .5)

    def test_pending_costs_and_selection_votes_are_not_counted_as_solving(self):
        costs = phase_costs([
            {"kind": "round_work", "cost_usd": 2, "reasoning_tokens": 25},
            {"kind": "vote", "cost_usd": 1},
            {"kind": "finalize", "reserved_usd": 3},
        ])
        self.assertEqual(costs["solving_cost_usd"], 2)
        self.assertEqual(costs["solving_share_of_confirmed_cost"], 2 / 3)
        self.assertEqual(costs["phases"]["finalize"]["reserved_usd"], 3)
        self.assertEqual(costs["phases"]["round_work"]["reasoning_tokens"], 25)
