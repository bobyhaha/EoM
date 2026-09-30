import json
import math
from pathlib import Path
import tempfile
import unittest

from hayekmas.base.agent import BaseAgent
from hayekmas.base.population import Population
from hayekmas.adapters.teams.agent import TeamAction, TeamAgent
from hayekmas.adapters.teams.config import TeamConfig, TokenBudget
from hayekmas.adapters.teams.env import ExactTaskEnv, Task, demo_tasks, load_tasks
from hayekmas.adapters.teams.mas import BudgetExceeded, TeamMAS, dumps
from hayekmas.adapters.teams.policy import DemoPolicy, INSTRUCTIONS
from hayekmas.adapters.teams.replay import replay_data, write_replay
from hayekmas.adapters.teams.runtime import run


class Script:
    label = "test_policy"

    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def respond(self, phase, agent, observation, max_tokens):
        self.calls.append((phase, agent.name, observation, max_tokens))
        value = self.callback(phase, agent, observation)
        return value if isinstance(value, str) else json.dumps(value)


def engine(callback=None, **config):
    policy = Script(callback or (lambda phase, agent, obs: {}))
    return TeamMAS(TeamConfig(**{"num_agents": 3, **config}), policy)


class MembershipTests(unittest.TestCase):
    def test_uses_original_eom_primitives(self):
        mas = engine()
        self.assertIsInstance(mas.population, Population)
        self.assertTrue(all(isinstance(a, BaseAgent) for a in mas.agents))
        self.assertEqual({a.get_system_prompt() for a in mas.agents}, {mas.agents[0].get_system_prompt()})
        self.assertTrue(all(a.agent_tags == () for a in mas.agents))

    def test_consent_exclusivity_and_voluntary_exit(self):
        mas = engine()
        a, b, c = mas.agents
        invitation = mas.team_manager.invite(a, b, 0)
        self.assertIsNone(b.team_tag)
        with self.assertRaises(ValueError):
            mas.team_manager.accept_invite(c, invitation["id"], 0)
        tag, _ = mas.team_manager.accept_invite(b, invitation["id"], 0)
        self.assertEqual(a.team_tag, tag)
        self.assertEqual(b.team_tag, tag)
        with self.assertRaises(ValueError):
            mas.team_manager.invite(c, b, 0)
        mas.team_manager.leave_team(b)
        self.assertIsNone(a.team_tag)
        self.assertIsNone(b.team_tag)

    def test_expired_and_stale_invites_cannot_admit(self):
        mas = engine()
        a, b, c = mas.agents
        invitation = mas.team_manager.invite(a, b, 0)
        with self.assertRaises(ValueError):
            mas.team_manager.accept_invite(b, invitation["id"], 1)
        mas.team_manager.create_team([a, c], 0)
        with self.assertRaises(ValueError):
            mas.team_manager.accept_invite(b, invitation["id"], 0)

    def test_team_size_cap_checked_at_acceptance(self):
        mas = engine(max_team_size=2)
        a, b, c = mas.agents
        invitation = mas.team_manager.invite(a, c, 0)
        mas.team_manager.create_team([a, b], 0)
        with self.assertRaises(ValueError):
            mas.team_manager.accept_invite(c, invitation["id"], 0)
        with self.assertRaises(ValueError):
            mas.team_manager.invite(a, c, 0)

    def test_membership_schedule_and_fixed_conditions(self):
        for condition in ("dynamic", "self_selected_fixed", "individual", "random_fixed"):
            mas = engine(condition=condition)
            self.assertEqual(mas.formation_enabled(), condition in ("dynamic", "self_selected_fixed"))
            mas.round = 1
            self.assertFalse(mas.formation_enabled())
            mas.round = 5
            self.assertEqual(mas.formation_enabled(), condition == "dynamic")
            if condition != "dynamic":
                before = mas.roster()
                mas.apply_formation(mas.agents[0], {"action": "leave"})
                self.assertEqual(before, mas.roster())

    def test_no_kicking_or_arbitrary_state_assignment(self):
        mas = engine()
        a, b, _ = mas.agents
        mas.team_manager.create_team([a, b], 0)
        before = mas.roster()
        mas.apply_formation(a, {"action": "kick", "target": b.name})
        mas.apply_formation(a, {"action": []})
        self.assertEqual(mas.roster(), before)
        self.assertEqual(mas.invalid_actions, 2)


class AuctionTests(unittest.TestCase):
    def test_negotiation_observes_peer_messages_and_revised_pledges(self):
        def callback(phase, a, obs):
            if phase == "negotiate":
                return {
                    "contribution": 2 + obs["turn"],
                    "message": f"Please share costs, {a.name}",
                    "contributions": {name: 999 for name in obs["pledges"]},
                }
            return {"contribution": 1}

        mas = engine(callback)
        mas.team_manager.create_team(mas.agents[:2], 0)
        winner, members, contributions, _ = mas.auction({"id": "x", "problem": "task"})
        calls = [c for c in mas.policy.calls if c[0] == "negotiate"]
        self.assertEqual(len(calls), 4)
        self.assertIn("Please share costs", calls[1][2]["discussion"])
        self.assertEqual(sum(calls[1][2]["pledges"].values()), 2)
        self.assertEqual(contributions, {"agent-0": 3, "agent-1": 3, "agent-2": 1})
        self.assertEqual(winner, "team-1")
        self.assertEqual([a.wealth for a in mas.agents], [17, 17, 20])
        self.assertEqual(len(members), 2)
        mas.assert_accounting()

    def test_invalid_pledges_never_overspend(self):
        for value in (-1, 21, True, "5", None, float("nan"), float("inf")):
            mas = engine(lambda phase, a, obs: {"contribution": value}, condition="individual")
            winner, _, contribution, _ = mas.auction({"id": "x", "problem": "x"})
            self.assertIsNone(winner)
            self.assertEqual(set(contribution.values()), {0})
            self.assertEqual([a.wealth for a in mas.agents], [20, 20, 20])

    def test_invalid_revision_preserves_last_valid_personal_commitment(self):
        mas = engine(lambda phase, a, obs: {"contribution": 2 if obs["turn"] == 0 else -100}, num_agents=2)
        mas.team_manager.create_team(mas.agents, 0)
        _, _, contribution, _ = mas.auction({"id": "x", "problem": "x"})
        self.assertEqual(set(contribution.values()), {2})

    def test_only_winner_pays_and_zero_contributor_gets_equal_reward(self):
        def callback(phase, a, obs):
            if phase in {"contribute", "negotiate"}:
                return {"contribution": {"agent-0": 4, "agent-1": 0, "agent-2": 1}[a.name]}
            if phase == "discuss":
                return {"candidate": "7", "message": "seven"}
            if phase == "vote":
                return {"candidate_id": obs["candidates"][0]["id"]}
            return {}

        mas = engine(callback, condition="self_selected_fixed")
        mas.team_manager.create_team(mas.agents[:2], 0)
        mas.round = 1  # membership frozen after initial formation
        mas.run_one_episode(ExactTaskEnv(Task("x", "3+4", 7), 12))
        self.assertEqual([a.wealth for a in mas.agents], [22, 26, 20])
        self.assertEqual(mas.metrics[-1]["reward"], 12)
        mas.assert_accounting()

    def test_cost_rate_and_all_zero_bids(self):
        mas = engine(lambda phase, a, obs: {"contribution": 2}, bid_cost_rate=0.25)
        mas.auction({"id": "x", "problem": "x"})
        self.assertEqual(mas.bid_burn_total, 0.5)
        mas.assert_accounting()
        empty = engine()
        empty.run_one_episode(ExactTaskEnv(Task("x", "x", "secret"), 12))
        self.assertEqual(empty.metrics[-1]["reward"], 0)

    def test_bad_votes_do_not_invent_a_submission(self):
        mas = engine(lambda phase, a, obs: {"candidate": "7"} if phase == "discuss" else {"candidate_id": []})
        result = mas.collaborate(mas.agents[:2], {"id": "x", "problem": "x"})
        self.assertIsNone(result)


class ObservationTests(unittest.TestCase):
    def test_shared_scratchpad_has_no_role_instructions(self):
        mas = engine()
        context = mas.scratchpad({"id": "x", "problem": "x"}, mas.agents, [])
        for key in ("task", "environment_state", "members", "discussion", "objective"):
            self.assertIn(key, context)
        self.assertIn("public_summary", context["members"][0])
        prompt = TeamAgent.FROZEN_SYSTEM_PROMPT + INSTRUCTIONS["discuss"]
        for word in ("solver", "critic", "verifier", "assign roles", "critique"):
            self.assertNotIn(word, prompt.lower())

    def test_ground_truth_and_private_competitor_traces_never_reach_policy(self):
        mas = engine(lambda phase, a, obs: {"contribution": 1}, condition="individual")
        mas.run_one_episode(ExactTaskEnv(Task("x", "public question", "HIDDEN_LABEL_SENTINEL"), 12))
        self.assertNotIn("HIDDEN_LABEL_SENTINEL", dumps(mas.policy.calls))
        mas.agents[1].trajectory = ["COMPETITOR_PRIVATE_SENTINEL"]
        mas.agents[1].summary = "COMPETITOR_PRIVATE_SENTINEL"
        self.assertNotIn("COMPETITOR_PRIVATE_SENTINEL", dumps(mas.reflection_observation(mas.agents[0])))

    def test_reflection_evidence_and_two_generation_budgets(self):
        def callback(phase, a, obs):
            return {"reflect": True, "analysis": "observed cost", "strategy": "be careful"}

        mas = engine(callback, num_agents=2, evidence_tokens=128, inspect_tokens=80, update_tokens=96)
        mas.team_manager.create_team(mas.agents, 0)
        for a in mas.agents:
            a.trajectory = ['large "quoted" 世界 😃 ' * 1000]
            a.summary = 'large "quoted" 世界 😃 ' * 1000
        # The decision summary itself is normally bounded by remember().
        for a in mas.agents:
            a.summary = mas.tokens.clip(a.summary, mas.config.summary_tokens)
        evidence = mas.reflection_observation(mas.agents[0])
        self.assertLessEqual(mas.tokens.count(dumps(evidence)), 128)
        mas.reflect()
        self.assertEqual([c[3] for c in mas.policy.calls if c[0] == "inspect"], [80, 80])
        self.assertEqual([c[3] for c in mas.policy.calls if c[0] == "improve"], [96, 96])
        self.assertEqual(mas.reflection_burn_total, 4)
        self.assertEqual(mas.agents[0].trainable_system_prompt, "be careful")
        mas.assert_accounting()

    def test_no_free_reflection_or_oversized_output(self):
        mas = engine(lambda phase, a, obs: {"reflect": True}, initial_wealth=1, reflection_cost=2)
        mas.reflect()
        self.assertEqual(mas.calls, 3)
        self.assertEqual(mas.reflection_burn_total, 0)
        mas.policy = Script(lambda phase, a, obs: {"analysis": "long text " * 1000})
        self.assertEqual(mas.ask("inspect", mas.agents[0], {}, 64), {})

    def test_fixed_call_budget_stops_before_request(self):
        mas = engine(max_calls=1)
        mas.ask("reflect", mas.agents[0], {})
        with self.assertRaises(BudgetExceeded):
            mas.ask("reflect", mas.agents[0], {})
        self.assertEqual(len(mas.policy.calls), 1)

    def test_reference_token_clipping_unicode(self):
        budget = TokenBudget()
        for n in (1, 2, 5, 64):
            for tail in (False, True):
                value = budget.clip('"世界😃\\\n' * 100, n, tail=tail)
                self.assertLessEqual(budget.count(value), n)


class RuntimeTests(unittest.TestCase):
    def test_repeatability_and_round_state_replay(self):
        outputs = []
        for _ in range(2):
            mas = TeamMAS(TeamConfig(num_agents=4, rounds=6), DemoPolicy(7))
            for task in demo_tasks(7, 6):
                mas.run_one_episode(ExactTaskEnv(task, 12))
            outputs.append(mas.metrics)
            replay = replay_data(mas, "complete")
            self.assertEqual(len(replay["rounds"]), 6)
            final = replay["rounds"][-1]
            self.assertEqual(final["agents"], mas.state()["agents"])
            self.assertTrue(any(e["event"] == "team_message" and e["channel"] == "bidding" for e in final["events"]))
            self.assertTrue(any(e["event"] == "strategy_updated" for e in replay["rounds"][4]["events"]))
        self.assertEqual(*outputs)

    def test_report_escapes_untrusted_script_and_keeps_messages(self):
        mas = TeamMAS(TeamConfig(num_agents=2), DemoPolicy(7))
        mas.run_one_episode(ExactTaskEnv(demo_tasks(7, 1)[0], 12))
        malicious = '</script><script>alert("x")</script>'
        mas.events.insert(
            -1,
            {
                "event": "team_message",
                "round": 0,
                "agent": "agent-0",
                "members": ["agent-0", "agent-1"],
                "text": malicious,
                "channel": "team",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            write_replay(mas, directory, "complete")
            text = (Path(directory) / "replay.html").read_text()
            self.assertNotIn(malicious, text)
            payload = text.split('<script id="replay-data" type="application/json">')[1].split("</script>")[0]
            decoded = json.loads(payload)
            self.assertEqual(decoded["rounds"][0]["events"][-1]["text"], malicious)

    def test_run_outputs_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            raw = {"teams": {"num_agents": 2, "rounds": 2}}
            result = run(raw, out=path, plots=False)
            self.assertEqual(result["completed_rounds"], 2)
            for name in (
                "manifest.json",
                "population.json",
                "events.jsonl",
                "replay.html",
                "replay.json",
                "agent_features.csv",
            ):
                self.assertTrue((path / name).exists())
            self.assertEqual(json.loads((path / "manifest.json").read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                run(raw, out=path, plots=False)

    def test_interruption_preserves_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            with self.assertRaises(BudgetExceeded):
                run({"teams": {"num_agents": 2, "rounds": 2, "max_calls": 1}}, out=path, plots=False)
            self.assertTrue((path / "interrupted_state.json").exists())
            self.assertEqual(json.loads((path / "replay.json").read_text())["status"], "interrupted")

    def test_config_validation(self):
        for settings in (
            {"reward": math.nan},
            {"num_agents": True},
            {"condition": "bad"},
            {"bid_cost_rate": 2},
            {"fixed_team_size": 5},
        ):
            with self.assertRaises(ValueError):
                TeamConfig(**settings)

    def test_jsonl_split_and_exact_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"id": "1", "split": "train", "problem": "x", "answer": 7},
                        {"id": "2", "split": "test", "problem": "x", "answer": "Yes"},
                    ]
                )
            )
            tasks = load_tasks(path, "test")
            self.assertEqual([t.id for t in tasks], ["2"])
            self.assertEqual(ExactTaskEnv(tasks[0], 12).apply(TeamAction(" YES ")), 12)
            self.assertEqual(ExactTaskEnv(Task("x", "x", 7), 12).apply(TeamAction("nan")), 0)


if __name__ == "__main__":
    unittest.main()
