import json
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ExactTaskEnv, ResearchTaskEnv, Task
from hayekmas.adapters.teams.mas import BudgetExceeded, TeamMAS
from hayekmas.adapters.teams.replay import replay_data
from hayekmas.adapters.teams.evaluation import compare
from hayekmas.experiments.team_checkpoint import dump_team, load_team


class RoundPolicy:
    label = "scripted_test"

    def __init__(self, switch=False, no_work=False, abstain=False, no_bids=False):
        self.switch, self.no_work, self.abstain, self.no_bids = switch, no_work, abstain, no_bids
        self.observations = []
        self.judgments = []
        self.client = self

    def generate(self, prompt, **kwargs):
        self.judgments.append(prompt)
        return "SCORE: 1\nREASON: test grade"

    def respond(self, phase, agent, observation, max_tokens):
        self.observations.append((phase, agent.name, deepcopy(observation)))
        result = {}
        if phase == "formation":
            result = {"action": "pass"}
            if self.switch and observation.get("stage") == "round_start" and observation["step"] == 1:
                turn = observation["turn"]
                if turn == 0 and agent.team_tag:
                    result = {"action": "leave"}
                elif turn == 1 and not agent.team_tag:
                    target = f"agent-{int(agent.name.split('-')[1]) ^ 2}"
                    result = {"action": "invite", "target": target, "text": "Try this next round?"}
                elif turn == 2 and not agent.team_tag and observation["invitations"]:
                    result = {"action": "accept", "invitation": observation["invitations"][0]["id"]}
            result["reason"] = "This partnership matches my plan for the next round."
        elif phase in {"round_chat", "round_recap"}:
            result = {"message": f"Discuss freely with {agent.name}"}
        elif phase == "round_bid":
            # Only agent-0 funds the team. All members still receive path credit.
            active = agent.name == "agent-0" and not self.no_bids
            result = {"act": active, "contribution": 1 if active else 0, "message": "My decision"}
        elif phase == "coordinate":
            result = {"participate": True, "reason": "Worth this fee"}
        elif phase == "round_work":
            result = {"message": "Our intermediate calculation", "candidate": None if self.no_work else "PUBLIC WORK"}
        elif phase == "finalize":
            result = {"abstain": True} if self.abstain else {"candidate": "7", "final": True}
        elif phase == "vote":
            result = {"candidate_id": observation["candidates"][0]["id"]}
        elif phase == "reflect":
            result = {"reflect": False}
        return json.dumps(result)


def engine(policy=None, **options):
    cfg = TeamConfig(**{"interaction_protocol": "rounds", "num_agents": 4, "max_steps": 3,
                        "formation_turns": 3, "max_calls": 1000, **options})
    result = TeamMAS(cfg, policy or RoundPolicy())
    result.team_manager.create_team(result.agents[:2], 0)
    result.team_manager.create_team(result.agents[2:], 0)
    return result


def run(mas):
    return mas.run_one_episode(ExactTaskEnv(Task("example", "What is 3+4?", 7), 12), reflection=False)


class RoundProtocolTests(unittest.TestCase):
    def test_zero_coordination_fee_has_no_consent_calls_or_burn(self):
        mas = engine(coordination_fee_lambda=0)
        metric = run(mas)
        self.assertEqual(metric["coordination_burn_total"], 0)
        self.assertEqual(sum(metric["coordination_paid"].values()), 0)
        self.assertFalse(any(phase == "coordinate" for phase, _, _ in mas.policy.observations))

    def test_coordination_fee_burns_for_winners_and_losers_and_restores_checkpoint(self):
        mas = engine(coordination_fee_lambda=.5)
        metric = run(mas)
        self.assertEqual(metric["coordination_paid"], {f"agent-{i}": 1.5 for i in range(4)})
        self.assertEqual(mas.coordination_burn_total, 6)
        self.assertEqual(sum(metric["wealth"].values()), 80 + 24 - 1 - 6)
        loaded = load_team(dump_team(mas), mas.config, RoundPolicy())
        self.assertEqual(loaded.coordination_burn_total, 6)
        loaded.assert_accounting()

    def test_declining_member_leaves_before_fees_and_others_pay_reduced_amount(self):
        mas = engine(coordination_fee_lambda=.5)
        respond = mas.policy.respond
        mas.policy.respond = lambda phase, a, o, cap: (json.dumps({"participate": False, "reason": "Too costly"})
            if phase == "coordinate" and a.name == "agent-1" else respond(phase, a, o, cap))
        metric = run(mas)
        self.assertIsNone(mas.lookup("agent-1").team_tag)
        self.assertEqual(metric["coordination_paid"]["agent-1"], 0)
        self.assertEqual(metric["coordination_paid"]["agent-0"], 0)
        self.assertEqual(mas.coordination_burn_total, 3)
        mas.assert_accounting()

    def test_unaffordable_or_invalid_consent_never_charges_member(self):
        for fee, malformed in [(30, False), (.5, True)]:
            mas = engine(coordination_fee_lambda=fee)
            if malformed:
                respond = mas.policy.respond
                mas.policy.respond = lambda phase, a, o, cap: ('{}' if phase == "coordinate" else respond(phase, a, o, cap))
            metric = run(mas)
            self.assertEqual(metric["coordination_burn_total"], 0)
            self.assertTrue(all(a.team_tag is None for a in mas.agents))
            mas.assert_accounting()

    def test_no_fee_when_budget_skips_discussion(self):
        mas = engine(coordination_fee_lambda=.5, max_calls=25)
        metric = run(mas)
        self.assertEqual(metric["coordination_burn_total"], 0)
        self.assertFalse(any(phase == "coordinate" for phase, _, _ in mas.policy.observations))
        self.assertEqual(metric["score"], 1)

    def test_shared_discussion_precedes_activation_and_auction_every_round(self):
        mas = engine()
        metric = run(mas)
        self.assertEqual(metric["step_count"], 3)
        for step in range(3):
            events = [event for event in mas.events if event["step"] == step]
            chat = [i for i, event in enumerate(events) if event["event"] == "team_message" and event["channel"] == "pre_bid"]
            bids = [i for i, event in enumerate(events) if event["event"] == "activation"]
            auction = next(i for i, event in enumerate(events) if event["event"] == "auction")
            formed = next(i for i, event in enumerate(events) if event["event"] == "membership_committed")
            self.assertLess(formed, min(chat))
            self.assertEqual(len(chat), 8)
            self.assertLess(max(chat), min(bids))
            self.assertLess(max(bids), auction)
        bids = [obs for phase, _, obs in mas.policy.observations if phase == "round_bid"]
        self.assertTrue(all("Discuss freely" in obs["discussion"] for obs in bids))
        self.assertEqual([s["final"] for s in metric["steps"]], [False, False, True])
        self.assertEqual(sum(e["event"] == "membership_window" for e in mas.events), 3)
        final_settlement = max(i for i, e in enumerate(mas.events) if e["event"] == "settlement")
        self.assertFalse(any(e["event"] == "membership_window" for e in mas.events[final_settlement:]))
        self.assertTrue(all(e["reason"] for e in mas.events if e["event"] == "membership_decision"))

    def test_switching_preserves_historical_credit_and_previous_bid_recipients(self):
        mas = engine(RoundPolicy(switch=True))
        metric = run(mas)
        self.assertEqual([set(item["members"]) for item in metric["credit_path"]],
                         [{"agent-0", "agent-1"}, {"agent-0", "agent-2"}, {"agent-0", "agent-2"}])
        self.assertEqual(metric["reward_income"], {"agent-0": 12, "agent-1": 4, "agent-2": 8, "agent-3": 0})
        self.assertEqual((metric["reward"], metric["reward_issued"]), (12, 24))
        self.assertEqual(metric["bid_income"], {"agent-0": 1, "agent-1": 0.5, "agent-2": 0.5, "agent-3": 0})
        self.assertEqual(metric["paid"], {"agent-0": 3, "agent-1": 0, "agent-2": 0, "agent-3": 0})
        self.assertAlmostEqual(sum(metric["wealth"].values()), 80 + 24 - 1)
        settlements = [event for event in mas.events if event["event"] == "settlement"]
        self.assertTrue(all(event["reward"] == 0 and event["credits"] == {} for event in settlements[:-1]))
        mas.assert_accounting()

    def test_repeated_participation_and_zero_pledges_get_full_shares(self):
        mas = engine()
        metric = run(mas)
        self.assertEqual(metric["reward_income"], {"agent-0": 12, "agent-1": 12, "agent-2": 0, "agent-3": 0})
        self.assertEqual(metric["contributing_rounds"], 3)
        self.assertTrue(all(item["credits"] == {"agent-0": 4, "agent-1": 4} for item in metric["credit_path"]))
        self.assertEqual(metric["paid"]["agent-1"], 0)

    def test_empty_intermediate_rounds_continue_and_do_not_dilute_credit(self):
        mas = engine(RoundPolicy(no_work=True))
        metric = run(mas)
        self.assertEqual(metric["step_count"], 3)
        self.assertEqual(metric["contributing_rounds"], 1)
        self.assertEqual(metric["credit_path"][0]["step"], 2)
        self.assertEqual(metric["reward_income"]["agent-1"], 12)

    def test_all_zero_bids_still_attempt_finalization_without_charging(self):
        mas = engine(RoundPolicy(no_bids=True))
        metric = run(mas)
        self.assertEqual(metric["score"], 1)
        self.assertEqual(metric["contributing_rounds"], 1)
        self.assertEqual(sum(metric["paid"].values()), 0)
        self.assertEqual(sum(metric["reward_income"].values()), 24)
        self.assertTrue(any(event["event"] == "finalization_recovery" for event in mas.events))

    def test_final_abstention_does_not_invent_answer_or_reward(self):
        mas = engine(RoundPolicy(abstain=True))
        metric = run(mas)
        self.assertEqual(metric["score"], 0)
        self.assertEqual(metric["reward_issued"], 0)
        self.assertFalse(any(event["event"] == "submission" and event["final"] for event in mas.events))

    def test_grades_only_final_answer_and_keeps_rubric_private(self):
        mas = engine()
        env = ResearchTaskEnv(Task("private", "Public question", "SECRET RUBRIC"), 12, mas.judge)
        result = mas.run_one_episode(env, reflection=False)
        self.assertEqual(result["score"], 1)
        self.assertEqual(len(mas.policy.judgments), 1)
        self.assertNotIn("SECRET RUBRIC", json.dumps(mas.policy.observations))
        self.assertIn("SECRET RUBRIC", mas.policy.judgments[0])

    def test_act_false_cannot_commit_money(self):
        mas = engine()
        respond = mas.policy.respond
        mas.policy.respond = lambda phase, a, o, cap: (json.dumps({"act": False, "contribution": 15})
            if phase == "round_bid" and a.name == "agent-2" else respond(phase, a, o, cap))
        metric = run(mas)
        self.assertEqual(metric["paid"]["agent-2"], 0)
        self.assertTrue(all(e["contributions"]["agent-2"] == 0 for e in mas.events if e["event"] == "auction"))

    def test_no_credit_or_bid_transfer_across_episodes_and_checkpoint_preserves_rules(self):
        mas = engine()
        first = run(mas)
        loaded = load_team(dump_team(mas), mas.config, RoundPolicy())
        self.assertIn("Do not divide by team size", loaded.agents[0].get_system_prompt())
        second = run(loaded)
        self.assertEqual(first["reward_income"], second["reward_income"])
        self.assertEqual(loaded.reward_total, 48)
        first_transfer = next(event for event in loaded.events if event["event"] == "bid_transfer")
        self.assertIsNone(first_transfer["receiver"])
        loaded.assert_accounting()

    def test_impossible_call_budget_is_rejected_before_decisions(self):
        mas = engine(max_calls=1)
        with self.assertRaises(BudgetExceeded):
            run(mas)
        self.assertEqual(mas.calls, 0)
        self.assertEqual(mas.bid_paid_total, 0)

    def test_replay_contains_round_membership_activation_and_full_reward_credit(self):
        mas = engine(RoundPolicy(switch=True))
        run(mas)
        events = replay_data(mas, "complete")["rounds"][0]["events"]
        kinds = {event["event"] for event in events}
        self.assertTrue({"decision_round_started", "membership_window", "activation", "path_reward"} <= kinds)
        payout = next(event for event in events if event["event"] == "path_reward")
        self.assertEqual(payout["credits"]["agent-1"], 4)

    def test_incompatible_protocols_rejected_and_legacy_default_retained(self):
        self.assertEqual(TeamConfig().interaction_protocol, "legacy")
        for options in [{"bidding_mode": "sealed"}, {"collaboration_mode": "reviewed"}, {"finalization_enabled": False}]:
            with self.assertRaises(ValueError):
                TeamConfig(interaction_protocol="rounds", **options)

    def test_trained_evaluation_keeps_round_formation_but_disables_strategy_reflection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text(json.dumps({"id": "train", "problem": "sum: [1, 2]", "answer": 3}) + "\n")
            compare({
                "teams": {"interaction_protocol": "rounds", "num_agents": 2, "rounds": 1, "max_steps": 2},
                "comparison": {"samples": 1, "ks": [1], "train_rounds": 1, "train_dataset": str(train)},
            }, out=root / "run")
            requests = [json.loads(line) for line in (root / "run/requests.jsonl").read_text().splitlines()]
            tests = [r for r in requests if r["phase"] == "test"]
            self.assertTrue(any(r["kind"] == "formation" for r in tests))
            self.assertTrue(any(r["kind"] == "round_recap" for r in tests))
            self.assertFalse(any(r["kind"] in {"reflect", "improve"} for r in tests))
