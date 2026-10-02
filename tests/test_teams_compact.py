"""Protocol invariants: real membership consent, collective abstention and own funds."""

from collections import Counter
from copy import deepcopy
import json
import unittest

from test_teams_round_protocol import RoundPolicy, run
from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.policy import DemoPolicy
from hayekmas.adapters.teams.replay import replay_data
from hayekmas.experiments.team_checkpoint import dump_team, load_team


class CompactPolicy(RoundPolicy):
    def __init__(self, switch=False, votes=None):
        super().__init__()
        self.switch, self.votes = switch, votes or {}

    def respond(self, phase, agent, observation, max_tokens):
        if phase not in {"round_membership", "round_join", "round_commit"}:
            return super().respond(phase, agent, observation, max_tokens)
        self.observations.append((phase, agent.name, deepcopy(observation)))
        if phase == "round_membership":
            change = self.switch and observation["step"] == 1
            result = {"leave": change, "invite": f"agent-{int(agent.name.split('-')[1]) ^ 2}" if change else None,
                      "reason": "Choose partners for this round", "message": "My membership proposal"}
        elif phase == "round_join":
            result = {"invitation": observation["invitations"][0]["id"], "reason": "I consent to join"}
        else:
            result = {"act": self.votes.get(agent.name, True), "contribution": 1 if agent.name == "agent-0" else 0,
                      "reason": "PRIVATE BALLOT REASON"}
        return json.dumps(result)


def compact(policy=None, **options):
    cfg = TeamConfig(**{"interaction_protocol": "rounds", "round_schedule": "compact", "num_agents": 4,
                        "max_steps": 3, "max_calls": 1000, **options})
    mas = TeamMAS(cfg, policy or CompactPolicy())
    for start in range(0, len(mas.agents) - 1, 2):
        mas.team_manager.create_team(mas.agents[start:start + 2], 0)
    return mas


class CompactTests(unittest.TestCase):
    def test_stable_k10_uses_30_preauction_calls_and_no_repeated_bidding(self):
        mas = compact(num_agents=10)
        run(mas)
        requests = [e for e in mas.events if e["event"] == "decision_request"]
        for step in range(3):
            counts = Counter(e["phase"] for e in requests if e["step"] == step)
            self.assertEqual({p: counts[p] for p in ("round_membership", "round_chat", "round_commit")},
                             {"round_membership": 10, "round_chat": 10, "round_commit": 10})
            self.assertEqual(sum(counts[p] for p in ("round_join", "round_recap", "round_bid", "formation")), 0)
            events = [e for e in mas.events if e["step"] == step]
            formed = next(i for i, e in enumerate(events) if e["event"] == "membership_committed")
            chat = [i for i, e in enumerate(events) if e["event"] == "team_message" and e["channel"] == "pre_bid"]
            activated = next(i for i, e in enumerate(events) if e["event"] == "team_activation")
            self.assertLess(formed, min(chat))
            self.assertLess(max(chat), activated)

    def test_leave_and_invite_can_switch_in_one_window_and_preserve_past_credit(self):
        mas = compact(CompactPolicy(switch=True))
        metric = run(mas)
        self.assertEqual([set(p["members"]) for p in metric["credit_path"]],
                         [{"agent-0", "agent-1"}, {"agent-0", "agent-2"}, {"agent-0", "agent-2"}])
        self.assertEqual(metric["reward_income"], {"agent-0": 12, "agent-1": 4, "agent-2": 8, "agent-3": 0})
        self.assertEqual(metric["bid_income"], {"agent-0": 1, "agent-1": .5, "agent-2": .5, "agent-3": 0})
        self.assertEqual(metric["paid"]["agent-2"], 0)
        for step in range(3):
            requests = [e for e in mas.events if e["event"] == "decision_request" and e["step"] == step
                        and e["phase"] in {"round_membership", "round_join"}]
            self.assertLessEqual(len(requests), 8)
        mas.assert_accounting()

    def test_invitation_after_recipient_proposal_is_still_accepted(self):
        mas = compact()
        for agent in mas.agents:
            agent.team_tag = None
        order = list(mas.agents)
        mas.rng.shuffle(order)
        recipient, inviter = order[0].name, order[-1].name
        mas.rng.seed(mas.config.seed)
        respond = mas.policy.respond
        mas.policy.respond = lambda phase, a, o, cap: (json.dumps({"leave": False,
            "invite": recipient if a.name == inviter else None, "reason": "Join?"})
            if phase == "round_membership" else respond(phase, a, o, cap))
        run(mas)
        self.assertIsNotNone(mas.lookup(recipient).team_tag)
        self.assertEqual(mas.lookup(recipient).team_tag, mas.lookup(inviter).team_tag)
        self.assertEqual(sum(p == "round_join" for p, _, _ in mas.policy.observations), 1)

    def test_declined_or_malformed_acceptance_never_joins_or_crashes(self):
        for selected in [None, [], "unknown"]:
            mas = compact(CompactPolicy(switch=True))
            respond = mas.policy.respond
            mas.policy.respond = lambda phase, a, o, cap: (json.dumps({"invitation": selected, "reason": "No"})
                if phase == "round_join" else respond(phase, a, o, cap))
            run(mas)
            self.assertTrue(all(a.team_tag is None for a in mas.agents))
            self.assertFalse(any(e["event"] == "joined" for e in mas.events))

    def test_tied_teams_abstain_cancel_pledges_and_are_not_forced_to_finalize(self):
        mas = compact(CompactPolicy(votes={"agent-1": False, "agent-3": False}))
        metric = run(mas)
        self.assertEqual(metric["step_count"], 3)
        self.assertEqual(metric["reward"], 0)
        self.assertEqual(metric["reward_issued"], 0)
        self.assertEqual(sum(metric["paid"].values()), 0)
        self.assertTrue(all(e["bid"] == 0 and not e["active"] for e in mas.events if e["event"] == "team_activation"))
        self.assertFalse(any(p in {"round_work", "finalize", "vote"} for p, _, _ in mas.policy.observations))
        self.assertFalse(any(e["event"] == "finalization_recovery" for e in mas.events))
        mas.assert_accounting()

    def test_strict_majority_can_act_and_zero_pledge_no_voter_still_gets_credit(self):
        mas = compact(CompactPolicy(votes={"agent-2": False}))
        for agent in mas.agents:
            agent.team_tag = None
        mas.team_manager.create_team(mas.agents[:3], 0)
        metric = run(mas)
        self.assertEqual(metric["reward_income"]["agent-2"], 12)
        self.assertEqual(metric["paid"]["agent-2"], 0)
        events = [e for e in mas.events if e["event"] == "team_activation" and len(e["members"]) == 3]
        self.assertTrue(all(e["yes_votes"] == 2 and e["active"] for e in events))

    def test_invalid_or_unauthorized_pledge_is_not_a_valid_yes_vote(self):
        for bad in [{"act": True, "contribution": 10000, "reason": "Overspend"},
                    {"act": True, "contribution": -1, "reason": "Negative"},
                    {"act": True, "contribution": 1},
                    {"act": "true", "contribution": 1, "reason": "Wrong type"}]:
            mas = compact()
            respond = mas.policy.respond
            mas.policy.respond = lambda phase, a, o, cap: (json.dumps(bad) if phase == "round_commit"
                and a.name == "agent-1" else respond(phase, a, o, cap))
            metric = run(mas)
            self.assertEqual(sum(metric["paid"].values()), 0)
            self.assertEqual(metric["reward_issued"], 0)
            self.assertTrue(any(e["event"] == "invalid_action" for e in mas.events))

    def test_current_ballots_are_hidden_until_commit_completes(self):
        mas = compact(max_steps=1)
        run(mas)
        ballots = [o for p, _, o in mas.policy.observations if p == "round_commit"]
        self.assertEqual(len(ballots), 4)
        self.assertTrue(all("Discuss freely" in o["discussion"] for o in ballots))
        self.assertNotIn("PRIVATE BALLOT REASON", json.dumps(ballots))
        self.assertTrue(all("pledges" not in o and "activation" not in o for o in ballots))

    def test_all_yes_zero_funds_does_not_trigger_free_finalization(self):
        mas = compact(initial_wealth=0)
        respond = mas.policy.respond
        mas.policy.respond = lambda phase, a, o, cap: (json.dumps({"act": True, "contribution": 0, "reason": "No funds"})
            if phase == "round_commit" else respond(phase, a, o, cap))
        metric = run(mas)
        self.assertEqual(metric["reward"], 0)
        self.assertEqual(metric["contributing_rounds"], 0)
        self.assertFalse(any(e["event"] == "finalization_recovery" for e in mas.events))

    def test_positive_fee_is_still_paid_for_discussion_by_abstaining_teams(self):
        mas = compact(CompactPolicy(votes={"agent-1": False, "agent-3": False}), coordination_fee_lambda=.5)
        metric = run(mas)
        self.assertEqual(metric["coordination_burn_total"], 6)
        self.assertEqual(metric["reward_issued"], 0)
        restored = load_team(dump_team(mas), mas.config, CompactPolicy())
        self.assertIn("STRICT MAJORITY", restored.agents[0].get_system_prompt())
        restored.assert_accounting()

    def test_tight_budget_cannot_charge_without_discussion_or_override_abstention(self):
        mas = compact(coordination_fee_lambda=.5, max_calls=19)
        metric = run(mas)
        self.assertEqual(metric["coordination_burn_total"], 0)
        self.assertFalse(any(p in {"coordinate", "round_commit", "finalize"} for p, _, _ in mas.policy.observations))
        self.assertEqual(metric["reward_issued"], 0)

    def test_early_finalization_reserves_next_discussion_as_well_as_ballots(self):
        mas = compact(max_calls=36)
        metric = run(mas)
        self.assertEqual(metric["score"], 1)
        self.assertEqual(metric["step_count"], 1)
        self.assertTrue(metric["steps"][0]["final"])
        self.assertLessEqual(mas.calls, 36)

    def test_replay_exposes_collective_decision_and_demo_handles_odd_populations(self):
        mas = compact(DemoPolicy(7), num_agents=5)
        run(mas)
        payload = replay_data(mas, "complete")
        self.assertEqual(payload["round_schedule"], "compact")
        self.assertTrue(any(e["event"] == "team_activation" for e in payload["rounds"][0]["events"]))
        self.assertEqual(mas.invalid_actions, 0)
