"""Regression coverage for answer delivery, bounded recovery and legacy behavior."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ExactTaskEnv, ResearchTaskEnv, Task, demo_tasks
from hayekmas.adapters.teams.evaluation import ExperimentPolicy
from hayekmas.adapters.teams.mas import BudgetExceeded, TeamMAS
from hayekmas.adapters.teams.policy import DemoPolicy
from hayekmas.adapters.teams.replay import replay_data
from hayekmas.experiments.team_checkpoint import dump_team, load_team


class DeliveryPolicy:
    label = "test"

    def __init__(self, discuss=None, finalize=None, vote=None):
        self.discuss = discuss
        self.finalize = finalize
        self.vote = vote
        self.requests = []
        self.judgments = []
        self.client = self

    def generate(self, prompt, **kwargs):
        self.judgments.append(prompt)
        return "SCORE: 1\nREASON: correct"

    def respond(self, phase, agent, observation, cap):
        self.requests.append((phase, agent.name, deepcopy(observation), cap))
        callback = getattr(self, phase, None)
        if phase in {"discuss", "finalize", "vote"} and callback:
            answer = callback(agent, observation)
        elif phase in {"negotiate", "contribute"}:
            answer = {"contribution": 1, "message": "shared notes"}
        elif phase == "discuss":
            answer = {"message": "3 + 4 equals 7", "candidate": None, "final": False}
        elif phase == "finalize":
            answer = {"candidate": "7", "final": True}
        elif phase == "vote":
            answer = {"candidate_id": observation["candidates"][0]["id"]}
        else:
            answer = {}
        return answer if isinstance(answer, str) else json.dumps(answer)


def make_engine(policy=None, **config):
    cfg = TeamConfig(**{"num_agents": 2, "condition": "random_fixed", **config})
    return TeamMAS(cfg, policy or DeliveryPolicy())


def run_task(engine, *, reflection=False):
    env = ResearchTaskEnv(Task("test", "What is 3 + 4?", "PRIVATE RUBRIC"), 12, engine.judge)
    result = engine.run_one_episode(env, formation=False, reflection=reflection)
    return result, env


class FinalizationTests(unittest.TestCase):
    def test_discussion_without_candidate_recovers_and_grades_once(self):
        engine = make_engine()
        metric, env = run_task(engine)
        self.assertEqual(env.native.final_answer, "7")
        self.assertEqual(metric["score"], 1)
        self.assertEqual(metric["step_count"], 1)
        self.assertEqual(len(engine.policy.judgments), 1)
        self.assertNotIn("PRIVATE RUBRIC", json.dumps(engine.policy.requests))
        self.assertIn("PRIVATE RUBRIC", engine.policy.judgments[0])
        self.assertEqual([a.wealth for a in engine.agents], [25, 25])
        self.assertEqual(engine.bid_paid_total, 2)
        self.assertEqual(len([e for e in engine.events if e["event"] == "auction"]), 1)
        final_requests = [r for r in engine.policy.requests if r[0] == "finalize"]
        self.assertEqual(len(final_requests), 2)
        self.assertTrue(all(r[2]["must_finalize"] for r in final_requests))
        self.assertTrue(all("3 + 4 equals 7" in r[2]["discussion"] for r in final_requests))
        self.assertEqual({r[3] for r in final_requests}, {engine.config.solution_tokens})
        replay = replay_data(engine, "complete")
        self.assertTrue(any(e["event"] == "finalization_started" for e in replay["rounds"][0]["events"]))
        engine.assert_accounting()

    def test_existing_final_answer_does_not_trigger_extra_generation(self):
        policy = DeliveryPolicy(discuss=lambda a, o: {"candidate": "7", "final": True})
        engine = make_engine(policy)
        metric, _ = run_task(engine)
        self.assertEqual(metric["score"], 1)
        self.assertFalse(any(r[0] == "finalize" for r in policy.requests))

    def test_intermediate_step_then_empty_discussion_recovers_with_payment_chain(self):
        policy = DeliveryPolicy(discuss=lambda a, o: (
            {"candidate": "PUBLIC INTERMEDIATE", "final": False} if o["step"] == 0 else {}
        ))
        engine = make_engine(policy)
        metric, _ = run_task(engine)
        self.assertEqual(metric["step_count"], 2)
        self.assertEqual(metric["score"], 1)
        self.assertEqual((engine.bid_paid_total, engine.bid_burn_total, engine.bid_transfer_total), (4, 2, 2))
        observations = [r[2] for r in policy.requests if r[0] == "finalize"]
        self.assertTrue(all("PUBLIC INTERMEDIATE" in o["environment_state"]["state"] for o in observations))
        self.assertEqual(len(policy.judgments), 1)
        engine.assert_accounting()

    def test_last_step_intermediate_is_not_promoted_to_final(self):
        policy = DeliveryPolicy(
            discuss=lambda a, o: {"candidate": "UNFINISHED", "final": False},
            finalize=lambda a, o: {"abstain": True},
        )
        engine = make_engine(policy, max_steps=1)
        metric, env = run_task(engine)
        self.assertEqual(metric["score"], 0)
        self.assertFalse(env.native.final_answer)
        self.assertEqual(policy.judgments, [])
        self.assertEqual(len([r for r in policy.requests if r[0] == "finalize"]), 2)

    def test_malformed_final_proposal_gets_one_correction(self):
        invalid = [
            "{", "[]", {}, {"candidate": " "}, {"candidate": 7, "final": True},
            {"candidate": "7", "final": "true"}, {"candidate": "7", "final": False},
            {"candidate": "7", "final": True, "abstain": True},
            {"message": "7"}, {"candidate": "long " * 1000, "final": True},
        ]
        for response in invalid:
            with self.subTest(response=str(response)[:80]):
                policy = DeliveryPolicy(finalize=lambda a, o: (
                    response if o["attempt"] == 1 else {"candidate": "7", "final": True}
                ))
                engine = make_engine(policy)
                metric, _ = run_task(engine)
                self.assertEqual(metric["score"], 1)
                requests = [r for r in policy.requests if r[0] == "finalize"]
                self.assertEqual(len(requests), 4)
                self.assertTrue(all(r[2]["correction"] for r in requests if r[2]["attempt"] == 2))

    def test_repeated_invalid_output_exhausts_recovery_without_fake_answer(self):
        policy = DeliveryPolicy(finalize=lambda a, o: {"message": "not a candidate"})
        engine = make_engine(policy)
        metric, env = run_task(engine)
        self.assertEqual(metric["score"], 0)
        self.assertFalse(env.native.final_answer)
        self.assertEqual(len([r for r in policy.requests if r[0] == "finalize"]), 4)
        self.assertFalse(any(r[0] == "vote" for r in policy.requests))
        self.assertEqual(policy.judgments, [])

    def test_explicit_abstention_is_not_retried(self):
        policy = DeliveryPolicy(finalize=lambda a, o: {"abstain": True})
        engine = make_engine(policy)
        metric, _ = run_task(engine)
        self.assertEqual(metric["score"], 0)
        self.assertEqual(len([r for r in policy.requests if r[0] == "finalize"]), 2)
        self.assertEqual(len([e for e in engine.events if e["event"] == "finalization_abstained"]), 2)
        self.assertEqual(policy.judgments, [])

    def test_one_abstention_does_not_discard_another_members_answer(self):
        policy = DeliveryPolicy(finalize=lambda a, o: (
            {"abstain": True} if a.name == "agent-0" else {"candidate": "7", "final": True}
        ))
        engine = make_engine(policy)
        metric, _ = run_task(engine)
        self.assertEqual(metric["score"], 1)
        self.assertEqual(len([e for e in engine.events if e["event"] == "candidate"]), 1)

    def test_all_invalid_votes_do_not_invent_a_winner(self):
        policy = DeliveryPolicy(vote=lambda a, o: {"candidate_id": "missing"})
        engine = make_engine(policy)
        metric, env = run_task(engine)
        self.assertEqual(metric["score"], 0)
        self.assertFalse(env.native.final_answer)
        self.assertEqual(policy.judgments, [])


class FinalizationBudgetTests(unittest.TestCase):
    def test_tight_budget_reserves_corrections_votes_and_real_grader(self):
        policy = DeliveryPolicy(finalize=lambda a, o: (
            {} if o["attempt"] == 1 else {"candidate": "7", "final": True}
        ))
        engine = make_engine(policy, max_calls=11)
        metric, _ = run_task(engine, reflection=True)
        self.assertEqual(metric["score"], 1)
        self.assertEqual(engine.calls, 11)  # 4 bids + 4 finalization + 2 votes + 1 judge
        self.assertFalse(any(r[0] in {"discuss", "reflect"} for r in policy.requests))
        self.assertEqual(len(policy.judgments), 1)
        self.assertTrue(any(e["event"] == "phase_skipped" and e["phase"] == "reflection" for e in engine.events))
        engine.assert_accounting()

    def test_too_small_budget_rejects_auction_before_calls_or_payments(self):
        engine = make_engine(max_calls=10)
        with self.assertRaises(BudgetExceeded):
            run_task(engine)
        self.assertEqual(engine.calls, 0)
        self.assertEqual(engine.bid_paid_total, 0)
        self.assertEqual([a.wealth for a in engine.agents], [20, 20])

    def test_shortened_discussion_reports_actual_remaining_turns(self):
        engine = make_engine(max_calls=13)
        metric, _ = run_task(engine)
        self.assertEqual(metric["score"], 1)
        requests = [r for r in engine.policy.requests if r[0] == "discuss"]
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r[2]["discussion_window"]["your_turns_remaining_after_this"] == 0 for r in requests))
        self.assertLessEqual(engine.calls, 13)

    def test_finalizes_instead_of_starting_unaffordable_next_auction(self):
        policy = DeliveryPolicy(discuss=lambda a, o: {"candidate": "DRAFT", "final": False})
        engine = make_engine(policy, max_calls=15)
        metric, _ = run_task(engine)
        self.assertEqual(metric["score"], 1)
        self.assertEqual(metric["step_count"], 1)
        self.assertEqual(len([e for e in engine.events if e["event"] == "auction"]), 1)
        votes = [r for r in policy.requests if r[0] == "vote"]
        self.assertTrue(all(c["final"] and c["answer"] == "7" for r in votes for c in r[2]["candidates"]))

    def test_optional_formation_cannot_spend_answer_budget(self):
        engine = make_engine(condition="dynamic", max_calls=6)
        env = ResearchTaskEnv(Task("x", "3 + 4", "PRIVATE"), 12, engine.judge)
        metric = engine.run_one_episode(env)
        self.assertEqual(metric["score"], 1)
        self.assertFalse(any(r[0] == "formation" for r in engine.policy.requests))
        self.assertLessEqual(engine.calls, 6)

    def test_reservation_respects_global_comparison_budget(self):
        with tempfile.TemporaryDirectory() as d:
            policy = DeliveryPolicy(finalize=lambda a, o: (
                {} if o["attempt"] == 1 else {"candidate": "7", "final": True}
            ))
            cfg = TeamConfig(num_agents=2, condition="random_fixed")
            metered = ExperimentPolicy(policy, cfg, 21, Path(d))
            metered.totals["calls"] = 10  # Other comparison arms already spent these calls.
            engine = TeamMAS(cfg, metered)
            metric, _ = run_task(engine)
            self.assertEqual(metric["score"], 1)
            self.assertEqual(metered.totals["calls"], 21)
            self.assertEqual(engine.calls, 11)


class FinalizationCompatibilityTests(unittest.TestCase):
    def test_legacy_flag_reproduces_prechange_trace(self):
        engine = TeamMAS(TeamConfig(num_agents=4, rounds=6, finalization_enabled=False), DemoPolicy(7))
        for task in demo_tasks(7, 6):
            engine.run_one_episode(ExactTaskEnv(task, 12))
        trace = {"metrics": engine.metrics, "events": engine.events, "state": engine.state()}
        trace["state"]["config"].pop("finalization_enabled")
        trace["events"][0]["config"].pop("finalization_enabled")
        # New opt-in fields only add configuration metadata to legacy traces.
        for config in (trace["state"]["config"], trace["events"][0]["config"]):
            for key in ("bidding_mode", "collaboration_mode", "bid_tokens", "interaction_protocol", "coordination_fee_lambda", "round_schedule",
                        "token_profile", "discussion_tokens", "formation_context_tokens", "pledge_context_tokens", "evolution_enabled", "population_cap_multiplier", "birth_interval",
                        "num_births_per_interval", "p_a", "p_b", "periodical_good_p", "rent", "rent_interval"):
                config.pop(key)
        digest = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
        # Captured from the unmodified engine before implementing finalization.
        self.assertEqual(digest, "0e4cb6f8352d84ba9e1e3397d387a6b04c160e64df76fafca1ae4a4caeb1598f")

    def test_legacy_empty_discussion_still_ends_without_recovery(self):
        engine = make_engine(finalization_enabled=False)
        metric, _ = run_task(engine)
        self.assertEqual(metric["score"], 0)
        self.assertFalse(any(r[0] == "finalize" for r in engine.policy.requests))

    def test_recovery_checkpoint_continuation_matches_uninterrupted(self):
        engine = make_engine()
        run_task(engine)
        data = json.loads(json.dumps(dump_team(engine)))
        clone = load_team(data, engine.config, DeliveryPolicy())
        self.assertEqual(run_task(engine)[0], run_task(clone)[0])
        self.assertEqual(dump_team(engine), dump_team(clone))

    def test_old_config_loads_and_flag_is_strict_boolean(self):
        self.assertTrue(TeamConfig(num_agents=2).finalization_enabled)
        for invalid in (0, 1, "false", None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                TeamConfig(finalization_enabled=invalid)


if __name__ == "__main__":
    unittest.main()
