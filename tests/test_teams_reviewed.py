"""Behavioral coverage for the opt-in cheaper bidding and reviewed submission."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ExactTaskEnv, ResearchTaskEnv, Task, demo_tasks
from hayekmas.adapters.teams.evaluation import ExperimentPolicy
from hayekmas.adapters.teams.mas import BudgetExceeded, TeamMAS
from hayekmas.adapters.teams.policy import DemoPolicy, INSTRUCTIONS
from hayekmas.adapters.teams.replay import replay_data
from hayekmas.experiments.team_checkpoint import dump_team, load_team
from hayekmas.experiments.team_finalization_evaluation import run, sha
from hayekmas.experiments.campaign_client import canonical_decision
from test_team_finalization_evaluation import fixture


class ReviewPolicy:
    label = "test"

    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.requests, self.judgments = [], []
        self.client = self

    def generate(self, prompt, **kwargs):
        self.judgments.append(prompt)
        return "SCORE: 1\nREASON: correct"

    def respond(self, phase, agent, observation, cap):
        self.requests.append((phase, agent.name, deepcopy(observation), cap))
        answer = self.answer(phase, agent.name, observation)
        return answer if isinstance(answer, str) else json.dumps(answer)

    def answer(self, phase, name, observation):
        if phase in self.overrides:
            return self.overrides[phase](name, observation)
        if phase in {"assess_bid", "negotiate", "contribute"}:
            return {"contribution": 1, "message": "SEALED_" + name}
        if phase in {"draft", "independent_check"}:
            return {"candidate": "PRIVATE_DRAFT_" + name, "final": True, "checklist": ["Compute 3 + 4"]}
        if phase == "review":
            return {"checks": [{"requirement": "Compute 3 + 4", "status": "missing",
                                "evidence": "The final sum must be 7"}], "issues": ["FIX_SUM_TO_7"]}
        if phase == "revise":
            return {"candidate": "7", "final": True, "checklist": ["Compute 3 + 4"],
                    "resolutions": ["Recomputed the sum as 7"]}
        return {}


def engine(policy=None, **kwargs):
    return TeamMAS(TeamConfig(**{
        "num_agents": 2, "condition": "random_fixed", "bidding_mode": "sealed",
        "collaboration_mode": "reviewed", "solution_tokens": 512, **kwargs,
    }), policy or ReviewPolicy())


def episode(e):
    env = ResearchTaskEnv(Task("test", "What is 3 + 4?", "SECRET_RUBRIC"), 12, e.judge)
    metric = e.run_one_episode(env, formation=False, reflection=False)
    return metric, env


class ReviewedTests(unittest.TestCase):
    def test_independent_drafts_review_then_revision_with_real_grader_and_payments(self):
        e = engine()
        metric, env = episode(e)
        self.assertEqual(env.native.final_answer, "7")
        self.assertEqual(metric["step_count"], 1)
        self.assertEqual(metric["score"], 1)
        self.assertEqual(len(e.policy.judgments), 1)
        self.assertNotIn("SECRET_RUBRIC", json.dumps(e.policy.requests))
        self.assertIn("SECRET_RUBRIC", e.policy.judgments[0])
        self.assertEqual([a.wealth for a in e.agents], [25, 25])
        self.assertEqual(e.bid_paid_total, 2)
        for phase, name, obs, cap in e.policy.requests:
            if phase == "assess_bid":
                self.assertNotIn("SEALED_", json.dumps(obs))
                self.assertNotIn("pledges", obs)
                self.assertEqual(cap, 256)
            if phase in {"draft", "independent_check"}:
                self.assertNotIn("PRIVATE_DRAFT", json.dumps(obs))
                self.assertTrue(obs["must_finalize"])
            if phase == "review":
                self.assertIn("PRIVATE_DRAFT_agent-0", json.dumps(obs))
                self.assertIn("PRIVATE_DRAFT_agent-1", json.dumps(obs))
            if phase == "revise":
                self.assertIn("FIX_SUM_TO_7", json.dumps(obs))
        self.assertEqual(e.calls, 7)  # two bids, two drafts, one review/revision/grader
        self.assertFalse(any(r[0] in {"vote", "discuss", "negotiate"} for r in e.policy.requests))
        e.assert_accounting()

    def test_twelve_agents_one_bid_each_only_winners_solve(self):
        e = engine(num_agents=12)
        metric, _ = episode(e)
        bids = [r for r in e.policy.requests if r[0] == "assess_bid"]
        self.assertEqual(len(bids), 12)
        self.assertEqual(len({r[1] for r in bids}), 12)
        self.assertEqual(e.calls, 17)
        winners = set(metric["steps"][0]["members"])
        self.assertTrue(all(r[1] in winners for r in e.policy.requests if r[0] != "assess_bid"))

    def test_worst_case_repairs_fit_reserved_budget_and_no_second_auction(self):
        def invalid_first(name, obs):
            return "{" if obs["attempt"] == 1 else {
                "candidate": "7", "final": True, "checklist": ["Sum"], "resolutions": ["Corrected JSON"],
            }
        overrides = {p: invalid_first for p in ("draft", "independent_check", "revise")}
        overrides["assess_bid"] = lambda n, o: {"contribution": 0}
        e = engine(ReviewPolicy(overrides), max_calls=12)
        metric, _ = episode(e)
        self.assertEqual(metric["score"], 1)
        self.assertEqual(e.calls, 12)
        self.assertEqual(len([v for v in e.events if v["event"] == "auction"]), 1)
        self.assertEqual(len(e.policy.judgments), 1)

    def test_insufficient_budget_rejected_before_calls_and_money(self):
        e = engine(max_calls=11)
        with self.assertRaises(BudgetExceeded):
            episode(e)
        self.assertEqual(e.calls, 0)
        self.assertEqual(e.bid_paid_total, 0)
        self.assertEqual([a.wealth for a in e.agents], [20, 20])

    def test_global_budget_honored(self):
        with tempfile.TemporaryDirectory() as d:
            e = engine()
            e.policy = ExperimentPolicy(e.policy, e.config, 13, d)
            e.policy.totals["calls"] = 4
            with self.assertRaises(BudgetExceeded):
                episode(e)
            self.assertEqual(e.calls, 0)

    def test_zero_bids_never_fabricate_winner(self):
        e = engine(ReviewPolicy({p: lambda n, o: {"contribution": 0} for p in ("assess_bid", "negotiate")}))
        _, env = episode(e)
        self.assertFalse(env.native.final_answer)
        self.assertFalse(e.policy.judgments)
        self.assertEqual(e.bid_paid_total, 0)
        self.assertEqual(sum(r[0] == "negotiate" for r in e.policy.requests), 2)

    def test_zero_initial_bids_get_one_visible_voluntary_funding_round(self):
        e = engine(ReviewPolicy({"assess_bid": lambda n, o: {"contribution": 0}}))
        _, env = episode(e)
        self.assertEqual(env.native.final_answer, "7")
        repairs = [r for r in e.policy.requests if r[0] == "negotiate"]
        self.assertEqual(len(repairs), 2)
        self.assertEqual(sum(repairs[0][2]["pledges"].values()), 0)
        self.assertEqual(sum(repairs[1][2]["pledges"].values()), 1)
        self.assertEqual(e.bid_paid_total, 2)
        self.assertEqual(e.calls, 9)

    def test_provider_wrapped_new_decisions_normalize_without_changing_values(self):
        for phase, value in (
            ("assess_bid", {"contribution": 0.4, "message": "assessment"}),
            ("draft", {"candidate": "7", "final": True, "checklist": ["sum"]}),
            ("independent_check", {"candidate": "7", "final": True, "checklist": ["sum"]}),
            ("review", {"checks": [], "issues": ["missing evidence"]}),
            ("revise", {"candidate": "7", "final": True, "checklist": ["sum"], "resolutions": []}),
        ):
            raw = 'Commentary\n' + json.dumps(value) + '\nextra text\n' + json.dumps(value)
            self.assertEqual(json.loads(canonical_decision(raw, phase)), value)
        raw = '{"candidate":"old answer","final":true}\nChanged my mind: {"abstain":true}'
        self.assertEqual(json.loads(canonical_decision(raw, "revise")), {"abstain": True})

    def test_malformed_review_is_flagged_and_revision_still_delivered(self):
        e = engine(ReviewPolicy({"review": lambda n, o: {"checks": "looks good"}}))
        _, env = episode(e)
        self.assertEqual(env.native.final_answer, "7")
        obs = next(r[2] for r in e.policy.requests if r[0] == "revise")
        self.assertIn("Review unavailable", json.dumps(obs))
        self.assertEqual(sum(r[0] == "review" for r in e.policy.requests), 1)

    def test_invalid_revision_uses_only_explicit_complete_proposal(self):
        e = engine(ReviewPolicy({"revise": lambda n, o: {"message": "not an answer"}}))
        _, env = episode(e)
        self.assertTrue(env.native.final_answer.startswith("PRIVATE_DRAFT_"))
        self.assertEqual(sum(r[0] == "revise" for r in e.policy.requests), 2)
        submission = next(x for x in e.events if x["event"] == "submission")
        self.assertEqual(submission["selection"], "draft_fallback")

    def test_all_abstain_no_fake_answer_or_grading(self):
        e = engine(ReviewPolicy({p: lambda n, o: {"abstain": True} for p in ("draft", "independent_check")}))
        _, env = episode(e)
        self.assertFalse(env.native.final_answer)
        self.assertEqual(e.calls, 4)
        self.assertFalse(e.policy.judgments)

    def test_revision_abstention_withdraws_own_proposal(self):
        e = engine(ReviewPolicy({"revise": lambda n, o: {"abstain": True}}))
        _, env = episode(e)
        author = next(r[1] for r in e.policy.requests if r[0] == "revise")
        self.assertNotIn(author, env.native.final_answer)
        self.assertEqual(sum(r[0] == "revise" for r in e.policy.requests), 1)

    def test_singleton_self_review_and_explicit_abstention(self):
        e = engine(num_agents=1)
        self.assertEqual(episode(e)[1].native.final_answer, "7")
        self.assertEqual(e.calls, 5)
        e = engine(ReviewPolicy({"revise": lambda n, o: {"abstain": True}}), num_agents=1)
        self.assertFalse(episode(e)[1].native.final_answer)

    def test_larger_team_reviews_each_peer_without_votes(self):
        e = engine(num_agents=4, fixed_team_size=4)
        episode(e)
        self.assertEqual(sum(r[0] == "independent_check" for r in e.policy.requests), 3)
        self.assertEqual(sum(r[0] == "review" for r in e.policy.requests), 3)
        self.assertEqual(e.calls, 13)
        self.assertEqual(e.collaboration_reserve(4), 14)

    def test_truncated_evidence_is_explicit(self):
        def long_answer(name, obs):
            return {"candidate": "Long independent explanation. " * 50, "final": True,
                    "checklist": ["Compute the sum"]}
        e = engine(ReviewPolicy({p: long_answer for p in ("draft", "independent_check")}), evidence_tokens=128)
        episode(e)
        reviews = [r for r in e.policy.requests if r[0] in {"review", "revise"}]
        self.assertTrue(any(d["truncated"] for r in reviews for d in r[2]["drafts"]))

    def test_checkpoint_continuation_and_replay_preserve_new_protocol(self):
        e = engine()
        episode(e)
        clone = load_team(json.loads(json.dumps(dump_team(e))), e.config, ReviewPolicy())
        episode(e)
        episode(clone)
        self.assertEqual(dump_team(e), dump_team(clone))
        kinds = {x["event"] for x in replay_data(e, "complete")["rounds"][0]["events"]}
        self.assertTrue({"review_started", "review_draft", "review_feedback", "review_revision"} <= kinds)

    def test_default_finalization_trace_is_unchanged_except_new_config_metadata(self):
        e = TeamMAS(TeamConfig(num_agents=4, rounds=6), DemoPolicy(7))
        for t in demo_tasks(7, 6):
            e.run_one_episode(ExactTaskEnv(t, 12))
        trace = {"metrics": e.metrics, "events": e.events, "state": e.state()}
        for config in (trace["state"]["config"], trace["events"][0]["config"]):
            for key in ("bidding_mode", "collaboration_mode", "bid_tokens", "interaction_protocol", "coordination_fee_lambda", "round_schedule",
                        "token_profile", "discussion_tokens", "formation_context_tokens", "pledge_context_tokens", "evolution_enabled", "population_cap_multiplier", "birth_interval",
                        "num_births_per_interval", "p_a", "p_b", "periodical_good_p", "rent", "rent_interval",
                        "objective_mode", "team_bid_rule", "team_base_bid", "terminal_policy", "shared_draft_passes"):
                config.pop(key)
        self.assertEqual(hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
                         "eb70de17b906daf50e9047b66957cb5d5f6ca30377400c6f46e10634b14f8678")

    def test_invalid_modes_fail_at_configuration(self):
        for kwargs in ({"bidding_mode": "x"}, {"collaboration_mode": "x"}, {"bid_tokens": 0},
                       {"collaboration_mode": "reviewed", "finalization_enabled": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TeamConfig(**kwargs)


class ReviewedEvaluationTests(unittest.TestCase):
    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_all_19_tasks_through_http_adapter_replay_and_resume(self, post):
        policy = ReviewPolicy()
        phases = []

        def response(url, **kwargs):
            payload = kwargs["json"]
            if "response_format" in payload:
                req = json.loads(payload["messages"][-1]["content"])
                phase = next(k for k, v in INSTRUCTIONS.items() if v == req["instruction"])
                phases.append(phase)
                obs = req["observation"]
                self.assertEqual(payload["reasoning"]["effort"], "none" if phase in {"assess_bid", "negotiate"} else "medium")
                decision = ({"contribution": 0, "message": "brief assessment"} if phase == "assess_bid"
                            else policy.answer(phase, obs["you"]["name"], obs))
                content = 'Provider wrapper\n' + json.dumps(decision) + '\nextra text\n' + json.dumps(decision)
            else:
                content = "SCORE: 0.6\nREASON: Mock grade, not a scientific result"
            result = Mock(status_code=200)
            result.json.return_value = {
                "usage": {"cost": 0.0001, "prompt_tokens": 10, "completion_tokens": 10},
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            }
            return result

        post.side_effect = response
        with tempfile.TemporaryDirectory() as d:
            root, source = fixture(Path(d), protocol_variant="reviewed", condition="random_fixed")
            initial_hash = sha(source / "teams/epoch-1.json")
            summary = run(root, "unit-test-only")
            self.assertEqual(summary["status"], "complete", summary.get("error"))
            self.assertEqual(summary["submitted"], 19)
            rows = [json.loads(p.read_text()) for p in sorted((root / "teams/evaluation").glob("*/result.json"))]
            self.assertEqual([x["task_id"] for x in rows], json.loads((root / "plan.json").read_text())["task_ids"])
            self.assertTrue(all(r["population_before"] == rows[0]["population_before"] for r in rows))
            self.assertTrue(all(r["review_windows"] == 1 and r["review_fallbacks"] == 0 for r in rows))
            self.assertTrue(all(r["funding_repairs"] == 1 for r in rows))
            self.assertEqual(initial_hash, sha(source / "teams/epoch-1.json"))
            self.assertEqual(initial_hash, sha(root / "checkpoint.json"))
            self.assertEqual(phases.count("assess_bid"), 38)
            self.assertNotIn("formation", phases)
            self.assertNotIn("reflect", phases)
            self.assertTrue((root / "STOP").exists())
            calls = post.call_count
            run(root, "unit-test-only")
            self.assertEqual(post.call_count, calls)
            self.assertNotIn("unit-test-only", (root / "teams/requests.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
