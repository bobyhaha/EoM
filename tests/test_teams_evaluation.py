import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hayekmas.adapters.teams.config import TeamConfig, TokenBudget
from hayekmas.adapters.teams.env import load_tasks
from hayekmas.adapters.teams.evaluation import (
    BASELINE_SYSTEM,
    compare,
    estimate_pass_at_k,
    evaluation_copy,
    parse_answer,
    validate,
)
from hayekmas.adapters.teams.mas import BudgetExceeded, TeamMAS
from hayekmas.adapters.teams.policy import DemoPolicy


ROOT = Path(__file__).resolve().parents[1]


class PassKTests(unittest.TestCase):
    def test_estimator_matches_exhaustive_subsets(self):
        for n in range(1, 9):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    outcomes = [any(i < c for i in subset) for subset in itertools.combinations(range(n), k)]
                    self.assertAlmostEqual(estimate_pass_at_k(n, c, k), sum(outcomes) / len(outcomes))
        self.assertAlmostEqual(estimate_pass_at_k(10, 2, 3), 1 - math.comb(8, 3) / math.comb(10, 3))

    def test_invalid_k_rejected_not_silently_omitted(self):
        for values in [(3, 2, 4), (3, 4, 1), (3, -1, 1), (3, 1, 0), (True, 1, 1)]:
            with self.assertRaises(ValueError):
                estimate_pass_at_k(*values)

    def test_bad_answers_are_not_repaired_using_reference(self):
        for raw in ["not json", '{"answer":null}', '{"answer":true}', '{"answer":NaN}', "[]"]:
            answer, error = parse_answer(raw, TokenBudget(), 256)
            self.assertIsNone(answer)
            self.assertIsNotNone(error)
        self.assertEqual(parse_answer('{"answer":0}', TokenBudget(), 256), (0, None))


class ScienceDataTests(unittest.TestCase):
    def test_bundled_frontier_science_and_no_overlap(self):
        path = ROOT / "third_party/benchmarks/frontier-science-research/data"
        train = load_tasks(path / "research_train.jsonl", "train")
        test = load_tasks(path / "research_test.jsonl", "test")
        self.assertEqual((len(train), len(test)), (40, 19))
        self.assertFalse({t.id for t in train} & {t.id for t in test})
        self.assertEqual({t.subject for t in test}, {"physics", "chemistry", "biology"})
        self.assertNotIn("answer", test[0].public())
        self.assertNotIn("rubric", test[0].public())

    def test_config_validation_checks_full_science_before_model(self):
        raw = json.loads((ROOT / "global_configs/compare_science_openrouter.json").read_text())
        raw["dataset"] = str(ROOT / raw["dataset"])
        raw["comparison"]["train_dataset"] = str(ROOT / raw["comparison"]["train_dataset"])
        raw["teams"]["rounds"] = 19
        raw["comparison"]["train_rounds"] = 40
        self.assertEqual(len(validate(raw)[4]), 19)
        raw["teams"]["rounds"] = 20
        with self.assertRaises(ValueError):
            validate(raw)
        raw["teams"]["rounds"] = 1
        raw["comparison"]["train_dataset"] = raw["dataset"]
        raw["comparison"]["train_rounds"] = 1
        with self.assertRaises(ValueError):
            validate(raw)


class ComparisonTests(unittest.TestCase):
    def test_demo_matched_tasks_shared_pool_and_no_test_reflection(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            result = compare(
                {
                    "backend": "demo",
                    "teams": {"num_agents": 4, "rounds": 2},
                    "comparison": {"samples": 3, "ks": [1, 2, 3], "seeds": [7, 17]},
                },
                out=out,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_pairs"], 4)
            self.assertTrue(all(row["pass_rate"] == 1 for row in result["summary"]))
            self.assertEqual(len(result["summary"]), 5)
            requests = [json.loads(x) for x in (out / "requests.jsonl").read_text().splitlines()]
            solves = [r for r in requests if r["event"] == "request" and r["kind"] == "solve"]
            self.assertEqual(len(solves), 12)  # Single agent reuses first sample, not 4 extra calls.
            for task in {r["task_id"] for r in solves}:
                self.assertEqual(len({r["prompt"] for r in solves if r["task_id"] == task}), 1)
            self.assertFalse(any(r["kind"] == "reflect" for r in requests))
            for pair in result["pairs"]:
                self.assertTrue((out / pair["team"]["replay"]).exists())
            self.assertEqual(result["total_actual_usage"]["cost_usd"], 0)
            self.assertEqual(result["summary"][1]["usage"]["calls"], 4)
            self.assertEqual(result["summary"][-1]["usage"]["calls"], 12)
            self.assertTrue((out / "comparison.html").exists())
            with self.assertRaises(FileExistsError):
                compare({"teams": {"rounds": 1}}, out=out)

    def test_global_call_cap_persists_incomplete_without_headline_scores(self):
        for enabled, expected_calls in ((True, 0), (False, 1)):
            with self.subTest(finalization_enabled=enabled), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "run"
                with self.assertRaises(BudgetExceeded):
                    compare(
                        {"teams": {"num_agents": 2, "rounds": 1, "finalization_enabled": enabled},
                         "comparison": {"max_total_calls": 1}}, out=out,
                    )
                result = json.loads((out / "comparison.json").read_text())
                self.assertEqual(result["status"], "interrupted")
                self.assertEqual(result["summary"], [])
                self.assertEqual(result["total_actual_usage"]["calls"], expected_calls)
                self.assertTrue((out / "seed-7/task-001/team/interrupted_state.json").exists())

    def test_trained_snapshot_copies_population_and_manager_together(self):
        base = TeamMAS(TeamConfig(num_agents=2), DemoPolicy(7))
        base.team_manager.create_team(base.agents, 0)
        base.agents[0].wealth = 25
        base.agents[0].trainable_system_prompt = "learned strategy"
        base.agents[0].summary = "TRAIN HISTORY"
        clone = evaluation_copy(base, 9, base.policy)
        clone.agents[0].wealth = 1
        clone.agents[0].trainable_system_prompt = "TEST CHANGE"
        self.assertEqual(base.agents[0].wealth, 25)
        self.assertEqual(base.agents[0].trainable_system_prompt, "learned strategy")
        self.assertIs(clone.team_manager.agent("agent-0"), clone.agents[0])
        self.assertEqual(clone.initial_total, 45)
        second = evaluation_copy(base, 10, base.policy)
        self.assertEqual(second.agents[0].summary, "TRAIN HISTORY")
        self.assertNotIn("TEST CHANGE", second.agents[0].get_system_prompt())

    def test_training_and_testing_have_separate_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text(json.dumps({"id": "train", "problem": "sum: [1, 2]", "answer": 3}) + "\n")
            result = compare(
                {
                    "teams": {"num_agents": 2, "rounds": 1},
                    "comparison": {"samples": 1, "ks": [1], "train_rounds": 1, "train_dataset": str(train)},
                },
                out=root / "run",
            )
            requests = [json.loads(x) for x in (root / "run/requests.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["kind"] == "reflect" and r["phase"] == "train" for r in requests))
            self.assertFalse(any(r["kind"] in {"reflect", "formation"} and r["phase"] == "test" for r in requests))
            self.assertEqual(result["training"][0]["tasks"], 1)
            self.assertGreater(result["training"][0]["usage"]["calls"], 0)
            self.assertEqual(
                result["total_actual_usage"]["calls"],
                result["training"][0]["usage"]["calls"] + sum(r["usage"]["calls"] for r in result["summary"][:2]),
            )

    @patch("hayekmas.adapters.teams.evaluation.ModelPolicy")
    def test_science_judge_is_shared_and_private(self, model):
        class Native:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, system_prompt=None, max_tokens=None):
                self.calls.append((prompt, system_prompt, max_tokens))
                if system_prompt == BASELINE_SYSTEM:
                    return '{"answer":"independent solution"}'
                return "SCORE: 0.75\nREASON: partial"

        class Policy:
            label = "llm"

            def __init__(self):
                self.client = Native()

            def respond(self, phase, a, o, max_tokens):
                if phase == "contribute":
                    return '{"contribution":1}'
                if phase == "discuss":
                    return '{"candidate":"team solution","final":true}'
                if phase == "vote":
                    return json.dumps({"candidate_id": o["candidates"][0]["id"]})
                return "{}"

        policy = Policy()
        model.return_value = policy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "test.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "task_group_id": "science",
                        "problem": "PUBLIC QUESTION",
                        "answer": "SECRET RUBRIC",
                        "subject": "physics",
                    }
                )
                + "\n"
            )
            result = compare(
                {
                    "backend": "llm",
                    "environment": "researchworld",
                    "dataset": str(data),
                    "teams": {"condition": "individual", "num_agents": 1, "rounds": 1, "solution_tokens": 512},
                    "comparison": {"samples": 2, "ks": [1, 2], "pass_threshold": 0.7},
                },
                out=root / "run",
            )
            self.assertTrue(result["pairs"][0]["team"]["passed"])
            self.assertEqual(result["total_actual_usage"]["judge_calls"], 3)
            generation = [p for p, s, c in policy.client.calls if s == BASELINE_SYSTEM]
            judges = [p for p, s, c in policy.client.calls if s is None]
            self.assertEqual(generation[0], generation[1])
            self.assertTrue(all("SECRET RUBRIC" not in p for p in generation))
            self.assertTrue(all("SECRET RUBRIC" in p for p in judges))
            self.assertIsNone(result["total_actual_usage"]["cost_usd"])
            self.assertEqual(result["summary"][1]["usage"]["calls"], 2)  # one solve + one judge


class OpenRouterComparisonTests(unittest.TestCase):
    @patch("hayekmas.adapters.teams.openrouter.requests.post")
    def test_one_shared_provider_ledger_covers_team_baselines_and_judges(self, post):
        from unittest.mock import Mock

        def respond(url, **kwargs):
            prompt = kwargs["json"]["messages"][-1]["content"]
            try:
                value = json.loads(prompt)
            except ValueError:
                result = "SCORE: 0.75\nREASON: partial"
            else:
                if "task" in value:
                    result = '{"answer":"complete scientific solution"}'
                else:
                    instruction = value["instruction"]
                    obs = value["observation"]
                    if instruction.startswith("As a singleton"):
                        result = '{"contribution":1}'
                    elif instruction.startswith("Use the team channel"):
                        result = '{"candidate":"complete scientific solution","final":true}'
                    elif instruction.startswith("Choose a candidate"):
                        result = json.dumps({"candidate_id": obs["candidates"][0]["id"]})
                    else:
                        result = "{}"
            response = Mock(status_code=200)
            response.json.return_value = {
                "id": f"gen-{post.call_count}",
                "model": "test-model",
                "provider": "mock",
                "usage": {"cost": 0.000001, "prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"message": {"content": result}}],
            }
            return response

        post.side_effect = respond
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "test.jsonl"
            data.write_text(json.dumps({"id": "x", "problem": "scientific question", "answer": "RUBRIC"}) + "\n")
            result = compare(
                {
                    "backend": "llm",
                    "environment": "researchworld",
                    "dataset": str(data),
                    "model": {"api": "openrouter", "name": "test-model", "api_key": "do-not-log-this-key"},
                    "budget": {"max_usd": 1, "prompt_usd_per_million": 1, "completion_usd_per_million": 2},
                    "teams": {"num_agents": 1, "condition": "individual", "rounds": 1},
                    "comparison": {"samples": 2, "ks": [1, 2]},
                },
                out=root / "run",
            )
            usage = result["total_actual_usage"]
            self.assertEqual(usage["calls"], post.call_count)
            self.assertEqual(usage["judge_calls"], 3)
            self.assertAlmostEqual(usage["cost_usd"], 0.000001 * post.call_count)
            self.assertEqual(usage["native_prompt_tokens"], 10 * post.call_count)
            ledger = [json.loads(x) for x in (root / "run/api_usage.jsonl").read_text().splitlines()]
            self.assertEqual({e["method"] for e in ledger}, {"team", "independent"})
            self.assertEqual(len([e for e in ledger if e["event"] == "api_reserved"]), post.call_count)
            self.assertNotIn(
                "do-not-log-this-key", "".join(p.read_text() for p in (root / "run").rglob("*") if p.is_file())
            )
