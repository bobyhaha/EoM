import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import load_tasks
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.policy import DemoPolicy, INSTRUCTIONS
from hayekmas.experiments.team_checkpoint import dump_team
from hayekmas.experiments.team_finalization_evaluation import REPO, prepare, run, sha


def fixture(base, *, protocol_variant="finalization", condition="dynamic"):
    source = base / "old"
    (source / "teams").mkdir(parents=True)
    config = TeamConfig(num_agents=2, max_steps=1, condition=condition, context_tokens=65536,
                        environment_tokens=16384, evidence_tokens=8192)
    engine = TeamMAS(config, DemoPolicy(7))
    engine.round = 40
    (source / "teams/protocol.json").write_text(json.dumps({"model": "openai/gpt-6-luna", "team_config": asdict(config)}))
    (source / "teams/epoch-1.json").write_text(json.dumps(dump_team(engine)))
    tasks = load_tasks(REPO / "third_party/benchmarks/frontier-science-research/data/research_test.jsonl", "test")
    for index, task in enumerate(tasks):
        folder = source / "teams/evaluation/epoch-1" / f"{index+1:02d}-{task.id}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({"task_id": task.id, "score": 0.2}))
    root = base / "new"
    prepare(root, source, protocol_variant=protocol_variant)
    return root, source


class EvaluationTests(unittest.TestCase):
    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_all_19_tasks_isolated_and_completed_run_never_repeats_calls(self, post):
        phases = []

        def response(url, **kwargs):
            payload = kwargs["json"]
            if "response_format" in payload:
                request = json.loads(payload["messages"][-1]["content"])
                phase = next(k for k, v in INSTRUCTIONS.items() if v == request["instruction"])
                phases.append(phase)
                observation = request["observation"]
                if phase in ("contribute", "negotiate"):
                    decision = {"contribution": 1}
                elif phase == "discuss":
                    decision = {"message": "Reasoning notes", "candidate": None, "final": False}
                elif phase == "finalize":
                    decision = {"candidate": "Mock final answer", "final": True}
                elif phase == "vote":
                    decision = {"candidate_id": observation["candidates"][0]["id"]}
                else:
                    self.fail(f"Unexpected phase {phase}")
                content = json.dumps(decision)
            else:
                content = "SCORE: 0.6\nREASON: Mocked grade for runner verification"
            result = Mock(status_code=200)
            result.json.return_value = {
                "usage": {"cost": 0.0001, "prompt_tokens": 10, "completion_tokens": 10},
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            }
            return result

        post.side_effect = response
        with tempfile.TemporaryDirectory() as d:
            root, source = fixture(Path(d))
            before = sha(source / "teams/epoch-1.json")
            summary = run(root, "unit-test-credential")
            self.assertEqual(summary["status"], "complete", summary.get("error"))
            self.assertEqual(summary["completed"], 19)
            self.assertEqual(summary["submitted"], 19)
            self.assertAlmostEqual(summary["mean_score"], 0.6)
            self.assertEqual(before, sha(source / "teams/epoch-1.json"))
            self.assertEqual(before, sha(root / "checkpoint.json"))
            results = [json.loads(p.read_text()) for p in sorted((root / "teams/evaluation").glob("*/result.json"))]
            self.assertTrue(all(r["population_before"] == results[0]["population_before"] for r in results))
            self.assertTrue(all(r["finalization_windows"] == 1 for r in results))
            self.assertNotIn("formation", phases)
            self.assertNotIn("reflect", phases)
            self.assertTrue((root / "STOP").exists())
            calls = post.call_count
            run(root, "unit-test-credential")
            self.assertEqual(post.call_count, calls)
            self.assertNotIn("unit-test-credential", (root / "teams/requests.jsonl").read_text())

    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_stop_file_prevents_spending_and_partial_run_has_no_full_mean(self, post):
        with tempfile.TemporaryDirectory() as d:
            root, _ = fixture(Path(d))
            (root / "STOP").touch()
            summary = run(root, "unit-test-credential")
            self.assertEqual(summary["status"], "stopped")
            self.assertEqual(summary["completed"], 0)
            self.assertIsNone(summary["mean_score"])
            self.assertEqual(summary["usage"]["cost_usd"], 0)
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
