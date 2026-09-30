import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hayekmas.adapters.teams.openrouter import OpenRouterClient, SpendLimitExceeded
from hayekmas.adapters.teams.runtime import run


BUDGET = {"max_usd": 1, "prompt_usd_per_million": 1, "completion_usd_per_million": 2}


def response(cost=0.0001, content="{}"):
    result = Mock(status_code=200)
    result.json.return_value = {
        "id": "gen-test",
        "model": "test",
        "provider": "mock",
        "usage": {"cost": cost, "prompt_tokens": 8, "completion_tokens": 2},
        "choices": [{"message": {"content": content}}],
    }
    return result


class OpenRouterTests(unittest.TestCase):
    @patch("hayekmas.adapters.teams.openrouter.requests.post")
    def test_provider_protocol_and_reported_cost(self, post):
        post.return_value = response()
        client = OpenRouterClient("test", budget=BUDGET, api_key="secret-sentinel")
        recorded = []
        client.sink = recorded.append
        self.assertEqual(client.generate("hi", max_tokens=64), "{}")
        self.assertEqual(client.usage()["reported_cost_usd"], 0.0001)
        self.assertEqual([e["event"] for e in recorded], ["api_reserved", "api_charged"])
        self.assertNotIn("secret-sentinel", json.dumps(recorded))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["provider"]["max_price"], {"prompt": 1, "completion": 2, "request": 0})
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        self.assertEqual(post.call_args.args[0], OpenRouterClient.URL)

    @patch("hayekmas.adapters.teams.openrouter.requests.post")
    def test_model_default_sampling_omits_temperature(self, post):
        post.return_value = response()
        client = OpenRouterClient("openai/gpt-6-luna", budget=BUDGET, api_key="test", temperature=None)
        client.generate("hi", max_tokens=64)
        self.assertNotIn("temperature", post.call_args.kwargs["json"])

    @patch("hayekmas.adapters.teams.openrouter.requests.post")
    def test_reserves_before_dispatch_and_stops_before_exceeding_budget(self, post):
        client = OpenRouterClient("test", budget={**BUDGET, "max_usd": 0.000001}, api_key="test")
        with self.assertRaises(SpendLimitExceeded):
            client.generate("hi")
        post.assert_not_called()

    @patch("hayekmas.adapters.teams.openrouter.requests.post")
    def test_missing_cost_and_transport_errors_keep_reservation_and_never_retry(self, post):
        for failure in ("missing_cost", "transport", "expensive"):
            client = OpenRouterClient("test", budget=BUDGET, api_key="test")
            post.reset_mock()
            post.side_effect = RuntimeError("timeout") if failure == "transport" else None
            post.return_value = response(None if failure == "missing_cost" else 0.5)
            with self.assertRaises(RuntimeError):
                client.generate("hi", max_tokens=64)
            with self.assertRaises(SpendLimitExceeded):
                client.generate("hi", max_tokens=64)
            self.assertEqual(post.call_count, 1)
            self.assertTrue(client.usage()["blocked"])
            if failure != "expensive":
                self.assertGreater(client.usage()["reserved_unconfirmed_usd"], 0)

    @patch("hayekmas.adapters.teams.openrouter.requests.post")
    def test_runtime_persists_accounting_on_error_without_key(self, post):
        post.return_value = response(None)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            with self.assertRaises(RuntimeError):
                run(
                    {
                        "backend": "llm",
                        "teams": {"num_agents": 2, "rounds": 1},
                        "budget": BUDGET,
                        "model": {"api": "openrouter", "name": "test", "api_key": "secret-sentinel"},
                    },
                    out=out,
                    plots=False,
                )
            self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "interrupted")
            self.assertGreater(json.loads((out / "usage.json").read_text())["provider"]["reserved_unconfirmed_usd"], 0)
            ledger = (out / "api_usage.jsonl").read_text()
            self.assertIn("api_reserved", ledger)
            self.assertIn("api_stopped", ledger)
            self.assertNotIn("secret-sentinel", "\n".join(p.read_text() for p in out.iterdir()))
