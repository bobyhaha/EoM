from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from hayekmas.adapters.teams.agent import TeamAction
from hayekmas.adapters.teams.env import ResearchTaskEnv, Task
from hayekmas.adapters.teams.runtime import run


class NativeEnvironmentTests(unittest.TestCase):
    def test_original_researchworld_grader_is_used(self):
        prompts = []

        def judge(prompt):
            prompts.append(prompt)
            return "SCORE: 0.75\nREASON: Partial credit."

        env = ResearchTaskEnv(Task("science", "Public problem", "SECRET RUBRIC"), 12, judge)
        env.initialize()
        self.assertNotIn("SECRET RUBRIC", env.get_state_description())
        reward = env.apply(TeamAction("A proposed solution", "team-1"))
        self.assertEqual(reward, 9)
        self.assertIn("SECRET RUBRIC", prompts[0])
        self.assertEqual(env.get_terminal_score(), 0.75)
        self.assertTrue(env.reach_termination())
        self.assertEqual(env.native.final_answer, "A proposed solution")

    def test_upstream_swallowed_judge_failure_is_surfaced(self):
        def judge(prompt):
            raise ConnectionError("offline")

        env = ResearchTaskEnv(Task("x", "q", "rubric"), 12, judge)
        with self.assertRaises(ConnectionError):
            env.apply(TeamAction("answer"))
        env = ResearchTaskEnv(Task("x", "q", "rubric"), 12, lambda prompt: "SCORE: nan")
        with self.assertRaises(ValueError):
            env.apply(TeamAction("answer"))


class LocalhostIntegrationTests(unittest.TestCase):
    def test_live_policy_through_original_eom_http_client(self):
        self.check_live_policy(recovery=False)

    def test_finalization_through_original_eom_http_client(self):
        self.check_live_policy(recovery=True)

    def check_live_policy(self, recovery):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(payload)
                request = json.loads(payload["messages"][-1]["content"])
                instruction, obs = request["instruction"], request["observation"]
                if instruction.startswith("Discuss how"):
                    response = {"contribution": 1, "message": "Can we each contribute one?"}
                elif instruction.startswith("Use the team channel"):
                    response = {"message": "My candidate is seven.", "candidate": None if recovery else "7"}
                elif instruction.startswith("Discussion has ended"):
                    response = {"candidate": "7", "final": True}
                elif instruction.startswith("Choose a candidate"):
                    response = {"candidate_id": obs["candidates"][0]["id"]}
                elif instruction.startswith("Choose whether"):
                    response = {"reflect": True}
                elif instruction.startswith("Inspect only"):
                    response = {"analysis": "We earned a reward after a joint contribution."}
                elif instruction.startswith("Use your inspection"):
                    response = {"strategy": "Consider past profits when discussing contributions."}
                else:
                    response = {}
                content = json.dumps({"choices": [{"message": {"content": json.dumps(response)}}]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(content.encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / "tasks.jsonl"
                data.write_text(json.dumps({"id": "x", "problem": "3+4", "answer": 7}) + "\n")
                result = run(
                    {
                        "backend": "llm",
                        "dataset": str(data),
                        "model": {
                            "api": "localhost",
                            "name": "test-model",
                            "api_key": "test-secret-sentinel",
                            "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                        },
                        "teams": {
                            "condition": "random_fixed",
                            "num_agents": 4,
                            "rounds": 1,
                            "inspect_tokens": 80,
                            "update_tokens": 96,
                        },
                    },
                    out=root / "run",
                    plots=False,
                )
                self.assertEqual(result["mean_score"], 1)
                self.assertEqual(result["invalid_actions"], 0)
                self.assertEqual(
                    sum("Discussion has ended" in r["messages"][-1]["content"] for r in requests),
                    2 if recovery else 0,
                )
                self.assertEqual(len(requests), result["decision_calls"])
                self.assertEqual(sum(r["max_tokens"] == 80 for r in requests), 4)
                self.assertEqual(sum(r["max_tokens"] == 96 for r in requests), 4)
                self.assertNotIn("test-secret-sentinel", (root / "run" / "manifest.json").read_text())
                replay = json.loads((root / "run" / "replay.json").read_text())
                self.assertEqual(replay["backend"], "llm")
                self.assertTrue(any(e["event"] == "strategy_updated" for e in replay["rounds"][0]["events"]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
