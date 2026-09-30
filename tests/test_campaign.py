import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from hayekmas.experiments.campaign_client import CampaignClient, CampaignStop, canonical_decision
from hayekmas.experiments.team_checkpoint import dump_team, load_team
from hayekmas.experiments.campaign_report import episode_rows, diagnostics, live_team_episode, build
from hayekmas.experiments.campaign_audit import closed_ledger_preservation, primary_preservation, request_range
from hayekmas.experiments.campaign_statistics import paired_intervals
from hayekmas.experiments.campaign_findings import collect_findings, render_findings
from hayekmas.experiments.campaign_grader import recheck_budget, recheck_gate, run_recheck, score_from_text
from hayekmas.experiments.campaign_grader_report import recheck_audit, recheck_summary
from hayekmas.experiments.research_campaign import repetition_gate, second_epoch_started
from hayekmas.experiments.campaign_native_trace import native_trace
from hayekmas.experiments.campaign_negotiation import negotiation_trace
from hayekmas.experiments.campaign_participation import participation
from hayekmas.experiments.campaign_diagnostic import interface_prompt, study_budget, run_diagnostic, diagnostic_gate
from hayekmas.adapters.teams.policy import INSTRUCTIONS
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.policy import DemoPolicy
from hayekmas.adapters.teams.env import ExactTaskEnv, Task


def response(content="OK", cost=0.0001):
    r = Mock(status_code=200)
    r.json.return_value = {
        "usage": {"cost": cost, "prompt_tokens": 4, "completion_tokens": 12},
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }
    return r


class FormatTests(unittest.TestCase):
    def test_extracts_final_complete_decision_without_changing_values(self):
        raw = 'Thinking... {"message":"first", "contribution":1}\n{"message":"final", "contribution":0.25}'
        self.assertEqual(json.loads(canonical_decision(raw, "negotiate")), {"message": "final", "contribution": 0.25})
        self.assertEqual(canonical_decision("bad output", "negotiate"), "bad output")
        self.assertEqual(canonical_decision("SCORE: 0.4", "judge"), "SCORE: 0.4")


class ClientTests(unittest.TestCase):
    @patch("hayekmas.experiments.campaign_client.time.sleep")
    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_missing_cost_keeps_reservation_discards_answer_and_retries(self, post, sleep):
        with tempfile.TemporaryDirectory() as d:
            missing = response("unpriced answer must not reach the agent")
            missing.json.return_value["usage"].pop("cost")
            missing.json.return_value["error"] = "receipt omitted sentinel"
            post.side_effect = [missing, response("priced answer")]
            client = CampaignClient("sentinel", d, time.time() + 120)
            self.assertEqual(client.generate("test", max_tokens=100), "priced answer")
            self.assertEqual(client.records[1]["status"], "uncertain_missing_cost")
            self.assertGreater(client.usage()["unconfirmed_usd"], 0)
            self.assertEqual(client.usage()["cost_usd"], 0.0001)
            self.assertNotIn("sentinel", Path(d, "transport_errors.jsonl").read_text())
            self.assertNotIn("unpriced answer", Path(d, "responses.jsonl").read_text())
            restored = CampaignClient("sentinel", d, time.time() + 120)
            self.assertEqual(restored.usage(), client.usage())

    @patch("hayekmas.experiments.campaign_client.time.sleep")
    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_missing_cost_retries_are_bounded_and_fully_reserved(self, post, sleep):
        with tempfile.TemporaryDirectory() as d:
            missing = response("unpriced")
            missing.json.return_value["usage"].pop("cost")
            post.return_value = missing
            client = CampaignClient("sentinel", d, time.time() + 120)
            with self.assertRaisesRegex(CampaignStop, "Retries exhausted: Missing/invalid API cost"):
                client.generate("test", max_tokens=100)
            self.assertEqual(post.call_count, 3)
            self.assertEqual(client.usage()["requests"], 3)
            self.assertTrue(all("cost_usd" not in r for r in client.records.values()))

    @patch("hayekmas.experiments.campaign_client.time.sleep")
    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_partial_text_with_provider_error_is_charged_but_never_used(self, post, sleep):
        with tempfile.TemporaryDirectory() as d:
            broken = response("partial text that must never reach an agent")
            broken.json.return_value["choices"][0]["finish_reason"] = "error"
            post.side_effect = [broken, response("valid response")]
            c = CampaignClient("sentinel", d, time.time() + 60)
            self.assertEqual(c.generate("hello", max_tokens=100), "valid response")
            self.assertEqual(c.usage()["cost_usd"], 0.0002)
            self.assertEqual(c.records[1]["status"], "charged_provider_error")
            self.assertEqual(c.records[2]["status"], "charged")

    @patch("hayekmas.experiments.campaign_client.time.sleep")
    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_empty_retries_charge_both_and_restart_restores_ledger(self, post, sleep):
        with tempfile.TemporaryDirectory() as d:
            post.side_effect = [response(""), response("OK")]
            c = CampaignClient("sentinel", d, time.time() + 60)
            self.assertEqual(c.generate("hello", max_tokens=100), "OK")
            self.assertEqual(c.usage()["cost_usd"], 0.0002)
            restored = CampaignClient("sentinel", d, time.time() + 60)
            self.assertEqual(restored.usage(), c.usage())
            self.assertNotIn("sentinel", Path(d, "api_usage.jsonl").read_text())
            self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 200)
            restored.phase_cap = 0.00021
            with self.assertRaises(CampaignStop):
                restored.generate("hello", max_tokens=100)
            self.assertEqual(post.call_count, 2)

    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_deadline_prevents_call(self, post):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(CampaignStop):
                CampaignClient("key", d, time.time()).generate("hello")
            post.assert_not_called()


class TeamCheckpointTests(unittest.TestCase):
    def test_continuation_matches_uninterrupted_team(self):
        cfg = TeamConfig(num_agents=4, max_steps=2, rounds=2)
        engine = TeamMAS(cfg, DemoPolicy(7))
        engine.run_one_episode(ExactTaskEnv(Task("one", "Sum these integers: [2, 3]", 5), cfg.reward))
        data = json.loads(json.dumps(dump_team(engine)))
        clone = load_team(data, cfg, deepcopy(engine.policy))
        self.assertEqual(json.loads(json.dumps(dump_team(clone))), data)
        task = Task("two", "Sum these integers: [3, 4]", 7)
        a = engine.run_one_episode(ExactTaskEnv(task, cfg.reward))
        b = clone.run_one_episode(ExactTaskEnv(task, cfg.reward))
        self.assertEqual(a, b)


class DiagnosticTests(unittest.TestCase):
    @patch("hayekmas.experiments.campaign_diagnostic.episode_rows")
    def test_diagnostic_requires_both_primary_studies_and_time_to_start(self, episodes):
        full = [{"phase": "training", "epoch": 1}] * 40 + [{"phase": "evaluation", "epoch": 1}] * 19
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            episodes.side_effect = [full, full[:-1]]
            self.assertIn("incomplete", diagnostic_gate(root, 5000, now=0))
            episodes.side_effect = None
            episodes.return_value = full
            self.assertIsNone(diagnostic_gate(root, 5000, now=0))
            self.assertIn("Insufficient", diagnostic_gate(root, 5000, now=3000))
            marker = root / "teams/interface-diagnostic/started.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}")
            self.assertIsNone(diagnostic_gate(root, 5000, now=3000))
            self.assertIn("Insufficient", diagnostic_gate(root, 5000, now=5000))

    def test_clarification_preserves_existing_observation_and_baseline_prompt(self):
        cfg = TeamConfig(discussion_turns=2)
        observation = {"turn": 1, "task": {"problem": "test"}, "objective": "Maximize your own long-term wealth."}
        saved = deepcopy(observation)
        instruction, visible = interface_prompt("discuss", observation, cfg, "baseline")
        self.assertEqual(instruction, INSTRUCTIONS["discuss"])
        self.assertEqual(visible, observation)
        instruction, visible = interface_prompt("discuss", observation, cfg, "clarified")
        self.assertEqual(observation, saved)
        self.assertEqual({k: visible[k] for k in saved}, saved)
        self.assertEqual(visible["discussion_window"]["your_turns_remaining_after_this"], 0)
        self.assertTrue(instruction.startswith(INSTRUCTIONS["discuss"]))
        self.assertEqual(
            interface_prompt("formation", {}, cfg, "baseline"), interface_prompt("formation", {}, cfg, "clarified")
        )

    def test_diagnostic_budget_does_not_replenish_on_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "started.json"
            first = study_budget(path, {"cost_usd": 4.0})
            later = study_budget(path, {"cost_usd": 4.8})
            self.assertEqual(first, later)
            self.assertEqual(later["absolute_cap_usd"], 5.0)
            capped = study_budget(Path(d) / "near-limit.json", {"cost_usd": 24.7})
            self.assertEqual(capped["absolute_cap_usd"], 25.0)

    def test_clarification_preserves_shortened_window_and_legacy_consequence(self):
        window = {"total_turns_per_member": 1, "your_turns_remaining_after_this": 0}
        observation = {"turn": 0, "discussion_window": window}
        instruction, visible = interface_prompt("discuss", observation, TeamConfig(), "clarified")
        self.assertIn("a separate final-answer window opens", instruction)
        self.assertEqual(visible["discussion_window"]["your_turns_remaining_after_this"], 0)
        legacy, _ = interface_prompt(
            "discuss", observation, TeamConfig(finalization_enabled=False), "clarified"
        )
        self.assertIn("the episode ends without a submitted action", legacy)

    @patch("hayekmas.experiments.campaign_diagnostic.diagnostic_gate", return_value=None)
    @patch(
        "hayekmas.experiments.campaign_diagnostic.ResearchTaskEnv",
        side_effect=lambda task, reward, judge: ExactTaskEnv(task, reward),
    )
    @patch("hayekmas.experiments.campaign_diagnostic.load_tasks")
    @patch("hayekmas.experiments.campaign_client.requests.post")
    def test_diagnostic_isolated_artifacts_resume_without_new_calls(self, post, tasks, env, gate):
        def scripted_response(url, **kwargs):
            prompt = json.loads(kwargs["json"]["messages"][-1]["content"])
            phase = next(k for k, instruction in INSTRUCTIONS.items() if prompt["instruction"].startswith(instruction))
            decision = (
                {"reflect": False}
                if phase == "reflect"
                else {"action": "pass"}
                if phase == "formation"
                else {"contribution": 0.1}
                if phase == "contribute"
                else {"message": "test", "candidate": "5", "final": True}
                if phase == "discuss"
                else {"candidate_id": prompt["observation"]["candidates"][0]["id"]}
            )
            return response(json.dumps(decision))

        post.side_effect = scripted_response
        tasks.return_value = [Task(str(i), "Sum these integers: [2, 3]", 5) for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "teams"
            out.mkdir()
            (root / "plan.json").write_text(json.dumps({"deadline_unix": time.time() + 600, "protocol_version": 3}))
            (out / "protocol.json").write_text(
                json.dumps({"team_config": asdict(TeamConfig(num_agents=2, rounds=3, max_steps=2))})
            )
            (out / "status.json").write_text(json.dumps({"training_completed": 40, "eval_completed": {"1": 19}}))
            (out / "training_checkpoint.json").write_text('"primary sentinel"')
            run_diagnostic(root, "sentinel")
            results = list((out / "interface-diagnostic").glob("*/*/result.json"))
            self.assertEqual(len(results), 6)
            self.assertTrue(all(json.loads(p.read_text())["passed"] for p in results))
            self.assertFalse((out / "episodes.jsonl").exists())
            self.assertEqual((out / "training_checkpoint.json").read_text(), '"primary sentinel"')
            calls = post.call_count
            run_diagnostic(root, "sentinel")
            self.assertEqual(post.call_count, calls)
            self.assertEqual(json.loads((out / "status.json").read_text())["training_completed"], 40)
            report = build(root)
            self.assertEqual(report["arms"]["teams"]["episodes"], [])
            self.assertEqual(report["arms"]["teams"]["accounting"]["main_cost_usd"], 0)
            self.assertEqual([v["n"] for v in report["interface_diagnostic"]["variants"]], [3, 3])
            self.assertAlmostEqual(report["interface_diagnostic"]["cost_usd"], calls * 0.0001)


class ReportTests(unittest.TestCase):
    def test_closed_ledgers_detect_same_length_edits_and_stop_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for arm in ("original", "teams"):
                for name in ("api_usage", "requests", "responses"):
                    path = root / arm / f"{name}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"original\n")
                    files[str(path.relative_to(root))] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                    }
            (root / "closed-ledger-manifest.json").write_text(json.dumps({"files": files}))
            (root / "STOP").touch()
            self.assertTrue(closed_ledger_preservation(root)["preserved"])
            path.write_bytes(b"modified\n")
            self.assertEqual(closed_ledger_preservation(root)["changed_or_missing_files"], [str(path.relative_to(root))])
            path.write_bytes(b"original\n")
            (root / "STOP").unlink()
            self.assertFalse(closed_ledger_preservation(root)["preserved"])
            (root / "STOP").touch()
            (root / "RESTART-original").touch()
            self.assertFalse(closed_ledger_preservation(root)["preserved"])

    def test_participation_scopes_candidate_ids_by_auction_step(self):
        events = [
            {"event": "candidate", "step": 0, "id": "candidate-0", "author": "a", "answer": "first", "final": False},
            {"event": "submission", "step": 0, "candidate_id": "candidate-0", "answer": "first", "final": False},
            {"event": "candidate", "step": 1, "id": "candidate-0", "author": "b", "answer": "second", "final": True},
            {"event": "submission", "step": 1, "candidate_id": "candidate-0", "answer": "second", "final": True},
        ]
        counts = participation(events, ["a", "b"])
        self.assertEqual(counts["a"]["selected_intermediate"], 1)
        self.assertEqual(counts["a"]["selected_final"], 0)
        self.assertEqual(counts["b"]["selected_final"], 1)
        events[-1]["answer"] = "mismatched answer"
        with self.assertRaises(ValueError):
            participation(events, ["a", "b"])

    def test_primary_preservation_detects_changes_missing_files_and_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "original/epoch-1.json"
            second = root / "teams/evaluation/epoch-1/01-task/result.json"
            for path in (first, second):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"saved":true}')
            protected = {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)
            }
            manifest = root / "primary-results-manifest.json"
            manifest.write_text(json.dumps({"files": protected}))
            self.assertTrue(primary_preservation(root)["preserved"])
            first.write_text('{"saved":false}')
            self.assertEqual(primary_preservation(root)["changed_or_missing_files"], ["original/epoch-1.json"])
            first.write_text('{"saved":true}')
            second.unlink()
            self.assertFalse(primary_preservation(root)["preserved"])
            self.assertIn(str(second.relative_to(root)), primary_preservation(root)["changed_or_missing_files"])
            second.write_text('{"saved":true}')
            manifest.write_text(json.dumps({"files": {"original/epoch-1.json": protected["original/epoch-1.json"]}}))
            self.assertEqual(primary_preservation(root)["missing_manifest_entries"], [str(second.relative_to(root))])
            self.assertFalse(primary_preservation(root)["preserved"])

    def test_negotiation_trace_reconciles_unequal_pledges_and_detects_bad_transfers(self):
        events = [{"event": "initialized", "agents": [{"name": "a"}, {"name": "b"}], "config": {"bid_cost_rate": 1}}]
        for step, first, final, opening, credits, reward, wealth in [
            (0, {"a": 0, "b": 3}, {"a": 2, "b": 3}, {"a": 20, "b": 20}, {}, 0, {"a": 18, "b": 17}),
            (1, {"a": 0, "b": 2}, {"a": 0, "b": 2}, {"a": 18, "b": 17}, {"a": 1, "b": 1}, 6, {"a": 22, "b": 19}),
        ]:
            for turn, pledges in enumerate((first, final)):
                events.extend(
                    {"event": "pledge", "step": step, "group": "team", "agent": name, "turn": turn, "amount": value}
                    for name, value in pledges.items()
                )
            paid = sum(final.values())
            events.append(
                {
                    "event": "auction",
                    "step": step,
                    "winner": "team",
                    "members": ["a", "b"],
                    "bids": {"team": paid},
                    "contributions": final,
                    "opening_wealth": opening,
                    "paid": paid,
                    "credits": credits,
                    "burned": 0 if credits else paid,
                }
            )
            events.append(
                {
                    "event": "settlement",
                    "step": step,
                    "members": ["a", "b"],
                    "reward": reward,
                    "share": reward / 2,
                    "wealth": wealth,
                }
            )
        result = negotiation_trace(events)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["counts"]["revised_personal_pledges"], 1)
        self.assertEqual(result["counts"]["winning_teams_with_zero_money_contributor"], 1)
        self.assertEqual(result["counts"]["wealth_values_checked"], 4)
        broken = deepcopy(events)
        broken[-2]["credits"] = {"a": 2, "b": 0}
        errors = negotiation_trace(broken)["errors"]
        self.assertTrue(any("previous winning members" in e["requirement"] for e in errors))
        self.assertTrue(any("wealth reconciles" in e["requirement"] for e in errors))

    def test_epoch_start_gate_allows_resume_after_first_attempt_or_committed_progress(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertFalse(second_epoch_started(root, {"training_completed": 40}))
            self.assertTrue(second_epoch_started(root, {"training_completed": 41}))
            (root / "training/epoch-2/01-task").mkdir(parents=True)
            self.assertFalse(second_epoch_started(root, {"training_completed": 40}))
            (root / "training/epoch-2/01-task/attempt-1").mkdir()
            self.assertTrue(second_epoch_started(root, {"training_completed": 40}))

    def test_native_trace_preserves_replay_order_and_rounded_credit_checks(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trajectory.log"
            path.write_text("""┌─ Step 2/10 ─┐
💰 Payment: Answer-1234 → Planner-1234 ($0.10)
🔍 CREDIT CHECK:
   Answer-1234 1: $-0.00 💀 BANKRUPT
   Planner-1234 2: $0.50 ✓ solvent
   💀 REMOVING Answer-1234 (id=1, wealth=$-0.00)
   🔄 Restoring surviving agents' wealth/capability to snapshot (preserving bid/status)
🔁 REPLAY trial 2/2
┌─ Step 1/10 ─┐
🏆 WINNER: Planner-1234 (id=2, bid=0.10)
🧬 SPAWNING: Planner-4321 3 via GOOD BIRTH (parent: Planner-1234)
Action: ResearchAction(STEP, '🧬 SPAWNING: quoted data is not a lifecycle event')
""")
            trace = native_trace(path)
            self.assertEqual(trace["counts"]["removed"], 1)
            self.assertEqual(trace["counts"]["birth"], 1)
            self.assertEqual(trace["events"][-1]["trial"], 2)
            self.assertEqual(trace["events"][-1]["step"], 1)
            self.assertIsNone(next(e for e in trace["events"] if e["kind"] == "replay")["step"])
            self.assertTrue(trace["credit_checks"][0]["agents"][0]["bankrupt"])
            self.assertEqual(trace["credit_checks"][0]["agents"][1]["rounded_wealth"], 0.5)
            self.assertEqual(trace["events"][1]["line"], 2)

    def test_repeat_gates_require_both_primary_arms_and_two_hours_for_each_new_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIn("incomplete", repetition_gate(root, "teams", 1, 10000, now=0))
            for arm in ("original", "teams"):
                folder = root / arm
                for phase, n in (("training", 40), ("evaluation", 19)):
                    for index in range(n):
                        task = folder / phase / "epoch-1" / str(index)
                        task.mkdir(parents=True)
                        (task / "result.json").write_text("{}")
                (folder / "epoch-1.json").write_text("{}")
            self.assertIsNone(repetition_gate(root, "teams", 1, 10000, now=0))
            self.assertIn("Insufficient", repetition_gate(root, "teams", 1, 10000, now=3000))
            marker = root / "teams/replications/repeat-1/started.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}")
            self.assertIsNone(repetition_gate(root, "teams", 1, 10000, now=3000))
            self.assertIn("Insufficient", repetition_gate(root, "original", 1, 10000, now=3000))
            self.assertIn("Insufficient", repetition_gate(root, "teams", 2, 10000, now=3000))
            self.assertIn("Insufficient", repetition_gate(root, "teams", 1, 10000, now=10000))

    def test_repeat_records_never_replace_or_enter_primary_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for phase, repeat, location in [
                ("evaluation", None, "evaluation/epoch-1"),
                ("replication", 1, "replications/repeat-1"),
                ("replication", 2, "replications/repeat-2"),
            ]:
                folder = root / location / "01-task"
                folder.mkdir(parents=True)
                row = {"phase": phase, "epoch": 1, "index": 1, "task_id": "task", "score": 0.4}
                if repeat is not None:
                    row["repeat"] = repeat
                (folder / "result.json").write_text(json.dumps(row))
            self.assertEqual(len(episode_rows(root)), 1)
            all_rows = episode_rows(root, include_replications=True)
            self.assertEqual([r.get("repeat") for r in all_rows], [None, 1, 2])

    def test_repeat_scores_and_costs_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "plan.json").write_text(json.dumps({"protocol_version": 3}))
            for arm in ("original", "teams"):
                folder = root / arm
                ledger = []
                for repeat in (None, 1, 2):
                    phase = "replication" if repeat else "evaluation"
                    task = folder / (f"replications/repeat-{repeat}" if repeat else "evaluation/epoch-1") / "01-task"
                    task.mkdir(parents=True)
                    row = {
                        "arm": arm,
                        "phase": phase,
                        "epoch": 1,
                        "index": 1,
                        "task_id": "task",
                        "subject": "physics",
                        "score": (repeat or 0) / 2,
                        "passed": bool(repeat),
                        "has_final_answer": True,
                        "protocol_version": 3,
                        "artifact": str(task.relative_to(root)),
                    }
                    record = {
                        "request": (repeat or 0) + 1,
                        "phase": phase,
                        "epoch": 1,
                        "task_id": "task",
                        "protocol_version": 3,
                        "status": "charged",
                        "cost_usd": (repeat or 0) + 1,
                    }
                    if repeat is not None:
                        row["repeat"] = record["repeat"] = repeat
                    ledger.append(record)
                    (task / "result.json").write_text(json.dumps(row))
                ledger.append(
                    {
                        "request": 4,
                        "phase": "evaluation",
                        "epoch": 1,
                        "task_id": "task",
                        "protocol_version": 3,
                        "status": "reserved",
                        "reserved_usd": 0.2,
                    }
                )
                (folder / "api_usage.jsonl").write_text("\n".join(map(json.dumps, ledger)))
            result = build(root)
            self.assertEqual(len(result["paired"]), 1)
            self.assertEqual(len(result["repeated"]), 2)
            self.assertEqual(result["paired"][0]["tasks"][0]["original_cost"], 1)
            self.assertEqual(result["paired"][0]["tasks"][0]["original_reserved"], 0.2)
            self.assertEqual([r["tasks"][0]["original_reserved"] for r in result["repeated"]], [0, 0])
            self.assertEqual([r["tasks"][0]["original_cost"] for r in result["repeated"]], [2, 3])
            self.assertEqual([r["mean_score"]["teams"] for r in result["repeated"]], [0.5, 1])
            self.assertEqual([r["n"] for r in result["arms"]["original"]["summary"]], [1, 1, 1])

    def test_native_missing_transfer_telemetry_is_not_reported_as_zero(self):
        result = diagnostics({"arm": "original", "has_final_answer": False}, [])
        self.assertIsNone(result["total_bid_paid"])
        self.assertIsNone(result["total_bid_transferred"])
        self.assertIsNone(result["messages"])

    def test_live_membership_tracks_inviter_and_dissolved_singleton(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "teams"
            attempt = folder / "training/epoch-1/01-task/attempt-1"
            attempt.mkdir(parents=True)
            state = {"status": "running", "phase": "training", "epoch": 1, "task_number": 1, "task_id": "task"}
            events = [
                {
                    "event": "initialized",
                    "agents": [{"name": "agent-0", "team": None}, {"name": "agent-1", "team": None}],
                },
                {"event": "joined", "agent": "agent-1", "inviter": "agent-0", "team": "team-1"},
            ]
            path = attempt / "events.jsonl"
            path.write_text("\n".join(map(json.dumps, events)))
            live = live_team_episode(root, folder, state, {})
            self.assertEqual([a["team"] for a in live["population_after"]], ["team-1", "team-1"])
            events.extend(
                [
                    {"event": "left", "agent": "agent-1", "team": "team-1"},
                    {"event": "team_dissolved", "team": "team-1", "remaining": "agent-0"},
                ]
            )
            path.write_text("\n".join(map(json.dumps, events)))
            live = live_team_episode(root, folder, state, {})
            self.assertEqual([a["team"] for a in live["population_after"]], [None, None])

    def test_paired_bootstrap_preserves_constant_advantage(self):
        tasks = [{"teams": 0.75, "original": 0.5, "teams_pass": True, "original_pass": True}] * 19
        self.assertIsNone(paired_intervals(tasks[:18]))
        result = paired_intervals(tasks, samples=500)
        self.assertEqual(result["score_difference"], [0.25, 0.25])
        self.assertEqual(result["pass_rate_difference"], [0.0, 0.0])

    def test_replayed_attempt_audit_excludes_abandoned_requests(self):
        ledger = {
            1: {"request": 1, "time": 1, "finish_reason": "error"},
            2: {"request": 2, "time": 2},
            3: {"request": 3, "time": 3},
        }
        explicit = {"request_start": 2, "request_end": 3, "usage": {"requests": 2}, "time": 4}
        legacy = {"usage": {"requests": 2}, "time": 4}
        self.assertEqual([r["request"] for r in request_range(explicit, ledger)], [2, 3])
        self.assertEqual(request_range(legacy, ledger), request_range(explicit, ledger))
        diagnostic = {key: value for key, value in explicit.items() if key != "usage"}
        self.assertEqual(request_range(diagnostic, ledger), request_range(explicit, ledger))

    def test_commit_survives_missing_index_append_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "training" / "epoch-1" / "01-task"
            folder.mkdir(parents=True)
            row = {"phase": "training", "epoch": 1, "index": 1, "task_id": "task", "score": 0.4}
            (folder / "result.json").write_text(json.dumps(row))
            self.assertEqual(episode_rows(root), [row])
            stale = {**row, "score": 0.1}
            (root / "episodes.jsonl").write_text(json.dumps(stale) + "\n" + json.dumps(stale) + "\n{partial")
            self.assertEqual(episode_rows(root), [row])

    def test_episode_invalid_counts_are_not_cumulative_and_ids_survive_renaming(self):
        row = {
            "has_final_answer": False,
            "invalid_actions": 97,
            "population_before": [{"id": 1, "name": "old"}, {"id": 2, "name": "removed"}],
            "population_after": [{"id": 1, "name": "new"}, {"id": 3, "name": "born"}],
        }
        events = [
            {"event": "invalid_action", "reason": "invalid vote"},
            {"event": "no_submission", "reason": "no valid votes"},
        ]
        result = diagnostics(row, events)
        self.assertEqual(result["invalid_actions_this_episode"], 1)
        self.assertEqual(result["outcome"], "no valid votes")
        self.assertEqual((result["births"], result["removed_agents"]), (1, 1))


class GraderRecheckTests(unittest.TestCase):
    def test_grading_means_require_valid_grades_and_missing_answers_stay_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [{"index": i + 1, "task_id": f"task-{i+1}", "has_final_answer": i == 0,
                        "primary_score": .5 if i == 0 else 0} for i in range(19)]
            (root / "grader-recheck-design.json").write_text(json.dumps({"arms": {"original": sources}}))
            study = root / "original/grader-recheck"
            for replicate, grade in [(1, {"score": .4, "valid": True}), (2, {"score": None, "valid": False})]:
                path = study / f"replicate-{replicate}/01-task-1.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(grade))
            (study / "started.json").write_text("{}")
            records = {1: {"study": "grader_recheck", "cost_usd": .02}, 2: {"study": "grader_recheck", "reserved_usd": .03}}
            summary = recheck_summary(root, "original", records)
            first, invalid, missing = summary["replicates"]
            self.assertTrue(first["complete"])
            self.assertAlmostEqual(first["mean_score"], .4/19)
            self.assertEqual(first["expected_grades"], 1)
            self.assertFalse(invalid["complete"])
            self.assertFalse(missing["complete"])
            self.assertIsNone(invalid["mean_score"])
            self.assertIsNone(missing["mean_score"])
            self.assertEqual(summary["completed_grades"], 2)
            self.assertEqual(summary["expected_grades"], 3)
            self.assertEqual((summary["billed_usd"], summary["reserved_usd"]), (.02, .03))

    def test_invalid_grades_are_missing_and_budget_cannot_replenish(self):
        self.assertEqual(score_from_text("Reason\nSCORE: 0.45\n"), 0.45)
        for text in ("No score", "SCORE: nan", "SCORE: 2", "SCORE: 0.5\nSCORE: 0.7"):
            self.assertIsNone(score_from_text(text))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "started.json"
            first = recheck_budget(path, {"cost_usd": 3.0}, 25)
            second = recheck_budget(path, {"cost_usd": 4.0}, 25)
            self.assertEqual(first, second)
            self.assertEqual(second["absolute_cap_usd"], 3.5)
            self.assertIn("finish first", recheck_gate(root, "original", 5000, now=0))
            for arm in ("original", "teams"):
                for repeat in (1, 2):
                    for index in range(19):
                        result = root / arm / f"replications/repeat-{repeat}/{index}/result.json"
                        result.parent.mkdir(parents=True, exist_ok=True)
                        result.write_text("{}")
            self.assertIn("diagnostic must finish", recheck_gate(root, "original", 5000, now=0))
            diagnostic = root / "teams/interface-diagnostic/status.json"
            diagnostic.parent.mkdir()
            diagnostic.write_text('{"status":"complete"}')
            self.assertIsNone(recheck_gate(root, "original", 5000, now=0))
            self.assertIn("Insufficient", recheck_gate(root, "original", 5000, now=4000))

    def test_saved_prompts_are_replayed_without_answers_changing_or_resume_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "original"
            out.mkdir()
            (root / "plan.json").write_text(json.dumps({"deadline_unix": time.time()+3600, "arm_max_usd": 25, "protocol_version": 3}))
            (out / "status.json").write_text(json.dumps({"status": "complete", "training_completed": 40}))
            result, answer = out / "primary.json", out / "answer.md"
            result.write_text('{"score":0.5}')
            answer.write_text("Untouched primary answer")
            protected = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (result, answer)}
            source = {"index": 1, "task_id": "task-1", "has_final_answer": True, "primary_score": .5,
                      "result": "original/primary.json", "result_sha256": protected["original/primary.json"],
                      "answer": "original/answer.md", "answer_sha256": protected["original/answer.md"],
                      "source_request_number": 100,
                      "judge_request": {"prompt": "Frozen original judge prompt", "system": "Frozen system",
                                        "reasoning_effort": "medium", "max_tokens": 8192}}
            missing = {**source, "index": 2, "task_id": "task-2", "has_final_answer": False, "primary_score": 0}
            (out / "requests.jsonl").write_text(json.dumps({"request": 100, **source["judge_request"]})+'\n')
            (root / "grader-recheck-design.json").write_text(json.dumps({"arms": {"original": [source, missing]}}))
            (root / "primary-results-manifest.json").write_text(json.dumps({"files": protected}))
            responses = [response("SCORE: 0.4"), response("Invalid grade syntax"), response("SCORE: 0.6")]
            with patch("hayekmas.experiments.campaign_grader.recheck_gate", return_value=None), patch("hayekmas.experiments.campaign_client.requests.post", side_effect=responses) as post:
                run_recheck(root, "original", "test-key")
                self.assertEqual(post.call_count, 3)
                for call in post.call_args_list:
                    payload = call.kwargs["json"]
                    self.assertEqual(payload["messages"], [{"role": "system", "content": "Frozen system"}, {"role": "user", "content": "Frozen original judge prompt"}])
                    self.assertEqual(payload["max_tokens"], 8192)
                run_recheck(root, "original", "test-key")
                self.assertEqual(post.call_count, 3)
            grades = sorted((out / "grader-recheck").glob("replicate-*/*.json"))
            self.assertEqual(len(grades), 3)
            invalid = json.loads(grades[1].read_text())
            self.assertIsNone(invalid["score"])
            self.assertFalse(invalid["valid"])
            self.assertEqual(answer.read_text(), "Untouched primary answer")
            self.assertEqual({name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in protected}, protected)
            def records(name):
                return {row["request"]: row for line in (out / name).read_text().splitlines() if (row := json.loads(line))}
            ledger, requests, responses = (records(name) for name in ("api_usage.jsonl", "requests.jsonl", "responses.jsonl"))
            def checked():
                return {name: (valid, evidence) for name, valid, evidence in recheck_audit(root, "original", ledger, requests, responses)}
            self.assertTrue(checked()["original: grading check call scope and prompt fidelity"][0])
            self.assertTrue(checked()["original: grading check outputs trace to charged responses"][0])
            ledger[1]["grading_replicate"] = 2
            self.assertFalse(checked()["original: grading check outputs trace to charged responses"][0])
            ledger[1]["grading_replicate"] = 1
            requests[1]["prompt"] = "changed prompt"
            self.assertEqual(checked()["original: grading check call scope and prompt fidelity"][1]["prompt_errors"], [1])
            altered = json.loads(grades[0].read_text())
            altered["score"] = .9
            grades[0].write_text(json.dumps(altered))
            self.assertFalse(checked()["original: grading check outputs trace to charged responses"][0])


class FindingsTests(unittest.TestCase):
    def test_primary_repeat_cost_and_completion_boundaries(self):
        plan = {
            "test_tasks": 19, "primary_train_tasks": 40, "deadline_unix": 100,
            "deadline_utc": "1970-01-01T00:01:40Z", "model": "test-model",
            "arm_max_usd": 25, "max_usd": 50, "comparison_limits": ["system comparison"],
        }
        primary = [
            {"phase": "evaluation", "epoch": 1, "score": 0.25, "passed": False,
             "has_final_answer": True, "full_episode_cost": 0.1,
             "full_episode_unconfirmed_usd": 0, "diagnostics": {"outcome": "graded"}}
            for _ in range(19)
        ]
        partial_repeat = [
            {**primary[0], "phase": "replication", "repeat": 1, "score": 1.0, "passed": True}
        ]
        pair = {"epoch": 1, "repeat": None, "complete": True, "paired_score_difference": 0,
                "wins_ties_losses": {"original_higher": 0, "teams_higher": 0, "tied": 19}}
        data = {
            "plan": plan, "updated_at": 90,
            "arms": {arm: {"episodes": primary + partial_repeat} for arm in ("original", "teams")},
            "paired": [pair], "repeated": [{**pair, "repeat": 1}],
        }
        audit = {"checked_at": 91, "failures": 0, "pending": 1, "arms": {
            arm: {"billed_usd": 3.0, "reserved_usd": 0.25} for arm in ("original", "teams")
        }}
        result = collect_findings(data, audit, {}, {}, now=95)
        self.assertEqual(result["research_window"], "in progress")
        self.assertTrue(result["runs"][0]["complete"])
        self.assertEqual(result["runs"][0]["arms"]["teams"]["mean_score"], 0.25)
        self.assertIsNone(result["runs"][1]["paired_comparison"])
        self.assertFalse(result["runs"][1]["complete"])
        self.assertIsNone(result["runs"][2]["arms"]["original"]["mean_score"])
        self.assertEqual(result["costs"]["teams"], {"billed_usd": 3, "reserved_usd": 0.25})
        text = render_findings(result)
        self.assertIn("REPEAT 1 — INCOMPLETE", text)
        self.assertIn("$3.000000 billed + $0.250000 unconfirmed", text)
        self.assertIn("no full-run paired conclusion", text)
        after = collect_findings(data, audit, {}, {}, now=101)
        self.assertEqual(after["research_window"], "ended")
        self.assertFalse(after["runs"][1]["complete"])


if __name__ == "__main__":
    unittest.main()
