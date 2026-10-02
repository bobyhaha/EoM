import json
from pathlib import Path
import tempfile
import unittest

from hayekmas.experiments.population_study import KS, estimate, prepare, prior_spend


class PopulationStudyTests(unittest.TestCase):
    def test_estimates_increase_with_population_and_context(self):
        for arm in ("original", "teams"):
            values = [estimate(k)[arm]["usd"] for k in KS]
            self.assertEqual(values, sorted(values))
            self.assertGreater(values[-1], values[0])
            for k in KS:
                low, mid, high = [estimate(k, s)[arm]["usd"] for s in ("low", "central", "high")]
                self.assertLess(low, mid)
                self.assertLess(mid, high)

    def test_shorter_schedule_reduces_cost_without_changing_dataset(self):
        self.assertLess(estimate(100, steps=4)["teams"]["usd"], estimate(100)["teams"]["usd"])
        self.assertEqual(estimate(100)["teams"]["phases"]["judge"]["calls"], 59)

    def test_copied_receipts_are_not_double_counted_and_unknown_cost_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ("original", "archive/original"):
                out = root / "runs/eom-vs-teams-12h" / arm
                out.mkdir(parents=True)
                rows = [{"request": 1, "generation_id": "same", "cost_usd": 2},
                        {"request": 2, "time": 123, "kind": "solve", "reserved_usd": .1}]
                (out / "api_usage.jsonl").write_text('\n'.join(map(json.dumps, rows)))
            result = prior_spend(root)
            self.assertEqual(result["billed_usd"], 2)
            self.assertEqual(result["reserved_usd"], .1)
            self.assertAlmostEqual(result["remaining_usd"], 47.9)

    def test_preparation_has_eight_unlaunched_cells_and_disjoint_complete_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "study"
            plan = prepare(root)
            self.assertFalse(plan["execution_authorized"])
            self.assertEqual(len(plan["cells"]), 8)
            self.assertEqual([len(plan["tasks"][s]) for s in ("train", "test")], [40, 19])
            self.assertTrue((root / "index.html").exists())
            for path in (root / "cells").glob("*.json"):
                cell = json.loads(path.read_text())
                self.assertEqual(cell["budget"], {"execution_authorized": False, "allocated_usd": 0})
            with self.assertRaises(FileExistsError):
                prepare(root)
