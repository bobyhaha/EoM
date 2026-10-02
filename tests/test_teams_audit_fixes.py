"""Regression checks for independent tests, repair supervision and financial stops."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ExactTaskEnv, ResearchTaskEnv, Task
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.openrouter import SpendLimitExceeded
from hayekmas.adapters.teams.policy import DemoPolicy
from hayekmas.adapters.teams.runtime import run


class AuditFixTests(unittest.TestCase):
    def test_runtime_test_tasks_reset_full_agent_state_and_keep_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            run({'backend': 'demo', 'split': 'test', 'plots': False,
                 'teams': {'interaction_protocol': 'rounds', 'num_agents': 4,
                           'rounds': 2, 'max_steps': 1, 'evolution_enabled': True,
                           'birth_interval': 1, 'rent': 100, 'rent_interval': 1}}, out=out, plots=False)
            events = [json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
            starts = [e for e in events if e['event'] == 'decision_round_started']
            resets = [e for e in events if e['event'] == 'evaluation_reset']
            completed = [e for e in events if e['event'] == 'round_complete']
            self.assertEqual(starts[0]['wealth'], starts[1]['wealth'])
            self.assertEqual(starts[0]['membership'], starts[1]['membership'])
            self.assertEqual(resets[0]['agents'], resets[1]['agents'])
            self.assertNotEqual(completed[0]['metrics']['wealth'], starts[1]['wealth'])
            self.assertEqual([e['round'] for e in completed], [0, 1])
            self.assertFalse(any(e['event'] in {'agent_born', 'agent_removed', 'population_rent', 'reflection_paid'} for e in events))
            usage = json.loads((out/'usage.json').read_text())
            self.assertEqual(usage['decision_calls'], sum(e['event'] == 'decision_request' for e in events))
            replay = json.loads((out/'replay.json').read_text())
            self.assertEqual(len(replay['rounds']), 2)
            self.assertTrue(all(r['events'][0]['event'] == 'evaluation_reset' for r in replay['rounds']))

    def test_repair_receives_reference_but_solver_and_tests_do_not(self):
        reference = 'PRIVATE_TRAINING_REFERENCE_738194'
        task = Task('audit', 'Compute 1 + 1', reference)
        for kind in ('exact', 'research'):
            with self.subTest(kind=kind):
                env = (ExactTaskEnv(task, 12) if kind == 'exact' else
                       ResearchTaskEnv(task, 12, lambda _: 'SCORE: 0\nREASON: wrong'))
                cfg = TeamConfig(interaction_protocol='rounds', num_agents=2, max_steps=1,
                                 evolution_enabled=True, birth_interval=1,
                                 periodical_good_p=0, context_tokens=32768)
                engine = TeamMAS(cfg, DemoPolicy(7))
                self.assertEqual(env.get_correct_answer(), reference)
                engine.run_one_episode(env, reflection=False)
                requests = [e for e in engine.events if e['event'] == 'decision_request']
                births = [e for e in requests if e['phase'] == 'birth_bad']
                self.assertTrue(births)
                self.assertTrue(all(reference in json.dumps(e['observation']) for e in births))
                self.assertTrue(all(reference not in json.dumps(e['observation']) for e in requests if e['phase'] != 'birth_bad'))
                testing = TeamMAS(cfg, DemoPolicy(7))
                testing.run_one_episode(env, training=False)
                self.assertFalse(any(reference in json.dumps(e['observation']) for e in testing.events if e['event'] == 'decision_request'))

    def test_budget_and_uncertain_provider_errors_interrupt_runtime_births(self):
        for failure in ('limit', 'blocked', 'wrapped_blocked'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                class StopPolicy(DemoPolicy):
                    def respond(self, phase, *args):
                        if phase.startswith('birth_'):
                            if failure == 'limit':
                                raise SpendLimitExceeded('synthetic financial stop')
                            native = SimpleNamespace(blocked=True)
                            self.client = native if failure == 'blocked' else SimpleNamespace(native=native)
                            raise RuntimeError('synthetic uncertain provider cost')
                        return super().respond(phase, *args)
                out = Path(tmp)/'run'
                with patch('hayekmas.adapters.teams.runtime.DemoPolicy', StopPolicy):
                    with self.assertRaises(SpendLimitExceeded if failure == 'limit' else RuntimeError):
                        run({'backend': 'demo', 'teams': {'interaction_protocol': 'rounds',
                             'num_agents': 2, 'rounds': 1, 'max_steps': 1,
                             'evolution_enabled': True, 'birth_interval': 1}}, out=out, plots=False)
                manifest = json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'interrupted')
                self.assertTrue((out/'interrupted_state.json').exists())
                events = [json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
                self.assertFalse(any(e['event'] in {'agent_born', 'birth_failed', 'round_complete'} for e in events))
