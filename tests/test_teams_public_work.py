import json
import unittest

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ExactTaskEnv, Task
from hayekmas.adapters.teams.round_protocol import select_action
from tests.test_teams_round_protocol import engine, RoundPolicy


class WorkPolicy(RoundPolicy):
    def respond(self, phase, agent, observation, max_tokens):
        if phase == 'round_commit':
            return json.dumps({'act': True, 'contribution': 0 if self.no_bids else 1, 'reason': 'test'})
        if phase == 'shared_work':
            self.observations.append((phase, agent.name, dict(observation)))
            return json.dumps({'message': 'extend', 'candidate': (observation['shared_draft'] or '') + agent.name})
        return super().respond(phase, agent, observation, max_tokens)


def execute(mas):
    return mas.run_one_episode(ExactTaskEnv(Task('t', '3+4?', 7), 12), formation=False, reflection=False)


class PublicWorkTests(unittest.TestCase):
    def test_closing_uses_last_public_team_without_payment_or_extra_credit(self):
        mas = engine(WorkPolicy(), round_schedule='compact', terminal_policy='public_work', discussion_turns=1)
        result = execute(mas)
        self.assertEqual(result['score'], 1)
        self.assertEqual(result['contributing_rounds'], 2)
        self.assertEqual(len([e for e in mas.events if e['event'] == 'auction']), 2)
        closure = next(e for e in mas.events if e['event'] == 'public_work_finalization')
        self.assertEqual(closure['members'], result['credit_path'][-1]['members'])
        self.assertEqual(sum(result['steps'][-1]['contributions'].values()), 0)
        self.assertTrue(result['steps'][-1]['closing_public_work'])
        self.assertEqual(result['reward_issued'], 24)
        mas.assert_accounting()

    def test_no_public_work_does_not_activate_unfunded_closure(self):
        mas = engine(WorkPolicy(no_bids=True), round_schedule='compact', terminal_policy='public_work')
        result = execute(mas)
        self.assertEqual(result['score'], 0)
        self.assertFalse(any(e['event'] == 'public_work_finalization' for e in mas.events))

    def test_finalizers_can_abstain_without_fabricating_answer_or_reward(self):
        mas = engine(WorkPolicy(abstain=True), round_schedule='compact', terminal_policy='public_work')
        result = execute(mas)
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['reward_issued'], 0)
        self.assertFalse(result['steps'][-1]['accepted'])

    def test_shared_draft_is_updated_sequentially_and_only_latest_is_selected(self):
        mas = engine(WorkPolicy(), shared_draft_passes=2)
        mas.step = 0
        action = select_action(mas, {'id': 't', 'problem': 'test'}, 'group', mas.agents[:2], False)
        seen = [o for p, _, o in mas.policy.observations if p == 'shared_work']
        self.assertEqual(len(seen), 4)
        self.assertIsNone(seen[0]['shared_draft'])
        self.assertTrue(all(o['shared_draft'] for o in seen[1:]))
        self.assertEqual(len(action.answer), 4 * len('agent-0'))
        votes = [o for p, _, o in mas.policy.observations if p == 'vote']
        self.assertEqual(len(votes[0]['candidates']), 1)
        self.assertEqual(votes[0]['candidates'][0]['answer'], action.answer)

    def test_closure_does_not_read_current_team_private_scratchpad(self):
        mas = engine(WorkPolicy(), terminal_policy='public_work')
        mas.step = 2
        mas.scratchpads['old-team'] = [{'author': 'new-member', 'text': 'PRIVATE NEW MEMBERS'}]
        select_action(mas, {'id': 't', 'problem': 'test'}, 'old-team', mas.agents[:2], True, closing_public_work=True)
        seen = [o for p, _, o in mas.policy.observations if p == 'finalize']
        self.assertTrue(seen)
        self.assertTrue(all('PRIVATE NEW MEMBERS' not in json.dumps(o) for o in seen))

    def test_revision_budget_preserves_finalization(self):
        mas = engine(WorkPolicy(), shared_draft_passes=2, max_calls=7)
        mas.step = 2
        action = select_action(mas, {'id': 't', 'problem': 'test'}, 'group', mas.agents[:2], True)
        self.assertTrue(action.final)
        self.assertFalse(any(p == 'shared_work' for p, _, _ in mas.policy.observations))

    def test_combined_options_keep_final_candidate_ids_unique(self):
        mas = engine(WorkPolicy(), round_schedule='compact', terminal_policy='public_work', shared_draft_passes=2)
        result = execute(mas)
        self.assertEqual(result['score'], 1)
        self.assertEqual(result['contributing_rounds'], 2)
        for step in range(3):
            ids = [e['id'] for e in mas.events if e['event'] == 'candidate' and e['step'] == step]
            self.assertEqual(len(ids), len(set(ids)))
        mas.assert_accounting()

    def test_null_revision_retains_previous_draft(self):
        mas = engine(WorkPolicy(), shared_draft_passes=1)
        original = mas.policy.respond
        count = 0
        def respond(phase, agent, observation, cap):
            nonlocal count
            if phase == 'shared_work':
                count += 1
                return json.dumps({'message': 'keep or extend', 'candidate': 'saved draft' if count == 1 else None})
            return original(phase, agent, observation, cap)
        mas.policy.respond = respond
        mas.step = 0
        action = select_action(mas, {'id': 't', 'problem': 'test'}, 'group', mas.agents[:2], False)
        self.assertEqual(action.answer, 'saved draft')

    def test_options_require_round_protocol_and_nonnegative_passes(self):
        for kwargs in ({'terminal_policy': 'public_work'}, {'shared_draft_passes': 1},
                       {'interaction_protocol': 'rounds', 'shared_draft_passes': -1}):
            with self.assertRaises(ValueError):
                TeamConfig(**kwargs)
