import json
import unittest
from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.experiments.team_checkpoint import dump_team, load_team


class Ballots:
    label = 'test'

    def __init__(self, ballots):
        self.ballots = ballots
        self.observations = []

    def respond(self, phase, agent, observation, max_tokens):
        self.observations.append(observation)
        return json.dumps(self.ballots.get(agent.name, {'act': False, 'authorize_base_bid': False, 'reason': 'abstain'}))


class FixedBidTests(unittest.TestCase):
    def engine(self, ballots, **kw):
        cfg = TeamConfig(interaction_protocol='rounds', token_profile='solve_first', num_agents=4,
                         team_bid_rule='fixed', **kw)
        e = TeamMAS(cfg, Ballots(ballots))
        e.team_manager.create_team(e.agents[:3], 0)
        e.step = 0
        return e

    def test_consent_cost_transfer_and_no_charge_for_dissenter(self):
        yes = {'act': True, 'authorize_base_bid': True, 'reason': 'fund useful work'}
        e = self.engine({'agent-0': yes, 'agent-1': yes})
        winner, members, contributions, _ = e.auction({'id': 'test', 'problem': 'x'})
        self.assertIsNotNone(winner)
        self.assertEqual(contributions, {'agent-0': .05, 'agent-1': .05, 'agent-2': 0, 'agent-3': 0})
        self.assertEqual(e.lookup('agent-2').wealth, 20)
        self.assertAlmostEqual(e.bid_burn_total, .1)
        e.assert_accounting()
        self.assertTrue(all(o['bidding_rule'] == 'fixed' and o['team_base_bid'] == .1 for o in e.policy.observations))
        e.previous_members = members
        e.previous_winner = winner
        e.auction({'id': 'test', 'problem': 'x'})
        self.assertAlmostEqual(e.bid_transfer_total, .1)
        e.assert_accounting()

    def test_missing_consent_or_insufficient_funds_cannot_fund(self):
        for ballot in ({'act': True, 'contribution': 0, 'reason': 'not explicit consent'},
                       {'act': True, 'authorize_base_bid': 'true', 'reason': 'wrong type'},
                       {'act': True, 'authorize_base_bid': False, 'reason': 'no'},
                       {'act': True, 'authorize_base_bid': True, 'reason': 'yes'}):
            e = self.engine({f'agent-{i}': ballot for i in range(4)})
            if ballot.get('authorize_base_bid') is True:
                for a in e.agents:
                    a.wealth = .01
                e.initial_total = .04
            winner, _, _, _ = e.auction({'id': 'test', 'problem': 'x'})
            self.assertIsNone(winner)
            self.assertEqual(e.bid_paid_total, 0)
            e.assert_accounting()

    def test_objective_checkpoint_and_config_validation(self):
        for objective in ('wealth', 'society'):
            e = self.engine({}, objective_mode=objective, evolution_enabled=True)
            clone = load_team(dump_team(e), e.config, e.policy)
            prompt = clone.agents[0].get_system_prompt()
            self.assertEqual('Your sole objective' in prompt, objective == 'wealth')
            self.assertIn('fixed total team bid', prompt)
        for amount in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                self.engine({}, team_base_bid=amount)
