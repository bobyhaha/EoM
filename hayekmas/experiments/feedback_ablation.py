"""Predeclared follow-up: voluntary bids with factual previous-auction feedback."""
import argparse
import getpass
import json
import os
from pathlib import Path

from . import bid_ablation as study
from .campaign_client import CampaignStop, atomic_json
from .k10_pilot import PilotPolicy


class FeedbackPolicy(PilotPolicy):
    engine = None

    def respond(self, phase, agent, observation, max_tokens):
        if phase in {'round_chat', 'round_commit'}:
            engine = self.engine
            auctions = [e for e in engine.events if e['event'] == 'auction']
            latest = auctions[-1] if auctions else None
            feedback = {
                'completed_auctions': len(auctions),
                'last_winner': latest['winner'] if latest else None,
                'last_all_bids_zero': all(v == 0 for v in latest['bids'].values()) if latest else None,
                'your_last_pledge': latest['contributions'].get(agent.name) if latest else None,
                'accepted_public_contributions': sum(e['event'] == 'submission' for e in engine.events),
                'rule': 'Voting yes or discussing privately does not submit work or earn credit. A team must have a positive total bid and win to publish work. If all bids are zero, nobody acts. You may abstain; only authorize your own funds.',
            }
            observation = {**observation, 'auction_feedback': feedback}
            if phase == 'round_commit':
                from hayekmas.adapters.teams.policy import INSTRUCTIONS
                size = engine.tokens.count(agent.get_system_prompt()) + engine.tokens.count(json.dumps(
                    {'instruction': INSTRUCTIONS[phase], 'observation': observation}, ensure_ascii=False))
                if size > engine.config.pledge_context_tokens:
                    raise CampaignStop('Feedback would exceed the unchanged pledge context cap')
        return super().respond(phase, agent, observation, max_tokens)


class FeedbackEngine(study.TeamMAS):
    def __init__(self, config, policy, event_sink=None):
        super().__init__(config, policy, event_sink)
        policy.engine = self


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    root = args.root.resolve()
    plan = study.prepare(root, args.deadline)
    cell = 'wealth-voluntary-feedback'
    (root / cell).mkdir()
    plan.update(cells={cell: ['wealth', 'voluntary']}, cell_cap_usd=5, total_partitioned_cap_usd=5,
                budget_note='Additional $5 partition; primary $80 + feedback $5 + earlier pilot caps below $2 remain below $100.',
                protocol=plan['protocol'] + ' Exploratory follow-up: add factual prior-auction feedback and restate existing positive-bid rule at chat and commitment. Same monetary mechanism, token limits and reasoning settings. Requests ledger contains the additional supplied feedback.')
    atomic_json(root / 'plan.json', plan)
    study.PilotPolicy = FeedbackPolicy
    study.TeamMAS = FeedbackEngine
    try:
        study.worker(root, cell, key)
    finally:
        study.report(root)
