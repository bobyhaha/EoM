"""Exploratory feedback ablation: clarify existing invite/leave semantics only."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from . import bid_ablation as study
from .bid_replication import upper_bound
from .campaign_client import CampaignStop, append, atomic_json
from .feedback_ablation import FeedbackPolicy, FeedbackEngine


MEMBERSHIP_RULES = (
    'invite names an agent you want to recruit into YOUR group; it is NOT a request to join '
    'that agent\'s existing team. Only recipients who are solo after departures can accept. '
    'To request an invitation from an existing team, use your public message; this does not '
    'guarantee an invitation or admission. leave=true leaves your current team before '
    'invitations are processed, even if no new membership follows. leave=false means you do '
    'not request departure; others may still leave and a team with fewer than two members '
    'dissolves. Any non-null accepted invitation must be one actually offered to you.'
)


class MembershipPolicy(FeedbackPolicy):
    def respond(self, phase, agent, observation, max_tokens):
        if phase in {'round_membership', 'round_join'}:
            from hayekmas.adapters.teams.policy import INSTRUCTIONS
            observation = {**observation, 'membership_operation_semantics': MEMBERSHIP_RULES}
            size = self.engine.tokens.count(agent.get_system_prompt()) + self.engine.tokens.count(
                json.dumps({'instruction': INSTRUCTIONS[phase], 'observation': observation}, ensure_ascii=False))
            if size > min(self.engine.config.formation_context_tokens, self.engine.config.context_tokens):
                raise CampaignStop('Membership clarification exceeds unchanged formation input limit')
        return super().respond(phase, agent, observation, max_tokens)


def prepare(root, primary, deadline):
    if deadline - time.time() < 1800:
        raise CampaignStop('Less than thirty minutes remain; do not start another condition')
    bounds = upper_bound(primary)
    repeat = Path('runs/k10-ablation-seed17-20261001')
    if (repeat/'plan.json').exists():
        for cell in study.CELLS:
            path = repeat/cell/'status.json'
            state = json.loads(path.read_text()) if path.exists() else None
            usage = state['usage'] if state else {}
            bound = usage['cost_usd'] + usage['unconfirmed_usd'] if state and state['status'] in {'completed', 'stopped'} else 10
            bounds.append({'source': str(path.parent), 'bound_usd': bound})
    total = sum(r['bound_usd'] for r in bounds) + 2
    if total > 100:
        raise CampaignStop('Aggregate worst-case budget exceeds $100')
    plan = study.prepare(root, deadline)
    cell = 'wealth-voluntary-feedback-membership'
    (root/cell).mkdir()
    plan.update(cells={cell: ['wealth', 'voluntary']}, cell_cap_usd=2, total_partitioned_cap_usd=2,
        aggregate_worst_case_usd=total, budget_sources=bounds,
        budget_note='Separate $2 partition after primary completion; active replications counted at full caps.',
        protocol='Exploratory comparison against wealth-voluntary-feedback: only add accurate descriptions of existing invite/leave operations to membership and invitation-reply prompts. Same feedback, original membership mechanics, seed7, k10, three fresh tasks, ten rounds, and unchanged token/reasoning limits. Does not implement direct admission or atomic switching.',
        membership_rules=MEMBERSHIP_RULES)
    atomic_json(root/'plan.json', plan)
    for name, digest in plan['source_sha256'].items():
        source = Path(name)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        destination = root/'source'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    append(primary/'research_journal.jsonl', {'time':time.time(), 'event':'membership_semantics_ablation_declared',
        'root':str(root), 'aggregate_worst_case_usd':total, 'hypothesis':'Clarifying invitation direction reduces rejected invitations and unintended isolation. This need not improve science scores.',
        'selection':'All three existing development tasks; exploratory, not primary or held-out.'})
    return cell


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    root = args.root.resolve()
    cell = prepare(root, args.primary.resolve(), args.deadline)
    study.PilotPolicy = MembershipPolicy
    study.TeamMAS = FeedbackEngine
    try:
        study.worker(root, cell, key)
    finally:
        study.report(root)
