"""Independent read-only checks and behavior summaries for the bid ablation."""
import argparse
from collections import Counter
from html import escape
import json
import math
from pathlib import Path
import re
import time

from .campaign_client import atomic_json
from hayekmas.adapters.teams.token_allocation import phase_costs


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def analyze(folder, rule):
    es = records(folder / 'events.jsonl')
    auctions = [e for e in es if e['event'] == 'auction']
    ballots = [e for e in es if e['event'] == 'decision_response' and e.get('phase') == 'round_commit']
    valid = []
    issues = []
    for e in ballots:
        try:
            valid.append((e, json.loads(e['response'])))
        except (ValueError, TypeError):
            continue
    activations = {(e['step'], e['group']): e for e in es if e['event'] == 'team_activation'}
    for e in auctions:
        step, winner = e['step'], e['winner']
        for group, bid in e['bids'].items():
            a = activations.get((step, group))
            if not a:
                issues.append(f'step {step}: missing activation for {group}')
                continue
            summed = math.fsum(e['contributions'][n] for n in a['members'])
            if not math.isclose(bid, summed, abs_tol=1e-9):
                issues.append(f'step {step}: bid != sum of personal pledges')
            if a['active'] != (a['yes_votes'] > len(a['members']) / 2):
                issues.append(f'step {step}: majority rule mismatch')
            if rule == 'fixed' and a['active']:
                if not math.isclose(bid, .1, abs_tol=1e-9):
                    issues.append(f'step {step}: fixed bid != .1')
                for n in a['members']:
                    expected = .1 / a['yes_votes'] if a['votes'][n] else 0
                    if not math.isclose(e['contributions'][n], expected, abs_tol=1e-9):
                        issues.append(f'step {step}: fixed cost allocation mismatch for {n}')
                    matching = [(x, b) for x, b in valid if x['step'] == step and x['agent'] == n]
                    if expected > 0 and (not matching or matching[-1][1].get('authorize_base_bid') is not True):
                        issues.append(f'step {step}: missing explicit payment consent for {n}')
        if winner is None and any(v > 0 for v in e['bids'].values()):
            issues.append(f'step {step}: positive bid without winner')
        if winner is not None:
            if e['bids'][winner] <= 0 or not math.isclose(e['bids'][winner], max(e['bids'].values()), abs_tol=1e-9):
                issues.append(f'step {step}: winner not maximal positive bid')
            # All study cells use bid_cost_rate=1.
            if not math.isclose(e['paid'], sum(e['contributions'][n] for n in e['members']), abs_tol=1e-9):
                issues.append(f'step {step}: winner payment mismatch')
        if e['paid'] < 0 or any(not math.isfinite(v) or v < 0 for v in e['contributions'].values()):
            issues.append(f'step {step}: invalid money')
    settlements = {e['step']: e for e in es if e['event'] == 'settlement'}
    previous_members = []
    for e in auctions:
        settlement = settlements.get(e['step'])
        if settlement:
            for name, opening in e['opening_wealth'].items():
                expected = opening - (e['contributions'][name] if name in e['members'] else 0)
                expected += e.get('credits', {}).get(name, 0) + settlement.get('credits', {}).get(name, 0)
                if not math.isclose(settlement['wealth'][name], expected, abs_tol=1e-8):
                    issues.append(f"step {e['step']}: per-agent wealth flow mismatch for {name}")
        expected_credits = {name: e['paid']/len(previous_members) for name in previous_members} if e['winner'] and previous_members else {}
        if set(e.get('credits', {})) != set(expected_credits) or any(
                not math.isclose(e['credits'][name], value, abs_tol=1e-9) for name, value in expected_credits.items()):
            issues.append(f"step {e['step']}: previous-winner transfer mismatch")
        if e['members']:
            previous_members = e['members']
    for e in es:
        if e['event'] != 'path_reward':
            continue
        n = e['contributing_rounds']
        share = e['environment_reward']/n if n else 0
        totals = Counter()
        if len(e['payouts']) != n:
            issues.append('Path length differs from credited rounds')
        for payout in e['payouts']:
            if set(payout['credits']) != set(payout['members']):
                issues.append('Historical member set differs from reward recipients')
            for name, amount in payout['credits'].items():
                if not math.isclose(amount, share, abs_tol=1e-9):
                    issues.append('Per-member path reward is not R/N')
                totals[name] += amount
        for name, amount in e['credits'].items():
            if not math.isclose(amount, totals[name], abs_tol=1e-9):
                issues.append('Aggregate member reward differs from path credits')
        if not math.isclose(e['issued'], sum(totals.values()), abs_tol=1e-9):
            issues.append('Issued reward differs from sum of member credits')
    submissions = [e for e in es if e['event'] == 'submission']
    repeats = []
    previous = {}
    for e in es:
        if e['event'] != 'team_message' or e.get('channel') != 'pre_bid':
            continue
        words = set(re.findall(r'\w+', e.get('text', '').lower()))
        name = e['agent']
        if name in previous and len(words) > 20:
            old = previous[name]
            sim = len(words & old) / max(1, len(words | old))
            if sim > .65:
                repeats.append({'agent': name, 'step': e['step'], 'word_set_similarity': round(sim, 3)})
        previous[name] = words
    quotes = [{'step': e['step'], 'agent': e['agent'], 'reason': b.get('reason'),
               'pledge': b.get('contribution'), 'act': b.get('act'), 'consent': b.get('authorize_base_bid')}
              for e, b in valid if b.get('reason')]
    result_path = folder / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    return {'task': folder.name, 'complete': bool(result), 'score': result['score'] if result else None,
            'final_answer': result['has_final_answer'] if result else None,
            'auctions': len(auctions), 'funded': sum(e['winner'] is not None for e in auctions),
            'positive_pledges': sum(v > 0 for e in auctions for v in e['contributions'].values()),
            'ballots': len(ballots), 'yes_votes': sum(b.get('act') is True for _, b in valid),
            'explicit_consents': sum(b.get('authorize_base_bid') is True for _, b in valid),
            'submissions': len(submissions), 'joins': sum(e['event'] == 'joined' for e in es),
            'leaves': sum(e['event'] == 'left' for e in es),
            'invalid_actions': dict(Counter(e.get('phase') for e in es if e['event'] == 'invalid_action')),
            'verification_issues': issues, 'repetition_flags': repeats, 'ballot_reasons': quotes,
            'latest': {k: es[-1].get(k) for k in ('event', 'step', 'phase')} if es else {}}


def report(root):
    plan = json.loads((root / 'plan.json').read_text())
    output = {'audit_version': 2, 'updated_at': time.time(), 'cells': {}}
    for cell, (_, rule) in plan['cells'].items():
        out = root / cell
        latest = {r['request']: r for r in records(out / 'api_usage.jsonl')}
        output['cells'][cell] = {
            'tasks': [analyze(p.parent, rule) for p in sorted(out.glob('*/events.jsonl'))],
            'costs': phase_costs(list(latest.values())),
            'transport': dict(Counter(r['status'] for r in latest.values()))}
    atomic_json(root / 'behavior.json', output)
    body = '<html><head><meta http-equiv="refresh" content="60"><title>Agent behavior audit</title></head><body><h1>Agent behavior audit</h1><p>Read-only event checks. Similarity flags are lexical repetition indicators, not judgments of scientific quality.</p>'
    for cell, data in output['cells'].items():
        body += f'<h2>{escape(cell)}</h2>'
        for task in data['tasks']:
            small = {k: v for k, v in task.items() if k not in ('ballot_reasons', 'repetition_flags')}
            body += '<pre>' + escape(json.dumps(small, indent=2)) + '</pre>'
            body += '<details><summary>Ballot reasons and pledges</summary><pre>' + escape(json.dumps(task['ballot_reasons'], indent=2)) + '</pre></details>'
        body += '<details><summary>Phase costs</summary><pre>' + escape(json.dumps(data['costs'], indent=2)) + '</pre></details>'
    body += '</body></html>'
    (root / 'behavior.html.tmp').write_text(body)
    (root / 'behavior.html.tmp').replace(root / 'behavior.html')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    deadline = json.loads((a.root / 'plan.json').read_text())['deadline']
    while True:
        report(a.root)
        if not a.watch or time.time() >= deadline + 130:
            break
        time.sleep(45)
