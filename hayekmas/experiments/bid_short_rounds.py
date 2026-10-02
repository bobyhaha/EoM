"""Exploratory cost ablation: society objective, fixed bid, three rather than ten rounds."""
import argparse
import getpass
import os
from pathlib import Path
from . import bid_ablation as study
from .campaign_client import atomic_json

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    root = a.root.resolve()
    plan = study.prepare(root, a.deadline)
    cell = 'society-fixed-3rounds'
    (root / cell).mkdir()
    plan['config']['max_steps'] = 3
    plan.update(cells={cell: ['society', 'fixed']}, cell_cap_usd=2, total_partitioned_cap_usd=2,
                budget_note='Additional $2 ceiling within the authorized $100 session maximum.',
                protocol=plan['protocol'].replace('max10 rounds', 'max3 rounds') +
                ' Exploratory cost ablation: finalization on round3 instead of round10. All other team settings match society-fixed. This is a fixed horizon change, not an agent-selected early stop.')
    atomic_json(root / 'plan.json', plan)
    try:
        study.worker(root, cell, key)
    finally:
        study.report(root)
