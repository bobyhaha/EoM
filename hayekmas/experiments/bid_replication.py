"""Launch all four seed17 replications only after primary completion and a fresh budget check."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .bid_ablation import prepare, CELLS, report
from .campaign_client import atomic_json, append


def upper_bound(primary):
    sources = [primary/c for c in CELLS] + [Path('runs')/p for p in (
        'k10-feedback-3h-20261001/wealth-voluntary-feedback', 'k10-short-rounds-20261001/society-fixed-3rounds',
        'k10-original-reference-20261001', 'k10-independent-20261001', 'k10-regrade-20261001',
        'k10-society-prompt-20261001', 'k10-pilot-20261001')]
    for name in ('k10-membership-clarity-20261001', 'k10-membership-clarity-v2-20261001'):
        membership = Path('runs')/name/'wealth-voluntary-feedback-membership'
        if (membership/'status.json').exists():
            sources.append(membership)
    regrades = Path('runs/k10-regrade-seed17-20261001')
    if (regrades/'status.json').exists():
        sources.append(regrades)
    for name in ('k10-original-high-reference-20261001','k10-regrade-original-high-20261001'):
        extra = Path('runs')/name
        if (extra/'status.json').exists():
            sources.append(extra)
    rows=[]
    for p in sources:
        if not (p/'status.json').exists():
            raise RuntimeError(f'Missing budget status: {p}')
        s=json.loads((p/'status.json').read_text());u=s['usage']
        used=u['cost_usd']+u['unconfirmed_usd']
        bound=used if s['status'] in {'completed','stopped','deadline_reached'} else u['cap_usd']
        if (p/'selection-plan.json').exists():
            selection=json.loads((p/'selection-status.json').read_text()) if (p/'selection-status.json').exists() else {}
            if selection.get('status') not in {'completed','stopped','not_started'}:
                bound=u['cap_usd']  # Reserve the same ledger for the declared pending selector.
        rows.append({'source':str(p),'status':s['status'],'bound_usd':bound})
    return rows


def run(primary, destination, key):
    deadline=json.loads((primary/'plan.json').read_text())['deadline']
    watch=primary/'replication-status.json'
    while True:
        states=[json.loads((primary/c/'status.json').read_text()) for c in CELLS]
        remaining=deadline-time.time()
        if all(s['status']=='completed' for s in states) and remaining>=3600:
            break
        if remaining<3600 or any(s['status']=='stopped' for s in states):
            atomic_json(watch,{'status':'not_started','reason':'Predeclared completion/time gate not met','updated_at':time.time()})
            return
        atomic_json(watch,{'status':'waiting_for_primary_completion','updated_at':time.time(),'remaining_seconds':remaining})
        time.sleep(30)
    budgets=upper_bound(primary)
    bound=sum(x['bound_usd'] for x in budgets)+40
    if bound>100:
        atomic_json(watch,{'status':'not_started','reason':'Aggregate worst-case budget would exceed $100','bound_usd':bound})
        return
    critical=['agent','compact_rounds','config','env','mas','policy','round_protocol','team','token_allocation']
    for name in critical:
        relative=Path('hayekmas/adapters/teams')/(name+'.py')
        if hashlib.sha256(relative.read_bytes()).hexdigest()!=hashlib.sha256((primary/'source'/relative).read_bytes()).hexdigest():
            atomic_json(watch,{'status':'not_started','reason':f'Model-facing source changed: {relative}'})
            return
    plan=prepare(destination,deadline,seed=17,cell_cap=10)
    plan.update(budget_note='Four disjoint $10 caps. Completed cells released to confirmed plus unresolved costs; all active partitions counted at full ceilings.',
                aggregate_worst_case_usd=bound,budget_sources=budgets,
                replication_of=str(primary),model_facing_sources_identical_to_primary=critical)
    atomic_json(destination/'plan.json',plan)
    atomic_json(primary/'replication-budget.json',{'checked_at':time.time(),'upper_bound_usd':bound,'new_caps_usd':40,'prior_sources':budgets})
    append(primary/'research_journal.jsonl',{'time':time.time(),'event':'seed17_replication_started','aggregate_worst_case_usd':bound,'selection':'All four conditions, all three tasks; not selected by score.'})
    children=[]
    for cell in CELLS:
        with (destination/cell/'worker.log').open('w') as log:
            p=subprocess.Popen([sys.executable,'-m','hayekmas.experiments.bid_ablation','--root',str(destination),'--cell',cell],
                env={**os.environ,'OPENROUTER_API_KEY':key},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p)
    atomic_json(destination/'workers.json',dict(zip(CELLS,[p.pid for p in children])))
    atomic_json(watch,{'status':'running','started_at':time.time(),'aggregate_bound_usd':bound})
    while any(p.poll() is None for p in children):
        report(destination)
        time.sleep(15)
    report(destination)
    atomic_json(watch,{'status':'finished','updated_at':time.time(),'destination':str(destination)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    key=os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    run(a.primary.resolve(),a.out.resolve(),key)
