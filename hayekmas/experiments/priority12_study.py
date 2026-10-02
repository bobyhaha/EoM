"""Matched closure × shared-draft pilot, with disjoint caps totaling $10."""
import argparse
import getpass
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .campaign_client import atomic_json


def prepare(root, hours=4):
    from .bid_ablation import prepare as base_prepare
    plan = base_prepare(root, time.time() + hours * 3600, cell_cap=1.1)
    for name in plan['cells']:
        (root / name).rmdir()
    variants = {'baseline': ('funded', 0), 'closure': ('public_work', 0),
                'draft': ('funded', 2), 'combined': ('public_work', 2)}
    cells, overrides, jobs = {}, {}, []
    for seed in (7, 17):
        jobs.append({'name': f'original-s{seed}', 'seed': seed, 'original': True, 'cap': .6})
        for name, (terminal, passes) in variants.items():
            cell = f'{name}-s{seed}'
            cells[cell] = ['wealth', 'fixed']
            overrides[cell] = {'seed': seed, 'terminal_policy': terminal, 'shared_draft_passes': passes}
            (root / cell).mkdir()
            jobs.append({'name': cell, 'seed': seed, 'original': False, 'cap': 1.1})
    plan.update(cells=cells, cell_overrides=overrides, jobs=jobs, user_max_usd=10,
                total_partitioned_cap_usd=10, concurrency=4, seeds=[7, 17],
                budget_note='Eight $1.10 team ledgers and two $0.60 original ledgers; no restarts or expansion.',
                protocol='Closure × shared drafts, fixed .1 bids, wealth objective, fresh k10 per task. Three development tasks, seeds7/17, max10 steps, no evolution/reflection. Original solver16384/high. Not held-out or trained-population evidence.')
    assert abs(sum(j['cap'] for j in jobs) - 10) < 1e-9
    atomic_json(root / 'plan.json', plan)
    return plan


def verify(plan):
    for name, digest in {**plan['source_sha256'], plan['dataset']: plan['dataset_sha256']}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Input changed: {name}')


def report(root, scheduler):
    plan = json.loads((root / 'plan.json').read_text())
    rows, cost, reserved = [], 0., 0.
    for job in plan['jobs']:
        out = root / job['name']
        status = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {'status': scheduler.get(job['name'], 'queued')}
        usage = status.get('usage', {})
        cost += usage.get('cost_usd', 0)
        reserved += usage.get('unconfirmed_usd', 0)
        results = status.get('results', [])
        rows.append({**status, 'name': job['name'], 'cap': job['cap'], 'completed': len(results)})
    summary = {'updated_at': time.time(), 'cost_usd': cost, 'reserved_usd': reserved, 'cap_usd': 10, 'cells': rows, 'scheduler': scheduler}
    atomic_json(root / 'summary.json', summary)
    body = '<html><head><meta http-equiv="refresh" content="20"><title>k10 priority 1+2 pilot</title></head><body><h1>k=10 closure × shared-draft pilot</h1><p>24 team episodes + 6 original EoM episodes. Seeds 7 and 17; same three development tasks; fresh populations. Not held-out evidence.</p>'
    body += f'<p>Confirmed ${cost:.4f}; reserved ${reserved:.4f}; hard aggregate cap $10. Scores are grader scores, not success rates.</p><table border="1" cellpadding="8"><tr><th>Cell</th><th>Status</th><th>Completed / 3</th><th>Scores</th><th>Answers</th><th>USD</th></tr>'
    for row in rows:
        results = row.get('results', [])
        values = [row['name'], row['status'], row['completed'], [r['score'] for r in results], sum(r['has_final_answer'] for r in results), round(row.get('usage', {}).get('cost_usd', 0), 4)]
        body += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in values) + '</tr>'
    body += '</table><h2>Completed replays / original results</h2><ul>'
    for p in sorted(root.glob('*/*/replay.html')) + sorted(root.glob('original-*/*/result.json')):
        rel = str(p.relative_to(root))
        body += f'<li><a href="{escape(rel)}">{escape(rel)}</a></li>'
    body += '</ul></body></html>'
    (root / 'index.tmp').write_text(body)
    (root / 'index.tmp').replace(root / 'index.html')


def supervise(root):
    plan = json.loads((root / 'plan.json').read_text())
    pending, active, scheduler = list(plan['jobs']), {}, {}
    while pending or active:
        for name, process in list(active.items()):
            if process.poll() is not None:
                scheduler[name] = f'exited {process.returncode}'
                del active[name]
        if time.time() >= plan['deadline'] or (root / 'STOP').exists():
            for job in pending:
                scheduler[job['name']] = 'not started: stop/deadline'
            pending.clear()
        while pending and len(active) < plan['concurrency']:
            job = pending.pop(0)
            verify(plan)
            with (root / f"{job['name']}.log").open('a') as log:
                active[job['name']] = subprocess.Popen([sys.executable, '-m', 'hayekmas.experiments.priority12_study', '--root', str(root), '--job', job['name']], stdout=log, stderr=log, stdin=subprocess.DEVNULL)
            scheduler[job['name']] = 'running'
        report(root, scheduler)
        if pending or active:
            time.sleep(10)
    report(root, scheduler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--job')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.launch:
        prepare(root)
        key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
        with (root / 'supervisor.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'hayekmas.experiments.priority12_study', '--root', str(root), '--supervise'], env={**os.environ, 'OPENROUTER_API_KEY': key}, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        atomic_json(root / 'supervisor.json', {'pid': process.pid, 'started_at': time.time()})
        print(f'Started supervisor PID {process.pid}; {root}')
    elif args.supervise:
        supervise(root)
    else:
        plan = json.loads((root / 'plan.json').read_text())
        verify(plan)
        job = next(j for j in plan['jobs'] if j['name'] == args.job)
        key = os.environ['OPENROUTER_API_KEY']
        if job['original']:
            from .bid_original_reference import run
            run(root / job['name'], key, plan['deadline'], solver_tokens=16384, solver_reasoning='high', cap=job['cap'], seed=job['seed'])
        else:
            from .bid_ablation import worker
            worker(root, job['name'], key)


if __name__ == '__main__':
    main()
