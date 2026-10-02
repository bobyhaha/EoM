"""Bounded 2x2 objective-by-bid ablation; independent task populations and persisted receipts."""
import argparse
from dataclasses import asdict
import fcntl
import getpass
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ResearchTaskEnv, load_tasks
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.replay import write_replay
from hayekmas.utils.logger import logger
from .campaign_client import CampaignClient, CampaignStop, atomic_json, append
from .k10_pilot import PilotPolicy
from .team_checkpoint import dump_team

CELLS = {'wealth-voluntary': ('wealth', 'voluntary'), 'society-voluntary': ('society', 'voluntary'),
         'wealth-fixed': ('wealth', 'fixed'), 'society-fixed': ('society', 'fixed')}


def prepare(root, deadline, *, seed=7, cell_cap=20):
    if type(seed) is not int or seed < 0 or not 0 < cell_cap <= 20:
        raise ValueError("Invalid seed or cell cap")
    root.mkdir(parents=True, exist_ok=False)
    dataset = Path('third_party/benchmarks/frontier-science-research/data/research_train.jsonl')
    tasks = load_tasks(dataset, 'train')[:3]
    config = json.loads(Path('runs/population-study-evolution-f2dbe94/cells/teams-k10.json').read_text())['team_config']
    config.update(rounds=1, evolution_enabled=False, seed=seed)
    plan = {'deadline': deadline, 'started_at': time.time(), 'cells': CELLS, 'cell_cap_usd': cell_cap,
            'total_partitioned_cap_usd': 4 * cell_cap, 'user_max_usd': 100,
            'budget_note': 'Four disjoint $20 ledgers cap this study at $80; prior pilots remain below $2. No automatic expansion.',
            'tasks': [t.id for t in tasks], 'dataset': str(dataset),
            'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'config': config,
            'model': 'openai/gpt-6-luna', 'seed': seed,
            'protocol': '2x2 objective x bid rule. Fresh k10 population per task; max10 rounds; no reflection/evolution; lambda0. Fixed bid=.1 per team, equally paid by explicit consenting yes voters, strict majority required. No feedback fix in primary cells. Same judge and token/reasoning settings. Development tasks, one seed; unequal actual compute is measured.',
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('hayekmas').rglob('*.py'))}}
    atomic_json(root / 'plan.json', plan)
    for name in CELLS:
        (root / name).mkdir()
    return plan


def worker(root, cell, key):
    plan = json.loads((root / 'plan.json').read_text())
    out = root / cell
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'api_usage.jsonl').exists():
        raise RuntimeError('Paid cell already started; explicit recovery required')
    for name, digest in plan['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed before launch: {name}')
    dataset = Path(plan['dataset'])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != plan['dataset_sha256']:
        raise RuntimeError('Dataset changed')
    tasks = load_tasks(dataset, 'train')[:3]
    objective, bidding = plan['cells'][cell]
    cfg = TeamConfig(**{**plan['config'], 'objective_mode': objective, 'team_bid_rule': bidding})
    state = {'cell': cell, 'status': 'running', 'results': [], 'started_at': time.time(), 'pid': os.getpid()}

    def tick():
        state.update(usage=client.usage(), updated_at=time.time())
        atomic_json(out / 'status.json', state)

    class BoundedClient(CampaignClient):
        def _generate_impl(self, *args, **kwargs):
            if (root / 'STOP').exists() or (out / 'STOP').exists():
                raise CampaignStop('Study stop requested')
            return super()._generate_impl(*args, **kwargs)

    client = BoundedClient(key, out, plan['deadline'], cap=plan['cell_cap_usd'], tick=tick)
    atomic_json(out / 'config.json', asdict(cfg))
    tick()
    logger.configure(verbose=False, log_dir=str(out), profile=cell)
    engine = None
    folder = None
    try:
        for index, task in enumerate(tasks):
            if time.time() >= plan['deadline']:
                raise CampaignStop('Study time window ended')
            state.update(task_id=task.id, task_number=index + 1)
            client.context = {'cell': cell, 'task_id': task.id, 'task_number': index + 1, 'phase': 'development'}
            folder = out / f'{index+1:02d}-{task.id}'
            folder.mkdir()
            engine = TeamMAS(cfg, PilotPolicy(client), event_sink=lambda e: append(folder / 'events.jsonl', e))
            assert len(engine.agents) == 10
            before = client.usage()['cost_usd']
            tick()
            env = ResearchTaskEnv(task, cfg.reward, lambda prompt: client.generate(
                prompt, max_tokens=cfg.judge_tokens, reasoning_effort='medium', kind='judge'))
            with logger.scoped_task_log(folder / 'trajectory.log'):
                metric = engine.run_one_episode(env, training=False, formation=True, reflection=False)
            engine.assert_accounting()
            atomic_json(folder / 'checkpoint.json', dump_team(engine))
            finals = [e for e in engine.events if e['event'] == 'submission' and e.get('final')]
            answer = finals[-1]['answer'] if finals else None
            result = {'cell': cell, 'task_id': task.id, 'score': env.get_terminal_score(),
                      'has_final_answer': bool(answer), 'answer': answer, 'metric': metric,
                      'cost_usd': client.usage()['cost_usd'] - before,
                      'completed_at': time.time(), 'population': engine.state()['agents']}
            atomic_json(folder / 'result.json', result)
            state['results'].append({k: result[k] for k in ('cell', 'task_id', 'score', 'has_final_answer', 'cost_usd')})
            write_replay(engine, folder, 'completed')
            tick()
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {str(exc).replace(key, "[redacted]")}')
        if engine is not None and folder is not None:
            write_replay(engine, folder, 'interrupted')
        if not isinstance(exc, CampaignStop):
            raise
    finally:
        tick()


def report(root):
    plan = json.loads((root / 'plan.json').read_text())
    cells = []
    for cell in plan['cells']:
        out = root / cell
        s = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {'status': 'pending'}
        events = []
        for path in sorted(out.glob('*/events.jsonl')):
            # A read may catch an unfinished append. Ignore only an incomplete final line.
            for line in path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        auctions = [e for e in events if e['event'] == 'auction']
        last = events[-1] if events else {}
        rows = [json.loads(p.read_text()) for p in sorted(out.glob('*/result.json'))]
        item = {**s, 'cell': cell, 'auctions': len(auctions),
                'funded_auctions': sum(e['winner'] is not None for e in auctions),
                'public_submissions': sum(e['event'] == 'submission' for e in events),
                'latest_step': last.get('step'), 'latest_phase': last.get('phase', last.get('event')),
                'completed_tasks': len(rows), 'scores': [r['score'] for r in rows]}
        cells.append(item)
    summary = {'updated_at': time.time(), 'deadline': plan['deadline'], 'cells': cells,
               'cost_usd': sum(c.get('usage', {}).get('cost_usd', 0) for c in cells),
               'reserved_usd': sum(c.get('usage', {}).get('unconfirmed_usd', 0) for c in cells)}
    atomic_json(root / 'summary.json', summary)
    body = '<html><head><meta http-equiv="refresh" content="30"><title>k10 bid ablation</title></head><body><h1>k=10 prompt × bidding ablation</h1><p>Three matched development tasks, fresh populations, seed 7. Primary conditions remain fixed. Not a trained-population or held-out study.</p>'
    body += f'<p>Confirmed ${summary["cost_usd"]:.4f}; pending reservations ${summary["reserved_usd"]:.4f}. Four $20 partitions; $80 primary ceiling.</p>'
    body += '<table border="1" cellpadding="8"><tr><th>Condition</th><th>Status</th><th>Task / round / phase</th><th>Funded / auctions</th><th>Submissions</th><th>Scores</th><th>Cost</th></tr>'
    for c in cells:
        values = [c['cell'], c['status'], f'{c.get("task_number", "-")} / {c["latest_step"]} / {c["latest_phase"]}',
                  f'{c["funded_auctions"]}/{c["auctions"]}', c['public_submissions'], c['scores'], round(c.get('usage', {}).get('cost_usd', 0), 4)]
        body += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in values) + '</tr>'
    body += '</table><h2>Completed task replays</h2><ul>'
    for p in sorted(root.glob('*/*/replay.html')):
        rel = p.relative_to(root)
        body += f'<li><a href="{escape(str(rel))}">{escape(str(rel.parent))}</a></li>'
    body += '</ul></body></html>'
    (root / 'index.html.tmp').write_text(body)
    (root / 'index.html.tmp').replace(root / 'index.html')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cell', choices=CELLS)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.report:
        print(json.dumps(report(root)))
    elif args.launch:
        if not args.deadline or args.deadline <= time.time():
            raise SystemExit('Future explicit deadline required')
        key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
        prepare(root, args.deadline)
        children = []
        for cell in CELLS:
            with (root / cell / 'worker.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-m', 'hayekmas.experiments.bid_ablation', '--root', str(root), '--cell', cell],
                    env={**os.environ, 'OPENROUTER_API_KEY': key}, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
        atomic_json(root / 'workers.json', dict(zip(CELLS, [p.pid for p in children])))
        while any(p.poll() is None for p in children):
            report(root)
            time.sleep(15)
        report(root)
    elif args.cell:
        worker(root, args.cell, os.environ['OPENROUTER_API_KEY'])
    else:
        parser.error('Choose --launch, --cell, or --report')
