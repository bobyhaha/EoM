"""Launch only untouched queued cells and monitor existing workers without restarting them."""
import argparse
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hayekmas.experiments.campaign_client import atomic_json
from hayekmas.experiments.priority12_study import report, verify


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def monitor(root):
    plan = json.loads((root / 'plan.json').read_text())
    children = {}
    for job in plan['jobs']:
        folder = root / job['name']
        if (folder / 'status.json').exists():
            continue
        if (folder / 'api_usage.jsonl').exists():
            raise RuntimeError('Refusing to restart a paid cell without a status')
        if time.time() >= plan['deadline'] or (root / 'STOP').exists():
            break
        verify(plan)
        with (root / f"{job['name']}.log").open('a') as log:
            children[job['name']] = subprocess.Popen(
                [sys.executable, '-m', 'hayekmas.experiments.priority12_study', '--root', str(root), '--job', job['name']],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    while True:
        scheduler, running = {}, False
        for job in plan['jobs']:
            name = job['name']
            child = children.get(name)
            path = root / name / 'status.json'
            if child is not None and child.poll() is not None and not path.exists():
                scheduler[name] = f'failed to start: exit {child.returncode}'
            elif path.exists():
                status = json.loads(path.read_text())
                if status['status'] == 'running' and not alive(status['pid']):
                    status.update(status='stopped', error='Worker exited unexpectedly; not restarted', updated_at=time.time())
                    atomic_json(path, status)
                scheduler[name] = status['status']
                running |= status['status'] == 'running'
            elif child is not None and child.poll() is None:
                scheduler[name] = 'starting'
                running = True
            else:
                scheduler[name] = 'not started: stop/deadline'
        report(root, scheduler)
        if not running:
            break
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.monitor:
        monitor(root)
        return
    plan = json.loads((root / 'plan.json').read_text())
    verify(plan)
    assert abs(sum(j['cap'] for j in plan['jobs']) - 10) < 1e-9
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    old = json.loads((root / 'supervisor.json').read_text())
    # Signal this supervisor only; existing model workers retain their processes.
    os.kill(old['pid'], signal.SIGTERM)
    for _ in range(50):
        if not alive(old['pid']):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Old supervisor did not exit; no new workers launched')
    plan.update(concurrency=10, concurrency_change={'at':time.time(), 'from':4, 'to':10,
                'reason':'User requested remaining seed in parallel; budgets and treatments unchanged.'})
    atomic_json(root / 'plan.json', plan)
    with (root / 'parallel-supervisor.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--monitor'],
            env={**os.environ, 'OPENROUTER_API_KEY':key}, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    atomic_json(root / 'supervisor.json', {'pid':process.pid,'started_at':time.time(),'previous_pid':old['pid'],'concurrency':10})
    print('Parallel supervisor started:', process.pid)


if __name__ == '__main__':
    main()
