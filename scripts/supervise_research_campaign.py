"""Keep credentials only in memory; attach to workers and permit safe later restarts."""
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from hayekmas.experiments.campaign_client import atomic_json

root=Path('runs/eom-vs-teams-12h').resolve()
plan=json.loads((root/'plan.json').read_text())
key=os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden): ')
env=dict(os.environ,OPENROUTER_API_KEY=key)
workers=json.loads((root/'workers.json').read_text()) if (root/'workers.json').exists() else {}
children={}

def alive(arm):
    if arm in children:
        return children[arm].poll() is None
    try:
        os.kill(workers[arm],0)
        return True
    except (KeyError,ProcessLookupError):
        return False


def start(arm):
    folder=root/arm;folder.mkdir(exist_ok=True)
    with (folder/'worker.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-m','hayekmas.experiments.research_campaign','--root',str(root),'--arm',arm],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children[arm]=child;workers[arm]=child.pid
    atomic_json(root/'workers.json',workers)

print('Supervisor attached; existing workers preserved.',flush=True)
while time.time()<plan['deadline_unix'] and not (root/'STOP_SUPERVISOR').exists():
    for arm in ('original','teams'):
        marker=root/f'RESTART-{arm}'
        if marker.exists() and not alive(arm):
            marker.unlink();start(arm)
    atomic_json(root/'supervisor.json',{'pid':os.getpid(),'updated_at':time.time(),'deadline':plan['deadline_unix'],
        'workers':workers,'alive':{arm:alive(arm) for arm in ('original','teams')}})
    time.sleep(min(20,max(0,plan['deadline_unix']-time.time())))
for arm in ('original','teams'):
    if alive(arm):
        os.kill(workers[arm],signal.SIGTERM)
print('Research window ended; remaining workers stopped.',flush=True)
