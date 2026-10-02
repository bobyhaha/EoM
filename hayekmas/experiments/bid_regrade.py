"""Two repeat grades of every completed submitted answer; never replace primary scores."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import time

from .campaign_client import CampaignClient, CampaignStop, atomic_json
from .campaign_grader import score_from_text
from .bid_behavior import records


def run(root, sources, key, deadline):
    root.mkdir(parents=True, exist_ok=False)
    plan = {'sources': [str(p) for p in sources], 'replicates': 2, 'cap_usd': 1,
            'deadline': deadline, 'declared_at': time.time(),
            'selection': 'Every completed submitted answer from the declared source directories, regardless of score. Missing answers remain zero without API calls. Interrupted episodes are not scored.',
            'method': 'Repeat the saved successful original judge prompt, with identical system text and reasoning setting. Arm labels are not added to judge prompts. Grades are reported separately; no answer generation or grade replacement.',
            'limitation': 'Same-model grader variability, not independent expert validation. Replicates are not independent research tasks.'}
    atomic_json(root / 'plan.json', plan)
    state = {'status': 'running', 'grades': [], 'started_at': time.time()}

    def tick():
        state.update(usage=client.usage(), updated_at=time.time())
        atomic_json(root / 'status.json', state)

    client = CampaignClient(key, root, deadline, cap=1, tick=tick)
    tick()
    seen = set()
    try:
        while time.time() < deadline - 5:
            for source in sources:
                calls = {r['request']: r for r in records(source / 'api_usage.jsonl')}
                prompts = {r['request']: r for r in records(source / 'requests.jsonl')}
                for path in sorted(source.glob('*/result.json')):
                    result = json.loads(path.read_text())
                    for repeat in (1, 2):
                        identifier = (str(path), repeat)
                        if identifier in seen:
                            continue
                        record = {'source': str(path), 'task_id': result['task_id'], 'repeat': repeat,
                                  'primary_score': result['score'], 'answer_sha256': hashlib.sha256(str(result.get('answer') or '').encode()).hexdigest()}
                        if not result['has_final_answer']:
                            record.update(score=0, reason='No submitted answer; no API call')
                        else:
                            eligible = [r for r in calls.values() if r.get('kind') == 'judge'
                                        and r.get('task_id') == result['task_id'] and r.get('status') == 'charged']
                            if not eligible:
                                record.update(score=None, error='No confirmed original judge request found')
                            else:
                                original = eligible[-1]
                                request = prompts[original['request']]
                                client.context = {'source_result': str(path), 'task_id': result['task_id'], 'repeat': repeat}
                                text = client.generate(request['prompt'], system_prompt=request.get('system'),
                                    max_tokens=request['max_tokens'], reasoning_effort=request['reasoning_effort'], kind='judge_recheck')
                                record.update(score=score_from_text(text), judgment=text,
                                    judge_prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest())
                        state['grades'].append(record)
                        seen.add(identifier)
                        tick()
            statuses = [json.loads((s / 'status.json').read_text()).get('status') for s in sources if (s / 'status.json').exists()]
            if len(statuses) == len(sources) and all(s in {'completed', 'stopped'} for s in statuses):
                state['status'] = 'completed'
                break
            time.sleep(30)
        if state['status'] == 'running':
            state['status'] = 'deadline_reached'
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {str(exc).replace(key, "[redacted]")}')
        if not isinstance(exc, CampaignStop):
            raise
    finally:
        tick()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--sources', type=Path, nargs='+', required=True)
    a = p.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    run(a.root.resolve(), [p.resolve() for p in a.sources], key, a.deadline)
