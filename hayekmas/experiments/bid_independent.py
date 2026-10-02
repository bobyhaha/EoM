"""Ten independent solutions per matched task; pass@k is oracle judged, not a selection algorithm."""
import argparse
import getpass
import hashlib
import math
import os
from pathlib import Path
import time

from hayekmas.adapters.teams.env import ResearchTaskEnv, load_tasks
from hayekmas.adapters.teams.agent import TeamAction
from .campaign_client import CampaignClient, CampaignStop, atomic_json


def run(root, key, deadline):
    root.mkdir(parents=True, exist_ok=False)
    dataset = Path('third_party/benchmarks/frontier-science-research/data/research_train.jsonl')
    tasks = load_tasks(dataset, 'train')[:3]
    atomic_json(root / 'plan.json', {'task_ids':[t.id for t in tasks], 'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'samples_per_task':10,'cap_usd':2,'deadline':deadline,'model':'openai/gpt-6-luna',
        'solver_max_tokens':16384,'solver_reasoning':'high','judge_max_tokens':8192,'judge_reasoning':'medium',
        'interpretation':'No shared state, economics, tools, or access to other samples. First sample is the prespecified single-agent baseline. Pass@k uses judge success >=.5; oracle outcome, not a deployable answer selector. Report partial tasks explicitly; no replacing samples based on scores.'})
    state = {'status':'running','results':[],'started_at':time.time()}

    def tick():
        state.update(usage=client.usage(),updated_at=time.time())
        atomic_json(root/'status.json',state)

    client = CampaignClient(key,root,deadline,cap=2,tick=tick)
    tick()
    try:
        for task in tasks:
            folder=root/task.id
            folder.mkdir()
            for sample in range(1,11):
                state.update(task_id=task.id,sample=sample)
                client.context={'task_id':task.id,'sample':sample,'condition':'independent'}
                before=client.usage()['cost_usd']
                answer=client.generate(task.problem,system_prompt='Solve this scientific problem carefully. Provide a complete answer with the derivations, assumptions, and justifications needed to assess it.',
                    max_tokens=16384,reasoning_effort='high',kind='independent_solve')
                env=ResearchTaskEnv(task,12,lambda prompt:client.generate(prompt,max_tokens=8192,reasoning_effort='medium',kind='judge'))
                env.initialize()
                env.apply(TeamAction(answer,'anonymous-independent',final=True))
                result={'task_id':task.id,'sample':sample,'score':env.get_terminal_score(),'has_final_answer':bool(answer),
                        'answer':answer,'cost_usd':client.usage()['cost_usd']-before}
                atomic_json(folder/f'sample-{sample:02d}.json',result)
                state['results'].append({k:result[k] for k in ('task_id','sample','score','cost_usd')})
                tick()
        state['status']='completed'
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {str(exc).replace(key,"[redacted]")}')
        if not isinstance(exc,CampaignStop):
            raise
    finally:
        stats=[]
        for task in tasks:
            scores=[r['score'] for r in state['results'] if r['task_id']==task.id]
            n=len(scores);c=sum(v>=.5 for v in scores)
            stats.append({'task_id':task.id,'n':n,'correct':c,'single_agent_first_score':scores[0] if scores else None,
                'mean_score':sum(scores)/n if n else None,'best_score':max(scores) if scores else None,
                'pass_at_k':{k:(1-math.comb(n-c,k)/math.comb(n,k) if n-c>=k else 1) if n>=k else None for k in (1,3,5,10)}})
        atomic_json(root/'statistics.json',stats)
        tick()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    key=os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    run(a.root.resolve(),key,a.deadline)
