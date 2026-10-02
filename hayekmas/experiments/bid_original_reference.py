"""Original EoM reference on the three ablation tasks, fresh k10 population per task."""
from dataclasses import asdict
import argparse
import getpass
import json
import os
from pathlib import Path
import random
import time

from hayekmas.adapters.researchworld.runtime import (
    ResearchTrainer, create_research_agents, create_research_env, load_research_runtime_config,
)
from hayekmas.adapters.researchworld.env import load_research_tasks
from hayekmas.base.mas import HayekMAS
from hayekmas.utils.logger import logger
from .campaign_client import CampaignClient, CampaignStop, atomic_json


def run(root, key, deadline, *, solver_tokens=8192, solver_reasoning='medium', cap=2, seed=7):
    if type(seed) is not int or seed < 0:
        raise ValueError('Invalid seed')
    if solver_tokens not in (8192,16384) or solver_reasoning not in ('medium','high') or not 0 < cap <= 2:
        raise ValueError('Unsupported original-reference settings')
    root.mkdir(parents=True, exist_ok=False)
    raw = json.loads(Path('global_configs/train_research.json').read_text())
    spec = json.loads(Path('global_configs/k10_bid_ablation.json').read_text())
    for section, overrides in spec['original_overrides'].items():
        raw['mas'][section].update(overrides)
    cfg = load_research_runtime_config(raw)
    state = {'status': 'running', 'results': [], 'started_at': time.time(), 'pid': os.getpid()}

    def tick():
        state.update(usage=client.usage(), updated_at=time.time())
        atomic_json(root / 'status.json', state)

    class BoundedClient(CampaignClient):
        def _generate_impl(self, *args, **kwargs):
            if (root / 'STOP').exists() or (root.parent / 'STOP').exists():
                raise CampaignStop('Study stop requested')
            return super()._generate_impl(*args, **kwargs)

    client = BoundedClient(key, root, deadline, cap=cap, tick=tick)
    atomic_json(root / 'plan.json', {'seed': seed, 'scope': f'Fresh untrained original EoM k10 reference; three matched development tasks; seed{seed}; no cross-task carryover.',
        'config': asdict(cfg.mas), 'cap_usd': cap, 'deadline': deadline,
        'solver_tokens':solver_tokens, 'solver_reasoning':solver_reasoning,
        'comparability': f'Native original loop and fixed individual bids; {solver_tokens} {solver_reasoning} solver output vs team16384 high. Roles, early stopping and total compute still differ.'})
    logger.configure(verbose=False, log_dir=str(root), profile='original-reference')
    tasks = load_research_tasks([Path('third_party/benchmarks/frontier-science-research/data/research_train.jsonl')], limit=3)
    try:
        for index, task in enumerate(tasks):
            state.update(task_id=task.id, task_number=index+1)
            client.context = {'arm': 'original-reference', 'task_id': task.id, 'task_number': index+1}
            tick()
            random.seed(seed)
            engine = HayekMAS(cfg.mas)
            trainer = ResearchTrainer(client, cfg)
            agents = create_research_agents(client) + create_research_agents(client)
            for i, agent in enumerate(agents):
                agent.backbone_llm = client.as_callable(max_tokens=solver_tokens, reasoning_effort=solver_reasoning, kind='solve')
                agent.name = f'{agent.name}-reference-{i}'
                agent.initialize(initial_wealth=cfg.mas.engine.initial_wealth)
                engine.population.add_agent(agent)
            engine._initial_agents = list(agents)
            engine.set_agent_factory(trainer.create_agent_factory_good_birth(), trainer.create_agent_factory_bad_birth())
            engine.wakeup_llm_override = client.as_callable(max_tokens=1024, reasoning_effort='none', kind='wakeup')
            engine.eval()
            folder = root / f'{index+1:02d}-{task.id}'
            folder.mkdir()
            before = client.usage()['cost_usd']
            env = create_research_env(task, llm_client=client, reward_config=cfg.mas.reward,
                                      use_judge=True, judge_threshold=.5, max_steps=10)
            env.llm_fn = client.as_callable(max_tokens=8192, reasoning_effort='medium', kind='judge')
            with logger.scoped_task_log(folder / 'trajectory.log'):
                engine.run_one_episode(env, step_ckpt_save_path=str(folder / 'steps'))
            engine.save(str(folder / 'checkpoint.json'))
            result = {'task_id': task.id, 'score': env.get_terminal_score(), 'has_final_answer': bool(env.final_answer),
                      'answer': env.final_answer, 'judge': env._last_judge_reason,
                      'steps': env.step_count, 'actions': env.action_history,
                      'cost_usd': client.usage()['cost_usd'] - before}
            atomic_json(folder / 'result.json', result)
            state['results'].append({k: result[k] for k in ('task_id','score','has_final_answer','cost_usd','steps')})
            tick()
        state['status'] = 'completed'
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
    p.add_argument('--solver-tokens', type=int, default=8192, choices=(8192,16384))
    p.add_argument('--solver-reasoning', default='medium', choices=('medium','high'))
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--cap', type=float, default=2)
    a = p.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    run(a.root.resolve(), key, a.deadline, solver_tokens=a.solver_tokens, solver_reasoning=a.solver_reasoning, cap=a.cap, seed=a.seed)
