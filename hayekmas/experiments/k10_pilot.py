"""Three matched training tasks with a single $5 ledger; no automatic restart."""
import argparse
from dataclasses import asdict
import fcntl
import getpass
import hashlib
from html import escape
import json
import os
from pathlib import Path
import random
import subprocess
import time

from hayekmas.adapters.researchworld.runtime import (
    ResearchTrainer, create_research_agents, create_research_env, load_research_runtime_config,
)
from hayekmas.adapters.researchworld.env import load_research_tasks
from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ResearchTaskEnv, load_tasks
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.policy import INSTRUCTIONS
from hayekmas.adapters.teams.replay import write_replay
from hayekmas.base.mas import HayekMAS
from hayekmas.utils.logger import logger
from .campaign_client import CampaignClient, CampaignStop, append, atomic_json
from .team_checkpoint import dump_team


class PilotPolicy:
    label = 'llm'

    def __init__(self, client):
        self.client = client

    def respond(self, phase, agent, observation, max_tokens):
        return self.client.generate(
            json.dumps({'instruction': INSTRUCTIONS[phase], 'observation': observation}, ensure_ascii=False),
            system_prompt=agent.get_system_prompt(), max_tokens=max_tokens,
            reasoning_effort=agent.phase_reasoning_efforts.get(phase, 'none'), kind=phase, json_mode=True,
        )


def run(root, key, *, task_count=3, arms=("original", "teams"), cap=5):
    if not 1 <= task_count <= 3 or not 0 < cap <= 5:
        raise ValueError("Pilot requires 1–3 tasks and a cap of at most $5")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'api_usage.jsonl').exists():
        raise RuntimeError('Existing paid ledger: inspect before any restart; automatic replay is disabled.')
    specs = Path('runs/population-study-evolution-f2dbe94/cells')
    original_spec = json.loads((specs / 'original-k10.json').read_text())
    team_spec = json.loads((specs / 'teams-k10.json').read_text())
    dataset = Path('third_party/benchmarks/frontier-science-research/data/research_train.jsonl')
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert digest == original_spec['dataset_sha256']['train'] == team_spec['dataset_sha256']['train']
    tasks = load_research_tasks([dataset], limit=task_count)
    team_tasks = load_tasks(dataset, 'train')[:task_count]
    raw = json.loads(Path('global_configs/train_research.json').read_text())
    for section, overrides in original_spec['original_overrides'].items():
        raw['mas'][section].update(overrides)
    cfg = load_research_runtime_config(raw)
    tcfg = TeamConfig(**{**team_spec['team_config'], 'rounds': task_count})
    state = {'status': 'running', 'started_at': time.time(), 'results': [], 'cap_usd': cap}
    atomic_json(root / 'plan.json', {
        'scope': f'{task_count} training task(s), arms {arms}, fresh populations, seed 7; not a held-out evaluation.',
        'task_ids': [t.id for t in tasks], 'dataset_sha256': digest,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path('hayekmas').rglob('*.py'))},
        'original_config': asdict(cfg.mas), 'team_config': asdict(tcfg),
        'shared_cap_usd': cap, 'included_in_prior_campaign_budget': True,
        'model': 'openai/gpt-6-luna', 'restart': 'Requires explicit review; never silently replay paid tasks.',
    })

    def tick():
        state['usage'] = client.usage()
        state['updated_at'] = time.time()
        atomic_json(root / 'status.json', state)
        rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(r[k]))}</td>'
                       for k in ('arm', 'task_id', 'score', 'has_final_answer', 'cost_usd')) + '</tr>'
                       for r in state['results'])
        page = ('<html><head><meta http-equiv="refresh" content="20"><title>k=10 pilot</title></head>'
                f'<body><h1>k=10 pilot</h1><p>{task_count} training task(s); arms {escape(str(arms))}; initial 10, cap 20; ${cap} limit.</p>'
                '<p>Training/debugging evidence, not a held-out performance estimate.</p><pre>'
                + escape(json.dumps({k: v for k, v in state.items() if k != 'results'}, indent=2))
                + '</pre><table border="1"><tr><th>Arm</th><th>Task</th><th>Score</th><th>Answer</th><th>USD</th></tr>'
                + rows + '</table><p><a href="teams/replay.html">Team replay (updates after each task)</a></p></body></html>')
        (root / 'index.html.tmp').write_text(page)
        (root / 'index.html.tmp').replace(root / 'index.html')

    client = CampaignClient(key, root, time.time() + 6 * 3600, cap=cap, tick=tick)
    tick()
    logger.configure(verbose=False, log_dir=str(root), profile='pilot')
    trainer = ResearchTrainer(client, cfg)
    original = HayekMAS(cfg.mas)
    agents = create_research_agents(client) + create_research_agents(client)
    for i, agent in enumerate(agents):
        agent.name = f'{agent.name}-pilot-{i}'
        agent.initialize(initial_wealth=cfg.mas.engine.initial_wealth)
        original.population.add_agent(agent)
    assert len(original.population.get_all()) == 10
    assert len({a.id for a in agents}) == 10
    original._initial_agents = list(agents)
    original.set_agent_factory(trainer.create_agent_factory_good_birth(), trainer.create_agent_factory_bad_birth())
    original.wakeup_llm_override = client.as_callable(max_tokens=1024, reasoning_effort='none', kind='wakeup')
    original.train()
    teamdir = root / 'teams'
    teamdir.mkdir(exist_ok=True)
    teams = TeamMAS(tcfg, PilotPolicy(client), event_sink=lambda e: append(teamdir / 'events.jsonl', e))
    assert len(teams.agents) == 10
    write_replay(teams, teamdir)
    try:
        for index, task in enumerate(tasks):
            for arm in arms:
                random.seed(7 + index)
                state.update(arm=arm, task_id=task.id, task_number=index + 1)
                client.context = {'arm': arm, 'task_id': task.id, 'task_number': index + 1, 'phase': 'training'}
                tick()
                folder = root / arm / f'{index + 1:02d}-{task.id}'
                folder.mkdir(parents=True, exist_ok=True)
                before = client.usage()['cost_usd']
                with logger.scoped_task_log(folder / 'trajectory.log'):
                    if arm == 'original':
                        env = create_research_env(task, llm_client=client, reward_config=cfg.mas.reward,
                                                  use_judge=True, judge_threshold=.5, max_steps=10)
                        env.llm_fn = client.as_callable(max_tokens=8192, reasoning_effort='medium', kind='judge')
                        original.run_one_episode(env, step_ckpt_save_path=str(folder / 'steps'))
                        original.save(str(folder / 'checkpoint.json'))
                        answer = env.final_answer
                        details = {'actions': env.action_history, 'steps': env.step_count,
                                   'termination': original.last_termination_reason.value,
                                   'population': [a.serialize() for a in original.population.get_all()]}
                    else:
                        env = ResearchTaskEnv(team_tasks[index], tcfg.reward,
                            lambda prompt: client.generate(prompt, max_tokens=8192, reasoning_effort='medium', kind='judge'))
                        metric = teams.run_one_episode(env, training=True, formation=True, reflection=True)
                        atomic_json(folder / 'checkpoint.json', dump_team(teams))
                        write_replay(teams, teamdir)
                        finals = [e for e in teams.events if e['event'] == 'submission'
                                  and e.get('final') and e['round'] == teams.round - 1]
                        answer = finals[-1]['answer'] if finals else None
                        details = {'metric': metric, 'population': teams.state()['agents']}
                    result = {'arm': arm, 'task_id': task.id, 'score': env.get_terminal_score(),
                              'has_final_answer': bool(answer), 'answer': answer,
                              'cost_usd': client.usage()['cost_usd'] - before, **details}
                    atomic_json(folder / 'result.json', result)
                    state['results'].append({k: result[k] for k in ('arm', 'task_id', 'score', 'has_final_answer', 'cost_usd')})
                    tick()
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {str(exc).replace(key, "[redacted]")}')
        if not isinstance(exc, CampaignStop):
            raise
    finally:
        write_replay(teams, teamdir, state['status'])
        tick()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument("--team-prompt-probe", action="store_true", help="One team task with a $1 ceiling")
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden): ')
    if not key.strip():
        raise SystemExit('Missing API key')
    options = dict(task_count=1, arms=("teams",), cap=1) if args.team_prompt_probe else {}
    run(args.root.resolve(), key.strip(), **options)
