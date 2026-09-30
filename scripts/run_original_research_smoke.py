"""One original EoM training episode, with metered OpenRouter transport only."""
import argparse
import getpass
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

from hayekmas.adapters.researchworld.runtime import (
    ResearchTrainer, create_research_agents, create_research_env,
    load_research_runtime_config,
)
from hayekmas.adapters.researchworld.env import load_research_tasks
from hayekmas.adapters.teams.openrouter import OpenRouterClient
from hayekmas.base.mas import HayekMAS
from hayekmas.utils.logger import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/original-eom-physics-1')
    parser.add_argument('--model', default='qwen/qwen3-30b-a3b-instruct-2507')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    raw = json.loads(Path('global_configs/train_research.json').read_text())
    raw['mas']['wakeup'].update(wakeup_model=None, wakeup_parallel_enabled=False)
    raw['mas']['evaluation']['periodic_test_enabled'] = False
    raw['run']['num_epochs'] = 1
    temperature = None if args.model.startswith('openai/gpt-6') else 0.9
    raw['model'] = {'api': 'openrouter', 'name': args.model, 'max_tokens': 8192, 'temperature': temperature}
    cfg = load_research_runtime_config(raw)
    budget = {'max_usd': 2, 'prompt_usd_per_million': 0.15, 'completion_usd_per_million': 0.6}
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden): ')
    class ReportingClient(OpenRouterClient):
        last_error = None
        def _generate_impl(self, *args, **kwargs):
            try:
                return super()._generate_impl(*args, **kwargs)
            except Exception as exc:
                if self.last_error is None:
                    self.last_error = str(exc)
                    (out / 'api_error.txt').write_text(self.last_error)
                raise
    client = ReportingClient(raw['model']['name'], budget=budget, api_key=key, temperature=temperature)
    del key
    client.default_max_tokens = 8192
    def sink(record):
        with (out / 'api_usage.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
    client.sink = sink
    random.seed(7)
    task = load_research_tasks([Path(cfg.data.train.files[0])], limit=1)[0]
    (out / 'task.json').write_text(json.dumps(asdict(task), indent=2))
    (out / 'config.json').write_text(json.dumps({'runtime': asdict(cfg), 'budget': budget, 'seed': 7, 'note': 'Original engine/roles/prompts/environment; fresh population, one training episode; OpenRouter transport, 8192 output cap, same model for wakeup and judge, serial wakeups.'}, indent=2))
    logger.configure(verbose=False, log_dir=str(out), profile='original_eom')
    mas = HayekMAS(cfg.mas)
    agents = create_research_agents(client)
    for agent in agents:
        agent.initialize(initial_wealth=cfg.mas.engine.initial_wealth)
        mas.population.add_agent(agent)
    mas._initial_agents = list(agents)
    trainer = ResearchTrainer(client, cfg)
    mas.set_agent_factory(trainer.create_agent_factory_good_birth(), trainer.create_agent_factory_bad_birth())
    env = create_research_env(task, llm_client=client, reward_config=cfg.mas.reward, use_judge=True, judge_threshold=cfg.judge.threshold, max_steps=cfg.mas.engine.max_steps_per_episode)
    status, error = 'complete', None
    try:
        mas.train().run_one_episode(env, step_ckpt_save_path=str(out / 'steps.json'))
    except Exception as exc:
        status, error = 'failed', str(exc)
    finally:
        if client.blocked or not env.final_answer or env.get_terminal_score() is None:
            status = 'incomplete'
            error = client.last_error or error or 'No final graded answer.'
        result = {'status': status, 'error': error, 'task_id': task.id, 'model': client.model, 'score': env.get_terminal_score(), 'passed': env.is_successful(), 'steps': env.step_count, 'usage': client.usage(), 'metrics': mas.last_episode_metrics}
        (out / 'result.json').write_text(json.dumps(result, indent=2, default=str))
        (out / 'answer.md').write_text(env.final_answer or 'No final answer produced.\n')
        (out / 'judge.txt').write_text(env._last_judge_reason or 'No judge result.\n')
        (out / 'actions.json').write_text(json.dumps(env.action_history, indent=2, default=str))
        mas.save(str(out / 'population.json'))
        logger.close()
        print(json.dumps({k:v for k,v in result.items() if k != 'metrics'}, indent=2))
    if status != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
