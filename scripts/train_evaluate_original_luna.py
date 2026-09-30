"""Budgeted original-EoM training followed by frozen held-out evaluation."""
import argparse
import getpass
import hashlib
import html
import json
import os
from pathlib import Path
import random
import time

from hayekmas.adapters.researchworld.env import load_research_tasks
from hayekmas.adapters.researchworld.runtime import (
    ResearchTrainer, ResearchEvaluator, load_research_runtime_config,
    make_research_agent_deserializer,
)
from hayekmas.adapters.teams.openrouter import OpenRouterClient, SpendLimitExceeded
from hayekmas.base.mas import HayekMAS
from hayekmas.utils.logger import logger


class RunStopped(BaseException):
    """Escape upstream exception handlers: API errors must not become task failures."""


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def frozen_signature(mas):
    return [a.serialize() for a in mas.population.get_all()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/original-luna-trained-5usd')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    raw = json.loads(Path('global_configs/train_research.json').read_text())
    raw['profile'] = 'original_luna_5usd'
    raw['model'] = {'api': 'openrouter', 'name': 'openai/gpt-6-luna', 'max_tokens': 8192, 'temperature': None}
    raw['run'].update(output=str(out / 'trained_population.json'), num_epochs=2, verbose=False,
                      checkpoint_steps=str(out / 'training_steps'))
    raw['mas']['wakeup'].update(wakeup_model=None, wakeup_parallel_enabled=False)
    raw['mas']['evaluation']['periodic_test_enabled'] = False
    cfg = load_research_runtime_config(raw)
    train = load_research_tasks([Path(p) for p in cfg.data.train.files], limit=40)
    test = load_research_tasks([Path(p) for p in cfg.data.test.files])
    assert len(train) == 40 and len(test) == 19
    assert not ({t.id for t in train} & {t.id for t in test})
    assert not ({t.problem.strip() for t in train} & {t.problem.strip() for t in test})
    plan = {'model': raw['model'], 'training_tasks': len(train), 'epochs': 2, 'test_tasks': len(test),
            'total_allowance_usd': 5, 'training_allowance_usd': 3.5, 'seed': 7,
            'protocol': 'Original EoM training; frozen checkpoint reloaded separately for every held-out task.',
            'differences_from_paper': ['Luna instead of Gemini-3-Flash', 'Repository has 19 test tasks, paper reports 20',
                 'Repository has five initial roles including Answer; paper describes four partial roles',
                 '8192 output token cap, serial wakeups, no periodic test sweeps; final checkpoint evaluation only'],
            'dataset_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in cfg.data.train.files + cfg.data.test.files}}
    if args.validate_only:
        print(json.dumps(plan, indent=2)); return
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'plan.json', plan)
    write_json(out / 'config.json', raw)
    status = {'status': 'running', 'phase': 'training', 'training_completed': 0, 'training_target': 80,
              'test_completed': 0, 'test_target': len(test), 'started_at': time.time(), 'pid': os.getpid()}
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden): ')

    class GuardedClient(OpenRouterClient):
        def _generate_impl(self, *args, **kwargs):
            if (out / 'STOP').exists():
                raise RunStopped('User stop file exists')
            for attempt in range(3):
                try:
                    return super()._generate_impl(*args, **kwargs)
                except Exception as exc:
                    # Keep the failed reservation: a retry never refunds uncertain spend.
                    if 'HTTP 429;' in str(exc) and attempt < 2:
                        self.blocked = False
                        status['last_rate_limit'] = time.time()
                        update()
                        time.sleep(20 * (attempt + 1))
                        continue
                    raise RunStopped(str(exc)) from exc

    client = GuardedClient('openai/gpt-6-luna', budget={'max_usd': 3.5, 'prompt_usd_per_million': .15,
                           'completion_usd_per_million': .6}, api_key=key, temperature=None)
    del key
    client.default_max_tokens = 8192

    def update():
        status['usage'] = client.usage()
        status['updated_at'] = time.time()
        write_json(out / 'status.json', status)
        content = html.escape(json.dumps(status, indent=2))
        (out / 'progress.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20">'
            '<title>Original EoM · Luna training and evaluation</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:20px}'
            'pre{background:#f3f5f7;padding:24px;white-space:pre-wrap}</style><h1>Original EoM · Luna</h1>'
            '<p>40 training tasks, up to two epochs, then 19 held-out tasks. Total allowance: $5. Refreshes every 20 seconds.</p>'
            '<p><a href="plan.json">Protocol</a> · <a href="episodes.jsonl">Task outcomes</a> · <a href="api_usage.jsonl">API costs</a></p><pre>' + content + '</pre>')

    def sink(record):
        with (out / 'api_usage.jsonl').open('a') as f:
            f.write(json.dumps({**record, 'phase': status['phase'], 'task_id': status.get('task_id'), 'time': time.time()}) + '\n')
        update()
    client.sink = sink
    update()
    random.seed(7)

    def persist_episode(env, task, phase, index, success):
        directory = out / phase / f'{index:03d}_{task.id}'
        directory.mkdir(parents=True, exist_ok=True)
        result = {'index': index, 'task_id': task.id, 'phase': phase, 'score': env.get_terminal_score(),
                  'success': success, 'has_final_answer': bool(env.final_answer), 'steps': env.step_count,
                  'cumulative_cost_usd': client.usage()['reported_cost_usd']}
        write_json(directory / 'result.json', result)
        write_json(directory / 'actions.json', env.action_history)
        (directory / 'answer.md').write_text(env.final_answer or 'No final answer produced.\n')
        (directory / 'judge.txt').write_text(env._last_judge_reason or 'No judge result.\n')
        with (out / 'episodes.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        return result

    class ObservedTrainer(ResearchTrainer):
        def create_env(self, task):
            status['task_id'] = task.id
            status['active_task_started_at'] = time.time()
            update()
            return super().create_env(task)

        def check_success(self, env, task):
            success = super().check_success(env, task)
            persist_episode(env, task, 'training', status['training_completed'] + 1, success)
            return success

        def _save_progress(self, **kwargs):
            super()._save_progress(**kwargs)
            status['training_completed'] = kwargs['tasks_completed']
            status['training_outputs'] = str(self.outputs_dir.resolve())
            if status['training_completed'] % len(train) == 0:
                self.mas.save(str(out / f'epoch_{status["training_completed"] // len(train)}.json'))
            update()

    trainer = ObservedTrainer(client, cfg)
    try:
        try:
            summary = trainer.train(data_files=cfg.data.train.files, max_tasks=40)
            trainer.save()
            write_json(out / 'training_summary.json', summary)
        except RunStopped as exc:
            status['training_stop_reason'] = str(exc)
            # Evaluate only a completed full pass, never a partially updated episode.
            if not ('cost reservation exceeds' in str(exc) and status['training_completed'] >= 40):
                raise
        epochs = status['training_completed'] // 40
        if not epochs:
            raise RunStopped('No full training epoch completed; held-out evaluation not started')
        checkpoint = out / f'epoch_{epochs}.json'
        status.update(phase='evaluation', checkpoint=str(checkpoint), checkpoint_training_episodes=epochs * 40)
        client.budget['max_usd'] = 5
        client.blocked = False  # Only completed training / clean budget stop reaches here.
        cfg.run.checkpoint = str(checkpoint)
        evaluator = ResearchEvaluator(client, cfg)
        deserializer = make_research_agent_deserializer(client)
        outcomes = []
        for index, task in enumerate(test, 1):
            status.update(task_id=task.id, active_task_started_at=time.time())
            update()
            mas = HayekMAS.load(str(checkpoint), agent_deserializer=deserializer)
            mas.eval()
            evaluator.mas = mas
            before = frozen_signature(mas)
            env = evaluator.create_env(task)
            directory = out / 'evaluation' / f'{index:03d}_{task.id}'
            directory.mkdir(parents=True, exist_ok=True)
            with logger.scoped_task_log(directory / 'trajectory.log'):
                mas.run_one_episode(env, step_ckpt_save_path=str(directory / 'steps'))
            assert frozen_signature(mas) == before, 'Evaluation mutated frozen population'
            success = evaluator.check_success(env, task)
            outcomes.append(persist_episode(env, task, 'evaluation', index, success))
            status['test_completed'] = index
            update()
        summary = {'total': len(outcomes), 'passed': sum(r['success'] for r in outcomes),
                   'pass_threshold': cfg.judge.threshold, 'pass_rate': sum(r['success'] for r in outcomes) / len(outcomes),
                   'mean_rubric_score': sum(r['score'] or 0 for r in outcomes) / len(outcomes),
                   'ungraded_no_answer': sum(r['score'] is None for r in outcomes),
                   'checkpoint_training_episodes': epochs * 40, 'usage': client.usage()}
        write_json(out / 'evaluation_summary.json', summary)
        status.update(status='complete', summary=summary)
    except (RunStopped, Exception) as exc:
        status.update(status='stopped', error=str(exc))
    finally:
        update()
        logger.close()
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
