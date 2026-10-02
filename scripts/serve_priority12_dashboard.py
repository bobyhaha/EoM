"""Read-only live dashboard; never imports or changes experiment code."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

REPO = Path(__file__).resolve().parents[1]
KEEP = {'initialized', 'decision_round_started', 'membership_committed', 'membership_decision',
        'invited', 'joined', 'left', 'team_message', 'team_activation', 'auction', 'candidate',
        'submission', 'step_complete', 'settlement', 'public_work_finalization', 'path_reward', 'round_complete'}


def episode(root, cell, task):
    plan = json.loads((root / 'plan.json').read_text())
    if cell not in {j['name'] for j in plan['jobs']} or task not in plan['tasks']:
        raise ValueError('Unknown episode')
    folder = root / cell / f"{plan['tasks'].index(task)+1:02d}-{task}"
    events, problem = [], ''
    file = folder / 'events.jsonl'
    if file.exists():
        with file.open() as stream:
            for line in stream:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # an in-progress final append
                if not problem and e.get('event') == 'decision_request':
                    problem = e.get('observation', {}).get('task', {}).get('problem', '')
                if e.get('event') in KEEP:
                    events.append(e)
    result_file = folder / 'result.json'
    result = json.loads(result_file.read_text()) if result_file.exists() else None
    return {'events': events, 'problem': problem, 'result': result, 'original': cell.startswith('original-')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8791)
    args = parser.parse_args()
    root = args.root.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path in ('/', '/dashboard.html'):
                data = (REPO / 'dashboard/priority12.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
            elif url.path == '/api/episode':
                query = parse_qs(url.query)
                try:
                    payload = episode(root, query.get('cell', [''])[0], query.get('task', [''])[0])
                except ValueError as exc:
                    self.send_error(400, str(exc))
                    return
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
            else:
                return super().do_GET()
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
