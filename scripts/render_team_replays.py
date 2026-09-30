"""Render archived team interactions with the current viewer, without changing sources."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path


def render(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Use a separate destination outside the source archive")
    template = (Path(__file__).resolve().parents[1] / "hayekmas/adapters/teams/replay.html").read_text()
    destination.mkdir(parents=True, exist_ok=True)
    entries, rows = [], []
    for path in sorted(source.rglob("replay.json")):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("schema") != 1 or not payload.get("rounds"):
            continue
        relative = path.relative_to(source)
        target = destination / relative.with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        safe = raw.decode().replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        target.write_text(template.replace("__REPLAY_DATA__", safe))
        (destination / relative).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        entries.append({"source": str(path), "sha256": digest, "viewer": str(target.relative_to(destination))})
        count = sum(len(r["events"]) for r in payload["rounds"])
        label = str(relative.parent).replace("/attempt-1", "")
        rows.append(f'<tr><td><a href="{escape(relative.with_suffix(".html").as_posix(), quote=True)}">{escape(label)}</a></td><td>{count}</td></tr>')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during rendering: {path}")
    (destination / "manifest.json").write_text(json.dumps({"source": str(source), "viewers": entries}, indent=2))
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Team interaction archive</title><style>body{max-width:1050px;margin:36px auto;padding:0 20px;'
        'font:16px/1.6 system-ui;background:#0c1420;color:#ecf1f7}a{color:#8de4d2}table{width:100%;border-collapse:collapse}'
        'td,th{text-align:left;padding:10px;border-bottom:1px solid #29384a}</style>'
        '<h1>Team interaction archive</h1><p>Open a task to replay individual messages, bids, votes, and rewards. '
        'The transcript scrolls freely. These views use the original recorded events; archived results are unchanged.</p>'
        f'<p>{len(entries)} recorded interactions.</p><table><thead><tr><th>Task</th><th>Events</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table>'
    )
    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Rendered {render(args.source, args.destination)} archived interactions")
