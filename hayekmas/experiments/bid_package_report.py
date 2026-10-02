"""Package a completed local study with portable replays and cost evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from .bid_session_report import ROOT


def copy_selected(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    names = ('plan.json','status.json','summary.json','config.json','api_usage.jsonl',
             'statistics.json','selection-plan.json','selection-status.json','source_manifest.json',
             'behavior.json','behavior.html','index.html','replay.html')
    for name in names:
        if (source/name).exists():
            shutil.copy2(source/name,destination/name)
    results = list(source.glob('*/result.json')) + list(source.glob('*/sample-*.json')) + list(source.glob('*/selection.json'))
    for path in results:
        target=destination/path.parent.name
        target.mkdir(exist_ok=True)
        shutil.copy2(path,target/path.name)
    for path in source.glob('*/replay.html'):
        target=destination/path.parent.name
        target.mkdir(exist_ok=True)
        shutil.copy2(path,target/path.name)


def package(destination):
    analysis=json.loads((ROOT/'analysis.json').read_text())
    if any(t['status'] not in {'completed','stopped','deadline_reached'} for t in analysis['teams']):
        raise RuntimeError('Cannot publish final package while a team condition is running')
    for name,value in analysis['references'].items():
        if isinstance(value,dict) and 'usage' in value and value.get('status') not in {'completed','stopped','deadline_reached'}:
            raise RuntimeError(f'Reference still running: {name}')
    verification=json.loads((ROOT/'verification.json').read_text())
    if verification['issues'] or not verification['all_paid_sources_terminal']:
        raise RuntimeError('Final independent audit has not passed with all paid sources terminal')
    html=(ROOT/'report.html').read_text()
    if 'Final research report' not in html:
        raise RuntimeError('Render and verify the final report before packaging')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('analysis.json','plan.json','session_budget.json','research_journal.jsonl',
                 'summary.json','behavior.json','behavior.html','verification.json','physics-check.json','replication-status.json',
                 'replication-budget.json','matched-original-budget.json','regrade-expansion-budget.json'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,destination/name)
    shutil.copytree(ROOT/'figures',destination/'figures',dirs_exist_ok=True)
    shutil.copy2(ROOT/'index.html',destination/'primary.html')
    (destination/'overview.html').write_text((ROOT/'overview.html').read_text().replace('href="index.html"','href="primary.html"'))
    for cell in json.loads((ROOT/'plan.json').read_text())['cells']:
        copy_selected(ROOT/cell,destination/cell)
    aliases=('seed17','followup-feedback','followup-short','followup-membership','followup-membership-v2',
             'original-reference','original-high-reference','independent-baseline',
             'grader-repeats','grader-repeats-seed17','grader-repeats-original-high')
    for alias in aliases:
        source=ROOT/alias
        if not source.exists():continue
        target=destination/alias
        copy_selected(source,target)
        plan=json.loads((source/'plan.json').read_text()) if (source/'plan.json').exists() else {}
        for cell in plan.get('cells',{}):copy_selected(source/cell,target/cell)
    html=html.replace('href="index.html"','href="primary.html"')
    html=html.replace('Each condition directory contains provider receipts, exact request/response logs, per-task event logs, result JSON, checkpoints, and replay HTML.',
        'This portable bundle contains provider cost receipts, result JSON, and self-contained replay HTML. Exact request/response logs, raw event logs, and checkpoints remain in the original local runs directories. Replay data preserves the recorded conversation and economic events.')
    html=html.replace('Live session overview','Saved session overview')
    (destination/'index.html').write_text(html)
    (destination/'report.html').write_text(html)
    for path in destination.rglob('*'):
        if path.is_file() and path.suffix in {'.html','.json','.jsonl','.py','.txt'}:
            if re.search(r'sk-or-v1-[A-Za-z0-9]{20,}',path.read_text(errors='ignore')):
                raise RuntimeError(f'Credential pattern in package: {path}')
    for path in (destination/'figures').glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    manifest={str(path.relative_to(destination)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(destination.rglob('*')) if path.is_file() and path.name!='artifact-sha256.json'}
    (destination/'artifact-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination/'index.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('reports/k10-bid-study-20261001'))
    args=parser.parse_args()
    print(package(args.out))
