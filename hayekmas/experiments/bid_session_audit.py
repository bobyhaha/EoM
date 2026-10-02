"""Independent file-level completion, protocol and ledger audit for the k10 session."""
import ast
import hashlib
import json
import math
from pathlib import Path
import time

from .bid_behavior import analyze, records
from .bid_session_report import ROOT, collect
from .campaign_client import atomic_json


def audit():
    data=collect()
    issues=[]
    sources={Path(t['folder']):t['usage'] for t in data['teams']}
    sources.update({Path(v['ledger_source']):v['usage'] for v in data['references'].values()
                    if isinstance(v,dict) and 'usage' in v})
    ledger_rows=[]
    bound=0
    for source in sources:
        state=json.loads((source/'status.json').read_text())
        latest=list({r['request']:r for r in records(source/'api_usage.jsonl')}.values())
        confirmed=math.fsum(r.get('cost_usd',0) for r in latest)
        unresolved=math.fsum(r.get('reserved_usd',0) for r in latest if 'cost_usd' not in r)
        terminal=state['status'] in {'completed','stopped','deadline_reached'}
        limit=state['usage']['cap_usd']
        if terminal and not math.isclose(confirmed,state['usage']['cost_usd'],abs_tol=1e-9):
            issues.append(f'{source}: terminal cost differs from receipt ledger')
        if terminal and not math.isclose(unresolved,state['usage']['unconfirmed_usd'],abs_tol=1e-9):
            issues.append(f'{source}: terminal reservation differs from ledger')
        if confirmed+unresolved>limit+1e-9:
            issues.append(f'{source}: per-ledger ceiling exceeded')
        if any(r.get('time',0)>data['deadline'] for r in latest):
            issues.append(f'{source}: request initiated after authorized deadline')
        bound+=confirmed+unresolved if terminal else limit
        ledger_rows.append({'source':str(source),'status':state['status'],'terminal':terminal,
                            'confirmed_usd':confirmed,'unresolved_usd':unresolved,'cap_usd':limit,
                            'requests':len(latest),'models':sorted({r['model'] for r in latest if r.get('model')}),
                            'providers':sorted({r['provider'] for r in latest if r.get('provider')})})
    if bound>100+1e-9:issues.append('Aggregate conservative budget bound exceeds $100')
    primary=json.loads((ROOT/'plan.json').read_text())
    task_checks=[]
    for team in data['teams']:
        rule='fixed' if 'fixed' in team['label'] else 'voluntary'
        for path in sorted(Path(team['folder']).glob('*/events.jsonl')):
            events=records(path)
            if not events:continue
            first=events[0];roster=first['roster'];config=first['config']
            if len(roster)!=10 or len({r['name'] for r in roster})!=10:
                issues.append(f'{path}: population is not ten unique initial agents')
            if any(r['wealth']!=20 or r['team'] is not None for r in roster):
                issues.append(f'{path}: initial wealth or membership differs from fresh-population protocol')
            if config['evolution_enabled'] or config['coordination_fee_lambda']!=0:
                issues.append(f'{path}: unexpected evolution or coordination fee')
            starts=[e['step'] for e in events if e['event']=='decision_round_started']
            if starts!=list(range(len(starts))) or len(starts)>config['max_steps']:
                issues.append(f'{path}: round sequence or limit violated')
            for event in events:
                if event['event']=='auction' and any(value>event['opening_wealth'][name]+1e-9
                                                      for name,value in event['contributions'].items()):
                    issues.append(f'{path}: personal pledge exceeds opening wealth')
            check=analyze(path.parent,rule)
            issues.extend(f'{path}: {issue}' for issue in check['verification_issues'])
            if check['complete']:
                result=json.loads((path.parent/'result.json').read_text())
                if result['task_id'] not in primary['tasks']:
                    issues.append(f'{path}: unexpected task ID')
                if not result['has_final_answer'] and result['score']!=0:
                    issues.append(f'{path}: nonzero score without an answer')
            task_checks.append({'condition':team['label'],'task':path.parent.name,'complete':check['complete'],
                                'rounds_started':len(starts),'funded':check['funded'],'auctions':check['auctions'],
                                'economic_issues':check['verification_issues']})
    critical=[Path('hayekmas/adapters/teams')/(name+'.py') for name in
              ('agent','compact_rounds','config','env','mas','policy','round_protocol','team','token_allocation')]
    critical += [Path('hayekmas/experiments/k10_pilot.py'),Path('hayekmas/experiments/campaign_client.py'),Path('hayekmas/utils/llm.py')]
    critical += list(Path('hayekmas/base').glob('*.py'))+list(Path('hayekmas/adapters/researchworld').glob('*.py'))
    for root in (ROOT,Path('runs/k10-ablation-seed17-20261001')):
        plan=json.loads((root/'plan.json').read_text())
        if hashlib.sha256(Path(plan['dataset']).read_bytes()).hexdigest()!=plan['dataset_sha256']:
            issues.append(f'{root}: dataset changed')
        for path in critical:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=plan['source_sha256'][str(path)]:
                issues.append(f'{root}: model-facing source changed: {path}')
        worker=lambda path: ast.dump(next(n for n in ast.parse(path.read_text()).body
                                        if isinstance(n,ast.FunctionDef) and n.name=='worker'),include_attributes=False)
        path=Path('hayekmas/experiments/bid_ablation.py')
        if worker(path)!=worker(root/'source'/path):issues.append(f'{root}: worker body differs from saved source')
    output={'checked_at':time.time(),'all_paid_sources_terminal':all(r['terminal'] for r in ledger_rows),
            'aggregate_conservative_bound_usd':bound,'ledger_checks':ledger_rows,'task_checks':task_checks,
            'issues':issues,'scope':'Independent receipt/status reconciliation, fresh populations, horizons, economic events, dataset/source fingerprints. Does not certify scientific answer correctness or replace manual report review.'}
    atomic_json(ROOT/'verification.json',output)
    return output


if __name__=='__main__':
    result=audit()
    print(json.dumps({k:result[k] for k in ('all_paid_sources_terminal','aggregate_conservative_bound_usd','issues')},indent=2))
