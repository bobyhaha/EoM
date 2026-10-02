"""Collect the k10 session without mixing incomplete episodes into score means."""
import argparse
from collections import Counter
from html import escape
import json
import math
from pathlib import Path
import statistics
import time

from .bid_behavior import analyze, records
from .campaign_client import atomic_json
from hayekmas.adapters.teams.token_allocation import phase_costs

RUNS = Path('runs')
ROOT = RUNS / 'k10-ablation-3h-20261001'


def independent_sensitivity(results):
    """Recompute oracle success at explicit rubric thresholds, including incomplete n."""
    rows = []
    for task_id in dict.fromkeys(r['task_id'] for r in results):
        scores = [r['score'] for r in results if r['task_id'] == task_id]
        n = len(scores)
        for threshold in (.5, .8, 1.):
            c = sum(score >= threshold for score in scores)
            rows.append({'task_id': task_id, 'threshold': threshold, 'n': n, 'c': c,
                         'pass_at_k': {str(k): None if k > n else
                                       1 - math.comb(n-c, k) / math.comb(n, k)
                                       for k in (1, 3, 5, 10)}})
    return rows


def grade_ranges(grades):
    grouped = {}
    for grade in grades:
        row = grouped.setdefault(grade['source'], {'source': grade['source'],
            'task_id': grade['task_id'], 'primary': grade['primary_score'], 'repeats': []})
        row['repeats'].append(grade['score'])
    for row in grouped.values():
        scores = [row['primary']] + [s for s in row['repeats'] if s is not None]
        row.update(min=min(scores), max=max(scores), span=max(scores)-min(scores))
    return list(grouped.values())


def collect_team(label, folder, rule):
    results = [json.loads(p.read_text()) for p in sorted(folder.glob('*/result.json'))]
    trajectories = [analyze(p.parent, rule) for p in sorted(folder.glob('*/events.jsonl'))]
    usage = {r['request']: r for r in records(folder / 'api_usage.jsonl')}
    state = json.loads((folder / 'status.json').read_text())
    auctions = funded = paid_slots = unpaid_slots = 0
    wins = Counter()
    sizes = Counter()
    quotes = []
    for path in sorted(folder.glob('*/events.jsonl')):
        for e in records(path):
            if e['event'] == 'auction':
                auctions += 1
                if e['winner'] is not None:
                    funded += 1
                    sizes[len(e['members'])] += 1
                    for name in e['members']:
                        wins[name] += 1
                        paid_slots += e['contributions'][name] > 0
                        unpaid_slots += e['contributions'][name] == 0
            if e['event'] == 'decision_response' and e.get('phase') == 'round_commit':
                try:
                    b = json.loads(e['response'])
                except (ValueError, TypeError):
                    continue
                reason = str(b.get('reason', ''))
                if any(s in reason.lower() for s in ('rely', 'zero', 'preserv', 'fund', 'society')):
                    quotes.append({'task':path.parent.name,'step':e['step'],'agent':e['agent'],**b})
    return {'label': label, 'folder':str(folder), 'status':state['status'], 'n':len(results),
            'scores':[r['score'] for r in results],
            'mean_completed_score': statistics.mean(r['score'] for r in results) if results else None,
            'submitted':sum(r['has_final_answer'] for r in results),
            'passed':sum(r['score'] >= .5 for r in results), 'usage':state['usage'],
            'auctions':auctions,'funded':funded,'funded_fraction':funded/auctions if auctions else None,
            'winning_size_counts':dict(sizes), 'paid_winner_slots':paid_slots,'unpaid_winner_slots':unpaid_slots,
            'joins':sum(t['joins'] for t in trajectories), 'leaves':sum(t['leaves'] for t in trajectories),
            'verification_issues':[x for t in trajectories for x in t['verification_issues']],
            'phase_costs':phase_costs(list(usage.values())), 'representative_ballot_candidates':quotes,
            'tasks':[{k:r[k] for k in ('task_id','score','has_final_answer','cost_usd')} for r in results]}


def collect():
    plan = json.loads((ROOT/'plan.json').read_text())
    teams = [collect_team(cell, ROOT/cell, rule) for cell,(_,rule) in plan['cells'].items()]
    for label, folder, rule in [
        ('wealth-voluntary-feedback',RUNS/'k10-feedback-3h-20261001/wealth-voluntary-feedback','voluntary'),
        ('society-fixed-3rounds',RUNS/'k10-short-rounds-20261001/society-fixed-3rounds','fixed')]:
        if (folder/'status.json').exists():teams.append(collect_team(label,folder,rule))
    repeatroot=RUNS/'k10-ablation-seed17-20261001'
    if (repeatroot/'plan.json').exists():
        for cell,(_,rule) in json.loads((repeatroot/'plan.json').read_text())['cells'].items():
            if (repeatroot/cell/'status.json').exists():teams.append(collect_team(cell+' / seed17',repeatroot/cell,rule))
    extra={}
    for label,name in [('original','k10-original-reference-20261001'),('independent','k10-independent-20261001'),
                       ('regrades','k10-regrade-20261001'),('society_pilot','k10-society-prompt-20261001'),
                       ('stopped_pilot','k10-pilot-20261001')]:
        path=RUNS/name/'status.json'
        if path.exists():extra[label]=json.loads(path.read_text())
    stats=RUNS/'k10-independent-20261001/statistics.json'
    if stats.exists():extra['independent_statistics']=json.loads(stats.read_text())
    extra['independent_sensitivity'] = independent_sensitivity(extra.get('independent', {}).get('results', []))
    extra['grade_ranges'] = grade_ranges(extra.get('regrades', {}).get('grades', []))
    receipts=[t['usage'] for t in teams]+[v['usage'] for k,v in extra.items() if isinstance(v,dict) and 'usage' in v]
    data={'generated_at':time.time(),'deadline':plan['deadline'],'teams':teams,'references':extra,
          'total_confirmed_usd':sum(v['cost_usd'] for v in receipts),
          'total_reserved_usd':sum(v['unconfirmed_usd'] for v in receipts)}
    atomic_json(ROOT/'analysis.json',data)
    return data


def render(data, final=False, discussion=''):
    def fmt(x):return f'{x:.3f}' if isinstance(x,(int,float)) else str(x)
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    html='''<!doctype html><html><head><meta charset="utf-8"><title>k10 ablation report</title><style>
body{font:16px/1.6 system-ui;max-width:1150px;margin:45px auto;padding:0 25px;color:#172537;background:#fafbfd}h1{font-size:36px;line-height:1.2}h2{margin-top:38px}table{border-collapse:collapse;font-size:14px;width:100%;background:white}td,th{padding:10px;border:1px solid #dbe3ed;text-align:left}th{background:#edf2f8}a{color:#155bb3}pre{white-space:pre-wrap;background:#edf2f8;padding:15px}.note{border-left:4px solid #4589bf;padding-left:16px}blockquote{border-left:3px solid #bbb;padding-left:18px}img{max-width:100%}</style></head><body>'''
    html+='<h1>k=10: objective prompts, team funding, and solution quality</h1>'
    html+='<p class="note">'+('Final research report' if final else 'Interim analysis — runs may still be in progress')+'</p>'
    html+=f'<p>Total confirmed session spending: <strong>${data["total_confirmed_usd"]:.4f}</strong>; unresolved reservations: <strong>${data["total_reserved_usd"]:.4f}</strong>. Maximum authorized: $100. This total includes the stopped pilot and standalone society-prompt pilot, but excludes historical research campaigns.</p>'
    html+='<h2>Study design</h2><p>The primary experiment crosses two objective prompts (individual wealth versus society success) with two bidding packages (voluntary personal amounts versus a fixed 0.1 total team bid). Each task starts with ten fresh, role-free agents; maximum team size four, ten decision rounds, λ=0, no cross-task learning, reflection, or population evolution. Scheduling seed 7 is common across conditions; model samples are stochastic and not provider-seed controlled. Additional seed17 results, if present, are reported separately.</p>'
    html+='<p>The tasks are the first three bundled Frontier Science research training tasks, chosen before these runs: lattice nucleon spectrum (physics), optogenetic signaling design (biology), and paclitaxel micelle formulation (chemistry). These are development tasks, not a fresh held-out set. This study does not replace training populations across all 40 tasks and evaluating the full 19-task test set.</p>'
    html+='<p>Each round permits membership proposals and replies, private problem discussion, a team activation ballot, an auction, and public work by the winner. Finalization occurs at the fixed horizon. Every historical member of each accepted contributing round receives the full R/N at episode end; rewards are not divided by team size. Bid payments go to the previous winning team, except the first winning bid is burned. Virtual wealth and bids are not API dollars.</p>'
    html+='<h2>What the base-bid variant changes</h2><p>The fixed variant is more than a numerical minimum. An agent explicitly returns act=true and authorize_base_bid=true to authorize a maximum personal charge of 0.1 × bid_cost_rate if its team wins. At least a strict majority must consent and be able to afford that maximum; the actual 0.1 bid is split equally among these consenting members. Others pay zero. Eligible teams tie at 0.1 and a seeded lottery selects the winner. The voluntary variant allows yes votes with zero pledges and selects the largest positive sum. Thus the ablation tests a fixed-price consent package, not the causal effect of price alone.</p>'
    html+='<h2>Team results</h2><p>Scores are native model-judge scores in [0,1]. Pass means score ≥0.5. Means use completed tasks only; interrupted episodes are not silently counted as zero. A completed episode with no answer legitimately receives zero. Funding totals include completed auctions from any currently partial episode.</p>'
    rows=[]
    for t in data['teams']:
        rows.append([t['label'],t['status'],str(t['n'])+'/3',', '.join(fmt(x) for x in t['scores']),fmt(t['mean_completed_score']),str(t['submitted'])+'/'+str(t['n']),str(t['passed'])+'/'+str(t['n']),str(t['funded'])+'/'+str(t['auctions']),f'${t["usage"]["cost_usd"]:.4f}'])
    html+=table(['Condition','Status','Tasks','Scores P / B / C','Mean','Answers','Passes','Funded auctions','USD'],rows)
    html+='<h2>Original EoM and independent samples</h2><p>The original reference uses the unchanged HayekMAS engine and five upstream research roles instantiated twice, with fresh untrained populations in evaluation mode. Native individual fixed bids, wakeups, and early finalization remain. It uses 8192 output tokens with medium reasoning versus 16384/high for team work, so it is a reference, not a compute-matched causal comparison. Its economics are frozen in evaluation, whereas team economics operate within each disposable episode.</p>'
    original=data['references'].get('original',{})
    html+=table(['Task','Score','Steps','USD'],[[r['task_id'],fmt(r['score']),r['steps'],f'${r["cost_usd"]:.4f}'] for r in original.get('results',[])])
    html+='<p>The independent baseline generates ten separate answers per task at 16384/high, without communication or access to other answers. Sample1 is the prespecified single-agent result. Pass@k is 1 − C(n−c,k)/C(n,k), where c is the number of judge-passing samples among n completed samples. This is an oracle success estimate, not a working selector, and judge calls are included in cost.</p>'
    stats=data['references'].get('independent_statistics',[])
    html+=table(['Task','Samples','First-sample score','Mean score','Best score','Pass@1','Pass@3','Pass@5','Pass@10'],[[r['task_id'],r['n'],fmt(r['single_agent_first_score']),fmt(r['mean_score']),fmt(r['best_score'])]+[fmt(r['pass_at_k'].get(str(k),r['pass_at_k'].get(k))) for k in (1,3,5,10)] for r in stats])
    html+='<p>A score of 0.5 represents partial rubric credit, not a fully correct scientific solution. The stricter thresholds below help avoid equating this operational pass criterion with correctness.</p>'
    html+=table(['Task','Threshold','Passing / samples','Pass@1','Pass@3','Pass@5','Pass@10'],[[r['task_id'],r['threshold'],f'{r["c"]}/{r["n"]}']+[fmt(r['pass_at_k'][str(k)]) for k in (1,3,5,10)] for r in data['references'].get('independent_sensitivity',[])])
    html+='<h3>Choosing an answer without the reference</h3><p>A separate selector sees the problem and ten shuffled candidate answers, but no rubric or scores. Its chosen answer keeps its previously recorded grade. This is a deployable selection procedure, unlike oracle best-of-ten, although it is not a fresh efficacy test.</p>'
    html+=table(['Task','Selected sample','Previously recorded score','Selection reason'],[[r['task_id'],r.get('selected_sample'),fmt(r.get('score')),r.get('reason')] for r in data['references'].get('independent',{}).get('selected_results',[])])
    html+='<h3>Repeated scoring of unchanged answers</h3><p>Each available primary, feedback, and original-reference answer is graded twice more with the same grading prompt and settings. Primary grades stay unchanged. Ranges describe grading variation, not confidence intervals; zero-answer cases require no judge call.</p>'
    html+=table(['Condition / task','Primary','Repeat scores','Range'],[[' / '.join(Path(r['source']).parts[-3:-1]),fmt(r['primary']),', '.join(fmt(s) for s in r['repeats']),f'{r["min"]:.3f}–{r["max"]:.3f}'] for r in data['references'].get('grade_ranges',[])])
    html+='<h2>Agent behavior and economic verification</h2><img src="figures/physics-membership.png" alt="Physics membership and funding by agent and decision round">'
    html+=table(['Condition','Joins','Leaves','Winning team sizes: rounds','Paying / nonpaying winner slots','Verification issues'],[[t['label'],t['joins'],t['leaves'],t['winning_size_counts'],f'{t["paid_winner_slots"]} / {t["unpaid_winner_slots"]}',len(t['verification_issues'])] for t in data['teams']])
    html+='<p>A nonpaying member can still contribute intellectual work; this count alone is not evidence of shirking. Membership counts describe behavior in this protocol, not proof that the resulting teams are optimal. All agents share the same model and initial strategy; there are no assigned specialist roles in team conditions.</p>'
    html+='<p>Independent event checks verify bid sums, majority activation, explicit fixed-price consent, winner eligibility, per-agent balance changes, transfers to historical previous winners, and R/N reward credits to the membership saved at submission. The mechanism passed 142 team tests at launch, including new consent and affordability checks. Repeated grades test the same model’s scoring variability; they are not independent expert verification.</p>'
    html+='<h2>Where inference cost went</h2><img src="figures/phase-costs.png" alt="Inference cost by phase">'
    html+=table(['Condition','Requests','Problem-solving share of billed cost','Pending reservation'],[[t['label'],t['usage']['requests'],fmt(t['phase_costs']['solving_share_of_confirmed_cost']),f'${t["usage"]["unconfirmed_usd"]:.5f}'] for t in data['teams']])
    html+='<p>Problem-solving includes private discussion, winner work, and final-answer generation. A high share does not guarantee useful public progress: unselected private reasoning still costs tokens. Budget ceilings are safety limits, not equal-spend constraints; the actual compute differs across conditions.</p>'
    html+='<h2>Findings, failures, and recommendations</h2>'+discussion
    html+='<h2>Limits of the evidence</h2><ul><li>Only three previously available development tasks; no population training, no fresh held-out efficacy claim.</li><li>Few scheduling seeds, stochastic provider outputs, and the same model used to solve and judge. Repeat grades cannot establish scientific correctness.</li><li>The fixed-price package changes both amount selection and who authorizes payment. It does not isolate the numeric effect of a bid floor.</li><li>Fresh populations and rent=0 do not test long-term bankruptcy selection or the k-to-2k growth policy.</li><li>Early pilots, the feedback experiment, and the three-round experiment have different designs and are not pooled into the primary factorial means.</li></ul>'
    html+='<h2>Artifacts and reproducibility</h2><ul><li><a href="overview.html">Live session overview</a></li><li><a href="index.html">Primary progress and task replays</a></li><li><a href="behavior.html">Behavior audit</a></li><li><a href="analysis.json">Complete numeric analysis and evidence candidates</a></li><li><a href="plan.json">Primary protocol and source hashes</a></li><li><a href="session_budget.json">Budget partitions</a></li><li><a href="research_journal.jsonl">Timestamped research decisions</a></li><li><a href="grader-repeats/status.json">Separate repeat grades</a></li></ul><p>Each condition directory contains provider receipts, exact request/response logs, per-task event logs, result JSON, checkpoints, and replay HTML. Primary and feedback/short-round source snapshots were saved and hash verified. Credentials are not stored in those artifacts.</p></body></html>'
    (ROOT/'report.html').write_text(html)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--discussion', type=Path)
    args = parser.parse_args()
    discussion = args.discussion.read_text() if args.discussion else '<p>Research is in progress. Final interpretation will be added after the declared research window.</p>'
    render(collect(), final=args.final, discussion=discussion)
