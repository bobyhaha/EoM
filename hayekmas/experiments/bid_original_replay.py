"""Read-only action replay for the original-engine references in the bid study."""
import argparse
import json
from pathlib import Path


def write(root):
    plan = json.loads((root/'plan.json').read_text())
    tasks = []
    for path in sorted(root.glob('*/result.json')):
        result = json.loads(path.read_text())
        checkpoint = json.loads((path.parent/'checkpoint.json').read_text())
        result['agents'] = [{'name':a['name'], 'role':a['type'].replace('ResearchAgent','').lower()}
                            for a in checkpoint['initial_population']]
        result['actions'] = [{'step':int(step), **action} for step,action in sorted(
            result['actions'].items(), key=lambda item:int(item[0]))]
        tasks.append(result)
    payload = json.dumps({'plan':plan,'tasks':tasks},ensure_ascii=False).replace('<','\\u003c')
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Original EoM action replay</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:36px auto;padding:0 24px;background:#f5f8fc;color:#182636}h1{line-height:1.2}select,button{padding:8px;margin:4px}input{width:100%}.card{padding:20px;margin:20px 0;background:white;border:1px solid #d8e3ee;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #d8e3ee;text-align:center}td:first-child,th:first-child{text-align:left}.acted{background:#3b83ac;color:white}.current{outline:3px solid #e49435}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.5 ui-monospace,monospace}summary{cursor:pointer;font-weight:650}details{margin:16px 0;border-left:3px solid #3b83ac;padding-left:16px}.muted{color:#52677a}</style></head>
<body><h1>Original EoM: executed actions over time</h1><p class="muted">Fresh k=10 population per task, five original roles × two. Evaluation freezes wealth and strategies. This replay shows executed public actions; wakeup and auction details remain in the local request ledger and trajectory log.</p>
<div class="card"><label>Task <select id="task"></select></label><p id="settings"></p><button id="previous">Previous action</button><button id="next">Next action</button><button id="all">Show complete trajectory</button><input id="step" type="range" min="0" value="0"><p id="position"></p><p id="outcome"></p></div>
<div class="card"><h2>Who acted</h2><p>Blue cells mark actions already executed at the selected point. Orange outline marks the latest action. Hover an agent label to see its full saved name.</p><table id="agents"></table></div>
<div class="card"><h2>Public solution history</h2><div id="history"></div></div>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const labels=['Physics','Biology','Chemistry'];
for(let i=0;i<data.tasks.length;i++){const option=document.createElement('option');option.value=i;option.textContent=labels[i]+' · '+data.tasks[i].task_id;$('task').append(option)}
function draw(){
 const task=data.tasks[Number($('task').value)];if(!task){$('position').textContent='No completed task yet.';return}
 const count=task.actions.length;$('step').max=count;const cursor=Math.min(Number($('step').value),count);$('step').value=cursor;
 $('settings').textContent='Solver: '+(data.plan.solver_tokens||8192)+' initial output-token cap / '+(data.plan.solver_reasoning||'medium')+' reasoning. Transport retries may expand the cap. Native early finalization is retained.';
 $('position').textContent=cursor+' / '+count+' actions revealed';$('outcome').textContent=cursor===count?'Final score: '+task.score+' · answer submitted: '+task.has_final_answer+' · API cost including grading: $'+task.cost_usd.toFixed(5):'Final outcome hidden until the complete trajectory is revealed.';
 $('previous').disabled=cursor===0;$('next').disabled=cursor===count;
 const table=$('agents');table.replaceChildren();const head=table.insertRow();let th=document.createElement('th');th.textContent='Agent / original role';head.append(th);
 for(const action of task.actions){th=document.createElement('th');th.textContent='Step '+action.step;head.append(th)}
 task.agents.forEach((agent,index)=>{const row=table.insertRow();const label=row.insertCell();label.textContent='#'+index+' '+agent.role;label.title=agent.name;task.actions.forEach((action,i)=>{const cell=row.insertCell();if(i<cursor&&action.author===agent.name){cell.textContent=action.is_final?'Final':'Act';cell.className='acted'+(i===cursor-1?' current':'')}})});
 const history=$('history');history.replaceChildren();for(const action of task.actions.slice(0,cursor)){const details=document.createElement('details');details.open=action.step===cursor;const summary=document.createElement('summary');summary.textContent='Step '+action.step+' · '+action.role+' · '+action.author+(action.is_final?' · final answer':'');const pre=document.createElement('pre');pre.textContent=action.text;details.append(summary,pre);history.append(details)}
}
$('task').onchange=()=>{$('step').value=0;draw()};$('step').oninput=draw;$('previous').onclick=()=>{$('step').value=Number($('step').value)-1;draw()};$('next').onclick=()=>{$('step').value=Number($('step').value)+1;draw()};$('all').onclick=()=>{$('step').value=$('step').max;draw()};draw();
</script></body></html>'''.replace('PAYLOAD',payload)
    (root/'replay.html').write_text(html)
    return root/'replay.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(write(args.root))
