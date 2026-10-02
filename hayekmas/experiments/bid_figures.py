"""Static publication-style summaries of membership, funding, scores and cost."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from .bid_behavior import records
from .bid_session_report import collect,ROOT


def plot():
    data=collect()
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    cells=['wealth-voluntary','society-voluntary','wealth-fixed','society-fixed']
    palette=['#e4e8ed','#4489ba','#e89f51','#75ae82','#b28fc2','#d5787b','#78b8b9','#c3b05c','#9b9fad','#73945f','#c18f71']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,cell in zip(axes.flat,cells):
        paths=sorted((ROOT/cell).glob('*/events.jsonl'))
        es=records(paths[0]) if paths else []
        grid=np.full((10,10),np.nan)
        for e in es:
            if e['event']=='membership_committed':
                for name,team in e['membership'].items():
                    grid[int(name.split('-')[-1]),e['step']]=int(team.split('-')[-1]) if team else 0
        ax.imshow(grid,aspect='auto',interpolation='none',cmap=ListedColormap(palette),norm=BoundaryNorm(np.arange(-.5,11.5),11))
        for e in es:
            if e['event']=='auction':
                for name,v in e['contributions'].items():
                    if v>0:ax.scatter(e['step'],int(name.split('-')[-1]),marker='o',s=24,facecolor='black')
                for name in e['members']:
                    ax.scatter(e['step'],int(name.split('-')[-1]),marker='s',s=120,facecolor='none',edgecolor='black',linewidth=1)
        ax.set_title(cell);ax.set_xticks(range(10),range(1,11));ax.set_yticks(range(10),[f'agent-{i}' for i in range(10)])
        ax.set_xlabel('Decision round');ax.set_ylabel('Agent')
    fig.suptitle('Physics task: team membership and funding\nGray = solo; colors = team IDs within each panel; dot = positive pledge; square = winning member',fontsize=12)
    fig.savefig(out/'physics-membership.png',dpi=160);fig.savefig(out/'physics-membership.svg');plt.close(fig)
    teams=data['teams'];labels=[x['label'] for x in teams]
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    means=[x['mean_completed_score'] if x['mean_completed_score'] is not None else float('nan') for x in teams]
    axes[0].barh(labels,means,color='#3887ab');axes[0].set_xlim(0,1);axes[0].set_xlabel('Mean score across completed tasks')
    for i,t in enumerate(teams):axes[0].text(min(means[i]+.015,.90) if np.isfinite(means[i]) else .015,i,f'n={t["n"]}/3',va='center',fontsize=9)
    costs=[x['usage']['cost_usd'] for x in teams]
    axes[1].barh(labels,costs,color='#9b7ac0');axes[1].set_xlabel('Confirmed API USD, including any partial episode')
    axes[0].invert_yaxis();axes[1].invert_yaxis();fig.suptitle('Development results — unequal compute; partial coverage explicitly labeled')
    fig.savefig(out/'scores-costs.png',dpi=160);fig.savefig(out/'scores-costs.svg');plt.close(fig)
    phases=[('Membership',['round_membership','round_join']),('Private solving',['round_chat']),('Ballots',['round_commit']),
            ('Winning work',['round_work','vote']),('Final answer',['finalize']),('Judge',['judge'])]
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True);left=np.zeros(len(teams))
    for label,keys in phases:
        vals=np.array([sum(t['phase_costs']['phases'].get(k,{}).get('cost_usd',0) for k in keys) for t in teams])
        ax.barh(labels,vals,left=left,label=label);left+=vals
    ax.invert_yaxis();ax.set_xlabel('Confirmed API USD');ax.set_title('Where the model budget went');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13))
    fig.savefig(out/'phase-costs.png',dpi=160);fig.savefig(out/'phase-costs.svg');plt.close(fig)
    return str(out)


if __name__=='__main__':print(plot())
