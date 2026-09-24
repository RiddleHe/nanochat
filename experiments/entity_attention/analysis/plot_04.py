"""Render only blocked-minus-ordinary panels from validated saved means."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
_parser=argparse.ArgumentParser()
_parser.add_argument('--result-dir',type=Path,required=True)
_parser.add_argument('--out-dir',type=Path,required=True)
_args=_parser.parse_args()
ROOT=_args.result_dir.resolve()
OUTPUT=_args.out_dir.resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
summary_path=ROOT/'attention_summary.json'
validation=json.loads((ROOT/'validation.json').read_text())
assert hashlib.sha256(summary_path.read_bytes()).hexdigest()==validation['artifact_sha256']['attention_summary.json']
summary=json.loads(summary_path.read_text())
names=['direct_fact','person_in_list','friend','visitor_register','name_badge']
vmin,vmax=validation['difference_scale']
plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
fig,axes=plt.subplots(1,5,figsize=(27,7),layout='constrained',sharey=True)
for ax,name in zip(axes,names):
    entry=summary[name]
    data=np.array(entry['blocked_minus_ordinary'])
    assert len(entry['entity_columns'])==2
    im=ax.imshow(data,aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=vmin,vmax=vmax)
    ax.set_title(f'{name.replace("_"," ").title()} (n={entry["n"]})',fontsize=15,pad=12)
    labels=entry['token_labels']
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f'{i}: {token}' for i,token in enumerate(labels)],rotation=90,fontsize=9)
    ax.set_yticks([0,5,10,15,20,23,25,30,35])
    ax.set_xlabel('Prompt token')
    for col in entry['entity_columns']:
        ax.axvline(col-.5,color='#dfa81c',lw=.9)
        ax.axvline(col+.5,color='#dfa81c',lw=.9)
        label=ax.get_xticklabels()[col]
        label.set_color('#a00000');label.set_weight('bold')
axes[0].set_ylabel('Layer')
fig.colorbar(im,ax=axes,shrink=.88,pad=.012,label='Blocked − ordinary')
fig.suptitle('Two-token exact-name → given-name-only cases · final prompt query attention\nBlocked − ordinary · Blue: decrease; red: increase · NAME_1 and NAME_2 separate',fontsize=17)
paths=[]
for ext in ('png','pdf'):
    target=OUTPUT/f'attention_diff_1x5.{ext}'
    fig.savefig(target,dpi=180,bbox_inches='tight');paths.append(target)
plt.close(fig)
record={'source_summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),'difference_scale':[vmin,vmax],'templates':names,'display_name_slots':2,'artifact_sha256':{str(path.relative_to(OUTPUT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
(OUTPUT/'attention_diff_1x5.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
