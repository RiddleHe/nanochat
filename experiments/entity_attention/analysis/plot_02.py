"""Render a 2x3 comparison from saved summaries, without model inference."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
_parser=argparse.ArgumentParser()
_parser.add_argument('--result-dir',type=Path,required=True)
_parser.add_argument('--out-dir',type=Path,required=True)
_args=_parser.parse_args()
ROOT=_args.result_dir.resolve()
OUTPUT=_args.out_dir.resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
summary=json.loads((ROOT/'attention_summary.json').read_text())
names=['person_in_list','friend']
matrices={}
for name in names:
 s=summary[name]
 ordinary=np.array(s['ordinary']);blocked=np.array(s['blocked'])
 difference=blocked-ordinary
 np.testing.assert_allclose(difference,-np.array(s['ordinary_minus_blocked']),rtol=2e-5,atol=2e-7)
 matrices[name]=(ordinary,blocked,difference)
vmax=max(m.max() for trio in matrices.values() for m in trio[:2])
dmax=max(np.abs(trio[2]).max() for trio in matrices.values())
plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(26,11),layout='constrained')
for row,name in enumerate(names):
 s=summary[name];labels=s['token_labels'];entity=s['entity_column']
 for col,(title,matrix) in enumerate(zip(['Ordinary','Blocked','Blocked − ordinary'],matrices[name])):
  ax=axes[row,col]
  im=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap='RdBu_r' if col==2 else 'viridis',vmin=-dmax if col==2 else 0,vmax=dmax if col==2 else vmax)
  ax.set_title(f'{name.replace("_"," ").title()} (n={s["n"]})\n{title}',fontsize=14,pad=10)
  ax.set_xticks(range(len(labels)))
  ax.set_xticklabels([f'{i}: {t}' for i,t in enumerate(labels)],rotation=90,fontsize=9)
  ax.set_yticks([0,5,10,15,20,23,25,30,35]);ax.set_ylabel('Layer')
  ax.set_xlabel('Prompt token')
  ax.axvline(entity-.5,color='#e8ac17',lw=1.1);ax.axvline(entity+.5,color='#e8ac17',lw=1.1)
  ax.get_xticklabels()[entity].set_weight('bold');ax.get_xticklabels()[entity].set_color('#a00000')
  if col==0:attention_image=im
  if col==2:difference_image=im
fig.colorbar(attention_image,ax=axes[:,:2],shrink=.8,pad=.02,label='Mean across cases of max-head attention')
fig.colorbar(difference_image,ax=axes[:,2],shrink=.8,pad=.02,label='Blocked − ordinary')
fig.suptitle('Final prompt query attention • selected one-token different-name / non-name cases\nDifference: blue = decreased under blocking; red = increased',fontsize=18)
plots=OUTPUT;plots.mkdir(exist_ok=True)
outputs=[]
for ext in ('png','pdf'):
 p=plots/f'attention_comparison_2x3.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight');outputs.append(p)
plt.close(fig)
provenance={'layout':'2 rows: Person in list (11), Friend (35); 3 columns: ordinary, blocked, blocked minus ordinary','difference_sign':'blocked minus ordinary','model_inference_rerun':False,'source_summary_sha256':hashlib.sha256((ROOT/'attention_summary.json').read_bytes()).hexdigest(),'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'difference_checked_against_previous_summary':True,'shared_attention_scale':[0,float(vmax)],'shared_difference_scale':[-float(dmax),float(dmax)],'figures_sha256':{str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
(OUTPUT/'grid_rendering.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
