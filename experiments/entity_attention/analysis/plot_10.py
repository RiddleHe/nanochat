"""Plot reviewed categories for every sliding window. Run after assemble_report.py."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
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
data=json.loads((ROOT/"category_by_span.json").read_text())
templates=[("direct_fact","Direct fact"),("person_in_list","Person in list"),("friend","Friend"),("visitor_register","Visitor register"),("name_badge","Name badge")]
categories=[("full_name","Full / expanded name","#2378ae"),("given_only","Given name only","#8dc9e8"),("surname_only","Surname only","#55a795"),("other_name_category","Other name category","#b49ace"),("different_name","Different name","#ed9a43"),("non_name","Non-name","#a7afb7"),("malformed","Malformed / prompt continuation","#c84557")]
fig,axes=plt.subplots(5,2,figsize=(16,15),sharey=True)
for i,(name,label) in enumerate(templates):
 for j,width in enumerate([4,6]):
  ax=axes[i,j];rs=sorted([r for r in data["span_results"] if r["template_name"]==name and r["width"]==width],key=lambda r:r["start_layer"])
  x=np.array([r["start_layer"] for r in rs]);bottom=np.zeros(len(rs))
  for key,lab,col in categories:
   values=np.array([100*r[key]/r["n"] for r in rs])
   ax.bar(x,values,bottom=bottom,width=.92,color=col,label=lab,linewidth=0)
   bottom+=values
  assert np.allclose(bottom,100)
  ax.set_title(f"{label} · width {width} · n={rs[0]['n']} per window",loc="left",fontsize=12)
  ax.set_ylim(0,100);ax.set_xlim(-.7,max(x)+.7);ax.set_xticks(list(range(0,max(x)+1,2)));ax.set_yticks([0,25,50,75,100])
  ax.set_xlabel("Window start layer (0-based)");ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
  if j==0:ax.set_ylabel("Outputs (%)")
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.965),ncol=4,frameon=False)
fig.suptitle("Entity-value blocking in final-prompt layer windows",fontsize=19,y=.995)
fig.text(.5,.970,"Intermediate prompt and generated queries blocked at all layers; final prompt query blocked only within each window.",ha="center",fontsize=11)
fig.text(.5,.015,"446 pairs selected from experiment 01. Each bar sums to 100%. Full includes prose and expansions; given includes prose. Categories pending user sign-off.",ha="center",fontsize=10)
fig.tight_layout(rect=[0,.035,1,.935],h_pad=2.1)
out=OUTPUT;out.mkdir(exist_ok=True)
for ext in ["png","pdf"]:fig.savefig(out/f"category_rates_by_layer_span.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)
print("Saved category_rates_by_layer_span.png and .pdf")
