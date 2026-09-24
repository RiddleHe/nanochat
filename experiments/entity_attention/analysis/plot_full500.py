"""Plot reviewed 06–09 full500 categories; never infer labels from answer text."""
from pathlib import Path
import argparse,collections,csv,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--results",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
groups=[
("Full / expanded name",{"full_supplied_name","full_supplied_name_in_prose","expanded_name","expanded_name_in_prose"},"#2378ae"),
("Given name only",{"first_name_only","given_name_in_prose"},"#8dc9e8"),
("Surname only",{"surname_only"},"#55a795"),
("Other same-person name",{"related_given_name_not_supplied","full_name_spelling_or_spacing_variant","shortened_name_or_initials","same_person_alternate_name"},"#b49ace"),
("Different name",{"altered_or_different_name"},"#ed9a43"),
("Non-name",{"other_response"},"#a7afb7"),
("Malformed / continuation",{"malformed_or_prompt_continuation"},"#c84557")]
templates=["direct_fact","person_in_list","friend","visitor_register","name_badge"]
conditions=[("06","No description"),("07","Correct description"),("08","Incorrect description"),("09","A human")]
allrows=[];table=[]
for n,label in conditions:
 d=next(a.results.glob(n+"-*"))
 rr=[json.loads(x) for x in (d/"review/proposed_categories.jsonl").read_text().splitlines()]
 selected=[r for r in rr if r["run_condition"]=="entity_blocked"]
 assert len(selected)==500 and len({(r["template_id"],r["entity_id"]) for r in selected})==500
 for r in selected:assert sum(r["proposed_category"] in members for _,members,_ in groups)==1
 allrows.extend(dict(r,experiment=n) for r in selected)
 for name in templates:
  cc=collections.Counter(r["proposed_category"] for r in selected if r["template_name"]==name);assert sum(cc.values())==100
  for title,members,_ in groups:
   table.append(dict(experiment=n,description=label,template=name,category=title,n=100,count=sum(cc[c] for c in members)))
with (a.out/"category_rates_full500.csv").open("w") as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
fig,axes=plt.subplots(1,5,figsize=(22,7),sharey=True)
for ax,name in zip(axes,templates):
 bottom=[0]*4
 for title,members,color in groups:
  values=[next(r["count"] for r in table if r["experiment"]==n and r["template"]==name and r["category"]==title) for n,_ in conditions]
  ax.bar(range(4),values,bottom=bottom,label=title,color=color)
  bottom=[x+y for x,y in zip(bottom,values)]
 assert bottom==[100]*4
 ax.set_xticks(range(4),[label for _,label in conditions],rotation=35,ha="right")
 ax.set_title(name.replace("_"," ").title());ax.set_ylim(0,100)
axes[0].set_ylabel("Outputs in category (%)")
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
fig.suptitle("Entity blocked, intermediate access preserved · all 100 names per template",fontsize=17)
fig.text(.5,.02,"Experiments 06–09 · full500 only · proposed categories pending user sign-off · each bar sums to 100%",ha="center")
fig.tight_layout(rect=[0,.13,1,.8])
for ext in ["png","pdf"]:fig.savefig(a.out/f"category_rates_full500.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)
print(a.out)
