from pathlib import Path
import importlib.util,json
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("portable_runner",ROOT/"run.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.mark.parametrize("n,count",[(1,1000),(2,92),(3,138),(4,394),(5,591),(6,1000),(7,1000),(8,1000),(9,1000),(10,29436)])
def test_cohort_and_interventions(n,count):
 c=json.loads((ROOT/"configs"/f"{n:02}.json").read_text())
 jobs=m.tasks(c,Path("/tmp/new-run"),"MODEL")
 assert sum(x["expected"] for x in jobs)==count
 for job in jobs:
  cmd=job["command"];condition=job["condition"]
  assert cmd[cmd.index("--entities-json")+1].endswith("/data/entities100.json")
  assert ("--block-intermediate-prompt-entity" in cmd)==(condition in ["blocked","signed_restored","sweep"])
  if condition=="signed_restored":
   assert cmd[cmd.index("--restore-entity-attention-policy")+1]=="signed"
   assert cmd[cmd.index("--restore-entity-attention-start-layer")+1]=="0"
  if condition=="entity_blocked":assert cmd[cmd.index("--widths")+1]=="36"
  if condition=="sweep":assert cmd[cmd.index("--widths")+1]=="4,6"
 if n in [6,7,8,9]:assert all(g["entity_ids"]==list(range(100)) for g in c["templates"])
 if n==10:assert [x["condition"] for x in jobs[:10]].count("sweep")==0
def test_current_entities_and_templates():
 import hashlib
 assert hashlib.sha256((ROOT/"data/entities100.json").read_bytes()).hexdigest()=="354ce3a954251ab5c596cdbd1ffade227d144ed8d81a8704d6790dc2cfd28e99"
 assert {x["template_id"] for x in json.loads((ROOT/"data/templates.json").read_text())}=={0,1,2,3,7}
