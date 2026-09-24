"""Verify private result payload hashes and fixed experiment sizes. No model/GPU needed."""
from pathlib import Path
import argparse,hashlib,json
def verify(root):
 manifest=json.loads((root/"manifest.json").read_text())
 for record in manifest["files"]:
  p=root/record["path"]
  assert p.is_file(),p
  assert p.stat().st_size==record["bytes"],p
  assert hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],p
 expected={"01":1000,"02":92,"03":138,"04":394,"05":591,"06":1000,"07":1000,"08":1000,"09":1000,"10":28544}
 for n,count in expected.items():
  d=next(root.glob(n+"-*"));rows=[json.loads(x) for x in (d/"generations.jsonl").read_text().splitlines()]
  assert len(rows)==count,(n,len(rows),count)
  if n in ["06","07","08","09"]:
   assert len({(r["template_id"],r["entity_id"]) for r in rows})==500
   labels=[json.loads(x) for x in (d/"review/proposed_categories.jsonl").read_text().splitlines()]
   assert len(labels)==1000
  if n=="10":assert len((d/"controls.jsonl").read_text().splitlines())==892
 assert not any(root.glob("0[678]-*/full500")),"Export should contain only flattened full500"
 print(f'PASS: {len(manifest["files"])} hashed files; 01–10 row counts; full500 coverage.')
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("results",type=Path);a=p.parse_args();verify(a.results)
