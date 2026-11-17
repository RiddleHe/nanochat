"""Provenance and exact ScalarRes source selection for the vocabulary VO probe."""
from pathlib import Path
import hashlib,json
import torch

def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def hashes(root):return {str(p.relative_to(root)):sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
def select(state,config,candidates):
 sizes=[]
 for l in range(config['n_layer']):sizes.extend([2*l+1]*3+[2*l+2])
 sizes.append(2*config['n_layer']+1)
 offsets=[0]
 for n in sizes:offsets.append(offsets[-1]+n)
 z=state['scalar_res_logits_flat'][:offsets[-1]].float()
 root=torch.sqrt(1+z.square());g=torch.where(z>=0,root+z,(root-z.clamp_max(0)).reciprocal())
 applied=g.to(torch.bfloat16).float()
 assert torch.isfinite(applied).all() and (applied>0).all()
 rows=[]
 for l in candidates:
  lo,hi=offsets[4*l+2:4*l+4];v=applied[lo:hi]
  assert len(v)==2*l+1 and l>0
  j=int(torch.argmax(v[1:]))+1
  rows.append(dict(layer=l,embedding=float(v[0]),largest_other=float(v[j]),largest_other_index=j,largest_other_source=('A' if j%2 else 'M')+str((j-1)//2),selected=bool(v[0]>v[1:].max()),coefficients=v.tolist()))
 return rows
