"""Small provenance helpers shared by frozen PCA stages."""
from pathlib import Path
import hashlib,json

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def hashes(root):return {str(p.relative_to(root)):sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
