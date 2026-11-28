"""Record/verify a source, tokenizer, corpus inventory and held-out-shard manifest.

Training corpus identity is path/size/mtime plus Parquet footer hashes, NOT full
content hashes of the entire corpus. Tokenizer, validation and confirmation files
are fully hashed. Shared training data must remain immutable throughout both runs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def snapshot(root,base,confirmation):
    source={}
    for sub,extensions in [("nanochat",{'.py'}),("scripts",{'.py'}),("configs/weighted12",{'.json'})]:
        for p in sorted((root/sub).rglob("*")):
            if p.is_file() and p.suffix in extensions:source[str(p.relative_to(root))]=sha(p)
    files=sorted((base/"base_data_climbmix").glob("*.parquet"))
    assert files and confirmation.name not in {p.name for p in files}, "Confirmation shard must be excluded from train and primary val"
    corpus=[]
    for p in files:
        s=p.stat()
        with p.open('rb') as f:
            f.seek(-8,2);end=f.read(8);assert end[4:]==b'PAR1'
            footer_len=struct.unpack('<I',end[:4])[0]
            f.seek(-(8+footer_len),2);footer=f.read(footer_len+8)
        corpus.append(dict(path=str(p.resolve()),name=p.name,size=s.st_size,mtime_ns=s.st_mtime_ns,footer_sha256=hashlib.sha256(footer).hexdigest()))
    tokenizer={str(p.relative_to(base/"tokenizer")):sha(p) for p in sorted((base/"tokenizer").rglob('*')) if p.is_file()}
    assert tokenizer
    return dict(source_commit="weighted12-queue-20261126",source_sha256=source,
        tokenizer_sha256=tokenizer,corpus_inventory=corpus,
        validation=dict(path=str(files[-1].resolve()),sha256=sha(files[-1]),tokens=41943040),
        confirmation=dict(path=str(confirmation.resolve()),sha256=sha(confirmation),tokens=4194304,
            sequence_length=2048,device_batch_size=4,world_size=2,packing="ExactBOSBestFitDataLoader defaults; each rank reads rowgroups rank::2 from this shard alone; prefix4194304predictiontokens; excluded from updates; previously inspected and used as development evidence, not a new unseen confirmation set",
            exclusion="Different shard identity; no claim of corpus-wide text deduplication"),
        runtime={pkg:importlib.metadata.version(pkg) for pkg in ['torch','numpy','pyarrow','tokenizers','kernels','triton']})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--verify',action='store_true')
    ap.add_argument('--confirmation',required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];base=Path(os.environ['NANOCHAT_BASE_DIR'])
    current=snapshot(root,base,Path(a.confirmation));path=Path(a.manifest)
    if a.verify:
        stored=json.loads(path.read_text());assert stored['snapshot']==current,"Frozen source/data/runtime changed"
        print(json.dumps(dict(verified=True,manifest_sha256=sha(path))))
    else:
        assert not path.exists(),"Refuse to overwrite frozen manifest"
        path.write_text(json.dumps(dict(created=time.time(),snapshot=current),indent=2)+'\n')
        print(json.dumps(dict(frozen=True,manifest_sha256=sha(path),source_files=len(current['source_sha256']),corpus_files=len(current['corpus_inventory']))))

if __name__=='__main__':main()
