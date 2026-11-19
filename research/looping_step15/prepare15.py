"""Reserve a NEW confirmation selection before training, then freeze sources and data.

python prepare15.py           -> data/confirmation.npy (+ registry, manifest)
python prepare15.py --freeze  -> manifest.json (requires passing CPU tests and GPU preflight)
"""
import hashlib
import json
import sys
import numpy as np
import pyarrow.parquet as pq
from shared15 import ROOT, STEP1, STEP2, PROTOCOL, previous, now, sha, sources, verify, write_json


def main():
    verify()
    dest = ROOT / 'data'
    dest.mkdir(exist_ok=True)
    if (dest / 'manifest.json').exists():
        raise RuntimeError('Data already prepared; refusing to overwrite')
    from nanochat.tokenizer import get_tokenizer
    old = json.loads((STEP1 / 'data/manifest.json').read_text())
    excluded, registries = set(), {}
    for name, meta in old['splits'].items():
        p = STEP1 / 'data' / f'{name}_documents.jsonl'
        assert sha(p) == meta['documents_sha256'], name
        registries[f'step1_{name}'] = dict(path=str(p), sha256=sha(p))
        excluded.update(json.loads(line)['text_sha256'] for line in p.read_text().splitlines())
    s2 = json.loads((STEP2 / 'data/manifest.json').read_text())
    diag = STEP2 / 'data/diagnostic_exclusions.json'
    assert sha(diag) == s2['diagnostic_exclusions_sha256']
    excluded.update(json.loads(diag.read_text())['text_sha256'])
    registries['step2_D1_D2'] = dict(path=str(diag), sha256=sha(diag))

    path = previous.BASE / 'base_data_climbmix/shard_06542.parquet'
    expected = {r['name']: r['sha256'] for r in previous.verify_source()['dataset_files']}
    assert sha(path) == expected[path.name]
    tokenizer = get_tokenizer()
    n = PROTOCOL['confirmation_sequences']
    need = n * 2049
    tokens, docs, seen, row = [], [], set(excluded), 0
    for records in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['text'], use_threads=False):
        texts = records.column(0).to_pylist()
        encoded = tokenizer.encode(texts, prepend=tokenizer.get_bos_token_id(), num_threads=4)
        for text, ids in zip(texts, encoded):
            digest = hashlib.sha256(text.encode()).hexdigest()
            source_row, row = row, row + 1
            if digest in seen or len(ids) < 2:
                continue
            seen.add(digest)
            take = min(len(ids), need - len(tokens))
            docs.append(dict(file=path.name, row=source_row, text_sha256=digest,
                             token_start=len(tokens), tokens_used=take, document_tokens=len(ids)))
            tokens.extend(ids[:take])
            if len(tokens) == need:
                break
        if len(tokens) == need:
            break
    if len(tokens) != need:
        raise RuntimeError(f'Shard 6542 exhausted: {len(tokens)} of {need} tokens')
    assert not ({d['text_sha256'] for d in docs} & excluded)
    out = dest / 'confirmation.npy'
    np.save(out, np.asarray(tokens, dtype=np.uint16).reshape(n, 2049))
    reg = dest / 'confirmation_documents.jsonl'
    with open(reg, 'x') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')
    write_json(dest / 'manifest.json', dict(
        created_utc=now(),
        confirmation=dict(path=str(out), sha256=sha(out), sequences=n, prediction_tokens=n * 2048,
                          documents=len(docs), documents_sha256=sha(reg),
                          first_row=docs[0]['row'], last_row=docs[-1]['row']),
        reused=dict(train=dict(path=str(STEP1 / 'data/train.npy'), sha256=old['splits']['train']['sha256']),
                    development=dict(path=str(STEP1 / 'data/development.npy'),
                                     sha256=old['splits']['development']['sha256'])),
        excluded_registries=registries, excluded_documents=len(excluded),
        packing=old['packing'],
        note='Old Step 1 confirmation was inspected, so it is excluded and not reused.',
        limitations='Exact text only; not near-duplicate or pretraining-corpus filtering.'))
    print(json.dumps(dict(event='confirmation_reserved', documents=len(docs),
                          rows=[docs[0]['row'], docs[-1]['row']])), flush=True)


def freeze():
    assert not (ROOT / 'manifest.json').exists(), 'Already frozen'
    verify()
    src = sources()
    for name in ('tests_cpu.json', 'tests_gpu.json'):
        report = json.loads((ROOT / name).read_text())
        assert report['passed'], name
        assert report['source_hashes'] == src, f'Code changed since {name}'
    s1 = json.loads((STEP1 / 'experiment_manifest.json').read_text())
    write_json(ROOT / 'manifest.json', dict(
        created_utc=now(), protocol=PROTOCOL, source_hashes=src,
        step1_manifest_sha256=sha(STEP1 / 'experiment_manifest.json'),
        step2_manifest_sha256=sha(STEP2 / 'manifest.json'),
        checkpoint=s1['checkpoint'], checkpoint_sha256=s1['checkpoint_sha256'],
        data_manifest_sha256=sha(ROOT / 'data/manifest.json'), runtime=s1['runtime']))
    verify()
    print('FROZEN', sha(ROOT / 'manifest.json'), flush=True)


if __name__ == '__main__':
    freeze() if '--freeze' in sys.argv else main()
