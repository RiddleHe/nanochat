"""Compare validated full-shard embedding and selected sublayer norms for two models."""
import argparse
from pathlib import Path
import json
import shutil
import numpy as np
from .common import read_json, sha256, write_json


def load(run, model_type, family='attention'):
    manifest = read_json(run/'manifest.json'); result = read_json(run/'results.json')
    assert manifest['config'].get('model_type', 'gpt_base') == model_type
    assert result['state'] == 'complete'
    for path, key in [('derived/statistics.npz','statistics_sha256'), ('derived/schema.json','schema_sha256'), ('validation/summary.json','validation_sha256')]:
        assert sha256(run/path) == result[key], path
    assert read_json(run/'validation/summary.json')['passed']
    schema = read_json(run/'derived/schema.json'); points = []
    with np.load(run/'derived/statistics.npz') as z:
        assert np.all(z['count'] == result['tokens'])
        np.testing.assert_allclose(z['std']**2, z['variance'], rtol=1e-14)
        for source in schema['sources']:
            if family != 'both' and source['family'] not in ('embedding',family): continue
            i = source['index']; mean = float(z['mean'][i]); std = float(z['std'][i])
            assert np.isfinite(mean) and mean > 0 and np.isfinite(std) and std >= 0
            points.append(dict(source, x=i-1 if family == 'both' else source['layer'], mean=mean, std=std,
                               variance=float(z['variance'][i]), count=int(z['count'][i]),
                               lower=max(0.,mean-std), upper=mean+std))
    assert [p['x'] for p in points] == list(range(-1,48 if family == 'both' else 24))
    if family == 'both':
        assert [(p['family'],p['layer']) for p in points] == [('embedding',-1)] + [(f,l) for l in range(24) for f in ('attention','mlp')]
    return manifest, result, points, schema['dimensions']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--variant-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--y-scale', choices=['linear','log'], default='log')
    p.add_argument('--family', choices=['attention','mlp','both'], default='attention')
    p.add_argument('--normalized-embedding', type=Path, help='Validated baseline post-RMSNorm statistics.json; retain its raw mean as one dashed reference')
    args = p.parse_args(); baseline = args.baseline_run.resolve(); variant = args.variant_run.resolve()
    bm, br, bp, bd = load(baseline, 'gpt_base', args.family)
    vm, vr, vp, vd = load(variant, 'gpt_base_no_embed_norm', args.family)
    for name in ('token_ids.u32','document_offsets.npy','token_frequencies.npy','validation.parquet'):
        assert bm['inputs_hashes'][name] == vm['inputs_hashes'][name], name
    for name in ('sequence_length','token_policy'):
        assert bm['config'][name] == vm['config'][name]
    assert br['tokens'] == vr['tokens'] and br['documents'] == vr['documents'] and bd == vd
    assert [(p['index'],p['family'],p['layer']) for p in bp] == [(p['index'],p['family'],p['layer']) for p in vp]
    references=[]; normalized_input=None
    if args.normalized_embedding:
        path=args.normalized_embedding.resolve(); measured=read_json(path)
        assert measured['passed'] and measured['count']==br['tokens']
        assert measured['manifest_sha256']==sha256(baseline/'manifest.json')
        assert measured['checkpoint_sha256']==bm['inputs_hashes']['checkpoint/model_007809.pt']
        assert measured['frequencies_sha256']==bm['inputs_hashes']['token_frequencies.npy']
        assert measured['per_vocabulary_sha256']==sha256(path.parent/'per_vocabulary.npz')
        assert measured['checks'] and all(c['exact_first_block_input_norm_match'] for c in measured['checks'])
        references=[dict(id='baseline-raw-embedding',label='GPTBase raw Emb',value=bp[0]['mean'],dash='7 4')]
        mean,std=measured['mean'],measured['std']
        bp[0].update(mean=mean,std=std,variance=measured['variance'],lower=max(0.,mean-std),upper=mean+std,boundary=measured['boundary'])
        normalized_input=dict(path=str(path),sha256=sha256(path),job_sha256=sha256(path.parent/'job.json'))
    series = [dict(id='baseline',label='GPTBase',values=bp), dict(id='variant',label='No norm(x0)',values=vp)]
    data = dict(run_id=baseline.name+'__'+variant.name, comparison=True, model_label='GPTBase vs no norm(x0)',
                tokens=br['tokens'], documents=br['documents'], dimensions=bd, families=args.family,
                y_scale=args.y_scale, references=references, series=series, points=bp+vp,
                baseline_normalized_embedding=bool(args.normalized_embedding),
                clipped_lower_band=any(p['lower']<=0 for p in bp+vp))
    here = Path(__file__).parent; template = here/('both_view.html' if args.family == 'both' else 'view.html'); output = args.output.resolve()
    fragment = template.read_text().replace('__NORM_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
    assert len(fragment.encode()) < 1_000_000 and (not output.exists() or output.read_text()==fragment)
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(fragment)
    archive = variant/'visualization'/('comparison_norms_'+sha256(output)[:12]); archive.mkdir(parents=True,exist_ok=True)
    code = [Path(__file__), here/'common.py', template]
    for path in [*code,output]: shutil.copy2(path,archive/path.name)
    if args.normalized_embedding:
        shutil.copy2(args.normalized_embedding,archive/'normalized_embedding_statistics.json')
        shutil.copy2(args.normalized_embedding.parent/'job.json',archive/'normalized_embedding_job.json')
    write_json(archive/'data.json',data)
    write_json(archive/'settings.json',dict(
        runs={s:dict(path=str(r),manifest_sha256=sha256(r/'manifest.json'),results_sha256=sha256(r/'results.json'),
                     statistics_sha256=res['statistics_sha256']) for s,r,res in [('baseline',baseline,br),('variant',variant,vr)]},
        source_code_hashes={p.name:sha256(p) for p in code}, inline_path=str(output),inline_sha256=sha256(output),
        theme='Codex dark',x_scale='interleaved Emb,A0,M0,...,A23,M23' if args.family == 'both' else 'linear',y_scale=args.y_scale,references=references,normalized_embedding_input=normalized_input,
        connections='separate dashed attention and MLP curves; standalone Emb' if args.family == 'both' else 'successive displayed sources',
        band='mean ± one population SD over all token occurrences; lower bound clipped at zero',
        y_domain='shared across visible series; rescales on legend toggle',
        embedding='GPTBase after actual BF16 RMSNorm; variant raw lookup' if args.normalized_embedding else 'raw lookup in both models', source_family=args.family, source_boundary='module output before residual addition',
        matching_tokens_and_contexts=True,tokens=br['tokens']))
    print(archive)


if __name__ == '__main__': main()
