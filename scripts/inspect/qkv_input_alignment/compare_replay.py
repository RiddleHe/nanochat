"""Compare every shared-threshold token alignment against the original run."""
import argparse
from pathlib import Path
import numpy as np
from .common import read_json, require, sha256, stage_entry, verify_chunk, verify_run, verify_svd, write_json
from .reuse_svd import source_run


def compare(run):
    run=Path(run);m,_=verify_run(run);verify_svd(run,m)
    source,sm,ref=source_run(run,m)
    for r in (run,source):
        results=read_json(r/'results.json')
        require(results['state']=='analysis_complete' and read_json(r/'validation/summary.json')['passed'], 'Run incomplete')
        require(sha256(r/'summaries/statistics.npz')==results['statistics_sha256'], 'Summary changed')
    require(m['tokens_shape']==sm['tokens_shape'] and m['config'].get('measurement','sources')==sm['config'].get('measurement','sources'), 'Replay measurement mismatch')
    nt=m['config']['squared_energy_thresholds'];ot=sm['config']['squared_energy_thresholds']
    shared=[(nt.index(t),i,t) for i,t in enumerate(ot) if t in nt]
    require(bool(shared),'No shared thresholds')
    exact=True;maximum=0.;values=0;chunk_records=[]
    # These tolerances allow FP32 replay rounding; exact equality is reported separately.
    rtol,atol=1e-6,2e-7
    for i in range(m['tokens_shape'][0]):
        name=f'chunk_{i:05d}.npz';npath=run/'tokens'/name;opath=source/'tokens'/name
        for r,path,manifest in ((run,npath,m),(source,opath,sm)):
            rec=verify_chunk(r,path,sha256(r/'manifest.json'),sha256(r/'svd/index.json'))
            require(rec['sequence_id']==i and rec['tokens_sha256']==manifest['tokens_sha256'], 'Chunk identity mismatch')
        with np.load(npath) as new,np.load(opath) as old:
            require(set(new.files)==set(old.files), 'Different measurement arrays')
            for key in new.files:
                a,b=new[key],old[key]
                pairs=[(a[...,ni],b[...,oi]) for ni,oi,_ in shared] if key.startswith('alignment_') else [(a,b)]
                for x,y in pairs:
                    require(x.shape==y.shape and np.array_equal(np.isnan(x),np.isnan(y)), 'Replay masks/dimensions differ')
                    np.testing.assert_allclose(x,y,rtol=rtol,atol=atol,equal_nan=True)
                    exact &= np.array_equal(x,y,equal_nan=True)
                    finite=np.isfinite(x);maximum=max(maximum,float(np.max(np.abs(x[finite]-y[finite]),initial=0)))
                    values += int(finite.sum())
        chunk_records.append(dict(chunk=name,new_sha256=sha256(npath),reference_sha256=sha256(opath)))
        if (i+1)%16==0: print(f'Compared {i+1}/{m["tokens_shape"][0]} chunks; max absolute error={maximum:g}',flush=True)
    result=dict(passed=True,reference=ref,thresholds=[t for _,_,t in shared],chunks=len(chunk_records),
        tokens=int(np.prod(m['tokens_shape'])),finite_values_compared=values,all_values_exactly_equal=bool(exact),
        max_absolute_error=maximum,rtol=rtol,atol=atol,manifest_sha256=sha256(run/'manifest.json'),chunk_hashes=chunk_records)
    write_json(run/'validation/replay_comparison.json',result)
    print(f'Replay comparison passed; exact={exact}, max absolute error={maximum:g}.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();compare(stage_entry(a.run_dir,'compare_replay'))
