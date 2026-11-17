"""Verify that an expanded layer collection reproduces the prior shared layers."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from .common import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference-run', type=Path, required=True)
    a = p.parse_args(); root=a.run_dir.resolve(); old=a.reference_run.resolve()
    parents=[read(r/'manifest.json') for r in (root,old)]
    assert parents[0]['tokens_sha256']==parents[1]['tokens_sha256']
    overlap=sorted(set(parents[0]['layers']) & set(parents[1]['layers'])); assert overlap
    checks={}
    for model in parents[0]['models']:
        records=[]
        for r in (root,old):
            m=read(r/model/'manifest.json'); result=read(r/model/'results.json')
            assert result['state']=='complete' and result['manifest_sha256']==sha(r/model/'manifest.json')
            assert result['statistics_sha256']==sha(r/model/'derived/statistics.npz')
            records.append(m)
        assert records[0]['inputs_hashes']['checkpoint/model_007809.pt']==records[1]['inputs_hashes']['checkpoint/model_007809.pt']
        error=0.;exact=True
        with np.load(root/model/'derived/statistics.npz') as new, np.load(old/model/'derived/statistics.npz') as previous:
            for layer in overlap:
                for prefix in ('energy_V','energy_share_V','denominator'):
                    for suffix in ('mean','variance','std'):
                        key=f'{prefix}_{layer:02d}_{suffix}'
                        np.testing.assert_allclose(new[key],previous[key],rtol=1e-6,atol=1e-10)
                        error=max(error,float(np.max(np.abs(new[key]-previous[key]))));exact=exact and np.array_equal(new[key],previous[key])
        checks[model]=dict(passed=True,arrays_exactly_equal=exact,max_absolute_difference=error,
            new_statistics_sha256=sha(root/model/'derived/statistics.npz'),reference_statistics_sha256=sha(old/model/'derived/statistics.npz'))
    archive=root/'validation'; archive.mkdir(exist_ok=True)
    write(archive/'overlap.json',dict(passed=True,reference_run=str(old),layers=overlap,rtol=1e-6,atol=1e-10,checks=checks))
    shutil.copy2(Path(__file__),archive/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('common.py'),archive/'common.py')
    print(archive/'overlap.json')


if __name__=='__main__':main()
