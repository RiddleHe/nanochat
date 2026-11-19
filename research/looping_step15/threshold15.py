"""Derive the Step 1.5 minimum useful BPB gain from Step 1 results only.

python threshold15.py /hdd/mh3897/cc/looping-experiments
Prints paired-bootstrap SE and dev/confirmation differences for each Step 1 comparison.
"""
import json, math, sys, numpy as np
D = sys.argv[1] + '/step1_20261020/results/'
def rows(p): return json.load(open(p))['rows']
def arr(r): return np.array([x['byte_nats'] for x in r]), np.array([x['bytes'] for x in r])
def bpb(bn, nb, idx=None):
    if idx is None: return bn.sum()/(math.log(2)*nb.sum())
    return bn[idx].sum(-1)/(math.log(2)*nb[idx].sum(-1))
rng = np.random.default_rng(0)
out = {}
base = {s: arr(rows(D+f'baseline_eval/{s}.json')) for s in ('development','confirmation')}
for split, fname in (('confirmation','confirmation_200.json'),('development','development_200.json')):
    runs = {f'{g}_{v}': arr(rows(D+f'{g}_{v}/{fname}')) for g in ('qk','vo','mlp') for v in ('shared_small','independent','shared_large')}
    n = len(base[split][0]); idx = rng.integers(0,n,(4000,n))
    for g in ('qk','vo','mlp'):
        for a,b in (('shared_large','independent'),('shared_small','independent'),('shared_small','shared_large')):
            A,B = runs[f'{g}_{a}'], runs[f'{g}_{b}']
            d = bpb(*A)-bpb(*B); boots = bpb(*A,idx)-bpb(*B,idx)
            out[f'{split}:{g}:{a}-{b}'] = (d, boots.std(), *np.quantile(boots,[.025,.975]))
        U = base[split]; S = runs[f'{g}_shared_small']
        d = bpb(*U)-bpb(*S); boots = bpb(*U,idx)-bpb(*S,idx)
        out[f'{split}:{g}:unchanged-shared_small'] = (d, boots.std(), *np.quantile(boots,[.025,.975]))
for k,v in out.items(): print(f'{k:45s} diff={v[0]*1e3:+.4f}e-3  se={v[1]*1e3:.4f}e-3  ci=[{v[2]*1e3:+.4f},{v[3]*1e3:+.4f}]e-3')
