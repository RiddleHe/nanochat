"""Step 1.5 analysis with the frozen decision rule.

python analyze15.py /hdd/mh3897/cc/looping-experiments <out summary.json> <frozen_rule.json>
Rule file: step15_local_20261026/analysis/frozen_rule.json (frozen before outcomes were opened).
"""
import json, math, sys, numpy as np
R = sys.argv[1] + '/step15_local_20261026/'
OUT = sys.argv[2]
rule = json.load(open(sys.argv[3]))
DELTA = rule['delta_bpb']
LOCS = ['L14','L19','L07','L10','L04','L21']
GROUP = {'L14':'high','L19':'high','L07':'low','L10':'low','L04':'random','L21':'random'}
V = ['shared_small','independent','shared_large']
def rows(p): return json.load(open(p))['rows']
def arr(r):
    assert [x['sequence'] for x in r] == list(range(len(r)))
    return np.array([x['byte_nats'] for x in r]), np.array([x['bytes'] for x in r])
def bpb(a, idx=None):
    bn, nb = a
    return bn.sum()/(math.log(2)*nb.sum()) if idx is None else bn[idx].sum(-1)/(math.log(2)*nb[idx].sum(-1))
IDX = np.random.default_rng(42).integers(0, 2048, (4000, 2048))
def diff(a, b):
    boots = bpb(a, IDX) - bpb(b, IDX)
    lo, hi = np.quantile(boots, [.025, .975])
    return dict(d=float(bpb(a)-bpb(b)), se=float(boots.std()), lo=float(lo), hi=float(hi))
base = {s: arr(rows(R+f'results/baseline_eval/{s}.json')) for s in ('development','confirmation')}
grad = json.load(open(R+'results/gradient/results.json'))
sel = json.load(open(R+'selection.json'))
def ab(name, loc):
    r = [x for x in grad['reports'][name]['rows'] if x['location']==loc][0]
    return r
early = {}
for line in open(sys.argv[1]+'/step2_early_20261025/analysis/early_scores.csv').read().splitlines()[1:]:
    ck, s, loc, ca, cb, nrm = line.split(',')
    early[(int(ck), s, loc)] = float(cb)
res = {'delta': DELTA, 'baseline': {s: bpb(base[s]) for s in base}, 'locations': {}}
for L in LOCS:
    runs = {}
    for v in V:
        rr = json.load(open(R+f'results/{L}_{v}/results.json'))
        runs[v] = dict(conf=arr(rows(R+f'results/{L}_{v}/confirmation_200.json')),
                       dev=arr(rows(R+f'results/{L}_{v}/development_200.json')),
                       curve={s: json.load(open(R+f'results/{L}_{v}/development_{s:03d}.json'))['summary']['bpb'] for s in (0,50,100,200)},
                       params=rr['trainable_parameters'], hours=rr['elapsed_seconds']/3600)
    e = {}
    for split, key in (('confirmation','conf'),('development','dev')):
        e[split] = dict(
            bpb={v: float(bpb(runs[v][key])) for v in V},
            gain_vs_large=diff(runs['shared_large'][key], runs['independent'][key]),
            gain_vs_small=diff(runs['shared_small'][key], runs['independent'][key]),
            large_vs_small=diff(runs['shared_small'][key], runs['shared_large'][key]),
            adapt_vs_unchanged={v: diff(base[split], runs[v][key]) for v in V})
    g, gd = e['confirmation']['gain_vs_large'], e['development']['gain_vs_large']
    same = np.sign(g['d']) == np.sign(gd['d'])
    if g['lo'] > 0 and same: cls = 'clear_benefit' if g['d'] >= DELTA else 'small_detectable'
    elif g['hi'] < 0 and same: cls = 'shared_better_clear' if -g['d'] >= DELTA else 'shared_better_small'
    else: cls = 'no_evidence'
    rD1 = ab('D1', L+'.up')
    e.update(group=GROUP[L], C_D1=rD1['conflict_B'], C_A_D1=rD1['conflict_A'],
             C_D2=ab('D2', L+'.up')['conflict_B'], early_C_ck1000=early.get((1000,'D1',L+'.up')),
             early_C_ck100=early.get((100,'D1',L+'.up')), cls=cls,
             curves={v: runs[v]['curve'] for v in V}, params={v: runs[v]['params'] for v in V},
             hours={v: runs[v]['hours'] for v in V})
    res['locations'][L] = e
sc = json.load(open(R+'results/scaling/results.json'))
res['scaling'] = dict(confirmed=sc['confirmed'], picks=sc['picks'], conf_unchanged=sc['confirmation_unchanged'],
                      dev_unchanged=sc['dev_subset_unchanged'],
                      conditions=[dict(location=c['location'], round=c['round'], scale=c['scale'], d=c['vs_unchanged']['delta_bpb'], ci=c['vs_unchanged']['ci95']) for c in sc['conditions']])
res['all24'] = [dict(location=r['location'], C_B=r['conflict_B'], C_A=r['conflict_A'],
                     C_B_D2=[x for x in grad['reports']['D2']['rows'] if x['location']==r['location']][0]['conflict_B'],
                     norm=r['mean_shared_gradient_norm'], B=r['B'], A=r['A']) for r in grad['reports']['D1']['rows']]
res['stability'] = sel['stability']
json.dump(res, open(OUT, 'w'), indent=1)
print('baseline', res['baseline'])
for L in LOCS:
    e = res['locations'][L]; c = e['confirmation']
    print(L, e['group'], f"C={e['C_D1']:+.3f}", {v: round(c['bpb'][v], 6) for v in V},
          'gainL=%+.4fe-3 [%+.4f,%+.4f] dev=%+.4fe-3' % (c['gain_vs_large']['d']*1e3, c['gain_vs_large']['lo']*1e3, c['gain_vs_large']['hi']*1e3, e['development']['gain_vs_large']['d']*1e3),
          'gainS=%+.4fe-3' % (c['gain_vs_small']['d']*1e3), 'adapt(ss)=%+.4fe-3' % (c['adapt_vs_unchanged']['shared_small']['d']*1e3), e['cls'])
for k, v in res['scaling']['confirmed'].items(): print('scale', k, v['round'], v['scale'], '%+.4fe-3' % (v['vs_unchanged']['delta_bpb']*1e3), [round(x*1e3, 4) for x in v['vs_unchanged']['ci95']])
