"""Single-token, wider-window replication; immutable protocol and resumable rows.

Experiment 1: final prompt query open at every layer.
Experiment 2: final prompt query open only for S < L <= T.
Both use the unchanged three-pass relay, middle/generated entity-value removal,
and frozen baseline-qualified fixed adjacent pairs. Unknown control answers are
pending review, NEVER silently failed or treated as relay failures.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

WIDTHS = [6, 8, 10, 12]
WIDTH_ORDER = [10, 8, 12, 6]
TIDS = [0, 1, 2, 3, 7]
TEMPLATE_ORDER = [0, 3, 1, 7, 2]
PASS = {'full_name', 'same_entity_expanded', 'same_entity_alias', 'case_variant', 'format_variant'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(entity, text, stop):
    return json.dumps([entity, text, stop], ensure_ascii=False, separators=(',', ':'))


def windows(width, depth=36):
    if width not in WIDTHS:
        raise ValueError('Only the four prespecified widths are supported')
    return [(s, s + width) for s in range(depth - width)]


def select_pairs(selection, names):
    single = set(names[:50])
    assert len(single) == 50
    cases = {(r['template'], r['entity']): r for r in selection['cases']}
    pairs = [p for p in selection['pairs'] if p['span_length'] == 1]
    assert len({p['id'] for p in pairs}) == len(pairs)
    for p in pairs:
        assert p['template'] in TIDS
        assert p['donor'] in single and p['recipient'] in single
        assert p['donor_id'] ^ 1 == p['recipient_id']
        assert p['donor'] != p['recipient']
        assert all(cases[p['template'], n]['eligible'] for n in (p['donor'], p['recipient']))
    return pairs


def control_decision(row, labels, extra=None):
    key = signature(row['entity'], row['clean_output'], row['stop_reason'])
    category = (extra or {}).get('by_signature', {}).get(key, {}).get('category')
    origin = 'additional explicit whole-response review'
    if category is None:
        category = labels['by_signature'].get(key, {}).get('category')
        origin = 'frozen prior whole-response review'
    if category is None and row['stop_reason'] == 'eos':
        identities = set(labels['whole_response_aliases'].get(row['clean_output'], []))
        if identities == {row['entity']}:
            category = 'same_entity_alias'
            origin = 'frozen unambiguous whole-response alias'
        elif identities and row['entity'] not in identities:
            category = 'different_name_or_word'
            origin = 'whole answer names a different reviewed identity'
    if category is None:
        output_key = json.dumps([row['clean_output'], row['stop_reason']], ensure_ascii=False, separators=(',', ':'))
        category = labels['by_output'].get(output_key, {}).get('category')
        origin = 'frozen prior output review'
    return {'eligible': None if category is None else category in PASS,
            'category': category, 'reason': origin if category else 'pending whole-response review'}


def read_rows(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    idx = {r['id']: r for r in rows}
    if len(idx) != len(rows):
        raise RuntimeError(f'Duplicate rows: {path}')
    return idx


def preflight(a):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from scripts.inspect import qwen_relay_supplied_common as c
    from scripts.inspect import qwen_relay_paper_runtime as w
    from scripts.inspect import qwen_relay_paper_queue as queue
    from scripts.inspect.qwen_relay_paper_provenance import validate_baseline_sources, MANIFEST
    baseline = json.loads((a.baseline_dir / 'protocol.json').read_text())
    selection = json.loads(a.selection.read_text())
    assert selection['baseline_generations_sha256'] == digest(a.baseline_dir / 'generations.jsonl')
    assert selection['baseline_protocol_sha256'] == digest(a.baseline_dir / 'protocol.json')
    validate_baseline_sources(baseline)
    snapshot = snapshot_download(c.ab.MODEL, local_files_only=True)
    assert snapshot == baseline['snapshot']
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    names, _, all_prompts = c.inputs(tok, extras=False)
    assert names == baseline['names']
    pairs = select_pairs(selection, names)
    prompts = {(tid, n): all_prompts[tid, n] for tid in TIDS for n in names[:50]}
    manifest = []
    for (tid, n), enc in prompts.items():
        assert len(enc.entity_positions) == 1, (tid, n)
        assert enc.ids[enc.entity_positions[0]] not in tok.all_special_ids
        manifest.append({'template': tid, 'entity': n, 'prompt': enc.prompt,
                         'ids': enc.ids, 'entity_positions': list(enc.entity_positions)})
    for p in pairs:
        c.validate_pair(prompts[p['template'], p['donor']], prompts[p['template'], p['recipient']])
    sources = [Path(__file__), Path(c.__file__), Path(c.relay.__file__), c.DATA, c.SOURCE,
               Path(w.__file__), Path(queue.__file__), MANIFEST,
               Path(inspect.getsourcefile(validate_baseline_sources))]
    spec = {'version': 'wide-single-paper-package-v1', 'model': c.ab.MODEL, 'snapshot': snapshot,
            'names': names[:50], 'templates': TIDS, 'widths': WIDTHS, 'width_order': WIDTH_ORDER,
            'pairs': pairs, 'manifest': manifest, 'precision': baseline['precision'],
            'max_new_tokens': 12, 'greedy': True, 'depth': 36,
            'windows': {str(k): [[s,t] for s,t in windows(k)] for k in WIDTHS},
            'baseline_rechecks': 500, 'experiment1_generations': len(pairs) * 108,
            'window_controls': len(pairs) * 108, 'experiment2_max_generations': len(pairs) * 108,
            'screening': 'ordinary and middle-open full-identity baselines valid for BOTH members; fixed ID XOR1 pairs; experiment2 additionally needs recipient same-width same-S no-replacement control to pass',
            'experiment1': 'final prompt entity-value access at all layers; no extra readout mask',
            'experiment2': 'final prompt entity-value access only S < L <= T in every pass',
            'shared': 'middle prompt positions and generated queries lose entity-value contribution in all layers; entity itself unchanged; postsoftmax no renormalization; entity K remains in denominator',
            'relay': 'donor entity after S -> recipient entity once; capture last prompt position after T -> fresh recipient last position after T once',
            'indexing': 'zero-based block outputs; width=T-S; first processed donor-influenced block S+1; no embedding-boundary window',
            'review': 'whole-response identity, explicit aliases; unknown pending, not rejected; assistant labels pending author sign-off',
            'selection_sha256': digest(a.selection), 'labels_sha256': digest(a.labels),
            'baseline_protocol_sha256': digest(a.baseline_dir/'protocol.json'),
            'baseline_generations_sha256': digest(a.baseline_dir/'generations.jsonl'),
            'source_hashes': {str(p.resolve()): digest(p) for p in sources}}
    path = a.out_dir / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == spec, 'Protocol/source changed: use a NEW directory'
    else:
        w.save_json(path, spec)
        frozen = a.out_dir/'source'; frozen.mkdir(exist_ok=True)
        for src in sources + [a.selection, a.labels]:
            shutil.copy2(src, frozen / src.name)
    return spec, prompts


def run(a, status):
    import torch
    from scripts.inspect import qwen_relay_supplied_common as c
    from scripts.inspect import qwen_relay_paper_runtime as w
    from scripts.inspect import qwen_relay_paper_queue as queue
    spec, prompts = preflight(a)
    if a.preflight_only:
        status('preflight_complete', prompts=len(prompts), pairs=len(spec['pairs']),
               experiment1_generations=spec['experiment1_generations']); return
    labels = json.loads(a.labels.read_text())
    baseline = read_rows(a.baseline_dir/'generations.jsonl')
    prior = {(r['template'], r['entity'], r['condition']): r for r in baseline.values()}
    cached = {k: read_rows(a.out_dir/(k+'.jsonl')) for k in ('baseline_rechecks', 'controls', 'relay')}
    checks = json.loads((a.out_dir/'checks.json').read_text()) if (a.out_dir/'checks.json').exists() else {}
    checked = set(checks)
    gpu, gpu_lock = queue.wait_gpu(status, 5, 30)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']; torch.set_num_threads(8)
    status('loading', gpu=gpu)
    device = torch.device('cuda')
    model, tok, layers = c.ab.load_model(spec['snapshot'], device)
    hook = c.configure_model(model); assert len(layers) == 36
    streams = {k: (a.out_dir/(k+'.jsonl')).open('a') for k in cached}
    states = {}
    job = {}

    def progress(state='running', **kw):
        status(state, gpu=gpu, **job, baseline_saved=len(cached['baseline_rechecks']),
               controls_saved=len(cached['controls']), controls_total=spec['window_controls'],
               experiment1_saved=sum(r['mode']=='open' for r in cached['relay'].values()),
               experiment1_total=spec['experiment1_generations'],
               experiment2_saved=sum(r['mode']=='window' for r in cached['relay'].values()),
               checks_passed=len(checks), **kw)

    def emit(kind, row):
        assert row['id'] not in cached[kind]
        streams[kind].write(json.dumps(row, ensure_ascii=False)+'\n'); streams[kind].flush()
        cached[kind][row['id']] = row
        if sum(len(v) for v in cached.values()) % 16 == 0: progress()

    def check(key, test, **kw):
        checks[key] = {'test': test, 'passed': True, **kw}
        w.save_json(a.out_dir/'checks.json', checks)

    def entity_states(tid, name):
        if (tid, name) not in states:
            states[tid, name], _ = c.capture(model, layers, prompts[tid, name], device)
        return states[tid, name]

    def decisions():
        extra = json.loads(a.review.read_text()) if a.review and a.review.exists() else {}
        if a.review and a.review.exists():
            assert extra['protocol_sha256'] == digest(a.out_dir/'protocol.json')
            frozen = a.out_dir/'source'/('review_'+digest(a.review)+'.json')
            if not frozen.exists(): shutil.copy2(a.review, frozen)
        return {key: control_decision(row, labels, extra) for key, row in cached['controls'].items()}

    def save_gates():
        gates = decisions()
        w.save_json(a.out_dir/'gates.json', {'protocol_sha256': digest(a.out_dir/'protocol.json'),
                    'control_sha256': digest(a.out_dir/'controls.jsonl'), 'decisions': gates})
        pending = [dict(cached['controls'][key], decision=value) for key, value in gates.items()
                   if value['eligible'] is None]
        w.save_json(a.out_dir/'pending_controls.json', pending)
        return gates

    try:
        with torch.inference_mode():
            # Verify all 50 names x 5 templates, not just the survivors.
            job.update(stage='baseline_rechecks')
            for tid in TIDS:
                for name in spec['names']:
                    for condition in ('ordinary', 'middle_open'):
                        key = f'{tid}:{name}:{condition}'
                        if key in cached['baseline_rechecks']: continue
                        gen = c.generate(model, tok, layers, prompts[tid, name], device,
                                         ordinary=condition=='ordinary')
                        assert gen['generated_token_ids'] == prior[tid,name,condition]['generated_token_ids'], key
                        check('baseline:'+key, 'frozen baseline token-exact parity')
                        emit('baseline_rechecks', {'id': key, 'template': tid, 'entity': name,
                                                  'condition': condition, **gen})
            # Reproduce the prior width-4 Einstein->Newton outputs as an integration check.
            if 'width4_reference' not in checked:
                old = read_rows(a.old_run/'sweep_t0'/'generations.jsonl')
                ref = {(r['donor'],r['recipient'],r['S']): r for r in old.values()
                       if r['kind']=='relay' and r['readout_policy']=='open'}
                for s in (0,20,30,31):
                    gen = c.run_relay(model,tok,layers,prompts[0,'Newton'],device,
                                      entity_states(0,'Einstein')[s],s,s+4)
                    assert gen['generated_token_ids'] == ref['Einstein','Newton',s]['generated_token_ids']
                check('width4_reference', 'four frozen width-4 relay outputs reproduced')
            for width in WIDTH_ORDER:
                for tid in TEMPLATE_ORDER:
                    pairs = [p for p in spec['pairs'] if p['template']==tid]
                    recipients = list(dict.fromkeys(p['recipient'] for p in pairs))
                    job.update(width=width, template=tid, stage='window_controls')
                    for s,t in windows(width):
                        disabled = w.outside_layers(36,s,t)
                        assert set(range(36))-set(disabled) == set(range(s+1,t+1))
                        for name in recipients:
                            key = f'{tid}:{width}:{s}:{name}'
                            if key not in cached['controls']:
                                gen = c.generate(model,tok,layers,prompts[tid,name],device,disabled)
                                emit('controls', {'id':key,'template':tid,'width':width,'S':s,'T':t,
                                                  'entity':name, **gen})
                        if recipients and s in (0,(36-width)//2,35-width):
                            name=recipients[0]
                            for mode in ('open','window'):
                                ck=f'identity:{tid}:{width}:{s}:{mode}'
                                if ck in checks: continue
                                gen=c.run_relay(model,tok,layers,prompts[tid,name],device,
                                                entity_states(tid,name)[s],s,t,
                                                () if mode=='open' else disabled)
                                ref=(prior[tid,name,'middle_open'] if mode=='open' else
                                     cached['controls'][f'{tid}:{width}:{s}:{name}'])
                                assert gen['generated_token_ids']==ref['generated_token_ids'], ck
                                check(ck,'self-replacement equals same-mask no-replacement control')
                    # Freeze gates from controls BEFORE relay for this template/width.
                    gates=save_gates()
                    job.update(stage='relay'); progress()
                    for s,t in windows(width):
                        for p in pairs:
                            control_id=f'{tid}:{width}:{s}:{p["recipient"]}'
                            for mode in ('open','window'):
                                key=f'{p["id"]}:{width}:{s}:{mode}'
                                if key in cached['relay']: continue
                                if mode=='window' and gates[control_id]['eligible'] is not True: continue
                                disabled=() if mode=='open' else w.outside_layers(36,s,t)
                                gen=c.run_relay(model,tok,layers,prompts[tid,p['recipient']],device,
                                                entity_states(tid,p['donor'])[s],s,t,disabled)
                                emit('relay', {'id':key, 'pair_id':p['id'],'template':tid,
                                               'width':width,'S':s,'T':t,'mode':mode,
                                               'donor':p['donor'],'recipient':p['recipient'],
                                               'span_length':1,'control_id':control_id if mode=='window' else None,
                                               'gate':gates[control_id] if mode=='window' else None, **gen})
                    progress('running', completed_job=f't{tid}_w{width}')
                    states.clear()
            gates=save_gates()
            assert len(cached['controls']) == spec['window_controls']
            assert sum(r['mode']=='open' for r in cached['relay'].values()) == spec['experiment1_generations']
            expected=sum(v['eligible'] is True for v in gates.values())
            assert sum(r['mode']=='window' for r in cached['relay'].values()) == expected
            pending=sum(v['eligible'] is None for v in gates.values())
            progress('awaiting_control_review' if pending else 'complete_needs_output_review',
                     window_controls_pass=expected, window_controls_pending=pending,
                     window_controls_fail=sum(v['eligible'] is False for v in gates.values()))
    finally:
        for stream in streams.values(): stream.close()
        hook.remove(); gpu_lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--review',type=Path)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    for field in ('out_dir','baseline_dir','selection','labels','old_run','review'):
        value=getattr(a,field)
        if value is not None: setattr(a,field,value.resolve())
    if a.launch:
        a.out_dir.mkdir(parents=True,exist_ok=True)
        guard=(a.out_dir/'.launch.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker_lock=(a.out_dir/'.worker.lock').open('a');fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.out_dir/'launch.json').exists() and not a.resume: raise RuntimeError('Existing launch: refuse duplicate')
        if (a.out_dir/'launch.json').exists():
            previous=json.loads((a.out_dir/'launch.json').read_text())
            try:os.kill(previous['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Previous worker PID still alive; refuse duplicate')
        command=[sys.executable,'-m','scripts.inspect.qwen_relay_wide_single']
        for field in ('out_dir','baseline_dir','selection','labels','old_run','review'):
            value=getattr(a,field)
            if value is not None: command.extend(['--'+field.replace('_','-'),str(value)])
        if a.resume: command.append('--resume')
        worker_lock.close()
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
             'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'8'}
        with (a.out_dir/'worker.log').open('a') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  env=env,start_new_session=True)
        from scripts.inspect.qwen_relay_paper_runtime import save_json,stamp
        record={'pid':proc.pid,'time':stamp(),'command':command}
        save_json(a.out_dir/'launch.json',record);print(json.dumps(record));return
    a.out_dir.mkdir(parents=True,exist_ok=True)
    guard=(a.out_dir/'.worker.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.out_dir/'relay.jsonl').exists() and not a.resume: raise RuntimeError('Use --resume; never overwrite results')
    from scripts.inspect.qwen_relay_paper_runtime import save_json,stamp
    def status(state,**kw):
        row={'state':state,'pid':os.getpid(),'time':stamp(),**kw}
        save_json(a.out_dir/'status.json',row);print(json.dumps(row),flush=True)
    try:run(a,status)
    except BaseException as exc:status('failed',error=repr(exc));traceback.print_exc();raise


if __name__=='__main__': main()
