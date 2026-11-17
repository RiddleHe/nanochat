"""Validate identical frozen code snapshots and seal CPU-ready manifests."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index',type=Path,default=Path(__file__).with_name('norm-input-prepared-runs.json'))
args=parser.parse_args()
runs=json.loads(args.index.read_text())
first=Path(next(iter(runs.values())))
tests=['test_embedding_value_norm_flags.py','test_gpt_base_pooled_qk_embed_no_embed_norm.py',
       'test_gpt_base_single_qk_embed_no_embed_norm.py','test_gpt_base_split_embed_head_out_no_embed_norm.py',
       'test_gpt_base_no_embed_norm.py','test_gpt_base_head_scaled_embed_no_embed_norm.py',
       'test_gpt_base_split_embed_shared_out_no_embed_norm.py','test_norm_input_queue.py',
       'test_norm_input_launcher.py','test_split_embed_launcher.py']
if (first/'code/tests/test_gpt_base_single_qk_value_proj_no_embed_norm.py').exists():
 tests.append('test_gpt_base_single_qk_value_proj_no_embed_norm.py')
python='/hdd/mh3897/nanochat/.venv/bin/python'
command=[python,'-m','pytest','-q']+['tests/'+t for t in tests]
env=dict(os.environ,OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(first/'code'))
with (first/'cpu_validation.log').open('w') as log:
 subprocess.run(command,cwd=first/'code',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print((first/'cpu_validation.log').read_text(),flush=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reference=json.loads((first/'manifest.json').read_text())['source_sha256']
for run in runs.values():
 root=Path(run);m=json.loads((root/'manifest.json').read_text())
 assert m['source_sha256']==reference
 for name in tests:assert digest(root/'code/tests'/name)==digest(first/'code/tests'/name)
 for name in ('launch.py','launcher_core.py','queue_norm_input.py'):
  assert digest(root/name)==digest(first/name)
 if root!=first:shutil.copy2(first/'cpu_validation.log',root/'cpu_validation.log')
 cpu=dict(status='passed',command=command,validated_snapshot=str(first/'code'),
          equivalence='Identical source hashes, test files, and launcher/queue files across the indexed snapshots',
          summary=(root/'cpu_validation.log').read_text().strip().splitlines()[-1])
 (root/'cpu_validation.json').write_text(json.dumps(cpu,indent=2)+'\n')
 m['validation']=dict(status='cpu_passed_pending_gpu',cpu=cpu,gpu='automatic_on_explicit_launch',
                      distributed_status='pending_automatic_preflight',training_started=False)
 names=['launch.py','launcher_core.py','launch.sh','queue_norm_input.py','gpu_smoke.py','distributed_smoke.py',
        'finalize_validation.py','prepare_snapshot.py','environment_packages.json','cpu_validation.json','cpu_validation.log']
 files=[root/n for n in names]+list((root/'code/tests').rglob('*.py'))
 m['artifact_sha256']={str(p.relative_to(root)):digest(p) for p in sorted(files)}
 (root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 (root/'status.json').write_text(json.dumps(dict(status='prepared',run_id=m['model_tag'],training_started=False,
      validation='cpu_passed_pending_gpu'),indent=2)+'\n')
 print(m['research_name'],m['expected_steps'],m['estimated_flops_per_token'],flush=True)
