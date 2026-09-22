"""Change only the description; preserve the existing final-query ablation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import torch
import transformers
import qwen_entity_attention_ablation as core


def main():
    parser = argparse.ArgumentParser()
    for key in ['model', 'cases', 'manifest', 'outdir']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    assert Counter(r['template_id'] for r in cases) == {t: 100 for t in range(4)}
    device = torch.device('cuda:0')
    model, tokenizer, blocks = core.load_model(args.model, device)
    assert len(blocks) == 36
    templates = {t.template_id: t for t in core.TEMPLATES}
    prepared, validation = [], []
    for case in cases:
        entity = case['entity']; wrong = manifest[entity]['wrong_description']
        old = case['entity_description']
        assert wrong != old
        enc = core.encode_prompt(tokenizer, templates[case['template_id']], case['entity_id'], entity, case['entity_group'], wrong)
        assert enc.prompt == case['prompt'].replace(f'{entity} ({old})', f'{entity} ({wrong})', 1)
        old_pos, new_pos = set(case['description_positions']), set(enc.description_positions)
        assert [v for i,v in enumerate(case['prompt_token_ids']) if i not in old_pos] == [v for i,v in enumerate(enc.ids) if i not in new_pos]
        assert enc.entity_position == case['entity_position']
        assert enc.ids[enc.entity_position] == case['prompt_token_ids'][case['entity_position']]
        validation.append(dict(template_id=case['template_id'], entity_id=case['entity_id'], entity=entity,
            original_description=old, wrong_description=wrong, old_description_tokens=len(old_pos),
            new_description_tokens=len(new_pos), token_length_delta=len(enc.ids)-len(case['prompt_token_ids']),
            same_entity_token=True, all_tokens_outside_description_unchanged=True,
            original_readout_position=case['readout_position'], readout_position=len(enc.ids)-1))
        prepared.append((case, enc))
    (out / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(f'Validated {len(prepared)} prompts: only description token spans change.', flush=True)
    with (out / 'generations.jsonl').open('w') as f:
        for index, (case, enc) in enumerate(prepared):
            with core.disable_entity_attention(layers=range(36), entity_position=enc.entity_position,
                    readout_position=len(enc.ids)-1, prompt_length=len(enc.ids), block_intermediate_prompt=False) as state:
                result = core.greedy_completion(model, tokenizer, enc, device, 12)
                core.validate_policy_counts(state, enc, 36, len(result['generated_token_ids']), 36, False)
                policy = dict(readout=state.applications, intermediate=state.intermediate_applications,
                              generated=state.generated_applications, restoration=state.restore_applications)
            row = dict(condition='wrong_description_last_token_disabled', template_id=enc.template_id,
                template_name=enc.template_name, entity_id=enc.entity_id, entity=enc.entity, entity_group=enc.entity_group,
                entity_description=enc.entity_description, original_description=case['entity_description'],
                prompt=enc.prompt, original_prompt=case['prompt'], prompt_token_ids=enc.ids, prompt_tokens=enc.tokens,
                entity_position=enc.entity_position, entity_token_id=enc.ids[enc.entity_position],
                description_positions=enc.description_positions, readout_position=len(enc.ids)-1,
                disabled_layers=list(range(36)), block_intermediate_prompt_entity=False, policy_counts=policy, **result)
            f.write(json.dumps(row)+'\n'); f.flush()
            print(f"{index+1}/400 {enc.template_name}/{enc.entity} ({enc.entity_description}): {result['completion']!r}", flush=True)
    metadata = dict(completed_at=datetime.now(timezone.utc).isoformat(), args=vars(args), completed_cases=len(prepared),
        ordinary_run=False, torch=torch.__version__, transformers=transformers.__version__, dtype=str(next(model.parameters()).dtype),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(), max_new_tokens=12, do_sample=False,
        policy='Last prompt and generated queries lose entity value in L0–35; intermediate queries native; no renormalization.',
        cases_sha256=hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (out / 'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('COMPLETE: raw outputs ready for manual review; no semantic score calculated.', flush=True)


if __name__ == '__main__':
    main()
