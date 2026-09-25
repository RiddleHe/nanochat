"""Span-aware relay using the user's frozen 2026-09-23 attention implementation."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import torch
from scripts.inspect import qwen_entity_relay_fixed_window as relay
from scripts.inspect.qwen_relay_paper_runtime import clean, hashes, configure_model

MATERIALS = Path(__file__).resolve().parent
DATA = MATERIALS / 'entity_attention_data' / 'entities100_balanced12.json'
SOURCE = MATERIALS / 'qwen_entity_attention_ablation.py'
_spec = importlib.util.spec_from_file_location('relay_frozen_ablation_20260923', SOURCE)
ab = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ab
_spec.loader.exec_module(ab)
TIDS = [0, 1, 2, 3, 7]
EXTRA_RECIPIENTS = ['John', 'apple', 'chair', 'table', 'x', 'X', 'a', 'A']


def validate_pair(donor, recipient):
    if donor.entity_positions != recipient.entity_positions or len(donor.ids) != len(recipient.ids):
        raise ValueError(f'Unequal spans or sequence lengths: {donor.entity}/{recipient.entity}')
    span = set(donor.entity_positions)
    if any(a != b and i not in span for i, (a, b) in enumerate(zip(donor.ids, recipient.ids))):
        raise ValueError(f'Tokens differ outside entity: {donor.entity}/{recipient.entity}')


def inputs(tok, extras=True):
    entities, _ = ab.load_entities(DATA)
    names = [n for n, _ in entities]
    assert len(names) == len(set(names)) == 100
    all_entities = entities + [(n, 'extra_diagnostic') for n in EXTRA_RECIPIENTS if n not in names] if extras else entities
    prompts = {(tid, n): ab.encode_prompt(tok, ab.TEMPLATES[tid], i, n, group)
               for tid in TIDS for i, (n, group) in enumerate(all_entities)}
    for tid in TIDS:
        for i, name in enumerate(names):
            enc = prompts[tid, name]
            assert len(enc.entity_positions) == (1 if i < 50 else 2), (tid, name, enc.entity_positions)
            validate_pair(enc, prompts[tid, names[i ^ 1]])
        for n, _ in all_entities:
            enc = prompts[tid, n]
            assert not set(enc.ids[p] for p in enc.entity_positions).intersection(tok.all_special_ids)
            if n not in names:
                validate_pair(enc, prompts[tid, names[0]])
    return names, [n for n, _ in all_entities], prompts


def policy(enc, disabled=(), middle=True):
    return ab.disable_entity_attention(disabled, enc.entity_position, len(enc.ids)-1,
                                       len(enc.ids), middle, entity_positions=enc.entity_positions)


def counts(pol, enc, disabled, middle, n_layers, generated_n=1):
    ab.validate_policy_counts(pol, enc, n_layers, generated_n, len(disabled), middle)
    return dict(zip(('readout', 'intermediate', 'generated', 'restore'),
                    (pol.applications, pol.intermediate_applications,
                     pol.generated_applications, pol.restore_applications)))


@torch.inference_mode()
def generate(model, tok, layers, enc, device, disabled=(), middle=True,
             state=None, t=None, ordinary=False):
    with (nullcontext() if ordinary else policy(enc, disabled, middle)) as pol:
        result = relay.greedy_completion(model, tok, layers, enc, device, 12,
                                         relay_layer=t, relay_state=state)
        result['policy_counts'] = None if ordinary else counts(
            pol, enc, disabled, middle, len(layers), len(result['generated_token_ids']))
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [tok.eos_token_id])
    result['stop_reason'] = 'eos' if result['generated_token_ids'][-1] in eos else 'max_new_tokens'
    result['clean_output'] = clean(result['completion'])
    return result


@torch.inference_mode()
def capture(model, layers, enc, device, disabled=(), middle=True):
    with policy(enc, disabled, middle) as pol:
        states = relay.capture_donor_entity_states(model, layers, enc, device)
        pc = counts(pol, enc, disabled, middle, len(layers))
    assert all(x.shape[0] == len(enc.entity_positions) for x in states)
    return states, pc


@torch.inference_mode()
def run_relay(model, tok, layers, enc, device, source, s, t, disabled=(), middle=True):
    if source.shape != (len(enc.entity_positions), model.config.hidden_size):
        raise ValueError('Source must cover the entire aligned entity span')
    with policy(enc, disabled, middle) as pol:
        state = relay.build_relay_state(model, layers, enc, s, t, source, device)
        pc = counts(pol, enc, disabled, middle, len(layers))
    gen = generate(model, tok, layers, enc, device, disabled, middle, state, t)
    gen['pass2_policy_counts'] = pc
    return gen
