"""Extended base-model evaluation via lm-evaluation-harness.

Runs perplexity-based (multiple_choice) tasks on a nanochat gpt_base checkpoint
and aggregates avg@N across few-shot orderings to reduce variance — matching
the methodology in the Ling 2.0 paper (https://arxiv.org/pdf/2510.26692) for
base-model evaluation: "we employ perplexity-based evaluation for MMLU,
MMLU-Redux, GPQA-Diamond, and C-Eval ... to mitigate the high variance ... we
report the mean score across N independent runs."

Default tasks: mmlu (5-shot via task default), gpqa_diamond_zeroshot (0-shot).
MMLU-Redux is intentionally omitted — lm-eval-harness 0.4.x ships only a
generative variant which is the wrong harness for tiny base models. If needed,
write a custom loglikelihood YAML for mmlu_redux and pass via --include-path.

Single-GPU per invocation. For multi-model parallelism, launch multiple
instances pinned via CUDA_VISIBLE_DEVICES.

Output: {output_dir}/{model_tag}.csv  with columns
        task, mean_acc, std, n_seeds, per_seed (semicolon-separated values).

Examples:
    # smoke test on 50 examples, 1 seed:
    CUDA_VISIBLE_DEVICES=4 uv run python -m scripts.base_eval_extended \\
        --model-tag arch_d24_gpt_base --num-seeds 1 --limit 50

    # full avg@8 eval:
    CUDA_VISIBLE_DEVICES=4 uv run python -m scripts.base_eval_extended \\
        --model-tag arch_d24_gpt_base
"""
import argparse
import csv
import os
import statistics
import time

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import LM

from nanochat.checkpoint_manager import load_model


class NanochatLM(LM):
    """Minimal lm-eval-harness LM adapter for nanochat base models.

    Only implements loglikelihood — sufficient for multiple_choice tasks like
    MMLU and GPQA-Diamond. loglikelihood_rolling and generate_until raise.
    Single-process, single-GPU. Right-pads with EOS; truncates from left if
    context+continuation exceeds sequence_len.
    """

    def __init__(self, model, tokenizer, device, batch_size=16, max_length=None):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._device = device
        self.batch_size = batch_size
        self.max_length = max_length or model.config.sequence_len
        self.vocab_size = tokenizer.get_vocab_size()
        self.eos_token_id = tokenizer.get_bos_token_id()

    # lm-eval-harness LM exposes these as read-only properties; override.
    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    @torch.no_grad()
    def _forward_logits(self, input_ids):
        return self.model(input_ids).float()  # (B, T, V)

    def loglikelihood(self, requests):
        """Return list of (logprob_sum_over_continuation, is_greedy)."""
        results = [None] * len(requests)
        encoded = []
        for idx, req in enumerate(requests):
            ctx_str, cont_str = req.args
            ctx_ids = self.tokenizer.encode(ctx_str)
            cont_ids = self.tokenizer.encode(cont_str)
            # Always prepend BOS — every nanochat training document begins with
            # BOS, so test-time prompts need the same framing or the model
            # produces repetitive degenerate output (verified at d24 sanity-check).
            ctx_ids = [self.eos_token_id] + ctx_ids
            full = ctx_ids + cont_ids
            if len(full) > self.max_length:
                excess = len(full) - self.max_length
                full = full[excess:]
                ctx_len = max(1, len(ctx_ids) - excess)
            else:
                ctx_len = len(ctx_ids)
            cont_len = len(full) - ctx_len
            encoded.append((idx, full, ctx_len, cont_len))
        # Sort longest-first for better batching
        encoded.sort(key=lambda x: -len(x[1]))
        for start in range(0, len(encoded), self.batch_size):
            batch = encoded[start:start + self.batch_size]
            max_len = max(len(e[1]) for e in batch)
            B = len(batch)
            input_ids = torch.full((B, max_len), self.eos_token_id,
                                   dtype=torch.long, device=self._device)
            for i, (_, full, _, _) in enumerate(batch):
                input_ids[i, :len(full)] = torch.tensor(full, dtype=torch.long, device=self._device)
            logits = self._forward_logits(input_ids)
            for i, (orig_idx, full, ctx_len, cont_len) in enumerate(batch):
                # logits at pos p predict token at p+1 → continuation predictions
                # are logits[ctx_len-1 : ctx_len-1+cont_len].
                pred = logits[i, ctx_len-1:ctx_len-1+cont_len, :]
                targets = torch.tensor(full[ctx_len:ctx_len+cont_len],
                                       dtype=torch.long, device=self._device)
                logp = F.log_softmax(pred, dim=-1)
                tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                total = float(tok_logp.sum().item())
                is_greedy = bool((pred.argmax(dim=-1) == targets).all().item())
                results[orig_idx] = (total, is_greedy)
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented "
                                  "(this adapter is for multiple_choice tasks)")

    def generate_until(self, requests):
        raise NotImplementedError("generate_until not implemented "
                                  "(this adapter is for multiple_choice tasks)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-tag', required=True,
                   help='checkpoint dir name under base_checkpoints/, e.g. arch_d24_gpt_base')
    p.add_argument('--step', type=int, default=None,
                   help='model step to load (default: last)')
    p.add_argument('--tasks', type=str, default='mmlu,gpqa_diamond_zeroshot',
                   help='comma-separated lm-eval-harness task names')
    p.add_argument('--num-seeds', type=int, default=8,
                   help='number of seeds for avg@N variance reduction')
    p.add_argument('--batch-size', type=int, default=16,
                   help='loglikelihood batch size')
    p.add_argument('--limit', type=int, default=None,
                   help='limit examples per task (for smoke testing)')
    p.add_argument('--output-dir', type=str,
                   default='/local-ssd/mh3897/base_eval_extended')
    args = p.parse_args()

    assert torch.cuda.is_available(), 'requires CUDA'
    device = torch.device('cuda')
    print(f'[base_eval_extended] loading {args.model_tag}  device={device}')
    model, tokenizer, meta = load_model('base', device, phase='eval',
                                        model_tag=args.model_tag, step=args.step)
    wrapper = NanochatLM(model, tokenizer, device, batch_size=args.batch_size)

    # Coherence sanity-check: generate from a fixed prompt and print. If output
    # is gibberish (random chars / pure repetition), the checkpoint or wrapper is
    # broken and there's no point running the full eval.
    sanity_prompt = "The capital of France is"
    bos = tokenizer.get_bos_token_id()
    sanity_ids = [bos] + tokenizer.encode(sanity_prompt)
    gen = list(model.generate(sanity_ids, max_tokens=40, temperature=0.0))
    completion = tokenizer.decode(sanity_ids + gen)
    print(f'\n[sanity prompt + greedy completion]\n  {completion!r}\n')
    # Bail out if the completion looks degenerate (single token repeated 5+ times),
    # which indicates the wrapper/tokenizer is misconfigured.
    decoded_words = completion.split()
    if len(decoded_words) >= 5 and len(set(decoded_words[-5:])) == 1:
        print('[sanity FAILED] degenerate repetition detected — aborting before eval')
        raise SystemExit(2)

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    print(f'[base_eval_extended] tasks={tasks}  num_seeds={args.num_seeds}  '
          f'batch_size={args.batch_size}  limit={args.limit}')

    per_seed = {}
    for seed in range(args.num_seeds):
        t0 = time.time()
        res = simple_evaluate(
            model=wrapper,
            tasks=tasks,
            num_fewshot=None,  # use each task's default
            batch_size=args.batch_size,
            limit=args.limit,
            random_seed=seed,
            numpy_random_seed=1234 + seed,
            torch_random_seed=1234 + seed,
            fewshot_random_seed=1234 + seed,
        )
        elapsed = time.time() - t0
        print(f'\n[seed {seed}] took {elapsed:.1f}s')
        for task_name, metrics in res['results'].items():
            acc = metrics.get('acc,none', metrics.get('acc'))
            if acc is None:
                acc = next((v for k, v in metrics.items()
                            if k.startswith('acc')), None)
            print(f'  {task_name}: acc={acc}')
            per_seed.setdefault(task_name, []).append(float(acc))

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f'{args.model_tag}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'mean_acc', 'std', 'n_seeds', 'per_seed'])
        for task, vals in per_seed.items():
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
            w.writerow([task, f'{mean:.4f}', f'{std:.4f}', len(vals),
                        ';'.join(f'{v:.4f}' for v in vals)])
    print(f'\n[base_eval_extended] saved {out_csv}')


if __name__ == '__main__':
    main()
