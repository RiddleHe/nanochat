"""TEMP: language-coherence check for the two d24@100B models, replicating the
sampling block in scripts/base_eval.py (conditioned greedy + unconditioned
temp=1.0 free-running). Lets us see raw fluency outside the few-shot task."""
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

MODELS = [
    ("arch_d24_gpt_base_100B",                        "GPT-base d24 (100B)"),
    ("arch_d24_gpt_base_v_from_value_emb_learn_100B", "V-from-value-emb d24 (100B)"),
]
PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]
DEVICE = torch.device("cuda")


def run(tag, label):
    model, tokenizer, meta = load_model("base", DEVICE, phase="eval",
                                        model_tag=tag, step=None)
    engine = Engine(model, tokenizer)
    print("=" * 90)
    print(f"MODEL: {label}   [{tag}, step {meta['step']}, val_bpb {meta['val_bpb']:.4f}]")
    print("=" * 90)

    print("\n--- Conditioned (greedy, 16 tok) ---")
    for prompt in PROMPTS:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16,
                                          temperature=0)
        print("  " + tokenizer.decode(sample[0]).replace("\n", " ⏎ "))

    print("\n--- Unconditioned (temp=1.0, 128 tok, 5 samples) ---")
    torch.manual_seed(0)  # same RNG draws across both models
    tokens = tokenizer("", prepend="<|bos|>")
    uncond, _ = engine.generate_batch(tokens, num_samples=5, max_tokens=128,
                                      temperature=1.0)
    for i, s in enumerate(uncond):
        print(f"\n  [sample {i}] " + tokenizer.decode(s).replace("\n", " ⏎ "))

    del model
    torch.cuda.empty_cache()
    print()


def main():
    for tag, label in MODELS:
        run(tag, label)


if __name__ == "__main__":
    main()
