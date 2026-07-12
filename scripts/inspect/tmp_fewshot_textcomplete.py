"""TEMP control for the patchscopes experiment.

Instead of injecting a residual into the `x` slot of the few-shot description
template, we put the *literal entity text* there and let each d24@100B model
greedily complete it. This isolates "can the model do the few-shot describe
task at all (given the real text)?" from "can it unpack a patched residual?".
"""
import torch
from nanochat.checkpoint_manager import load_model

# Same few-shot template as patchscope_few_shot.TARGET_DEFAULT, minus the `x`.
PREFIX = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer "
    "electronics corporation, "
)
# Canonical entity phrases (same as --source-set canonical).
ENTITIES = [
    "Diana, princess of Wales",
    "Alexander the Great",
    "Muhammad Ali",
    "Jurassic Park",
    "New York City",
]
MODELS = [
    ("arch_d24_gpt_base_100B",                        "GPT-base d24 (100B)"),
    ("arch_d24_gpt_base_v_from_value_emb_learn_100B", "V-from-value-emb d24 (100B)"),
]
MAX_TOKENS = 30
DEVICE = torch.device("cuda")


def complete(model, tokenizer, prompt, max_tokens):
    ids = tokenizer.encode(prompt)
    out = []
    for tok in model.generate(list(ids), max_tokens=max_tokens, temperature=0.0):
        out.append(tok)
    return tokenizer.decode(out)


def main():
    for tag, label in MODELS:
        model, tokenizer, meta = load_model("base", DEVICE, phase="eval",
                                            model_tag=tag, step=None)
        print("=" * 90)
        print(f"MODEL: {label}   [{tag}, step {meta['step']}]")
        print("=" * 90)
        for ent in ENTITIES:
            prompt = PREFIX + ent
            cont = complete(model, tokenizer, prompt, MAX_TOKENS)
            cont_oneline = cont.split("\n")[0]
            print(f"\n  prompt tail : ...corporation, {ent}")
            print(f"  completion  : {ent}{cont_oneline}")
        del model
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
