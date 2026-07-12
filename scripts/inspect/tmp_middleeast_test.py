"""TEMP: causal test for 'where does Middle East come from'. We swap the Syria
example's description from 'in the Middle East' to 'in Western Asia' and rerun
the literal-text completion. If the entity completions stop saying 'Middle
East', it was copied from the Syria in-context example (template bleed)."""
import torch
from nanochat.checkpoint_manager import load_model

PREFIX_ORIG = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer "
    "electronics corporation, "
)
PREFIX_SWAP = (
    "Syria: Country in Western Asia, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer "
    "electronics corporation, "
)
ENTITIES = ["Alexander the Great", "New York City", "Muhammad Ali"]
MODELS = [
    ("arch_d24_gpt_base_100B",                        "GPT-base d24 (100B)"),
    ("arch_d24_gpt_base_v_from_value_emb_learn_100B", "V-from-value-emb d24 (100B)"),
]
DEVICE = torch.device("cuda")


def complete(model, tok, prompt, n=20):
    ids = tok.encode(prompt)
    out = [t for t in model.generate(list(ids), max_tokens=n, temperature=0.0)]
    return tok.decode(out).split("\n")[0]


def main():
    for tag, label in MODELS:
        model, tok, meta = load_model("base", DEVICE, phase="eval",
                                      model_tag=tag, step=None)
        print("=" * 90)
        print(f"MODEL: {label}")
        print("=" * 90)
        for ent in ENTITIES:
            orig = complete(model, tok, PREFIX_ORIG + ent)
            swap = complete(model, tok, PREFIX_SWAP + ent)
            print(f"\n  {ent}")
            print(f"    Syria=Middle East : {ent}{orig}")
            print(f"    Syria=Western Asia: {ent}{swap}")
        del model
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
