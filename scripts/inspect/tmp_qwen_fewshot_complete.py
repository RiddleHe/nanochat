"""TEMP: 'standard completion' ceiling on Qwen3-8B-Base.

This is the limiting case of the patchscope few-shot setup: instead of patching
an intermediate-layer residual into the `x` slot, we put the literal entity text
there and let the model complete normally. Equivalent to 'patchscoping the last
activation, targeting the last layer' (the entity flows through all layers ->
final activation -> standard final_norm+lm_head readout). Apples-to-apples with
the d24 tmp_fewshot_textcomplete.py run, on a strong model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer "
    "electronics corporation, "
)
ENTITIES = [
    "Diana, princess of Wales",
    "Alexander the Great",
    "Muhammad Ali",
    "Jurassic Park",
    "New York City",
]
HF_MODEL = "Qwen/Qwen3-8B-Base"
MAX_TOKENS = 30
DEVICE = "cuda"


@torch.inference_mode()
def complete(model, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    print("=" * 90)
    print(f"MODEL: {HF_MODEL}  (n_layer={model.config.num_hidden_layers})")
    print("=" * 90)

    print("\n--- (A) Few-shot template + literal entity (matches d24 run) ---")
    for ent in ENTITIES:
        cont = complete(model, tok, PREFIX + ent, MAX_TOKENS).split("\n")[0]
        print(f"\n  {ent}{cont}")

    print("\n\n--- (B) Plain natural completion of the bare entity name ---")
    for ent in ENTITIES:
        cont = complete(model, tok, ent, MAX_TOKENS).split("\n")[0]
        print(f"\n  {ent}{cont}")


if __name__ == "__main__":
    main()
