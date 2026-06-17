"""Q/K vs V context-ablation on a standard pretrained model.

Tests, on a model that was NEVER trained to be context-free, whether the deep
layers actually need context in the value path the way they need it in the
query/key path. For a chosen set of layers and a chosen stream (q, k, or v),
we replace the INPUT to that projection with the token embedding (the layer-0
representation, which depends only on the token id, not the accumulated
context). The projection then produces a "context-free" q/k/v.

Crucial control (the lesson from the patchscope norm confound): the token
embedding has a much smaller norm than a deep-layer residual, and feeding an
out-of-scale vector breaks the computation by itself. So every replacement is
rescaled per position to match the norm of the residual it replaces. The only
thing that changes is the CONTENT (token identity vs contextualized), not the
scale.

We measure next-token cross-entropy on a held-out passage and report the damage
(delta CE vs the unablated baseline) for each (layer-region, stream).

Prediction if "deep value does not need context, Q/K do":
  - late-third V context-free  -> small damage
  - late-third Q or K context-free -> large damage
  - and late-third V damage < early-third V damage

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m scripts.inspect.qkv_context_ablation \
      --model Qwen/Qwen3-0.6B-Base --out results/qkv_ablation
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F


# A held-out passage of varied encyclopedic prose (in-distribution for a base
# model). Loss differences are measured on the SAME text across conditions, so
# this is a paired comparison; one passage is enough for a directional signal.
TEXT = """The history of the printing press is closely tied to the spread of literacy across Europe. Before movable type, books were copied by hand, a slow and expensive process that kept written knowledge in the hands of a small elite. When Johannes Gutenberg introduced his press in the middle of the fifteenth century, the cost of producing a book fell sharply, and within a few decades printing shops had opened in most major cities. The sudden abundance of printed material helped fuel the Reformation, the scientific revolution, and the rise of vernacular literature.

Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in sugars. Inside the chloroplast, pigments such as chlorophyll absorb photons and drive a chain of reactions that split water molecules, release oxygen, and ultimately fix carbon dioxide into glucose. This single process supplies almost all of the chemical energy that sustains life on Earth, and it is responsible for the oxygen that animals breathe.

The Roman Republic gradually expanded from a small city-state on the Italian peninsula into a power that controlled the entire Mediterranean basin. Its political system balanced the authority of elected magistrates, a powerful senate, and popular assemblies. Over time, the strains of governing a vast territory, combined with the ambitions of individual generals, eroded the old institutions, and the republic gave way to the rule of emperors.

In modern physics, the behavior of very small particles is described by quantum mechanics, a theory that often defies everyday intuition. Particles can exist in superpositions of states, and the act of measurement appears to collapse these possibilities into a single outcome. Despite its strangeness, quantum mechanics is one of the most precisely tested theories in all of science, underpinning technologies from lasers to semiconductors.

The Amazon rainforest stretches across several South American countries and contains an extraordinary diversity of plant and animal life. Its dense canopy regulates regional rainfall and stores enormous quantities of carbon. Scientists worry that deforestation, driven by logging and agriculture, could push the forest past a tipping point beyond which large areas would dry out and turn into savanna.

The development of vaccines stands as one of the great achievements of medicine. By exposing the immune system to a harmless piece of a pathogen, a vaccine trains the body to recognize and fight the real infection later. Widespread vaccination has eliminated smallpox, brought polio to the brink of eradication, and dramatically reduced the toll of measles and other childhood diseases."""


def load_hf(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.to(device).eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):       # Llama/Qwen
        layers = model.model.layers
        embed = model.model.embed_tokens
        attn_of = lambda blk: blk.self_attn
    elif hasattr(model, "gpt_neox"):                                      # Pythia
        layers = model.gpt_neox.layers
        embed = model.gpt_neox.embed_in
        attn_of = lambda blk: blk.attention
    else:
        raise RuntimeError(f"unknown architecture for {model_name}")
    return model, tok, layers, embed, attn_of


def proj_modules(attn, stream):
    """Return the projection module(s) for a stream, across arch naming."""
    names = {
        "q": ["q_proj", "query_key_value", "query"],
        "k": ["k_proj", "query_key_value", "key"],
        "v": ["v_proj", "query_key_value", "value"],
    }[stream]
    for n in names:
        if hasattr(attn, n):
            return n, getattr(attn, n)
    raise RuntimeError(f"no {stream} projection found on {type(attn)}")


def make_posthook(emb_holder):
    """Replace the projection OUTPUT with the projection of the token embedding,
    rescaled per position to the natural output's norm. Output-side matching
    treats q/k/v uniformly (Qwen3's q_norm/k_norm would otherwise re-normalize
    q/k but not v, confounding a V-vs-Q/K comparison done at the input)."""
    def hook(module, inp, output):
        x = inp[0]
        e = emb_holder["e"].to(x.dtype)
        if e.shape[:2] != output.shape[:2]:
            return output  # fused/packed layout -> skip
        free = F.linear(e, module.weight, getattr(module, "bias", None))  # bypass this hook
        scale = output.norm(dim=-1, keepdim=True) / free.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return free * scale
    return hook


@torch.no_grad()
def ce_loss(model, ids):
    logits = model(ids).logits[:, :-1, :].float()
    target = ids[:, 1:]
    return float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/qkv_ablation")
    args = ap.parse_args()
    device = torch.device(args.device)

    model, tok, layers, embed, attn_of = load_hf(args.model, device)
    n = len(layers)
    early = list(range(0, n // 3))
    late = list(range(2 * n // 3, n))
    print(f"{args.model}: {n} layers. early third {early[0]}-{early[-1]}, "
          f"late third {late[0]}-{late[-1]}", flush=True)

    ids = tok(TEXT, return_tensors="pt").input_ids.to(device)
    emb_holder = {"e": embed(ids)}  # context-free token embeddings, fixed
    print(f"passage: {ids.shape[1]} tokens", flush=True)

    base = ce_loss(model, ids)
    print(f"\nbaseline CE {base:.4f}  ppl {torch.tensor(base).exp():.2f}\n", flush=True)

    results = {"model": args.model, "n_layer": n, "baseline_ce": base,
               "early_third": [early[0], early[-1]],
               "late_third": [late[0], late[-1]], "conditions": {}}

    def run(region_name, region, stream):
        handles = []
        for li in region:
            attn = attn_of(layers[li])
            _, mod = proj_modules(attn, stream)
            handles.append(mod.register_forward_hook(make_posthook(emb_holder)))
        try:
            ce = ce_loss(model, ids)
        finally:
            for h in handles:
                h.remove()
        d = ce - base
        results["conditions"][f"{region_name}_{stream}"] = {"ce": ce, "delta": d}
        print(f"  {region_name:5s} {stream}  context-free:  CE {ce:.4f}  "
              f"dCE {d:+.4f}", flush=True)

    for region_name, region in [("early", early), ("late", late)]:
        for stream in ["q", "k", "v"]:
            run(region_name, region, stream)
        print(flush=True)

    os.makedirs(args.out, exist_ok=True)
    slug = args.model.replace("/", "_")
    with open(os.path.join(args.out, f"{slug}__qkv_ablation.json"), "w") as f:
        json.dump(results, f, indent=1)

    # Summary: the two key comparisons.
    c = results["conditions"]
    print("=== KEY COMPARISONS ===")
    print(f"late third:  V dCE {c['late_v']['delta']:+.4f}   "
          f"Q dCE {c['late_q']['delta']:+.4f}   K dCE {c['late_k']['delta']:+.4f}")
    print(f"  -> deep V {'<' if c['late_v']['delta'] < c['late_q']['delta'] else '>='} deep Q damage")
    print(f"V damage:  late {c['late_v']['delta']:+.4f}  vs  early {c['early_v']['delta']:+.4f}")
    print(f"saved {os.path.join(args.out, slug + '__qkv_ablation.json')}", flush=True)


if __name__ == "__main__":
    main()
