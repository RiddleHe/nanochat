"""Train a linear layer translator and evaluate regular few-shot patchscope.

The translator is a single nn.Linear(d, d) trained on ordinary hidden-state
reconstruction pairs: source-layer residual at an entity-token position maps to
readout-layer residual at the same position. Evaluation then compares true,
raw, translated, random, and wrong-entity vectors under the original regular
few-shot patchscope target prompt.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scripts.inspect.patchscope_few_shot import (
        CRITERIA,
        ENTITIES,
        ENTITY_TITLES,
        SOURCE_SETS,
        TARGET_DEFAULT,
    )
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from scripts.inspect.patchscope_few_shot import (  # noqa: E402
        CRITERIA,
        ENTITIES,
        ENTITY_TITLES,
        SOURCE_SETS,
        TARGET_DEFAULT,
    )


HELD_OUT_NAMES = {
    "Diana, princess of Wales",
    "Alexander the Great",
    "Muhammad Ali",
    "Jurassic Park",
    "New York City",
}

GENERIC_ENTITIES = [
    "Ada Lovelace", "Alan Turing", "Albert Einstein", "Amelia Earhart",
    "Aristotle", "Barack Obama", "Beyonce", "Bill Gates", "Charles Darwin",
    "Cleopatra", "Cristiano Ronaldo", "Elon Musk", "Florence Nightingale",
    "Frida Kahlo", "Galileo Galilei", "George Washington", "Isaac Newton",
    "Jane Austen", "Joan of Arc", "Leonardo da Vinci", "Mahatma Gandhi",
    "Marie Curie", "Martin Luther King Jr.", "Maya Angelou", "Michael Jordan",
    "Nelson Mandela", "Nikola Tesla", "Pablo Picasso", "Queen Victoria",
    "Rosa Parks", "Serena Williams", "Socrates", "Taylor Swift",
    "Vincent van Gogh", "William Shakespeare", "Wolfgang Amadeus Mozart",
    "Amazon", "Apple Inc.", "BBC", "BMW", "Boeing", "Coca-Cola", "Disney",
    "Google", "Honda", "IBM", "IKEA", "Intel", "Microsoft", "NASA",
    "Netflix", "Nintendo", "Nokia", "OpenAI", "Pepsi", "Samsung", "Sony",
    "Toyota", "UNESCO", "United Nations", "Volkswagen", "Walmart",
    "Wikipedia", "World Health Organization", "YouTube", "Zurich",
    "Argentina", "Australia", "Brazil", "Canada", "China", "Egypt",
    "France", "Germany", "India", "Indonesia", "Italy", "Japan", "Kenya",
    "Mexico", "Morocco", "Nigeria", "Norway", "Peru", "Portugal",
    "South Africa", "South Korea", "Spain", "Sweden", "Thailand", "Turkey",
    "United Kingdom", "United States", "Vietnam", "Amsterdam", "Athens",
    "Bangkok", "Barcelona", "Beijing", "Berlin", "Boston", "Buenos Aires",
    "Cairo", "Chicago", "Delhi", "Dubai", "Hong Kong", "Istanbul",
    "Jerusalem", "Kyoto", "London", "Los Angeles", "Madrid", "Melbourne",
    "Miami", "Moscow", "Mumbai", "Paris", "Prague", "Rio de Janeiro",
    "Rome", "San Francisco", "Seoul", "Shanghai", "Singapore", "Sydney",
    "Tokyo", "Toronto", "Vienna", "Washington, D.C.", "The Godfather",
    "Star Wars", "Titanic", "The Matrix", "Casablanca", "Hamlet",
    "Pride and Prejudice", "Moby-Dick", "The Odyssey", "The Beatles",
    "The Rolling Stones", "World War II", "French Revolution",
    "Industrial Revolution", "Cold War", "Apollo 11", "Mount Everest",
    "Pacific Ocean", "Amazon River", "Sahara Desert", "Great Barrier Reef",
]

TRAIN_TEMPLATES = [
    "{entity}",
    "The topic is {entity}",
    "A short article about {entity}",
    "This passage discusses {entity}",
    "Important facts about {entity}",
]

STRICT_ALIASES = {
    "diana": ["diana", "princess of wales"],
    "alexander": ["alexander the great"],
    "ali": ["muhammad ali"],
    "jurassic": ["jurassic park"],
    "nyc": ["new york city", "nyc"],
}

CATEGORY_ALIASES = {
    "diana": ["princess", "royal", "royalty", "british royal"],
    "alexander": ["king", "conqueror", "macedon", "macedonian", "ancient greek"],
    "ali": ["boxer", "boxing", "heavyweight", "athlete"],
    "jurassic": ["film", "movie", "dinosaur", "spielberg", "science fiction"],
    "nyc": ["city", "metropolis", "new york", "united states", "manhattan"],
}

CONDITIONS = [
    "oracle_readout",
    "raw_late",
    "translated_late",
    "random_norm_control",
    "wrong_entity_control",
]


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_hf(hf_model, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model.to(device)
    model.eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        blocks = model.gpt_neox.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    else:
        raise RuntimeError(f"Cannot locate transformer blocks for {hf_model}")

    def encode(text):
        return tok.encode(text, add_special_tokens=False)

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_tokens(ids, max_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
        out = model.generate(
            x,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=eos if eos is not None else 0,
        )
        return out[0, x.shape[1]:].tolist()

    return {
        "model": model,
        "tokenizer": tok,
        "blocks": blocks,
        "n_layer": len(blocks),
        "encode": encode,
        "decode": decode,
        "generate_tokens": generate_tokens,
        "name": hf_model.replace("/", "_"),
    }


@torch.inference_mode()
def capture_layers(adapter, ids, pos, layers, device):
    wanted = set(layers)
    hiddens = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            hiddens[idx] = tensor_out[0, pos, :].detach().float().cpu()
        return hook

    for idx in wanted:
        handles.append(adapter["blocks"][idx].register_forward_hook(make_hook(idx)))
    try:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _ = adapter["model"](x)
    finally:
        for handle in handles:
            handle.remove()
    missing = wanted.difference(hiddens)
    if missing:
        raise RuntimeError(f"Did not capture layers: {sorted(missing)}")
    return hiddens


def entity_position(adapter, text):
    ids = adapter["encode"](text)
    if not ids:
        raise ValueError(f"Prompt tokenized to empty ids: {text!r}")
    return ids, len(ids) - 1


def make_examples(limit=None):
    entities = [e for e in GENERIC_ENTITIES if e not in HELD_OUT_NAMES]
    rows = []
    for entity in entities:
        for template in TRAIN_TEMPLATES:
            rows.append({"entity": entity, "prompt": template.format(entity=entity)})
    if limit is not None:
        rows = rows[:limit]
    return rows


def split_examples(examples, val_fraction, seed):
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def collect_pairs(adapter, examples, source_layer, readout_layer, device):
    xs, ys = [], []
    for i, row in enumerate(examples, 1):
        ids, pos = entity_position(adapter, row["prompt"])
        captured = capture_layers(adapter, ids, pos, [source_layer, readout_layer], device)
        xs.append(captured[source_layer])
        ys.append(captured[readout_layer])
        if i % 100 == 0:
            print(f"[pairs] collected {i}/{len(examples)}", flush=True)
    return torch.stack(xs), torch.stack(ys)


def metrics(model, x, y, batch_size, device):
    model.eval()
    total_mse = 0.0
    total_cos = 0.0
    n = 0
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            pred = model(xb)
            bs = xb.shape[0]
            total_mse += F.mse_loss(pred, yb, reduction="mean").item() * bs
            total_cos += F.cosine_similarity(pred, yb, dim=-1).mean().item() * bs
            n += bs
    return {"mse": total_mse / n, "cosine": total_cos / n}


def train_translator(train_x, train_y, val_x, val_y, args, device):
    d_model = train_x.shape[1]
    translator = nn.Linear(d_model, d_model, bias=not args.no_bias).to(device)
    opt = torch.optim.AdamW(translator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        translator.train()
        perm = torch.randperm(train_x.shape[0], generator=generator)
        total = 0.0
        n = 0
        for start in range(0, train_x.shape[0], args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)
            pred = translator(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = xb.shape[0]
            total += loss.item() * bs
            n += bs
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            tr = metrics(translator, train_x, train_y, args.batch_size, device)
            va = metrics(translator, val_x, val_y, args.batch_size, device)
            row = {
                "epoch": epoch,
                "train_loss": total / n,
                "train_mse": tr["mse"],
                "train_cosine": tr["cosine"],
                "val_mse": va["mse"],
                "val_cosine": va["cosine"],
            }
            rows.append(row)
            print(
                f"[train] epoch={epoch} train_mse={tr['mse']:.6g} "
                f"train_cos={tr['cosine']:.4f} val_mse={va['mse']:.6g} "
                f"val_cos={va['cosine']:.4f}",
                flush=True,
            )
    return translator, rows


def norm_match(vec, target_norm):
    return vec * (target_norm / (vec.norm() + 1e-8))


def patched_generate(adapter, tgt_ids, target_layer, tgt_pos, patch_vec, max_tokens, device):
    patch_norm = None
    target_norm = None

    def patch_hook(_mod, _inp, out):
        nonlocal patch_norm, target_norm
        is_tuple = isinstance(out, tuple)
        x = out[0] if is_tuple else out
        if x.shape[1] <= tgt_pos:
            return out
        new_x = x.clone()
        normal = new_x[0, tgt_pos, :].float()
        target_norm = float(normal.norm().item())
        v = norm_match(patch_vec.to(device=device, dtype=torch.float32), normal.norm())
        patch_norm = float(v.norm().item())
        new_x[0, tgt_pos, :] = v.to(new_x.dtype)
        return (new_x,) + out[1:] if is_tuple else new_x

    handle = adapter["blocks"][target_layer].register_forward_hook(patch_hook)
    try:
        gen = adapter["generate_tokens"](tgt_ids, max_tokens)
        text = adapter["decode"](gen).lstrip().split("\n")[0]
    finally:
        handle.remove()
    return text, target_norm, patch_norm


def score_original(ent_key, text):
    t = text.lower()
    criteria = CRITERIA[ent_key]
    if any(neg in t for neg in criteria["neg"]):
        return 0
    return int(any(pos in t for pos in criteria["pos"]))


def score_aliases(ent_key, text, aliases):
    t = text.lower()
    return int(any(alias in t for alias in aliases[ent_key]))


def eval_vectors(adapter, translator, source_layer, readout_layer, args, device):
    target_ids = adapter["encode"](args.target)
    if len(target_ids) < 2:
        raise ValueError(f"Target prompt too short after tokenization: {args.target!r}")
    tgt_pos = len(target_ids) - 1
    translator.eval()

    captured_by_entity = {}
    translated_by_entity = {}
    for ent_key in ENTITIES:
        prompt = SOURCE_SETS["canonical"][ent_key]
        ids, src_pos = entity_position(adapter, prompt)
        captured = capture_layers(adapter, ids, src_pos, [source_layer, readout_layer], device)
        h_l = captured[source_layer]
        h_k = captured[readout_layer]
        with torch.inference_mode():
            translated = translator(h_l.to(device)).detach().float().cpu()
        captured_by_entity[ent_key] = {"h_l": h_l, "h_k": h_k}
        translated_by_entity[ent_key] = translated

    rows = []
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    for i, ent_key in enumerate(ENTITIES):
        wrong_key = ENTITIES[(i + 1) % len(ENTITIES)]
        h_l = captured_by_entity[ent_key]["h_l"]
        h_k = captured_by_entity[ent_key]["h_k"]
        random_vec = torch.randn(h_k.shape, generator=generator)
        random_vec = norm_match(random_vec, h_k.norm())
        condition_vecs = {
            "oracle_readout": h_k,
            "raw_late": h_l,
            "translated_late": translated_by_entity[ent_key],
            "random_norm_control": random_vec,
            "wrong_entity_control": translated_by_entity[wrong_key],
        }
        for condition in CONDITIONS:
            text, target_norm, patch_norm = patched_generate(
                adapter,
                target_ids,
                readout_layer,
                tgt_pos,
                condition_vecs[condition],
                args.max_tokens,
                device,
            )
            row = {
                "entity_key": ent_key,
                "entity": ENTITY_TITLES[ent_key],
                "condition": condition,
                "source_layer": source_layer,
                "readout_layer": readout_layer,
                "target_prompt": args.target,
                "source_prompt": SOURCE_SETS["canonical"][ent_key],
                "wrong_entity_key": wrong_key if condition == "wrong_entity_control" else "",
                "wrong_entity": ENTITY_TITLES[wrong_key] if condition == "wrong_entity_control" else "",
                "max_tokens": args.max_tokens,
                "generated_text": text,
                "score_original_substring": score_original(ent_key, text),
                "score_strict_entity_name": score_aliases(ent_key, text, STRICT_ALIASES),
                "score_category_description": score_aliases(ent_key, text, CATEGORY_ALIASES),
                "normal_target_norm": target_norm,
                "patched_vector_norm": patch_norm,
                "source_vec_norm": float(condition_vecs[condition].norm().item()),
            }
            rows.append(row)
            print(
                f"[eval] {ent_key} {condition}: "
                f"orig={row['score_original_substring']} strict={row['score_strict_entity_name']} "
                f"cat={row['score_category_description']} text={text!r}",
                flush=True,
            )
    return rows


def summarize(rows, group_fields):
    score_fields = [
        "score_original_substring",
        "score_strict_entity_name",
        "score_category_description",
    ]
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item["n"] = len(group)
        for field in score_fields:
            item[field + "_hit_rate"] = sum(int(r[field]) for r in group) / len(group)
        out.append(item)
    return out


def make_readme(args, out_dir, command):
    text = f"""# Layer Translator Patchscope

Created: {datetime.now(timezone.utc).isoformat()}

Model: `{args.hf_model}`

Source layer: `{args.source_layer}`

Readout layer: `{args.readout_layer}`

The translator is one `nn.Linear(d, d)` trained to reconstruct readout-layer
hidden states from source-layer hidden states at the same entity-token position.
The five regular patchscope entities are excluded from train and validation.

Evaluation uses the regular few-shot target prompt and `max_tokens={args.max_tokens}`.
All patched vectors are norm-matched to the normal readout-layer hidden-state
norm at the target placeholder position before injection.

Command:

```bash
{command}
```
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)


def run_smoke_test(args):
    os.makedirs(args.out_dir, exist_ok=True)
    examples = make_examples(limit=20)
    train_examples, val_examples = split_examples(examples, 0.2, args.seed)
    d = 16
    train_x = torch.randn(len(train_examples), d)
    true_w = torch.randn(d, d) / math.sqrt(d)
    train_y = train_x @ true_w
    val_x = torch.randn(len(val_examples), d)
    val_y = val_x @ true_w
    translator, metric_rows = train_translator(train_x, train_y, val_x, val_y, args, torch.device("cpu"))
    metric_path = os.path.join(args.out_dir, "train_val_metrics.csv")
    write_csv(metric_path, metric_rows, list(metric_rows[0].keys()))
    torch.save(translator.state_dict(), os.path.join(args.out_dir, "translator_weights.pt"))
    cfg = vars(args).copy()
    cfg["train_examples"] = len(train_examples)
    cfg["val_examples"] = len(val_examples)
    write_json(os.path.join(args.out_dir, "translator_config.json"), cfg)
    sample_rows = [{
        "entity_key": "diana",
        "entity": "Diana, princess of Wales",
        "condition": "translated_late",
        "source_layer": args.source_layer,
        "readout_layer": args.readout_layer,
        "target_prompt": args.target,
        "source_prompt": "Diana, princess of Wales",
        "wrong_entity_key": "",
        "wrong_entity": "",
        "max_tokens": args.max_tokens,
        "generated_text": "British royal and princess",
        "score_original_substring": 1,
        "score_strict_entity_name": 0,
        "score_category_description": 1,
        "normal_target_norm": 1.0,
        "patched_vector_norm": 1.0,
        "source_vec_norm": 1.0,
    }]
    eval_fields = list(sample_rows[0].keys())
    write_csv(os.path.join(args.out_dir, "eval_generations.csv"), sample_rows, eval_fields)
    write_csv(os.path.join(args.out_dir, "eval_summary.csv"), summarize(sample_rows, ["condition"]),
              ["condition", "n", "score_original_substring_hit_rate",
               "score_strict_entity_name_hit_rate", "score_category_description_hit_rate"])
    write_csv(os.path.join(args.out_dir, "eval_summary_by_entity.csv"),
              summarize(sample_rows, ["entity_key", "entity", "condition"]),
              ["entity_key", "entity", "condition", "n", "score_original_substring_hit_rate",
               "score_strict_entity_name_hit_rate", "score_category_description_hit_rate"])
    with open(os.path.join(args.out_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    make_readme(args, args.out_dir, " ".join(sys.argv))
    print(f"[smoke] wrote smoke artifacts to {args.out_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--source-layer", type=int, required=True)
    ap.add_argument("--readout-layer", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    command = " ".join(sys.argv)
    with open(os.path.join(args.out_dir, "command.txt"), "w") as f:
        f.write(command + "\n")

    if args.smoke_test:
        run_smoke_test(args)
        return

    device = torch.device(args.device)
    adapter = load_hf(args.hf_model, device)
    if not (0 <= args.source_layer < adapter["n_layer"]):
        raise ValueError(f"source layer {args.source_layer} out of range 0..{adapter['n_layer'] - 1}")
    if not (0 <= args.readout_layer < adapter["n_layer"]):
        raise ValueError(f"readout layer {args.readout_layer} out of range 0..{adapter['n_layer'] - 1}")

    examples = make_examples(limit=args.train_limit)
    train_examples, val_examples = split_examples(examples, args.val_fraction, args.seed)
    print(
        f"[setup] model={args.hf_model} n_layer={adapter['n_layer']} "
        f"source_layer={args.source_layer} readout_layer={args.readout_layer} "
        f"train={len(train_examples)} val={len(val_examples)}",
        flush=True,
    )
    train_x, train_y = collect_pairs(adapter, train_examples, args.source_layer, args.readout_layer, device)
    val_x, val_y = collect_pairs(adapter, val_examples, args.source_layer, args.readout_layer, device)
    translator, metric_rows = train_translator(train_x, train_y, val_x, val_y, args, device)

    metric_fields = ["epoch", "train_loss", "train_mse", "train_cosine", "val_mse", "val_cosine"]
    write_csv(os.path.join(args.out_dir, "train_val_metrics.csv"), metric_rows, metric_fields)
    torch.save(translator.state_dict(), os.path.join(args.out_dir, "translator_weights.pt"))

    cfg = vars(args).copy()
    cfg.update({
        "model_name": adapter["name"],
        "n_layer": adapter["n_layer"],
        "d_model": train_x.shape[1],
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "held_out_entities": sorted(HELD_OUT_NAMES),
        "train_templates": TRAIN_TEMPLATES,
        "conditions": CONDITIONS,
    })
    write_json(os.path.join(args.out_dir, "translator_config.json"), cfg)

    eval_rows = eval_vectors(adapter, translator, args.source_layer, args.readout_layer, args, device)
    eval_fields = [
        "entity_key", "entity", "condition", "source_layer", "readout_layer",
        "target_prompt", "source_prompt", "wrong_entity_key", "wrong_entity",
        "max_tokens", "generated_text", "score_original_substring",
        "score_strict_entity_name", "score_category_description",
        "normal_target_norm", "patched_vector_norm", "source_vec_norm",
    ]
    write_csv(os.path.join(args.out_dir, "eval_generations.csv"), eval_rows, eval_fields)

    summary_fields = [
        "condition", "n", "score_original_substring_hit_rate",
        "score_strict_entity_name_hit_rate", "score_category_description_hit_rate",
    ]
    write_csv(os.path.join(args.out_dir, "eval_summary.csv"),
              summarize(eval_rows, ["condition"]), summary_fields)
    by_entity_fields = [
        "entity_key", "entity", "condition", "n",
        "score_original_substring_hit_rate", "score_strict_entity_name_hit_rate",
        "score_category_description_hit_rate",
    ]
    write_csv(os.path.join(args.out_dir, "eval_summary_by_entity.csv"),
              summarize(eval_rows, ["entity_key", "entity", "condition"]), by_entity_fields)
    make_readme(args, args.out_dir, command)
    print(f"[done] results written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
