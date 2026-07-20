"""Adaptive adapter search for late-to-readout patchscope vectors.

This script is intentionally standalone: it does not modify or depend on the
previous dense translator experiment. It collects ordinary hidden-state pairs
from many generic prompts, fits constrained adapters from late source layers to
readout layers, and evaluates each adapter with the regular few-shot
patchscope target prompt.
"""
import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
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

BASE_ENTITIES = [
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
    "Netflix", "Nintendo", "Nokia", "OpenAI", "Pepsi", "Sony", "Toyota",
    "UNESCO", "United Nations", "Volkswagen", "Walmart", "Wikipedia",
    "World Health Organization", "YouTube", "Zurich", "Argentina",
    "Australia", "Brazil", "Canada", "China", "Egypt", "France", "Germany",
    "India", "Indonesia", "Italy", "Japan", "Kenya", "Mexico", "Morocco",
    "Nigeria", "Norway", "Peru", "Portugal", "South Africa", "South Korea",
    "Spain", "Sweden", "Thailand", "Turkey", "United Kingdom",
    "United States", "Vietnam", "Amsterdam", "Athens", "Bangkok",
    "Barcelona", "Beijing", "Berlin", "Boston", "Buenos Aires", "Cairo",
    "Chicago", "Delhi", "Dubai", "Hong Kong", "Istanbul", "Jerusalem",
    "Kyoto", "London", "Los Angeles", "Madrid", "Melbourne", "Miami",
    "Moscow", "Mumbai", "Paris", "Prague", "Rio de Janeiro", "Rome",
    "San Francisco", "Seoul", "Shanghai", "Singapore", "Sydney", "Tokyo",
    "Toronto", "Vienna", "Washington, D.C.", "The Godfather", "Star Wars",
    "Titanic", "The Matrix", "Casablanca", "Hamlet", "Pride and Prejudice",
    "Moby-Dick", "The Odyssey", "The Beatles", "The Rolling Stones",
    "World War II", "French Revolution", "Industrial Revolution", "Cold War",
    "Apollo 11", "Mount Everest", "Pacific Ocean", "Amazon River",
    "Sahara Desert", "Great Barrier Reef", "Python", "JavaScript", "Linux",
    "Photosynthesis", "Quantum mechanics", "General relativity",
    "The Roman Empire", "The Renaissance", "The Internet", "Solar System",
]

TRAIN_TEMPLATES = [
    "{entity}",
    "The topic is {entity}.",
    "A concise encyclopedia entry about {entity}.",
    "Important facts about {entity} include its history, context, and influence.",
    "In a classroom lesson, the teacher introduced {entity} and explained why it matters.",
    "Researchers often mention {entity} when discussing culture, geography, science, or history.",
    "The article compared {entity} with related people, places, organizations, and events.",
    "Background: {entity} is a subject that appears in reference books and general knowledge tests.",
    "A short paragraph says that {entity} has a recognizable name and associated description.",
    "Students wrote notes about {entity}, then summarized the topic in one sentence.",
    "For a quiz clue, {entity} would be identified from a brief factual description.",
    "The phrase {entity} appears in a neutral document with surrounding generic context.",
]

MAPPINGS = [(28, 10), (28, 8), (28, 6), (30, 10), (30, 8), (24, 10)]
RIDGE_ALPHAS = [1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
DIAGONAL_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
LOW_RANKS = [4, 8, 16, 32, 64]

STOP_REQUESTED = False


def _handle_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


def log(msg):
    print(msg, flush=True)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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
    def forward_once(ids):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _ = model(x)

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
        "forward_once": forward_once,
        "generate_tokens": generate_tokens,
        "name": hf_model.replace("/", "_"),
    }


@torch.inference_mode()
def capture_prompt_layers(adapter, ids, layers, device):
    wanted = sorted(set(layers))
    hiddens = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            hiddens[idx] = tensor_out[0].detach().float().cpu()
        return hook

    for idx in wanted:
        handles.append(adapter["blocks"][idx].register_forward_hook(make_hook(idx)))
    try:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _ = adapter["model"](x)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(wanted).difference(hiddens)
    if missing:
        raise RuntimeError(f"Did not capture layers: {sorted(missing)}")
    return hiddens


def make_examples():
    entities = [e for e in BASE_ENTITIES if e not in HELD_OUT_NAMES]
    rows = []
    for entity in entities:
        for template in TRAIN_TEMPLATES:
            rows.append({"entity": entity, "prompt": template.format(entity=entity)})
    return rows


def split_entities(examples, val_fraction, seed):
    rng = random.Random(seed)
    entities = sorted({row["entity"] for row in examples})
    rng.shuffle(entities)
    n_val = max(1, int(round(len(entities) * val_fraction)))
    val_entities = set(entities[:n_val])
    train_rows = [row for row in examples if row["entity"] not in val_entities]
    val_rows = [row for row in examples if row["entity"] in val_entities]
    return train_rows, val_rows, sorted(set(entities) - val_entities), sorted(val_entities)


def collect_pairs(adapter, rows, layers, target_pairs, seed, device, label, deadline):
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    data = {layer: [] for layer in layers}
    meta = []
    pair_count = 0
    for i, row in enumerate(shuffled, 1):
        if STOP_REQUESTED or time.monotonic() > deadline:
            break
        ids = adapter["encode"](row["prompt"])
        if not ids:
            continue
        captured = capture_prompt_layers(adapter, ids, layers, device)
        positions = list(range(len(ids)))
        rng.shuffle(positions)
        for pos in positions:
            token_text = adapter["decode"]([ids[pos]]).replace("\n", "\\n")
            for layer in layers:
                data[layer].append(captured[layer][pos].clone())
            meta.append({
                "split": label,
                "entity": row["entity"],
                "prompt": row["prompt"],
                "position": pos,
                "token_id": ids[pos],
                "token_text": token_text,
            })
            pair_count += 1
            if pair_count >= target_pairs:
                break
        if i % 50 == 0 or pair_count >= target_pairs:
            log(f"[pairs] {label} prompts={i}/{len(shuffled)} pairs={pair_count}/{target_pairs}")
        if pair_count >= target_pairs:
            break
    tensors = {layer: torch.stack(vals) for layer, vals in data.items() if vals}
    return tensors, meta


def centered_cosine(pred, y):
    return F.cosine_similarity(pred, y, dim=-1).mean().item()


def compute_metrics(name, mapping, adapter_obj, train_x, train_y, val_x, val_y):
    with torch.inference_mode():
        pred_train = adapter_obj.predict(train_x)
        pred_val = adapter_obj.predict(val_x)
        return {
            "candidate": name,
            "source_layer": mapping[0],
            "readout_layer": mapping[1],
            "train_mse": F.mse_loss(pred_train, train_y).item(),
            "val_mse": F.mse_loss(pred_val, val_y).item(),
            "train_cosine": centered_cosine(pred_train, train_y),
            "val_cosine": centered_cosine(pred_val, val_y),
            "n_train_pairs": train_x.shape[0],
            "n_val_pairs": val_x.shape[0],
        }


class MeanShiftNorm:
    def fit(self, x, y):
        self.x_mean = x.mean(0)
        self.y_mean = y.mean(0)
        xc = x - self.x_mean
        yc = y - self.y_mean
        self.scale = yc.norm(dim=1).mean() / (xc.norm(dim=1).mean() + 1e-8)
        return self

    def predict(self, x):
        return (x - self.x_mean) * self.scale + self.y_mean

    def state(self):
        return {"type": "mean_shift_norm", "x_mean": self.x_mean, "y_mean": self.y_mean, "scale": self.scale}


class ScalarAffine:
    def fit(self, x, y):
        self.x_mean = x.mean(0)
        self.y_mean = y.mean(0)
        xc = x - self.x_mean
        yc = y - self.y_mean
        self.a = (xc * yc).sum() / ((xc * xc).sum() + 1e-8)
        self.b = self.y_mean - self.a * self.x_mean
        return self

    def predict(self, x):
        return self.a * x + self.b

    def state(self):
        return {"type": "scalar_affine", "a": self.a, "b": self.b}


class DiagonalAffine:
    def __init__(self, alpha=1e-6):
        self.alpha = float(alpha)

    def fit(self, x, y):
        self.x_mean = x.mean(0)
        self.y_mean = y.mean(0)
        xc = x - self.x_mean
        yc = y - self.y_mean
        self.a = (xc * yc).sum(0) / ((xc * xc).sum(0) + self.alpha)
        self.b = self.y_mean - self.a * self.x_mean
        return self

    def predict(self, x):
        return self.a * x + self.b

    def state(self):
        return {"type": "diagonal_affine", "alpha": self.alpha, "a": self.a, "b": self.b}


class OrthogonalProcrustes:
    def fit(self, x, y):
        self.x_mean = x.mean(0)
        self.y_mean = y.mean(0)
        xc = x - self.x_mean
        yc = y - self.y_mean
        cross = xc.T @ yc
        u, _s, vh = torch.linalg.svd(cross, full_matrices=False)
        self.r = u @ vh
        return self

    def predict(self, x):
        return (x - self.x_mean) @ self.r + self.y_mean

    def state(self):
        return {"type": "orthogonal_procrustes", "x_mean": self.x_mean, "y_mean": self.y_mean, "r": self.r}


class RidgeLinear:
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def fit(self, x, y):
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
        xa = torch.cat([x, ones], dim=1)
        gram = xa.T @ xa
        reg = torch.eye(gram.shape[0], dtype=gram.dtype) * self.alpha
        reg[-1, -1] = 0.0
        rhs = xa.T @ y
        self.w = torch.linalg.solve(gram + reg, rhs)
        return self

    def predict(self, x):
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
        xa = torch.cat([x, ones], dim=1)
        return xa @ self.w

    def state(self):
        return {"type": "ridge_linear", "alpha": self.alpha, "w": self.w}


class LowRankResidual:
    def __init__(self, rank, epochs, lr, batch_size, seed, device, weight_decay=1e-3):
        self.rank = int(rank)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = device
        self.weight_decay = float(weight_decay)

    def fit(self, x, y, deadline):
        self.base = DiagonalAffine().fit(x, y)
        d = x.shape[1]
        torch.manual_seed(self.seed)
        self.down = nn.Linear(d, self.rank, bias=False).to(self.device)
        self.up = nn.Linear(self.rank, d, bias=False).to(self.device)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.up.weight)
        opt = torch.optim.AdamW(list(self.down.parameters()) + list(self.up.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        residual = y - self.base.predict(x)
        generator = torch.Generator().manual_seed(self.seed)
        for epoch in range(1, self.epochs + 1):
            if STOP_REQUESTED or time.monotonic() > deadline:
                break
            perm = torch.randperm(x.shape[0], generator=generator)
            for start in range(0, x.shape[0], self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = x[idx].to(self.device)
                rb = residual[idx].to(self.device)
                pred = self.up(self.down(xb))
                loss = F.mse_loss(pred, rb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if epoch == 1 or epoch % 10 == 0:
                log(f"[low_rank] rank={self.rank} epoch={epoch} loss={loss.item():.6g}")
        self.down_w = self.down.weight.detach().float().cpu()
        self.up_w = self.up.weight.detach().float().cpu()
        self.down = self.down.cpu()
        self.up = self.up.cpu()
        return self

    def predict(self, x):
        base = self.base.predict(x)
        residual = (x @ self.down_w.T) @ self.up_w.T
        return base + residual

    def state(self):
        state = self.base.state()
        state.update({"type": "low_rank_residual", "rank": self.rank, "weight_decay": self.weight_decay, "down_w": self.down_w, "up_w": self.up_w})
        return state


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


def entity_position(adapter, text):
    ids = adapter["encode"](text)
    if not ids:
        raise ValueError(f"Prompt tokenized to empty ids: {text!r}")
    return ids, len(ids) - 1


@torch.inference_mode()
def capture_layers_at_pos(adapter, ids, pos, layers, device):
    captured = capture_prompt_layers(adapter, ids, layers, device)
    return {layer: captured[layer][pos].clone() for layer in layers}


def score_original(ent_key, text):
    t = text.lower()
    criteria = CRITERIA[ent_key]
    if any(neg in t for neg in criteria["neg"]):
        return 0
    return int(any(pos in t for pos in criteria["pos"]))


def score_aliases(ent_key, text, aliases):
    t = text.lower()
    return int(any(alias in t for alias in aliases[ent_key]))


def eval_adapter(adapter, candidate_name, adapter_obj, mapping, args, device):
    source_layer, readout_layer = mapping
    target_ids = adapter["encode"](args.target)
    if len(target_ids) < 2:
        raise ValueError(f"Target prompt too short after tokenization: {args.target!r}")
    tgt_pos = len(target_ids) - 1
    captured_by_entity = {}
    translated_by_entity = {}
    for ent_key in ENTITIES:
        prompt = SOURCE_SETS["canonical"][ent_key]
        ids, src_pos = entity_position(adapter, prompt)
        captured = capture_layers_at_pos(adapter, ids, src_pos, [source_layer, readout_layer], device)
        h_l = captured[source_layer]
        h_k = captured[readout_layer]
        translated = adapter_obj.predict(h_l.unsqueeze(0)).squeeze(0)
        captured_by_entity[ent_key] = {"h_l": h_l, "h_k": h_k}
        translated_by_entity[ent_key] = translated

    rows = []
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1701)
    for i, ent_key in enumerate(ENTITIES):
        wrong_key = ENTITIES[(i + 1) % len(ENTITIES)]
        h_l = captured_by_entity[ent_key]["h_l"]
        h_k = captured_by_entity[ent_key]["h_k"]
        random_vec = norm_match(torch.randn(h_k.shape, generator=generator), h_k.norm())
        condition_vecs = {
            "oracle_readout": h_k,
            "raw_late": h_l,
            "translated_late": translated_by_entity[ent_key],
            "random_norm_control": random_vec,
            "wrong_entity_control": translated_by_entity[wrong_key],
        }
        for condition in CONDITIONS:
            text, target_norm, patch_norm = patched_generate(
                adapter, target_ids, readout_layer, tgt_pos,
                condition_vecs[condition], args.max_tokens, device,
            )
            row = {
                "candidate": candidate_name,
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
            log(
                f"[eval] {candidate_name} {ent_key} {condition}: "
                f"orig={row['score_original_substring']} strict={row['score_strict_entity_name']} "
                f"cat={row['score_category_description']} text={text!r}"
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
            item[field + "_hits"] = sum(int(r[field]) for r in group)
            item[field + "_hit_rate"] = sum(int(r[field]) for r in group) / len(group)
        out.append(item)
    return out


def condition_hits(eval_rows, candidate, condition):
    rows = [r for r in eval_rows if r["candidate"] == candidate and r["condition"] == condition]
    return {
        "original": sum(int(r["score_original_substring"]) for r in rows),
        "strict": sum(int(r["score_strict_entity_name"]) for r in rows),
        "category": sum(int(r["score_category_description"]) for r in rows),
    }


def make_leaderboard(metric_rows, eval_rows):
    by_metric = {r["candidate"]: r for r in metric_rows}
    candidates = sorted(by_metric)
    out = []
    for cand in candidates:
        translated = condition_hits(eval_rows, cand, "translated_late")
        raw = condition_hits(eval_rows, cand, "raw_late")
        random_ctl = condition_hits(eval_rows, cand, "random_norm_control")
        wrong_ctl = condition_hits(eval_rows, cand, "wrong_entity_control")
        oracle = condition_hits(eval_rows, cand, "oracle_readout")
        row = {
            "candidate": cand,
            "source_layer": by_metric[cand]["source_layer"],
            "readout_layer": by_metric[cand]["readout_layer"],
            "translated_original_hits": translated["original"],
            "translated_strict_hits": translated["strict"],
            "translated_category_hits": translated["category"],
            "raw_original_hits": raw["original"],
            "random_original_hits": random_ctl["original"],
            "wrong_original_hits": wrong_ctl["original"],
            "oracle_original_hits": oracle["original"],
            "val_cosine": by_metric[cand]["val_cosine"],
            "val_mse": by_metric[cand]["val_mse"],
            "promising": int(
                translated["original"] >= 2
                and translated["original"] > raw["original"]
                and random_ctl["original"] == 0
                and wrong_ctl["original"] < translated["original"]
            ),
            "good": int(
                (translated["original"] >= 4
                 or (translated["strict"] + translated["category"]) >= max(1, oracle["strict"] + oracle["category"] - 1))
                and random_ctl["original"] == 0
                and wrong_ctl["original"] == 0
                and translated["original"] >= raw["original"] + 2
            ),
            "strong": int(
                (translated["original"] >= 4
                 or (translated["strict"] + translated["category"]) >= max(1, oracle["strict"] + oracle["category"] - 1))
                and random_ctl["original"] == 0
                and wrong_ctl["original"] == 0
                and translated["original"] >= raw["original"] + 2
            ),
        }
        out.append(row)
    out.sort(key=lambda r: (
        r["translated_original_hits"],
        r["translated_strict_hits"],
        r["translated_category_hits"],
        r["val_cosine"],
    ), reverse=True)
    return out


def write_readme(path, args, leaderboard, stopped_reason):
    best = leaderboard[0] if leaderboard else None
    if best:
        best_text = (
            f"Best candidate: `{best['candidate']}` with "
            f"{best['translated_original_hits']}/5 translated original hits, "
            f"{best['translated_strict_hits']}/5 strict hits, "
            f"{best['translated_category_hits']}/5 category hits."
        )
        beat_raw = best["translated_original_hits"] > best["raw_original_hits"]
        controls = best["random_original_hits"] == 0 and best["wrong_original_hits"] < best["translated_original_hits"]
        rec = (
            "Rerun the best constrained adapter on a fresh split and then test neighboring readout layers."
            if best["promising"]
            else "Increase pair diversity with longer factual contexts or try a learned low-rank adapter with entity-position-focused pairs."
        )
    else:
        best_text = "Best candidate: none completed."
        beat_raw = False
        controls = False
        rec = "Rerun with a smaller initial pair target to produce at least one completed candidate."
    text = f"""# Patchscope Adapter Search

Created: {datetime.now(timezone.utc).isoformat()}

Model: `{args.hf_model}`

Target prompt: `{args.target}`

{best_text}

Beat raw_late: `{beat_raw}`

Controls stayed clean: `{controls}`

Stopped reason: `{stopped_reason}`

If no adapter worked, the likely failure is that reconstruction quality alone
did not put late-layer entity information into the readout layer's decodable
format under the regular target prompt.

Recommended next experiment: {rec}
"""
    with open(path, "w") as f:
        f.write(text)


def save_artifacts(args, metric_rows, eval_rows, leaderboard, best_state, stopped_reason):
    metric_fields = [
        "candidate", "source_layer", "readout_layer", "train_mse", "val_mse",
        "train_cosine", "val_cosine", "n_train_pairs", "n_val_pairs",
    ]
    eval_fields = [
        "candidate", "entity_key", "entity", "condition", "source_layer",
        "readout_layer", "target_prompt", "source_prompt", "wrong_entity_key",
        "wrong_entity", "max_tokens", "generated_text", "score_original_substring",
        "score_strict_entity_name", "score_category_description",
        "normal_target_norm", "patched_vector_norm", "source_vec_norm",
    ]
    summary_fields = [
        "candidate", "condition", "n",
        "score_original_substring_hits", "score_original_substring_hit_rate",
        "score_strict_entity_name_hits", "score_strict_entity_name_hit_rate",
        "score_category_description_hits", "score_category_description_hit_rate",
    ]
    by_entity_fields = [
        "candidate", "entity_key", "entity", "condition", "n",
        "score_original_substring_hits", "score_original_substring_hit_rate",
        "score_strict_entity_name_hits", "score_strict_entity_name_hit_rate",
        "score_category_description_hits", "score_category_description_hit_rate",
    ]
    leader_fields = [
        "candidate", "source_layer", "readout_layer", "translated_original_hits",
        "translated_strict_hits", "translated_category_hits", "raw_original_hits",
        "random_original_hits", "wrong_original_hits", "oracle_original_hits",
        "val_cosine", "val_mse", "promising", "good", "strong",
    ]
    write_csv(os.path.join(args.out_dir, "train_val_metrics_by_candidate.csv"), metric_rows, metric_fields)
    write_csv(os.path.join(args.out_dir, "eval_generations.csv"), eval_rows, eval_fields)
    write_csv(os.path.join(args.out_dir, "eval_summary.csv"), summarize(eval_rows, ["candidate", "condition"]), summary_fields)
    write_csv(
        os.path.join(args.out_dir, "eval_summary_by_entity.csv"),
        summarize(eval_rows, ["candidate", "entity_key", "entity", "condition"]),
        by_entity_fields,
    )
    write_csv(os.path.join(args.out_dir, "adapter_leaderboard.csv"), leaderboard, leader_fields)
    write_csv(os.path.join(args.out_dir, "all_candidates.csv"), leaderboard, leader_fields)
    if leaderboard:
        write_json(os.path.join(args.out_dir, "best_so_far.json"), leaderboard[0])
    if best_state is not None:
        torch.save(best_state, os.path.join(args.out_dir, "best_adapter.pt"))
    write_readme(os.path.join(args.out_dir, "README.md"), args, leaderboard, stopped_reason)


def candidate_plan(include_heavy=True):
    items = [
        ("mean_shift_norm", lambda args, device: MeanShiftNorm()),
        ("scalar_affine", lambda args, device: ScalarAffine()),
        ("diagonal_affine", lambda args, device: DiagonalAffine()),
    ]
    for alpha in DIAGONAL_ALPHAS:
        items.append((f"diagonal_affine_alpha_{alpha:g}", lambda args, device, a=alpha: DiagonalAffine(a)))
    items.append(("orthogonal_procrustes", lambda args, device: OrthogonalProcrustes()))
    if include_heavy:
        for alpha in RIDGE_ALPHAS:
            items.append((f"ridge_linear_alpha_{alpha:g}", lambda args, device, a=alpha: RidgeLinear(a)))
        for rank in LOW_RANKS:
            items.append((f"low_rank_residual_rank_{rank}", lambda args, device, r=rank: LowRankResidual(
                r, args.low_rank_epochs, args.low_rank_lr, args.batch_size, args.seed, device, args.low_rank_weight_decay
            )))
    return items


def fit_candidate(name, maker, args, device, train_x, train_y, deadline):
    log(f"[candidate] fitting {name}")
    obj = maker(args, device)
    if isinstance(obj, LowRankResidual):
        obj.fit(train_x, train_y, deadline)
    else:
        obj.fit(train_x, train_y)
    return obj


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--initial-pairs", type=int, default=12000)
    ap.add_argument("--expanded-pairs", type=int, default=24000)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--time-limit-seconds", type=int, default=7200)
    ap.add_argument("--candidate-soft-limit-seconds", type=int, default=900)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--low-rank-epochs", type=int, default=80)
    ap.add_argument("--low-rank-lr", type=float, default=1e-3)
    ap.add_argument("--low-rank-weight-decay", type=float, default=1e-3)
    ap.add_argument("--smoke-test", action="store_true")
    return ap.parse_args()


def run_smoke(args):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    x = torch.randn(128, 32)
    y = 1.5 * x + 0.1 * torch.randn_like(x)
    train_x, val_x = x[:96], x[96:]
    train_y, val_y = y[:96], y[96:]
    metric_rows, eval_rows = [], []
    obj = ScalarAffine().fit(train_x, train_y)
    metric_rows.append(compute_metrics("scalar_affine_h28_to_h10", (28, 10), obj, train_x, train_y, val_x, val_y))
    leaderboard = [{
        "candidate": "scalar_affine_h28_to_h10",
        "source_layer": 28,
        "readout_layer": 10,
        "translated_original_hits": 0,
        "translated_strict_hits": 0,
        "translated_category_hits": 0,
        "raw_original_hits": 0,
        "random_original_hits": 0,
        "wrong_original_hits": 0,
        "oracle_original_hits": 0,
        "val_cosine": metric_rows[0]["val_cosine"],
        "val_mse": metric_rows[0]["val_mse"],
        "promising": 0,
        "strong": 0,
    }]
    save_artifacts(args, metric_rows, eval_rows, leaderboard, obj.state(), "smoke_test")
    write_json(os.path.join(args.out_dir, "adapter_search_config.json"), vars(args))
    log(f"[smoke] wrote {args.out_dir}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    existing = [name for name in os.listdir(args.out_dir) if name != "run.log"]
    if existing:
        raise FileExistsError(f"Refusing to overwrite non-empty out_dir: {args.out_dir}")
    command = " ".join(sys.argv)
    with open(os.path.join(args.out_dir, "command.txt"), "w") as f:
        f.write(command + "\n")
    if args.smoke_test:
        run_smoke(args)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.monotonic()
    deadline = start + args.time_limit_seconds
    device = torch.device(args.device)
    config = vars(args).copy()
    config.update({
        "mappings": MAPPINGS,
        "ridge_alphas": RIDGE_ALPHAS,
        "low_ranks": LOW_RANKS,
        "diagonal_alphas": DIAGONAL_ALPHAS,
        "held_out_entities": sorted(HELD_OUT_NAMES),
        "conditions": CONDITIONS,
        "train_templates": TRAIN_TEMPLATES,
    })
    write_json(os.path.join(args.out_dir, "adapter_search_config.json"), config)
    adapter = load_hf(args.hf_model, device)
    all_layers = sorted({layer for mapping in MAPPINGS for layer in mapping})
    for layer in all_layers:
        if not (0 <= layer < adapter["n_layer"]):
            raise ValueError(f"Layer {layer} out of range 0..{adapter['n_layer'] - 1}")

    examples = make_examples()
    train_rows, val_rows, train_entities, val_entities = split_entities(examples, args.val_fraction, args.seed)
    with open(os.path.join(args.out_dir, "train_entities.txt"), "w") as f:
        f.write("\n".join(train_entities) + "\n")
    with open(os.path.join(args.out_dir, "val_entities.txt"), "w") as f:
        f.write("\n".join(val_entities) + "\n")
    log(
        f"[setup] model={args.hf_model} n_layer={adapter['n_layer']} "
        f"train_entities={len(train_entities)} val_entities={len(val_entities)}"
    )

    train_tensors, train_meta = collect_pairs(
        adapter, train_rows, all_layers, args.initial_pairs, args.seed, device, "train", deadline
    )
    val_tensors, val_meta = collect_pairs(
        adapter, val_rows, all_layers, max(1000, args.initial_pairs // 5), args.seed + 1, device, "val", deadline
    )
    write_csv(
        os.path.join(args.out_dir, "pair_metadata_sample.csv"),
        (train_meta + val_meta)[:1000],
        ["split", "entity", "prompt", "position", "token_id", "token_text"],
    )

    metric_rows = []
    eval_rows = []
    leaderboard = []
    best_state = None
    stopped_reason = "time_limit"
    expanded = False

    try:
        for mapping_index, mapping in enumerate(MAPPINGS):
            if STOP_REQUESTED or time.monotonic() > deadline:
                break
            include_heavy = True
            source_layer, readout_layer = mapping
            train_x, train_y = train_tensors[source_layer], train_tensors[readout_layer]
            val_x, val_y = val_tensors[source_layer], val_tensors[readout_layer]
            log(f"[mapping] h{source_layer}->h{readout_layer} train_pairs={train_x.shape[0]} val_pairs={val_x.shape[0]}")
            for base_name, maker in candidate_plan(include_heavy=include_heavy):
                if STOP_REQUESTED or time.monotonic() > deadline:
                    break
                candidate_deadline = min(deadline, time.monotonic() + args.candidate_soft_limit_seconds)
                cand_name = f"{base_name}_h{source_layer}_to_h{readout_layer}"
                obj = fit_candidate(cand_name, maker, args, device, train_x, train_y, candidate_deadline)
                metrics = compute_metrics(cand_name, mapping, obj, train_x, train_y, val_x, val_y)
                metric_rows.append(metrics)
                eval_rows.extend(eval_adapter(adapter, cand_name, obj, mapping, args, device))
                leaderboard = make_leaderboard(metric_rows, eval_rows)
                best_name = leaderboard[0]["candidate"] if leaderboard else cand_name
                if best_name == cand_name:
                    best_state = {"candidate": cand_name, "mapping": mapping, "state": obj.state(), "metrics": metrics}
                save_artifacts(args, metric_rows, eval_rows, leaderboard, best_state, stopped_reason)
                log(f"[leader] best={leaderboard[0] if leaderboard else 'none'}")

                best = leaderboard[0]
                if best["strong"]:
                    log("[adaptive] strong adapter found; confirming on alternate split seed")
                    confirm_args = argparse.Namespace(**vars(args))
                    confirm_args.seed = args.seed + 99
                    confirm_name = cand_name + "_confirm_seed"
                    confirm_train_rows, confirm_val_rows, _, _ = split_entities(examples, args.val_fraction, confirm_args.seed)
                    confirm_train, _ = collect_pairs(adapter, confirm_train_rows, all_layers, min(args.initial_pairs, 12000), confirm_args.seed, device, "confirm_train", deadline)
                    confirm_val, _ = collect_pairs(adapter, confirm_val_rows, all_layers, max(1000, args.initial_pairs // 5), confirm_args.seed + 1, device, "confirm_val", deadline)
                    cobj = fit_candidate(confirm_name, maker, confirm_args, device, confirm_train[source_layer], confirm_train[readout_layer], candidate_deadline)
                    cmetrics = compute_metrics(confirm_name, mapping, cobj, confirm_train[source_layer], confirm_train[readout_layer], confirm_val[source_layer], confirm_val[readout_layer])
                    metric_rows.append(cmetrics)
                    eval_rows.extend(eval_adapter(adapter, confirm_name, cobj, mapping, args, device))
                    leaderboard = make_leaderboard(metric_rows, eval_rows)
                    if leaderboard[0]["candidate"] == confirm_name:
                        best_state = {"candidate": confirm_name, "mapping": mapping, "state": cobj.state(), "metrics": cmetrics}
                    stopped_reason = "strong_adapter_confirmed"
                    save_artifacts(args, metric_rows, eval_rows, leaderboard, best_state, stopped_reason)
                    return

                simple_done = len(metric_rows) >= 8 and mapping_index == 0
                no_promising = leaderboard and not any(row["promising"] for row in leaderboard)
                if simple_done and no_promising and not expanded and args.expanded_pairs > args.initial_pairs:
                    log("[adaptive] no simple adapter worked; expanding training-pair count before heavier candidates")
                    extra_train, extra_train_meta = collect_pairs(
                        adapter, train_rows, all_layers, args.expanded_pairs, args.seed + 10, device, "train_expanded", deadline
                    )
                    extra_val, extra_val_meta = collect_pairs(
                        adapter, val_rows, all_layers, max(2000, args.expanded_pairs // 5), args.seed + 11, device, "val_expanded", deadline
                    )
                    train_tensors, val_tensors = extra_train, extra_val
                    train_x, train_y = train_tensors[source_layer], train_tensors[readout_layer]
                    val_x, val_y = val_tensors[source_layer], val_tensors[readout_layer]
                    write_csv(
                        os.path.join(args.out_dir, "pair_metadata_sample_expanded.csv"),
                        (extra_train_meta + extra_val_meta)[:1000],
                        ["split", "entity", "prompt", "position", "token_id", "token_text"],
                    )
                    expanded = True
        stopped_reason = "completed_candidate_plan" if not STOP_REQUESTED and time.monotonic() <= deadline else "time_limit"
    finally:
        leaderboard = make_leaderboard(metric_rows, eval_rows) if metric_rows else []
        save_artifacts(args, metric_rows, eval_rows, leaderboard, best_state, stopped_reason)
        elapsed = time.monotonic() - start
        log(f"[done] stopped_reason={stopped_reason} elapsed_seconds={elapsed:.1f} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
