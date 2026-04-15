"""
Prepare the rStar-Coder dataset into the canonical JSONL format consumed
by the nanorl pipeline (nanorl/data.py).

Reads from HF: RiddleHe/rStar-Coder (preprocessed: test cases capped,
schema simplified, JSON blobs pre-parsed into proper columns).

Writes to: <base_dir>/data/rl/rstar_seed_train.jsonl

Usage:
    python -m nanorl.scripts.prepare_rstar --tokenizer Qwen/Qwen3-0.6B
    python -m nanorl.scripts.prepare_rstar --tokenizer Qwen/Qwen3-0.6B --limit 10
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections.abc import Iterable

from transformers import AutoTokenizer
from datasets import load_dataset

from nanochat.common import get_base_dir


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def load_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1]
                    os.environ["HF_TOKEN"] = token
                    return token
    return None


def build_prompt(question: str, starter_code: str, tokenizer) -> str:
    user_content = question.strip()
    if starter_code:
        user_content += (
            "\n\nUse this starter code:\n```python\n"
            + starter_code.strip()
            + "\n```"
        )
    user_content += "\n\nProvide your complete solution as a single Python code block."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_filtered_jsonl(path: str, limit: int | None = None, skip_rows: int = 0) -> Iterable[dict]:
    """Yield rows from a locally filtered rStar JSONL.

    Expected schema matches the output of `scripts/filter_rstar.py`:
      question_id, question, kind, starter_code, func_name, inputs, outputs, n_tests
    """
    yielded = 0
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            if i <= skip_rows:
                continue
            if limit is not None and yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yielded += 1
            yield json.loads(line)


def _ast_to_python(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_to_python(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return [_ast_to_python(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {_ast_to_python(k): _ast_to_python(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_to_python(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return +_ast_to_python(node.operand)
    if isinstance(node, ast.NameConstant):  # py<3.8 compatibility shape
        return node.value
    raise ValueError(f"unsupported AST literal: {ast.dump(node, include_attributes=False)}")


def _call_args_from_ast(call: ast.Call) -> list:
    if call.keywords:
        raise ValueError("keyword args in assert-style tests are unsupported")
    return [_ast_to_python(arg) for arg in call.args]


def _extract_param_names(starter_code: str, fn_name: str) -> list[str]:
    def _normalize_args(args_src: str) -> list[str]:
        names = []
        for raw in args_src.split(","):
            raw = raw.strip()
            if not raw:
                continue
            raw = raw.split("=", 1)[0].strip()
            raw = raw.split(":", 1)[0].strip()
            raw = raw.lstrip("*").strip()
            if raw and raw not in {"self", "cls"}:
                names.append(raw)
        return names

    if not starter_code.strip():
        return []
    try:
        module = ast.parse(starter_code)
    except SyntaxError:
        module = None

    if module is not None:
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                return [arg.arg for arg in node.args.args]
            if isinstance(node, ast.ClassDef) and node.name == "Solution":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == fn_name:
                        args = [arg.arg for arg in child.args.args]
                        if args and args[0] == "self":
                            args = args[1:]
                        return args

    header_pat = re.compile(rf"def\s+{re.escape(fn_name)}\s*\((.*?)\)\s*:")
    m = header_pat.search(starter_code)
    if m:
        return _normalize_args(m.group(1))
    return []


def _normalize_call_based_case(input_s: str, output_s: str, param_names: list[str]) -> tuple[list, object]:
    s = input_s.strip()
    if s.startswith("assert "):
        module = ast.parse(s)
        if len(module.body) != 1 or not isinstance(module.body[0], ast.Assert):
            raise ValueError("unsupported assert-style test shape")
        test = module.body[0].test

        if isinstance(test, ast.Compare):
            if len(test.ops) != 1 or len(test.comparators) != 1 or not isinstance(test.ops[0], ast.Eq):
                raise ValueError("only == compares are supported in assert-style tests")
            if not isinstance(test.left, ast.Call):
                raise ValueError("assert compare does not call the target function")
            args = _call_args_from_ast(test.left)
            expected = _ast_to_python(test.comparators[0])
            return args, expected

        if isinstance(test, ast.Call):
            return _call_args_from_ast(test), True

        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Call):
            return _call_args_from_ast(test.operand), False

        raise ValueError("unsupported assert-style test expression")

    if len(param_names) == 1:
        prefix = f'{{"{param_names[0]}": '
        if input_s.startswith(prefix) and input_s.endswith("}"):
            raw_value = input_s[len(prefix):-1].strip()
            if raw_value.startswith('"') and raw_value.endswith('"'):
                body = raw_value[1:-1]
                if "\\" not in body:
                    parsed_arg = body
                else:
                    parsed_arg = json.loads(raw_value)
            else:
                parsed_arg = json.loads(raw_value)

            if output_s and (output_s.isdigit() or (output_s[0] == "-" and output_s[1:].isdigit())):
                return [parsed_arg], output_s
            if output_s.startswith('"') and output_s.endswith('"'):
                out_body = output_s[1:-1]
                if "\\" not in out_body:
                    return [parsed_arg], out_body
            try:
                expected = json.loads(output_s)
            except json.JSONDecodeError:
                expected = output_s
            return [parsed_arg], expected

    try:
        parsed_input = json.loads(input_s)
    except json.JSONDecodeError:
        try:
            module = ast.parse(s)
        except SyntaxError:
            args = [s]
        else:
            if module.body and all(isinstance(stmt, ast.Assign) for stmt in module.body):
                values_by_name = {}
                source_order = []
                for stmt in module.body:
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                        args = [s]
                        break
                    name = stmt.targets[0].id
                    values_by_name[name] = _ast_to_python(stmt.value)
                    source_order.append(name)
                else:
                    if param_names and all(name in values_by_name for name in param_names):
                        args = [values_by_name[name] for name in param_names]
                    else:
                        args = [values_by_name[name] for name in source_order]
            else:
                args = [s]
    else:
        if isinstance(parsed_input, list):
            if len(param_names) == 1:
                args = [parsed_input]
            else:
                args = parsed_input
        elif isinstance(parsed_input, dict):
            if param_names and all(name in parsed_input for name in param_names):
                args = [parsed_input[name] for name in param_names]
            else:
                # Preserve source key order as a fallback.
                args = list(parsed_input.values())
        else:
            args = [parsed_input]

    try:
        expected = json.loads(output_s)
    except json.JSONDecodeError:
        expected = output_s
    return args, expected


def main():
    parser = argparse.ArgumentParser(description="Prepare rStar-Coder for RL")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HF tokenizer name (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N rows (for smoke testing)")
    parser.add_argument("--time-limit", type=float, default=4.0,
                        help="Per-test time limit in seconds")
    parser.add_argument("--memory-limit-mb", type=int, default=256,
                        help="Per-test memory limit in MB")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path")
    parser.add_argument("--dataset", type=str, default="RiddleHe/rStar-Coder",
                        help="HF dataset repo (default: RiddleHe/rStar-Coder)")
    parser.add_argument("--input-jsonl", type=str, default=None,
                        help="Use a local filtered rStar JSONL instead of loading from HF")
    parser.add_argument("--unsupported-output", type=str, default=None,
                        help="Optional JSONL path for call-based rows that required fallback parsing")
    parser.add_argument("--skip-rows", type=int, default=0,
                        help="Skip the first N locally filtered rows before converting")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Print progress every N emitted rows")
    parser.add_argument("--progress-path", type=str, default=None,
                        help="Optional JSON path updated on every row with current progress")
    args = parser.parse_args()

    hf_token = load_hf_token()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=hf_token)

    if args.input_jsonl:
        print(f"Loading local filtered JSONL: {args.input_jsonl}")
        rows = iter_filtered_jsonl(args.input_jsonl, limit=args.limit, skip_rows=args.skip_rows)
        loaded_msg = f"Loaded up to {args.limit} rows after skipping {args.skip_rows}" if args.limit else f"Streaming all rows after skipping {args.skip_rows}"
        print(loaded_msg)
    else:
        # Load preprocessed dataset (small enough to load directly)
        split = "train"
        if args.limit:
            split = f"train[:{args.limit}]"
        print(f"Loading {args.dataset} split={split} ...")
        rows = load_dataset(args.dataset, split=split, token=hf_token)
        print(f"Loaded {len(rows)} rows")

    # Output path
    if args.output:
        out_path = args.output
    else:
        base = get_base_dir()
        out_dir = os.path.join(base, "data", "rl")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "rstar_seed_train.jsonl")

    n_call_based = 0
    n_stdin_stdout = 0
    n_emitted = 0
    n_call_based_fallback = 0

    unsupported_f = open(args.unsupported_output, "w") if args.unsupported_output else None

    def write_progress(stage: str, row_idx: int, row: dict, kind: str | None = None) -> None:
        if not args.progress_path:
            return
        progress = {
            "stage": stage,
            "row_idx": row_idx,
            "question_id": row.get("question_id", ""),
            "kind": kind or row.get("kind", ""),
            "emitted": n_emitted,
        }
        with open(args.progress_path, "w") as pf:
            json.dump(progress, pf)
            pf.write("\n")

    with open(out_path, "w") as f:
        for row_idx, row in enumerate(rows, start=1):
            kind = row["kind"]
            starter_code = row["starter_code"] or ""
            inputs = row["inputs"]    # already a list[str]
            outputs = row["outputs"]  # already a list[str]

            write_progress("start", row_idx, row, kind)

            # For call-based, parse each element from JSON string to native
            if kind == "code_call_based":
                param_names = _extract_param_names(starter_code, row.get("func_name", ""))
                normalized = []
                row_used_fallback = False
                for inp, out in zip(inputs, outputs):
                    try:
                        normalized.append(_normalize_call_based_case(inp, out, param_names))
                    except Exception as exc:
                        row_used_fallback = True
                        try:
                            expected = json.loads(out)
                        except Exception:
                            expected = out
                        normalized.append(([inp.strip()], expected))
                        if unsupported_f is not None:
                            unsupported_f.write(json.dumps({
                                "question_id": row["question_id"],
                                "func_name": row.get("func_name", ""),
                                "class_name": row.get("class_name", ""),
                                "starter_code": starter_code,
                                "input": inp,
                                "output": out,
                                "error": f"{type(exc).__name__}: {exc}",
                            }) + "\n")
                inputs_parsed = [args for args, _ in normalized]
                outputs_parsed = [expected for _, expected in normalized]
                n_call_based += 1
                if row_used_fallback:
                    n_call_based_fallback += 1
            else:
                inputs_parsed = inputs
                outputs_parsed = outputs
                n_stdin_stdout += 1
            write_progress("normalized", row_idx, row, kind)

            prompt_str = build_prompt(row["question"], starter_code, tokenizer)
            write_progress("prompted", row_idx, row, kind)

            payload = {
                "inputs": inputs_parsed,
                "outputs": outputs_parsed,
                "time_limit_s": args.time_limit,
                "memory_limit_mb": args.memory_limit_mb,
            }
            if kind == "code_call_based":
                payload["fn_name"] = row["func_name"]
                if starter_code:
                    payload["starter_code"] = starter_code

            jsonl_row = {
                "id": f"rstar/{row['question_id']}",
                "prompt": prompt_str,
                "kind": kind,
                "payload": payload,
                "meta": {
                    "source": "rstar_seed",
                    "n_tests": row["n_tests"],
                },
            }
            line = json.dumps(jsonl_row)
            write_progress("serialized", row_idx, row, kind)
            f.write(line + "\n")
            n_emitted += 1
            write_progress("done", row_idx, row, kind)
            if n_emitted % args.progress_every == 0:
                print(
                    f"  emitted {n_emitted} rows (row_idx={row_idx}, qid={row['question_id']}, kind={kind})...",
                    flush=True,
                )

    if unsupported_f is not None:
        unsupported_f.close()

    print(f"\n--- Summary ---")
    print(f"emitted {n_emitted} rows to {out_path}")
    print(f"  call_based:   {n_call_based}")
    print(f"  stdin_stdout: {n_stdin_stdout}")
    print(f"  call_fallback:{n_call_based_fallback}")
    if args.unsupported_output:
        print(f"  unsupported:  {args.unsupported_output}")


if __name__ == "__main__":
    main()
