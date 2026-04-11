"""
Standalone rollout worker for strict-synchronous RL training.

This process owns a single vLLM engine on a dedicated GPU. The trainer talks to
it over localhost HTTP:

  - GET  /health
  - POST /generate   {prompts, num_samples, max_new_tokens, temperature, top_k}
  - POST /reload     {model_path}

The trainer keeps semantics strict by:
  1. generating step-t rollouts from weights W_t
  2. updating the policy to W_{t+1}
  3. checkpointing W_{t+1}
  4. asking this worker to reload the checkpoint
  5. only then starting step t+1
"""

import argparse
import gc
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer

from nanochat.rl_rollout import generate_rollouts, vllm_reload_model


class RolloutState:
    def __init__(self, model_path, tokenizer_path, dtype, gpu_memory_utilization):
        self.tokenizer_path = tokenizer_path
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_path = model_path
        self.engine = vllm_reload_model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def reload(self, model_path):
        old_engine = self.engine
        self.engine = None
        del old_engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1.0)
        self.model_path = model_path
        self.engine = vllm_reload_model(
            model_path=model_path,
            tokenizer_path=self.tokenizer_path,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "nanochat-rollout-worker/0.1"

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(body.decode("utf-8"))

    def _write_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # pragma: no cover - keep worker stdout clean
        sys.stdout.write("%s - - [%s] %s\n" % (
            self.address_string(),
            self.log_date_time_string(),
            fmt % args,
        ))

    def do_GET(self):
        if self.path != "/health":
            self._write_json({"ok": False, "error": "not found"}, status=404)
            return
        state = self.server.state
        self._write_json({
            "ok": True,
            "model_path": state.model_path,
        })

    def do_POST(self):
        if self.path == "/generate":
            payload = self._read_json()
            state = self.server.state
            rollouts = generate_rollouts(
                state.engine,
                state.tokenizer,
                payload["prompts"],
                payload["num_samples"],
                payload["max_new_tokens"],
                payload["temperature"],
                payload["top_k"],
            )
            self._write_json({"ok": True, "rollouts": rollouts})
            return

        if self.path == "/reload":
            payload = self._read_json()
            state = self.server.state
            state.reload(payload["model_path"])
            self._write_json({
                "ok": True,
                "model_path": state.model_path,
            })
            return

        self._write_json({"ok": False, "error": "not found"}, status=404)


def main():
    parser = argparse.ArgumentParser(description="nanochat rollout worker")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8047)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model
    state = RolloutState(
        model_path=args.model,
        tokenizer_path=tokenizer_path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state
    print(f"rollout worker listening on http://{args.host}:{args.port} model={args.model}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
