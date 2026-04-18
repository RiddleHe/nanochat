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
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer
from vllm import LLM

from nanorl.rollout import generate_rollouts, vllm_reload_weights_inplace


class RolloutState:
    def __init__(self, model_path, tokenizer_path, dtype, gpu_memory_utilization, tensor_parallel_size):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_path = model_path
        self.engine = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            disable_log_stats=True,
        )

    def reload(self, model_path):
        vllm_reload_weights_inplace(self.engine, model_path)
        self.model_path = model_path


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
            self._write_json({"ok": True, "model_path": state.model_path})
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model
    state = RolloutState(
        model_path=args.model,
        tokenizer_path=tokenizer_path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state
    print(f"rollout worker listening on http://{args.host}:{args.port} model={args.model}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
