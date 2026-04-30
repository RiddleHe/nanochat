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
import threading
import time
import logging
from typing import Literal
from transformers import AutoTokenizer
from vllm import LLM
from vllm.config import WeightTransferConfig
from nanorl.rollout import generate_rollouts, vllm_reload_weights_inplace

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nanorl.rollout_worker")

class RolloutState:
    def __init__(self, model_path, tokenizer_path, dtype, gpu_memory_utilization, tensor_parallel_size,weight_transfer_backend):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_path = model_path
        self._pause_lock = threading.Lock()
        self._update_thread = None
        self._update_error = None
        self._update_in_flight = False
        self._is_generation_paused = False
        self._weight_transfer_backend = weight_transfer_backend
        llm_kwargs = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "disable_log_stats": True,
        }
        if weight_transfer_backend:
            llm_kwargs["load_format"] = "dummy"
            # it can either be IPC (colocate trainer and inference) or nccl (different gpus)
            llm_kwargs["weight_transfer_config"] = WeightTransferConfig(backend=weight_transfer_backend)
        self.engine = LLM(
            **llm_kwargs,
        )

    def reload(self, model_path):
        vllm_reload_weights_inplace(self.engine, model_path)
        self.model_path = model_path

    def wait_for_generation_slot(self):
        while True:
            with self._pause_lock:
                if not self._is_generation_paused:
                    return
            time.sleep(0.01)

    def _run_update(self, payload):
        try:
            logger.info(f"Applying in-place vLLM update with {len(payload['names'])=}, {payload['packed']=}.")
            self.engine.update_weights({"update_info": payload})
        except Exception as exc:  # pragma: no cover - runtime errors are environment-specific.
            self._update_error = exc
        finally:
            self._update_in_flight = False
    
    def start_update_weights(self, payload):
        with self._pause_lock:
            if self._update_in_flight:
                raise RuntimeError("weight update already in progress")
            self._is_generation_paused = True
            self._update_in_flight = True
            self._update_error = None
            self._update_thread = threading.Thread(
                target=self._run_update,
                args=(payload,),
                daemon=True,
            )
            self._update_thread.start()
    
    def init_weight_transfer(self, payload):
        logger.info(
            "Initializing worker weight transfer with "
            f"{payload['master_address']=}, {payload['master_port']=}, "
            f"{payload['rank_offset']=}, {payload['world_size']=}."
        )
        # Run in a background thread so the HTTP response returns immediately.
        # The trainer must call trainer_init() concurrently to complete the
        # NCCL rendezvous; blocking here would deadlock.
        threading.Thread(
            target=self.engine.init_weight_transfer_engine,
            args=({"init_info": payload},),
            daemon=True,
        ).start()

    def finish_update_weights(self):
        if self._update_thread is None:
            raise RuntimeError("no weight update has been started")
        self._update_thread.join()
        self._update_thread = None
        if self._update_error is not None:
            err = self._update_error
            self._update_error = None
            self._is_generation_paused = False
            raise RuntimeError(f"in-place weight update failed: {err}") from err
        # Clear out the KV cache in vLLM
        self.engine.reset_prefix_cache()
        with self._pause_lock:
            self._is_generation_paused = False
        logger.info("In-place vLLM update completed and prefix cache reset.")

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
            state.wait_for_generation_slot()
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

        if self.path == "/update_weights_start":
            payload = self._read_json()
            state = self.server.state
            state.start_update_weights(payload)
            self._write_json({"ok": True, "status": "started"})
            return

        if self.path == "/init_weight_transfer":
            payload = self._read_json()
            state = self.server.state
            state.init_weight_transfer(payload)
            self._write_json({"ok": True, "status": "initialized"})
            return

        if self.path == "/update_weights_finish":
            state = self.server.state
            state.finish_update_weights()
            self._write_json({"ok": True, "status": "completed"})
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
    parser.add_argument("--weight-transfer-backend", type=str, default="nccl",
                        choices=["nccl", "ipc"], help="Backend for inplace weight transfer (nccl or ipc)")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model
    state = RolloutState(
        model_path=args.model,
        tokenizer_path=tokenizer_path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        weight_transfer_backend=args.weight_transfer_backend,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state
    print(f"rollout worker listening on http://{args.host}:{args.port} model={args.model}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
