"""
Sandboxed execution of model-generated Python code against test cases.

Each call to `run_test` spawns a fresh short-lived subprocess with:
  - resource.setrlimit(RLIMIT_AS) for memory cap
  - subprocess.communicate(timeout=...) for wall-clock cap
  - os.setsid + killpg on timeout so threads/children can't escape
  - returns a TestResult with passed: bool + detail: str + duration_s

`detail` is a free-form string carrying whatever the failure looked like —
the failing test, the exception message, "timeout (wall clock)", a stderr
tail, etc. We deliberately don't enumerate failure categories: training
only consumes pass/fail, and any finer breakdown can be recovered from
detail strings when investigating. If a failure category ever becomes
training signal (e.g. shaping the reward to penalize timeouts harder),
add it at that point.

Two modes share the same transport:
  - code_call_based:  model defines a function; we call it via a JSON driver
  - code_stdin_stdout: model is a complete program; we pipe stdin, read stdout

Safety scope: this is *bounded* execution, not full isolation. It catches
infinite loops, runaway recursion, memory blowups, and the usual mistakes
in LLM-generated code. It is NOT a defense against deliberately malicious
code (which could still touch the filesystem or network). Cwd is a fresh
tempdir per call.

Threading note: preexec_fn is not safe from a multi-threaded parent. Call
this from process-based pools (multiprocessing) or single-threaded code,
not from a thread pool sharing state.
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import shutil
import resource
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class TestResult:
    passed: bool
    detail: str = ""      # empty on success; reason / stderr tail on failure
    duration_s: float = 0.0


# ----------------------------------------------------------------------------
# Driver script for call-based mode.
#
# Reads {"code": str, "fn_name": str, "args": list} on stdin, exec's the code
# with user prints suppressed (so they can't pollute our result line), calls
# fn_name(*args), and writes exactly one JSON line on stdout: {"ok": <value>}
# on success or {"err": <message>} on any failure.
# ----------------------------------------------------------------------------
_CALL_BASED_DRIVER = r'''
import sys, json, io, contextlib
try:
    spec = json.loads(sys.stdin.read())
    ns = {"__name__": "__user__"}
    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        exec(spec["code"], ns)
        fn_name = spec["fn_name"]
        fn = ns.get(fn_name)
        if fn is None:
            # LeetCode-style wrapper: class Solution: def fn_name(self, ...)
            cls = ns.get("Solution")
            if cls is not None:
                fn = getattr(cls(), fn_name, None)
        if fn is None:
            raise NameError(f"function {fn_name!r} not defined (also tried Solution().{fn_name})")
        result = fn(*spec["args"])
    print(json.dumps({"ok": result}))
except MemoryError:
    print(json.dumps({"err": "MemoryError"}))
except Exception as e:
    print(json.dumps({"err": f"{type(e).__name__}: {e}"}))
'''


def _make_preexec(memory_limit_mb: int):
    """preexec_fn that puts the child in its own session and applies rlimits."""
    mem_bytes = memory_limit_mb * 1024 * 1024

    def _child_setup():
        os.setsid()  # own process group → killpg sweeps spawned children too
        for limit, value in (
            (resource.RLIMIT_AS, (mem_bytes, mem_bytes)),
            (resource.RLIMIT_CORE, (0, 0)),
            (resource.RLIMIT_NPROC, (64, 64)),
        ):
            try:
                resource.setrlimit(limit, value)
            except (ValueError, OSError):
                pass  # not all platforms allow all limits from non-root

    return _child_setup


def _spawn_and_communicate(
    cmd: list[str],
    stdin_data: str,
    cwd: str,
    time_limit_s: float,
    memory_limit_mb: int,
) -> tuple[int, str, str, bool]:
    """Run `cmd`. Returns (returncode, stdout, stderr, timed_out).

    On timeout, kills the entire process group so spawned threads/children
    can't outlive the wall-clock cap.
    """
    timed_out = False
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
        preexec_fn=_make_preexec(memory_limit_mb),
        close_fds=True,
    )
    try:
        out, err = proc.communicate(input=stdin_data, timeout=time_limit_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            out, err = "", ""
    rc = proc.returncode if proc.returncode is not None else 0
    return rc, out or "", err or "", timed_out


def run_call_based_test(
    code: str,
    fn_name: str,
    args: list,
    expected: Any,
    time_limit_s: float = 4.0,
    memory_limit_mb: int = 256,
) -> TestResult:
    """Run one call-based test.

    Loads `code` in a subprocess, calls `fn_name(*args)`, and compares the
    return value (by `==`) against `expected`. Both must be JSON-serializable.
    """
    t0 = time.time()
    cwd = tempfile.mkdtemp(prefix="rl_sandbox_")
    try:
        spec = json.dumps({"code": code, "fn_name": fn_name, "args": args})
        rc, out, err, timed_out = _spawn_and_communicate(
            [sys.executable, "-I", "-c", _CALL_BASED_DRIVER],
            stdin_data=spec,
            cwd=cwd,
            time_limit_s=time_limit_s,
            memory_limit_mb=memory_limit_mb,
        )
        duration = time.time() - t0

        if timed_out:
            return TestResult(False, "timeout (wall clock)", duration)

        # Driver always prints exactly one JSON line on stdout — even for its
        # own caught exceptions. If we got nothing parseable, the interpreter
        # itself died (segfault, OOM-killer, etc.).
        last_line = out.strip().splitlines()[-1] if out.strip() else ""
        try:
            payload = json.loads(last_line) if last_line else None
        except json.JSONDecodeError:
            payload = None
        if payload is None:
            return TestResult(False, f"interpreter crash: {err.strip()[-300:]}", duration)

        if "err" in payload:
            return TestResult(False, payload["err"], duration)
        if payload.get("ok") == expected:
            return TestResult(True, "", duration)
        return TestResult(False, f"got {payload.get('ok')!r}, expected {expected!r}", duration)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def run_stdin_stdout_test(
    code: str,
    stdin: str,
    expected_stdout: str,
    time_limit_s: float = 4.0,
    memory_limit_mb: int = 256,
) -> TestResult:
    """Run one stdin/stdout test.

    `code` is a complete Python program. We write it to a temp file, run it
    as a script with `stdin` piped to its stdin, and compare normalized
    stdout against `expected_stdout`.
    """
    t0 = time.time()
    cwd = tempfile.mkdtemp(prefix="rl_sandbox_")
    code_path = os.path.join(cwd, "solution.py")
    try:
        with open(code_path, "w") as f:
            f.write(code)
        rc, out, err, timed_out = _spawn_and_communicate(
            [sys.executable, "-I", code_path],
            stdin_data=stdin,
            cwd=cwd,
            time_limit_s=time_limit_s,
            memory_limit_mb=memory_limit_mb,
        )
        duration = time.time() - t0

        if timed_out:
            return TestResult(False, "timeout (wall clock)", duration)
        if rc != 0:
            return TestResult(False, err.strip()[-300:], duration)
        if _normalize_output(out) == _normalize_output(expected_stdout):
            return TestResult(True, "", duration)
        return TestResult(False, f"got {out[:200]!r}", duration)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def _normalize_output(s: str) -> str:
    """Whitespace normalization for competitive-programming output comparison.

    Strips trailing whitespace per line, drops trailing empty lines, leaves
    internal blank lines and leading whitespace intact.
    """
    lines = [line.rstrip() for line in s.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def run_test(
    kind: str,
    code: str,
    test: dict,
    *,
    fn_name: str | None = None,
    time_limit_s: float = 4.0,
    memory_limit_mb: int = 256,
) -> TestResult:
    """Top-level dispatcher used by the verifier registry in nanochat.rl_data.

    `kind` is one of:
      - "code_call_based":   test = {"args": list, "expected": Any}; needs fn_name
      - "code_stdin_stdout": test = {"stdin": str, "expected": str}
    """
    if kind == "code_call_based":
        if fn_name is None:
            raise ValueError("code_call_based requires fn_name")
        return run_call_based_test(
            code, fn_name, test["args"], test["expected"],
            time_limit_s=time_limit_s, memory_limit_mb=memory_limit_mb,
        )
    if kind == "code_stdin_stdout":
        return run_stdin_stdout_test(
            code, test["stdin"], test["expected"],
            time_limit_s=time_limit_s, memory_limit_mb=memory_limit_mb,
        )
    raise ValueError(f"unknown sandbox test kind: {kind!r}")
