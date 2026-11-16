#!/usr/bin/env python3
"""Create missing private memory files or retrieve bounded project context.

Python 3.9+, standard library only. No network, subprocesses, or model calls.
"""
import argparse
import json
from pathlib import Path
import re
import sys


def local_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path resolves outside the selected project")
    return path


def section(text, title):
    """Select one ATX Markdown heading, including its nested subsections."""
    lines = text.splitlines(keepends=True)
    headings = []
    fence = None
    for i, line in enumerate(lines):
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\n"))
        if fence_match:
            marker, suffix = fence_match.groups()
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
                fence = None
            continue
        if fence is not None:
            continue
        match = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*$", line)
        if match:
            name = re.sub(r"\s+#+\s*$", "", match[2])
            headings.append((i, len(match[1]), name))
    matches = [h for h in headings if h[2] == title]
    if len(matches) != 1:
        raise ValueError("Heading must match exactly once; found %d" % len(matches))
    start, level, _ = matches[0]
    end = next((i for i, depth, _ in headings if i > start and depth <= level), len(lines))
    return "".join(lines[start:end])


def initialize(root):
    assets = Path(__file__).resolve().parent.parent / "assets"
    memory = local_path(root, ".research")
    memory.mkdir(exist_ok=True)
    content = {".gitignore": "# Local research memory is private by default.\n*\n"}
    for name in ("STATE.md", "CURRENT_TASK.md", "INDEX.md", "RUNBOOK.md"):
        content[name] = (assets / name).read_text(encoding="utf-8")
    changes = []
    for name, value in content.items():
        dest = local_path(root, str(memory / name))
        try:
            with dest.open("x", encoding="utf-8") as handle:
                handle.write(value)
            changes.append("Created .research/" + name)
        except FileExistsError:
            if not dest.is_file():
                raise ValueError("Expected a file: " + str(dest))
            changes.append("Preserved .research/" + name)
    changes.append("Fill state/task from the actual request. Existing tracked files stay tracked;")
    changes.append("verify Git ignores private records before committing. Agent rules were not edited.")
    return "\n".join(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--max-chars", type=int, default=12000)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "resume", "index"):
        commands.add_parser(command)
    read = commands.add_parser("read")
    read.add_argument("path")
    selection = read.add_mutually_exclusive_group()
    selection.add_argument("--heading")
    selection.add_argument("--key", help="Dot-separated JSON object keys or list indices")
    args = parser.parse_args()
    try:
        root = args.project.resolve(strict=True)
        if not root.is_dir() or args.max_chars <= 0:
            raise ValueError("Use an existing project directory and positive --max-chars")
        if args.command == "init":
            output = initialize(root)
        elif args.command in ("resume", "index"):
            names = ["STATE.md", "CURRENT_TASK.md"] if args.command == "resume" else ["INDEX.md"]
            output = "\n\n".join(
                "--- " + name + " ---\n" + local_path(root, ".research/" + name).read_text(encoding="utf-8")
                for name in names
            )
        else:
            output = local_path(root, args.path).read_text(encoding="utf-8")
            if args.heading is not None:
                output = section(output, args.heading)
            elif args.key is not None:
                value = json.loads(output)
                for key in args.key.split("."):
                    value = value[int(key)] if isinstance(value, list) else value[key]
                output = json.dumps(value, ensure_ascii=False, indent=2)
        if len(output) > args.max_chars:
            raise ValueError("Output is %d characters; cap is %d. Narrow the selection or explicitly raise --max-chars."
                             % (len(output), args.max_chars))
        print(output)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print("Context error: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
