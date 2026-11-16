"""Run with python3; tests use temporary projects, never real research records."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("context.py")


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_context(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), "--project", str(self.root), *args],
                              capture_output=True, text=True)

    def test_initialization_preserves_existing_work(self):
        self.assertEqual(self.run_context("init").returncode, 0)
        state = self.root / ".research/STATE.md"
        state.write_text("Existing validated finding\n")
        (self.root / "AGENTS.md").write_text("Existing project rules\n")
        result = self.run_context("init")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state.read_text(), "Existing validated finding\n")
        self.assertEqual((self.root / "AGENTS.md").read_text(), "Existing project rules\n")
        self.assertIn("*", (self.root / ".research/.gitignore").read_text().splitlines())

    def test_resume_reads_only_current_files(self):
        self.run_context("init")
        (self.root / ".research/archive.json").write_text("PRIVATE_ARCHIVE" * 50000)
        result = self.run_context("resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("PRIVATE_ARCHIVE", result.stdout)
        self.assertLess(len(result.stdout), 4000)

    def test_section_includes_children_not_code_headings(self):
        (self.root / "notes.md").write_text(
            "# Notes\n## Keep\nA\n```md\n## End\n```\n### Child\nB\n## End\nC\n")
        result = self.run_context("read", "notes.md", "--heading", "Keep")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("### Child\nB", result.stdout)
        self.assertNotIn("\nC", result.stdout)
        self.assertNotEqual(self.run_context("read", "notes.md", "--heading", "Absent").returncode, 0)

    def test_json_selection_and_output_limit(self):
        (self.root / "result.json").write_text(json.dumps({"runs": [{"score": 0.5}], "other": "X" * 20000}))
        result = self.run_context("read", "result.json", "--key", "runs.0.score")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), 0.5)
        result = self.run_context("read", "result.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("Narrow the selection", result.stderr)

    def test_outside_paths_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "unrelated.txt"
            secret.write_text("UNRELATED_FILE_CONTENT")
            (self.root / "link").symlink_to(secret)
            for path in (str(secret), "link"):
                result = self.run_context("read", path)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("UNRELATED_FILE_CONTENT", result.stdout + result.stderr)
            (self.root / ".research").symlink_to(outside, target_is_directory=True)
            self.assertNotEqual(self.run_context("init").returncode, 0)
            self.assertEqual(list(Path(outside).iterdir()), [secret])

    def test_missing_memory_does_not_initialize_itself(self):
        self.assertNotEqual(self.run_context("resume").returncode, 0)
        self.assertFalse((self.root / ".research").exists())


if __name__ == "__main__":
    unittest.main()
