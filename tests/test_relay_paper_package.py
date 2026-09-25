"""CPU-only packaging, source-equivalence and GPU-admission guards."""
import ast
import copy
import hashlib
import json
import unittest
from scripts.inspect import qwen_relay_paper_provenance as provenance
from scripts.inspect import qwen_relay_paper_queue as queue


class DropImports(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        # Python 3.12 added empty type_params fields to non-generic functions.
        # Normalize that representation across the archived Python 3.10 runtime.
        if 'type_params' in node._fields:
            assert not node.type_params
            node._fields = tuple(f for f in node._fields if f != 'type_params')
        return node

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def ast_value(node):
    if isinstance(node, ast.AST):
        return [type(node).__name__, [[key, ast_value(value)] for key, value in ast.iter_fields(node)]]
    if isinstance(node, list):
        return [ast_value(value) for value in node]
    return node


class PaperPackageTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(provenance.MANIFEST.read_text())

    def test_reviewed_source_hashes(self):
        provenance.validate_baseline_sources({'hashes': self.manifest['baseline_source_hashes']})

    def test_different_baseline_source_rejected(self):
        hashes = dict(self.manifest['baseline_source_hashes'])
        hashes['qwen_relay_supplied_common.py'] = 'changed'
        with self.assertRaises(ValueError):
            provenance.validate_baseline_sources({'hashes': hashes})

    def test_executed_function_bodies_preserved(self):
        for entry in self.manifest['function_fingerprints']:
            with self.subTest(function=entry['function'], path=entry['path']):
                tree = ast.parse((provenance.ROOT / entry['path']).read_text())
                node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == entry['function'])
                node = DropImports().visit(copy.deepcopy(node))
                digest = hashlib.sha256(json.dumps(ast_value(node), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(digest, entry['ast_without_imports_sha256'])

    def test_idle_gpu_thresholds(self):
        idle = dict(has_process=False, memory_mib=999, utilization=4)
        self.assertTrue(queue.available(idle))
        for change in ({'has_process': True}, {'memory_mib': 1000}, {'utilization': 5}):
            self.assertFalse(queue.available({**idle, **change}))


if __name__ == '__main__':
    unittest.main()
