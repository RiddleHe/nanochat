"""CPU-only statistics checks; optionally audit externally stored gradient results."""
import argparse
import json
import math
import statistics
import unittest
from pathlib import Path

RECORDS = None


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))


class GradientStatisticsTest(unittest.TestCase):
    def test_average_order_can_flip_sign(self):
        first, second = [(4, 1), (-1, -4)], [(-1, -4), (4, 1)]
        mean_cosine = statistics.mean(cosine(a, b) for a, b in zip(first, second))
        mean_first = [statistics.mean(v) for v in zip(*first)]
        mean_second = [statistics.mean(v) for v in zip(*second)]
        self.assertAlmostEqual(mean_cosine, -8/17)
        self.assertAlmostEqual(cosine(mean_first, mean_second), 1)

    def test_all_recorded_heatmap_cells(self):
        if RECORDS is None:
            self.skipTest("Pass --input RUN_DIR to audit measured heatmap cells")
        batches = [json.loads(line) for line in (RECORDS / "batches.jsonl").read_text().splitlines()]
        summary = json.loads((RECORDS / "results.json").read_text())
        self.assertEqual([b["batch"] for b in batches], list(range(32)))
        for item in summary["summary"]:
            rows = [next(m for m in b["matrices"] if m["name"] == item["name"]) for b in batches]
            for a in range(4):
                for b in range(a+1, 4):
                    values = [r["cosine"][a][b] for r in rows if r["valid_norms"][a] and r["valid_norms"][b]]
                    with self.subTest(matrix=item["name"], pair=(a, b)):
                        self.assertEqual(len(values), 32)
                        self.assertAlmostEqual(statistics.mean(values), item["mean_cosine"][a][b], places=14)
        self.assertLess(summary["max_sum_relative_error"], 1e-6)

    def test_direction_examples(self):
        self.assertAlmostEqual(cosine((1, 0), (2, 0)), 1)
        self.assertAlmostEqual(cosine((1, 0), (-2, 0)), -1)
        self.assertAlmostEqual(cosine((1, 0), (0, 2)), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Run directory containing results/gradient/")
    args, remaining = parser.parse_known_args()
    if args.input is not None:
        RECORDS = args.input / "results/gradient"
    unittest.main(argv=[__file__, *remaining])
