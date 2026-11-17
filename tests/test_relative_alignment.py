import unittest
import numpy as np
from scripts.inspect.qkv_input_alignment.relative_alignment import normalize_alignment, Moments


class RelativeAlignmentTest(unittest.TestCase):
    def test_normalize_before_token_aggregation(self):
        q, _, n = normalize_alignment([[.01, .04, .09], [.01, .01, .64]], metric='length')
        np.testing.assert_allclose(q, [[.5, 1, 1.5], [.3, .3, 2.4]])
        np.testing.assert_equal(n, [3, 3])
        moments = Moments((3,))
        for chunk in np.split(q, 2):
            moments.add(chunk)
        result = moments.arrays()
        np.testing.assert_allclose(result['token_mean'], [.4, .65, 1.95])
        np.testing.assert_allclose(result['token_std'], [.1, .35, .45])

    def test_zero_projection_is_valid_but_nan_is_excluded(self):
        q, denominator, n = normalize_alignment([[0, .25, np.nan]])
        np.testing.assert_allclose(q, [[0, 2, np.nan]], equal_nan=True)
        np.testing.assert_allclose(denominator, [.125])
        np.testing.assert_equal(n, [2])

    def test_zero_denominator_and_missing_bank(self):
        q, _, n = normalize_alignment([[0, 0], [np.nan, np.nan]])
        self.assertTrue(np.isnan(q).all())
        np.testing.assert_equal(n, [2, 0])
        moments = Moments((2,)); moments.add(q)
        self.assertTrue(np.isnan(moments.arrays()['token_mean']).all())
        np.testing.assert_equal(moments.count, [0, 0])

    def test_single_source_and_scale_invariance(self):
        q, _, _ = normalize_alignment([[.01], [.81]])
        np.testing.assert_equal(q, [[1], [1]])
        a = np.array([[.01, .04, .09]])
        np.testing.assert_allclose(normalize_alignment(a)[0], normalize_alignment(a*4)[0])
        np.testing.assert_allclose(normalize_alignment(a[:, ::-1])[0], normalize_alignment(a)[0][:, ::-1])

    def test_invalid_input_is_rejected(self):
        for a in [[-.1], [1.1], [np.inf]]:
            with self.assertRaises(ValueError):
                normalize_alignment(a)

    def test_squared_metric_preserves_squared_ratios(self):
        q, denominator, n = normalize_alignment([[.04, .01], [.01, .01]])
        np.testing.assert_allclose(q, [[1.6, .4], [1, 1]])
        np.testing.assert_allclose(denominator, [.025, .01])
        np.testing.assert_equal(n, [2, 2])
        self.assertAlmostEqual(q[0, 0] / q[0, 1], 4)
        moments = Moments((2,))
        moments.add(q[:1]); moments.add(q[1:])
        result = moments.arrays()
        np.testing.assert_allclose(result['token_mean'], [1.3, .7])
        np.testing.assert_allclose(result['token_std'], [.3, .3])
        # The per-token mean differs from normalizing after token averaging.
        self.assertNotAlmostEqual(result['token_mean'][0], .025 / .0175)

    def test_squared_metric_uses_every_source_and_excludes_nan(self):
        q, denominator, n = normalize_alignment([[.04, .01, .05, np.nan]])
        np.testing.assert_allclose(q, [[1.2, .3, 1.5, np.nan]], equal_nan=True)
        np.testing.assert_allclose(denominator, [.1 / 3])
        np.testing.assert_equal(n, [3])
        with self.assertRaises(ValueError):
            normalize_alignment([[.1]], metric='unknown')


if __name__ == '__main__':
    unittest.main()
