import heapq
import itertools
import json
from pathlib import Path
import unittest

import numpy as np
from scipy import ndimage as ndi

from algorithms.tracing.app2_gsdt import distance_transform, SOURCE
from algorithms.tracing.gcut_adapter import component_maxima


def reference(raw, threshold, connectivity):
    """Independent small-volume shortest-path oracle including APP2 boundary seeds."""
    offsets = [p for p in itertools.product((-1, 0, 1), repeat=3)
               if 0 < sum(abs(v) for v in p) <= connectivity]
    def neighbors(p):
        for delta in offsets:
            q = tuple(a+b for a, b in zip(p, delta))
            if all(0 <= a < b for a, b in zip(q, raw.shape)):
                yield q, np.sqrt(sum(v*v for v in delta))
    bg = raw <= threshold
    field = np.where(bg, raw, np.inf).astype(np.float32)
    heap = []
    for p in map(tuple, np.argwhere(~bg)):
        adjacent = [float(raw[q]) for q, _ in neighbors(p) if bg[q]]
        if adjacent:
            field[p] = min(adjacent) + float(raw[p])
            heapq.heappush(heap, (float(field[p]), p))
    while heap:
        value, p = heapq.heappop(heap)
        if value != field[p]:
            continue
        for q, step in neighbors(p):
            if bg[q]:
                continue
            value2 = np.float32(value + float(raw[q])*step)
            if value2 < field[q]:
                field[q] = value2
                heapq.heappush(heap, (float(value2), q))
    return field


class TestApp2GSDT(unittest.TestCase):
    def test_native_matches_independent_oracle_all_connectivities(self):
        rng = np.random.default_rng(37)
        raw = rng.integers(0, 200, (7, 8, 9), dtype=np.uint16)
        raw[[0, -1]] = 0
        for connectivity in (1, 2, 3):
            field, _ = distance_transform(raw, 35.8, connectivity)
            np.testing.assert_allclose(field, reference(raw, 35, connectivity), atol=1e-4)

    def test_gray_values_matter_and_axis_order_preserved(self):
        raw = np.zeros((13, 15, 23), np.uint16)
        raw[2:11, 2:13, 2:11] = 50
        raw[2:11, 2:13, 13:22] = 100
        field, _ = distance_transform(raw, 0)
        self.assertEqual(field.shape, raw.shape)
        self.assertEqual(field[6, 7, 6], 250)
        self.assertEqual(field[6, 7, 17], 500)

    def test_peak_selection_can_move_with_fixed_geometric_gate(self):
        raw = np.zeros((21, 25, 29), np.uint8)
        raw[2:19, 2:23, 2:27] = np.arange(25)*7+20
        edt = ndi.distance_transform_edt(raw > 10)
        labels, count = ndi.label(edt > 5)
        field, _ = distance_transform(raw, 10)
        a, b = component_maxima(edt, labels, count), component_maxima(field, labels, count)
        self.assertEqual(count, 1)
        self.assertNotEqual(a, b)
        self.assertGreater(edt[b[0]], 5)
        self.assertGreater(field[b[0]], field[a[0]])

    def test_background_retains_native_gray_and_no_background_is_rejected(self):
        raw = np.full((3, 4, 5), 7, np.uint8)
        np.testing.assert_array_equal(distance_transform(raw, 7)[0], raw)
        with self.assertRaisesRegex(ValueError, 'no background'):
            distance_transform(raw, 0)

    def test_invalid_inputs_fail_explicitly(self):
        for raw in (np.zeros((3, 4), np.uint8), np.full((3, 4, 5), np.nan, np.float32),
                    np.full((3, 4, 5), -1, np.float32)):
            with self.assertRaises(ValueError):
                distance_transform(raw, 0)

    def test_vendored_headers_match_recorded_upstream(self):
        import hashlib
        provenance = json.loads((SOURCE/'provenance.json').read_text())
        for name, digest in provenance['files_sha256'].items():
            self.assertEqual(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
