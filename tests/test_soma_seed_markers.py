import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.compare_soma_seed_markers import spatial_pairs, marker_coordinates, write_marker


class SomaMarkerTests(unittest.TestCase):
    def test_maximum_cardinality_precedes_distance(self):
        pairs, old_only, new_only = spatial_pairs([[0,0,0],[3,0,0]], [[1,0,0],[-2,0,0]], 3)
        self.assertEqual({(i,j) for i,j,_ in pairs}, {(0,1),(1,0)})
        self.assertEqual(old_only+new_only, [])

    def test_one_to_one_boundary_and_no_distant_pairing(self):
        pairs, old_only, new_only=spatial_pairs([[0,0,0],[0,0,1]], [[0,0,0],[20,0,0]], 6)
        self.assertEqual(len(pairs),1)
        self.assertEqual(old_only,[1]);self.assertEqual(new_only,[1])
        self.assertEqual(len(spatial_pairs([[0,0,0]], [[6,0,0]], 6)[0]),1)
        self.assertEqual(len(spatial_pairs([[0,0,0]], [[6.01,0,0]], 6)[0]),0)
        self.assertEqual(spatial_pairs([],[[1,2,3]])[2],[0])

    def test_image_axes_and_tiff_transform(self):
        np.testing.assert_array_equal(marker_coordinates([[2,3,4]],[7,11,13]),[[3,4,5]])
        np.testing.assert_array_equal(marker_coordinates([[2,3,4]],[7,11,13],True),[[3,8,5]])
        with self.assertRaises(ValueError):marker_coordinates([[13,3,4]],[7,11,13])

    def test_marker_roundtrip_and_empty_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'points.marker'
            points=[dict(xyz=[2,3,4],name='new_C001',comment='display, only',color=[32,255,120])]
            result=write_marker(path,points,[7,11,13])
            self.assertEqual(result['count'],1)
            with path.open() as f:
                next(f);row=next(csv.reader(f))
            self.assertEqual(float(row[3]),8)
            self.assertEqual(row[6],'display, only')
            self.assertEqual(points[0]['xyz'],[2,3,4])
            self.assertEqual(write_marker(path,[],[7,11,13])['count'],0)

if __name__=='__main__':unittest.main()
