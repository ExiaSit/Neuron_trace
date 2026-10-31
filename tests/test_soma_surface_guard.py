"""Soma-top limits preserve original surface foreground and normal lateral growth."""
import unittest
import numpy as np

from algorithms.mask_repair.soma_surface_guard import restrict, select_top


class SomaSurfaceGuardTests(unittest.TestCase):
    def test_preserves_original_surface_and_allows_lateral_and_deep_additions(self):
        original=np.zeros((15,15,15),bool)
        original[6:10,6:9,6:9]=True
        original[1,3,3]=True  # preexisting surface artifact must remain unchanged
        proposed=original.copy();proposed[2:12,5:10,5:10]=True
        result,report=restrict(original,proposed,6)
        np.testing.assert_array_equal(result[:6],original[:6])
        self.assertTrue(result[original].all())
        self.assertTrue(result[6,5,6])  # at soma top: lateral growth is allowed
        self.assertTrue(result[11,6,6])
        self.assertGreater(report['blocked_above_soma'],0)
        self.assertEqual(report['final_added_voxels'],np.count_nonzero(result&~original))

    def test_cut_bridge_does_not_leave_a_new_floating_fragment(self):
        original=np.zeros((12,12,12),bool);original[2,4,4]=True
        proposed=original.copy();proposed[2:10,4,4]=True
        result,report=restrict(original,proposed,6)
        np.testing.assert_array_equal(result,original)
        self.assertEqual(report['blocked_above_soma'],3)
        self.assertEqual(report['detached_additions_removed'],4)

    def test_original_disconnected_deep_fragment_is_always_preserved(self):
        original=np.zeros((12,12,12),bool);original[2,4,4]=original[9,4,4]=True
        proposed=original.copy();proposed[2:10,4,4]=True
        result,_=restrict(original,proposed,6)
        self.assertTrue(result[original].all())
        self.assertTrue(result[6:10,4,4].all())
        self.assertFalse(result[3:6].any())

    def test_unrelated_preexisting_deep_soma_proposal_is_unchanged(self):
        original=np.zeros((12,12,12),bool);original[2,2,2]=True
        proposed=original.copy();proposed[8:10,8:10,8:10]=True
        result,report=restrict(original,proposed,6)
        np.testing.assert_array_equal(result,proposed)
        self.assertEqual(report['detached_additions_removed'],0)

    def test_uses_target_completed_body_and_not_another_soma_or_center_z(self):
        target=dict(center_xyz=[10,10,20],completed_soma_top_z=13,
                    completed_soma_bbox_zyx=[[13,28],[5,16],[5,16]])
        other=dict(center_xyz=[30,10,20],completed_soma_top_z=2,
                   completed_soma_bbox_zyx=[[2,28],[5,16],[25,36]])
        top,sources=select_top([10,10,20],dict(anchors=[other,target]),
                               dict(soma=dict(status='supported_soma',completed_soma_top_z=12)))
        self.assertEqual(top,12)
        self.assertEqual([s['top_z'] for s in sources],[13,12])

    def test_no_supported_body_withholds_additions_instead_of_guessing_a_plane(self):
        top,sources=select_top([5,5,5],{},dict(soma=dict(status='no_signal_at_metadata')))
        self.assertIsNone(top);self.assertEqual(sources,[])
        original=np.zeros((10,10,10),bool);original[5,5,5]=True
        proposed=original.copy();proposed[6,5,5]=True
        result,report=restrict(original,proposed,top)
        np.testing.assert_array_equal(result,original)
        self.assertEqual(report['status'],'no_supported_target_soma')

    def test_unrestricted_mode_and_bad_proposals(self):
        original=np.zeros((9,9,9),bool);original[5,5,5]=True
        proposed=original.copy();proposed[4,5,5]=True
        np.testing.assert_array_equal(restrict(original,proposed,None,'unrestricted')[0],proposed)
        with self.assertRaises(ValueError):restrict(original,np.zeros_like(original),5)
        with self.assertRaises(ValueError):restrict(original,proposed,-1)


if __name__=='__main__':unittest.main()
