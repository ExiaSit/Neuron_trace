"""Per-body ceilings, mask-geodesic attribution and ambiguity safeguards."""
import unittest
import numpy as np
from algorithms.mask_repair.soma_surface_guard import completed_bodies,restrict_per_soma


def body(point,source='f'):
    p=np.array([point],int)
    return dict(points_zyx=p,center_xyz=p[0,::-1].tolist(),top_z=int(p[0,0]),source=source)


class PerSomaGuardTests(unittest.TestCase):
    def test_two_depths_have_independent_ceilings_and_lateral_growth(self):
        old=np.zeros((18,12,20),bool);old[4,5,3]=old[11,5,15]=True
        proposal=old.copy();proposal[2:9,5,3]=True;proposal[8:16,5,15]=True
        proposal[4,5,4]=True;proposal[11,5,16]=True
        result,report,owners=restrict_per_soma(old,proposal,[body([4,5,3]),body([11,5,15])])
        self.assertTrue(result[4,5,4]);self.assertTrue(result[11,5,16])
        self.assertFalse(result[:4,5,3].any());self.assertFalse(result[8:11,5,15].any())
        self.assertTrue(result[7,5,3]);self.assertTrue(result[14,5,15])
        self.assertEqual(owners[7,5,3],1);self.assertEqual(owners[14,5,15],2)
        self.assertTrue(result[old].all());self.assertEqual(report['soma_count'],2)
        self.assertEqual(sum(r['final_added_voxels'] for r in report['somas']),np.count_nonzero(result&~old))

    def test_nearby_disconnected_soma_cannot_steal_a_branch(self):
        old=np.zeros((15,15,15),bool);old[8,2,2]=old[6,4,10]=True
        proposal=old.copy();proposal[8,2,2:12]=True
        result,_,owners=restrict_per_soma(old,proposal,[body([8,2,2]),body([6,4,10])])
        self.assertEqual(owners[8,2,10],1)  # soma 2 is much closer in Euclidean space
        self.assertTrue(result[8,2,10])

    def test_mixed_component_competes_and_withholds_tied_additions(self):
        old=np.zeros((14,14,14),bool);old[3,5,2]=old[9,5,8]=True
        proposal=old.copy()
        for k in range(7):proposal[3+k,5,2+k]=True
        result,report,owners=restrict_per_soma(old,proposal,[body([3,5,2]),body([9,5,8])])
        self.assertEqual(owners[6,5,5],0);self.assertFalse(result[6,5,5])
        self.assertGreater(report['ambiguous_additions_withheld'],0)
        self.assertEqual(owners[4,5,3],1);self.assertTrue(result[4,5,3])
        self.assertTrue(result[old].all())

    def test_body_must_remain_reachable_after_its_upper_bridge_is_cut(self):
        old=np.zeros((16,12,12),bool);old[8,5,2]=old[12,5,8]=True
        proposal=old.copy();proposal[4:9,5,2]=True;proposal[4,5,2:9]=True;proposal[4:13,5,8]=True
        result,report,_=restrict_per_soma(old,proposal,[body([8,5,2])])
        np.testing.assert_array_equal(result,old)
        self.assertEqual(report['detached_additions_removed'],4)

    def test_no_soma_and_unseeded_components_keep_original_only(self):
        old=np.zeros((12,12,12),bool);old[3,3,3]=True
        proposal=old.copy();proposal[4,3,3]=True;proposal[9,9,9]=True
        result,report,_=restrict_per_soma(old,proposal,[])
        np.testing.assert_array_equal(result,old)
        self.assertEqual(report['unassigned_additions_withheld'],2)
        result,report,_=restrict_per_soma(old,proposal,[body([3,3,3])])
        self.assertTrue(result[4,3,3]);self.assertFalse(result[9,9,9])

    def test_identified_new_soma_can_seed_its_own_independent_region(self):
        old=np.zeros((12,12,12),bool);old[2,2,2]=True
        proposal=old.copy();proposal[7:10,8,8]=True
        result,_,_=restrict_per_soma(old,proposal,[body([7,8,8])])
        np.testing.assert_array_equal(result,proposal)

    def test_same_target_merges_by_overlap_but_distinct_f_bodies_do_not(self):
        a=body([4,5,3]);b=body([9,5,9]);shape=(15,15,15)
        report=dict(completed_somas=[{k:v for k,v in q.items() if k!='points_zyx'} for q in [a,b]])
        archive={'body_0':a['points_zyx'],'body_1':b['points_zyx']}
        bodies=completed_bodies(report,archive,[body([4,5,3],'g_target')],shape)
        self.assertEqual(len(bodies),2);self.assertEqual(bodies[0]['sources'],['f','g_target'])
        bodies=completed_bodies(report,archive,[body([4,5,4],'g_target')],shape)
        self.assertEqual(len(bodies),3)  # proximity without overlap is insufficient

    def test_empty_mask_and_inconsistent_completed_region(self):
        old=np.zeros((4,4,4),bool)
        result,report,_=restrict_per_soma(old,old,[])
        self.assertFalse(result.any());self.assertEqual(report['graph_nodes'] if 'graph_nodes' in report else report['ownership']['graph_nodes'],0)
        with self.assertRaises(ValueError):restrict_per_soma(old,old,[body([2,2,2])])


if __name__=='__main__':unittest.main()
