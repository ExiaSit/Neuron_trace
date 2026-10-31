import numpy as np
import unittest
from algorithms.tracing.skeleton_input import skeleton_with_soma


class SkeletonInputTests(unittest.TestCase):
    def test_skeleton_soma_preserves_seed_branches_and_source(self):
        z,y,x=np.indices((35,35,35))
        soma=(x-17)**2+(y-17)**2+(z-17)**2<=25
        mask=soma.copy()
        mask[16:19,16:19,3:32]=True
        mask[16:19,17:32,16:19]=True
        original=mask.copy()
        sk,body,merged,a=skeleton_with_soma(mask,soma,[17,17,17])
        assert np.array_equal(mask,original)
        assert merged[17,17,17] and np.all(merged[body])
        assert not (body & ~soma).any()
        assert not (merged & ~mask).any()
        assert merged[15:20,15:20,3:7].any()
        assert merged[15:20,28:32,15:20].any()
        assert a['components_before']==a['components_after']==1
        assert a['target_skeleton_unreachable']==0


    def test_no_bridge_between_separate_objects(self):
        mask=np.zeros((25,25,25),bool)
        mask[2:9,2:9,2:9]=True
        mask[16:23,16:23,16:23]=True
        soma=np.zeros_like(mask);soma[2:9,2:9,2:9]=True
        *_,a=skeleton_with_soma(mask,soma,[5,5,5])
        assert a['components_before']==a['components_after']==2


    def test_empty_or_misaligned_soma_rejected(self):
        mask=np.ones((9,9,9),bool)
        with unittest.TestCase().assertRaisesRegex(ValueError,'does not contain'):
            skeleton_with_soma(mask,np.zeros_like(mask),[4,4,4])
        with unittest.TestCase().assertRaisesRegex(ValueError,'outside'):
            skeleton_with_soma(mask,mask,[9,4,4])

    def test_opening_removes_soma_spike_but_preserves_neurite_skeleton(self):
        z,y,x=np.indices((41,41,41))
        soma=((x-20)/8)**2+((y-20)/10)**2+((z-20)/7)**2<=1
        soma[20,20,27:34]=True
        mask=soma.copy();mask[20,20,30:39]=True
        old_sk,_,old,_=skeleton_with_soma(mask,soma,[20,20,20],0)
        sk,body,merged,a=skeleton_with_soma(mask,soma,[20,20,20])
        self.assertFalse(body[20,20,32])
        np.testing.assert_array_equal(sk,old_sk)
        self.assertTrue(merged[20,20,36])
        self.assertTrue(np.all(old[merged]))
        self.assertEqual(a['target_skeleton_unreachable'],0)
        self.assertGreaterEqual(body.sum()/soma.sum(),.8)

    def test_small_soma_falls_back_without_erasing_seed(self):
        soma=np.zeros((13,13,13),bool);soma[5:8,5:8,5:8]=True
        sk,body,merged,a=skeleton_with_soma(soma,soma,[6,6,6])
        np.testing.assert_array_equal(body,soma)
        self.assertEqual(a['soma_regularization']['selected_radius_um'],0)
        self.assertTrue(merged[6,6,6])

    def test_disabled_opening_reproduces_measured_soma_union(self):
        rng=np.random.default_rng(9)
        soma=rng.random((11,11,11))>.3;soma[5,5,5]=True
        sk,body,merged,a=skeleton_with_soma(soma,soma,[5,5,5],0)
        np.testing.assert_array_equal(body,soma)
        np.testing.assert_array_equal(merged,sk|soma)

    def test_connectivity_guard_restores_soma_bridge_needed_by_skeleton(self):
        from unittest.mock import patch
        z,y,x=np.indices((31,31,41))
        soma=(x-15)**2+(y-15)**2+(z-15)**2<=64
        soma[15,15,23:33]=True
        mask=soma.copy();mask[15,15,32:39]=True
        skeleton=np.zeros_like(mask);skeleton[15,15,15]=True;skeleton[15,15,32:39]=True
        with patch('algorithms.tracing.skeleton_input.skeletonize',return_value=skeleton):
            _,body,merged,a=skeleton_with_soma(mask,soma,[15,15,15])
        np.testing.assert_array_equal(body,soma)
        self.assertEqual(a['soma_regularization']['selected_radius_um'],0)
        self.assertEqual(a['target_skeleton_unreachable'],0)
        self.assertTrue(any(t['status']=='rejected_connectivity' for t in a['soma_regularization']['trials']))

if __name__ == "__main__": unittest.main()
