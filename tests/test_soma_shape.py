import unittest
import numpy as np
from scipy import ndimage as ndi
from algorithms.tracing.skeleton_input import skeleton_with_soma


class SomaSurfaceTests(unittest.TestCase):
    def geometry(self):
        z,y,x=np.indices((49,49,49))
        body=((x-24)/10)**2+((y-24)/12)**2+((z-24)/9)**2<=1
        signal=((x-25)/10)**2+((y-24)/12)**2+((z-24)/9)**2<=1
        mask=ndi.binary_dilation(body,iterations=3)
        mask[24,24,3:46]=True
        raw=10+200*ndi.gaussian_filter(signal.astype(float),.8)
        return mask,body,raw.astype(np.float32)

    def test_image_fit_preserves_skeleton_seed_and_mask(self):
        mask,body,raw=self.geometry()
        old=mask.copy()
        sk,_,_,_=skeleton_with_soma(mask,body,[24,24,24],raw=raw,soma_shape='opening')
        sk2,fitted,merged,a=skeleton_with_soma(mask,body,[24,24,24],raw=raw)
        self.assertEqual(a['soma_regularization']['surface_fit']['status'],'accepted')
        np.testing.assert_array_equal(mask,old)
        np.testing.assert_array_equal(sk,sk2)
        self.assertTrue(fitted[24,24,24])
        self.assertFalse((merged & ~mask).any())
        self.assertEqual(a['target_skeleton_unreachable'],0)
        # The measured image is shifted in X: the surface follows that evidence.
        self.assertGreater(np.argwhere(fitted).mean(0)[2],np.argwhere(body).mean(0)[2]+.15)

    def test_low_contrast_falls_back_to_opening(self):
        mask,body,raw=self.geometry();raw[:]=80
        _,expected,_,_=skeleton_with_soma(mask,body,[24,24,24],raw=raw,soma_shape='opening')
        _,actual,_,audit=skeleton_with_soma(mask,body,[24,24,24],raw=raw)
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(audit['soma_regularization']['surface_fit']['reason'],'low_contrast')

    def test_zero_radius_disables_all_body_processing(self):
        mask,body,raw=self.geometry()
        _,actual,_,_=skeleton_with_soma(mask,body,[24,24,24],0,raw=raw)
        np.testing.assert_array_equal(actual,body)

    def test_does_not_connect_neighboring_skeleton_component(self):
        mask,body,raw=self.geometry()
        mask[24,24,38:42]=False
        mask[19:30,19:30,42:47]=True;raw[19:30,19:30,42:47]=240
        sk,_,merged,a=skeleton_with_soma(mask,body,[24,24,24],raw=raw)
        labels,_=ndi.label(merged,np.ones((3,3,3)))
        self.assertNotEqual(labels[24,24,24],labels[sk & (np.indices(sk.shape)[2]>=42)].max())
        self.assertEqual(a['soma_regularization']['surface_fit'].get('newly_connected_skeleton_voxels',0),0)


if __name__=='__main__':unittest.main()
