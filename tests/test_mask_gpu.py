"""Numeric equivalence and CPU-only ownership semantics of mask acceleration."""
import os,unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage as cpu
from utils.gpu_ndimage import MaskNdimage
try:
 import cupy as cp
 CUDA=cp.cuda.runtime.getDeviceCount()>0
except Exception:
 CUDA=False

@unittest.skipUnless(CUDA,'CUDA/CuPy unavailable')
class MaskGpuTests(unittest.TestCase):
 def setUp(self):
  self.env=patch.dict(os.environ,{'RECONNECT_COMPUTE_BACKEND':'cuda','RECONNECT_GPU_MIN_VOXELS':'1'})
  self.env.start();self.addCleanup(self.env.stop)
  self.backend=MaskNdimage();self.rng=np.random.default_rng(32)
  self.array=(255*self.rng.random((32,37,41))).astype('float32')
 def test_float_filters(self):
  for name,args,kwargs in [('gaussian_filter',((0,.6,.6),),{}),('gaussian_filter',(4,),{}),('gaussian_filter1d',(.6,),dict(axis=0))]:
   actual=getattr(self.backend,name)(self.array,*args,**kwargs)
   expected=getattr(cpu,name)(self.array,*args,**kwargs)
   np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=8e-5)
   self.assertEqual(actual.dtype,expected.dtype)
 def test_distance_exact_and_sampling(self):
  a=self.array>90
  for sampling in [None,(2.,1.,.5)]:
   actual=self.backend.distance_transform_edt(a,sampling=sampling)
   np.testing.assert_allclose(actual,cpu.distance_transform_edt(a,sampling=sampling),rtol=0,atol=1e-12)
   self.assertEqual(actual.dtype,np.float64)
 def test_nearest_indices_remain_cpu(self):
  a=self.array>90
  result=self.backend.distance_transform_edt(a,return_indices=True)
  for x,y in zip(result,cpu.distance_transform_edt(a,return_indices=True)):np.testing.assert_array_equal(x,y)
  self.assertFalse(self.backend.stats['gpu_calls'])
 def test_all_foreground_remains_cpu(self):
  a=np.ones((4,5,6),bool)
  np.testing.assert_array_equal(self.backend.distance_transform_edt(a),cpu.distance_transform_edt(a))
  self.assertFalse(self.backend.stats['gpu_calls'])
 def test_discrete_filters_exact(self):
  binary=self.array>90
  for name,a,args,kwargs in [('maximum_filter',self.array,(),dict(size=3)),('minimum_filter',self.array,(),dict(size=3)),('maximum_filter1d',self.array,(5,),dict(axis=0)),('convolve',binary.astype('uint8'),(np.ones((1,3,3),np.uint8),),dict(mode='constant')),('binary_dilation',binary,(np.ones((3,3,3)),),dict(iterations=2)),('binary_erosion',binary,(np.ones((3,3,3)),),{})]:
   np.testing.assert_array_equal(getattr(self.backend,name)(a,*args,**kwargs),getattr(cpu,name)(a,*args,**kwargs))
 def test_cpu_backend_and_small_input(self):
  for env in [dict(RECONNECT_COMPUTE_BACKEND='cpu'),dict(RECONNECT_GPU_MIN_VOXELS='1000000')]:
   with patch.dict(os.environ,env):
    b=MaskNdimage();np.testing.assert_array_equal(b.gaussian_filter(self.array,.6),cpu.gaussian_filter(self.array,.6))
    self.assertFalse(b.stats['gpu_calls'])
 def test_oom_falls_back_for_rest_of_worker(self):
  b=self.backend;b.initialize()
  with patch.object(b.gpu,'gaussian_filter',side_effect=cp.cuda.memory.OutOfMemoryError(100,0)):
   result=b.gaussian_filter(self.array,.6)
  np.testing.assert_array_equal(result,cpu.gaussian_filter(self.array,.6))
  self.assertTrue(b.gpu_disabled)
  np.testing.assert_array_equal(b.gaussian_filter(self.array,1),cpu.gaussian_filter(self.array,1))
  self.assertEqual(b.stats['cpu_fallbacks'][0]['reason'],'cuda_out_of_memory')

if __name__=='__main__':unittest.main()
