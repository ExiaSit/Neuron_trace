import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from algorithms.mask_repair.image_evidence_cache import cached_arrays
from algorithms.mask_repair.multiscale_mip import configuration,multiscale_seeds_cpu


class EvidenceCacheTests(unittest.TestCase):
    def test_reuse_is_exact_and_content_or_config_changes_invalidate(self):
        a=np.arange(60,dtype=np.float32).reshape(3,4,5);calls=[]
        def compute():
            calls.append(1);return a.copy()*2,dict(count=len(a))
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,RECONNECT_WORK=d,RECONNECT_EVIDENCE_CACHE='1'):
            first=cached_arrays('test',[a],{'mode':'one'},compute)
            second=cached_arrays('test',[a.copy()],{'mode':'one'},compute)
            self.assertEqual(len(calls),1);np.testing.assert_array_equal(first[0],second[0]);self.assertEqual(first[1],second[1])
            a[0,0,0]=23
            changed=cached_arrays('test',[a],{'mode':'one'},compute)
            self.assertEqual(len(calls),2);self.assertEqual(changed[0][0,0,0],46)
            cached_arrays('test',[a],{'mode':'two'},compute);self.assertEqual(len(calls),3)
            cached_arrays('test',[a.reshape(5,4,3)],{'mode':'two'},compute);self.assertEqual(len(calls),4)
    def test_incomplete_generation_is_recomputed(self):
        a=np.ones((3,4,5),np.float32);calls=[]
        def compute():calls.append(1);return a.copy(),{}
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,RECONNECT_WORK=d,RECONNECT_EVIDENCE_CACHE='1'):
            cached_arrays('test',[a],{},compute)
            next(Path(d).rglob('0.npy')).write_bytes(b'truncated')
            result=cached_arrays('test',[a],{},compute)
            np.testing.assert_array_equal(a,result[0]);self.assertEqual(len(calls),2)
    def test_without_case_workspace_nothing_is_cached(self):
        calls=[]
        def compute():calls.append(1);return np.ones(2),{}
        with patch.dict(os.environ,clear=True):
            cached_arrays('test',[np.ones(2)],{},compute);cached_arrays('test',[np.ones(2)],{},compute)
        self.assertEqual(len(calls),2)


class CudaSlabParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if not cupy.cuda.runtime.getDeviceCount():raise RuntimeError('No CUDA device')
        except Exception as e:raise unittest.SkipTest(str(e))
    def check(self,raw,contrast,cfg):
        from algorithms.mask_repair.multiscale_mip_cuda import seeds_cuda
        expected,a=multiscale_seeds_cpu(raw,contrast,cfg);actual,b=seeds_cuda(raw,contrast,cfg)
        np.testing.assert_array_equal(actual,expected);self.assertEqual(a,b)
    def test_random_ties_edges_and_noncontiguous_axes(self):
        rng=np.random.default_rng(449)
        for dtype in [np.uint8,np.uint16]:
            raw=rng.integers(0,8,(23,29,31),dtype=dtype)
            contrast=rng.choice(np.array([0,2,np.nextafter(np.float32(2),np.float32(3)),8],np.float32),raw.shape)
            for axes in [(0,1,2),(2,0,1)]:
                self.check(raw.transpose(axes),contrast.transpose(axes),configuration(dict(thicknesses=[4,8,16],overlap=.4)))
    def test_flat_and_single_voxel_axes(self):
        for shape in [(1,9,11),(13,1,7),(8,9,1)]:
            raw=np.full(shape,255,np.uint8)
            self.check(raw,raw.astype('float32'),configuration())
    def test_three_view_consensus(self):
        rng=np.random.default_rng(23);raw=rng.integers(0,65535,(20,17,19),dtype=np.uint16)
        self.check(raw,raw.astype('float64')*.001,configuration(dict(min_axes=3,thicknesses=[6,12])))


class SlabDispatchTests(unittest.TestCase):
    def test_cuda_oom_retries_exact_cpu_computation(self):
        from types import SimpleNamespace
        from algorithms.mask_repair import multiscale_mip as mip
        class FakeOOM(Exception):pass
        backend=SimpleNamespace(gpu_disabled=False,stats={'minimum_gpu_voxels':1},
            cp=SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(OutOfMemoryError=FakeOOM))))
        raw=np.arange(9*11*13,dtype=np.uint16).reshape(9,11,13)
        contrast=raw.astype('float32');cfg=configuration()
        expected,report=multiscale_seeds_cpu(raw,contrast,cfg)
        with patch.dict(os.environ,RECONNECT_COMPUTE_BACKEND='cuda',RECONNECT_MIP_CUDA='1'),patch.object(mip,'ndi',backend),patch('algorithms.mask_repair.multiscale_mip_cuda.seeds_cuda',side_effect=FakeOOM):
            actual,actual_report=mip.multiscale_seeds(raw,contrast,cfg)
        np.testing.assert_array_equal(actual,expected);self.assertEqual(actual_report,report)
        self.assertTrue(backend.gpu_disabled)
        self.assertEqual(backend.stats['cpu_fallbacks'][0]['reason'],'cuda_out_of_memory')


class PairStreamingTests(unittest.TestCase):
    def test_order_and_radius_boundary_match_full_pair_set(self):
        from scipy.spatial import cKDTree
        from algorithms.mask_repair.endpoint_pairs import cross_component_pairs
        rng=np.random.default_rng(770)
        points=rng.integers(0,30,(170,3)).astype(float);points[:3]=[[0,0,0],[0,0,10],[0,6,8]]
        components=rng.integers(0,8,len(points))
        for radius in [0,10,70]:
            expected=[(i,j) for i,j in sorted(cKDTree(points).query_pairs(radius)) if components[i]!=components[j]]
            self.assertEqual(list(cross_component_pairs(points,components,radius,chunk_size=9)),expected)
    def test_single_component_and_empty_inputs(self):
        from algorithms.mask_repair.endpoint_pairs import cross_component_pairs
        self.assertEqual(list(cross_component_pairs([],[],70)),[])
        self.assertEqual(list(cross_component_pairs(np.zeros((100,3)),np.ones(100),70)),[])

class ProjectionCacheTests(unittest.TestCase):
    def test_projections_exact_and_file_changes_invalidate(self):
        import tifffile
        from visualization.diagnostics import _raw_projections
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.tif';a=np.arange(9*11*13,dtype=np.uint16).reshape(9,11,13)
            tifffile.imwrite(p,a,photometric='minisblack');st=p.stat()
            first=_raw_projections(str(p),st.st_size,st.st_mtime_ns)
            self.assertIs(first,_raw_projections(str(p),st.st_size,st.st_mtime_ns))
            for axis in range(3):np.testing.assert_array_equal(first.max(axis),a.max(axis))
            tifffile.imwrite(p,a+1,photometric='minisblack');os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns+1000000));st=p.stat()
            second=_raw_projections(str(p),st.st_size,st.st_mtime_ns)
            for axis in range(3):np.testing.assert_array_equal(second.max(axis),(a+1).max(axis))

if __name__=='__main__':unittest.main()
