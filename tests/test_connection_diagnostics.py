import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi
from utils import bootstrap
from algorithms.tracing import connection_diagnostics as cd


def image_of_paths(paths,shape=(64,64,80)):
    image=np.zeros(shape,np.float32)
    for path in paths:
        p=cd.resample_line(path,.25)[:,::-1]
        image[tuple(np.rint(p).astype(int).T)]=1
    image=ndi.gaussian_filter(image,.6)
    return np.rint(3+180*image/max(float(image.max()),1e-8)).astype(np.uint8)


def branching_tree():
    return [[1,1,4,32,32,1,-1], [2,3,20,32,32,1,1],
            [3,3,40,32,32,1,2],[4,3,20,52,32,1,2],
            [5,3,60,32,32,1,3]]


class ConnectionTests(unittest.TestCase):
    def run_diagnosis(self,tree,raw):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        out=Path(tmp.name)
        report=cd.diagnose_connections(tree,raw,out,np.asarray(tree[0][2:5]))
        self.assertTrue((out/'diagnosis.json').is_file())
        for site in report['sites']:
            if 'image' in site:self.assertTrue((out/site['image']).is_file())
        return report,out

    def test_real_branch_preserved(self):
        tree=branching_tree()
        raw=image_of_paths([[[4,32,32],[60,32,32]],[[20,32,32],[20,52,32]]])
        report,_=self.run_diagnosis(tree,raw)
        self.assertEqual(report['proposed_cut_ids'],[])
        self.assertEqual(cd.apply_connection_cuts(tree,report)[0],tree)
        self.assertEqual(report['sites'][0]['status'],'supported')

    def test_confirmed_gap_cuts_only_its_downstream_branch(self):
        tree=branching_tree()
        raw=image_of_paths([[[4,32,32],[60,32,32]],[[20,32,32],[20,52,32]]])
        raw[:,:,23:30]=3
        report,_=self.run_diagnosis(tree,raw)
        self.assertIn(3,report['proposed_cut_ids'])
        kept,audit=cd.apply_connection_cuts(tree,report)
        self.assertEqual({n[0] for n in kept},{1,2,4})
        self.assertEqual(audit['removed_ids'],[3,5])

    def test_separated_projection_crossing_does_not_delete_paths(self):
        tree=[[1,1,8,8,8,1,-1],[2,3,10,30,20,1,1],[3,3,60,30,20,1,2],
              [4,3,35,10,42,1,1],[5,3,35,55,42,1,4]]
        raw=image_of_paths([[[8,8,8],[10,30,20],[60,30,20]],[[8,8,8],[35,10,42],[35,55,42]]])
        report,_=self.run_diagnosis(tree,raw)
        crossings=[s for s in report['sites'] if s['kind']=='crossing']
        self.assertTrue(crossings)
        self.assertTrue(any(s['separation_um']>20 and 'XY' in s['overlap_planes'] for s in crossings))
        self.assertEqual(report['proposed_cut_ids'],[])

    def test_no_signal_is_uncertain_not_a_cut(self):
        report,_=self.run_diagnosis(branching_tree(),np.zeros((64,64,80),np.uint8))
        self.assertEqual(report['proposed_cut_ids'],[])
        self.assertEqual(report['sites'][0]['status'],'uncertain')

    def test_shallow_angle_crossing_is_not_excluded(self):
        tree=[[1,1,0,0,5,1,-1],[2,3,5,30,20,1,1],[3,3,70,30,20,1,2],
              [4,3,5,19,40,1,1],[5,3,70,41,40,1,4]]
        sites=cd.candidate_sites(tree)
        self.assertTrue(any(s['kind']=='crossing' and 'XY' in s['overlap_planes']
                            and 19<s['separation_um']<21 for s in sites))

    def test_self_crossing_is_diagnosed(self):
        tree=[[1,1,5,10,20,1,-1],[2,3,60,50,20,1,1],
              [3,3,5,50,20,1,2],[4,3,60,10,20,1,3]]
        self.assertTrue(any(s['kind']=='crossing' and 'XY' in s['overlap_planes']
                            for s in cd.candidate_sites(tree)))

    def test_mips_are_cropped_in_all_three_dimensions(self):
        tree=branching_tree();raw=np.zeros((64,64,80),np.uint8)
        raw[60,:,:]=255  # Bright sheet outside local box must not enter XY MIP.
        report,_=self.run_diagnosis(tree,raw)
        site=report['sites'][0]
        self.assertLess(site['bbox_xyz'][2][1],60)
        self.assertEqual(site['arms'][0]['diagnosis']['anchor_signal'],0)

    def test_saved_diagnosis_precedes_cuts_and_rejects_changed_tree(self):
        tree=branching_tree();raw=np.zeros((64,64,80),np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            original=cd.apply_connection_cuts
            def checked(rows,report):
                self.assertTrue((Path(tmp)/'diagnosis.json').exists())
                self.assertTrue((Path(tmp)/'index.html').exists())
                return original(rows,report)
            with patch.object(cd,'apply_connection_cuts',side_effect=checked):
                cd.diagnose_then_cut(tree,raw,tmp,np.array(tree[0][2:5]))
            report=json.loads((Path(tmp)/'diagnosis.json').read_text())
            with self.assertRaises(ValueError):original(tree[:-1],report)

    def test_dim_but_continuous_fiber_is_not_a_gap(self):
        raw=image_of_paths([[[4,32,32],[60,32,32]]])
        raw[:,:,23:30]=3+((raw[:,:,23:30].astype(float)-3)*.12).astype(np.uint8)
        d=cd.diagnose_path(raw,[0,0,0],np.array([[20,32,32],[40,32,32]]),cd.DEFAULTS)
        self.assertNotEqual(d['status'],'disconnected')

    def test_gap_hidden_by_out_of_depth_bright_fibers_is_detected(self):
        raw=image_of_paths([[[4,32,32],[60,32,32]]]);raw[:,:,23:30]=3
        raw[45,32,:]=240;raw[32,45,:]=240
        d=cd.diagnose_path(raw,[0,0,0],np.array([[20,32,32],[40,32,32]]),cd.DEFAULTS)
        self.assertEqual(d['status'],'disconnected')
        self.assertGreaterEqual(d['selected_gap']['resolving_views'],2)

    def test_bend_needs_independent_shaft_and_two_thin_views(self):
        center=[32,32,32]
        tree=[[1,1,8,32,8,1,-1],[2,3,*center,1,1],[3,3,22,42,42,1,2]]
        site=next(s for s in cd.extended_sites(tree,np.array([8,32,8]),cd.DEFAULTS)
                  if s['kind']=='crossing_switch')
        paths=[[[8,32,8],center,[22,42,42]]]
        d=cd.diagnose_switch(image_of_paths(paths),np.zeros(3),site,cd.DEFAULTS)
        self.assertEqual(d['status'],'uncertain')
        paths.append([[42,22,22],[22,42,42]])
        d=cd.diagnose_switch(image_of_paths(paths),np.zeros(3),site,cd.DEFAULTS)
        self.assertEqual(d['status'],'suspected_switch')
        self.assertEqual(d['evidence_type'],'ownership_heuristic_not_proven_gap')
        shifted=ndi.shift(image_of_paths(paths),[1,0,0],order=0,mode='nearest')
        shifted_d=cd.diagnose_switch(shifted,np.zeros(3),site,cd.DEFAULTS)
        self.assertEqual(shifted_d['status'],'suspected_switch')
        self.assertEqual(shifted_d['probes']['opposite']['centering_tolerance_um'],0.)
        # Extending the original incoming fiber makes a real crossing ambiguous.
        paths.append([center,[48,32,48]])
        d=cd.diagnose_switch(image_of_paths(paths),np.zeros(3),site,cd.DEFAULTS)
        self.assertEqual(d['status'],'uncertain')

    def test_topological_branch_is_not_a_switch_candidate(self):
        tree=[[1,1,8,32,8,1,-1],[2,3,32,32,32,1,1],
              [3,3,22,42,42,1,2],[4,3,42,22,22,1,2]]
        self.assertFalse(any(s['kind']=='crossing_switch'
                             for s in cd.extended_sites(tree,np.array([8,32,8]),cd.DEFAULTS)))

    def test_long_branch_windows_cover_far_end(self):
        tree=[[1,1,4,32,32,1,-1],[2,3,76,32,32,1,1]]
        sites=cd.extended_sites(tree,np.array([4,32,32]),cd.DEFAULTS)
        self.assertTrue(any(np.allclose(s['arms'][0]['xyz'][-1],[76,32,32]) for s in sites))


if __name__=='__main__':unittest.main()
