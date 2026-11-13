import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
from algorithms.tracing.soma_seed_integration import validate_context, matching_audit


class IntegrationTests(unittest.TestCase):
    def context(self):
        return dict(target_zyx=[10,10,10],centers_zyx=[[10,10,10],[10,10,12]],
                    radii=[None,4.2],seed_ids=['T','C001'])

    def test_target_and_units_validation(self):
        c=self.context()
        centers,radii,ids=validate_context(c,(30,30,30),[10,10,10])
        self.assertEqual(len(centers),2)
        self.assertEqual(radii[(10,10,12)],4.2)
        c['centers_zyx'][0]=[10,10,9]
        with self.assertRaises(ValueError):validate_context(c,(30,30,30),[10,10,10])

    def test_duplicate_ids_and_out_of_bounds_fail(self):
        c=self.context();c['seed_ids']=['T','T']
        with self.assertRaises(ValueError):validate_context(c,(30,30,30),[10,10,10])
        c=self.context();c['centers_zyx'][1]=[30,0,0]
        with self.assertRaises(ValueError):validate_context(c,(30,30,30),[10,10,10])

    def test_matches_collisions_and_unmatched_are_explicit(self):
        tree=[[1,1,10,10,10,1,-1],[2,3,11,10,10,1,1]]
        centers=[[10,10,10],[10,10,10.1],[25,25,25]]
        audit=matching_audit(tree,centers,{tuple(c):4 for c in centers},['T','A','B'])
        self.assertEqual([r['status'] for r in audit['records']],['matched','same_app2_node','unmatched'])
        self.assertEqual(audit['unique_matched_nodes'],[1])

    def test_supplied_seeds_bypass_nms_and_merge(self):
        from utils.case_worker import load_module
        root=Path(__file__).resolve().parents[1]
        cfg=SimpleNamespace(BASE_DIR=root,PATHS={'gcut':str(root/'vendor/gcut')},GCUT={})
        with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'pipeline_config':cfg}):
            folder=Path(td);raw=np.zeros((30,30,30),dtype=np.uint8);raw[5:20,5:20,5:20]=100
            tifffile.imwrite(folder/'raw.tif',raw);tifffile.imwrite(folder/'mask.tif',np.ones_like(raw))
            mod=load_module('gcut')
            with patch.object(mod,'apply_nms',side_effect=AssertionError('NMS called')), \
                 patch('algorithms.tracing.soma_signal_groups.filter_seeds',side_effect=AssertionError('merge called')), \
                 patch.object(mod,'_finish_gcut',return_value='passed') as finish:
                self.assertEqual(mod.process_image_to_gcut(folder/'raw.tif',folder/'mask.tif',None,folder,
                     [10,10,10],soma_seed_override=self.context()),'passed')
                self.assertEqual(finish.call_args.args[5],self.context()['centers_zyx'])
                self.assertEqual(finish.call_args.kwargs['seed_ids'],['T','C001'])
            saved=json.loads((folder/'soma-context.json').read_text())
            self.assertEqual(saved['radii'],[None,4.2])


if __name__=='__main__':unittest.main()
