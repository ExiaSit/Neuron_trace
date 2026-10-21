"""Numerical and cohort rules for the reusable version comparison."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from reconstruction_comparison.metrics import read_swc,swc_total_length
from reconstruction_comparison.report import structural,summarize,paired_deltas,write_csv,validate_topology
spec=importlib.util.spec_from_file_location('comparison_driver',ROOT/'tools/compare_reconstructions.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class ComparisonMethods(unittest.TestCase):
    def test_length_is_physical_and_independent_of_node_density(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'tree.swc'
            p.write_text('1 1 0 0 0 1 -1\n2 3 3 4 0 1 1\n3 3 3 4 12 1 2\n')
            self.assertEqual(swc_total_length(read_swc(p)),(17.,3,2,0))
            p.write_text('\n'.join(f'{i+1} 3 {i} 0 0 1 {i if i else -1}' for i in range(18)))
            self.assertEqual(swc_total_length(read_swc(p))[0],17.)

    def test_branch_metric_counts_tips_and_branchpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'tree.swc'
            p.write_text('1 1 0 0 0 1 -1\n2 3 1 0 0 1 1\n3 3 2 0 0 1 2\n4 3 3 1 0 1 3\n5 3 3 -1 0 1 3\n6 3 4 1 0 1 4\n')
            x=structural(read_swc(p),(10,10,10))
            self.assertEqual(x['tips'],2);self.assertEqual(x['branchpoints'],1);self.assertEqual(x['branches'],3)

    def rows(self):
        rows=[]
        for cid,key,length,error in [('1','old',10.,''),('2','old',100.,''),('1','new',20.,''),('2','new',0.,'missing')]:
            rows.append(dict(id=cid,version_key=key,error=error,length=length,nodes=3,branches=1,precision=.8,recall=.5,mean_snr=4.))
        return rows

    def test_shared_cohort_excludes_failure_from_every_version(self):
        result,common=summarize(self.rows(),[dict(key='old',label='old'),dict(key='new',label='new')])
        self.assertEqual(common,['1'])
        paired=[r for r in result if r['scope']=='paired_all_versions']
        self.assertEqual([r['count'] for r in paired],[1,1]);self.assertEqual([r['mean_length'] for r in paired],[10.,20.])
        self.assertEqual(result[0]['mean_length'],55.)

    def test_paired_bootstrap_is_cell_based_and_reproducible(self):
        versions=[dict(key='old',label='old'),dict(key='new',label='new')]
        x=paired_deltas(self.rows(),versions,['1']);y=paired_deltas(self.rows(),versions,['1'])
        self.assertEqual(x,y);r=x[0]
        self.assertEqual(r['metric'],'length');self.assertEqual(r['mean_delta'],10.)
        self.assertEqual(r['bootstrap_95_low'],10.);self.assertEqual(r['count'],1)

    def test_crop_conversion_preserves_origin_and_accepts_historical_padding(self):
        data={'bounds_1um_zxy':[0,3,10,17,20,25]}
        result,order=driver.normalize_crop(data,(3,5,7))
        self.assertEqual(order,'ZXY');self.assertEqual(result['bounds_1um_zxy'],[0,3,20,25,10,17])
        padded,_=driver.normalize_crop(data,(4,6,8))
        self.assertEqual(padded['bounds_1um_zxy'],result['bounds_1um_zxy'])
        self.assertNotIn('bounds_order',data)

    def test_missing_swc_raises_and_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):read_swc(Path(folder)/'missing.swc')

    def test_invalid_topology_is_rejected_before_legacy_traversal(self):
        from reconstruction_comparison.metrics import SwcData
        with self.assertRaisesRegex(ValueError,'Missing parent'):
            validate_topology(SwcData(nodes={1:(0,0,0,1)},parents={1:2}))
        with self.assertRaisesRegex(ValueError,'cycle'):
            validate_topology(SwcData(nodes={1:(0,0,0,1),2:(1,0,0,1)},parents={1:2,2:1}))

    def test_failure_csv_can_include_partial_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'failed.csv'
            write_csv(p,[dict(id='1',error='missing',length=12.)],fields=['id','error'])
            self.assertIn('missing',p.read_text());self.assertNotIn('length',p.read_text())

    def test_pr_algorithm_matches_registered_historical_source(self):
        import hashlib
        manifest=json.loads((ROOT/'comparison/metric_provenance.json').read_text())
        path=ROOT/'tools/reconstruction_comparison/legacy_prec_rec.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest['legacy_prec_rec_sha256'])


if __name__=='__main__':unittest.main()
