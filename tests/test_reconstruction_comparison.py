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
from reconstruction_comparison.report import branch_lengths,structural,summarize,paired_deltas,write_csv,validate_topology
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
            rows.append(dict(id=cid,version_key=key,error=error,length=length,nodes=3,branches=1,mean_branch_length=length/2,precision=.8,recall=.5,mean_snr=4.))
        return rows

    def test_branch_paths_use_3d_arc_length_and_partition_every_edge(self):
        from reconstruction_comparison.metrics import SwcData
        # A bent 5+12 um trunk, then two terminal paths of 5 and 6 um.
        swc=SwcData(nodes={1:(0,0,0,1),2:(3,4,0,1),3:(3,4,12,1),
                           4:(6,8,12,1),5:(3,4,18,1)},parents={1:-1,2:1,3:2,4:3,5:3})
        self.assertEqual(branch_lengths(swc),[17.,5.,6.])
        result=structural(swc,(30,30,30))
        self.assertEqual(result['branch_segments'],3)
        self.assertAlmostEqual(result['mean_branch_length'],28/3)
        self.assertEqual(sum(branch_lengths(swc)),swc_total_length(swc)[0])

    def test_branch_length_does_not_depend_on_sampling_density(self):
        from reconstruction_comparison.metrics import SwcData
        a=SwcData(nodes={1:(0,0,0,1),2:(3,4,0,1)},parents={1:-1,2:1})
        b=SwcData(nodes={1:(0,0,0,1),2:(1.5,2,0,1),3:(3,4,0,1)},parents={1:-1,2:1,3:2})
        self.assertEqual(branch_lengths(a),branch_lengths(b))

    def test_root_forks_forests_and_single_nodes(self):
        from reconstruction_comparison.metrics import SwcData
        swc=SwcData(nodes={1:(0,0,0,1),2:(1,0,0,1),3:(0,2,0,1),4:(5,0,0,1)},
                    parents={1:-1,2:1,3:1,4:-1})
        result=structural(swc,(10,10,10))
        self.assertEqual(result['branches'],4)  # Historical count includes root fork and isolated tip.
        self.assertEqual(result['branch_segments'],2)
        self.assertEqual(result['mean_branch_length'],1.5)
        single=SwcData(nodes={1:(0,0,0,1)},parents={1:-1})
        self.assertEqual(branch_lengths(single),[])
        self.assertEqual(structural(single,(1,1,1))['mean_branch_length'],0.)

    def test_branch_summary_weights_cells_equally(self):
        rows=self.rows();rows[1]['mean_branch_length']=100.
        summary,_=summarize(rows,[dict(key='old',label='old'),dict(key='new',label='new')])
        self.assertEqual(summary[0]['mean_branch_length'],52.5)
        self.assertEqual(summary[2]['mean_branch_length'],5.)

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

    def test_archived_app2_import_preserves_tree_and_never_moves_seed(self):
        from reconstruction_comparison.archived_app2 import convert
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'native.swc';destination=Path(folder)/'image.swc'
            source.write_text('# native\n0 1 3 4 5 2 -1\n2 3 6 7 8 1 0\n9 1 2 3 4 5 -1\n')
            before=source.read_bytes();a=read_swc(source)
            convert(source,destination,20);b=read_swc(destination)
            self.assertEqual(a.parents,b.parents)
            self.assertEqual(b.nodes[0],(3.,15.,5.,2.))
            self.assertEqual(b.nodes[9],(2.,16.,4.,5.))
            self.assertEqual(swc_total_length(a),swc_total_length(b))
            self.assertEqual(source.read_bytes(),before)
            convert(source,destination,20)  # Matching import is resumable.

    def test_archived_app2_import_rejects_double_conversion_and_overwrite(self):
        from reconstruction_comparison.archived_app2 import convert
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'native.swc';destination=Path(folder)/'image.swc'
            source.write_text('1 1 3 4 5 2 -1\n')
            with self.assertRaisesRegex(ValueError,'overwritten'):convert(source,source,20)
            convert(source,destination,20)
            with self.assertRaisesRegex(ValueError,'another Y flip'):convert(destination,Path(folder)/'twice.swc',20)
            with self.assertRaisesRegex(ValueError,'Conflicting'):convert(source,destination,21)

    def test_archived_app2_mode_disables_retracing(self):
        self.assertEqual(driver.stages({'app2_input_mode':'archived_native'}),('import_app2','gcut','gcut_resample','prune'))
        self.assertEqual(driver.stages({}),driver.STAGES)
        with self.assertRaisesRegex(ValueError,'Unknown'):driver.stages({'app2_input_mode':'typo'})

    def test_archived_app2_worker_checks_source_hash_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'manifests').mkdir()
            source=root/'native.swc';source.write_text('1 1 3 4 5 2 -1\n')
            record=dict(inputs={'archived_app2':str(source)},input_sha256={'archived_app2':driver.sha(source)},shape=[10,20,30])
            (root/'manifests/1.json').write_text(json.dumps(record))
            cfg=dict(output_root=str(root),app2_input_mode='archived_native')
            driver.stage_worker(cfg,'1','import_app2')
            self.assertTrue((root/'reconstruction/trace_app2/down_sampled_swcs_app2/image_1.swc').is_file())
            source.write_text('1 1 9 4 5 2 -1\n')
            with self.assertRaisesRegex(ValueError,'changed since preflight'):driver.stage_worker(cfg,'1','import_app2')
            with self.assertRaisesRegex(ValueError,'disabled'):driver.stage_worker(cfg,'1','app2_trace')

    def test_reuse_requires_identical_processing_parameters(self):
        from reconstruction_comparison.reuse import processing_config
        a=dict(output_root='a',neuron_ids=[1],metadata_file='a.csv',gcut={'threshold':2})
        b=dict(output_root='b',neuron_ids=[1,2],metadata_file='b.csv',gcut={'threshold':2})
        self.assertEqual(processing_config(a),processing_config(b))
        b['gcut']['threshold']=3
        self.assertNotEqual(processing_config(a),processing_config(b))

    def test_cohort_extension_preserves_failed_case_and_copy_integrity(self):
        from reconstruction_comparison.reuse import adopt
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'old';out=Path(folder)/'new';source.mkdir();out.mkdir()
            for root,fingerprint in [(source,'old'),(out,'new')]:
                driver.save(root/'run_manifest.json',dict(fingerprint=fingerprint,code={'test.py':'digest'},stages=['import_app2','gcut','gcut_resample','prune']))
                driver.save(root/'pipeline_config.json',dict(output_root=str(root),neuron_ids=[1],metadata_file=str(root/'meta.csv'),gcut={'value':1}))
            driver.save(source/'validation.json',dict(validation_passed=True))
            rec=dict(id='1',input_sha256={'archived_app2':'pinned'},metadata={'soma_x':3},shape=[10,20,30],crop_for_v081={})
            driver.save(source/'manifests/1.json',rec)
            driver.save(source/'status/1.json',dict(id='1',fingerprint='old',state='failed',current_stage='prune',completed_stages=['import_app2','gcut','gcut_resample'],seconds={}))
            (source/'logs/1').mkdir(parents=True);(source/'logs/1/prune.log').write_text('source failure\n')
            paths=['trace_app2/down_sampled_swcs_app2/image_1.swc','gcut_output/selected_for_pruning/image_1.swc','gcut_output/selected_resampled_step1/image_1.swc']
            for relative in paths:
                p=source/'reconstruction'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('1 1 3 4 5 2 -1\n')
            p=source/'reconstruction'/paths[0]
            driver.save(source/'metrics/1.json',[dict(version_key='app2_current',error='',swc_path=str(p),swc_sha256=driver.sha(p))])
            cfg=dict(reuse_from=str(source),output_root=str(out),app2_input_mode='archived_native',workers=1)
            adopt(cfg,[rec]);adopt(cfg,[rec])  # Resuming must keep the reuse record.
            self.assertEqual(json.loads((out/'reuse_manifest.json').read_text())['reused_count'],1)
            for relative in paths:self.assertEqual((source/'reconstruction'/relative).read_bytes(),(out/'reconstruction'/relative).read_bytes())
            result=driver.reconstruct(cfg,[rec])
            self.assertEqual(result[0]['state'],'failed')
            self.assertEqual(result[0]['reused_from'],str(source))


if __name__=='__main__':unittest.main()
