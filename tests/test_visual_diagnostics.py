"""Diagnostics must explain actual 3-D changes and preserve reference coordinates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
from utils import bootstrap
from visualization.diagnostics import mask_delta, mask_transition, record_mask_step, write_short_leaf_audit
from visualization.previous_app2 import resolve_previous_app2, reference_nodes
from web_console.server import artifacts
from algorithms.tracing.gcut_adapter import make_segmentor


class DiagnosticTests(unittest.TestCase):
    def test_three_dimensional_diff_before_projection(self):
        before=np.zeros((3,5,5),bool);before[0,2,2]=True
        after=before.copy();after[1,2,2]=True
        self.assertTrue(np.array_equal(before.any(0),after.any(0)))
        added,removed=mask_delta(before,after)
        self.assertEqual(int(added.sum()),1)
        self.assertEqual(int(removed.sum()),0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'42.png'
            stats=mask_transition(np.ones(before.shape,dtype=np.uint8),before,after,p,'overlap')
            self.assertEqual(stats['added_voxels'],1)
            self.assertGreater(p.stat().st_size,1000)
            self.assertEqual(json.loads(p.with_suffix('.json').read_text())['added_voxels'],1)

    def test_removal_and_shape_error(self):
        added,removed=mask_delta(np.ones((2,2,2)),np.zeros((2,2,2)))
        self.assertEqual(int(removed.sum()),8)
        self.assertFalse(added.any())
        with self.assertRaises(ValueError):mask_delta(np.zeros((2,2,2)),np.zeros((3,2,2)))

    def test_exact_id_priority_and_conversion_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'first';b=root/'second';a.mkdir();b.mkdir()
            (a/'image_116984.swc').write_text('1 1 2 3 4 1 -1\n')
            source=b/'image_16984.swc';source.write_text('1 1 2 3 4 1 -1\n')
            original=source.read_bytes()
            spec=dict(id='16984',config=dict(previous_app2_dirs=[str(a),str(b)]))
            self.assertEqual(resolve_previous_app2(spec),source)
            nodes,audit=reference_nodes(spec,10)
            self.assertEqual(nodes[1][0].tolist(),[2,6,4])
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(audit['transform'],'y = height - 1 - y')
            first=a/'16984.swc';first.write_text('# app2_native_to_image_zero_based_v2\n1 1 2 3 4 1 -1\n')
            nodes,audit=reference_nodes(spec,10)
            self.assertEqual(nodes[1][0].tolist(),[2,3,4])
            self.assertEqual(audit['path'],str(first))
            spec['id']='42'
            self.assertEqual(reference_nodes(spec,10)[1]['status'],'missing')

    def test_audit_excludes_soma_contraction_and_keeps_parent_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);swc=root/'in.swc';somas=root/'somas.txt'
            swc.write_text('1 1 0 0 0 1 -1\n2 3 .1 0 0 .01 1\n3 3 10 0 0 .01 2\n4 3 20 10 0 .01 3\n5 3 10 1 0 .01 3\n6 3 10 2 0 .01 5\n')
            somas.write_text('1\n')
            segmentor=make_segmentor(str(swc),str(somas),min_leaf_length_um=7.75);audit=segmentor.short_leaf_audit
            self.assertNotIn(2,{n[0] for n in audit['before_nodes']})
            self.assertNotIn(2,audit['removed_ids'])
            self.assertEqual(audit['removed_ids'],[5,6])
            self.assertIn(3,{n[0] for n in audit['before_nodes']})
            write_short_leaf_audit(audit,np.ones((3,25,25),dtype=np.uint8),root,'42')
            self.assertEqual(json.loads((root/'gcut_audits/42_short_leaves.json').read_text())['removed_ids'],[5,6])
            names={a['name'] for a in artifacts(root,'42')}
            self.assertIn('42_short_leaves.png',names)
            self.assertIn('42_short_leaves_before.swc',names)

    def test_pruning_summary_artifact_is_cell_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for cid in ['42','420']:
                p=root/'logs'/cid/'pruning-audit.json'
                p.parent.mkdir(parents=True)
                p.write_text('{"removal_counts": {"qc_merge_close_point": 2}}')
            entries=artifacts(root,'42')
            self.assertEqual([a['path'] for a in entries],['logs/42/pruning-audit.json'])
            self.assertEqual(entries[0]['name'],'42_pruning-audit.json')
            self.assertFalse(entries[0]['image'])
            with self.assertRaises(ValueError):artifacts(root,'../../other')

    def test_mask_step_uses_provided_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw.tif';a=root/'a.tif';b=root/'b.tif'
            tifffile.imwrite(raw,np.zeros((3,5,5),dtype=np.uint8))
            tifffile.imwrite(a,np.ones((3,5,5),dtype=np.uint8))
            tifffile.imwrite(b,np.ones((3,5,5),dtype=np.uint8)*255)
            spec=dict(id='42',raw=str(raw),config=dict(output_root=tmp))
            with patch('visualization.diagnostics.mask_transition') as render:
                record_mask_step(spec,1,a,b)
            self.assertTrue((render.call_args.args[1]==1).all())
            self.assertTrue((render.call_args.args[2]==255).all())
            self.assertEqual(render.call_args.args[3].name,'42_02_b.png')


if __name__=='__main__':unittest.main()
