"""Regression coverage for v0.84's restored base pruning pipeline."""
import json
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile
from utils import bootstrap
import sys
from types import SimpleNamespace
from utils.case_worker import load_module

_previous_config = sys.modules.get('pipeline_config')
sys.modules['pipeline_config'] = SimpleNamespace(BASE_DIR=Path('.'), PATHS={})
try:
    prune = load_module('prune')
finally:
    if _previous_config is None:
        sys.modules.pop('pipeline_config', None)
    else:
        sys.modules['pipeline_config'] = _previous_config


class FourRulePruningTests(unittest.TestCase):
    def run_pipeline(self, tree, raw=None, mask=None, soma=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        shape = (48, 64, 96)
        if raw is None:
            raw = np.zeros(shape, np.uint8)
        if mask is None:
            mask = np.zeros(shape, np.uint8)  # Disconnection must not prune.
        if soma is None:
            soma = np.zeros(shape, np.uint8)
            soma[24, 32, 12] = 1
        for name, data in [('42', mask), ('raw', raw), ('soma', soma)]:
            tifffile.imwrite(root/f'{name}.tif', data)
        (root/'soma.json').write_text(json.dumps(dict(bounds_1um_zxy=[0,48,0,96,0,64])))
        (root/'42.swc').write_text(''.join(' '.join(map(str, row))+'\n' for row in tree))
        (root/'out').mkdir(); (root/'mips').mkdir()
        meta = pd.DataFrame([dict(soma_x=12, soma_y=32, soma_z=24,
                                 xy_resolution=1000, z_resolution=1000)], index=[42])
        result, _ = prune.process_neuron_morphology(
            str(root/'42.tif'), str(root/'42.swc'), str(root/'raw.tif'),
            str(root/'out'), str(root/'mips'), meta, sphere_zradius=10,
            show_steps=False, soma_json_path=str(root/'soma.json'),
            soma_mask_path=str(root/'soma.tif'))
        audit = json.loads((root/'mips/42_pruning_reasons.json').read_text())
        # One root, no dangling parents or cycles after all cleanup stages.
        nodes = {int(n[0]): n for n in result}
        self.assertEqual(sum(int(n[6]) == -1 for n in result), 1)
        for nid in nodes:
            seen = set()
            while nid != -1:
                self.assertNotIn(nid, seen)
                seen.add(nid)
                self.assertIn(nid, nodes)
                nid = int(nodes[nid][6])
        self.assertEqual(audit['active_rules'], ['initial_short_terminal', 'local_connection_gap', 'remove_branches_entering_neighbor_soma', 'soma_cleanup',
                                               'qc_merge_close_point', 'qc_remove_zigzag'])
        return result, audit

    def test_only_foreign_soma_rule_cuts_dark_thick_disconnected_branch(self):
        tree = [[i+1, 1 if i == 0 else 3, x, 32, 24, 20, -1 if i == 0 else i]
                for i, x in enumerate(range(12, 81))]
        raw = np.zeros((48, 64, 96), np.uint8)
        raw[:, :, :20] = 180
        raw[:, :, 70:] = 250  # Deep valley and bright rebound.
        soma = np.zeros_like(raw)
        soma[24, 32, 12] = 1
        soma[22:27, 30:35, 70:80] = 1  # Separate foreign soma.
        result, audit = self.run_pipeline(tree, raw=raw, soma=soma)
        self.assertEqual({n[0] for n in result}, set(range(1,59)))
        self.assertEqual(audit['neighbor_soma_audit']['removed_ids'], list(range(59,70)))
        self.assertEqual({d['code'] for d in audit['removed_nodes'].values()},
                         {'remove_branches_entering_neighbor_soma'})
        for key in ['signal_gap_decisions', 'crossing_decisions', 'sheet_signal_decisions']:
            self.assertEqual(audit[key], [])
        self.assertEqual(audit['repair_evidence_audit']['status'], 'disabled')

    def test_soma_interior_cleanup_still_removes_internal_twig(self):
        tree = [[1, 1, 12,32,24,1,-1], [2,3,13,32,28,1,1],
                [3,3,14,32,24,1,2], [4,3,30,32,24,1,1],
                [5,3,50,32,24,1,4]]
        soma = np.zeros((48,64,96), np.uint8)
        soma[23:30,31:34,11:16] = 1
        result, audit = self.run_pipeline(tree, soma=soma)
        ids = {n[0] for n in result}
        self.assertTrue({1,4,5} <= ids)
        self.assertTrue({2,3}.isdisjoint(ids))
        self.assertIn('soma', audit['removed_nodes']['3']['code'])

    def test_geometry_does_not_reparent_multifurcation_or_cut_return_branch(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,32,32,24,1,1],
                [3,3,52,32,24,1,2], [4,3,22,45,24,1,2],
                [5,3,22,19,24,1,2]]
        self.assertEqual([list(n) for n in prune.apply_qc_geometry_steps(tree)], tree)

    def test_final_threshold_in_full_pipeline(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,32,32,24,1,1],
                [3,3,52,32,24,1,2], [4,3,32,39.749,24,1,2],
                [5,3,32,32,31.75,1,2]]
        result, audit = self.run_pipeline(tree)
        self.assertEqual({n[0] for n in result}, {1,2,3,5})
        self.assertEqual(audit['initial_short_leaf_audit']['removed_ids'], [4])
        self.assertEqual(audit['removed_nodes']['4']['code'], 'initial_short_terminal')

    def test_soma_cleanup_can_leave_only_the_root(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,13,32,24,1,1]]
        soma = np.zeros((48,64,96), np.uint8)
        soma[23:30,31:34,11:16] = 1
        result, _ = self.run_pipeline(tree, soma=soma)
        self.assertEqual([n[0] for n in result], [1])

    def test_inward_soma_exit_is_retained(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,14,32,24,1,1],
                [3,3,0,32,24,1,2], [4,3,34,32,24,1,2]]
        soma = np.zeros((48,64,96), np.uint8)
        soma[23:30,31:34,11:16] = 1
        result, _ = self.run_pipeline(tree, soma=soma)
        self.assertTrue({3,4} <= {n[0] for n in result})

    def test_empty_soma_and_misplaced_metadata_do_not_delete_tree(self):
        tree = [[1,1,52,32,24,1,-1], [2,3,72,32,24,1,1]]
        result, _ = self.run_pipeline(tree, soma=np.zeros((48,64,96), np.uint8))
        self.assertEqual({n[0] for n in result}, {1,2})

    def test_single_sharp_bend_in_secondary_branch_is_retained(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,32,32,24,1,1],
                [3,3,42,32,24,1,2], [4,3,37,37,24,1,3],
                [5,3,32,42,24,1,4], [6,3,32,12,24,1,2]]
        result, _ = self.run_pipeline(tree)
        self.assertTrue({n[0] for n in tree} <= {n[0] for n in result})

    def test_short_leaf_filter_runs_before_zigzag_and_is_not_repeated(self):
        # Initially >7.75 um; Zigzag correction later shortens it below 7.75.
        tree = [[1,1,12,32,24,1,-1], [2,3,16,32,24,1,1],
                [3,3,13,32.5,24,1,2], [4,3,17,32.5,24,1,3]]
        result, audit = self.run_pipeline(tree)
        self.assertEqual({n[0] for n in result}, {1,3,4})
        self.assertEqual(audit['initial_short_leaf_audit']['removed_ids'], [])
        self.assertIsNone(audit['final_short_leaf_audit'])
        self.assertEqual(audit['initial_short_leaf_audit']['stage'], 'initial_pruning')

    def test_diagnosis_receives_tree_after_short_leaf_removal(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,32,32,24,1,1],
                [3,3,52,32,24,1,2], [4,3,32,33,24,1,2]]
        actual = prune.diagnose_then_cut
        def check(rows, *args):
            self.assertNotIn(4, {n[0] for n in rows})
            return actual(rows, *args)
        with patch.object(prune, 'diagnose_then_cut', side_effect=check):
            self.run_pipeline(tree)

    def test_guard_does_not_restore_historical_tree(self):
        from utils.stages import load_stage
        stage = load_stage('guard')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root/'42.swc'
            original = b'1 1 12 32 24 1 -1\n2 3 32 32 24 1 1\n'
            final.write_bytes(original)
            spec = dict(id='42', metadata={}, guard_baseline_root='/missing/old/baseline',
                        config=dict(method_revision='test'))
            with patch.object(stage, 'setup_config', return_value=dict(root=root, final=final)):
                stage.run(spec)
            self.assertEqual(final.read_bytes(), original)
            audit = json.loads((root/'regression_guard/42.json').read_text())
            self.assertEqual(audit['decision'], 'validated_restored_base_pruning_output')

    def test_merge_and_zigzag_still_run(self):
        tree = [[1,1,0,0,0,1,-1], [2,3,10,0,0,1,1],
                [3,3,10.001,0,0,1,2], [4,3,20,0,0,1,3]]
        result = prune.apply_qc_geometry_steps(tree)
        self.assertEqual({n[0] for n in result}, {1,2,4})
        self.assertEqual(next(n[6] for n in result if n[0] == 4), 2)
        zigzag = [[1,1,0,0,0,1,-1], [2,3,10,0,0,1,1],
                  [3,3,5,1,0,1,2], [4,3,15,1,0,1,3]]
        result = prune.apply_qc_geometry_steps(zigzag)
        self.assertEqual({n[0] for n in result}, {1,3,4})
        self.assertEqual(next(n[6] for n in result if n[0] == 3), 1)

    def test_old_filters_are_not_called(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,32,32,24,1,1],
                [3,3,52,32,24,1,2]]
        names = ['check_root_ids_connectivity_in_region', 'qc_tune_branch',
                 'qc_correct_multifurcation', 'qc_remove_bad_binary_bifurcation',
                 'qc_remove_branch_towards_soma']
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(patch.object(prune, name, side_effect=AssertionError(name)))
            self.run_pipeline(tree)

    def test_foreign_soma_crossed_by_edge_with_both_endpoints_outside(self):
        mask = np.zeros((64,64,64), bool)
        mask[23:26,31:34,20:23] = True
        context = dict(mask=mask, origin_xyz=np.zeros(3), component_count=1)
        tree = [[1,1,10,32,24,1,-1], [2,3,30,32,24,1,1],
                [3,3,50,32,24,1,2], [4,3,10,52,24,1,1]]
        kept, removed = prune.remove_branches_entering_neighbor_soma(tree, context)
        self.assertEqual(removed, {2,3})
        self.assertEqual({n[0] for n in kept}, {1,4})

    def test_confirmed_same_soma_component_is_not_a_foreign_blocker(self):
        tree = [[1,1,12,32,24,1,-1], [2,3,50,32,24,1,1],
                [3,3,80,32,24,1,2]]
        soma = np.zeros((48,64,96), np.uint8)
        soma[24,32,12] = 1
        soma[22:27,30:35,70:80] = 1
        grouping = dict(same_auxiliary_components=[dict(center_zyx=[24,32,74])])
        # Other integration tests temporarily replace sys.modules. Resolve the
        # actual import target rather than a stale attribute on its parent package.
        group_module = importlib.import_module('algorithms.tracing.soma_signal_groups')
        with patch.object(group_module, 'auxiliary_groups',
                          return_value=(None,None,None,grouping)):
            kept, audit = self.run_pipeline(tree, soma=soma)
        self.assertTrue({1,2,3} <= {n[0] for n in kept})
        self.assertEqual(audit['neighbor_soma_audit']['removed_ids'], [])
        self.assertEqual(audit['neighbor_soma_audit']['neighbor_component_count'], 0)


if __name__ == '__main__':
    unittest.main()
