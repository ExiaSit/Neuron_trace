import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from utils import bootstrap
from algorithms.tracing.signal_support import SignalEvidence
from algorithms.tracing.soma_terminals import prune_soma_terminals


class SomaTerminalTests(unittest.TestCase):
    def setUp(self):
        z, y, x = np.indices((81, 81, 81))
        self.soma = (z-40)**2 + (y-40)**2 + (x-40)**2 <= 10**2
        self.raw = np.full(self.soma.shape, 20., dtype=np.float32)
        self.evidence = SignalEvidence(self.raw)
        self.root = [1, 1, 40., 40., 40., 1., -1]

    def run_rule(self, rows, evidence=None, soma=None):
        return prune_soma_terminals(rows, evidence or self.evidence,
                                    self.soma if soma is None else soma)

    def test_short_exterior_not_hidden_by_long_intracellular_path(self):
        rows = [self.root, [2, 3, 53., 40., 40., 1., 1]]
        kept, removed, audit = self.run_rule(rows)
        self.assertEqual(kept, [self.root])
        self.assertEqual(removed, {2})
        self.assertGreater(audit['decisions'][0]['length_um'], 7.75)
        self.assertLess(audit['decisions'][0]['exterior_length_um'], 7.75)

    def test_true_short_tubular_neurite_is_rescued_by_raw(self):
        z, y, x = np.indices(self.raw.shape)
        raw = 20 + 150*np.exp(-((y-40)**2+(z-40)**2)/2.)
        rows = [self.root, [2, 3, 55., 40., 40., 1., 1]]
        kept, removed, _ = self.run_rule(rows, SignalEvidence(raw.astype(np.float32)))
        self.assertEqual(kept, rows)
        self.assertFalse(removed)

    def test_long_external_branch_and_its_stem_remain_exact(self):
        rows = [self.root, [2, 3, 52., 40., 40., 1., 1],
                [3, 3, 70., 40., 40., 1., 2],
                [4, 3, 40., 53., 40., 1., 1]]
        kept, removed, _ = self.run_rule(rows)
        self.assertEqual(kept, rows[:3])
        self.assertEqual(removed, {4})

    def test_tiny_side_twigs_do_not_hide_spur_or_become_orphans(self):
        rows = [self.root, [2, 3, 51., 40., 40., 1., 1],
                [3, 3, 54., 40., 40., 1., 2],
                [4, 3, 51., 41., 40., 1., 2]]
        kept, removed, _ = self.run_rule(rows)
        self.assertEqual(kept, [self.root])
        self.assertEqual(removed, {2, 3, 4})

    def test_typed_soma_and_uncertain_target_are_preserved(self):
        rows = [self.root, [2, 1, 53., 40., 40., 1., 1]]
        self.assertEqual(self.run_rule(rows)[0], rows)
        displaced = np.roll(self.soma, 20, axis=0)
        self.assertEqual(self.run_rule(rows, soma=displaced)[0], rows)

    def test_short_branch_attached_outside_soma_is_not_candidate(self):
        rows = [self.root, [2, 3, 52., 40., 40., 1., 1],
                [3, 3, 70., 40., 40., 1., 2],
                [4, 3, 52., 49., 40., 1., 2]]
        self.assertEqual(self.run_rule(rows)[0], rows)

    def test_duplicate_requires_no_signal_valley(self):
        rows = [self.root, [2, 3, 55., 40., 40., 1., 1],
                [3, 3, 40., 42., 40., 1., 1],
                [4, 3, 70., 42., 40., 1., 3]]
        def broad_only(evidence, p):
            return np.tile([[False], [False], [True], [True]], (1, len(p)))
        with patch('algorithms.tracing.soma_terminals.tube_support', broad_only):
            kept, removed, _ = self.run_rule(rows)
            self.assertEqual(removed, {2})
            self.assertEqual(kept, [rows[i] for i in (0, 2, 3)])
            # Distinct ridges only two voxels apart must not be merged.
            smooth = self.raw.copy()
            smooth[:, 41, :] = 0
            evidence = SimpleNamespace(raw=self.raw, smooth=smooth)
            self.assertEqual(self.run_rule(rows, evidence)[0], rows)

    def test_image_edge_uncertainty_is_not_negative_evidence(self):
        rows = [self.root, [2, 3, 53., 40., 40., 1., 1]]
        moved = [[*n[:2], n[2], n[3], n[4]-35, *n[5:]] for n in rows]
        soma = np.roll(self.soma, -35, axis=0)
        self.assertEqual(self.run_rule(moved, soma=soma)[0], moved)
