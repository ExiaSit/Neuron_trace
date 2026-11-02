import unittest
import json
import numpy as np
from types import SimpleNamespace
from algorithms.tracing.soma_seed_ownership import validate_anchor_route,merge_anchor_assignments


class OwnershipTests(unittest.TestCase):
    def knife(self):
        graph={i:SimpleNamespace(out_neurites={}) for i in [1,4,6,8,9]}
        owners={(1,4):1,(4,6):6,(6,8):6,(6,9):9}
        for a,b in owners:graph[a].out_neurites[b]=0;graph[b].out_neurites[a]=0
        return SimpleNamespace(soma_ids={1,6,9},neurite_assignments=owners,
            topo_graph=SimpleNamespace(graph=graph,undirected_hash=lambda a,b:tuple(sorted((a,b)))))
    def test_group_only_verified_source_and_preserve_foreign_assignment(self):
        knife=self.knife();audit=merge_anchor_assignments(knife,1,[6])
        self.assertTrue(audit[0]['merged']);self.assertEqual(knife.soma_ids,{1,9})
        self.assertEqual(knife.neurite_assignments,{(1,4):1,(4,6):1,(6,8):1,(6,9):9})
    def test_foreign_owned_connector_vetoes_group(self):
        knife=self.knife();knife.neurite_assignments[4,6]=9
        original=knife.neurite_assignments.copy()
        audit=merge_anchor_assignments(knife,1,[6]);self.assertFalse(audit[0]['merged'])
        self.assertEqual(knife.neurite_assignments,original);self.assertEqual(knife.soma_ids,{1,6,9})
    def test_missing_contracted_anchor_is_skipped(self):
        knife=self.knife();knife.soma_ids.remove(6)
        self.assertFalse(merge_anchor_assignments(knife,1,[6])[0]['merged'])
    def test_tree_route_must_follow_measured_corridor_without_other_somas(self):
        tree=[[1,1,10,10,10,1,-1],[2,3,14,10,10,1,1],[3,3,18,10,10,1,2]]
        path=[[10,10,x] for x in range(10,19)]
        evidence=dict(gap=8,path_zyx=path,endpoint_radii=[9,4],interior_core_bridge=dict(path_zyx=path))
        record=validate_anchor_route(tree,np.int64(1),np.int64(3),set(),evidence)
        self.assertIsNotNone(record)
        json.dumps(record)  # matched SWC IDs originate from NumPy arrays
        self.assertIsNone(validate_anchor_route(tree,1,3,{2},evidence))
        tree[1][3]=16
        self.assertIsNone(validate_anchor_route(tree,1,3,set(),evidence))


if __name__=='__main__':unittest.main()
