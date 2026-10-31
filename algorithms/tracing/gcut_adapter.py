"""Bounded soma contraction and terminal pruning by 3-D path length.

Current pipeline runs terminal pruning last, after all downstream pruning.
G-Cut retains an explicit legacy replay option for historical diagnostics.
Coordinates are 1-um image XYZ. Include the attachment edge to the surviving
branch point or soma in each terminal path's length; protect known somas.
"""
import math

SHORT_LEAF_MIN_LENGTH_UM = 7.75


def prune_short_leaves_by_length(tree, stable_nodes, min_length_um=SHORT_LEAF_MIN_LENGTH_UM):
    """Iteratively delete terminal paths strictly shorter than min_length_um.

    Keep the attachment node. Update child counts after each deletion, as in
    the former node-count rule. Process leaves by ID for reproducible results.
    Return one audit record per deletion event (a path may contain many nodes).
    """
    if not math.isfinite(min_length_um) or min_length_um <= 0:
        raise ValueError('Short-leaf length threshold must be positive and finite')
    if tree.empty():
        return []
    protected = set(stable_nodes) | {tree.root_id()}
    removed_branches = []
    iteration = 0
    while True:
        iteration += 1
        tree.find_children(rescan=True)
        changed = False
        for leaf_id in sorted(tree.leaves()):
            if leaf_id in protected or leaf_id not in tree.tree:
                continue
            path, seen, length, current = [], set(), 0.0, leaf_id
            while True:
                node = tree.tree[current]
                parent_id = node.parent_id
                if parent_id not in tree.tree:
                    raise ValueError(f'Missing parent {parent_id} for terminal node {current}')
                parent = tree.tree[parent_id]
                edge_length = math.dist((node.x, node.y, node.z), (parent.x, parent.y, parent.z))
                if not math.isfinite(edge_length):
                    raise ValueError(f'Non-finite terminal edge at node {current}')
                path.append(current)
                seen.add(current)
                length += edge_length
                if parent_id in protected or tree.n_children(parent_id) > 1:
                    break
                if parent_id in seen:
                    raise ValueError('Cycle in terminal branch')
                current = parent_id
            if length < min_length_um:
                # Remove from the tip upward; each node is a leaf at removal.
                for nid in path:
                    tree.children[tree.tree[nid].parent_id].remove(nid)
                    tree.children.pop(nid)
                    tree.tree.pop(nid)
                removed_branches.append(dict(node_ids=path, anchor_id=parent_id,
                                             length_um=length, iteration=iteration))
                changed = True
        if not changed:
            return removed_branches


def make_segmentor(swc_path,soma_file_path,min_leaf_length_um=None):
    from neuron_segmentation import NeuronSegmentation
    from gcut.preprocess import HandyMan
    from gcut.gcut import GCut
    segmentor=NeuronSegmentation.__new__(NeuronSegmentation)
    segmentor.scale_z=1.0
    segmentor.scale_output_z=False
    class BoundedSomaPreprocessor(HandyMan):
        def sink_ancestors(self,node_id):
            # Upstream switches to the tree root before checking distance when
            # a soma is its direct child, merging even distant known somas.
            if node_id==self.tree_nodes.root_id():return node_id
            parent_id=self.tree_nodes.tree[node_id].parent_id
            while parent_id in self.tree_nodes.tree:
                if parent_id in self.soma_ids and parent_id!=node_id:break
                distance=self.tree_nodes.pair_distance(node_id,parent_id)
                if distance>self.tree_nodes.tree[node_id].radius+self.tree_nodes.tree[parent_id].radius:break
                if parent_id==self.tree_nodes.root_id():
                    self.soma_ids.discard(node_id);self.soma_ids.add(parent_id)
                    return parent_id
                self.tree_nodes.tree[node_id].radius=self._combine_radius(node_id,parent_id)
                grandparent=self.tree_nodes.tree[parent_id].parent_id
                self.tree_nodes.remove_node(parent_id);parent_id=grandparent
            return node_id
        def sink_descendants(self,node_id):
            queue=list(self.tree_nodes.children[node_id])
            while queue:
                child=queue.pop(0)
                if child in self.soma_ids and child!=node_id:continue
                distance=self.tree_nodes.pair_distance(node_id,child)
                if distance>self.tree_nodes.tree[node_id].radius+self.tree_nodes.tree[child].radius:continue
                self.tree_nodes.tree[node_id].radius=self._combine_radius(node_id,child)
                queue.extend(self.tree_nodes.children[child]);self.tree_nodes.remove_node(child)
    segmentor.handy_man=BoundedSomaPreprocessor(swc_path,soma_file_path,scale_z=1.0)
    segmentor.handy_man.extract_soma_ids({},size_threshold=None)
    segmentor.handy_man.soma_sinks()
    segmentor.short_leaf_audit = None
    # Explicit threshold is only for replaying historical preprocessing.
    if min_leaf_length_um is not None:
        def snapshot():
            return [[n.node_id,n.node_type,n.x,n.y,n.z,n.radius,n.parent_id]
                    for n in segmentor.handy_man.tree_nodes.tree.values()]
        before_nodes=snapshot()
        removed_branches=prune_short_leaves_by_length(
            segmentor.handy_man.tree_nodes,segmentor.handy_man.soma_ids,min_leaf_length_um)
        after_nodes=snapshot()
        segmentor.short_leaf_audit=dict(before_nodes=before_nodes,after_nodes=after_nodes,
            removed_ids=sorted({n[0] for n in before_nodes}-{n[0] for n in after_nodes}),
            soma_ids=sorted(segmentor.handy_man.soma_ids),min_leaf_branch_length_um=min_leaf_length_um,
            removed_branches=removed_branches,
            removed_length_um=sum(branch['length_um'] for branch in removed_branches),
            length_metric='3-D path length including attachment edge; 1-um image XYZ',
            comparison='strictly_less_than',leaf_order='ascending_node_id',
            rule=f'iterative terminal path length <{min_leaf_length_um:g} um; attachment node and somas protected',
            capture_point='after soma contraction, before G-Cut segmentation')
    segmentor.tree_nodes,segmentor.soma_ids=segmentor.handy_man.results()
    segmentor.knife=GCut()
    if min_leaf_length_um is None:
        print('G-Cut: terminal length pruning deferred to the final pruning stage', flush=True)
    else:
        print(f'Historical replay: short terminals <{min_leaf_length_um:g} um removed before G-Cut', flush=True)
    return segmentor


def component_maxima(values, labels, count):
    """Same per-component peak, scanning each bounding box instead of the full volume."""
    import numpy as np
    from scipy import ndimage as ndi
    result=[]
    for cid,box in enumerate(ndi.find_objects(labels,max_label=count),1):
        if box is None:continue
        local=np.where(labels[box]==cid,values[box],-np.inf)
        point=np.array(np.unravel_index(np.argmax(local),local.shape))
        result.append(tuple(map(int,point+np.array([s.start for s in box]))))
    return result
