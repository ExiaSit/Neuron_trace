"""Apply bounded soma contraction and the legacy G-Cut short-leaf pruning.

Before 1-um resampling, recursively remove terminal branches with fewer than
two nodes (excluding their starting branch point), protecting known somas.
This is a node-count rule, not a physical-length threshold.
"""
def make_segmentor(swc_path,soma_file_path):
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
    before_pruning=len(segmentor.handy_man.tree_nodes.tree)
    segmentor.handy_man.tree_nodes.prune_short_leaves(
        min_leaf_branch_length=2,stable_nodes=segmentor.handy_man.soma_ids)
    segmentor.tree_nodes,segmentor.soma_ids=segmentor.handy_man.results()
    segmentor.knife=GCut()
    print(f'G-Cut: legacy short-leaf pruning (<2 nodes, soma protected): '
          f'{before_pruning} -> {len(segmentor.tree_nodes.tree)} nodes',flush=True)
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
