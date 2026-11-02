"""Keep a verified target extension as a routing anchor, not another neuron.

Deleting its seed alone can hand the same shaft to a nearby soma. Retain its
directional source during G-Cut, then combine only its assigned neurites with
the target. Never transfer neurites owned by another independent soma.
"""
import numpy as np
from scipy.spatial import cKDTree
from algorithms.tracing.signal_support import resample_line


def validate_anchor_route(tree, target_id, anchor_id, independent_ids, evidence):
    nodes={int(n[0]):n for n in tree}
    if target_id==anchor_id or anchor_id in independent_ids:return None
    adjacency={i:[] for i in nodes}
    for i,n in nodes.items():
        parent=int(n[6])
        if parent in nodes:adjacency[i].append(parent);adjacency[parent].append(i)
    previous={target_id:None};queue=[target_id]
    for current in queue:
        if current==anchor_id:break
        for neighbor in adjacency[current]:
            if neighbor not in previous and neighbor not in independent_ids:
                previous[neighbor]=current;queue.append(neighbor)
    if anchor_id not in previous:return None
    route=[anchor_id]
    while route[-1]!=target_id:route.append(previous[route[-1]])
    route.reverse()
    points=np.array([nodes[i][2:5][::-1] for i in route],float)
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    if length>1.8*evidence['gap']:return None
    corridor=np.vstack([evidence['path_zyx'],evidence['interior_core_bridge']['path_zyx']])
    deviation=cKDTree(corridor).query(resample_line(points))[0]
    # Both routes can lie at different positions inside the same measured
    # shaft. Bound deviation by its radius, not the narrower saddle threshold.
    tolerance=max(2.,min(evidence['endpoint_radii']))
    if not len(deviation) or deviation.max()>tolerance:return None
    return dict(target_node=int(target_id),anchor_node=int(anchor_id),
                path_node_ids=list(map(int,route)),length_um=length,max_deviation_um=float(deviation.max()),
                corridor_tolerance_um=float(tolerance))


def merge_anchor_assignments(knife, target_id, anchor_ids):
    """Join source labels only across a route already owned by this group."""
    records=[]
    for anchor in anchor_ids:
        record=dict(target_node=int(target_id),anchor_node=int(anchor),merged=False)
        if target_id not in knife.soma_ids or anchor not in knife.soma_ids:
            records.append(dict(record,reason='source_missing_after_contraction'));continue
        allowed={target_id,anchor};seen={target_id};queue=[target_id]
        for current in queue:
            for neighbor in sorted(knife.topo_graph.graph[current].out_neurites):
                key=knife.topo_graph.undirected_hash(current,neighbor)
                if neighbor in seen or (neighbor in knife.soma_ids and neighbor not in allowed):continue
                if knife.neurite_assignments.get(key) not in allowed:continue
                seen.add(neighbor);queue.append(neighbor)
        if anchor not in seen:
            records.append(dict(record,reason='connecting_route_owned_by_other_soma'));continue
        changed=[]
        for key,owner in list(knife.neurite_assignments.items()):
            if owner==anchor:knife.neurite_assignments[key]=target_id;changed.append(list(key))
        knife.soma_ids.remove(anchor)
        records.append(dict(record,merged=True,reassigned_neurites=changed))
    return records
