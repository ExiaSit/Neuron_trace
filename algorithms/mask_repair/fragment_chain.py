"""Admit tiny measured fragments on a validated path without merging side branches."""
import numpy as np
from scipy.spatial import cKDTree


def chain_fragments(edge, touched, surfaces, soma_ids, root, members, contrast):
    endpoints=set(edge['components']);third=set(map(int,touched))-endpoints
    if not endpoints.issubset(touched):return [],'missing_endpoint_contact'
    if not third:return [],None
    if edge['kind']!='local_endpoint_mip':return [],'touches_third_fragment'
    if len(third)>4:return [],'too_many_chain_fragments'
    if sum(len(surfaces[c]) for c in third)>24:return [],'chain_fragments_too_large'
    path=np.asarray(edge['path_zyx']);tree=cKDTree(path)
    endpoint_roots={root(c) for c in endpoints}
    for cid in sorted(third):
        points=surfaces[cid]
        if cid in soma_ids:return [],'chain_touches_soma'
        if len(points)>8 or np.linalg.norm(np.ptp(points,axis=0))>4:
            return [],'chain_fragment_not_tiny'
        distances,indices=tree.query(points)
        if np.any(distances>2.5) or np.any(indices==0) or np.any(indices==len(path)-1):
            return [],'chain_fragment_off_path'
        values=contrast[tuple(points.T)]
        if np.mean((values>2)&(values/edge['noise_sigma']>2))<.85:
            return [],'chain_fragment_weak_signal'
        # A tiny original label may already have been attached to another neuron.
        group=root(cid)
        if group not in endpoint_roots and not members[group].issubset(third):
            return [],'chain_fragment_attached_elsewhere'
    return sorted(third),None


def unapproved_bridge_overlap(collision, origin, accepted, chain_ids, roots, root, solid_bridge, shape):
    """Only overlap with earlier attachments of this chain's own tiny fragments."""
    remaining=collision.copy()
    if not remaining.any():return False
    for old in accepted:
        if old['kind']!='small_fragment' or old['source'] not in chain_ids:continue
        if root(old['source']) not in roots or root(old['target']) not in roots:continue
        points=solid_bridge(old['path_zyx'],shape)-origin
        valid=np.all((points>=0)&(points<remaining.shape),axis=1)
        remaining[tuple(points[valid].T)]=False
    return bool(remaining.any())
