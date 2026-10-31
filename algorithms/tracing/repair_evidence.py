"""Candidate 3-D evidence; all coordinates ZYX, spacing 1 um."""
import numpy as np
from scipy import ndimage as ndi
from algorithms.tracing.signal_support import resample_line,tree_paths

def tube_support(evidence,p,radii=(2.,3.,4.,6.)):
 p=np.asarray(p,float)
 if len(p)<2:return np.ones(len(p),bool)
 t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-8)
 ref=np.tile([1.,0,0],(len(p),1));ref[np.abs(t[:,0])>.9]=[0,1,0]
 u=np.cross(t,ref);u/=np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-8);v=np.cross(t,u)
 good=np.zeros((len(radii),len(p)),bool)
 for offset in ([0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]):
  q=p+offset;center=ndi.map_coordinates(evidence.smooth,q.T,order=1,mode='nearest')
  for j,r in enumerate(radii):
   coords=np.array([q+r*(np.cos(a)*u+np.sin(a)*v) for a in np.linspace(0,2*np.pi,16,endpoint=False)])
   ring=ndi.map_coordinates(evidence.smooth,coords.reshape(-1,3).T,order=1,mode='nearest').reshape(16,-1)
   pair=np.maximum(ring[:8],ring[8:]);valid=np.all((coords>=0)&(coords<np.array(evidence.raw.shape)-1),axis=(0,2))
   good[j]|=((np.mean(center-pair>=np.maximum(2.,.1*center),axis=0)>=.75)&(center>=10)&(np.min(center-pair,axis=0)>=np.maximum(.5,.02*center))&valid)
 # Multi-scale rescue: evidence at any radius preserves a thick shaft.
 return good

def sample_mask(mask,p):
 q=np.rint(p).astype(int);ok=np.all((q>=0)&(q<np.array(mask.shape)),axis=1)
 hit=np.zeros(len(p),bool);hit[ok]=mask[tuple(q[ok].T)];return hit

def _prune_repair_artifacts_once(tree,evidence,original,soma,neighbor_points,neighbor_distance_um=30.,window_um=8.,support_fraction=.8,foreign_witness_ids=None):
 """Restrict rejection to newly repaired paths near a separately measured soma.

 Terminal twigs need a whole-path failure; internal connectors need an 8-um
 predominantly added interval with evidence of original mask on both sides.
 Multiple transverse scales and coherent +/-1 voxel centering rescue true tubes.
 Roots, soma and original-mask-supported paths are protected. No ID gates.
 """
 from scipy.spatial import cKDTree
 rows=[list(n) for n in tree]
 from algorithms.tracing.terminal_pruning import prune_final_short_leaves
 # Virtual view joins paths separated only by already-discardable tiny twigs.
 # Apply evidence cuts to the full input; length pruning still writes last.
 preview,_=prune_final_short_leaves(rows)
 nodes,children,paths=tree_paths(preview)
 audit=dict(rule='repair_without_bilateral_tube',parameters=dict(neighbor_distance_um=neighbor_distance_um,window_um=window_um,support_fraction=support_fraction,new_fraction=.9,radii_um=[2,3,4,6],connector_soma_protection_um=14.),decisions=[],removed_ids=[])
 if not len(neighbor_points):return rows,set(),audit
 kd=cKDTree(neighbor_points);cuts=[]
 roots=[n for n in rows if int(n[6])==-1]
 if len(roots)!=1:raise ValueError("Expected one root")
 root_zyx=np.array(roots[0][2:5])[::-1]
 foreign_witness_ids=set(foreign_witness_ids or [])
 for path in paths:
  xyz=np.array([nodes[i][2:5] for i in path],float);p=resample_line(xyz[:,::-1],.5)
  if len(p)<8:continue
  ds=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
  orig=sample_mask(original,p);own=sample_mask(soma,p);near=kd.query(p)[0]<=neighbor_distance_um
  far=np.linalg.norm(p-root_zyx,axis=1)>=14.
  safe=np.all((p>=7)&(p<np.array(original.shape)-8),axis=1)
  interior=(ds>=2)&~own&near
  if not np.any(interior&~orig):continue
  good=None;cut=None;reason=None;stats={}
  if not children[path[-1]] and ds[-1]>=6 and not own[-1] and interior.mean()>=.7 and safe[interior].all() and orig[interior].mean()<=.1:
   good=tube_support(evidence,p)
   fractions=good[:,interior].mean(axis=1)
   if fractions.max()<support_fraction:
    cut=path[1];reason='added_terminal_without_bilateral_tube';stats=dict(length_um=float(ds[-1]),support_by_radius=fractions.tolist(),added_fraction=float(1-orig[interior].mean()))
  ancestor=path[0];witness=False
  while ancestor in nodes:
   if ancestor in foreign_witness_ids:witness=True;break
   ancestor=int(nodes[ancestor][6])
  if cut is None and witness and ds[-1]>=window_um+6 and (~orig).sum()>=window_um/.5*.9:
   if good is None:good=tube_support(evidence,p)
   for j in range(len(p)):
    k=np.searchsorted(ds,ds[j]+window_um)
    if k>=len(p):break
    selection=slice(j,k+1)
    if orig[selection].mean()>.1 or own[selection].any() or not near[selection].all() or not safe[selection].all() or not far[selection].all():continue
    # Require this to be an actual connector, not an isolated endpoint or edge.
    before=(ds>=ds[j]-12)&(ds<ds[j]);after=(ds>ds[k])&(ds<=ds[k]+12)
    if orig[before].sum()<4 or orig[after].sum()<4:continue
    fractions=good[:,selection].mean(axis=1)
    if fractions.max()>=support_fraction:continue
    cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    index=max(1,int(np.searchsorted(cum,ds[j])))
    if any(int(nodes[n][1])==1 for n in path[index:]):continue
    cut=path[index];reason='added_connector_without_bilateral_tube';stats=dict(window_start_xyz=p[j][::-1].tolist(),window_end_xyz=p[k][::-1].tolist(),window_length_um=float(ds[k]-ds[j]),support_by_radius=fractions.tolist(),added_fraction=float(1-orig[selection].mean()));break
  if cut is not None:
   if any(int(nodes[n][1])==1 for n in path[1:]):continue
   cuts.append(cut);audit['decisions'].append(dict(reason=reason,cut_id=cut,segment_start=path[0],segment_end=path[-1],**stats))
 nodes,children,_=tree_paths(rows)
 removed=set();stack=cuts[:]
 while stack:
  nid=stack.pop()
  if nid in removed:continue
  removed.add(nid);stack.extend(children[nid])
 # Never remove a root or typed soma, including descendants of a candidate cut.
 if any(int(nodes[i][1])==1 or int(nodes[i][6]) not in nodes for i in removed):
  raise ValueError('Candidate would delete a protected soma/root')
 audit['removed_ids']=sorted(removed)
 return [n for n in rows if int(n[0]) not in removed],removed,audit


def prune_repair_artifacts(tree, evidence, original, soma, neighbor_points, **kwargs):
    rows = [list(n) for n in tree]
    removed = set()
    decisions = []
    iteration = 0
    while True:
        iteration += 1
        kept, deleted, audit = _prune_repair_artifacts_once(
            rows, evidence, original, soma, neighbor_points, **kwargs)
        for decision in audit['decisions']:
            decision['iteration'] = iteration
            decisions.append(decision)
        if not deleted:
            break
        assert len(kept) < len(rows)
        removed.update(deleted)
        rows = kept
    audit.update(decisions=decisions, removed_ids=sorted(removed), iterations=iteration)
    return rows, removed, audit


def foreign_entry_ancestors(tree, neighbor_points):
    """Preserve evidence of a sibling entering another soma before old pruning.

    Root alone is not an ownership witness. The candidate connector must share
    a non-root stem with a path that actually entered the measured foreign soma.
    """
    from scipy.spatial import cKDTree
    if not len(neighbor_points):
        return set()
    nodes, _, _ = tree_paths(tree)
    kd = cKDTree(neighbor_points)
    hits = []
    for nid, node in nodes.items():
        parent = int(node[6])
        if parent not in nodes:
            continue
        p = np.rint(resample_line(np.array([nodes[parent][2:5], node[2:5]])[:, ::-1], .5))
        if np.any(kd.query(p)[0] < .01):
            hits.append(nid)
    ancestors = set()
    for nid in hits:
        while nid in nodes and int(nodes[nid][6]) != -1 and nid not in ancestors:
            ancestors.add(nid)
            nid = int(nodes[nid][6])
    return ancestors


def prune_with_context(tree, evidence, context):
    """Load preserved input masks, leaving legacy callers without context intact."""
    if not context or not len(context.get('neighbor_points', [])):
        return tree, set(), dict(rule='repair_without_bilateral_tube',
                                status='no_distinct_neighbor_or_context',
                                decisions=[], removed_ids=[])
    import tifffile
    original = tifffile.imread(context['original_mask_path']) > 0
    soma = tifffile.imread(context['measured_soma_path']) > 0
    if original.shape != evidence.raw.shape or soma.shape != original.shape:
        raise ValueError('Repair pruning requires matching 1-um ZYX raw and masks')
    kept, removed, audit = prune_repair_artifacts(
        tree, evidence, original, soma, context['neighbor_points'],
        foreign_witness_ids=context.get('foreign_witness_ids', []))
    audit['foreign_entry_witness_ancestors'] = sorted(context.get('foreign_witness_ids', []))
    audit['status'] = 'pruned' if removed else 'no_candidates'
    return kept, removed, audit
