"""Conservative publication guard against loss of measured baseline branches.

This is an incremental update policy. Candidate reconstructions remain available
for inspection, including candidates that are not selected for publication.
"""
import numpy as np
from scipy.spatial import cKDTree
from algorithms.tracing.signal_support import SignalEvidence,resample_line


def coverage(baseline,candidate,raw):
    a=np.asarray(baseline,float);b=np.asarray(candidate,float);by={int(n[0]):n for n in a};samples=[];directions=[]
    for n in a:
        parent=by.get(int(n[6]))
        if parent is None:continue
        v=parent[2:5]-n[2:5];d=np.linalg.norm(v)
        if d<1e-8:continue
        k=max(2,int(np.ceil(d))+1);samples.extend(np.linspace(n[2:5],parent[2:5],k));directions.extend([v/d]*k)
    if not samples:return 1.
    p=np.asarray(samples);t=np.asarray(directions);ev=SignalEvidence(raw);good=ev.profile(p[:,::-1],t[:,::-1])['good']
    # Sampling candidate edges prevents sparse APP2 vertices looking like a loss.
    byb={int(n[0]):n for n in b};points=[]
    for n in b:
        par=byb.get(int(n[6]))
        if par is None:points.append(n[2:5]);continue
        d=np.linalg.norm(n[2:5]-par[2:5]);points.extend(np.linspace(n[2:5],par[2:5],max(2,int(np.ceil(d))+1)))
    dist=cKDTree(np.asarray(points)).query(p)[0]
    return float((dist[good]<=3).mean()) if good.any() else 1.


def choose(baseline,candidate,raw,min_coverage=.98,foreign_labels=None,foreign_origin_zyx=None):
    """Restore small supported losses, or retain the complete baseline.

    Restored edges come only from the validated old tree and must have current
    3-D ridge support. They attach through a nearby mapped ancestor, never a new
    arbitrary shortcut. Large structural changes are rejected as a whole.
    """
    a=np.asarray(baseline,float);b=np.asarray(candidate,float);before=coverage(a,b,raw)
    if foreign_labels is not None:
        old_hits=foreign_contacts(a,foreign_labels,foreign_origin_zyx)
        new_hits=foreign_contacts(b,foreign_labels,foreign_origin_zyx)
        introduced=sorted(new_hits-old_hits)
        if introduced:return a,dict(decision='baseline',coverage=before,reason='new_independent_soma_contact',foreign_components=int_list(introduced),restored_nodes=0)
    if before>=min_coverage:return b,dict(decision='candidate',coverage=before,restored_nodes=0)
    if before<.85:return a,dict(decision='baseline',coverage=before,reason='large_supported_branch_loss',restored_nodes=0)
    ev=SignalEvidence(raw);tree=cKDTree(b[:,2:5]);dist,near=tree.query(a[:,2:5]);old={int(n[0]):n for n in a};idx={int(n[0]):i for i,n in enumerate(a)}
    children={i:[] for i in old};roots=[]
    for i,n in old.items():
        if int(n[6]) in children:children[int(n[6])].append(i)
        else:roots.append(i)
    mapping={};output=[n.copy() for n in b];coordinates={int(n[0]):n[2:5] for n in b};nextid=int(b[:,0].max())+1;queue=roots[:];added=[]
    for oid in queue:
        queue.extend(children[oid]);n=old[oid];k=idx[oid]
        if dist[k]<=3:
            mapping[oid]=int(b[near[k],0]);continue
        parent=mapping.get(int(n[6]))
        if parent is None:continue
        source=old[int(n[6])][2:5];attach=coordinates[parent]
        if np.linalg.norm(attach-source)>3:continue
        route=np.array([attach,n[2:5]])
        samples=resample_line(route[:,::-1],.5)
        profile=ev.profile(samples,center_search=True)
        # The old centerline can be one voxel off the measured ridge. Recheck
        # within that voxel, and use a wider rim for bright soma-root plateaus.
        good=profile['good']
        broad=np.logical_or.reduce([ev.profile(samples,radius=r,center_search=True)['good'] for r in [8.,12.]])
        good|=(profile['center']>=45)&broad
        if good.mean()<.8:continue
        row=n.copy();row[0]=nextid;row[1]=3;row[6]=parent;output.append(row);coordinates[nextid]=row[2:5];mapping[oid]=nextid;added.append(nextid);nextid+=1
    result=np.asarray(output);after=coverage(a,result,raw)
    if foreign_labels is not None:
        introduced=sorted(foreign_contacts(result,foreign_labels,foreign_origin_zyx)-foreign_contacts(a,foreign_labels,foreign_origin_zyx))
        if introduced:return a,dict(decision='baseline',coverage=before,reason='restoration_enters_independent_soma',foreign_components=int_list(introduced),restored_nodes=0)
    if after<min_coverage:return a,dict(decision='baseline',coverage=before,recovered_coverage=after,reason='supported_loss_remains',restored_nodes=0)
    return result,dict(decision='candidate_with_supported_baseline_branches',coverage=before,recovered_coverage=after,restored_nodes=len(added))


def int_list(values):return [int(x) for x in values]


def foreign_contacts(tree,labels,origin_zyx):
    """Sample entire edges: sparse vertices must not hide soma crossings."""
    nodes={int(n[0]):n for n in np.asarray(tree)};points=[]
    for n in nodes.values():
        parent=nodes.get(int(n[6]));p=n[2:5][::-1]
        if parent is None:points.append(p);continue
        q=parent[2:5][::-1];points.extend(np.linspace(p,q,max(2,int(np.ceil(np.linalg.norm(q-p)/.5))+1)))
    p=np.rint(np.asarray(points)-np.asarray(origin_zyx)).astype(int);good=np.all(p>=0,axis=1)&np.all(p<labels.shape,axis=1)
    return set(int(x) for x in np.unique(labels[tuple(p[good].T)]) if x)


def foreign_context(raw,target_zyx,json_path,mask_path):
    """Known independent soma cores, without deleting ambiguous neighbors."""
    from algorithms.tracing.soma_signal_groups import auxiliary_groups
    from scipy import ndimage as ndi
    labels,origin,same,_=auxiliary_groups(raw,target_zyx,json_path,mask_path)
    q=np.rint(np.asarray(target_zyx)-origin).astype(int);own=int(labels[tuple(q)]) if np.all(q>=0) and np.all(q<labels.shape) else 0
    sizes=np.bincount(labels.ravel());keep=[i for i,n in enumerate(sizes) if i and n>=40 and i!=own and i not in same]
    mask=np.isin(labels,keep);core=ndi.binary_erosion(mask,np.ones((3,3,3)))
    return np.where(core,labels,0),origin
