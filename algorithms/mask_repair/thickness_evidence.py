"""Final merger guard with a measured same-soma exception.

This module does not alter thickness growth or its intensity thresholds.
"""
import numpy as np
from utils.gpu_ndimage import ndi


def soma_domains(raw, soma_mask, records, margin=3):
    """Grow c-stage soma evidence only within 3 voxels and its measured half-height.

    Bounding boxes alone never authorize a merger. Ambiguous overlaps are withheld.
    """
    domain=np.zeros(raw.shape,np.uint16)
    if len(records)>=65535:raise ValueError('Too many soma domains')
    for sid,rec in enumerate(records,1):
        sl=tuple(slice(a,b) for a,b in rec['bbox_zyx'])
        seed=np.asarray(soma_mask[sl],bool)
        if not seed.any():continue
        # Assign only the seed fragment nearest this independently measured core.
        local,n=ndi.label(seed,np.ones((3,3,3)))
        center=np.array(rec['center_xyz'][::-1])-np.array([s.start for s in sl])
        points=np.argwhere(seed);nearest=points[np.argmin(np.linalg.norm(points-center,axis=1))]
        if np.linalg.norm(nearest-center)>rec['core_radius']+3.5:continue
        seed=local==local[tuple(nearest)]
        patch=raw[sl].astype(float);sig=ndi.gaussian_filter(patch,.6)
        permitted=(ndi.distance_transform_edt(~seed)<=margin)&(patch>=rec['threshold'])&(sig>=rec['threshold'])
        region=ndi.binary_propagation(seed&permitted,structure=np.ones((3,3,3)),mask=permitted)
        old=domain[sl];conflict=region&(old!=0)&(old!=sid)
        old[region&(old==0)]=sid;old[conflict]=65535
    domain[domain==65535]=0
    return domain


def guard_mergers(mask, added, labels, growth_records, domains):
    """Allow a merged group only when every new inter-component contact is inside
    one and the same measured soma domain. Otherwise keep the original group guard.
    """
    structure=np.ones((3,3,3));out,nout=ndi.label(mask|added,structure)
    n=int(labels.max());groups={}
    for cid,box in enumerate(ndi.find_objects(labels),1):
        point=tuple(np.argwhere(labels[box]==cid)[0]);root=int(out[box][point])
        groups.setdefault(root,[]).append(cid)
    merged={root:ids for root,ids in groups.items() if len(ids)>1}
    report=dict(components_before=n,components_proposed=int(nout),mergers=[],rollback_voxels=0)
    if not merged:
        report['components_after']=n;return added,report
    points=np.concatenate([p for p,cid in growth_records])
    owners=np.concatenate([np.full(len(p),cid,np.int32) for p,cid in growth_records])
    flat=np.ravel_multi_index(points.T,mask.shape);order=np.argsort(flat)
    flat=flat[order];sorted_owners=owners[order]
    stats={root:dict(soma_ids=set(),outside=False,contact_count=0,examples=[]) for root in merged}
    roots=out[tuple(points.T)]
    for delta in np.ndindex(3,3,3):
        shift=np.array(delta)-1
        if not shift.any():continue
        neighbor=points+shift;valid=np.all((neighbor>=0)&(neighbor<mask.shape),axis=1)
        p=points[valid];q=neighbor[valid];own=owners[valid];group=roots[valid]
        other=labels[tuple(q.T)].copy()
        lookup=np.ravel_multi_index(q.T,mask.shape);ix=np.searchsorted(flat,lookup);safe=np.minimum(ix,len(flat)-1)
        found=(ix<len(flat))&(flat[safe]==lookup)&(other==0)
        other[found]=sorted_owners[safe[found]]
        contact=(other>0)&(other!=own)
        p=p[contact];q=q[contact];group=group[contact]
        if not len(p):continue
        a=domains[tuple(p.T)] if domains is not None else np.zeros(len(p),int)
        b=domains[tuple(q.T)] if domains is not None else np.zeros(len(p),int)
        for root in np.unique(group):
            root=int(root)
            if root not in stats:continue
            select=group==root;ok=(a[select]>0)&(a[select]==b[select]);stat=stats[root]
            stat['outside']|=bool((~ok).any());stat['soma_ids'].update(map(int,a[select][ok]))
            stat['contact_count']+=int(select.sum())
            if len(stat['examples'])<8:stat['examples']+=p[select][:8-len(stat['examples'])].tolist()
    rejected=[];allowed=[]
    for root,ids in merged.items():
        s=stats[root];accept=not s['outside'] and len(s['soma_ids'])==1 and s['contact_count']>0
        (allowed if accept else rejected).append(root)
        report['mergers'].append(dict(components=ids,status='allowed_same_soma' if accept else 'rolled_back',
            soma_ids=sorted(s['soma_ids']),has_contact_outside_soma=s['outside'],contact_count=s['contact_count'],contact_examples_zyx=s['examples']))
    rollback=added&np.isin(out,rejected);report['rollback_voxels']=int(rollback.sum());added[rollback]=False
    expected=n-sum(len(merged[root])-1 for root in allowed)
    report['components_after']=int(ndi.label(mask|added,structure)[1])
    assert report['components_after']==expected
    return added,report
