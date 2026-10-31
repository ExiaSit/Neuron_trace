"""Target soma anchoring and bounded, direction-aware raw-signal continuation.

Uses the measured 3-D volume, never fills a MIP column. Existing foreground is
preserved; a continuation stops at ambiguous directions or another soma.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

from pathlib import Path
import sys,json
import numpy as np
import tifffile
from utils.gpu_ndimage import ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from algorithms.tracing.signal_support import SignalEvidence
S=np.ones((3,3,3),bool)


def target_body(raw, xyz, json_path=None, soma_path=None):
    """Use the known soma component, or a small signal-supported seed region.

    Do not snap an empty metadata position to an unrelated foreground fragment.
    """
    p=np.rint(np.asarray(xyz)[::-1]).astype(int);lo=np.maximum(0,p-24);hi=np.minimum(raw.shape,p+25)
    sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));patch=raw[sl].astype(np.float32)
    sig=ndi.gaussian_filter(patch,.65);bg=ndi.gaussian_filter(sig,5);center=tuple(p-lo)
    body=np.zeros(patch.shape,bool);other=np.zeros_like(body)
    if json_path and soma_path and Path(json_path).exists() and Path(soma_path).exists():
        bounds=json.loads(Path(json_path).read_text())['bounds_1um_zxy'];origin=np.array([bounds[0],bounds[4],bounds[2]],int)
        aux=tifffile.imread(soma_path)>0;labs,_=ndi.label(aux,S);q=p-origin;cid=0
        if np.all(q>=0) and np.all(q<aux.shape):cid=int(labs[tuple(q)])
        if not cid:
            a=np.maximum(0,q-4);b=np.minimum(aux.shape,q+5)
            if np.all(b>a):
                local=labs[tuple(slice(int(x),int(y)) for x,y in zip(a,b))];pts=np.argwhere(local>0)
                if len(pts):
                    d=np.linalg.norm(pts+a-q,axis=1);j=int(np.argmin(d))
                    if d[j]<=4:cid=int(local[tuple(pts[j])])
        a=np.maximum(lo,origin);b=np.minimum(hi,origin+np.array(aux.shape))
        if np.all(b>a):
            dst=tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo));src=tuple(slice(int(x),int(y)) for x,y in zip(a-origin,b-origin));v=labs[src]
            body[dst]=(v==cid)&(v>0);other[dst]=(v>0)&(v!=cid)
    # Soma brightness varies by cell; use its local contrast, not a global 100 gate.
    core=(sig-bg>=max(5.,float(sig[center]-bg[center])*.18))&(sig>=max(12.,float(sig[center])*.22))
    if not body.any():
        if sig[center]<12 or sig[center]-bg[center]<4:return sl,body,other,dict(status='no_signal_at_metadata',seed_raw=int(raw[tuple(p)]))
        seed=np.zeros_like(core);seed[center]=True
        radius=np.linalg.norm(np.indices(core.shape).transpose(1,2,3,0)-(p-lo),axis=-1)
        body=ndi.binary_propagation(seed,S,mask=core&(radius<=7))
    else:
        # Segmentation is a spatial prior, not permission to add dark voxels.
        allowed=body&(sig>=max(8.,np.percentile(sig[body],80)*.14))
        body=ndi.binary_fill_holes(allowed)
    if not body.any():return sl,body,other,dict(status='empty_supported_soma')
    # Only a short measured shell; long neurites use the directional trace below.
    allowed=ndi.binary_dilation(body,S,iterations=3)&core&~ndi.binary_dilation(other,S)
    body=ndi.binary_propagation(body&~other,S,mask=body|allowed)
    top=int(np.argwhere(body)[:,0].min()+lo[0]) if body.any() else None
    return sl,body,other,dict(status='supported_soma',seed_raw=int(raw[tuple(p)]),voxels=int(body.sum()),completed_soma_top_z=top)


def directions():
    k=np.arange(96);z=1-2*(k+.5)/len(k);a=np.pi*(3-np.sqrt(5))*k;r=np.sqrt(1-z*z)
    return np.column_stack([z,r*np.cos(a),r*np.sin(a)])


def connect_roots(raw, mask, body, other_bodies=None):
    from algorithms.fragment_geometry import robust
    from .soma_bridge import solid_bridge
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from algorithms.mask_repair import signal_reconnect as ev
    labels,n=ndi.label(mask,S);audit=[]
    ids,counts=np.unique(labels[body],return_counts=True);target=int(ids[np.argmax(counts)])
    surface=np.argwhere(body&~ndi.binary_erosion(body,S));kt=cKDTree(surface)
    blo=surface.min(0);bhi=surface.max(0)
    protected=set(np.unique(labels[other_bodies]))-{0,target} if other_bodies is not None else set()
    contrast=ndi.gaussian_filter(raw.astype(np.float32),(0,.6,.6))-ndi.gaussian_filter(raw.astype(np.float32),(0,4,4))
    ev.RAW_CONTRAST=contrast;proposals=[]
    for cid,box in enumerate(ndi.find_objects(labels),1):
        if box is None or cid in protected:continue
        lo=np.array([v.start for v in box]);hi=np.array([v.stop for v in box])
        if np.linalg.norm(np.maximum(0,np.maximum(blo-hi,lo-bhi)))>70:continue
        local=labels[box]==cid
        tips,_=robust.tips_for(local,lo)
        seen=set()
        for point,direction in tips:
            if tuple(point) in seen:continue
            distance,j=kt.query(point)
            if not 2<=distance<=70:continue
            q=surface[j];cosine=direction@((q-point)/distance)
            if cosine<.6:continue
            seen.add(tuple(point))
            e,reason=ev.path_evidence(raw,contrast,labels,point,q,cid,target,'target_root_repair')
            if e:
                e['same_component']=cid==target;proposals.append(e)
            else:audit.append(dict(source=cid,status=reason,gap=float(distance)))
    added=np.zeros_like(mask);accepted=[];used=set()
    for e in sorted(proposals,key=lambda x:x['cost']):
        if e['source']!=target and e['source'] in used:continue
        pts=solid_bridge(e['path_zyx'],mask.shape)
        lo=np.maximum(0,pts.min(0)-1);hi=np.minimum(mask.shape,pts.max(0)+2);sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        path=np.zeros(hi-lo,bool);path[tuple((pts-lo).T)]=True
        touch=ndi.binary_dilation(path,S)
        if set(np.unique(labels[sl][touch]))-{0,e['source'],target}:continue
        if added[sl][touch].any():continue
        # Thin measured path, no unconstrained dilation into a nearby cell.
        added[tuple(pts.T)]=True;used.add(e['source']);accepted.append({k:v for k,v in e.items() if k not in ['path_zyx','layers']})
    return mask|added,dict(accepted=accepted,rejections=audit,added_voxels=int(np.count_nonzero(added&~mask)))

def extend(raw, mask, xyz=None, json_path=None, soma_path=None, completed_somas=None):
    from .continuation_evidence import ContinuationEvidence
    evidence=ContinuationEvidence(raw);final=mask.copy();report=dict(soma={},traces=[],rejected={});target=None
    body=np.zeros_like(mask);neighbors=np.zeros_like(mask)
    if xyz is not None:
        sl,b,other,rec=target_body(raw,xyz,json_path,soma_path);report['soma']=rec;body[sl]=b;neighbors[sl]=other;final|=body
        if b.any() and completed_somas is not None:
            points=np.argwhere(b)+np.array([s.start for s in sl])
            completed_somas.append(dict(center_xyz=list(xyz),points_zyx=points,
                                        top_z=int(points[:,0].min()),source='g_target'))
        if not b.any():return final,dict(report,added_voxels=int(np.count_nonzero(final&~mask)))
    if body.any():
        # Keep full auxiliary-crop neighbor context, including somas outside the
        # smaller root-growth ROI. A source already containing another soma is
        # never newly joined to the target soma.
        if json_path and soma_path and Path(json_path).exists() and Path(soma_path).exists():
            bounds=json.loads(Path(json_path).read_text())['bounds_1um_zxy'];origin=np.array([bounds[0],bounds[4],bounds[2]],int)
            aux=tifffile.imread(soma_path)>0;auxlabels,_=ndi.label(aux,S)
            lower=np.maximum(0,origin);upper=np.minimum(raw.shape,origin+np.array(aux.shape))
            if np.all(upper>lower):
                dst=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper));src=tuple(slice(int(a),int(b)) for a,b in zip(lower-origin,upper-origin))
                local=auxlabels[src];ids,counts=np.unique(local[body[dst]],return_counts=True)
                good=ids>0;own=int(ids[good][np.argmax(counts[good])]) if good.any() else 0
                volumes=np.bincount(auxlabels.ravel());valid_ids=np.flatnonzero(volumes>=40);valid_ids=valid_ids[(valid_ids!=0)&(valid_ids!=own)]
                neighbors[dst]|=np.isin(local,valid_ids)
        final,report['root_connections']=connect_roots(raw,final,body,neighbors)
    labels,n=ndi.label(final,S)
    if body.any():
        ids,ct=np.unique(labels[body],return_counts=True);target=int(ids[np.argmax(ct)])
    candidates=(labels==target) if target else final
    # The established mask may contain several cells; other known somas cannot
    # nominate roots or be crossed by newly added paths.
    sk=skeletonize(candidates);points=np.argwhere(sk);tree=cKDTree(points) if len(points) else None
    adj=tree.query_ball_point(points,1.75) if tree else []
    tips=[]
    for i,nb in enumerate(adj):
        ns=[j for j in nb if j!=i]
        if len(ns)!=1:continue
        chain=[i,ns[0]]
        while len(chain)<10:
            choices=[j for j in adj[chain[-1]] if j not in chain]
            if len(choices)!=1:break
            chain.append(choices[0])
        d=points[i]-points[chain[-1]];norm=np.linalg.norm(d)
        if norm>=4:tips.append((points[i].astype(float),d/norm,'mask_endpoint'))
    # Missing roots need raw candidates too. Skeletonize only the small soma ROI,
    # keeping the raw connection to the confirmed body and excluding neighbors.
    if body.any():
        lo=np.array([s.start for s in sl]);patch=evidence.smooth[sl];bg=ndi.gaussian_filter(patch,4)
        allowed=(patch-bg>=4)&(patch>=12)&~ndi.binary_dilation(other,S,iterations=2)
        near=ndi.distance_transform_edt(~b)
        allowed&=near<=18
        reached=ndi.binary_propagation(b,S,mask=allowed|b)
        rawsk=skeletonize(reached);pts=np.argwhere(rawsk);kt=cKDTree(pts);neighbors1=kt.query_ball_point(pts,1.75)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        pairs=kt.query_pairs(1.75,output_type='ndarray')
        if len(pairs):
            aa,bb=pairs.T;ww=np.linalg.norm(pts[aa]-pts[bb],axis=1)
            graph=coo_matrix((np.r_[ww,ww],(np.r_[aa,bb],np.r_[bb,aa])),shape=(len(pts),len(pts))).tocsr()
            seeds=np.flatnonzero(b[tuple(pts.T)])
            if len(seeds):
                distance,predecessors,_=dijkstra(graph,indices=seeds,min_only=True,return_predecessors=True)
                for i,nb in enumerate(neighbors1):
                    if len(nb)!=2 or near[tuple(pts[i])]<5 or not np.isfinite(distance[i]) or distance[i]>40:continue
                    chain=[i];cur=i
                    while predecessors[cur]>=0:
                        cur=int(predecessors[cur]);chain.append(cur)
                    pth=pts[chain][::-1]+lo
                    hits=np.flatnonzero(final[tuple(pth.T)])
                    if not len(hits):continue
                    pth=pth[hits[-1]:]
                    if len(pth)<3:continue
                    localpts=pth-lo
                    outside=near[tuple(localpts.T)]>=2
                    check=evidence.path(pth[outside])
                    if not check['supported']:continue
                    d=pth[-1]-pth[max(0,len(pth)-7)];norm=np.linalg.norm(d)
                    if norm<4:continue
                    # Connecting a raw exit cannot fuse a neighboring soma.
                    if other[tuple(localpts.T)].any():continue
                    from .soma_bridge import solid_bridge
                    thick_path=solid_bridge(pth,raw.shape)
                    final[tuple(thick_path.T)]=True
                    tips.append((pth[-1].astype(float),d/norm,'soma_exit'))
    dirs=directions();used=[];shape=np.array(raw.shape);added_paths=[]
    for start,u,kind in tips:
        if any(np.linalg.norm(start-s)<4 for s in used):continue
        used.append(start);p=start.copy();path=[p.copy()];stop='length_limit';step=1.5
        for iteration in range(384):
            dd=dirs[dirs@u>.75];dd=np.vstack([u,dd]);q=p+step*dd
            valid=np.all(q>=2,axis=1)&np.all(q<shape-2,axis=1);q=q[valid];dd=dd[valid]
            if not len(q):stop='image_boundary';break
            stats=evidence.profile(q,dd);score=stats['ridge']*(.45+.55*(dd@u))
            score[~stats['good']]=-np.inf
            if not np.isfinite(score).any():stop='insufficient_transverse_signal';break
            j=int(np.argmax(score));p1=q[j];v=dd[j];ip=np.rint(p1).astype(int)
            if neighbors[tuple(ip)]:stop='neighbor_soma';break
            # Test the full grown tube clearance, including diagonal contacts,
            # not just a floating-point centerline rounded to one voxel.
            contact_lo=np.maximum(0,ip-2);contact_hi=np.minimum(shape,ip+3)
            contact_box=tuple(slice(int(a),int(b)) for a,b in zip(contact_lo,contact_hi))
            if np.linalg.norm(p1-start)>6 and final[contact_box].any():
                # Stop before a new contact; connecting two existing fragments is
                # delegated to the established bilateral direction/cost matcher.
                stop='existing_fragment_contact';break
            if len(path)>8 and np.min(np.linalg.norm(np.array(path[:-5])-p1,axis=1))<2:stop='loop';break
            path.append(p1.copy());p=p1;u=.55*u+.45*v;u/=np.linalg.norm(u)
        path=np.asarray(path);length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
        if length<5:report['rejected'][stop]=report['rejected'].get(stop,0)+1;continue
        # Recheck the full centerline; widening is bounded by measured signal.
        from algorithms.tracing.signal_support import resample_line
        line=resample_line(path,.4);ix=np.unique(np.rint(line).astype(int),axis=0)
        lo=np.maximum(0,ix.min(0)-2);hi=np.minimum(shape,ix.max(0)+3);sl1=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));thin=np.zeros(hi-lo,bool);thin[tuple((ix-lo).T)]=True
        local=ndi.binary_dilation(thin,ndi.generate_binary_structure(3,1))
        sig=evidence.smooth[sl1];back=ndi.gaussian_filter(sig,2)
        local&=(sig>=12)&((sig-back)>=2)
        # Keep the minimal face-connected tube even where half-height widening stops.
        local|=ndi.binary_dilation(thin,ndi.generate_binary_structure(3,1))
        # New voxels must not expand into pre-existing disconnected fragments.
        old=labels[sl1];other_components=(old>0)&(old!=labels[tuple(np.rint(start).astype(int))])
        local&=~ndi.binary_dilation(other_components,S)
        # Widening must not close a loop onto the same component either.
        # The only allowed attachment is the original launch neighborhood.
        grid=np.indices(local.shape).transpose(1,2,3,0)+lo
        away=np.linalg.norm(grid-start,axis=-1)>6
        local&=~(ndi.binary_dilation(final[sl1],S)&away)
        # Do not leave disconnected tail pieces after collision clipping.
        attached=local & ndi.binary_dilation(final[sl1],S) & ~away
        local=ndi.binary_propagation(attached,S,mask=local)
        final[sl1]|=local;added_paths.append(path)
        report['traces'].append(dict(kind=kind,start_xyz=start[::-1].tolist(),end_xyz=path[-1,::-1].tolist(),length=length,stop=stop))
    # Keep only additions attached to the original mask or the supported target body.
    keep=ndi.binary_propagation(mask|body,S,mask=final);final=mask|keep
    report.update(added_voxels=int(np.count_nonzero(final&~mask)),original_preserved=bool(np.all(final[mask])),limits='3D signal heuristic; ambiguous contacts stop, no biological ownership ground truth')
    return final,report
