"""Image-coordinate 3-D evidence shared by continuation and pruning.

A MIP never supplies voxel evidence. All lengths below are 1-um voxel lengths.
"""
import numpy as np
from utils.gpu_ndimage import ndi


def resample_line(points, step=.75):
    points=np.asarray(points,float)
    if len(points)<2:return points
    ds=np.linalg.norm(np.diff(points,axis=0),axis=1)
    keep=np.r_[True,ds>1e-6];points=points[keep]
    if len(points)<2:return points
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    x=np.linspace(0,d[-1],max(2,int(np.ceil(d[-1]/step))+1))
    return np.column_stack([np.interp(x,d,points[:,j]) for j in range(3)])


class SignalEvidence:
    def __init__(self, raw):
        self.raw=raw
        self.smooth=ndi.gaussian_filter(raw.astype(np.float32),.65)

    def profile(self, points, tangents=None, radius=4., center_search=False):
        """Subvoxel center and eight transverse samples; coordinates are ZYX."""
        p=np.asarray(points,float)
        if not len(p):return dict(center=np.array([]),ridge=np.array([]),noise=np.array([]),good=np.array([],bool))
        if tangents is None:
            tangents=np.gradient(p,axis=0) if len(p)>1 else np.array([[0.,0.,1.]])
        t=np.asarray(tangents,float);t=t/np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-8)
        ref=np.tile([1.,0.,0.],(len(p),1));ref[np.abs(t[:,0])>.9]=[0,1,0]
        u=np.cross(t,ref);u/=np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-8);v=np.cross(t,u)
        center=ndi.map_coordinates(self.smooth,p.T,order=1,mode='nearest')
        if center_search:
            for offset in ([1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]):
                center=np.maximum(center,ndi.map_coordinates(self.smooth,(p+offset).T,order=1,mode='nearest'))
        ring=np.array([ndi.map_coordinates(self.smooth,(p+radius*(np.cos(a)*u+np.sin(a)*v)).T,order=1,mode='nearest') for a in np.linspace(0,2*np.pi,8,endpoint=False)])
        bg=np.median(ring,axis=0);ridge=center-bg
        # Ring dispersion includes clutter: dense neighborhoods demand stronger evidence.
        noise=np.maximum(1.,.25*1.4826*np.median(np.abs(ring-bg),axis=0))
        good=(ridge>=np.maximum.reduce([np.full_like(center,3.),2.5*noise,.10*center]))&(center>=10)&(np.mean(ring<center[None,:]-2,axis=0)>=.625)
        return dict(center=center,ridge=ridge,noise=noise,good=good)

    def path(self, points):
        p=resample_line(points)
        if len(p)<2:return dict(supported=False,fraction=0.,dark_length=0.,length=0.)
        stats=self.profile(p);d=np.linalg.norm(np.diff(p,axis=0),axis=1);run=longest=0.
        for i,ok in enumerate(stats['good']):
            run=0 if ok else run+(d[max(0,i-1)] if len(d) else 0);longest=max(longest,run)
        return dict(supported=bool(stats['good'].mean()>=.8 and longest<=2.),fraction=float(stats['good'].mean()),dark_length=float(longest),length=float(d.sum()),median_ridge=float(np.median(stats['ridge'])),median_intensity=float(np.median(stats['center'])))

    def supported_xyz(self, points):return self.path(np.asarray(points)[:,::-1])['supported']


def tree_paths(tree):
    nodes={int(n[0]):n for n in tree};children={i:[] for i in nodes}
    for i,n in nodes.items():
        if int(n[6]) in nodes:children[int(n[6])].append(i)
    paths=[]
    for nid,n in nodes.items():
        if int(n[6]) not in nodes or len(children[nid])!=1:
            for child in children[nid]:
                path=[nid,child]
                while len(children[path[-1]])==1:path.append(children[path[-1]][0])
                paths.append(path)
    return nodes,children,paths


def prune_unsupported(tree, evidence, soma_xyz, protect_radius=12., protected_mask=None):
    """Cut sustained raw-signal gaps, including gaps hidden inside long segments.

    Geometry alone is never sufficient here. Samples remain in 3-D, and short
    dips/beading are tolerated. Return original node IDs for the reason report.
    """
    nodes,children,paths=tree_paths(tree);cuts=[];reports=[]
    for path in paths:
        xyz=np.array([nodes[i][2:5] for i in path],float)
        p=resample_line(xyz[:,::-1],.75)
        if len(p)<8:continue
        v=evidence.profile(p);best=evidence.profile(p,center_search=True);ds=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        far=np.linalg.norm(p-np.asarray(soma_xyz)[::-1],axis=1)>protect_radius
        if protected_mask is not None:
            ix=np.rint(p).astype(int);valid=np.all(ix>=0,axis=1)&np.all(ix<protected_mask.shape,axis=1)
            protected=np.zeros(len(p),bool);protected[valid]=protected_mask[tuple(ix[valid].T)]
            far &= ~protected
        # Strictly absent ridge rather than merely failing the strong-support test.
        weak=((v['ridge']<np.maximum(1.2,1.25*v['noise']))|(v['center']<8))&far
        start=None
        for j,bad in enumerate(weak):
            if bad and start is None:start=j
            if start is not None and (not bad or j==len(weak)-1):
                stop=j if not bad else j+1
                length=ds[stop-1]-ds[start]
                nearby_weak=(best['ridge'][start:stop]<np.maximum(1.2,1.25*best['noise'][start:stop]))|(best['center'][start:stop]<8)
                if length>=4.5 and nearby_weak.mean()>=.5:
                    cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
                    k=min(len(path)-1,max(1,int(np.searchsorted(cum,ds[start]+1.))))
                    cuts.append(path[k]);reports.append(dict(node=int(path[k]),reason='sustained_3d_signal_gap',gap_length=float(length),xyz=list(map(float,nodes[path[k]][2:5]))));break
                start=None
    removed=set();stack=cuts[:]
    while stack:
        n=stack.pop()
        if n in removed:continue
        removed.add(n);stack.extend(children[n])
    return [n for n in tree if int(n[0]) not in removed],removed,reports


def prune_crossing_switches(tree, evidence, soma_xyz, protect_radius=14., protected_mask=None):
    """Reject a sharp switch onto an independently continuous transverse shaft.

    A bend alone is insufficient: the putative foreign shaft must continue on
    both sides in raw 3-D signal, while forward continuation of the incoming
    branch is absent. Broad halos fail the transverse-profile requirement.
    """
    nodes,children,paths=tree_paths(tree);cuts=[];reports=[]
    for nid,node in nodes.items():
        if int(node[6]) not in nodes or not children[nid]:continue
        center=np.asarray(node[2:5],float)
        from algorithms.tracing.soma_signal_groups import inside_confirmed
        if inside_confirmed([center],protected_mask):continue
        if np.linalg.norm(center-np.asarray(soma_xyz))<protect_radius:continue
        cur=nid;back=[center];length=0.
        while int(nodes[cur][6]) in nodes and length<7:
            par=int(nodes[cur][6]);q=np.asarray(nodes[par][2:5],float);length+=np.linalg.norm(q-back[-1]);back.append(q);cur=par
        if length<6:continue
        incoming=center-back[-1];incoming/=np.linalg.norm(incoming)
        for child in children[nid]:
            forward=[center,np.asarray(nodes[child][2:5],float)];cur=child;length=np.linalg.norm(forward[1]-center)
            while len(children[cur])==1 and length<7:
                cur=children[cur][0];q=np.asarray(nodes[cur][2:5],float);length+=np.linalg.norm(q-forward[-1]);forward.append(q)
            if length<6:continue
            outgoing=forward[-1]-center;outgoing/=np.linalg.norm(outgoing)
            dot=float(incoming@outgoing)
            if dot>.43:continue
            # Sample away from the actual junction, where local overlap alone is
            # expected even for a genuine bend.
            steps=np.linspace(3.,11.,12)
            opposite=(center[None,:]-steps[:,None]*outgoing)[...,::-1]
            absent=(center[None,:]+steps[:,None]*incoming)[...,::-1]
            other=evidence.profile(opposite,np.tile(outgoing[::-1],(len(steps),1)))
            ahead=evidence.profile(absent,np.tile(incoming[::-1],(len(steps),1)))
            own=evidence.path(np.asarray(forward)[:,::-1])
            before=evidence.path(np.asarray(back)[:,::-1])
            if other['good'].mean()>=.8 and ahead['good'].mean()<=.35 and own['supported'] and np.median(other['ridge'])>=4.:
                cuts.append(child);reports.append(dict(node=int(child),xyz=center.tolist(),reason='switch_to_transverse_shaft',angle=float(np.degrees(np.arccos(np.clip(dot,-1,1)))),opposite_support=float(other['good'].mean()),incoming_forward_support=float(ahead['good'].mean())))
    removed=set();stack=cuts[:]
    while stack:
        nid=stack.pop()
        if nid in removed:continue
        removed.add(nid);stack.extend(children[nid])
    return [n for n in tree if int(n[0]) not in removed],removed,reports
