"""Morphology-aware trial. Original TIFF voxel coordinates; no physical spacing calibration."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import json,sys
from pathlib import Path
import os
RAW_PATH=os.environ.get('RECONNECT_RAW','')
MASK_PATH=os.environ.get('RECONNECT_MASK','')
WORK=Path(os.environ.get('RECONNECT_WORK','.'))

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.measure import marching_cubes
from algorithms.mask_repair.soma_bridge import candidates as soma_candidates, solid_bridge
from algorithms.mask_repair.mip_backmap import backmap
RAW_CONTRAST=None
BACKMAP_STATS={}
O=WORK/'b'
BASE=WORK/'a'
CFG=dict(small_max_gap=18,small_neighbors=5,min_signal_fraction=.85,min_layer_fraction=.6,max_dark_run=2,max_detour=1.6,ambiguity_margin=.15)

def unambiguous(scores):
    return len(scores)==1 or (scores[1]-scores[0])/max(scores[1],1e-9)>=CFG['ambiguity_margin']

def path_evidence(raw,contrast,labels,p,q,a,b,kind):
    p=np.asarray(p,int);q=np.asarray(q,int);gap=float(np.linalg.norm(q-p))
    if gap<1:return None,'touching'
    lo=np.maximum(0,np.minimum(p,q)-7);hi=np.minimum(raw.shape,np.maximum(p,q)+8);sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))
    c=contrast[sl];l=labels[sl];start=p-lo;end=q-lo
    observed=RAW_CONTRAST[sl] if RAW_CONTRAST is not None else c
    # Local signal-to-noise scale, estimated from high-frequency background residual.
    residual=raw[sl].astype(float)-ndi.gaussian_filter(raw[sl].astype(float),(0,.6,.6))
    bg=residual[l==0];sigma=max(1.,float(np.median(abs(bg-np.median(bg)))/.6745)) if len(bg) else 1.
    strength=np.maximum(c,0)/sigma
    line=np.ones(c.shape,bool);ix=np.rint(np.linspace(start,end,int(gap*3)+2)).astype(int);line[tuple(ix.T)]=False
    allowed=(ndi.distance_transform_edt(line)<=5)&((l==0)|(l==a)|(l==b))
    cost=1+12/(1+strength);cost[~allowed]=np.inf
    try:
        path,total=route_through_array(cost,tuple(start),tuple(end),fully_connected=True)
    except ValueError:
        return None,'blocked_path'
    if not np.isfinite(total):return None,'blocked_path'
    path=np.asarray(path);gapmask=l[tuple(path.T)]==0;g=path[gapmask]
    if not len(g):return None,'no_gap'
    length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    if length>CFG['max_detour']*gap:return None,'detour'
    vals=c[tuple(g.T)];snr=strength[tuple(g.T)];supported=(vals>2)&(snr>2)
    observed_values=observed[tuple(g.T)]
    reinforced=(vals>observed_values+1e-5)&~((observed_values>2)&(observed_values/sigma>2))
    if np.mean(reinforced)>.35 or np.mean(observed_values>.75)<.85:
        return None,'too_much_inferred_support'
    dark=run=0
    for good in supported:
        run=0 if good else run+1;dark=max(dark,run)
    layers=[dict(z=int(z+lo[0]),fraction=float(supported[g[:,0]==z].mean())) for z in np.unique(g[:,0])]
    if supported.mean()<CFG['min_signal_fraction'] or min(x['fraction'] for x in layers)<CFG['min_layer_fraction'] or dark>CFG['max_dark_run']:
        return None,'insufficient_layer_signal'
    # A neurite must stand out from its transverse neighborhood, not just background.
    # Eight samples around each 3D path tangent suppress broad halos and plateaus.
    smooth_patch=ndi.gaussian_filter(raw[sl].astype(float),.6)
    path_float=path.astype(float)
    tangent=np.gradient(path_float,axis=0)[gapmask]
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
    reference=np.zeros_like(tangent);reference[:,0]=1
    parallel=np.abs(tangent[:,0])>.9;reference[parallel]=[0,1,0]
    u=np.cross(tangent,reference);u/=np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-9)
    v=np.cross(tangent,u)
    ring=[]
    for angle in np.linspace(0,2*np.pi,8,endpoint=False):
        points=g+4*(np.cos(angle)*u+np.sin(angle)*v)
        ring.append(ndi.map_coordinates(smooth_patch,points.T,order=1,mode='nearest'))
    ridge=smooth_patch[tuple(g.T)]-np.median(ring,axis=0)
    ridge_fraction=float(np.mean(ridge>max(2,1.5*sigma)))
    if ridge_fraction<.65:return None,'broad_halo_without_transverse_ridge'
    # Signal dominates cost; length and curvature discourage distant/indirect routes.
    score=float(np.mean(1/(1+snr))+.10*length/18+.25*(length/gap-1))
    return dict(components=sorted([a,b]),source=a,target=b,kind=kind,gap=gap,path_zyx=(path+lo).tolist(),signal_fraction=float(supported.mean()),noise_sigma=sigma,transverse_ridge_fraction=ridge_fraction,reinforced_fraction=float(reinforced.mean()),raw_signal_fraction=float(np.mean((observed_values>2)&(observed_values/sigma>2))),layers=layers,max_dark_run=dark,length=length,cost=score),None

def main():
    O.mkdir(parents=True,exist_ok=True)
    global RAW_CONTRAST,BACKMAP_STATS
    raw=tifffile.imread(RAW_PATH);mask=tifffile.imread(MASK_PATH)>0
    lab,n=ndi.label(mask,np.ones((3,3,3)));audit=json.loads((BASE/'report.json').read_text());records=audit['fragment_audit'];by={x['component']:x for x in records}
    contrast=ndi.gaussian_filter(raw.astype('float32'),(0,.6,.6))-ndi.gaussian_filter(raw.astype('float32'),(0,4,4))
    RAW_CONTRAST=contrast.copy()
    contrast,classes,mip,zmax,BACKMAP_STATS=backmap(raw,contrast)
    tifffile.imwrite(O/'xy-mip.tif',mip)
    tifffile.imwrite(O/'mip-z-argmax.tif',zmax.astype('uint16'))
    tifffile.imwrite(O/'backmapped-signal-labels.tif',classes,compression='zlib')
    tips=[(x['component'],np.array(t['point_zyx']),np.array(t['direction_zyx'])) for x in records for t in x['endpoint_directions']]
    soma,bodies=soma_candidates(raw,lab,tips);soma_ids={x['component'] for x in bodies}
    body_centers={x['component']:x['center_zyx'] for x in bodies}
    boxes=ndi.find_objects(lab);surfaces={};trees={}
    for cid,box in enumerate(boxes,1):
        region=lab[box]==cid;surface=region&~ndi.binary_erosion(region);pts=np.argwhere(surface)+np.array([s.start for s in box]);surfaces[cid]=pts;trees[cid]=cKDTree(pts)
    sk=tifffile.imread(BASE/'all-fragment-skeletons.tif')>0
    skeletons={cid:np.argwhere(sk[box]&(lab[box]==cid))+np.array([s.start for s in box]) for cid,box in enumerate(boxes,1)}
    rejected=[];proposals=[]
    def evaluate(p,q,a,b,kind):
        e,reason=path_evidence(raw,contrast,lab,p,q,a,b,kind)
        if reason:
            guided_reason=reason
            e,reason=path_evidence(raw,RAW_CONTRAST,lab,p,q,a,b,kind)
            if e:
                e['route_source']='raw_fallback'
                e['guided_rejection']=guided_reason
        elif e:
            e['route_source']='mip_guided'
        if reason:rejected.append(dict(source=a,target=b,kind=kind,reason=reason))
        return e
    # Keep directional candidates from the prior strict pass, but revalidate signal strength.
    for c in audit['accepted']:
        a,b=c['components'];p=np.array(c['path_zyx']);e=evaluate(p[0],p[-1],a,b,'neurite_to_neurite')
        if e:proposals.append(e)
    seen=set()
    for c in soma:
        p=np.array(c['metrics']['path_zyx'],int);a=int(lab[tuple(p[0])]);b=int(lab[tuple(p[-1])]);key=(a,b,tuple(p[0]))
        if key in seen:continue
        seen.add(key);e=evaluate(p[0],p[-1],a,b,'neurite_to_soma')
        if e:proposals.append(e)
    small_audit=[]
    for rec in records:
        a=rec['component']
        if rec['direction_candidates'] or a in soma_ids:continue
        # A compact large blob is not an uncertain tiny neurite fragment.
        if rec['voxels']>32:
            small_audit.append(dict(fragment=a,status='large_undirected_blob'));continue
        candidates=[]
        for b in surfaces:
            # Do not bootstrap a chain of ambiguous small fragments.
            if b==a or (not by[b]['direction_candidates'] and b not in soma_ids):continue
            ds,ix=trees[b].query(surfaces[a]);j=int(np.argmin(ds));gap=float(ds[j])
            if gap<=CFG['small_max_gap']:candidates.append((gap,b,surfaces[a][j],surfaces[b][ix[j]]))
        ranked=[]
        for gap,b,p,q in sorted(candidates,key=lambda x:x[0])[:CFG['small_neighbors']]:
            # For neurites, avoid lateral attachments inconsistent with the local tangent.
            if b not in soma_ids:
                pts=skeletons[b];cloud=pts[np.linalg.norm(pts-q,axis=1)<=6]
                if len(cloud)>=3:
                    eig,v=np.linalg.eigh(np.cov(cloud,rowvar=False));d=(p-q)/max(gap,1e-9)
                    if eig[-1]>2*max(eig[-2],1e-9) and abs(v[:,-1]@d)<.5:
                        rejected.append(dict(source=a,target=b,kind='small_fragment',reason='sideways_neurite_attachment'));continue
            e=evaluate(p,q,a,b,'small_fragment')
            if e:ranked.append(e)
        ranked.sort(key=lambda e:e['cost'])
        row=dict(fragment=a,neighbors_considered=[int(x[1]) for x in sorted(candidates,key=lambda x:x[0])[:CFG['small_neighbors']]],scores=[dict(target=e['target'],cost=e['cost']) for e in ranked])
        if not ranked:row['status']='no_supported_target'
        elif not unambiguous([e['cost'] for e in ranked]):row['status']='ambiguous_best_two'
        else:row['status']='proposed';proposals.append(ranked[0])
        small_audit.append(row)
    parent=list(range(n+1));root_somas={i:({i} if i in soma_ids else set()) for i in range(1,n+1)}
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    added=np.zeros(mask.shape,bool);accepted=[];used_small=set()
    # First anchor directional neurites, then resolve uncertain fragments.
    for e in sorted(proposals,key=lambda e:(e['kind']=='small_fragment',e['cost'])):
        a,b=e['components'];ra,rb=root(a),root(b);reason=None
        if ra==rb:reason='cycle'
        elif root_somas[ra] and root_somas[rb]:reason='would_merge_distinct_soma_components'
        elif e['kind']=='small_fragment' and e['source'] in used_small:reason='small_fragment_already_attached'
        coords=solid_bridge(e['path_zyx'],mask.shape);lo=np.maximum(0,coords.min(0)-1);hi=np.minimum(mask.shape,coords.max(0)+2);sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi));bridge=np.zeros(hi-lo,bool);bridge[tuple((coords-lo).T)]=True;halo=ndi.binary_dilation(bridge,structure=np.ones((3,3,3)))
        if set(np.unique(lab[sl][halo]))-{0}!={a,b}:reason=reason or 'touches_third_fragment'
        if np.any(added[sl]&halo):reason=reason or 'touches_prior_bridge'
        if reason:rejected.append(dict(source=e['source'],target=e['target'],kind=e['kind'],reason=reason));continue
        assert ndi.label(bridge)[1]==1
        fresh=~mask[tuple(coords.T)];added[tuple(coords[fresh].T)]=True;parent[ra]=rb;root_somas[rb]|=root_somas[ra]
        if e['kind']=='small_fragment':used_small.add(e['source'])
        e['added_voxels']=int(fresh.sum());accepted.append(e)
    accepted_small={e['source']:e['target'] for e in accepted if e['kind']=='small_fragment'}
    for row in small_audit:
        if row['status']=='proposed':
            row['status']='accepted' if row['fragment'] in accepted_small else 'blocked_by_topology_or_collision'
    connected=mask|added;after=ndi.label(connected,np.ones((3,3,3)))[1];assert after==n-len(accepted)
    faceslab=ndi.label(connected)[0]
    for e in accepted:
        p=np.array(e['path_zyx']);assert len(np.unique(faceslab[tuple(p.T)]))==1
    for name,a in [('connected-mask',connected),('bridges-only',added)]:tifffile.imwrite(O/(name+'.tif'),a.astype('uint8')*255,compression='zlib')
    report=dict(config=CFG,backmap=BACKMAP_STATS,soma_regions=bodies,soma_component_ids=sorted(soma_ids),fragments_before=n,fragments_after=after,added_voxels=int(added.sum()),accepted=accepted,small_fragment_audit=small_audit,rejected=rejected,limitations='Soma-like thick cores are heuristic; voxel spacing uncalibrated; minimum bridge thickness is not measured neurite thickness. Cost is not a probability. Existing mixed-neuron components cannot be separated by this trial.')
    (O/'report.json').write_text(json.dumps(report,indent=2));print('RESULT',n,after,len(accepted),int(added.sum()),flush=True)
    print('KINDS',{k:sum(e['kind']==k for e in accepted) for k in ['neurite_to_neurite','neurite_to_soma','small_fragment']},flush=True)

if __name__=='__main__':main()
