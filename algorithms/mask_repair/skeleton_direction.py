import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import sys,json
from pathlib import Path
import os
RAW_PATH=os.environ.get('RECONNECT_RAW','')
MASK_PATH=os.environ.get('RECONNECT_MASK','')
WORK=Path(os.environ.get('RECONNECT_WORK','.'))

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from skimage.measure import marching_cubes

from algorithms.fragment_geometry import robust
sys.path.insert(0,str(Path(__file__).parent))
from algorithms.mask_repair.soma_bridge import solid_bridge
O=WORK/'a';O.mkdir(parents=True,exist_ok=True)
raw=tifffile.imread(RAW_PATH);mask=tifffile.imread(MASK_PATH)>0
labels,n=ndi.label(mask,np.ones((3,3,3)));sk=np.zeros(mask.shape,bool);tips=[];records=[]
for cid,box in enumerate(ndi.find_objects(labels),1):
 local=labels[box]==cid; origin=np.array([s.start for s in box]);s=skeletonize(local)
 fallback=False
 if not s.any():
  # Lee thinning can erase symmetric tiny/thick components. Retain a geodesic centerline.
  pts=np.argwhere(local);fallback=True
  if len(pts)>1:
   _,vec=np.linalg.eigh(np.cov(pts.astype(float),rowvar=False));v=vec[:,-1];proj=pts@v;start=pts[proj.argmin()];end=pts[proj.argmax()]
   dt=ndi.distance_transform_edt(np.pad(local,1))[1:-1,1:-1,1:-1];cost=np.where(local,1/(dt+.1),np.inf);path,_=route_through_array(cost,tuple(start),tuple(end),fully_connected=True);s[tuple(np.array(path).T)]=True
  else:s[tuple(pts[0])]=True
 sk[box]|=s
 ts,_=robust.tips_for(local,origin)
 if not ts:
  pts=np.argwhere(s).astype(float)
  if len(pts)>=3:
   eig,vec=np.linalg.eigh(np.cov(pts,rowvar=False));v=vec[:,-1];proj=pts@v
   if eig[-1]>2*max(eig[-2],1e-6) and np.ptp(proj)>=3:
    ts=[(pts[proj.argmin()]+origin,-v),(pts[proj.argmax()]+origin,v)]
 tips.extend((cid,p,u) for p,u in ts)
 records.append(dict(component=cid,voxels=int(local.sum()),skeleton_voxels=int(s.sum()),direction_candidates=len(ts),geodesic_skeleton_fallback=fallback,endpoint_directions=[dict(point_zyx=p.tolist(),direction_zyx=u.tolist()) for p,u in ts]))
tifffile.imwrite(O/'all-fragment-skeletons.tif',sk.astype('uint8')*255,compression='zlib')
# Each Z plane has its own local background; do not use a full-volume MIP.
f=raw.astype('float32');contrast=ndi.gaussian_filter(f,(0,.6,.6))-ndi.gaussian_filter(f,(0,4,4)); del f
pairs=cKDTree([t[1] for t in tips]).query_pairs(40) if tips else set();best={};counts={}
def reject(reason):counts[reason]=counts.get(reason,0)+1
for i,j in sorted(pairs):
 a,p,u=tips[i];b,q,w=tips[j]
 if a==b:continue
 d=q-p;gap=np.linalg.norm(d);d/=gap
 if abs(u@w)<np.cos(np.deg2rad(35)) or u@d<.5 or w@(-d)<.5:
  reject('fragment_direction');continue
 key=tuple(sorted((a,b)))
 if key in best and best[key]['gap']<=gap:continue
 lo=np.maximum(0,np.minimum(p,q).astype(int)-7);hi=np.minimum(raw.shape,np.maximum(p,q).astype(int)+8);sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi));c=contrast[sl];l=labels[sl]
 start=np.rint(p-lo).astype(int);end=np.rint(q-lo).astype(int);line=np.ones(c.shape,bool);ix=np.rint(np.linspace(start,end,int(gap*3)+2)).astype(int);line[tuple(ix.T)]=False
 allowed=(ndi.distance_transform_edt(line)<=5)&((l==0)|(l==a)|(l==b));cost=1+20/(1+np.maximum(c,0));cost[~allowed]=np.inf
 try:
  path,total=route_through_array(cost,tuple(start),tuple(end),fully_connected=True)
 except ValueError:reject("no_path");continue
 if not np.isfinite(total):reject('no_path');continue
 path=np.array(path);in_gap=l[tuple(path.T)]==0;g=path[in_gap]
 if not len(g):continue
 if np.linalg.norm(np.diff(path,axis=0),axis=1).sum()>1.6*gap:reject('detour');continue
 # Every traversed Z layer must have strong local signal along the gap.
 layers=[]
 for z in np.unique(g[:,0]):
  pts=g[g[:,0]==z];vals=c[tuple(pts.T)];layers.append(dict(z=int(z+lo[0]),fraction=float(np.mean(vals>2)),mean_contrast=float(vals.mean())))
 if min(x['fraction'] for x in layers)<.6 or np.mean(c[tuple(g.T)]>2)<.85:reject('z_layer_signal');continue
 # Skeleton of the actual strong-signal corridor, with local PCA direction checks.
 signal=(c>2)&allowed;signal_sk=np.argwhere(skeletonize(signal))
 if len(signal_sk)<3:reject('signal_skeleton');continue
 tree=cKDTree(signal_sk);alignment=[]
 for point in g:
  ids=tree.query_ball_point(point,4)
  if len(ids)<3:continue
  cloud=signal_sk[ids].astype(float);eig,vec=np.linalg.eigh(np.cov(cloud,rowvar=False))
  if eig[-1]<=1.5*max(eig[-2],1e-6):continue
  axis=vec[:,-1];alignment.append(min(abs(axis@u),abs(axis@w)))
 if len(alignment)<max(1,int(.5*len(g))) or np.mean(np.array(alignment)>=np.cos(np.deg2rad(45)))<.7:reject('signal_skeleton_direction');continue
 best[key]=dict(components=list(key),gap=float(gap),path_zyx=(path+lo).tolist(),layers=layers,orientation_support=float(np.mean(np.array(alignment)>=np.cos(np.deg2rad(45)))))
print('signal-supported pairs',len(best),'rejections',counts,flush=True)
parent=list(range(n+1))
def root(x):
 while parent[x]!=x:x=parent[x]
 return x
added=np.zeros(mask.shape,bool);accepted=[]
for e in sorted(best.values(),key=lambda x:x['gap']):
 a,b=e['components']
 if root(a)==root(b):continue
 coords=solid_bridge(e['path_zyx'],mask.shape);lo=np.maximum(0,coords.min(0)-1);hi=np.minimum(mask.shape,coords.max(0)+2);sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi));bridge=np.zeros(hi-lo,bool);bridge[tuple((coords-lo).T)]=True;halo=ndi.binary_dilation(bridge,structure=np.ones((3,3,3)))
 if set(np.unique(labels[sl][halo]))-{0}!={a,b} or np.any(added[sl]&halo):continue
 assert ndi.label(bridge)[1]==1
 fresh=~mask[tuple(coords.T)];added[tuple(coords[fresh].T)]=True;parent[root(a)]=root(b);e['added_voxels']=int(fresh.sum());accepted.append(e)
connected=mask|added;after=ndi.label(connected,np.ones((3,3,3)))[1];assert after==n-len(accepted)
for name,a in [('connected-mask',connected),('bridges-only',added)]:tifffile.imwrite(O/(name+'.tif'),a.astype('uint8')*255,compression='zlib')
r=dict(fragments=n,fragments_after=after,skeletonized_fragments=len(records),accepted=accepted,fragment_audit=records,rejection_counts=counts,added_voxels=int(added.sum()),method='per-Z local contrast + gap signal skeleton direction; voxel units; minimum thickness, not measured radius')
(O/'report.json').write_text(json.dumps(r,indent=2));print('RESULT',n,after,len(accepted),int(added.sum()),flush=True)
