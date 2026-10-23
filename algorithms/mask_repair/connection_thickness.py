"""Restore signal-supported volume around accepted paths without new component mergers."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import os,json
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi

def restore(raw,mask,paths):
 labels,n=ndi.label(mask,np.ones((3,3,3)));added=np.zeros_like(mask);audit=[]
 for path in paths:
  path=np.asarray(path,int)
  lo=np.maximum(0,path.min(0)-9);hi=np.minimum(raw.shape,path.max(0)+10)
  sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));p=path-lo;sig=ndi.gaussian_filter(raw[sl].astype(float),.6)
  line=np.zeros(sig.shape,bool);line[tuple(p.T)]=True
  distance,ix=ndi.distance_transform_edt(~line,return_indices=True)
  # Local transverse profile separates an actual shaft from broad background.
  tangent=np.gradient(p.astype(float),axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
  ref=np.zeros_like(tangent);ref[:,0]=1;ref[np.abs(tangent[:,0])>.9]=[0,1,0]
  u=np.cross(tangent,ref);u/=np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-9);v=np.cross(tangent,u)
  ring=[ndi.map_coordinates(sig,(p+8*(np.cos(a)*u+np.sin(a)*v)).T,order=1,mode='nearest') for a in np.linspace(0,2*np.pi,12,endpoint=False)]
  bg=np.median(ring,axis=0);peak=sig[tuple(p.T)];threshold=bg+np.maximum(3,.45*(peak-bg))
  threshold_map=np.zeros(sig.shape,float);threshold_map[tuple(p.T)]=threshold;threshold_map=threshold_map[tuple(ix)]
  allowed=(distance<=6)&(sig>=threshold_map)&(raw[sl]>=threshold_map)
  ids=set(np.unique(labels[sl][line]))-{0}
  if len(ids)!=1:
   audit.append(dict(status='path_not_single_component'));continue
  cid=next(iter(ids));other=(labels[sl]>0)&(labels[sl]!=cid)
  # A two-voxel margin prevents independent grown paths bridging different groups.
  allowed&=~ndi.binary_dilation(other,structure=np.ones((3,3,3)),iterations=2)
  grown=ndi.binary_propagation(line&allowed,structure=np.ones((3,3,3)),mask=allowed)
  fresh=grown&~mask[sl]&~added[sl];added[sl]|=fresh
  audit.append(dict(status='grown',added_voxels=int(fresh.sum()),path_start_zyx=path[0].tolist(),path_end_zyx=path[-1].tolist(),median_profile_threshold=float(np.median(threshold))))
 # Explicit global guard against collisions between independently expanded paths.
 result=mask|added;out,nout=ndi.label(result,np.ones((3,3,3)))
 if nout!=n:
  # Reject all additions in merged output groups, preserving the validated baseline.
  groups={}
  for i,box in enumerate(ndi.find_objects(labels),1):
   point=np.argwhere(labels[box]==i)[0];root=int(out[box][tuple(point)]);groups.setdefault(root,[]).append(i)
  bad=[k for k,v in groups.items() if len(v)>1];added[np.isin(out,bad)]=False
 result=mask|added;assert ndi.label(result,np.ones((3,3,3)))[1]==n
 return result,added,audit

def main():
 work=Path(os.environ['RECONNECT_WORK']);raw=tifffile.imread(os.environ['RECONNECT_RAW']);mask=tifffile.imread(work/'d/connected-mask.tif')>0
 paths=[e['path_zyx'] for s in ['b','d'] for e in json.loads((work/s/'report.json').read_text())['accepted']]
 final,added,audit=restore(raw,mask,paths);out=work/'e';out.mkdir(exist_ok=True)
 for name,v in [('connected-mask',final),('thickness-added-only',added)]:tifffile.imwrite(out/(name+'.tif'),v.astype('uint8')*255,compression='zlib')
 (out/'report.json').write_text(json.dumps(dict(added_voxels=int(added.sum()),paths=audit,component_count_preserved=True,max_distance_voxels=6,profile_fraction=.45),indent=2))
if __name__=='__main__':main()
