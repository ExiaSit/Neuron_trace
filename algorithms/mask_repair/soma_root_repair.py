"""Soma-root completion with slice-local background and measured XY boundaries.
Reassess generated roots from stage e; never unconditionally retain older roots.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import os,json
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
S=np.ones((3,3,3),bool)

def boundary_evidence(signal,region,background,noise):
 """Use raw-image profiles normal to a contour, not the contour shape itself."""
 boundary=region&~ndi.binary_erosion(region,np.ones((3,3)))
 # Exclude crop edges where an outward measurement is unavailable.
 boundary[:3]=False;boundary[-3:]=False;boundary[:,:3]=False;boundary[:,-3:]=False
 points=np.argwhere(boundary)
 if len(points)<12:return 0.,0.,len(points)
 signed=ndi.distance_transform_edt(region)-ndi.distance_transform_edt(~region)
 gy,gx=np.gradient(ndi.gaussian_filter(signed,.8));normal=np.stack([gy[tuple(points.T)],gx[tuple(points.T)]],axis=1)
 normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-6)
 inside=ndi.map_coordinates(signal,(points+2*normal).T,order=1,mode='nearest')
 outside=ndi.map_coordinates(signal,(points-2*normal).T,order=1,mode='nearest')
 drop=inside-outside;relative=drop/np.maximum(inside-background,10)
 supported=(relative>=.55)&(drop>=max(4,2.5*noise))
 return float(np.mean(supported)),float(np.median(relative)),len(points)

def separate_additions(mask,added):
 """Keep source components distinct; cut only new contact boundaries."""
 labels,n=ndi.label(mask,S);out,nout=ndi.label(mask|added,S);rollback=np.zeros_like(mask)
 if nout!=n:
  groups={}
  for cid,box in enumerate(ndi.find_objects(labels),1):
   point=tuple(np.argwhere(labels[box]==cid)[0]);root=int(out[box][point]);groups.setdefault(root,[]).append(cid)
  boxes=ndi.find_objects(out)
  for root in [k for k,v in groups.items() if len(v)>1]:
   box=boxes[root-1];region=out[box]==root
   _,ix=ndi.distance_transform_edt(~mask[box],return_indices=True);owner=labels[box][tuple(ix)]
   high=ndi.maximum_filter(np.where(region,owner,0),size=3,mode='constant',cval=0)
   low=ndi.minimum_filter(np.where(region,owner,n+1),size=3,mode='constant',cval=n+1)
   cut=region&(high!=low)&added[box];rollback[box]|=cut;added[box]&=~cut
 reached=ndi.binary_propagation(mask,structure=S,mask=mask|added)
 rollback|=added&~reached;added&=reached
 assert ndi.label(mask|added,S)[1]==n
 return added,int(rollback.sum()),n

def restore_roots(raw,mask,previous_result=None):
 # previous_result is accepted only for API compatibility; never used as a seed.
 labels,_=ndi.label(mask,S);thick=ndi.distance_transform_edt(mask)>=2.5
 cores,_=ndi.label(thick,S);added=np.zeros_like(mask);audit=[]
 for cid,box in enumerate(ndi.find_objects(cores),1):
  pts=np.argwhere(cores[box]==cid)
  if len(pts)<4:continue
  ext=np.array([s.stop-s.start for s in box]);eig=np.linalg.eigvalsh(np.cov(pts.astype(float),rowvar=False));aspect=np.sqrt(eig[-1]/max(eig[0],1e-6))
  if (len(pts)>=12 and aspect>4) or max(ext)>40:continue
  lo=np.maximum(0,np.array([s.start for s in box])-26);hi=np.minimum(raw.shape,np.array([s.stop for s in box])+26)
  sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));seed=cores[sl]==cid
  if seed.sum()<12:seed=ndi.binary_dilation(seed,S)&mask[sl]
  patch=raw[sl].astype('float32');sig=ndi.gaussian_filter(patch,.8)
  projection=seed.any(0);dxy=ndi.distance_transform_edt(~projection);ring=(dxy>=8)&(dxy<=16)
  if ring.sum()<50:continue
  residual=patch-sig
  bg=np.median(sig[:,ring],axis=1)
  far_ring=(dxy>=20)&(dxy<=25)
  far_bg=np.median(sig[:,far_ring],axis=1) if far_ring.sum()>=50 else bg
  noise=1.4826*np.median(np.abs(residual[:,ring]-np.median(residual[:,ring],axis=1)[:,None]),axis=1)
  # Fixed XY anchor samples the same structure through Z, while background is
  # measured in each optical plane instead of averaged with empty planes.
  peak=np.percentile(sig[:,projection],80,axis=1);amp=np.maximum(peak-bg,0)
  valid=amp>=np.maximum(20,4*noise)
  zseed=np.flatnonzero(seed.any((1,2)));valid[:max(0,zseed[0]-4)]=False;valid[zseed[-1]+5:]=False
  high=bg+np.maximum.reduce([np.full_like(bg,12),4*noise,.35*amp])
  low=bg+np.maximum.reduce([np.full_like(bg,8),3*noise,.18*amp])
  distance=ndi.distance_transform_edt(~seed)
  component=int(labels[sl][seed][0]);own=labels[sl]==component;other=(labels[sl]>0)&~own
  guard=~ndi.binary_dilation(other,S,iterations=2)
  allowed=valid[:,None,None]&(distance<=20)&(sig>=high[:,None,None])&(patch>=high[:,None,None])
  strong=ndi.binary_propagation(seed&allowed,structure=S,mask=allowed)
  interior=ndi.binary_erosion(strong,S)
  center=(np.mean(np.argwhere(seed),axis=0)+lo)[::-1].tolist()
  rec=dict(center_xyz=center,core_voxels=int(seed.sum()))
  denom=np.maximum(amp[:,None,None],20)
  roughness=float(np.mean((np.abs(patch-sig)/denom)[interior])) if interior.any() else 1
  rec['roughness']=roughness
  if roughness>.055:
   audit.append(dict(rec,status='granular_or_unsupported_core'));continue
  edge=valid[:,None,None]&(distance<=26)&(sig>=low[:,None,None])&(patch>=low[:,None,None])
  edge&=ndi.gaussian_filter(sig,(0,1.2,1.2))>=low[:,None,None]
  edge&=ndi.distance_transform_edt(~strong)<=6
  edge&=ndi.distance_transform_edt(~own)<=18
  candidate=ndi.binary_propagation(strong,structure=S,mask=edge|strong)
  # Require a measured lateral boundary. Axially thin sheets and saturated halos
  # can have a sharp Z falloff but remain diffuse within the optical plane.
  planes=[];supported=np.zeros_like(candidate)
  for z in np.flatnonzero(candidate.sum((1,2))>=25):
   fraction,drop,count=boundary_evidence(sig[z],candidate[z],float(min(bg[z],far_bg[z])),float(noise[z]))
   ok=fraction>=.55
   planes.append(dict(z=int(z+lo[0]),background=float(bg[z]),noise=float(noise[z]),peak=float(peak[z]),threshold=float(low[z]),boundary_supported_fraction=fraction,median_boundary_drop=drop,accepted=ok))
   if ok:supported[z]=candidate[z]
  grown=ndi.binary_propagation(seed&supported&guard,structure=S,mask=supported&guard)
  fresh=grown&~mask[sl]&~added[sl]
  rec['planes']=planes
  if fresh.sum()>max(7000,6*np.count_nonzero(own&(distance<=6))):
   audit.append(dict(rec,status='excessive_growth',proposed_voxels=int(fresh.sum())));continue
  added[sl]|=fresh
  audit.append(dict(rec,status='grown' if fresh.any() else 'no_sharp_supported_boundary',added_voxels=int(fresh.sum())))
 added,rollback,n=separate_additions(mask,added);final=mask|added
 assert np.all(final[mask])
 return final,added,dict(anchors=audit,added_voxels=int(added.sum()),rollback_voxels=rollback,components_before=n,components_after=n,original_preserved=True,limits='Slice-local background/noise and XY boundary evidence. Reassesses older generated roots; preserves stage-e mask. Voxel-space uint8 heuristic; ambiguous or blurred real roots may be withheld.')

def main():
 work=Path(os.environ['RECONNECT_WORK']);raw=tifffile.imread(os.environ['RECONNECT_RAW']);mask=tifffile.imread(work/'e/connected-mask.tif')>0
 final,added,report=restore_roots(raw,mask);out=work/'f';out.mkdir(exist_ok=True)
 for name,v in [('connected-mask',final),('roots-added-only',added)]:tifffile.imwrite(out/(name+'.tif'),v.astype('uint8')*255,compression='zlib')
 (out/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
