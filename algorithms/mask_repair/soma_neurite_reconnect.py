"""Reconnect disjoint fragments to explicitly segmented somas in repaired v5."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

from pathlib import Path
import os
RAW_PATH=os.environ.get('RECONNECT_RAW','')
MASK_PATH=os.environ.get('RECONNECT_MASK','')
WORK=Path(os.environ.get('RECONNECT_WORK','.'))

import sys,json
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import tifffile
from skimage.measure import marching_cubes

from algorithms.fragment_geometry import robust
from algorithms.mask_repair import signal_reconnect as evidence
from algorithms.mask_repair.mip_backmap import backmap
from algorithms.mask_repair.soma_bridge import solid_bridge
O=WORK/'d';O.mkdir(parents=True,exist_ok=True)
BASE=WORK/'c'

def main():
 raw=tifffile.imread(RAW_PATH);mask=tifffile.imread(BASE/'connected-mask.tif')>0;somas=tifffile.imread(BASE/'soma-only.tif')>0
 labels,n=ndi.label(mask,np.ones((3,3,3)));soma_labels,ns=ndi.label(somas,np.ones((3,3,3)));sizes=np.bincount(labels.ravel());records=[];soma_components=set()
 for sid,box in enumerate(ndi.find_objects(soma_labels),1):
  body=soma_labels[box]==sid;origin=np.array([s.start for s in box]);coords=np.argwhere(body)+origin;cid=int(labels[tuple(coords[0])]);soma_components.add(cid)
  # Use nearby existing foreground as part of the soma target; conservative core
  # segmentation must not artificially move a real soma surface inward.
  lo=np.maximum(0,origin-6);hi=np.minimum(mask.shape,np.array([s.stop for s in box])+6)
  roi=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));core=soma_labels[roi]==sid
  local=labels[roi]==cid;thick=ndi.distance_transform_edt(np.pad(local,1))[1:-1,1:-1,1:-1]>=2
  target=core|(local&thick&(ndi.distance_transform_edt(~core)<=6))
  surf=np.argwhere(target&~ndi.binary_erosion(target))+lo
  records.append(dict(soma_id=sid,component=cid,center_xyz=coords.mean(0)[::-1].tolist(),surface=surf,tree=cKDTree(surf)))
 all_tips={};surfaces={};trees={};audit=[]
 for cid,box in enumerate(ndi.find_objects(labels),1):
  local=labels[box]==cid;origin=np.array([s.start for s in box]);surf=np.argwhere(local&~ndi.binary_erosion(local))+origin;surfaces[cid]=surf;trees[cid]=cKDTree(surf)
  if cid in soma_components:continue
  tips,skn=robust.tips_for(local,origin);all_tips[cid]=tips;audit.append(dict(component=cid,voxels=int(sizes[cid]),endpoint_direction_candidates=len(tips),skeleton_voxels=skn))
 c=ndi.gaussian_filter(raw.astype('float32'),(0,.6,.6))-ndi.gaussian_filter(raw.astype('float32'),(0,4,4));evidence.RAW_CONTRAST=c;guide,_,_,_,mapstats=backmap(raw,c)
 rejected=[];best={}
 def evaluate(p,q,a,b,kind):
  e,reason=evidence.path_evidence(raw,guide,labels,p,q,a,b,kind)
  if reason:
   e,reason=evidence.path_evidence(raw,c,labels,p,q,a,b,kind)
   if e:e['route_source']='raw_fallback'
  elif e:e['route_source']='mip_guided'
  return e,reason
 for rec in records:
  b=rec['component'];sid=rec['soma_id']
  for a,tips in all_tips.items():
   # Original repaired-v5 soma components are never merged with each other.
   if a in soma_components:continue
   options=[]
   for p,u in tips:
    gap,j=rec['tree'].query(p)
    if not 1<=gap<=50:continue
    q=rec['surface'][j];angle=float(np.degrees(np.arccos(np.clip(u@((q-p)/gap),-1,1))))
    if angle<=60:options.append((p,q,angle))
   if not tips and sizes[a]<=32:
    ds,js=rec['tree'].query(surfaces[a]);i=int(np.argmin(ds))
    if 1<=ds[i]<=18:options.append((surfaces[a][i],rec['surface'][js[i]],None))
   for p,q,angle in options:
    key=(a,sid);e,why=evaluate(p,q,a,b,'fragment_to_segmented_soma')
    if why:rejected.append(dict(component=a,soma_id=sid,reason=why));continue
    e.update(soma_id=sid,source_angle=angle,small_undirected=angle is None)
    if key not in best or e['cost']<best[key]['cost']:best[key]=e
 ranked={}
 for e in best.values():ranked.setdefault(e['source'],[]).append(e)
 proposals=[];competition=[]
 for a,es in ranked.items():
  es.sort(key=lambda e:e['cost']);winner=es[0];scores=[dict(soma_id=e['soma_id'],target=e['target'],cost=e['cost']) for e in es]
  if len(es)>1 and not evidence.unambiguous([e['cost'] for e in es]):
   competition.append(dict(component=a,candidates=scores,status='ambiguous_soma_targets'));continue
  # Tiny undirected fragments must prefer this soma over nearby non-soma alternatives.
  alternatives=[]
  if winner['small_undirected']:
   neighbors=[]
   for b in surfaces:
    if b==a or b in soma_components or sizes[b]<=32:continue
    ds,js=trees[b].query(surfaces[a]);i=int(np.argmin(ds))
    if 1<=ds[i]<=18:neighbors.append((float(ds[i]),b,surfaces[a][i],surfaces[b][js[i]]))
   for _,b,p,q in sorted(neighbors,key=lambda x:x[0])[:5]:
    e,why=evaluate(p,q,a,b,'competing_non_soma')
    if e:alternatives.append(dict(target=b,cost=e['cost']))
   if alternatives and winner['cost']>.85*min(x['cost'] for x in alternatives):
    competition.append(dict(component=a,candidates=scores,non_soma_alternatives=alternatives,status='non_soma_competition'));continue
  competition.append(dict(component=a,candidates=scores,non_soma_alternatives=alternatives,status='proposed'));proposals.append(winner)
 parent=list(range(n+1))
 def root(i):
  while parent[i]!=i:i=parent[i]
  return i
 root_somas={i:({i} if i in soma_components else set()) for i in range(1,n+1)};added=np.zeros(mask.shape,bool);accepted=[]
 for e in sorted(proposals,key=lambda e:e['cost']):
  a,b=e['source'],e['target'];ra,rb=root(a),root(b)
  if ra==rb or (root_somas[ra] and root_somas[rb]):rejected.append(dict(component=a,soma_id=e['soma_id'],reason='cycle_or_two_somas'));continue
  coords=solid_bridge(e['path_zyx'],mask.shape);lo=np.maximum(0,coords.min(0)-1);hi=np.minimum(mask.shape,coords.max(0)+2);sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi));bridge=np.zeros(hi-lo,bool);bridge[tuple((coords-lo).T)]=True;halo=ndi.binary_dilation(bridge,structure=np.ones((3,3,3)))
  if set(np.unique(labels[sl][halo]))-{0}!={a,b} or np.any(added[sl]&halo):rejected.append(dict(component=a,soma_id=e['soma_id'],reason='third_fragment_or_bridge_collision'));continue
  assert ndi.label(bridge)[1]==1
  fresh=~mask[tuple(coords.T)];added[tuple(coords[fresh].T)]=True;parent[ra]=rb;root_somas[rb]|=root_somas[ra];e['added_voxels']=int(fresh.sum());accepted.append(e)
 connected=mask|added;lab2,n2=ndi.label(connected,np.ones((3,3,3)));assert n2==n-len(accepted);face=ndi.label(connected)[0]
 for e in accepted:
  p=np.asarray(e['path_zyx']);assert len(np.unique(face[tuple(p.T)]))==1
 soma_roots=[int(lab2[tuple(surfaces[cid][0])]) for cid in soma_components];assert len(set(soma_roots))==len(soma_roots)
 for name,a in [('connected-mask',connected),('new-soma-bridges-only',added)]:tifffile.imwrite(O/(name+'.tif'),a.astype('uint8')*255,compression='zlib')
 report=dict(base=str(BASE/'connected-mask.tif'),components_before=n,components_after=n2,new_connections=len(accepted),new_voxels=int(added.sum()),accepted=accepted,somas=[{k:v for k,v in r.items() if k not in ['surface','tree']} for r in records],fragment_audit=audit,competition=competition,rejected=rejected,limits='Same-component internal gaps are skipped. Source direction 60 degrees; 50-voxel endpoint search (compact soma surface); 18-voxel small-fragment search. Voxel units, minimal bridge thickness; not biological validation.')
 report['backmap']=mapstats
 (O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['components_before','components_after','new_connections','new_voxels']}),flush=True)
 for e in accepted:print(e['source'],'-> soma',e['soma_id'],'gap',round(e['gap'],1),'cost',round(e['cost'],3),flush=True)

if __name__=='__main__':main()
