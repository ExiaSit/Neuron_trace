import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
CFG=dict(max_gap_voxels=120,source_angle_degrees=30,parallel_angle_degrees=30,target_facing_degrees=40,ray_miss_floor=3,ray_miss_fraction=.15,tangent_path_voxels=10,min_tangent_displacement=4,connectivity=26,central_xy_fraction=[.25,.75],new_bridge_voxels=False)

def tips_for(mask,origin):
 sk=skeletonize(mask);points=np.argwhere(sk);lut={tuple(p):j for j,p in enumerate(points)};adj=[[] for _ in points]
 offsets=[(a,b,c) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1) if (a,b,c)!=(0,0,0)]
 for j,p in enumerate(points):
  for off in offsets:
   k=lut.get(tuple(p+off))
   if k is not None:adj[j].append(k)
 result=[]
 for j,neighbors in enumerate(adj):
  if len(neighbors)!=1:continue
  prev=j;cur=neighbors[0];length=float(np.linalg.norm(points[cur]-points[j]));seen={j,cur}
  while length<CFG['tangent_path_voxels']:
   opts=[q for q in adj[cur] if q!=prev and q not in seen]
   if len(opts)!=1:break
   nxt=opts[0];length+=float(np.linalg.norm(points[nxt]-points[cur]));prev,cur=cur,nxt;seen.add(cur)
  direction=points[j].astype(float)-points[cur];norm=np.linalg.norm(direction)
  if norm>=CFG['min_tangent_displacement']:result.append((points[j].astype(float)+origin,direction/norm))
 return result,len(points)

def evaluate(p,u,q,w):
 vec=q-p;gap=float(np.linalg.norm(vec))
 if gap<=0:return None
 d=vec/gap;forward=float(u@d);parallel=float(abs(u@w));facing=float(w@(-d));miss=gap*np.sqrt(max(0,1-forward*forward))
 metrics=dict(gap=gap,source_angle=float(np.degrees(np.arccos(np.clip(forward,-1,1)))),parallel_angle=float(np.degrees(np.arccos(np.clip(parallel,-1,1)))),target_facing_angle=float(np.degrees(np.arccos(np.clip(facing,-1,1)))),ray_miss=float(miss),ray_tolerance=max(CFG['ray_miss_floor'],gap*CFG['ray_miss_fraction']))
 reasons=[]
 if gap>CFG['max_gap_voxels']:reasons.append('beyond_search_distance')
 if metrics['source_angle']>CFG['source_angle_degrees']:reasons.append('subject_not_extending_towards_fragment')
 if metrics['parallel_angle']>CFG['parallel_angle_degrees']:reasons.append('centerline_not_parallel')
 if metrics['target_facing_angle']>CFG['target_facing_degrees']:reasons.append('fragment_endpoint_not_facing_subject')
 if miss>metrics['ray_tolerance']:reasons.append('extension_ray_misses_fragment')
 return metrics,reasons
