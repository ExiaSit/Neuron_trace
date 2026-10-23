import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.graph import route_through_array
from algorithms.fragment_geometry.strict import tips_for as legacy_tips, CFG as STRICT_CFG
CFG=STRICT_CFG.copy()
CFG.update(ray_miss_floor=5,ray_miss_fraction=.5,tangent_path_voxels=24,tangent_scales=[8,16,24],signal_smoothing_sigma=.8,source_search="all_endpoints_within_radius",min_tangent_displacement=6,signal_fraction=.6,max_dark_run=8,signal_contrast=1.5,signal_path_tube_max=12)
RAW=None
SIGNAL_CACHE={}
def tips_for(mask,origin):
 points=np.argwhere(skeletonize(mask));n=len(points)
 if n<7:return [],n
 pairs=cKDTree(points).query_pairs(1.75,output_type='ndarray')
 if not len(pairs):return [],n
 a,b=pairs.T;weights=np.linalg.norm(points[a]-points[b],axis=1);graph=coo_matrix((np.r_[weights,weights],(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr();degree=np.diff(graph.indptr)
 endpoints=list(np.flatnonzero(degree==1))
 # Geodesic diameter adds end candidates when voxel-level loops hide strict degree-one tips.
 dist=dijkstra(graph,indices=0);finite=np.isfinite(dist);first=int(np.argmax(np.where(finite,dist,-1)));dist=dijkstra(graph,indices=first);second=int(np.argmax(np.where(np.isfinite(dist),dist,-1)));endpoints=sorted(set(endpoints+[first,second]));result=[]
 for j in endpoints:
  for scale in CFG['tangent_scales']:
   ds=dijkstra(graph,indices=j,limit=scale);good=np.isfinite(ds);k=int(np.argmax(np.where(good,ds,-1)));direction=points[j].astype(float)-points[k];norm=float(np.linalg.norm(direction))
   if norm>=6 and norm/max(ds[k],1)>.65:result.append((points[j].astype(float)+origin,direction/norm))
 previous,_=legacy_tips(mask,origin);result.extend(previous)
 return result,n

def signal_check(p,q):
 margin=14;lo=np.maximum(0,np.floor(np.minimum(p,q)).astype(int)-margin);hi=np.minimum(RAW.shape,np.ceil(np.maximum(p,q)).astype(int)+margin+1);raw=np.asarray(RAW[tuple(slice(a,b) for a,b in zip(lo,hi))],dtype=np.float32);contrast=ndi.gaussian_filter(raw,CFG['signal_smoothing_sigma'])-ndi.gaussian_filter(raw,4);contrast=ndi.maximum_filter(contrast,size=3)
 p0=p-lo;q0=q-lo;gap=np.linalg.norm(q-p);coords=np.linspace(p0,q0,max(2,int(np.ceil(gap))+1));sample=ndi.map_coordinates(contrast,coords.T,order=1,mode='nearest')
 def stats(vals):
  supported=vals>CFG['signal_contrast'];maxrun=run=0
  for b in supported:
   run=0 if b else run+1;maxrun=max(maxrun,run)
  return float(supported.mean()),maxrun
 fraction,dark=stats(sample);method='straight_corridor';path=coords
 if fraction<CFG['signal_fraction'] or dark>CFG['max_dark_run']:
  line=np.ones(raw.shape,bool);ii=np.rint(coords).astype(int);line[tuple(ii.T)]=False;distance=ndi.distance_transform_edt(line);radius=min(CFG['signal_path_tube_max'],max(4,gap*.15));cost=1+8/(1+np.maximum(contrast,0));cost[distance>radius]=np.inf
  try:
   path,costsum=route_through_array(cost,tuple(np.rint(p0).astype(int)),tuple(np.rint(q0).astype(int)),fully_connected=True,geometric=True);path=np.asarray(path,float);vals=ndi.map_coordinates(contrast,path.T,order=1);fraction,dark=stats(vals);method='raw_signal_guided_curved_corridor'
  except ValueError:pass
 length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum());passed=fraction>=CFG['signal_fraction'] and dark<=CFG['max_dark_run'] and length<=1.6*gap
 return dict(signal_fraction=fraction,max_dark_samples=dark,path_length=length,path_method=method,signal_passed=bool(passed),path_zyx=(path+lo).tolist())

def evaluate(p,u,q,w):
 vec=q-p;gap=float(np.linalg.norm(vec))
 if gap<=0:return None
 d=vec/gap;forward=float(u@d);parallel=float(abs(u@w));facing=float(w@(-d));miss=gap*np.sqrt(max(0,1-forward*forward));metrics=dict(gap=gap,source_angle=float(np.degrees(np.arccos(np.clip(forward,-1,1)))),parallel_angle=float(np.degrees(np.arccos(np.clip(parallel,-1,1)))),target_facing_angle=float(np.degrees(np.arccos(np.clip(facing,-1,1)))),ray_miss=float(miss),ray_tolerance=max(CFG['ray_miss_floor'],gap*CFG['ray_miss_fraction']));reasons=[]
 if gap>CFG['max_gap_voxels']:reasons.append('beyond_search_distance')
 if metrics['source_angle']>CFG['source_angle_degrees']:reasons.append('subject_not_extending_towards_fragment')
 if metrics['parallel_angle']>CFG['parallel_angle_degrees']:reasons.append('centerline_not_parallel')
 if metrics['target_facing_angle']>CFG['target_facing_degrees']:reasons.append('fragment_endpoint_not_facing_subject')
 if miss>metrics['ray_tolerance']:reasons.append('extension_ray_misses_fragment')
 if not reasons:
  key=tuple(p)+tuple(q)
  if key not in SIGNAL_CACHE:SIGNAL_CACHE[key]=signal_check(p,q)
  evidence=SIGNAL_CACHE[key];metrics.update(evidence)
  if not evidence['signal_passed']:reasons.append('insufficient_raw_signal_in_gap')
 return metrics,reasons
