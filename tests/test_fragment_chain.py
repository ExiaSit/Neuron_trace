import unittest,json,os,subprocess,sys,tempfile
from pathlib import Path
import tifffile
from scipy import ndimage as ndi
import numpy as np
from algorithms.mask_repair.fragment_chain import chain_fragments, unapproved_bridge_overlap
from algorithms.mask_repair.soma_bridge import solid_bridge
from algorithms.mask_repair.local_pair_mip import local_segment, segment_directions

class FragmentChainTests(unittest.TestCase):
 def setup_case(self):
  edge=dict(components=[1,2],kind='local_endpoint_mip',noise_sigma=1.,path_zyx=[[4,4,x] for x in range(2,15)])
  points={3:np.array([[4,4,7]]),4:np.array([[4,4,10],[4,4,11]])}
  members={i:{i} for i in range(1,6)}
  return edge,points,members,np.full((9,9,20),20.)
 def test_tiny_signal_fragments_join_the_chain(self):
  e,s,m,c=self.setup_case();ids,reason=chain_fragments(e,{1,2,3,4},s,set(),lambda i:i,m,c)
  self.assertEqual(ids,[3,4]);self.assertIsNone(reason)
 def test_large_fragment_soma_and_weak_signal_block(self):
  e,s,m,c=self.setup_case()
  for change in ['large','soma','weak','side']:
   ss={k:v.copy() for k,v in s.items()};cc=c.copy();bodies=set()
   if change=='large':ss[3]=np.array([[z,4,7] for z in range(9)])
   if change=='soma':bodies={3}
   if change=='weak':cc[4,4,7]=0
   if change=='side':ss[3]=np.array([[0,4,7]])
   ids,reason=chain_fragments(e,{1,2,3},ss,bodies,lambda i:i,m,cc)
   self.assertTrue(reason,change);self.assertFalse(ids)
 def test_already_attached_to_unrelated_component_blocks(self):
  e,s,m,c=self.setup_case();m[5]={3,5}
  ids,reason=chain_fragments(e,{1,2,3},s,set(),lambda i:5 if i==3 else i,m,c)
  self.assertEqual(reason,'chain_fragment_attached_elsewhere')
 def test_tiny_fragment_attached_to_endpoint_is_permitted(self):
  e,s,m,c=self.setup_case();m[1]={1,3}
  ids,reason=chain_fragments(e,{1,2,3},s,set(),lambda i:1 if i==3 else i,m,c)
  self.assertEqual(ids,[3]);self.assertIsNone(reason)
 def test_ordinary_candidate_keeps_original_third_fragment_guard(self):
  e,s,m,c=self.setup_case();e['kind']='neurite_to_neurite'
  self.assertEqual(chain_fragments(e,{1,2,3},s,set(),lambda i:i,m,c)[1],'touches_third_fragment')
 def test_only_own_small_fragment_bridge_overlap_is_allowed(self):
  shape=(9,9,20);old=dict(source=3,target=1,kind='small_fragment',path_zyx=[[4,4,x] for x in range(2,8)])
  collision=np.zeros(shape,bool);collision[4,4,5]=True;root=lambda i:1 if i==3 else i
  args=(collision,np.array([0,0,0]),[old],{3},{1,2},root,solid_bridge,shape)
  self.assertFalse(unapproved_bridge_overlap(*args))
  collision[4,7,15]=True
  self.assertTrue(unapproved_bridge_overlap(*args))
 def test_bent_endpoint_uses_shorter_reliable_tangent(self):
  points=np.array([[0,0,0],[0,1,-1],[1,2,-2],[1,2,-3],[1,2,-4],[2,3,-5],[3,2,-5],[4,1,-5],[5,0,-4]])
  segment,direction=local_segment(points,points[0],12)
  arc=np.linalg.norm(np.diff(segment,axis=0),axis=1).sum()
  self.assertLess(np.linalg.norm(segment[0]-segment[-1])/arc,.65)
  self.assertGreater(len(segment_directions(segment)),0)
  toward=np.array([-15,-19,25]);toward=toward/np.linalg.norm(toward)
  self.assertGreater(direction@toward,.5)

 def test_full_b_stage_merges_chain_and_serializes_audit(self):
  with tempfile.TemporaryDirectory() as folder:
   base=Path(folder);work=base/'mask_steps/1';(work/'a').mkdir(parents=True)
   z,y,x=np.indices((21,21,64));raw=(5+180*np.exp(-((z-10)**2+(y-10)**2)/2)).astype('uint8')
   mask=np.zeros(raw.shape,bool);mask[10,10,5:21]=True;mask[10,10,27]=True;mask[10,10,35:56]=True
   labels,n=ndi.label(mask,np.ones((3,3,3)));records=[]
   for cid in range(1,n+1):
    tips=[]
    for point,direction in [([10,10,20],[0,0,1]),([10,10,35],[0,0,-1])]:
     if labels[tuple(point)]==cid:tips.append(dict(point_zyx=point,direction_zyx=direction))
    records.append(dict(component=cid,voxels=int(np.count_nonzero(labels==cid)),direction_candidates=len(tips),endpoint_directions=tips))
   tifffile.imwrite(base/'raw.tif',raw);tifffile.imwrite(base/'mask.tif',mask.astype('uint8'))
   tifffile.imwrite(work/'a/all-fragment-skeletons.tif',mask.astype('uint8'))
   (work/'a/report.json').write_text(json.dumps(dict(fragment_audit=records,accepted=[])))
   env=dict(os.environ,RECONNECT_RAW=str(base/'raw.tif'),RECONNECT_MASK=str(base/'mask.tif'),RECONNECT_WORK=str(work),RECONNECT_COMPUTE_BACKEND='cpu',RECONNECT_LOCAL_PAIR_MIP_CONFIG='{}')
   env.pop('RECONNECT_GPU_STATS',None)
   script=Path(__file__).resolve().parents[1]/'algorithms/mask_repair/signal_reconnect.py'
   process=subprocess.run([sys.executable,str(script)],env=env,capture_output=True,text=True)
   self.assertEqual(process.returncode,0,process.stdout+process.stderr)
   report=json.loads((work/'b/report.json').read_text());result=tifffile.imread(work/'b/connected-mask.tif')>0
   self.assertTrue(result[mask].all());self.assertEqual(ndi.label(result,np.ones((3,3,3)))[1],1)
   self.assertTrue(any(e.get('chain_fragment_ids') for e in report['accepted']))
   self.assertEqual(report['fragments_before']-report['fragments_after'],sum(e['component_reduction'] for e in report['accepted']))
