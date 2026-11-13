"""Registered XY/XZ/YZ raw and hypothesis panels for a soma-rooted walk."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from visualization.ownership_audit import PLANES,clipped_segments
COLORS=['#ffb347','#62df80','#ef80f5']

def box(raw,points,pad):
    p=np.array(points);lo=np.maximum(0,np.floor(p.min(0)-pad)).astype(int);hi=np.minimum(np.array(raw.shape[::-1])-1,np.ceil(p.max(0)+pad)).astype(int)
    return lo,hi

def in_box(xyz,lo,hi):return bool(np.all((np.array(xyz)>=lo-.5)&(np.array(xyz)<=hi+.5)))

def window(raw,lo,hi):
    crop=raw[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1];black=float(np.percentile(crop,2));white=max(black+12,30.,float(np.percentile(crop,99.5)))
    return [black,min(255.,white)]

def panel(ax,raw,lo,hi,axis,plane,marks,routes,window_,title):
    a,b=plane;crop=raw[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1];ax.imshow(crop.max(axis),cmap='gray',vmin=window_[0],vmax=window_[1],interpolation='nearest',extent=[lo[a]-.5,hi[a]+.5,hi[b]+.5,lo[b]-.5])
    for points,color in routes:
        segs=clipped_segments([points],lo,hi,plane)
        if segs:ax.add_collection(LineCollection(segs,colors=color,linewidths=.8))
    visible={}
    for label,p in marks.items():
        if not in_box(p,lo,hi):continue
        visible[label]=dict(xyz_um=list(p),uv_um=[p[a],p[b]],projection_pixel_uv=[p[a]-int(lo[a]),p[b]-int(lo[b])],depth_um=p[2-axis])
        ax.plot(p[a],p[b],'+',color='#ffd66d',ms=4);ax.annotate(label,(p[a],p[b]),xytext=(3,3),textcoords='offset points',fontsize=6,color='#ffd66d')
    ax.set(xlim=(lo[a]-.5,hi[a]+.5),ylim=(hi[b]+.5,lo[b]-.5),xlabel='XYZ'[a]+' (um)',ylabel='XYZ'[b]+' (um)',aspect='equal');ax.set_title(title,fontsize=8);ax.tick_params(labelsize=7)
    return visible

def render(raw,case,target,folder,prefix='initial',focus=None,supplement=False,offset=0):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);center=case['center_xyz'] if focus is None else focus
    marks={'J':case['center_xyz'],'T':target['xyz_um']};routes=[(case['upstream_xyz'],'#5bdcff')]
    forward=[case['center_xyz']]
    for i,b in enumerate(case['branches']):
        p=np.array(b['xyz']);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];marks[b['branch_id']]=p[np.argmin(abs(arc-min(8,arc[-1])))].tolist()
        for s in b['future']:forward+=s['xyz'];routes.append((s['xyz'],COLORS[i%3]))
    upstream=case['upstream_xyz']
    if len(upstream)>1:
        for i,j in enumerate(np.linspace(0,len(upstream)-1,min(7,len(upstream)),dtype=int)):marks['U'+str(i)]=upstream[j]
    # Avoid clutter in local/raw panels; exact edge endpoints remain in the manifest.
    localmarks={k:v for k,v in marks.items() if not k.startswith('U')}
    views={};images=[]
    def save_panels(name,lo,hi,scope,raw_only=False):
        w=window(raw,lo,hi)
        for overlay in ([False] if raw_only else [False,True]):
            path=folder/f'{prefix}_{name}_{"hypothesis" if overlay else "raw"}.png';fig,axes=plt.subplots(1,3,figsize=(12,4.2),dpi=180,layout='constrained');records=[]
            for ax,(plane,axis,xy) in zip(axes,PLANES):
                vid=f'{prefix}_{name}_{plane}_{"hyp" if overlay else "raw"}';visible=panel(ax,raw,lo,hi,axis,xy,localmarks if scope=='local' else marks,routes if overlay else [],w,vid)
                v=dict(view_id=vid,raw=not overlay,scope=scope,plane=plane,array_projection_axis=axis,world_axes=list(xy),xyz_bounds=[lo.tolist(),hi.tolist()],crop_origin_xyz=lo.tolist(),window=w,image=str(path),landmarks=visible,
                    source_projection_shape_hw=[int(hi[xy[1]]-lo[xy[1]]+1),int(hi[xy[0]]-lo[xy[0]]+1)],coordinate_rule='projection pixel u=XYZ[horizontal]-origin[horizontal], v=XYZ[vertical]-origin[vertical]; source volume is ZYX, 1 um voxels')
                records.append((vid,v,ax,xy))
            fig.suptitle(('HYPOTHESES: cyan = traversed SWC; outlet colors = three future branch segments per path' if overlay else 'RAW: identical landmark IDs mean the same 3D point in every view')+f' | window {w[0]:.0f}..{w[1]:.0f}',fontsize=9)
            finish(fig,path,records);images.append(str(path))
    def finish(fig,path,records):
        fig.canvas.draw();fig.set_layout_engine(None);width,height=fig.canvas.get_width_height()
        for vid,v,ax,xy in records:
            origin=ax.transData.transform([0,0]);u=ax.transData.transform([1,0])-origin;vv=ax.transData.transform([0,1])-origin
            matrix=np.zeros((2,4));matrix[0,xy[0]]=u[0];matrix[0,xy[1]]=vv[0];matrix[0,3]=origin[0];matrix[1,xy[0]]=-u[1];matrix[1,xy[1]]=-vv[1];matrix[1,3]=height-origin[1]
            v.update(xyz_to_png_uv=matrix.tolist(),image_size_wh=[width,height]);views[vid]=v
        fig.savefig(path,dpi=180);plt.close(fig)
    lo,hi=box(raw,[center],18);save_panels('local',lo,hi,'local',raw_only=supplement)
    w=window(raw,lo,hi);path=folder/f'{prefix}_slices.png';fig,axes=plt.subplots(3,3,figsize=(9,9),dpi=180,layout='constrained');records=[]
    for row,(plane,axis,xy) in enumerate(PLANES):
        hidden=2-axis
        for col,delta in enumerate([-1,0,1]):
            depth=int(np.clip(round(center[hidden])+delta+offset,lo[hidden],hi[hidden]));sl=lo.copy();sh=hi.copy();sl[hidden]=sh[hidden]=depth
            vid=f'{prefix}_{plane}_slice_{depth}';visible=panel(axes[row,col],raw,sl,sh,axis,xy,localmarks,[],w,vid)
            records.append((vid,dict(view_id=vid,raw=True,scope='local',plane=plane,xyz_bounds=[sl.tolist(),sh.tolist()],landmarks=visible,image=str(path),window=w,crop_origin_xyz=sl.tolist(),array_projection_axis=axis,world_axes=list(xy)),axes[row,col],xy))
    fig.suptitle('RAW consecutive 1 um slices. A missing fiber in one slice is not proof of a false connection.',fontsize=9);finish(fig,path,records);images.append(str(path))
    if not supplement:
        fl,fh=box(raw,forward,8);save_panels('forward3',fl,fh,'forward_three_segments')
        ul,uh=box(raw,upstream+[target['xyz_um']],10);save_panels('upstream',ul,uh,'full_upstream')
    return images,views
