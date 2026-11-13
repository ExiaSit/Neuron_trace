"""Matched raw/overlay evidence with explicit bounds and correctly clipped segments."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from visualization.soma_ownership_context import clip_segment,unfolded
from algorithms.tracing.signal_support import tree_paths

PLANES=[('XY',0,(0,1)),('XZ',1,(0,2)),('YZ',2,(1,2))]

def bounds(raw,center,radius):
    return np.maximum(0,np.floor(np.array(center)-radius)).astype(int),np.minimum(np.array(raw.shape[::-1])-1,np.ceil(np.array(center)+radius)).astype(int)

def projection(raw,lo,hi,array_axis):
    return raw[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1].max(array_axis)

def clipped_segments(paths,lo,hi,axes):
    segs=[]
    for path in paths:
        for a,b in zip(path,path[1:]):
            clipped=(np.array(a),np.array(b))
            for axis in range(3):
                if clipped is None:break
                clipped=clip_segment(*clipped,axis,lo[axis]-.5,hi[axis]+.5)
            if clipped is not None:segs.append(np.array(clipped)[:,list(axes)])
    return segs

def view_panel(ax,raw,lo,hi,axis,plane,center,title,paths=None,somas=()):
    bg=projection(raw,lo,hi,axis);a,b=plane
    ax.imshow(bg,cmap='gray',vmin=0,vmax=180,extent=[lo[a]-.5,hi[a]+.5,hi[b]+.5,lo[b]-.5],interpolation='nearest')
    ax.plot(center[a],center[b],marker='+',color='#ffbc57',ms=6,mew=.8);ax.annotate('J',(center[a],center[b]),xytext=(4,5),textcoords='offset points',color='#ffbc57',fontsize=8)
    if paths:
        for route,color in paths:
            segs=clipped_segments([route],lo,hi,plane)
            if segs:ax.add_collection(LineCollection(segs,colors=color,linewidths=.8))
    for s in somas:
        p=s['center_xyz']
        if np.all((np.array(p)>=lo)&(np.array(p)<=hi)):
            ax.text(p[a],p[b],'T' if s['soma_id']==0 else 'S'+str(s['soma_id']),color='#ffff99',fontsize=8)
    ax.set_xlim(lo[a]-.5,hi[a]+.5);ax.set_ylim(hi[b]+.5,lo[b]-.5);ax.set_aspect('equal');ax.set_title(title,fontsize=9);ax.tick_params(labelsize=7)
    ax.set_xlabel('XYZ'[a]+' um',fontsize=7);ax.set_ylabel('XYZ'[b]+' um',fontsize=7)

def render_evidence(raw,tree,c,somas,folder,prefix='initial',center=None,radius=18,global_context=True,slice_offset=0):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    center=np.array(c['center_xyz'] if center is None else center);lo,hi=bounds(raw,center,radius)
    nodes,_,paths=tree_paths(tree)
    # Entire nearby SWC is available in overlays, not just the favored path.
    swcpaths=[[nodes[n][2:5] for n in p] for p in paths]
    nearby=[]
    for p in swcpaths:
        q=np.array(p)
        if np.all(q.max(0)>=lo)&np.all(q.min(0)<=hi):nearby.append((p,'#77bddd'))
    routes=nearby+[(c['upstream_xyz'],'cyan'),(c['other_route_xyz'],'magenta')]
    images=[];views={}
    landmarks=[]
    if len(c['upstream_xyz'])>1:landmarks.append(('A',c['upstream_xyz'][max(0,len(c['upstream_xyz'])-9)]))
    if len(c['downstream_xyz'])>1:landmarks.append(('B',c['downstream_xyz'][min(8,len(c['downstream_xyz'])-1)]))
    def register(vid,lo,hi,plane,scope,path,raw=True):
        views[vid]=dict(view_id=vid,xyz_bounds=[lo.tolist(),hi.tolist()],plane=plane,scope=scope,raw=raw,image=str(path),window=[0,180])
    # Separate raw image first, corresponding annotations second, identical coordinates/window.
    for overlay in [False,True]:
        p=folder/f'{prefix}_local_{"hypothesis" if overlay else "raw"}.png'
        fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,(name,axis,plane) in zip(axs,PLANES):
            vid=f'{prefix}_{name}_{"overlay" if overlay else "raw"}'
            view_panel(ax,raw,lo,hi,axis,plane,center,vid,routes if overlay else None)
            for label,point in landmarks:
                if np.all((np.array(point)>=lo)&(np.array(point)<=hi)):
                    ax.plot(point[plane[0]],point[plane[1]],marker='o',mfc='none',mec='#ffbc57',ms=4)
                    ax.annotate(label,(point[plane[0]],point[plane[1]]),xytext=(3,3),textcoords='offset points',color='#ffbc57',fontsize=8)
            register(vid,lo,hi,name,'local',p,not overlay)
        fig.suptitle('HYPOTHESES ONLY: cyan upstream SWC, blue nearby SWC, magenta image route' if overlay else 'RAW first: J is location only. 1 um XYZ; same field/window as hypotheses.',fontsize=10)
        fig.savefig(p,dpi=200);plt.close(fig);images.append(str(p))
    # Serial raw slices in every axis; each view has a precise coordinate inventory.
    p=folder/f'{prefix}_slices.png';fig,axs=plt.subplots(3,3,figsize=(9,9),layout='constrained')
    for row,(name,axis,plane) in enumerate(PLANES):
        hidden=2-axis
        for col,delta in enumerate([-1,0,1]):
            depth=int(np.clip(round(center[hidden])+delta+slice_offset,lo[hidden],hi[hidden]));sl=lo.copy();sh=hi.copy();sl[hidden]=sh[hidden]=depth
            vid=f'{prefix}_{name}_slice_{depth}'
            view_panel(axs[row,col],raw,sl,sh,axis,plane,center,vid)
            register(vid,sl,sh,name,'local',p)
    fig.suptitle('RAW consecutive 1 um slices; absent signal in one slice is not disconnection',fontsize=10);fig.savefig(p,dpi=200);plt.close(fig);images.append(str(p))
    if global_context:
        pp=[c['upstream_xyz'],c['other_route_xyz'],c['downstream_xyz'],[somas[0]['center_xyz']]];points=np.concatenate([p for p in pp if len(p)])
        gl=np.maximum(0,np.floor(points.min(0)-10)).astype(int);gh=np.minimum(np.array(raw.shape[::-1])-1,np.ceil(points.max(0)+10)).astype(int)
        for overlay in [False,True]:
            p=folder/f'{prefix}_global_{"hypothesis" if overlay else "raw"}.png';fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
            for ax,(name,axis,plane) in zip(axs,PLANES):
                vid=f'{prefix}_global_{name}_{"hypothesis" if overlay else "raw"}'
                view_panel(ax,raw,gl,gh,axis,plane,center,vid,[(c['upstream_xyz'],'cyan'),(c['downstream_xyz'],'orange'),(c['other_route_xyz'],'magenta')] if overlay else None,somas)
                register(vid,gl,gh,name,'global',p,not overlay)
            fig.suptitle('GLOBAL location context only. T known metadata soma; S unconfirmed; routes are hypotheses.',fontsize=10);fig.savefig(p,dpi=200);plt.close(fig);images.append(str(p))
    return images,views

def render_unfolded(raw,c,folder):
    p=Path(folder)/'path_unfolded_reference.png';fig,axs=plt.subplots(2,1,figsize=(12,5),layout='constrained')
    for ax,(name,xyz) in zip(axs,[('SWC hypothesis',c['upstream_xyz']),('Competitor skeleton hypothesis',c['other_route_xyz'])]):
        if len(xyz)<2:ax.text(.1,.5,'No route provided');continue
        data,arc=unfolded(raw,xyz);ax.imshow(data,cmap='gray',vmin=0,vmax=180,aspect='auto',extent=[0,arc[-1],4,-4]);ax.set_title(name+'; brightness is not ownership');ax.set_xlabel('Path distance um')
    fig.savefig(p,dpi=200);plt.close(fig);return str(p)
