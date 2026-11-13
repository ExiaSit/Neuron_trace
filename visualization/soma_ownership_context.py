"""Whole-route raw views for competing-soma ownership review (1 um voxels)."""
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from algorithms.tracing.signal_support import resample_line


def clip_segment(a,b,axis,low,high):
    a=np.asarray(a,float);b=np.asarray(b,float);d=b[axis]-a[axis]
    if abs(d)<1e-10:return (a,b) if low<=a[axis]<=high else None
    t0,t1=sorted(((low-a[axis])/d,(high-a[axis])/d));t0=max(0.,t0);t1=min(1.,t1)
    return None if t0>t1 else (a+t0*(b-a),a+t1*(b-a))


def unfolded(raw,xyz,step=.5):
    p=resample_line(xyz,step)
    if len(p)<2:return np.zeros((17,2)),np.array([0.,1.])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-8)
    ref=np.tile([0.,0.,1.],(len(t),1));ref[np.abs(t[:,2])>.85]=[0.,1.,0.]
    u=np.cross(t,ref);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-8);v=np.cross(t,u)
    # Keep signs consistent across a reference-axis change.
    for i in range(1,len(u)):
        if np.dot(u[i],u[i-1])<0:u[i]*=-1;v[i]*=-1
    offsets=np.linspace(-4,4,17);width=np.linspace(-1,1,5)
    q=p[None,:,None,:]+offsets[:,None,None,None]*u[None,:,None,:]+width[None,None,:,None]*v[None,:,None,:]
    a=map_coordinates(raw.astype(np.float32),q[...,::-1].reshape(-1,3).T,order=1,mode='constant',cval=0).reshape(17,len(p),5).max(2)
    return a,arc


def render_case(raw,case,somas,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);images=[]
    routes=[('SWC from target',case['upstream_xyz'],'cyan'),('To competing soma',case['other_route_xyz'],'magenta'),
            ('To target (image route)',case['target_route_xyz'],'lime'),('Candidate run',case['downstream_xyz'],'orange')]
    allp=np.concatenate([np.asarray(p) for _,p,_ in routes if len(p)])
    allp=np.vstack([allp,somas[0]['center_xyz'],somas[case['competing_soma_id']]['center_xyz']])
    lo=np.maximum(0,np.floor(allp.min(0)-12)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(allp.max(0)+13)).astype(int)
    case['bbox_xyz']=[lo.tolist(),hi.tolist()]
    crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    for axis,(name,plane) in enumerate([('XY',(0,1)),('XZ',(0,2)),('YZ',(1,2))]):
        bg=crop.max(axis);extent=[lo[plane[0]]-.5,hi[plane[0]]-.5,hi[plane[1]]-.5,lo[plane[1]]-.5]
        fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
        for col,ax in enumerate(axes):
            ax.imshow(bg,cmap='gray',vmin=0,vmax=max(50,float(np.percentile(bg,99.7))),extent=extent,interpolation='nearest')
            for s in somas:
                xyz=np.asarray(s['center_xyz'])
                if not np.all((xyz>=lo)&(xyz<hi)):continue
                selected=s['soma_id'] in [0,case['competing_soma_id']];color='lime' if s['is_target'] else ('magenta' if selected else '#bbbbbb')
                ax.annotate('T' if s['is_target'] else 'S'+str(s['soma_id']),(xyz[plane[0]],xyz[plane[1]]),xytext=(6,6),textcoords='offset points',color=color,fontsize=10,weight='bold')
                if col:ax.scatter(xyz[plane[0]],xyz[plane[1]],s=90,facecolors='none',edgecolors=color)
            if col:
                for label,p,color in routes:
                    if len(p):p=np.asarray(p);ax.plot(p[:,plane[0]],p[:,plane[1]],color=color,lw=.9,alpha=.9,label=label)
                for j,o in enumerate(case['cut_options']):
                    p=o['xyz'];ax.scatter(p[plane[0]],p[plane[1]],c='red',s=28,marker='x')
                    ax.annotate('C'+str(j),(p[plane[0]],p[plane[1]]),xytext=(3,-10),textcoords='offset points',color='red',fontsize=8)
                p=case['center_xyz'];ax.scatter(p[plane[0]],p[plane[1]],c='orange',s=35,marker='s')
            ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal');ax.set_xlabel('XYZ'[plane[0]]+' (um)');ax.set_ylabel('XYZ'[plane[1]]+' (um)')
            ax.set_title(name+(' raw + soma labels' if not col else ' route hypotheses + cut options'))
        axes[1].legend(loc='upper right',fontsize=7)
        fig.suptitle(f"Case {case['case_id']}: target T versus candidate soma S{case['competing_soma_id']}\nFull upstream and soma-to-branch paths. Colored lines do NOT prove connectivity.",fontsize=12)
        path=folder/(name.lower()+'.png');fig.savefig(path,dpi=110);plt.close(fig);images.append(str(path))
    fig,axes=plt.subplots(3,1,figsize=(12,7),layout='constrained')
    for ax,(label,p,color) in zip(axes,routes[:3]):
        a,arc=unfolded(raw,p);ax.imshow(a,cmap='gray',vmin=0,vmax=180,extent=[0,arc[-1],4,-4],aspect='auto',interpolation='nearest')
        ax.axhline(0,color=color,lw=.4,alpha=.5);ax.set_ylabel('Offset (um)');ax.set_title(label+' : raw transverse ribbon, origin at soma',color='black')
        ax.set_xlabel('Distance along path (um)')
        if label.startswith('SWC'):
            pp=resample_line(p,.5)
            for j,o in enumerate(case['cut_options']):
                k=np.argmin(np.linalg.norm(pp-o['xyz'],axis=1));ax.axvline(arc[k],color='red',lw=.6);ax.text(arc[k],-3.2,'C'+str(j),color='red',fontsize=8)
    fig.suptitle('Unfolded raw signal: dark gaps weaken a path; a bright ridge alone cannot prove neuron identity.\nTransverse +/-4 um; maximum over orthogonal +/-1 um; interpolation may change local appearance.',fontsize=11)
    path=folder/'path_unfolded.png';fig.savefig(path,dpi=110);plt.close(fig);images.append(str(path))
    return images


def render_cut_detail(raw,case,cut_id,folder):
    """Local raw box and thin-slab comparison; overlays are clipped to slabs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    center=np.asarray(next(o['xyz'] for o in case['cut_options'] if o['node_id']==cut_id));images=[]
    lo=np.maximum(0,np.floor(center-15)).astype(int);hi=np.minimum(raw.shape[::-1],np.ceil(center+16)).astype(int)
    crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    paths=[(case['upstream_xyz'],'cyan'),(case['other_route_xyz'],'magenta'),(case['target_route_xyz'],'lime')]
    for axis,(name,plane) in enumerate([('XY',(0,1)),('XZ',(0,2)),('YZ',(1,2))]):
        hidden=2-axis;depth=int(round(center[hidden]));low=max(lo[hidden],depth-1);high=min(hi[hidden],depth+2)
        slices=[slice(None)]*3;slices[axis]=slice(low-lo[hidden],high-lo[hidden]);bgs=[crop.max(axis),crop[tuple(slices)].max(axis)]
        extent=[lo[plane[0]]-.5,hi[plane[0]]-.5,hi[plane[1]]-.5,lo[plane[1]]-.5]
        fig,axes=plt.subplots(2,2,figsize=(8,8),layout='constrained')
        for row in range(2):
            for col in range(2):
                ax=axes[row,col];ax.imshow(bgs[row],cmap='gray',vmin=0,vmax=180,extent=extent,interpolation='nearest')
                if col:
                    for p,color in paths:
                        segs=[]
                        for a,b in zip(p,p[1:]):
                            segment=clip_segment(a,b,hidden,low-.5,high-.5) if row else (np.asarray(a),np.asarray(b))
                            if segment is not None:segs.append(np.asarray(segment)[:,list(plane)])
                        if segs:ax.add_collection(LineCollection(segs,colors=color,linewidths=1))
                    ax.scatter(center[plane[0]],center[plane[1]],marker='x',c='red')
                ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_title(('Box MIP' if not row else f'{high-low} um slab {"XYZ"[hidden]}={depth}')+(' + clipped paths' if col else ' raw'))
        fig.suptitle(f'Case {case["case_id"]}: proposed cut node {cut_id}, {name}\nCyan SWC; magenta competing-soma hypothesis; green target-soma hypothesis')
        p=Path(folder)/('cut_'+name.lower()+'.png');fig.savefig(p,dpi=110);plt.close(fig);images.append(str(p))
    return images
