"""Multiscale evidence for the local crossing ownership model."""
from pathlib import Path
import numpy as np

def render_encounter(raw,nodes,site,target,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    lo,hi=np.asarray(site['bbox_xyz']);center=np.asarray(site['center_xyz'])
    incoming=np.array([nodes[i][2:5] for i in site['incoming_ids']]);outgoing=np.array([nodes[i][2:5] for i in site['outgoing_ids']])
    ancestors=np.array([nodes[i][2:5] for i in site['ancestors']]);paths=[(incoming,'cyan'),(outgoing,'yellow')]
    images=[]
    def finish(fig,name):
        path=folder/name;fig.savefig(path,dpi=115);plt.close(fig);images.append(str(path))
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    ax.imshow(raw.max(0),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    ax.plot(ancestors[:,0],ancestors[:,1],color='#62b8e5',lw=.9)
    for xyz,color in paths:ax.plot(xyz[:,0],xyz[:,1],color=color,lw=1.5)
    ax.scatter(target[0],target[1],marker='*',s=140,color='lime');ax.text(target[0]+5,target[1],'TARGET',color='lime')
    ax.scatter(center[0],center[1],marker='x',s=80,color='red')
    ax.set_title(f"Site {site['site_id']} | node {site['node_id']} | global XY MIP\nCyan incoming; yellow outgoing; SWC lines are hypotheses")
    finish(fig,'global.png')
    crop=raw[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    for axis,(name,plane) in enumerate([('XY',(0,1)),('XZ',(0,2)),('YZ',(1,2))]):
        hidden=2-axis;depth=int(round(center[hidden]-lo[hidden]));section=[slice(None)]*3
        section[axis]=slice(max(0,depth-1),min(crop.shape[axis],depth+2))
        full=crop.max(axis);thin=crop[tuple(section)].max(axis)
        fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
        extent=[lo[plane[0]]-.5,hi[plane[0]]-.5,hi[plane[1]]-.5,lo[plane[1]]-.5]
        for row,bg in enumerate([full,thin]):
            for col in range(2):
                ax=axes[row,col];ax.imshow(bg,cmap='gray',vmin=0,vmax=max(40,float(np.percentile(full,99.7))),extent=extent,interpolation='nearest')
                if col:
                    for xyz,color in paths:ax.plot(xyz[:,plane[0]],xyz[:,plane[1]],color=color,lw=1.25)
                    ax.scatter(center[plane[0]],center[plane[1]],color='red',marker='x',s=40)
                ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal')
                ax.set_title(('Box MIP' if row==0 else f'3 um slab at {"XYZ"[hidden]}={center[hidden]:.1f}')+(' + SWC' if col else ' raw'))
                ax.set_xlabel('XYZ'[plane[0]]+' um');ax.set_ylabel('XYZ'[plane[1]]+' um')
        fig.suptitle(f'{name}: raw panels on left; overlays on right. Red X = candidate switch.\nOverlay paths can lie outside the thin slab; verify with clean image.')
        finish(fig,name.lower()+'.png')
    fig,axes=plt.subplots(3,3,figsize=(10,10),layout='constrained')
    for ax,dz in zip(axes.ravel(),range(-4,5)):
        z=int(np.clip(round(center[2])+dz,0,raw.shape[0]-1));bg=raw[z,lo[1]:hi[1],lo[0]:hi[0]]
        ax.imshow(bg,cmap='gray',vmin=0,vmax=max(40,float(np.percentile(crop,99.7))),extent=[lo[0]-.5,hi[0]-.5,hi[1]-.5,lo[1]-.5],interpolation='nearest')
        ax.set_title(f'Raw XY z={z} um')
    fig.suptitle('Consecutive raw slices across the encounter; no SWC overlays')
    finish(fig,'slices.png')
    return images
