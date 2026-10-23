from pathlib import Path
import numpy as np
import tifffile
def paths(spec):
    from utils.case_worker import paths as get
    return get(spec)
def check_swc(path):
    from utils.case_worker import check_swc as check
    return check(path)

def comparison(spec, failure=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    p=paths(spec); p['comparison'].parent.mkdir(parents=True,exist_ok=True); raw=tifffile.imread(p['raw']);before=tifffile.imread(spec['mask'])>0;after=tifffile.imread(p['mask'])>0 if p['mask'].is_file() else None
    bg=raw.max(0);vmax=max(25,float(np.percentile(bg,99.5)))
    fig,axes=plt.subplots(2,3,figsize=(18,12))
    titles=['Raw XY MIP','Original mask','Updated mask: red additions','Previous SWC','New APP2',spec['config'].get('method_version','v0.82')+' final SWC']
    for ax,title in zip(axes.flat,titles):ax.imshow(bg,cmap='gray',vmin=0,vmax=vmax);ax.set_title(title);ax.axis('off')
    for ax,added in [(axes[0,1],False),(axes[0,2],True)]:
        rgba=np.zeros((*bg.shape,4));rgba[before.any(0)]=[0,1,1,.65]
        if added and after is not None:rgba[(after&~before).any(0)]=[1,.1,0,.85]
        if added and after is None:ax.text(.5,.5,'Updated mask unavailable',transform=ax.transAxes,ha='center',color='yellow')
        ax.imshow(rgba)
    old=spec['old_swcs']
    for ax,files,color in [(axes[1,0],old,'cyan'),(axes[1,1],[str(p['app2'])],'orange'),(axes[1,2],[str(p['final'])],'lime')]:
        if not files:ax.text(.5,.5,'Previous SWC not supplied',transform=ax.transAxes,ha='center',color='yellow')
        for file in files:
            if not Path(file).is_file():
                ax.text(.5,.5,'No output: stage failed / not reached',transform=ax.transAxes,ha='center',color='yellow');continue
            nodes=check_swc(Path(file));edges=[[n.xy,nodes[n.parent].xy] for n in nodes.values() if n.parent in nodes]
            ax.add_collection(LineCollection(edges,colors=color,linewidths=.8))
    metadata=spec['metadata']
    soma_x=float(metadata['soma_x'])*float(metadata['xy_resolution'])/1000
    soma_y=float(metadata['soma_y'])*float(metadata['xy_resolution'])/1000
    for ax in axes.flat:
        ax.scatter([soma_x],[soma_y],s=12,c='red',marker='o',edgecolors='none',zorder=20)
        ax.set_xlim(-.5,bg.shape[1]-.5);ax.set_ylim(bg.shape[0]-.5,-.5)
    fig.suptitle(f"{spec['id']} | {spec['selection_reason']} | red dot: target soma | " + (f"INCOMPLETE: {failure}" if failure else "SWCs in image coordinates"))
    fig.tight_layout();fig.savefig(p['comparison'],dpi=140);plt.close(fig)
