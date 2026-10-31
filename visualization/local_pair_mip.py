"""Review the local crop, component segments, projections and accepted 3-D route."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_local_pairs(raw,audit,path,max_pairs=3):
    pairs=audit['pairs']
    if not pairs:return
    chosen=sorted(pairs,key=lambda p:(p['status']!='accepted',p['status']!='rejected_3d',-p['gap']))[:max_pairs]
    fig,axes=plt.subplots(len(chosen),3,figsize=(15,4.6*len(chosen)),squeeze=False)
    for row,rec in enumerate(chosen):
        lo=np.array([b[0] for b in rec['bbox_zyx']]);sl=tuple(slice(a,b) for a,b in rec['bbox_zyx'])
        crop=raw[sl]
        for axis,view in enumerate(rec['views']):
            ax=axes[row,axis];mip=crop.max(axis)
            ax.imshow(mip,cmap='gray',vmin=0,vmax=max(float(np.percentile(mip,99.5)),1))
            dims=[d for d in range(3) if d!=axis]
            for segment,color in zip(rec['local_segments_zyx'],['#39d5ff','#ffaa45']):
                points=np.asarray(segment)-lo
                ax.plot(points[:,dims[1]],points[:,dims[0]],color=color,lw=2.2)
            projected=np.asarray(view['path_rc'])
            ax.plot(projected[:,1],projected[:,0],'--',color='white',lw=1,alpha=.8)
            if 'path_zyx' in rec:
                points=np.asarray(rec['path_zyx'])-lo
                ax.plot(points[:,dims[1]],points[:,dims[0]],color='#ff5757',lw=1.6)
            ends=np.asarray(rec['endpoints_zyx'])-lo
            ax.scatter(ends[:,dims[1]],ends[:,dims[0]],c='yellow',s=22,zorder=5)
            ax.set_title(f"{rec['source']} -> {rec['target']} | {view['plane']} | support {view['signal_fraction']:.0%}\n"
                         f"gap {rec['gap']:.1f} um | {rec['status']}",fontsize=10)
            ax.set_xlabel('local voxel coordinate');ax.set_ylabel('local voxel coordinate')
    fig.suptitle('Pair-local MIP: cyan/orange = nearby segments; white = 2D route; red = 3D route\n'
                 'Yellow = gap endpoints. Each row uses one bounded 3D crop.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150);plt.close(fig)
