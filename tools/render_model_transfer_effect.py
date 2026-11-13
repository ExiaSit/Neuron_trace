"""Show exactly which original edges were removed by a completed model run."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visualization.swc_view import SwcMipView

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();out=Path(args.run)
    identity=json.loads((out/'input.json').read_text());audit=json.loads((out/'cuts.json').read_text())
    tree=np.loadtxt(identity['swc']);raw=tifffile.imread(identity['raw']);removed=set(audit['removed_ids'])
    nodes={int(n[0]):n for n in tree};kept_edges=[];removed_edges=[]
    for n in tree:
        if int(n[6]) not in nodes:continue
        (removed_edges if int(n[0]) in removed else kept_edges).append([nodes[int(n[6])][2:5],n[2:5]])
    removed_tree=tree[np.isin(tree[:,0],list(removed))].copy()
    if len(removed_tree):
        removed_tree[~np.isin(removed_tree[:,6],list(removed)),6]=-1
        np.savetxt(out/'removed_segments.swc',removed_tree,fmt=['%d','%d','%.5f','%.5f','%.5f','%.5f','%d'],header='Removed fragments for review; detached parents set to -1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained');view=SwcMipView(raw.shape,tree[:,2:5])
    for axis,(plane,name) in enumerate([([0,1],'XY'),([0,2],'XZ'),([1,2],'YZ')]):
        ax=axes[axis];ax.imshow(view.project(raw,axis),cmap='gray',vmin=0,vmax=255,extent=view.extent(axis))
        for group,color,width in [(kept_edges,'#44ddff',.5),(removed_edges,'#ff4444',1.4)]:
            if group:ax.add_collection(LineCollection(np.asarray(group)[:,:,plane],colors=color,linewidths=width))
        view.apply(ax,axis);ax.set_title(name);ax.set_aspect('equal')
    fig.suptitle(f'{Path(identity["swc"]).stem} | cyan = retained; red = removed original edges | {len(removed)} nodes removed\nExperimental model-assisted result; not ground-truth accuracy')
    fig.savefig(out/'removed_overlay.png',dpi=130);plt.close(fig)
    index=out/'index.html';text=index.read_text()
    if 'removed_overlay.png' not in text:
        link='<p><a href="removed_segments.swc">删除片段（独立复核用）</a></p>' if len(removed_tree) else '<p>本轮未删除节点。</p>'
        text=text.replace('<table>','<h2>保留与删除范围：青色保留，红色删除</h2><img src="removed_overlay.png">'+link+'<table>',1)
        index.write_text(text)
    print(f'Exported removed-edge overlay and {len(removed_tree)} review fragment nodes')

if __name__=='__main__':main()
