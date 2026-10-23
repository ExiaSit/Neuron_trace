from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from swc_handler import parse_swc, get_index_dict

def plot_swc(image: Optional[np.ndarray], swc_tree, figname='neuron_mip.png', scale: float=1.):
    if isinstance(swc_tree, str):
        swc_tree = parse_swc(swc_tree)
    elif not isinstance(swc_tree, list):
        raise NotImplementedError("Incorrect data type for @arg `swc_tree`")


    plt.figure(figsize=(10,10))
    if image is not None:
        plt.imshow(image.max(axis=0), cmap='gray',origin='lower')

    index_dict = get_index_dict(swc_tree)
    for node in swc_tree:
        idx, _, x1, y1, _, radius, pid = node
        if pid == -1 or pid not in index_dict:
            continue

        pnode = swc_tree[index_dict[pid]]
        x2, y2 = pnode[2:4]
        plt.plot([x1,x2], [y1,y2], 'dodgerblue', 
                #  marker='o',
                 linewidth=2,)

    coords = np.array([node[2:6] for node in swc_tree])
    if len(coords) > 0:
        plt.scatter(coords[:,0], coords[:,1], s=coords[:,-1]*scale, color='magenta')
    plt.tight_layout()
    plt.savefig(figname, dpi=300); plt.close()
