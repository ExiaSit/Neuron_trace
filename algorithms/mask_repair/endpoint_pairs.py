"""Stream inter-component endpoint pairs without materializing within-cell pairs."""
import numpy as np
from scipy.spatial import cKDTree


def cross_component_pairs(points, components, radius, chunk_size=64):
    points=np.asarray(points,float);components=np.asarray(components)
    if not len(points):return
    tree=cKDTree(points)
    for start in range(0,len(points),chunk_size):
        neighbors=tree.query_ball_point(points[start:start+chunk_size],radius,return_sorted=True)
        for i,near in enumerate(neighbors,start):
            near=np.asarray(near,dtype=np.intp)
            valid=near[(near>i)&(components[near]!=components[i])]
            for j in valid:yield i,int(j)
