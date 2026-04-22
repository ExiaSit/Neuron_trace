import os
import pandas as pd

def load_marker(marker_file):
    """
    Read the marker file and return the contents as a string.
    """
    markers = pd.read_csv(marker_file, sep=',', header=None, comment='#')
    # ##x,y,z,radius,shape,name,comment,color_r,color_g,color_b
    markers.columns = ['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g', 'color_b']
    # 数据类型
    markers['x'] = markers['x'].astype(float)
    markers['y'] = markers['y'].astype(float)
    markers['z'] = markers['z'].astype(float)
    markers['radius'] = markers['radius'].astype(float)
    markers['shape'] = markers['shape'].astype(str)
    markers['name'] = markers['name'].astype(str)
    markers['comment'] = markers['comment'].astype(str)
    markers['color_r'] = markers['color_r'].astype(int)
    markers['color_g'] = markers['color_g'].astype(int)
    markers['color_b'] = markers['color_b'].astype(int)

    return markers