from neuroutils.config.settings import TILE_SOMA_MAP_DIR, DEFAULT_RESOLUTION_UNIT
from neuroutils.meta.neuron import get_neuron_meta, get_xy_z_resolution, get_tile_id
import pandas as pd
import os
import re
from neuroutils.config.settings import TILE_SOMA_MAP_DIR


def find_soma_marker_file_for_ptrs_db(tile_id):
    ptrs_files = os.listdir(TILE_SOMA_MAP_DIR)
    pattern = re.compile(r'P\d{5}-T\d{3}-R\d{3}-S\d{3}.*')
    # print(tile_id)
    assert pattern.match(tile_id), f"Tile ID {tile_id} does not match expected pattern."
    for ptrs_file in ptrs_files:
        if(tile_id in str(ptrs_file)):
        # if(str(ptrs_file).startswith(tile_id)):
            if(".marker" in ptrs_file or ".apo" in ptrs_file):
                # print(f"Found soma marker file for tile ID {tile_id} in PTRS database: {ptrs_file}")
                return os.path.join(TILE_SOMA_MAP_DIR, ptrs_file)
    # print(f"No soma marker file found for tile ID {tile_id} in PTRS database.")
    return None


def get_somas_in_tile(tile_id):
    """
    Get the somas in a tile.

    Args:
        tile_id (str): The ID of the tile.

    Returns:
        list: A list of soma IDs in the tile.
    """
    # print(f"Getting somas in tile: {tile_id}")
    try:
        tile_soma_map_path = find_soma_marker_file_for_ptrs_db(tile_id)
    except Exception as e:
        print(f"Error finding soma marker file for tile ID {tile_id}: {e}")
        return None
    # print(f"Loading soma markers from: {tile_soma_map_path}")
    if not tile_soma_map_path:
        # raise FileNotFoundError(f"Tile soma map for {tile_id} does not exist.")
        # warning and return None
        print(f"Warning: Tile soma map for {tile_id} does not exist.")
        return None


    # somas = pd.read_csv(tile_soma_map_path, sep=',',
    #                            comment='#',
    #                            names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
    #                                   'color_b'])
    if(".apo" in tile_soma_map_path):
        # n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b
        soma_markers = pd.read_csv(tile_soma_map_path, sep=',',
                                   comment='#',
                                   names=['n', 'orderinfo', 'name', 'comment', 'z', 'x', 'y', 'pixmax', 'intensity',
                                          'sdev', 'volsize', 'mass', 'comment1', 'comment2', 'comment3', 'color_r', 'color_g', 'color_b'])

    elif(".marker" in tile_soma_map_path):
        soma_markers = pd.read_csv(tile_soma_map_path, sep=',',
                                   comment='#',
                                   names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                          'color_b'])

    return soma_markers


def get_tile_resolution(tile_id, default_xy_res=300, default_z_res=1000):
    '''
    Get the resolution of a tile.
    :param tile_id:
    :param default_xy_res:
    :param default_z_res:
    :return: xy_res, z_res
    '''
    soma_markers = get_somas_in_tile(tile_id)
    first_neuron_id = int(soma_markers['name'].values[0])
    xy_res, z_res = get_xy_z_resolution(first_neuron_id)
    # if not valid, use default
    if(not isinstance(xy_res, (int, float))):
        xy_res = default_xy_res
    if(not isinstance(z_res, (int, float))):
        z_res = default_z_res
    return xy_res, z_res


def get_relative_soma_positions(neuron_id: int, rescale=False) -> pd.DataFrame:
    """
    Compute the relative positions of all somas in the same tile with respect to the given neuron's soma.

    Args:
        neuron_id (int): The neuron ID.
        rescale (bool): Whether to rescale the coordinates to a common resolution.

    Returns:
        pd.DataFrame: A DataFrame with columns [soma_id, dx, dy, dz] relative to the neuron.
    """
    neuron_meta = get_neuron_meta(neuron_id)

    tile_id = neuron_meta["document_name"].values[0]
    xy_res = float(neuron_meta["xy_resolution"].values[0])
    z_res = float(neuron_meta["z_resolution"].values[0])
    x0 = float(neuron_meta["soma_x"].values[0])
    y0 = float(neuron_meta["soma_y"].values[0])
    z0 = float(neuron_meta["soma_z"].values[0])

    # Load somas in tile
    somas = get_somas_in_tile(tile_id)

    # Normalize to same resolution
    if(rescale):
        somas['x'] = somas['x'] * xy_res / DEFAULT_RESOLUTION_UNIT
        somas['y'] = somas['y'] * xy_res / DEFAULT_RESOLUTION_UNIT
        somas['z'] = somas['z'] * z_res / DEFAULT_RESOLUTION_UNIT

    # Compute relative positions
    somas['x'] = somas['x'] - x0
    somas['y'] = somas['y'] - y0
    somas['z'] = somas['z'] - z0

    return somas

def find_nearest_soma(neuron_id, eps=1e-6):
    """
    Find the nearest soma to a given neuron ID.

    Args:
        neuron_id (int): The ID of the neuron.

    Returns:
        str: The ID of the nearest soma.
        float: The distance to the nearest soma.
    """
    somas = get_relative_soma_positions(neuron_id, rescale=True)
    # Compute distances
    # 计算距离
    somas['distance'] = (somas['x'] ** 2 + somas['y'] ** 2 + somas['z'] ** 2) ** 0.5

    # 排除自身（距离为 极小值）
    somas_non_self = somas[somas['distance'] > eps]

    # 找最近的 soma
    nearest_soma = somas_non_self.loc[somas_non_self['distance'].idxmin()]
    return nearest_soma['name'], nearest_soma['distance']


def get_mapped_somas_in_neuron_block(neuron_id):
    """
    Get the mapped somas in the neuron block for a given neuron ID.
    :param neuron_id:
    :param ptrs_dir:
    :param output_marker_file:
    :return:
    """

    neuron_meta = get_neuron_meta(neuron_id)
    soma_x, soma_y, soma_z = neuron_meta[["soma_x", "soma_y", "soma_z"]].values[0]

    ptrs_flag = get_tile_id(neuron_id)
    ptrs_markers = get_somas_in_tile(ptrs_flag)
    if(ptrs_markers is None):
        print(f"Warning: No tile ID found for neuron {neuron_id}.")
        # ptrs_markers = pd.DataFrame(columns=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g', 'color_b'])
        # # 添加一行当前神经元的位置
        # # ptrs_markers = ptrs_markers.append({'x': float(soma_x), 'y': float(soma_y), 'z': float(soma_z), 'radius': 0, 'shape': "", 'name': str(neuron_id), 'comment': '', 'color_r': 255, 'color_g': 0, 'color_b': 0}, ignore_index=True)
        # new_row = pd.DataFrame([{'x': float(soma_x), 'y': float(soma_y), 'z': float(soma_z), 'radius': 0, 'shape': "", 'name': str(neuron_id), 'comment': '', 'color_r': 255, 'color_g': 0, 'color_b': 0}])
        # ptrs_markers = pd.concat([ptrs_markers, new_row], ignore_index=True)
        # soma_x, soma_y, soma_z是可以float的

        row_data = {
            'x': float(soma_x),
            'y': float(soma_y),
            'z': float(soma_z),
            'radius': 0,
            'shape': "",
            'name': str(neuron_id),
            'comment': '',
            'color_r': 255,
            'color_g': 0,
            'color_b': 0
        }
        # 2. 定义列顺序 (确保列的顺序是你想要的)
        cols = ['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g', 'color_b']
        ptrs_markers = pd.DataFrame([row_data], columns=cols)

        return ptrs_markers

    # print(ptrs_markers['name'])
    # print(neuron_id)
    current_soma_pos = ptrs_markers[ptrs_markers['name'] == neuron_id][['x', 'y', 'z']].values[0]

    ptrs_markers['x'] = (ptrs_markers['x'] - current_soma_pos[0] + float(soma_x))
    ptrs_markers['y'] = (ptrs_markers['y'] - current_soma_pos[1] + float(soma_y))
    ptrs_markers['z'] = (ptrs_markers['z'] - current_soma_pos[2] + float(soma_z))

    return ptrs_markers


def get_mapped_somas_in_neuron_block_rescaled_1um(neuron_id):
    ptrs_markers = get_mapped_somas_in_neuron_block(neuron_id)
    xy_res, z_res = get_xy_z_resolution(neuron_id)
    ptrs_markers['x'] = ptrs_markers['x'] * xy_res / DEFAULT_RESOLUTION_UNIT
    ptrs_markers['y'] = ptrs_markers['y'] * xy_res / DEFAULT_RESOLUTION_UNIT
    ptrs_markers['z'] = ptrs_markers['z'] * z_res / DEFAULT_RESOLUTION_UNIT
    return ptrs_markers

if __name__ == "__main__":
    # Example usage
    test_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_retrace_swcs/image_15896.swc"
    neuron_id = 15896
    nearest_soma, distance = find_nearest_soma(neuron_id)
    print(f"Nearest soma to neuron {neuron_id} is {nearest_soma} with distance {distance:.2f}")