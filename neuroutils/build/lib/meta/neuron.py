import os
import pandas as pd
from neuroutils.config.settings import NEURON_META_INFO_PATH, NEURON_META_DIR
from neuroutils.config.settings import V3D_IMAGE_ROOT


def split_neuron_meta_table(csv_path: str = None, output_dir: str = None):
    """
    Split a metadata table into per-neuron files based on neuron_id.

    Args:
        csv_path (str): Path to the metadata CSV file. Defaults to config path.
        output_dir (str): Directory to save individual neuron meta files. Defaults to config path.
    """
    csv_path = csv_path or NEURON_META_INFO_PATH
    output_dir = output_dir or NEURON_META_DIR

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(csv_path)

    if "cell_id" not in df.columns:
        raise ValueError("Metadata table must contain a 'cell_id' column.")

    for _, row in df.iterrows():
        nid = str(int(row["cell_id"]))  # normalize by removing leading zeros
        row.to_frame().T.to_csv(os.path.join(output_dir, f"{nid}.csv"), index=False, encoding="gbk")


def get_neuron_meta(neuron_id: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Given a file (e.g. image or swc), return the neuron's meta info.

    Args:
        neuron_id (int): The neuron ID to look up.
        use_cache (bool): Whether to use cached metadata files. Defaults to True.


    Returns:
        pd.DataFrame: A single-row DataFrame containing neuron metadata.
    """
    # 必须是纯数字
    if not isinstance(neuron_id, int):
        raise ValueError("Neuron ID must be an integer.")

    local_path = os.path.join(NEURON_META_DIR, f"{neuron_id}.csv")

    if os.path.exists(local_path) and use_cache:
        return pd.read_csv(local_path, encoding="gbk")

    # fallback: search in meta CSV and cache it
    if(NEURON_META_INFO_PATH.endswith(".xlsx")):
        full_df = pd.read_excel(NEURON_META_INFO_PATH, engine='openpyxl')
    elif(NEURON_META_INFO_PATH.endswith(".csv")):
        full_df = pd.read_csv(NEURON_META_INFO_PATH, encoding="gbk")
    else:
        raise ValueError("Unsupported metadata file format. Must be .csv or .xlsx.")
    # match = full_df[full_df["cell_id"].astype(str) == neuron_id]
    match = full_df[full_df["Cell ID"].astype(int) == neuron_id]
    if match.empty:
        raise ValueError(f"Neuron ID {neuron_id} not found in metadata.")

    if(use_cache):
        os.makedirs(NEURON_META_DIR, exist_ok=True)
        match.to_csv(local_path, index=False, encoding="gbk")
    return match


def get_tile_id(neuron_id):
    """
    Get the tile ID for a given neuron ID.

    Args:
        neuron_id (int): The ID of the neuron.

    Returns:
        str: The tile ID.
    """
    neuron_meta = get_neuron_meta(neuron_id)
    if("document_name" in neuron_meta.columns):
        return neuron_meta["document_name"].values[0]
    elif("PTRS(B)" in neuron_meta.columns):
        return neuron_meta["PTRS(B)"].values[0]
    elif("PTRSB" in neuron_meta.columns):
        return neuron_meta["PTRSB"].values[0]
    else:
        # raise ValueError(f"Cannot find tile ID for neuron ID {neuron_id}.")
        print("warning: cannot find tile ID for neuron ID ", neuron_id)
        return None


def get_xy_z_resolution(neuron_id: int) -> tuple:
    """
    Get the XY and Z resolution for a given neuron ID.

    Args:
        neuron_id (int): The neuron ID to look up.
    Returns:
        tuple: A tuple containing (xy_resolution, z_resolution).
    """
    meta = get_neuron_meta(neuron_id)
    if("xy_resolution" in meta.columns and "z_resolution" in meta.columns):
        xy_resolution = float(meta["xy_resolution"].values[0])
        z_resolution = float(meta["z_resolution"].values[0])
    elif("xy拍摄分辨率(*10e-3μm/px)" in meta.columns and "z拍摄分辨率(*10e-3μm/px)" in meta.columns):
        xy_resolution = float(meta["xy拍摄分辨率(*10e-3μm/px)"].values[0])
        z_resolution = float(meta["z拍摄分辨率(*10e-3μm/px)"].values[0])
    else:
        raise ValueError("Resolution columns not found in metadata.")

    # is 纯数字
    if(isinstance(xy_resolution, str) and xy_resolution.replace('.', '', 1).isdigit()):
        xy_resolution = int(xy_resolution)
    if(isinstance(z_resolution, str) and z_resolution.replace('.', '', 1).isdigit()):
        z_resolution = int(z_resolution)
    return xy_resolution, z_resolution


def get_soma_pos(neuron_id: int) -> tuple:
    """
    Get the soma position for a given neuron ID.

    Args:
        neuron_id (int): The neuron ID to look up.
    Returns:
        tuple: A tuple containing (soma_x, soma_y, soma_z).
    """
    meta = get_neuron_meta(neuron_id)
    if("soma_x" in meta.columns and "soma_y" in meta.columns and "soma_z" in meta.columns):
        soma_x = float(meta["soma_x"].values[0])
        soma_y = float(meta["soma_y"].values[0])
        soma_z = float(meta["soma_z"].values[0])
    else:
        raise ValueError("Soma position columns not found in metadata.")

    # is 纯数字
    if(isinstance(soma_x, str) and soma_x.replace('.', '', 1).isdigit()):
        soma_x = float(soma_x)
    if(isinstance(soma_y, str) and soma_y.replace('.', '', 1).isdigit()):
        soma_y = float(soma_y)
    if(isinstance(soma_z, str) and soma_z.replace('.', '', 1).isdigit()):
        soma_z = float(soma_z)
    return soma_x, soma_y, soma_z


def get_source_v3d_img_file(neuron_id : int) -> int:
    """
    Get the source V3D file path for a given neuron ID.

    Args:
        neuron_id (int): The neuron ID to look up.
    Returns:
        str: The source V3D file path.
    """
    meta = get_neuron_meta(neuron_id)
    if "v3dpbd_file" in meta.columns:
        cur_path =  meta["v3dpbd_file"].values[0]
        if(not "Cell_Image/" in cur_path):
            raise ValueError("Unexpected V3D file path format.")
        cur_path = cur_path.replace("Cell_Image/", "")
        full_path = os.path.join(V3D_IMAGE_ROOT, cur_path)
        return full_path
    else:
        raise ValueError("Source V3D file column not found in metadata.")


if __name__ == "__main__":
    # Example usage
    # split_neuron_meta_table()
    neuron_meta = get_neuron_meta("/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/compare_origin_and_retrace/image_03569.png")
    print(neuron_meta)
    print(neuron_meta["PTRSB"].values[0])
