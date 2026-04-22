
import os
import re
from neuroutils.config.settings import new_old_id_map_file
import pandas as pd

def extract_neuron_id(filename: str) -> int:
    """
    Extract a pure numeric neuron ID (e.g., '00123') from a given filename.
    Keeps leading zeros and ignores prefixes/suffixes.

    Args:
       filename (str): File name or full path (e.g., 'image_00123_0000.tif').

    Returns:
       str: Pure numeric neuron ID (e.g., '00123').

    Raises:
       ValueError: If no numeric ID is found.
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    # 搜索所有纯数字子串
    numeric_matches = re.findall(r'\d+', name_without_ext)
    # print(f"Extracted numeric matches: {numeric_matches}, {name_without_ext}")  # Debugging line
    if numeric_matches:
        # 返回第一个匹配，通常是编号
        return int(numeric_matches[0])

    raise ValueError(f"Could not find numeric neuron ID in: {filename}")


def new_old_cell_id_mapping(id, old_to_new=False):
    """
    Map cell IDs between old and new systems using an Excel mapping file.
    :param id:
    :param old_to_new:
    :return:
    """
    id_map_file = new_old_id_map_file
    id_map_df = pd.read_excel(id_map_file)[["cell_id_backUp", "cell_id"]]
    # 只保留前15000行
    id_map_df = id_map_df.iloc[:15000, :]
    # rename columns
    id_map_df = id_map_df.rename(columns={"cell_id_backUp": "old_id", "cell_id": "new_id"})
    # to map
    # id_map = {}
    # for i in range(len(id_map_df)):
    #     id_map[id_map_df.iloc[i, 1]] = id_map_df.iloc[i, 0] # new_id -> old_id

    if old_to_new:
        mapped_id = id_map_df.loc[id_map_df["old_id"] == id, "new_id"]
    else:
        mapped_id = id_map_df.loc[id_map_df["new_id"] == id, "old_id"]

    print("id: ", id, " mapped_id: ", mapped_id.values[0])  # Debugging line
    if not mapped_id.empty:
        return mapped_id.values[0]
    else:
        return None