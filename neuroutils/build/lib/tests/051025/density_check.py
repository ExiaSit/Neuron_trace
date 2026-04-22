"""
density_check.py
This script checks the density of a given dataset and generates a report.
对于数据集中的每一个神经元，计算距离它最近的神经元的距离，并做可视化、
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from neuroutils.meta.tile import find_nearest_soma
from neuroutils.meta.mapping import extract_neuron_id
from joblib import Parallel, delayed
from neuroutils.config.settings import DEFAULT_NUM_THREADS
import shutil
from tqdm import tqdm


def find_nearest_neuron_distance(swc_path, save_path=None):
    """
    Find the distance from a neuron to its nearest soma and optionally cache the result.

    If a cache file exists and is readable, return the cached result.
    Otherwise, compute the distance and store the result to cache (if save_path is provided).

    Args:
        swc_path (str): Path to the neuron's SWC file.
        save_path (str, optional): Path to save or read cached distance. Defaults to None.

    Returns:
        float or None: The nearest distance if successful, or None on failure or if skipped.
    """
    try:
        neuron_id = int(extract_neuron_id(swc_path))
        if neuron_id < 14000 or neuron_id > 62000:
            return None
    except Exception as e:
        print(f"Error extracting neuron_id from {swc_path}: {e}")
        return None

    # Step 1: Try loading cached result
    if save_path and os.path.exists(save_path):
        try:
            return float(np.load(save_path, allow_pickle=True))
        except Exception as e:
            print(f"Warning: failed to load cached result from {save_path}: {e}")

    # Step 2: Compute distance
    try:
        _, nearest_distance = find_nearest_soma(neuron_id)

        # Step 3: Cache result if path provided
        if save_path:
            try:
                np.save(save_path, nearest_distance)
            except Exception as e:
                print(f"Warning: failed to save result to {save_path}: {e}")

        return nearest_distance

    except Exception as e:
        print(f"Error computing distance for {swc_path}: {e}")

        # Save error message to cache if needed
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write(f"Error: {e}")
            except Exception as e:
                print(f"Warning: failed to write error to {save_path}: {e}")

        return None

def check_density(ws_dir, dataset_path):
    # Replace with your dataset path
    swcs = [f for f in os.listdir(dataset_path) if f.endswith('.swc')]
    os.makedirs(os.path.join(ws_dir, "dist"), exist_ok=True)
    save_files = [os.path.join(ws_dir, f"dist/{swc.split('.')[0]}_dist.npy") for swc in swcs]

    dist_list = []
    # Parallel processing to find nearest neuron distances
    dist_list = Parallel(n_jobs=DEFAULT_NUM_THREADS)(
        delayed(find_nearest_neuron_distance)(os.path.join(dataset_path, swc), save_path) for swc, save_path in zip(swcs, save_files)
    )
    # Filter out None values
    dist_list = [dist for dist in dist_list if dist is not None]
    print(f"Number of valid distances: {len(dist_list)}, {len(swcs) - len(dist_list)} invalid distances")
    # # 保存临时结果
    # np.save(os.path.join(ws_dir, "dist_list.npy"), dist_list)

    # visualize the distribution of distances
    plt.figure(figsize=(4, 4))
    plt.hist(dist_list, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Nearest Neuron Distances')
    plt.xlabel('Distance (um)')
    plt.ylabel('Frequency')
    # plt.show()
    plt.savefig(os.path.join(ws_dir, "hist.jpg"))
    plt.close()

    dist_threshold = 100
    # Filter distances based on threshold
    filtered_distances = [dist for dist in dist_list if dist > dist_threshold]
    print(f"Number of distances greater than {dist_threshold}: {len(filtered_distances)}")

    '''
    Number of valid distances: 11957, 12734 invalid distances
    Number of distances greater than 100: 4246
    '''

def filter_neuron(swc_path, target_path, threshold=100, size_threshold=2):
    '''
    找到满足条件的swc，并转存到新文件中
    条件1：与最近的neuron的距离大于threshold 100um
    条件2：文件大小大于threshold 2kb
    '''
    neuron_id = int(extract_neuron_id(swc_path))
    if neuron_id < 14000 or neuron_id > 62000:
        return
    # Step 1: Try loading cached result
    try:
        _, nearest_distance = find_nearest_soma(neuron_id)
    except Exception as e:
        print(f"Error computing distance for {swc_path}: {e}")
        return

    # Step 2: Check file size
    file_size = os.path.getsize(swc_path) / 1024  # Convert to KB
    if nearest_distance > threshold and file_size > size_threshold:
        # Step 3: Save the neuron to the target path
        shutil.copy(swc_path, target_path)

def filter_neurons_in_directory(source_dir, target_dir, threshold=100, size_threshold=1000):
    '''
    遍历目录，找到满足条件的swc，并转存到新文件中
    条件1：与最近的neuron的距离大于threshold 100um
    条件2：文件大小大于threshold 1000kb
    '''
    os.makedirs(target_dir, exist_ok=True)
    swcs = [f for f in os.listdir(source_dir) if f.endswith('.swc')]
    for swc in tqdm(swcs):
        source_path = os.path.join(source_dir, swc)
        target_path = os.path.join(target_dir, swc)
        filter_neuron(source_path, target_path, threshold, size_threshold)

if __name__ == "__main__":
    ws_dir = "/data2/kfchen/temp/051025"
    dataset_path = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_retrace_swcs"

    # check_density(ws_dir, dataset_path)

    target_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled/retrace_swcs_filtered"
    os.makedirs(target_dir, exist_ok=True)
    filter_neurons_in_directory(dataset_path, target_dir, threshold=100, size_threshold=2)





