from neuroutils.swc.parser import prune_swc
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from neuroutils.config.settings import DEFAULT_NUM_THREADS

def do_prune(in_file, out_file):
    """
    Prune a SWC file to remove small branches.
    :param in_file: Input SWC file path
    :param out_file: Output SWC file path
    """
    # Prune the SWC file
    if(os.path.exists(out_file)):
        if not check_if_valid_swc(out_file):
            print(f"Invalid SWC file generated: {out_file}")
        return

    prune_swc(in_file, out_file)

    # 检查是否合法



    # try:
    #     prune_swc(in_file, out_file)
    # except Exception as e:
    #     print(f"Error pruning SWC file: {e}")
    #     return

def check_if_valid_swc(file_path):
    """
    Check if the SWC file is valid.
    :param file_path: Path to the SWC file
    :return: True if valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) != 7:
                    return False
        return True
    except Exception as e:
        print(f"Error reading SWC file: {e}")
        return False

if __name__ == "__main__":
    in_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample"
    out_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_resample_prune"
    files = [f for f in os.listdir(in_dir) if f.endswith('.swc')]
    os.makedirs(out_dir, exist_ok=True)

    Parallel(n_jobs=DEFAULT_NUM_THREADS)(
        delayed(do_prune)(os.path.join(in_dir, file), os.path.join(out_dir, file))
        for file in tqdm(files)
    )

