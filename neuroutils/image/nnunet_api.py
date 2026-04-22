import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import argparse

import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy, preprocess_fromfiles_save_to_queue
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

import multiprocessing
import queue
from torch.multiprocessing import Event, Process, Queue, Manager

from time import sleep
from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager



NODE_NUM=3
GPUS_PER_NODE=2
NUM_PARTS=NODE_NUM*GPUS_PER_NODE

def my_preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                          list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                          output_filenames_truncated: Union[None, List[str]],
                                          plans_manager: PlansManager,
                                          dataset_json: dict,
                                          configuration_manager: ConfigurationManager,
                                          target_queue: Queue,
                                          done_event: Event,
                                          abort_event: Event,
                                          verbose: bool = False):
    label_manager = plans_manager.get_label_manager(dataset_json)
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    for idx in range(len(list_of_lists)):
        try:
            data, seg, data_properties = preprocessor.run_case(list_of_lists[idx],
                                                               list_of_segs_from_prev_stage_files[
                                                                   idx] if list_of_segs_from_prev_stage_files is not None else None,
                                                               plans_manager,
                                                               configuration_manager,
                                                               dataset_json)
        except Exception as e:
            print(f"Failed to preprocess {list_of_lists[idx]}")
            print(e)
            continue
        if list_of_segs_from_prev_stage_files is not None and list_of_segs_from_prev_stage_files[idx] is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

        item = {'data': data, 'data_properties': data_properties,
                'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None}
        success = False
        while not success:
            try:
                if abort_event.is_set():
                    return
                target_queue.put(item, timeout=0.01)
                success = True
            except queue.Full:
                pass
    done_event.set()

def my_preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                        list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                        output_filenames_truncated: Union[None, List[str]],
                                        plans_manager: PlansManager,
                                        dataset_json: dict,
                                        configuration_manager: ConfigurationManager,
                                        num_processes: int,
                                        pin_memory: bool = False,
                                        verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=my_preprocess_fromfiles_save_to_queue,
                             args=(
                                 list_of_lists[i::num_processes],
                                 list_of_segs_from_prev_stage_files[
                                 i::num_processes] if list_of_segs_from_prev_stage_files is not None else None,
                                 output_filenames_truncated[
                                 i::num_processes] if output_filenames_truncated is not None else None,
                                 plans_manager,
                                 dataset_json,
                                 configuration_manager,
                                 queue,
                                 event,
                                 abort_event,
                                 verbose
                             ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # 取出队列中的任务
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            # 检查所有进程是否活跃
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue

        # 如果设置了 pin_memory，执行 pin_memory 操作
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]

        # 返回当前处理的任务项
        yield item

    [p.join() for p in processes]

class My_nnUNetPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super(My_nnUNetPredictor, self).__init__(*args, **kwargs)

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return my_preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                   output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                   self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                   self.verbose_preprocessing)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            # for preprocessed in data_iterator:
            while(True):
                try:
                    preprocessed = next(data_iterator) # 只要有一个data出问题，迭代器直接挂掉
                except StopIteration:
                    print('StopIteration')
                    break
                except Exception as e:
                    print(e)
                    continue
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # 防止GPU预测速度过快导致磁盘被大量npy文件淹没
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # 异步导出分割结果
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    # 异步导出分割结果，不保存概率
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')

            ret = [i.get()[0] for i in r]

            if isinstance(data_iterator, MultiThreadedAugmenter):
                data_iterator._finish()


        # 清空 lru cache 和设备缓存
        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行 nnUNetPredictor，并指定 CUDA 设备")
    parser.add_argument('--cuda', type=int, default=0, help='指定要使用的 CUDA 设备 ID (默认: 0)')
    # parser.add_argument('--gpu_node', type=int, default=0, help='目前运行的节点 ID (默认: 0)')
    parser.add_argument('--num_parts', type=int, default=1, help='分片数量 (默认: 1)')
    parser.add_argument('--part_id', type=int, default=0, help='当前分片 ID (默认: 0)')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}')
    # part_id = args.gpu_node * 2 + args.cuda
    num_parts = args.num_parts
    part_id = args.part_id


    # instantiate the nnUNetPredictor
    predictor = My_nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        # join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres'),
        "/data/disk/nnUNet_local/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    # predictor.predict_from_files("/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_img",
    #                              "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_seg",
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              folder_with_segs_from_prev_stage=None, num_parts=NUM_PARTS, part_id=part_id)
    predictor.predict_from_files("data/disk/trace/60k_to_120k/down_sampled_soma",
                                 "data/disk/trace/60k_to_120k/down_sampled_soma_seg_prob",
                                 save_probabilities=True, overwrite=False,
                                 num_processes_preprocessing=8, num_processes_segmentation_export=8,
                                 folder_with_segs_from_prev_stage=None, num_parts=num_parts, part_id=part_id, )