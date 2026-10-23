##########################################################
#Author:          Yufeng Liu / Adjusted version
#Create time:     2025-09-27
#Description:     Neuron pruning and re-connection pipeline
##########################################################
import os
import re
import sys
import time
import json
import fcntl
import cv2
cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from tqdm import tqdm
import contextlib
import logging
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cc3d
import edt
import networkx as nx
import numpy as np
import pandas as pd
from skimage import graph as skgraph
from scipy.spatial import cKDTree
from skimage.graph import route_through_array, MCP_Geometric
from collections import deque

# Package-local dependencies and the same configurable paths as the orchestrator.
from pipeline_config import BASE_DIR, PATHS as PIPELINE_PATHS
PATHS = dict(PIPELINE_PATHS)
PATHS["gcut_selected_swc_dir"] = PATHS["gcut_resampled_swc_dir"]
PATHS["soma_crop_dir"] = PATHS["soma_reader_json_dir"]

sys.path.insert(0, PATHS["pylib"])
from swc_handler import parse_swc, write_swc, flip_swc, shift_swc
from morph_topo.morphology import Morphology, Topology

sys.path.insert(0, PATHS["TraceFlow"])
from core.visualization.debug_vis import plot_swc
from core.io.image_parser import ImageParser
from core.visualization.debug_vis import plot_image_2D
import glob
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


class SomaOnlySWCError(ValueError):
    """Raised when an SWC contains a soma root but no neurite nodes."""


class SingleNodeSWCError(ValueError):
    """Raised when an SWC contains one non-soma or otherwise invalid node."""


def classify_processing_exception(exc):
    """Return a stable CSV/log category and a concise user-facing reason."""
    if isinstance(exc, SomaOnlySWCError):
        return "soma_only", str(exc)
    if isinstance(exc, SingleNodeSWCError):
        return "single_node_invalid", str(exc)
    if isinstance(exc, FileNotFoundError):
        return "missing_file", str(exc)
    if isinstance(exc, MemoryError):
        return "memory_error", "处理过程中内存不足"
    if isinstance(exc, KeyError):
        return "missing_key_or_node", f"缺少必要的键、节点或元数据：{exc}"
    if isinstance(exc, ValueError):
        return "invalid_data", str(exc) or "输入数据或参数无效"
    return "unexpected_exception", f"{type(exc).__name__}: {exc}"


def classify_precheck_result(message):
    """Classify a failed preflight check as either skipped or failed."""
    message = str(message)
    if message.startswith("Already processed:"):
        return "skipped", "already_processed"
    if message.startswith("Skipped: Cell ID"):
        return "skipped", "below_start_cell_id"
    if message.startswith("Mask input missing:"):
        return "error", "missing_mask"
    if message.startswith("Raw image missing:"):
        return "error", "missing_raw_image"
    if "empty/invalid" in message:
        return "error", "empty_or_invalid_swc"
    if message.startswith("Selected G-Cut SWC missing:"):
        return "error", "missing_selected_swc"
    if message.startswith("Cannot extract cell ID"):
        return "error", "invalid_filename"
    if message.startswith("Not implemented:"):
        return "error", "unsupported_configuration"
    return "error", "precheck_failed"


def write_run_statistics_csv(
    csv_path,
    run_started_at,
    run_finished_at,
    records,
    total_discovered,
    pending_submitted,
):
    """Write run summary, failure category counts, and per-file details."""
    normalized_records = []
    for record in records:
        status = str(record.get("status", "failed"))
        if status == "error":
            status = "failed"
        normalized_records.append({
            "file": str(record.get("file", "")),
            "status": status,
            "category": str(record.get("category", "")),
            "reason": str(record.get("reason", "")),
        })

    success_count = sum(r["status"] == "success" for r in normalized_records)
    failure_count = sum(r["status"] == "failed" for r in normalized_records)
    skipped_count = sum(r["status"] == "skipped" for r in normalized_records)

    failure_category_counts = {}
    for record in normalized_records:
        if record["status"] != "failed":
            continue
        category = record["category"] or "unclassified_failure"
        failure_category_counts[category] = (
            failure_category_counts.get(category, 0) + 1
        )

    shared = {
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "total_discovered": int(total_discovered),
        "pending_submitted": int(pending_submitted),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "skipped_count": int(skipped_count),
    }
    rows = [{
        "record_type": "run_summary",
        **shared,
        "status": "summary",
        "category": "all",
        "category_count": len(normalized_records),
        "file": "",
        "reason": "本次运行总体统计",
    }]
    for category, count in sorted(failure_category_counts.items()):
        rows.append({
            "record_type": "failure_category_summary",
            **shared,
            "status": "failed",
            "category": category,
            "category_count": int(count),
            "file": "",
            "reason": "该失败类别的文件数量",
        })
    for record in sorted(normalized_records, key=lambda item: item["file"]):
        rows.append({
            "record_type": "file_detail",
            **shared,
            "status": record["status"],
            "category": record["category"],
            "category_count": "",
            "file": record["file"],
            "reason": record["reason"],
        })

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {
        "csv_path": csv_path,
        "success_count": success_count,
        "failure_count": failure_count,
        "skipped_count": skipped_count,
        "failure_category_counts": failure_category_counts,
    }


def swc_has_nodes(swc_file):
    if not os.path.exists(swc_file) or os.path.getsize(swc_file) == 0:
        return False
    with open(swc_file, 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if len(line.split()) >= 7:
                return True
    return False


def prune_output_swc_path(input_imgfile, out_swc_dir):
    prefix = os.path.splitext(os.path.basename(input_imgfile))[0]
    return os.path.join(out_swc_dir, f'{prefix}.swc')


def extract_cell_id_from_gcut_swc(gcut_swc_file):
    """Extract the numeric cell ID from a selected G-Cut SWC filename.

    Examples
    --------
    image_100007.swc -> 100007
    image_100007_0000_gcut_soma_1_TARGET.swc -> 100007
    """
    stem = os.path.splitext(os.path.basename(gcut_swc_file))[0]
    match = re.search(r'(?:^|_)image_(\d+)(?:_|$)', stem, flags=re.IGNORECASE)
    if match is None:
        # Backward-compatible fallback for names such as ``100007_xxx.swc``.
        match = re.search(r'(?<!\d)(\d+)(?!\d)', stem)
    if match is None:
        raise ValueError(
            f"Cannot extract cell ID from selected G-Cut SWC filename: {gcut_swc_file}"
        )
    return int(match.group(1))


def resolve_image_by_stem(image_dir, stem):
    """Return an existing .tif/.tiff image path, or the expected .tif path."""
    for suffix in ('.tif', '.tiff', '.TIF', '.TIFF'):
        candidate = os.path.join(image_dir, f'{stem}{suffix}')
        if os.path.exists(candidate):
            return candidate
    return os.path.join(image_dir, f'{stem}.tif')


class NeuronGraphProcessor:
    def __init__(
        self, graph, coords_dict, sxyz, scale_xyz, conf_length=5,
        rejection_callback=None
    ):
        self.graph = graph
        self.coords_dict = coords_dict
        self.sxyz = sxyz
        self.scale_xyz = scale_xyz
        self.conf_length = conf_length
        self.rejection_callback = rejection_callback
        self.last_rejection = None

    def _detect_rebound(self, new_parent, current_intensity, lookback_nodes=15):
        """长程反弹检测：专杀缓慢爬升进入邻居胞体的串线"""
        curr = new_parent
        min_history_int = current_intensity

        for _ in range(lookback_nodes):
            if not self._has_single_predecessor(curr): break
            pred = list(self.graph.predecessors(curr))[0]
            past_intensity = self.graph.edges[pred, curr]['intensity']
            min_history_int = min(min_history_int, past_intensity)
            curr = pred

        # 如果在过去 15 个节点中出现过低谷，而现在的亮度爬升到了谷底的 2 倍以上，
        # 且绝对值暴增了 30 以上，说明它已经脱离了自然衰减，正在爬另一个细胞！切断！
        if current_intensity > (min_history_int * 2.0 + 30):
            return True
        return False

    def successive_check(self, old_parent, old_child, intensity, radius, max_radius,
                         path_dist, vn, parent_to_child):
        new_parent, new_child = (old_parent, old_child) if parent_to_child else (old_child, old_parent)
        self.last_rejection = None

        if not self._has_single_predecessor(new_parent): return True

        parent_edges = self._get_predecessor_edges(new_parent)
        if not parent_edges: return True

        path_dist1, vn1, radius1, max_radius1, intensity1 = parent_edges[0]

        if self._check_angle_and_radius(path_dist, path_dist1, vn1, vn, radius, radius1, max_radius):
            self.last_rejection = (
                "分支半径绝对值过大",
                "NeuronGraphProcessor._check_angle_and_radius",
            )
            return False
        if self._check_intensity_with_parent(intensity, radius, path_dist, parent_edges):
            self.last_rejection = (
                "相邻长段亮度突然升高",
                "NeuronGraphProcessor._check_intensity_with_parent",
            )
            return False

        # ===================================================
        # 【新增调用】：执行长程爬升拦截
        # ===================================================
        if self._detect_rebound(new_parent, intensity):
            self.last_rejection = (
                "长程亮度从低谷异常反弹",
                "NeuronGraphProcessor._detect_rebound",
            )
            return False

        new_parent2 = list(self.graph.predecessors(new_parent))[0]
        if not self._has_single_predecessor(new_parent2): return True

        passed = self._check_intensity_and_radius(
            intensity, radius, path_dist, parent_edges
        )
        if not passed:
            self.last_rejection = (
                "当前亮度高于前两代分支",
                "NeuronGraphProcessor._check_intensity_and_radius",
            )
        return passed

    def _has_single_predecessor(self, node):
        return self.graph.has_node(node) and self.graph.in_degree(node) == 1

    def _detect_valley_crossing(self, new_parent, current_intensity, lookback_nodes=20):
        """长程回溯：检查是否跨越了深暗的背景沟壑后，又爬上了另一个高亮结构"""
        curr = new_parent
        min_intensity = current_intensity

        # 往回遍历最多 lookback_nodes 代祖先
        for _ in range(lookback_nodes):
            if not self._has_single_predecessor(curr):
                break
            pred = list(self.graph.predecessors(curr))[0]
            edge_data = self.graph.edges[pred, curr]
            past_intensity = edge_data['intensity']
            min_intensity = min(min_intensity, past_intensity)
            curr = pred

        # 核心拦截逻辑：如果在过去一段距离内，最暗的地方极暗（比如背景水平），
        # 而当前节点的亮度竟然是谷底的 3 倍以上，且绝对增量 > 30，直接切断！
        if current_intensity > (min_intensity * 3.0 + 30):
            return True # 发现深谷反弹，判定为跨细胞串线！

        return False

    def _get_predecessor_edges(self, node):
        edges_data = []
        current = node
        for _ in range(2):
            if not self._has_single_predecessor(current): break
            pred = list(self.graph.predecessors(current))[0]
            edge_data = self.graph.edges[pred, current]
            edges_data.append((edge_data['path_dist'], edge_data['vn'], edge_data['radius'], edge_data['max_radius'], edge_data['intensity']))
            current = pred
        return edges_data

    def _check_angle_and_radius(self, path_dist, path_dist1, vn1, vn, radius, radius1, max_radius):
        # 连续段的 107.5 度硬切已移除；角度异常改由二叉分叉 QC 联合判定。
        radius_hard = (radius > 12) or (max_radius > 15)
        return radius_hard

    def _check_intensity_with_parent(self, intensity, radius, path_dist, parent_edges):
        path_dist1, _, radius1, _, intensity1 = parent_edges[0]
        if (path_dist > self.conf_length) and (path_dist1 > self.conf_length):
            # 4. 亮度限制大幅放宽：允许真实的荧光“串珠”闪烁！
            # 只有当亮度发生了极其离谱的暴增（例如涨了 50% 且绝对值也增加了）才拦截
            if intensity > (intensity1 * 1.5 + 20):
                return True
        return False

    def _check_intensity_and_radius(self, intensity, radius, path_dist, parent_edges):
        if len(parent_edges) < 2:
            return True

        path_dist1, _, radius1, _, intensity1 = parent_edges[0]
        path_dist2, _, radius2, _, intensity2 = parent_edges[1]

        # 5. 废除“单调递减”的暴政！不再要求 intensity <= intensity1
        # 只要没有暴增到碰见其他胞体的程度，都让它活下去
        if path_dist > self.conf_length:
            max_int = max(intensity1, intensity2)
            if intensity > (max_int * 1.5 + 20):
                return False

        return True

    def add_edge_with_attributes(self, parent, child, weight, attributes):
        self.graph.add_edge(parent, child, weight=weight)
        nx.set_edge_attributes(self.graph, {(parent, child): attributes})

    def process_child_node(self, parent, child, fseg_dict, visited, topo, parent_to_child=True):
        if child in visited:
            return False
        if child not in fseg_dict:
            if self.rejection_callback:
                self.rejection_callback(
                    child,
                    "拓扑段未进入特征图（通常为暗谷过滤）",
                    "process_neuron_morphology: fseg_dict",
                )
            return False
        intensity, radius, max_radius, vn, path_dist, euc_dist = fseg_dict[child]

        if not self.successive_check(
            parent, child, intensity, radius, max_radius, path_dist, vn,
            parent_to_child
        ):
            if self.rejection_callback and self.last_rejection:
                reason, code = self.last_rejection
                self.rejection_callback(child, reason, code)
            return False

        attributes = {'intensity': intensity, 'radius': radius, 'max_radius': max_radius, 'vn': vn if parent_to_child else -vn, 'path_dist': path_dist, 'euc_dist': euc_dist}
        weight = 1 if parent_to_child else -1
        self.add_edge_with_attributes(parent, child, weight, attributes)
        visited.add(child)
        return True

    def traverse_sequential_branch(self, start_id, fseg_dict, visited, topo):
        processing = [start_id]
        while processing:
            current = processing.pop()
            parent = topo.pos_dict[current][6]
            good = self.process_child_node(parent, current, fseg_dict, visited, topo, parent_to_child=True)
            if not good: continue
            for child in topo.child_dict.get(current, []):
                if child not in visited: processing.append(child)

    def traverse_inverse_branch(self, id0, id1, fseg_dict, visited, topo):
        intensity, radius, max_radius, vn, path_dist, euc_dist = fseg_dict[id0]
        attributes = {'intensity': intensity, 'radius': radius, 'max_radius': max_radius, 'vn': -vn, 'path_dist': path_dist, 'euc_dist': euc_dist}
        self.add_edge_with_attributes(id0, id1, -1, attributes)
        visited.update([id0, id1])
        processing = [id1]

        while processing:
            current = processing.pop()
            for child in topo.child_dict.get(current, []):
                if self.process_child_node(current, child, fseg_dict, visited, topo, parent_to_child=True):
                    processing.append(child)

            parent_id = topo.pos_dict[current][6]
            if parent_id != -1 and parent_id not in visited:
                intensity, radius, max_radius, vn, path_dist, euc_dist = fseg_dict[current]
                if self.successive_check(parent_id, current, intensity, radius, max_radius, path_dist, -vn, parent_to_child=False):
                    attributes = {'intensity': intensity, 'radius': radius, 'max_radius': max_radius, 'vn': -vn, 'path_dist': path_dist, 'euc_dist': euc_dist}
                    self.add_edge_with_attributes(current, parent_id, -1, attributes)
                    processing.append(parent_id)
            visited.add(current)


class SWCQCNeuron:
    """Minimal in-memory adapter for the QC_utils geometry/topology operations."""

    def __init__(self, swc):
        self.initialize(swc)

    def initialize(self, swc):
        self.swc = np.array(swc, dtype=object)
        self.nid_hash = {int(node[0]): idx for idx, node in enumerate(self.swc)}
        self.children = [[] for _ in range(len(self.swc))]
        for idx, node in enumerate(self.swc):
            parent_idx = self.nid_hash.get(int(node[6]))
            if parent_idx is not None:
                self.children[parent_idx].append(idx)
        self.soma_indices = [
            idx for idx, node in enumerate(self.swc)
            if int(node[1]) == 1 and int(node[6]) == -1
        ]
        if not self.soma_indices:
            self.soma_indices = [
                idx for idx, node in enumerate(self.swc)
                if int(node[6]) not in self.nid_hash
            ]
        self.bifur_indices = [
            idx for idx, child_indices in enumerate(self.children)
            if len(child_indices) > 1
        ]
        return self

    def remove_indices(self, indices):
        if indices:
            indices = {
                int(idx) for idx in indices
                if 0 <= int(idx) < len(self.swc)
            }
            removed_ids = {int(self.swc[idx][0]) for idx in indices}
            parent_by_id = {
                int(node[0]): int(node[6]) for node in self.swc
            }

            # Rewiring in the QC passes can be transitive. For example, a
            # child may first be moved from node A to its near-duplicate B,
            # and B may then also be removed as a duplicate of C. The cached
            # ``children`` lists are intentionally rebuilt only after a QC
            # pass, so they do not contain that newly moved child. Resolve the
            # final parent chain here before deleting anything, ensuring every
            # surviving node points to the nearest surviving ancestor.
            for idx, node in enumerate(self.swc):
                if idx in indices:
                    continue
                parent_id = int(node[6])
                visited_removed_parents = set()
                while parent_id in removed_ids:
                    if parent_id in visited_removed_parents:
                        raise ValueError(
                            "Cycle detected while resolving removed SWC "
                            f"parent chain for node {int(node[0])}"
                        )
                    visited_removed_parents.add(parent_id)
                    parent_id = parent_by_id[parent_id]
                node[6] = parent_id

            keep = np.ones(len(self.swc), dtype=bool)
            keep[list(indices)] = False
            self.initialize(self.swc[keep])

    def as_swc_list(self):
        return [
            (
                int(node[0]), int(node[1]), float(node[2]), float(node[3]),
                float(node[4]), float(node[5]), int(node[6])
            )
            for node in self.swc
        ]


def _qc_squared_distance(point1, point2):
    diff = np.asarray(point1, dtype=float) - np.asarray(point2, dtype=float)
    return float(np.dot(diff, diff))


def _qc_is_sharp_turn(n1, n2, n3, degree_threshold=90.0):
    v1 = np.asarray(n2[2:5], dtype=float) - np.asarray(n1[2:5], dtype=float)
    v2 = np.asarray(n3[2:5], dtype=float) - np.asarray(n2[2:5], dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return False
    return np.dot(v1, v2) / (norm1 * norm2) <= np.cos(np.radians(degree_threshold))


def _qc_accumulate_parent_direction(neuron, start_idx, min_length):
    start_pos = np.asarray(neuron.swc[start_idx][2:5], dtype=float)
    cur_idx, accumulated, end_pos = start_idx, 0.0, start_pos
    for _ in range(len(neuron.swc)):
        if cur_idx != start_idx and len(neuron.children[cur_idx]) > 1:
            break
        parent_idx = neuron.nid_hash.get(int(neuron.swc[cur_idx][6]))
        if parent_idx is None:
            break
        parent_pos = np.asarray(neuron.swc[parent_idx][2:5], dtype=float)
        accumulated += np.linalg.norm(
            np.asarray(neuron.swc[cur_idx][2:5], dtype=float) - parent_pos
        )
        end_pos, cur_idx = parent_pos, parent_idx
        if accumulated >= min_length:
            break
    return start_pos - end_pos, accumulated


def _qc_accumulate_child_direction(neuron, start_idx, first_child_idx, min_length):
    start_pos = np.asarray(neuron.swc[start_idx][2:5], dtype=float)
    end_pos = np.asarray(neuron.swc[first_child_idx][2:5], dtype=float)
    accumulated = np.linalg.norm(end_pos - start_pos)
    cur_idx = first_child_idx
    for _ in range(len(neuron.swc)):
        if accumulated >= min_length or len(neuron.children[cur_idx]) != 1:
            break
        next_idx = neuron.children[cur_idx][0]
        next_pos = np.asarray(neuron.swc[next_idx][2:5], dtype=float)
        accumulated += np.linalg.norm(
            next_pos - np.asarray(neuron.swc[cur_idx][2:5], dtype=float)
        )
        end_pos, cur_idx = next_pos, next_idx
    return end_pos - start_pos, accumulated


def _qc_angle_degree(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return None
    cosine = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _qc_collect_subtree(neuron, root_idx, nodes_to_remove):
    stack = [root_idx]
    while stack:
        idx = stack.pop()
        if idx in nodes_to_remove:
            continue
        nodes_to_remove.add(idx)
        stack.extend(neuron.children[idx])


def qc_remove_bad_binary_bifurcation(
    neuron, window_length=5.0, child_child_angle_threshold=140.0,
    parent_child_angle_threshold=70.0
):
    nodes_to_remove = set()
    for bifur_idx in neuron.bifur_indices:
        children = neuron.children[bifur_idx]
        if len(children) != 2 or bifur_idx in nodes_to_remove:
            continue
        child1_idx, child2_idx = children
        parent_vec, parent_len = _qc_accumulate_parent_direction(
            neuron, bifur_idx, window_length
        )
        child1_vec, child1_len = _qc_accumulate_child_direction(
            neuron, bifur_idx, child1_idx, window_length
        )
        child2_vec, child2_len = _qc_accumulate_child_direction(
            neuron, bifur_idx, child2_idx, window_length
        )
        if min(parent_len, child1_len, child2_len) < window_length:
            continue
        child_angle = _qc_angle_degree(child1_vec, child2_vec)
        parent_child1_angle = _qc_angle_degree(parent_vec, child1_vec)
        parent_child2_angle = _qc_angle_degree(parent_vec, child2_vec)
        if None in (child_angle, parent_child1_angle, parent_child2_angle):
            continue
        if (
            child_angle > child_child_angle_threshold
            and parent_child1_angle > parent_child_angle_threshold
        ):
            _qc_collect_subtree(neuron, child2_idx, nodes_to_remove)
        if (
            child_angle > child_child_angle_threshold
            and parent_child2_angle > parent_child_angle_threshold
        ):
            _qc_collect_subtree(neuron, child1_idx, nodes_to_remove)
    neuron.remove_indices(nodes_to_remove)


def qc_remove_zigzag(neuron):
    nodes_to_remove = set()
    stack = deque(
        child_idx
        for soma_idx in neuron.soma_indices
        for child_idx in neuron.children[soma_idx]
    )
    while stack:
        cur_idx = stack.pop()
        parent1_idx = neuron.nid_hash.get(int(neuron.swc[cur_idx][6]))
        if parent1_idx is None:
            stack.extend(neuron.children[cur_idx])
            continue
        parent2_idx = neuron.nid_hash.get(int(neuron.swc[parent1_idx][6]))
        if parent2_idx is None:
            stack.extend(neuron.children[cur_idx])
            continue
        parent3_idx = neuron.nid_hash.get(int(neuron.swc[parent2_idx][6]))
        if parent3_idx is None:
            stack.extend(neuron.children[cur_idx])
            continue
        if (
            _qc_is_sharp_turn(
                neuron.swc[parent3_idx], neuron.swc[parent2_idx],
                neuron.swc[parent1_idx]
            )
            and _qc_is_sharp_turn(
                neuron.swc[parent2_idx], neuron.swc[parent1_idx],
                neuron.swc[cur_idx]
            )
        ):
            parent3_id = int(neuron.swc[parent3_idx][0])
            for child_idx in neuron.children[parent2_idx]:
                neuron.swc[child_idx][6] = parent3_id
            nodes_to_remove.add(parent2_idx)
        stack.extend(neuron.children[cur_idx])
    neuron.remove_indices(nodes_to_remove)


def qc_tune_branch(neuron):
    modified = False
    positions = np.asarray(neuron.swc[:, 2:5], dtype=float)
    stack = deque(
        child_idx
        for soma_idx in neuron.soma_indices
        for child_idx in neuron.children[soma_idx]
    )
    while stack:
        cur_idx = stack.pop()
        parent_idx = neuron.nid_hash.get(int(neuron.swc[cur_idx][6]))
        if parent_idx is None:
            continue
        if len(neuron.children[parent_idx]) > 1:
            stack.extend(neuron.children[cur_idx])
            continue
        gparent_idx = neuron.nid_hash.get(int(neuron.swc[parent_idx][6]))
        if gparent_idx is None:
            continue
        if len(neuron.children[gparent_idx]) == 1:
            stack.extend(neuron.children[cur_idx])
            continue
        ggparent_idx = neuron.nid_hash.get(int(neuron.swc[gparent_idx][6]))
        if ggparent_idx is None:
            stack.extend(neuron.children[cur_idx])
            continue

        min_dist, new_parent_idx = np.inf, None
        if not _qc_is_sharp_turn(
            neuron.swc[gparent_idx], neuron.swc[parent_idx], neuron.swc[cur_idx]
        ):
            min_dist = _qc_squared_distance(positions[parent_idx], positions[gparent_idx])
        if not _qc_is_sharp_turn(
            neuron.swc[ggparent_idx], neuron.swc[parent_idx], neuron.swc[cur_idx]
        ):
            distance = _qc_squared_distance(positions[parent_idx], positions[ggparent_idx])
            if distance < min_dist:
                min_dist, new_parent_idx = distance, ggparent_idx

        for sibling_idx in reversed(neuron.children[gparent_idx]):
            if sibling_idx == parent_idx:
                break
            if _qc_is_sharp_turn(
                neuron.swc[sibling_idx], neuron.swc[parent_idx], neuron.swc[cur_idx]
            ):
                continue
            distance = _qc_squared_distance(positions[parent_idx], positions[sibling_idx])
            if distance < min_dist:
                min_dist, new_parent_idx = distance, sibling_idx

        if new_parent_idx is not None:
            neuron.swc[parent_idx][6] = int(neuron.swc[new_parent_idx][0])
            modified = True
        stack.extend(neuron.children[cur_idx])
    if modified:
        neuron.initialize(neuron.swc)


def _qc_branch_path_length(neuron, root_idx, child_idx):
    total = np.linalg.norm(
        np.asarray(neuron.swc[child_idx][2:5], dtype=float)
        - np.asarray(neuron.swc[root_idx][2:5], dtype=float)
    )
    cur_idx = child_idx
    for _ in range(len(neuron.swc)):
        if len(neuron.children[cur_idx]) != 1:
            break
        next_idx = neuron.children[cur_idx][0]
        total += np.linalg.norm(
            np.asarray(neuron.swc[next_idx][2:5], dtype=float)
            - np.asarray(neuron.swc[cur_idx][2:5], dtype=float)
        )
        cur_idx = next_idx
    return total


def qc_remove_branch_towards_soma(
    neuron, min_path_length=10.0, window_length=5.0,
    towards_soma_degree=120.0
):
    if not neuron.soma_indices:
        return
    soma_center = np.mean(
        np.asarray(neuron.swc[neuron.soma_indices, 2:5], dtype=float), axis=0
    )
    cos_threshold = np.cos(np.radians(towards_soma_degree))
    nodes_to_remove = set()
    candidate_roots = set(neuron.bifur_indices) | set(neuron.soma_indices)
    for root_idx in candidate_roots:
        for child_idx in neuron.children[root_idx]:
            if child_idx in nodes_to_remove:
                continue
            if _qc_branch_path_length(neuron, root_idx, child_idx) <= min_path_length:
                continue
            branch_vec, branch_len = _qc_accumulate_child_direction(
                neuron, root_idx, child_idx, window_length
            )
            if branch_len < window_length:
                continue
            to_soma = soma_center - np.asarray(neuron.swc[root_idx][2:5], dtype=float)
            norm_branch, norm_soma = np.linalg.norm(branch_vec), np.linalg.norm(to_soma)
            if norm_branch == 0 or norm_soma == 0:
                continue
            cosine = np.dot(branch_vec, to_soma) / (norm_branch * norm_soma)
            if cosine >= cos_threshold:
                _qc_collect_subtree(neuron, child_idx, nodes_to_remove)
    neuron.remove_indices(nodes_to_remove)


def _qc_append_interpolated_parent(neuron, lower_idx, upper_idx, child_idx):
    lower = np.asarray(neuron.swc[lower_idx][2:5], dtype=float)
    upper = np.asarray(neuron.swc[upper_idx][2:5], dtype=float)
    child = np.asarray(neuron.swc[child_idx][2:5], dtype=float)
    segment = upper - lower
    norm2 = float(np.dot(segment, segment))
    t = 0.5 if norm2 == 0 else float(np.clip(np.dot(child - lower, segment) / norm2, 0, 1))
    position = lower + t * segment
    radius = (
        (1 - t) * float(neuron.swc[lower_idx][5])
        + t * float(neuron.swc[upper_idx][5])
    )
    new_id = int(max(int(node[0]) for node in neuron.swc)) + 1
    new_node = np.array([
        new_id, int(neuron.swc[lower_idx][1]), *position, radius,
        int(neuron.swc[upper_idx][0])
    ], dtype=object)
    neuron.initialize(np.vstack([neuron.swc, new_node]))
    return new_id


def qc_correct_multifurcation(neuron, window_length=5.0, max_iterations=500):
    for _ in range(max_iterations):
        soma_set = set(neuron.soma_indices)
        candidates = [
            idx for idx, children in enumerate(neuron.children)
            if len(children) > 2 and idx not in soma_set
        ]
        if not candidates:
            return
        multifur_idx = candidates[0]
        multifur_id = int(neuron.swc[multifur_idx][0])
        up_vec, up_len = _qc_accumulate_parent_direction(
            neuron, multifur_idx, window_length
        )
        if up_len < window_length:
            parent_idx = neuron.nid_hash.get(int(neuron.swc[multifur_idx][6]))
            if parent_idx is None:
                return
            up_vec = (
                np.asarray(neuron.swc[multifur_idx][2:5], dtype=float)
                - np.asarray(neuron.swc[parent_idx][2:5], dtype=float)
            )
        up_norm = np.linalg.norm(up_vec)
        if up_norm < 0.1:
            return

        child_info = []
        for child_idx in neuron.children[multifur_idx]:
            child_vec, child_len = _qc_accumulate_child_direction(
                neuron, multifur_idx, child_idx, window_length
            )
            if child_len < window_length:
                child_vec = (
                    np.asarray(neuron.swc[child_idx][2:5], dtype=float)
                    - np.asarray(neuron.swc[multifur_idx][2:5], dtype=float)
                )
            child_norm = np.linalg.norm(child_vec)
            if child_norm >= 0.1:
                cosine = float(np.dot(up_vec, child_vec) / (up_norm * child_norm))
                child_info.append((cosine, child_len, child_idx, child_vec, child_norm))
        if len(child_info) < 3:
            return
        child_info.sort(key=lambda item: (item[0], item[1]), reverse=True)
        main_idx, main_vec, main_norm = child_info[0][2:5]
        side_info = sorted(
            (
                float(np.dot(main_vec, child_vec) / (main_norm * child_norm)),
                child_idx
            )
            for _, _, child_idx, child_vec, child_norm in child_info[1:]
        )
        move_child_ids = [
            int(neuron.swc[child_idx][0]) for _, child_idx in side_info[1:]
        ]
        moved = False
        for child_id in move_child_ids:
            multifur_idx = neuron.nid_hash.get(multifur_id)
            child_idx = neuron.nid_hash.get(child_id)
            if multifur_idx is None or child_idx not in neuron.children[multifur_idx]:
                continue
            parent_idx = neuron.nid_hash.get(int(neuron.swc[multifur_idx][6]))
            if parent_idx is None:
                continue
            if len(neuron.children[parent_idx]) > 1:
                new_parent_id = _qc_append_interpolated_parent(
                    neuron, multifur_idx, parent_idx, child_idx
                )
                child_idx = neuron.nid_hash[child_id]
            else:
                new_parent_id = int(neuron.swc[parent_idx][0])
            neuron.swc[child_idx][6] = new_parent_id
            neuron.initialize(neuron.swc)
            moved = True
        if not moved:
            return


def qc_merge_close_point(neuron, threshold=0.01):
    threshold2 = threshold * threshold
    nodes_to_remove = set()
    stack = deque(
        child_idx
        for soma_idx in neuron.soma_indices
        for child_idx in neuron.children[soma_idx]
    )
    while stack:
        cur_idx = stack.pop()
        parent1_idx = neuron.nid_hash.get(int(neuron.swc[cur_idx][6]))
        if parent1_idx is None:
            continue
        parent2_idx = neuron.nid_hash.get(int(neuron.swc[parent1_idx][6]))
        if parent2_idx is None:
            stack.extend(neuron.children[cur_idx])
            continue
        if _qc_squared_distance(
            neuron.swc[parent1_idx][2:5], neuron.swc[parent2_idx][2:5]
        ) < threshold2:
            parent2_id = int(neuron.swc[parent2_idx][0])
            for child_idx in neuron.children[parent1_idx]:
                neuron.swc[child_idx][6] = parent2_id
            nodes_to_remove.add(parent1_idx)
        elif len(neuron.children[parent1_idx]) > 1:
            siblings = list(neuron.children[parent1_idx])
            kept_idx = siblings[-1]
            for sibling_idx in reversed(siblings[:-1]):
                if _qc_squared_distance(
                    neuron.swc[kept_idx][2:5], neuron.swc[sibling_idx][2:5]
                ) < threshold2:
                    kept_id = int(neuron.swc[kept_idx][0])
                    for child_idx in neuron.children[sibling_idx]:
                        neuron.swc[child_idx][6] = kept_id
                    nodes_to_remove.add(sibling_idx)
                else:
                    kept_idx = sibling_idx
        stack.extend(neuron.children[cur_idx])
    neuron.remove_indices(nodes_to_remove)


def apply_qc_geometry_steps(swc_list, removal_callback=None):
    neuron = SWCQCNeuron(swc_list)

    def run_removal_step(function, reason, code, **kwargs):
        before_ids = {int(node[0]) for node in neuron.swc}
        function(neuron, **kwargs)
        removed_ids = before_ids - {int(node[0]) for node in neuron.swc}
        if removal_callback and removed_ids:
            removal_callback(removed_ids, reason, code)

    run_removal_step(
        qc_merge_close_point,
        "删除距离过近的冗余节点",
        "qc_merge_close_point",
        threshold=0.01,
    )
    run_removal_step(
        qc_remove_zigzag,
        "连续两次急转形成 Zigzag",
        "qc_remove_zigzag",
    )
    qc_tune_branch(neuron)
    qc_correct_multifurcation(neuron, window_length=5.0)
    run_removal_step(
        qc_remove_bad_binary_bifurcation,
        "异常二叉分叉：子枝夹角大于140度且偏离父枝延伸线大于70度",
        "qc_remove_bad_binary_bifurcation",
        window_length=5.0,
        child_child_angle_threshold=140.0,
        parent_child_angle_threshold=70.0,
    )
    run_removal_step(
        qc_remove_branch_towards_soma,
        "分叉后的分支重新朝向胞体",
        "qc_remove_branch_towards_soma",
        min_path_length=10.0,
        window_length=5.0,
        towards_soma_degree=60.0,
    )
    return neuron.as_swc_list()


# ==============================================================================
# [完美版方案一]：专属 Mask 提取 + 动态外扩 (Swallow the Bird's Nest)
# ==============================================================================
def log_soma_mask_fallback(out_swc_dir, record):
    """Append one JSON record per sphere fallback, including failed later runs.

    Keep this separate from error_neurons.log, which is reset on each run.
    flock prevents records from different pruning workers from interleaving.
    """
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(out_swc_dir)), "soma_mask_fallback.log"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "event": "soma_mask_fallback",
        "cut_mode": "sphere",
        **record,
    }
    with open(log_path, "a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def get_soma_mask_inside_ids(
    sxyz, topo_ids, topo_coords, json_path, mask_path, halo_radius=10.0,
    min_neighbor_volume=20, diagnostics=None
):
    """
    通过真实的 3D Soma Mask 提取处于目标胞体内部及表面光晕区域的 SWC 节点 ID。
    halo_radius (um): 向外膨胀的“清理半径”，用于吞噬表面的鸟巢毛刺。默认 10 um。
    """
    if diagnostics is None:
        diagnostics = {}
    try:
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
        bounds = meta_data.get("bounds_1um_zxy", [0,0,0,0,0,0])
        z_min, x_min, y_min = bounds[0], bounds[2], bounds[4]
    except Exception as e:
        print(f"  [Error] Failed to read JSON: {e}")
        diagnostics.update(reason="json_read_error", detail=str(e))
        return None, 20.0, None

    try:
        mask_img = ImageParser(mask_path).load()
        mask_binary = (mask_img > 0).astype(np.uint8)
    except Exception as e:
        print(f"  [Error] Failed to read Mask: {e}")
        diagnostics.update(reason="mask_read_error", detail=str(e))
        return None, 20.0, None

    # 1. 计算连通域，区分不同的细胞
    ccs = cc3d.connected_components(mask_binary)
    z_dim, y_dim, x_dim = ccs.shape

    # 2. 映射中心点，只锁定我们的目标胞体 (完美避开邻居)
    local_x = int(round(sxyz[0] - x_min))
    local_y = int(round(sxyz[1] - y_min))
    local_z = int(round(sxyz[2] - z_min))

    target_cc_id = 0
    if 0 <= local_z < z_dim and 0 <= local_y < y_dim and 0 <= local_x < x_dim:
        target_cc_id = ccs[local_z, local_y, local_x]

        # 容错：如果中心刚好是空洞，搜索 5 像素内最大的块
        if target_cc_id == 0:
            search_r = 5
            sub_vol = ccs[max(0, local_z-search_r):min(z_dim, local_z+search_r+1),
                          max(0, local_y-search_r):min(y_dim, local_y+search_r+1),
                          max(0, local_x-search_r):min(x_dim, local_x+search_r+1)]
            valid_ids = sub_vol[sub_vol > 0]
            if len(valid_ids) > 0:
                target_cc_id = np.bincount(valid_ids.flatten()).argmax()

    if target_cc_id == 0:
        diagnostics.update(
            reason="target_component_not_found",
            local_target_xyz=[local_x, local_y, local_z],
            mask_shape_zyx=[int(z_dim), int(y_dim), int(x_dim)],
        )
        return None, 20.0, None

    # 3. 核心改进：创建目标胞体的专属 Mask，并计算到表面的距离
    target_mask = (ccs == target_cc_id)
    component_ids, component_volumes = np.unique(ccs[ccs > 0], return_counts=True)
    valid_neighbor_ids = {
        int(component_id)
        for component_id, component_volume in zip(component_ids, component_volumes)
        if int(component_id) != int(target_cc_id)
        and int(component_volume) >= int(min_neighbor_volume)
    }
    neighbor_mask = np.isin(ccs, list(valid_neighbor_ids))
    # edt.edt 计算背景像素到前景（胞体）的最短距离
    dist_from_soma = edt.edt(~target_mask, anisotropy=(1,1,1), black_border=False)

    # 估算等效半径 (供step02画图参考)
    volume = np.sum(target_mask)
    approx_radius = (volume * 3.0 / (4.0 * np.pi)) ** (1.0/3.0) + halo_radius

    # 4. 判断拓扑节点：不仅在胞体内部的要切，在表面 halo_radius 范围内的毛刺也要切！
    inside_ids = set()
    for nid, coord in zip(topo_ids, topo_coords):
        lx = int(round(coord[0] - x_min))
        ly = int(round(coord[1] - y_min))
        lz = int(round(coord[2] - z_min))

        if 0 <= lz < z_dim and 0 <= ly < y_dim and 0 <= lx < x_dim:
            # 如果在专属胞体内，或者在距离表面 halo_radius 微米的范围内
            if target_mask[lz, ly, lx] or dist_from_soma[lz, ly, lx] <= halo_radius:
                inside_ids.add(nid)

    neighbor_context = {
        "mask": neighbor_mask,
        "origin_xyz": np.array([x_min, y_min, z_min], dtype=float),
        "component_count": len(valid_neighbor_ids),
    }
    return inside_ids, max(approx_radius, 5.0), neighbor_context


def remove_branches_entering_neighbor_soma(
    swc_list, neighbor_context, sample_step=0.5
):
    """Remove downstream branches whose node or edge enters another soma mask."""
    if not neighbor_context or neighbor_context.get("component_count", 0) == 0:
        return swc_list, set()

    neighbor_mask = neighbor_context["mask"]
    origin_xyz = np.asarray(neighbor_context["origin_xyz"], dtype=float)
    z_dim, y_dim, x_dim = neighbor_mask.shape

    graph = nx.DiGraph()
    coords = {}
    nodes_by_id = {}
    for node in swc_list:
        node_id, parent_id = int(node[0]), int(node[6])
        coords[node_id] = np.asarray(node[2:5], dtype=float)
        nodes_by_id[node_id] = node
        graph.add_node(node_id)
        if parent_id != -1:
            graph.add_edge(parent_id, node_id)

    def point_hits_neighbor(point_xyz):
        local = np.rint(point_xyz - origin_xyz).astype(int)
        x, y, z = int(local[0]), int(local[1]), int(local[2])
        return (
            0 <= z < z_dim
            and 0 <= y < y_dim
            and 0 <= x < x_dim
            and bool(neighbor_mask[z, y, x])
        )

    def edge_hits_neighbor(parent_xyz, child_xyz):
        distance = np.linalg.norm(child_xyz - parent_xyz)
        sample_count = max(2, int(np.ceil(distance / sample_step)) + 1)
        for fraction in np.linspace(0.0, 1.0, sample_count):
            if point_hits_neighbor(parent_xyz + fraction * (child_xyz - parent_xyz)):
                return True
        return False

    hit_roots = set()
    for parent_id, child_id in graph.edges():
        if parent_id not in coords or child_id not in coords:
            continue
        if edge_hits_neighbor(coords[parent_id], coords[child_id]):
            hit_roots.add(child_id)

    nodes_to_remove = set()
    for root_id in hit_roots:
        if root_id in nodes_to_remove:
            continue
        nodes_to_remove.add(root_id)
        nodes_to_remove.update(nx.descendants(graph, root_id))

    filtered = [
        node for node in swc_list if int(node[0]) not in nodes_to_remove
    ]
    return filtered, nodes_to_remove

# ==========================================================
# (中间的通用函数)
# ==========================================================
def get_cube_vertices_max(points, image, fill_value=0.0):
    points = np.asarray(points, dtype=np.float64)
    image_shape = np.array(image.shape)
    floor_coords = np.floor(points).astype(np.int32)
    offsets = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    all_vertices = floor_coords[:, np.newaxis, :] + offsets
    all_vertices_flat = all_vertices.reshape(-1, 3)
    valid_mask = (
        (all_vertices_flat[:, 0] >= 0) & (all_vertices_flat[:, 0] < image_shape[2]) &
        (all_vertices_flat[:, 1] >= 0) & (all_vertices_flat[:, 1] < image_shape[1]) &
        (all_vertices_flat[:, 2] >= 0) & (all_vertices_flat[:, 2] < image_shape[0])
    )
    vertex_values = np.full(len(all_vertices_flat), fill_value, dtype=image.dtype)
    valid_indices = all_vertices_flat[valid_mask]
    vertex_values[valid_mask] = image[valid_indices[:, 2], valid_indices[:, 1], valid_indices[:, 0]]
    vertex_values_reshaped = vertex_values.reshape(len(points), 8)
    return np.max(vertex_values_reshaped, axis=1)

def extract_binary_region(img_binary, soma_coords, root_ids, coords_dict, expansion_ratio=0.2):
    all_points = [soma_coords]
    for root_id in root_ids: all_points.append(coords_dict[root_id])
    points_array = np.array(all_points)
    min_coords, max_coords = np.min(points_array, axis=0), np.max(points_array, axis=0)
    expansion = np.maximum(((max_coords - min_coords) * expansion_ratio).astype(int), np.array([10, 10, 5]))
    min_expanded = np.maximum(min_coords - expansion, 0)
    max_expanded = np.minimum(max_coords + expansion, np.array(img_binary.shape[::-1]) - 1)
    x_min, y_min, z_min = min_expanded.astype(int)
    x_max, y_max, z_max = max_expanded.astype(int)
    region_binary = img_binary[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    return region_binary, np.array([x_min, y_min, z_min]), (x_min, x_max, y_min, y_max, z_min, z_max)

def check_root_ids_connectivity_in_region(region_binary, offset, soma_coords, root_ids, coords_dict):
    ccs_region = cc3d.connected_components(region_binary)
    def to_region_coords(gc): return (gc[2]-offset[2], gc[1]-offset[1], gc[0]-offset[0])
    zi, yi, xi = to_region_coords(soma_coords)
    cc_soma_region = ccs_region[zi, yi, xi] if (0<=zi<ccs_region.shape[0] and 0<=yi<ccs_region.shape[1] and 0<=xi<ccs_region.shape[2] and region_binary[zi,yi,xi]>0) else 0
    good_points, bad_points, connectivity_info = [], [], {}
    for root_id in root_ids:
        zr, yr, xr = to_region_coords(np.round(coords_dict[root_id]).astype(int))
        if (0<=zr<ccs_region.shape[0] and 0<=yr<ccs_region.shape[1] and 0<=xr<ccs_region.shape[2] and region_binary[zr,yr,xr]>0):
            cc_root = ccs_region[zr, yr, xr]
            if cc_soma_region > 0 and cc_root == cc_soma_region:
                good_points.append(root_id); connectivity_info[root_id] = ("direct_connection", None)
            else:
                has_connection, ngb_xyz = False, None
                for dz in [-1,0,1]:
                    for dy in [-1,0,1]:
                        for dx in [-1,0,1]:
                            if dz==0 and dy==0 and dx==0: continue
                            nz, ny, nx = zr+dz, yr+dy, xr+dx
                            if (0<=nz<ccs_region.shape[0] and 0<=ny<ccs_region.shape[1] and 0<=nx<ccs_region.shape[2] and region_binary[nz,ny,nx]>0):
                                if cc_soma_region > 0 and ccs_region[nz,ny,nx] == cc_soma_region:
                                    has_connection, ngb_xyz = True, np.array([nx,ny,nz]); break
                        if has_connection: break
                    if has_connection: break
                if has_connection:
                    good_points.append(root_id); connectivity_info[root_id] = ("neighbor_connection", ngb_xyz)
                else:
                    bad_points.append(root_id); connectivity_info[root_id] = ("no_connection", None)
        else:
            bad_points.append(root_id); connectivity_info[root_id] = ("out_of_region", None)
    return good_points, bad_points, connectivity_info, ccs_region

def remove_bad_root_ids_and_descendants(new_graph, bad_root_ids):
    cleaned_graph = new_graph.copy()
    removed_nodes, removed_edges = [], []
    for bad_root_id in bad_root_ids:
        if not cleaned_graph.has_node(bad_root_id): continue
        if nx.is_directed_acyclic_graph(cleaned_graph):
            downstream_nodes = nx.descendants(cleaned_graph, bad_root_id)
            downstream_nodes.add(bad_root_id)
        else:
            try: downstream_nodes = set(nx.bfs_tree(cleaned_graph, bad_root_id).nodes())
            except nx.NetworkXError: continue
        nodes_to_remove = list(downstream_nodes)
        edges_to_remove = []
        for node in nodes_to_remove:
            if cleaned_graph.has_node(node):
                for pred in cleaned_graph.predecessors(node): edges_to_remove.append((pred, node))
                for succ in cleaned_graph.successors(node): edges_to_remove.append((node, succ))
        for edge in edges_to_remove:
            if cleaned_graph.has_edge(*edge):
                cleaned_graph.remove_edge(*edge)
                removed_edges.append(edge)
        for node in nodes_to_remove:
            if cleaned_graph.has_node(node):
                cleaned_graph.remove_node(node)
                removed_nodes.append(node)
    for node in list(nx.isolates(cleaned_graph)):
        cleaned_graph.remove_node(node)
        removed_nodes.append(node)
    return cleaned_graph, removed_nodes, removed_edges

def connect_roots_to_soma_region(img_binary, soma_coords, root_ids, coords_dict, scale_xyz, start_id, expansion_ratio=0.2, merge_radius_ratio=1.0):
    region_binary, offset, bbox = extract_binary_region(img_binary, soma_coords, root_ids, coords_dict, expansion_ratio)
    distance_transform = edt.edt(region_binary, anisotropy=(1,1,1), black_border=True).astype(np.float64)
    cost_array_region = 1.0 / (distance_transform + 1e-8)
    def to_region_coords(gc): return (gc[0]-offset[0], gc[1]-offset[1], gc[2]-offset[2])
    def to_global_coords(rc): return (rc[0]+offset[0], rc[1]+offset[1], rc[2]+offset[2])
    soma_region_coords = to_region_coords(soma_coords)
    all_paths, path_costs = calculate_all_paths_shortest_path_tree(root_ids, coords_dict, cost_array_region, soma_region_coords, to_region_coords, to_global_coords)
    DG, coord_to_id, root_to_id = paths_to_directed_tree(all_paths, start_id=start_id)
    scale = max(scale_xyz) / min(scale_xyz)
    for node_name, attrs in DG.nodes(data=True):
        pt_xyz = attrs['coords']
        radius = distance_transform[pt_xyz[2], pt_xyz[1], pt_xyz[0]] * scale
        nx.set_node_attributes(DG, {node_name: {'radius': radius}})
    for node, attrs in DG.nodes(data=True):
        if 'coords' in attrs: attrs['coords'] = to_global_coords(attrs['coords'])
    outDG = merge_with_soma(DG, scale_xyz, merge_ratio=merge_radius_ratio)
    return outDG, root_to_id

def calculate_all_paths_shortest_path_tree(root_ids, coords_dict, cost_array_region, soma_region_coords, to_region_coords, to_global_coords):
    all_paths, path_costs = {}, {}
    soma_z, soma_y, soma_x = soma_region_coords[2], soma_region_coords[1], soma_region_coords[0]
    mcp = MCP_Geometric(cost_array_region, fully_connected=True)
    cumulative_costs, traceback = mcp.find_costs(starts=[(soma_z, soma_y, soma_x)])
    start_points_int, start_points_info = [], []
    for root_id in root_ids:
        rrc = np.round(to_region_coords(coords_dict[root_id])).astype(int)
        start_points_int.append((rrc[2], rrc[1], rrc[0]))
        start_points_info.append({'root_id': root_id})
    batch_paths = []
    for spi in start_points_int:
        cp = mcp.traceback(spi)
        if len(cp) == 1: cp = cp + cp
        batch_paths.append(cp)
    for i, (spi, pr) in enumerate(zip(start_points_info, batch_paths)):
        root_id = spi['root_id']
        if pr:
            path_curr = [(p[2], p[1], p[0]) for p in pr[:-1]]
            all_paths[root_id] = path_curr
            path_costs[root_id] = cumulative_costs[start_points_int[i]]
    return all_paths, path_costs

def paths_to_directed_tree(all_paths, start_id=0):
    DG = nx.DiGraph()
    all_points = set()
    for path in all_paths.values():
        for point in path: all_points.add(tuple(point))
    coord_to_id = {tuple(p): i + start_id for i, p in enumerate(all_points)}
    for nid in coord_to_id.values(): DG.add_node(nid, coords=list(coord_to_id.keys())[list(coord_to_id.values()).index(nid)])
    root_to_id = {rid: coord_to_id[tuple(path[-1])] for rid, path in all_paths.items()}
    for rid, path in all_paths.items():
        for i in range(len(path)-1):
            DG.add_edge(coord_to_id[tuple(path[i])], coord_to_id[tuple(path[i+1])], weight=1)
    return DG, coord_to_id, root_to_id

def merge_with_soma(inDG, scale_xyz, merge_ratio=1.0):
    def _is_outside(node_coord, soma_coord, scale, threshold):
        return np.linalg.norm((soma_coord - node_coord) * scale) > threshold
    outDG = nx.DiGraph()
    soma_node = [node for node in inDG.nodes() if inDG.in_degree(node) == 0][0]
    leaf_nodes = [node for node in inDG.nodes() if inDG.out_degree(node) == 0 and node != soma_node]
    scale = np.array([1,1,scale_xyz[2]/scale_xyz[0]]) if scale_xyz[2] > scale_xyz[0] else np.array([scale_xyz[0]/scale_xyz[2], scale_xyz[0]/scale_xyz[2], 1])
    coords_all = {k: np.array(v) for k, v in nx.get_node_attributes(inDG, 'coords').items()}
    soma_coord = coords_all[soma_node]
    merge_threshold = inDG.nodes[soma_node].get('radius', 1.0) * merge_ratio
    outDG.add_node(soma_node, **inDG.nodes[soma_node])
    processed_nodes, node_mapping = set([soma_node]), {}

    def add_node_to_output(node, parent=None):
        if node in processed_nodes: return node_mapping.get(node, node)
        outDG.add_node(node, **inDG.nodes[node].copy())
        processed_nodes.add(node); node_mapping[node] = node
        if parent and not outDG.has_edge(parent, node):
            outDG.add_edge(parent, node, **(inDG.get_edge_data(parent, node) or {}))
        return node

    for leaf_node in leaf_nodes:
        if leaf_node in processed_nodes: continue
        try: path = nx.shortest_path(inDG, soma_node, leaf_node)
        except nx.NetworkXNoPath: continue
        current_parent, nodes_to_add = soma_node, []
        for i in range(len(path) - 1, 0, -1):
            current_node = path[i]
            if _is_outside(coords_all[current_node], soma_coord, scale, merge_threshold): nodes_to_add.append(current_node)
            else:
                if i == len(path) - 1: nodes_to_add.append(current_node)
                break
        for node in reversed(nodes_to_add):
            if node not in processed_nodes: current_parent = add_node_to_output(node, current_parent)
            else:
                mapped_node = node_mapping.get(node, node)
                if not outDG.has_edge(current_parent, mapped_node):
                    outDG.add_edge(current_parent, mapped_node, **(inDG.get_edge_data(current_parent, mapped_node) or {}))
                current_parent = mapped_node
    return outDG

# ==========================================================
# 主流程函数
# ==========================================================
def process_neuron_morphology(input_imgfile, swc_neu, raw_imgfile, out_swc_dir, mip_dir,
                            meta, sphere_zradius=20, downsample_scale=np.array([2,2,2]),
                            binary_input=True, pre_traced=True, debug=True, show_steps=True,
                            soma_img_dir=None, soma_mask_dir=None):
    swc_name = os.path.split(input_imgfile)[-1]
    cell_id = int(os.path.splitext(swc_name)[0].split('_')[1])
    prefix = os.path.splitext(swc_name)[0]

    viz_buffer = {}
    pruning_reasons = {}
    original_node_map = {}
    original_children = {}

    def _record_removed_ids(node_ids, reason, code, overwrite=False):
        for node_id in node_ids:
            node_id = int(node_id)
            if (
                node_id in original_node_map
                and (overwrite or node_id not in pruning_reasons)
            ):
                pruning_reasons[node_id] = {"reason": reason, "code": code}

    def _record_original_subtree(root_id, reason, code):
        root_id = int(root_id)
        stack = [root_id]
        subtree_ids = set()
        while stack:
            node_id = stack.pop()
            if node_id in subtree_ids:
                continue
            subtree_ids.add(node_id)
            stack.extend(original_children.get(node_id, []))
        _record_removed_ids(subtree_ids, reason, code)

    def _complete_pruning_reason_coverage(output_tree):
        output_ids = {int(node[0]) for node in output_tree}
        removed_ids = set(original_node_map) - output_ids
        input_parent = {
            int(node[0]): int(node[6]) for node in original_node_map.values()
        }
        removed_roots = [
            node_id for node_id in removed_ids
            if input_parent.get(node_id, -1) not in removed_ids
        ]

        for root_id in removed_roots:
            if root_id in pruning_reasons:
                continue
            stack = list(original_children.get(root_id, []))
            descendant_reasons = []
            while stack:
                node_id = stack.pop()
                if node_id not in removed_ids:
                    continue
                if node_id in pruning_reasons:
                    descendant_reasons.append(pruning_reasons[node_id])
                stack.extend(original_children.get(node_id, []))

            if descendant_reasons:
                counts = {}
                for info in descendant_reasons:
                    key = (info["reason"], info["code"])
                    counts[key] = counts.get(key, 0) + 1
                reason, code = max(counts, key=counts.get)
            else:
                reason = "胞体近端分支在胞体裁剪与重连阶段未保留"
                code = "process_neuron_morphology: soma cut and reconnection"
            _record_original_subtree(root_id, reason, code)

    def _graph_to_swc_list(graph, color_type=3):
        temp_tree = []
        for node in graph.nodes():
            if 'coords' in graph.nodes[node]:
                c = graph.nodes[node]['coords']
                r = graph.nodes[node].get('radius', 1.0)
            elif node in morph.pos_dict:
                c = morph.pos_dict[node][2:5]
                r = morph.pos_dict[node][5]
            else: continue
            preds = list(graph.predecessors(node))
            p_id = preds[0] if preds else -1
            temp_tree.append((node, color_type, c[0], c[1], c[2], r, p_id))
        return temp_tree

    # 将原来的 _visualize_step 定义修改为接收 annotations：
    def _visualize_step(step_name, tree_data=None, image_data=None, scale=1, annotations=None):
        current_swc = None
        if tree_data is not None:
            if isinstance(tree_data, (nx.Graph, nx.DiGraph)): current_swc = _graph_to_swc_list(tree_data)
            else: current_swc = tree_data

        if show_steps:
            viz_buffer[step_name] = {'tree': current_swc, 'img': image_data, 'scale': scale, 'annotations': annotations}
        elif step_name == 'step08_final_result':
            fname = os.path.join(mip_dir, f'{prefix}_{step_name}.png')
            img_to_plot = image_data if image_data is not None else raw_image
            plot_swc(img_to_plot, current_swc, figname=fname, scale=scale)

    def _save_combined_grid_plot():
        if not show_steps or not viz_buffer: return
        matplotlib.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
            "Arial Unicode MS", "DejaVu Sans"
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
        fig = plt.figure(figsize=(30, 35), dpi=110)
        grid = fig.add_gridspec(
            4, 6, height_ratios=[1.0, 1.0, 1.0, 1.45],
            hspace=0.10, wspace=0.04
        )
        axes = [
            fig.add_subplot(grid[row, col * 2:(col + 1) * 2])
            for row in range(3)
            for col in range(3)
        ]

        step_order = [
            ("step01_original_input", "Step 1: Original Input"),
            ("step02_sphere_cuts", "Step 2: Mask Cuts"),
            ("step03_graph_traversed", "Step 3: Graph Traversed"),
            ("step04_binary_mask", "Step 4: Binary Mask"),
            ("step05_soma_mask", "Step 5: Soma Mask"),
            ("step06_connectivity_check", "Step 6: Roots Check"),
            ("step07_new_connections", "Step 7: New Connections"),
            ("step08_sharp_bends", "Step 8: Z-Bends Pruning"),
            ("step09_final_result", "Step 9: Final Result"),
        ]

        type_colors = {1: 'white', 2: 'magenta', 3: 'cyan', 4: 'purple', 0: 'dodgerblue'}
        for i, (key, title) in enumerate(step_order):
            ax = axes[i]
            if key not in viz_buffer:
                ax.text(0.5, 0.5, 'Skipped', ha='center', va='center', color='white')
                ax.set_title(title)
                ax.axis('off'); ax.set_facecolor('black')
                continue
            data = viz_buffer[key]
            bg_img = data['img'] if data['img'] is not None else raw_image
            bg_img_2d = np.max(bg_img, axis=0) if bg_img.ndim == 3 else bg_img
            # np.percentile cannot interpolate boolean arrays. Step 4's binary
            # mask is bool, so convert all panel backgrounds to a numeric dtype
            # before calculating the display range.
            bg_img_2d = np.asarray(bg_img_2d, dtype=np.float32)
            vmax_val = np.percentile(bg_img_2d, 99.5)
            if vmax_val <= 0:
                vmax_val = np.max(bg_img_2d)
                if vmax_val <= 0: vmax_val = 1
            ax.imshow(bg_img_2d, cmap='gray', vmax=vmax_val)

            swc_list = data['tree']
            if swc_list and key != "step05_soma_mask":
                node_map = {n[0]: (n[2], n[3]) for n in swc_list}
                for node in swc_list:
                    nid, ntype, x, y, z, r, pid = node
                    color = type_colors.get(ntype, 'dodgerblue')
                    if pid != -1 and pid in node_map:
                        px, py = node_map[pid]
                        ax.plot([x, px], [y, py], color=color, linewidth=0.8, alpha=0.7)
                    if pid == -1: ax.scatter(x, y, s=30, c='red', linewidths=1.2, marker='o', zorder=10)

            if key == "step02_sphere_cuts":
                soma_node = morph.pos_dict.get(list(morph.pos_dict.keys())[0])
                for pid, info in morph.pos_dict.items():
                    if info[1] == 1:
                        soma_node = info; break
                raw_soma_x, raw_soma_y = soma_node[2], soma_node[3]
                soma_circle = plt.Circle((raw_soma_x, raw_soma_y), sphere_zradius, color='white', fill=False, linestyle='--', linewidth=1.5, alpha=0.9)
                ax.add_patch(soma_circle)
                ax.scatter(raw_soma_x, raw_soma_y, c='red', s=40, marker='x', zorder=10)

            # 🌟 统一渲染各种标注
            if 'annotations' in data and data['annotations']:
                for ann in data['annotations']:
                    if 'text_x' in ann:
                        ax.plot([ann['rx'], ann['text_x']], [ann['ry'], ann['text_y']], color='black', linewidth=1.5, zorder=5)
                        ax.text(ann['text_x'], ann['text_y'], ann['text'], color=ann['color'], fontsize=9,
                                ha='center', va='center', fontweight='bold', zorder=9,
                                bbox=dict(facecolor='black', alpha=0.8, edgecolor=ann['color'], pad=1.5))
                        if 'vsc_end_x' in ann:
                            ax.annotate('', xy=(ann['vsc_end_x'], ann['vsc_end_y']), xytext=(ann['rx'], ann['ry']),
                                        arrowprops=dict(arrowstyle="->", color='yellow', lw=2, alpha=0.9), zorder=8)
                        if 'vc_end_x' in ann:
                            ax.annotate('', xy=(ann['vc_end_x'], ann['vc_end_y']), xytext=(ann['rx'], ann['ry']),
                                        arrowprops=dict(arrowstyle="->", color='green', lw=2, alpha=0.9), zorder=8)

            ax.set_title(title, color='black', fontsize=12)
            ax.axis('off')

        comparison_input_ax = fig.add_subplot(grid[3, :3])
        comparison_output_ax = fig.add_subplot(grid[3, 3:])
        input_tree = viz_buffer.get("step01_original_input", {}).get("tree") or []
        output_tree = viz_buffer.get("step09_final_result", {}).get("tree") or []
        output_ids = {int(node[0]) for node in output_tree}
        removed_ids = {
            int(node[0]) for node in input_tree if int(node[0]) not in output_ids
        }

        def draw_comparison(ax, tree_data, title, show_removed_overlay):
            bg_img_2d = np.max(raw_image, axis=0) if raw_image.ndim == 3 else raw_image
            vmax_val = np.percentile(bg_img_2d, 99.5)
            if vmax_val <= 0:
                vmax_val = max(float(np.max(bg_img_2d)), 1.0)
            ax.imshow(bg_img_2d, cmap="gray", vmax=vmax_val)
            node_map = {
                int(node[0]): (float(node[2]), float(node[3]))
                for node in tree_data
            }
            for node in tree_data:
                node_id, parent_id = int(node[0]), int(node[6])
                if parent_id in node_map:
                    x, y = node_map[node_id]
                    px, py = node_map[parent_id]
                    ax.plot(
                        [px, x], [py, y], color="#38d9ff", linewidth=0.75,
                        alpha=0.75, zorder=3
                    )

            if show_removed_overlay:
                original_xy = {
                    int(node[0]): (float(node[2]), float(node[3]))
                    for node in input_tree
                }
                for node in input_tree:
                    node_id, parent_id = int(node[0]), int(node[6])
                    if node_id not in removed_ids or parent_id not in original_xy:
                        continue
                    x, y = original_xy[node_id]
                    px, py = original_xy[parent_id]
                    ax.plot(
                        [px, x], [py, y], color="red", linewidth=1.35,
                        linestyle="--", alpha=0.9, zorder=5
                    )
            ax.set_title(title, fontsize=15, fontweight="bold")
            ax.axis("off")

        draw_comparison(
            comparison_input_ax, input_tree,
            "剪枝前：红色虚线为最终被删除的原始分支", True
        )
        draw_comparison(
            comparison_output_ax, output_tree,
            "剪枝后：青色为保留结构，红色虚线为被删除分支", True
        )

        removed_roots = []
        input_parent = {int(node[0]): int(node[6]) for node in input_tree}
        for node_id in sorted(removed_ids):
            if input_parent.get(node_id, -1) not in removed_ids:
                removed_roots.append(node_id)

        input_xy = {
            int(node[0]): (float(node[2]), float(node[3])) for node in input_tree
        }
        reason_categories = {}
        root_entries = []
        for node_id in removed_roots:
            info = pruning_reasons.get(
                node_id,
                {
                    "reason": "前序筛选后未进入最终树",
                    "code": "process_neuron_morphology",
                },
            )
            category_key = (info["reason"], info["code"])
            if category_key not in reason_categories:
                reason_categories[category_key] = len(reason_categories) + 1
            root_entries.append((node_id, info, reason_categories[category_key]))

        category_counts = {
            category_number: sum(
                entry_category == category_number
                for _, _, entry_category in root_entries
            )
            for category_number in reason_categories.values()
        }

        for node_id, info, category_number in root_entries:
            x, y = input_xy.get(node_id, (None, None))
            if x is None:
                continue
            comparison_output_ax.scatter(
                [x], [y], c="yellow", edgecolors="red", s=32, zorder=8
            )
            comparison_output_ax.text(
                x, y, f"R{category_number}",
                fontsize=7.0, color="yellow", fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor="black",
                    edgecolor="red", alpha=0.78
                ),
                zorder=9,
            )

        legend_lines = ["剪枝原因编号："]
        for (reason, code), category_number in reason_categories.items():
            legend_lines.append(
                f"R{category_number}（{category_counts[category_number]}处）：{reason}"
            )
            legend_lines.append(f"代码：{code}")
        legend_text = "\n".join(legend_lines)
        comparison_output_ax.text(
            0.01, 0.99, legend_text,
            transform=comparison_output_ax.transAxes,
            fontsize=7.5,
            color="white",
            ha="left",
            va="top",
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.35", facecolor="black",
                edgecolor="red", alpha=0.78
            ),
            zorder=10,
        )

        fig.suptitle(
            f"{prefix} 剪枝流程与原因对照", fontsize=19, fontweight="bold", y=0.995
        )
        plt.savefig(
            os.path.join(mip_dir, f'{prefix}_combined_process.png'),
            bbox_inches="tight",
        )

        plt.close('all')
    # 1. 加载数据
    raw_image = ImageParser(raw_imgfile).load()
    tree = parse_swc(swc_neu)
    if len(tree) == 1:
        node = tree[0]
        node_id = int(node[0])
        node_type = int(node[1])
        parent_id = int(node[6])
        if node_type == 1 and parent_id == -1:
            raise SomaOnlySWCError(
                "SWC 只有一个 Soma 节点，没有任何神经突节点："
                f"node_id={node_id}, type={node_type}, parent={parent_id}"
            )
        raise SingleNodeSWCError(
            "SWC 只有一个节点，无法构建神经元拓扑，且该节点不是合法的 "
            f"Soma 根节点：node_id={node_id}, type={node_type}, "
            f"parent={parent_id}"
        )
    original_node_map.update({int(node[0]): node for node in tree})
    for node in tree:
        node_id, parent_id = int(node[0]), int(node[6])
        original_children.setdefault(node_id, [])
        if parent_id != -1:
            original_children.setdefault(parent_id, []).append(node_id)

    morph = Morphology(tree)

    curr_meta = meta.loc[cell_id]
    orig_scale_xyz = np.array([float(curr_meta['xy_resolution'])/1000., float(curr_meta['xy_resolution'])/1000., float(curr_meta['z_resolution'])/1000.])
    sxyz = np.array([float(curr_meta.soma_x)*orig_scale_xyz[0], float(curr_meta.soma_y)*orig_scale_xyz[1], float(curr_meta.soma_z)*orig_scale_xyz[2]])


    _visualize_step("step01_original_input", tree_data=tree, scale=2)

    # 2. 构建拓扑与特征
    coords_dict = {node[0]: np.array(node[2:5]) for node in morph.tree}
    topo_tree, seg_dict = morph.convert_to_topology_tree()
    topo = Topology(topo_tree)

    fseg_dict = {}
    for sid, seg_ids in seg_dict.items():
        pid = topo.pos_dict[sid][-1]
        if pid == -1: continue
        seg_ids_all = [sid] + seg_ids + [pid]
        curr_coords = np.array([coords_dict[idx] for idx in seg_ids_all])
        path_dists = np.linalg.norm((curr_coords[:-1] - curr_coords[1:]) * 1.0, axis=1)
        path_dist = path_dists.sum()
        vn = curr_coords[0] - curr_coords[-1]; vn /= (np.linalg.norm(vn) + 1e-10)
        radii = [morph.pos_dict[nid][5] for nid in seg_ids_all]
        radius, max_radius = np.median(radii) if len(radii) >= 3 else np.mean(radii), max(radii)
        intensities = get_cube_vertices_max(curr_coords, raw_image, fill_value=0.0)
        intensity = np.median(intensities) if len(intensities) >= 3 else np.mean(intensities)
        euc_dist = np.linalg.norm((coords_dict[pid] - coords_dict[sid]) * 1.0)

        if len(intensities) >= 5 and path_dist > 5.0:
            end_intensity_1 = np.mean(intensities[:2])
            end_intensity_2 = np.mean(intensities[-2:])
            min_end_base = min(end_intensity_1, end_intensity_2)
            middle_min = np.min(intensities[1:-1])

            # 【核心修改】：不仅要求跌幅巨大（<25%），还要要求绝对亮度极低（<15）
            if middle_min < min_end_base * 0.25 and middle_min < 15:
                _record_original_subtree(
                    sid,
                    "分支中间出现深暗谷值",
                    "process_neuron_morphology: dark-valley filter",
                )
                continue

        fseg_dict[sid] = (intensity, radius, max_radius, vn, path_dist, euc_dist)

    scale_xyz = np.array([1.0, 1.0, 1.0])

    # =====================================================================
    # 3. 提取 Soma 内部节点
    # =====================================================================
    topo_coords = np.array([node[2:5] for node in topo_tree])
    topo_ids = np.array([int(node[0]) for node in topo_tree])
    app2_root_id = [
        nid for nid in topo_ids if topo.pos_dict[nid][-1] == -1
    ]
    mask_target_xyz = (
        np.asarray(topo.pos_dict[app2_root_id[0]][2:5], dtype=float)
        if app2_root_id
        else sxyz
    )

    in_soma_ids = None
    neighbor_soma_context = None
    base_name = prefix.replace('_0000', '')
    json_file = os.path.join(soma_img_dir, f"{base_name}.json") if soma_img_dir else None
    soma_mask_file = os.path.join(soma_mask_dir, f"{base_name}.tif") if soma_mask_dir else None
    mask_diagnostics = {"reason": "soma_directories_not_configured"}

    if soma_img_dir and soma_mask_dir:
        json_file = os.path.join(soma_img_dir, f"{base_name}.json")
        soma_mask_file = os.path.join(soma_mask_dir, f"{base_name}.tif")
        mask_diagnostics = {
            "reason": "missing_soma_files",
            "missing_files": [
                path for path in (json_file, soma_mask_file) if not os.path.exists(path)
            ],
        }
        if os.path.exists(json_file) and os.path.exists(soma_mask_file):
            mask_diagnostics = {}
            result_ids, approx_radius, neighbor_soma_context = (
                get_soma_mask_inside_ids(
                    mask_target_xyz,
                    topo_ids,
                    topo_coords,
                    json_file,
                    soma_mask_file,
                    halo_radius=0,
                    min_neighbor_volume=20,
                    diagnostics=mask_diagnostics,
                )
            )
            if result_ids is not None:
                in_soma_ids = result_ids
                sphere_zradius = approx_radius

    if in_soma_ids is not None and len(in_soma_ids) == 0:
        print("  [Warning] Mask found but no SWC nodes fall inside it. Falling back to sphere cut.")
        in_soma_ids = None
        mask_diagnostics["reason"] = "mask_no_topology_nodes"

    if in_soma_ids is None:
        log_soma_mask_fallback(out_swc_dir, {
            **mask_diagnostics,
            "cell_id": int(cell_id),
            "input_swc": os.fspath(swc_neu),
            "soma_json": json_file,
            "soma_mask": soma_mask_file,
            "sphere_radius_um": float(sphere_zradius),
            "mask_target_xyz": [float(v) for v in mask_target_xyz],
            "sphere_center_xyz": [float(v) for v in sxyz],
        })
        print(f"  [Warning] Using fixed sphere radius {sphere_zradius} um for cutting.")
        dists2soma = np.linalg.norm((sxyz - topo_coords) * scale_xyz, axis=1)
        in_soma_ids = set(topo_ids[dists2soma < sphere_zradius])

    _, neighbor_soma_blocked_ids = remove_branches_entering_neighbor_soma(
        tree, neighbor_soma_context, sample_step=0.5
    )
    _record_removed_ids(
        neighbor_soma_blocked_ids,
        "分支进入周围其他 neuron 的胞体区域",
        "remove_branches_entering_neighbor_soma",
        overwrite=True,
    )
    for sid, seg_ids in seg_dict.items():
        pid = topo.pos_dict[sid][-1]
        segment_node_ids = {sid, *seg_ids}
        if pid != -1:
            segment_node_ids.add(pid)
        if segment_node_ids & neighbor_soma_blocked_ids:
            fseg_dict.pop(sid, None)

    _record_removed_ids(
        in_soma_ids,
        "节点位于胞体掩膜内部",
        "get_soma_mask_inside_ids / soma sphere cut",
    )

    # 查找穿出胞体边界的边 (Edges crossing the mask boundary)
    intersection_pairs = []
    for id0 in in_soma_ids:
        children = topo.child_dict.get(id0)
        if children:
            for id1 in children:
                if id1 not in in_soma_ids: intersection_pairs.append([id0, id1, 1])
        parent = topo.pos_dict[id0][-1]
        if parent != -1 and parent not in in_soma_ids:
            intersection_pairs.append([id0, parent, 0])

    # ==========================================
    # 🌟 新增的可视化与角度计算逻辑
    # ==========================================
    # ==========================================
    # 🌟 新增的可视化与角度计算逻辑 (带防重叠偏移)
    # ==========================================
    # ==========================================
    # 🌟 新增的可视化与角度计算逻辑 (防重叠延长线版)
    # ==========================================
    # ==========================================
    # 🌟 新增的可视化与角度计算逻辑 (只显示被Cut的，带双箭头可视化)
    # ==========================================
    print(app2_root_id)
    if app2_root_id:
        # 如果找到了，就用它的坐标当做真正的中心 (True Center)
        true_center = np.array(topo.pos_dict[app2_root_id[0]][2:5])
    else:
        # 极端兜底情况，找不到就退回用 sxyz
        true_center = sxyz
    is_pairs = []
    rejected_pairs = []       # 用来存放被剪掉的死树枝
    angle_annotations = []    # 用来存放要在图上打印的文字信息

    for pair in intersection_pairs:
        c0, c1 = np.array(topo.pos_dict[pair[0]][2:5]), np.array(topo.pos_dict[pair[1]][2:5])

        vsc, vc = (c0 - true_center), (c1 - c0)
        norm_vsc = np.linalg.norm(vsc)
        norm_vc = np.linalg.norm(vc) + 1e-10 # 给实际生长方向加个防0底线

        # 树枝真实的起点坐标
        root_x, root_y = c0[0], c0[1]

        # 先计算实际生长方向 (绿箭头方向)，这总是安全的
        dir_vc_x, dir_vc_y = vc[0] / norm_vc, vc[1] / norm_vc

        # 胞体出口不再按单边长度剪枝，仅保留方向判断。
        # 🌟 这里的防线顺便保护了除零错误！
        if norm_vsc < 1e-3:
            # 如果起点完全等于中心点(true_center)，它长向任何方向都是合理的，直接放行！
            is_pairs.append(pair)
            continue

        # 走到这里，说明 c0 不是中心点，可以安全地除以 norm_vsc 了
        dir_vsc_x, dir_vsc_y = vsc[0] / norm_vsc, vsc[1] / norm_vsc
        arrow_len = 25

        dot_val = np.dot(vsc/norm_vsc, vc/norm_vc)
        angle_deg = np.degrees(np.arccos(np.clip(dot_val, -1.0, 1.0)))

        if dot_val > 0:
            is_pairs.append(pair)
        else:
            rejected_pairs.append(pair)
            _record_original_subtree(
                pair[1],
                "穿出胞体边界时方向朝向胞体内部",
                "process_neuron_morphology: soma-boundary direction check",
            )
            text_x = root_x + dir_vc_x * 50
            text_y = root_y + dir_vc_y * 50
            text_str = f"CUT: {angle_deg:.0f}°\n(dot: {dot_val:.2f})"

            angle_annotations.append({
                'rx': root_x, 'ry': root_y,
                'text_x': text_x, 'text_y': text_y,
                'text': text_str, 'color': 'red',
                'vsc_end_x': root_x + dir_vsc_x * arrow_len,
                'vsc_end_y': root_y + dir_vsc_y * arrow_len,
                'vc_end_x': root_x + dir_vc_x * arrow_len,
                'vc_end_y': root_y + dir_vc_y * arrow_len
            })
    # --- 生成用于画图的 debug 树 ---
    debug_sphere_tree = []
    for pair in is_pairs:
        p0, p1 = morph.pos_dict[pair[0]], morph.pos_dict[pair[1]]
        debug_sphere_tree.append((pair[0], 2, *p0[2:5], p0[5], -1))
        debug_sphere_tree.append((pair[1], 2, *p1[2:5], p1[5], pair[0]))

    for pair in rejected_pairs:
        p0, p1 = morph.pos_dict[pair[0]], morph.pos_dict[pair[1]]
        debug_sphere_tree.append((pair[0], 4, *p0[2:5], p0[5], -1))
        debug_sphere_tree.append((pair[1], 4, *p1[2:5], p1[5], pair[0]))

    _visualize_step("step02_sphere_cuts", tree_data=debug_sphere_tree, scale=2, annotations=angle_annotations)
    # ==========================================
    # ==========================================
    # ==========================================


    # 4. 图构建
    graph = nx.DiGraph()
    processor = NeuronGraphProcessor(
        graph,
        coords_dict,
        sxyz,
        scale_xyz,
        conf_length=2.5,
        rejection_callback=_record_original_subtree,
    )
    visited = set()
    for pair in is_pairs:
        id0, id1, pc_type = pair
        if id1 in neighbor_soma_blocked_ids:
            continue
        visited.add(id0)
        if pc_type == 1: processor.traverse_sequential_branch(id1, fseg_dict, visited, topo)
        else: processor.traverse_inverse_branch(id0, id1, fseg_dict, visited, topo)

    _visualize_step("step03_graph_traversed", tree_data=graph, scale=2)

    # 5. 二值化
    img_binary = np.ascontiguousarray(
        ImageParser(input_imgfile).load() > 0,
        dtype=bool,
    )

    _visualize_step("step04_binary_mask", image_data=img_binary)

    # ==========================================================
    # 【极简版 Step 5】直接读取 soma_seg 原文件并拼接到全局画布
    # ==========================================================
    global_soma_mask = np.zeros_like(raw_image, dtype=np.uint8)
    base_name = prefix.replace('_0000', '')

    if soma_img_dir and soma_mask_dir:
        json_file = os.path.join(soma_img_dir, f"{base_name}.json")
        soma_mask_file = os.path.join(soma_mask_dir, f"{base_name}.tif")

        if os.path.exists(json_file) and os.path.exists(soma_mask_file):
            try:
                # 1. 读 JSON 拿坐标偏移
                with open(json_file, 'r') as f:
                    meta_data = json.load(f)
                bounds = meta_data.get("bounds_1um_zxy", [0,0,0,0,0,0])
                z_min, x_min, y_min = int(bounds[0]), int(bounds[2]), int(bounds[4])

                # 2. 读 TIF，只要有值的地方全变成 255 (纯白)
                mask_img = ImageParser(soma_mask_file).load()
                mask_binary = (mask_img > 0).astype(np.uint8) * 255

                # 3. 带越界保护的直接贴图
                z_dim, y_dim, x_dim = mask_binary.shape
                z_start, z_end = max(0, z_min), min(raw_image.shape[0], z_min + z_dim)
                y_start, y_end = max(0, y_min), min(raw_image.shape[1], y_min + y_dim)
                x_start, x_end = max(0, x_min), min(raw_image.shape[2], x_min + x_dim)

                mz_start, mz_end = z_start - z_min, z_end - z_min
                my_start, my_end = y_start - y_min, y_end - y_min
                mx_start, mx_end = x_start - x_min, x_end - x_min

                global_soma_mask[z_start:z_end, y_start:y_end, x_start:x_end] = mask_binary[mz_start:mz_end, my_start:my_end, mx_start:mx_end]
                if(soma_mask_file=="/data/disk/C6.0/app_test/soma_seg/image_60280.tif"):
                    print("Debug: Loaded mask shape:", mask_img.shape, "with bounds:", bounds)
                    print(f"DEBUG: raw_shape={raw_image.shape}, z_min={z_min}, x_min={x_min}, y_min={y_min}")
                    print(mz_start, mz_end, my_start, my_end, mx_start, mx_end)
            except Exception as e:
                print(f"  [Error] Failed to process soma mask for {base_name}: {e}")
                pass # 哪怕出错也无所谓，就显示纯黑画布

    # 直接将全局的画布传给 Step 5
    _visualize_step("step05_soma_mask", image_data=global_soma_mask)

    root_ids = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    soma_coords = (int(true_center[0]), int(true_center[1]), int(true_center[2]))
    region_binary, offset, bbox = extract_binary_region(img_binary, soma_coords, root_ids, coords_dict, expansion_ratio=0.2)

    good_points, bad_points, connectivity_info, ccs_region = check_root_ids_connectivity_in_region(
        region_binary, offset, soma_coords, root_ids, coords_dict
    )

    # ==========================================================
    # 🛡️ 核心手术：强制营救距离胞体极近的“假性断连”根节点
    # ==========================================================
    rescued_points = []
    for rid in bad_points:
        dist_to_soma = np.linalg.norm((coords_dict[rid] - sxyz) * scale_xyz)
        if dist_to_soma < sphere_zradius + 15.0:
            rescued_points.append(rid)

    good_points.extend(rescued_points)
    bad_points = [r for r in bad_points if r not in rescued_points]

    debug_conn_tree = []
    for rid in root_ids:
        c = coords_dict[rid]
        ctype = 3 if rid in good_points else 2
        debug_conn_tree.append((rid, ctype, *c, 5.0, -1))
    _visualize_step(
        "step06_connectivity_check", tree_data=debug_conn_tree, scale=2
    )

    for bad_root_id in bad_points:
        _record_original_subtree(
            bad_root_id,
            "分支根节点与胞体所在二值连通域不连通",
            "check_root_ids_connectivity_in_region",
        )
    graph, _, _ = remove_bad_root_ids_and_descendants(graph, bad_points)
    root_ids = [x for x in root_ids if x in good_points]

    if len(graph) <= 1:
        print('No salient tree found!')
        _save_combined_grid_plot()
        return None, None

    # 6. 路径重连
    # 6. 路径重连
    start_id = max(max(morph.pos_dict.keys()), max([node[6] for node in morph.tree])) + 1
    # （记得修复前面提到的 merge_ratio 漏传的 Bug）
    DG, root_to_id = connect_roots_to_soma_region(img_binary, soma_coords, root_ids, coords_dict, scale_xyz,
                                 start_id, expansion_ratio=0.2, merge_radius_ratio=0.2)

    # 🌟 修改点 2：强行让新生成的中心节点“认祖归宗”
    if app2_root_id:
        orig_root_id = app2_root_id[0]
        # 找到新生成的树的根节点 (没有父节点的那个)
        dg_root = [n for n in DG.nodes() if DG.in_degree(n) == 0][0]

        if dg_root != orig_root_id:
            # 1. 把新生成的根节点 ID 强制改回原 SWC 的根节点 ID
            DG = nx.relabel_nodes(DG, {dg_root: orig_root_id}, copy=False)

            # 2. 同步更新断点映射字典
            for k, v in root_to_id.items():
                if v == dg_root:
                    root_to_id[k] = orig_root_id

            # 3. 还原原根节点在原 SWC 中的真实半径 (可选，如果不加则使用 EDT 算出的半径)
            orig_radius = morph.pos_dict[orig_root_id][5]
            nx.set_node_attributes(DG, {orig_root_id: {'radius': orig_radius}})

    dg_list = _graph_to_swc_list(DG, color_type=2)

    dg_list = _graph_to_swc_list(DG, color_type=2)
    _visualize_step("step07_new_connections", tree_data=dg_list, scale=2)

    # ==========================================================
    # 6 & 7. [极简暴力版] 获取真实中心与直线重连
    # ==========================================================
    # 提取原 SWC 的真实根节点 ID 和信息
    app2_root_id = [nid for nid in topo_ids if topo.pos_dict[nid][-1] == -1]
    if app2_root_id:
        orig_root_id = app2_root_id[0]
        true_center = np.array(topo.pos_dict[orig_root_id][2:5])
        orig_radius = morph.pos_dict[orig_root_id][5]
    else:
        # 极端兜底：如果原 SWC 连 -1 根节点都没有，就用 CSV 的坐标，强行指定 ID=1
        orig_root_id = 1
        true_center = sxyz
        orig_radius = 10.0

    # 将处理后的有向图边转换为供输出的图结构
    new_graph = nx.DiGraph()
    for u,v,w in graph.edges(data='weight'):
        if w == 1:
            all_nodes = [v] + seg_dict[v] + [u]
            for i1, i2 in zip(all_nodes[:-1], all_nodes[1:]): new_graph.add_edge(i2, i1, weight=1)
        elif w == -1:
            all_nodes = [u] + seg_dict[u] + [v]
            for i1, i2 in zip(all_nodes[:-1], all_nodes[1:]): new_graph.add_edge(i1, i2, weight=-1)

    tree_final = []
    added_nodes = set()

    # 第一步：直接写入唯一的真·根节点 (锚定中心，parent 设为 -1)
    tree_final.append((orig_root_id, 1, true_center[0], true_center[1], true_center[2], orig_radius, -1))
    added_nodes.add(orig_root_id)

    # 第二步：写入所有外围存活的树枝
    for node in new_graph.nodes():
        if node in added_nodes: continue

        if new_graph.in_degree(node) >= 1:
            # 正常的树枝，顺着拓扑连向它的父节点
            p_first = list(new_graph.predecessors(node))[0]
            tree_final.append((*morph.pos_dict[node][:6], p_first))
        else:
            # 🌟 暴力连线核心：它是被切断的孤儿根节点！
            # 抛弃所有寻路算法，直接让它的 parent 强行指向 orig_root_id
            tree_final.append((*morph.pos_dict[node][:6], orig_root_id))

        added_nodes.add(node)
    # ==========================================================

    tree_final, neighbor_soma_removed_ids = remove_branches_entering_neighbor_soma(
        tree_final, neighbor_soma_context, sample_step=0.5
    )
    _record_removed_ids(
        neighbor_soma_removed_ids,
        "分支进入周围其他 neuron 的胞体区域",
        "remove_branches_entering_neighbor_soma",
    )

    # Apply the selected QC_utils topology repairs and branch-direction filters.
    tree_final = apply_qc_geometry_steps(
        tree_final, removal_callback=_record_removed_ids
    )

    def remove_sharp_bends(swc_list, min_order=2, angle_thresh=90.0):
        G_swc = nx.DiGraph()
        for node in swc_list:
            nid, ntype, x, y, z, r, pid = node
            G_swc.add_node(nid, type=ntype, coords=np.array([x, y, z]), r=r, pid=pid)
            if pid != -1: G_swc.add_edge(pid, nid)

        swc_roots = [n for n, d in G_swc.in_degree() if d == 0]
        nodes_to_remove = set()
        z_cut_annotations = []

        def dfs_prune(curr, current_order):
            if curr in nodes_to_remove: return

            children = list(G_swc.successors(curr))

            # 遇到分叉 (子节点>1)，则下一级的 order + 1
            next_order = current_order + 1 if len(children) > 1 else current_order

            # 只在 "Branch 内部" 检查 (1个父节点，1个子节点)
            preds = list(G_swc.predecessors(curr))
            if len(preds) == 1 and len(children) == 1 and current_order >= min_order:
                p = preds[0]
                c = children[0]
                cp, cn, cc = G_swc.nodes[p]['coords'], G_swc.nodes[curr]['coords'], G_swc.nodes[c]['coords']

                v1 = cn - cp  # 流入方向
                v2 = cc - cn  # 流出方向
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

                if n1 > 1e-5 and n2 > 1e-5:
                    dot = np.dot(v1/n1, v2/n2)
                    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

                    # 如果这根连贯的线上，突然自己跟自己打了个折返回来 (大于90度的锐角弯)
                    if angle > angle_thresh:
                        # 从子节点开始，把后面的乱线全部斩断
                        descendants = nx.descendants(G_swc, c)
                        descendants.add(c)
                        nodes_to_remove.update(descendants)

                        dir_x, dir_y = (v1[0]/n1), (v1[1]/n1)
                        z_cut_annotations.append({
                            'rx': cn[0], 'ry': cn[1],
                            'text': f"Z-CUT\n{angle:.0f}°",
                            'color': '#ff00ff', # 用洋红色特别标注死结
                            'text_x': cn[0] + dir_x * 40,
                            'text_y': cn[1] + dir_y * 40
                        })
                        return # 剪断了，不用再遍历下面了

            for child in children:
                if child not in nodes_to_remove:
                    dfs_prune(child, next_order)

        for r in swc_roots: dfs_prune(r, 1)

        pruned_tree = [node for node in swc_list if node[0] not in nodes_to_remove]
        return pruned_tree, z_cut_annotations

    # 执行剪枝：从 Order 2 起，只要转角大于 90 度，全部切掉
    tree_final_pruned, z_cut_anns = remove_sharp_bends(tree_final, min_order=2, angle_thresh=90.0)
    _record_removed_ids(
        {int(node[0]) for node in tree_final}
        - {int(node[0]) for node in tree_final_pruned},
        "二级及以上分支出现大于90度的局部急转",
        "remove_sharp_bends",
    )

    def remove_crosstalk(swc_list, min_order=2):
        G_swc = nx.DiGraph()
        for node in swc_list:
            nid, ntype, x, y, z, r, pid = node
            G_swc.add_node(nid, type=ntype, coords=np.array([x, y, z]), r=r, pid=pid)
            if pid != -1: G_swc.add_edge(pid, nid)

        swc_roots = [n for n, d in G_swc.in_degree() if d == 0]
        nodes_to_remove = set()
        crosstalk_annotations = []

        def dfs_crosstalk(curr, current_order, branch_start_coord, path_length):
            if curr in nodes_to_remove: return

            curr_coord = G_swc.nodes[curr]['coords']

            children = list(G_swc.successors(curr))
            is_bifurcation = len(children) > 1

            # 提取两个判定串线的核心指标
            euc_dist = np.linalg.norm(curr_coord - branch_start_coord) # 直线距离
            tortuosity = (path_length / euc_dist) if euc_dist > 1e-3 else 1.0 # 曲折度

            is_crosstalk = False
            reason = ""

            # 只有长到远端且路径极度曲折时才判定串线；不再根据远端增粗剪枝。
            if current_order >= min_order and euc_dist > 15.0:
                # 极度扭曲的跨界飞线：实际路径是直线距离的 2.5 倍以上。
                if tortuosity > 2.5:
                    is_crosstalk = True
                    reason = f"Wiggly\n(T:{tortuosity:.1f})"

            if is_crosstalk:
                descendants = nx.descendants(G_swc, curr)
                descendants.add(curr)
                nodes_to_remove.update(descendants)

                crosstalk_annotations.append({
                    'rx': curr_coord[0], 'ry': curr_coord[1],
                    'text': f"CROSSTALK\n{reason}",
                    'color': '#ffaa00', # 用橙色标注串线
                    'text_x': curr_coord[0] + 30,
                    'text_y': curr_coord[1] + 30
                })
                return # 发现串线，直接砍断整条分支

            # 继续顺着树枝往下爬
            for child in children:
                if child not in nodes_to_remove:
                    c_coord = G_swc.nodes[child]['coords']
                    step_len = np.linalg.norm(c_coord - curr_coord)

                    # 如果遇到分叉路口，下一级分支重新开始记录新的起点坐标
                    if is_bifurcation:
                        dfs_crosstalk(child, current_order + 1, curr_coord, step_len)
                    else:
                        # 如果没有分叉，继续累加这条路的长度
                        dfs_crosstalk(child, current_order, branch_start_coord, path_length + step_len)

        for r in swc_roots:
            dfs_crosstalk(r, 1, G_swc.nodes[r]['coords'], 0.0)

        pruned_tree = [node for node in swc_list if node[0] not in nodes_to_remove]
        return pruned_tree, crosstalk_annotations

    # 1. 执行跨界串线剪枝
    tree_final_pruned_2, crosstalk_anns = remove_crosstalk(tree_final_pruned, min_order=2)
    _record_removed_ids(
        {int(node[0]) for node in tree_final_pruned}
        - {int(node[0]) for node in tree_final_pruned_2},
        "远端分支曲折度大于2.5",
        "remove_crosstalk",
    )

    # 2. 把 Z-Bend 和 Cross-talk 的标注合并在一起，统一扔给 Step 8 显示
    combined_anns = z_cut_anns + crosstalk_anns
    _visualize_step("step08_sharp_bends", tree_data=tree_final_pruned_2, scale=2, annotations=combined_anns)

    # 3. 更新最终输出树
    tree_final = tree_final_pruned_2
    _complete_pruning_reason_coverage(tree_final)
    # ==========================================================

    if debug:
        swc_name = swc_name.replace('.tif', '')
        out_swc = os.path.join(out_swc_dir, f'{swc_name}.swc')
        tree_to_save = tree_final
        temporary_swc = out_swc + '.writing'
        write_swc(tree_to_save, temporary_swc)
        os.replace(temporary_swc, out_swc)

        # 注意：这里改成了 step09
        _visualize_step("step09_final_result", tree_data=tree_final, scale=1)
    if(int(swc_name.split("_")[1])%1==0):
        reason_report = {
            "cell_id": cell_id,
            "input_swc": swc_neu,
            "output_swc": out_swc,
            "removed_nodes": {
                str(node_id): info
                for node_id, info in sorted(pruning_reasons.items())
                if node_id not in {int(node[0]) for node in tree_final}
            },
        }
        with open(
            os.path.join(mip_dir, f"{prefix}_pruning_reasons.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(reason_report, stream, ensure_ascii=False, indent=2)
        _save_combined_grid_plot()
    return tree_final, graph

@contextlib.contextmanager
def suppress_stdout(suppress=True):
    if suppress:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try: yield
            finally: sys.stdout = old_stdout
    else: yield

def check_exist(
    gcut_swc_file,
    mask_dir,
    raw_image_dir,
    out_swc_dir,
    binary_input=True,
    pre_traced=True,
    start_cell_id=0,
    skip_existing_outputs=True,
):
    """
    Build a pruning task from one selected G-Cut SWC.

    The task source is now gcut_selected_swc_dir.  A selected G-Cut filename
    may contain additional suffixes (for example ``_0000_gcut_soma_1_TARGET``).
    Companion images are therefore matched by *cell ID*, not by the full SWC
    stem:
        gcut_output/image_<cell_id>_*.swc
        mask/image_<cell_id>.tif
        img/image_<cell_id>_0000.tif
    """
    if not os.path.exists(gcut_swc_file):
        return False, f"Selected G-Cut SWC missing: {gcut_swc_file}"
    if not swc_has_nodes(gcut_swc_file):
        return False, f"Selected G-Cut SWC is empty/invalid: {gcut_swc_file}"

    prefix = os.path.splitext(os.path.basename(gcut_swc_file))[0]
    out_swc_path = prune_output_swc_path(gcut_swc_file, out_swc_dir)
    # A non-empty, parseable output SWC is treated as a completed result.
    # This makes the script resumable: re-running it only submits pending cells.
    if skip_existing_outputs and swc_has_nodes(out_swc_path):
        return False, f"Already processed: {out_swc_path}"

    try:
        cell_id = extract_cell_id_from_gcut_swc(gcut_swc_file)
    except ValueError as exc:
        return False, str(exc)
    if cell_id < start_cell_id:
        return False, (
            f"Skipped: Cell ID {cell_id} < start_cell_id ({start_cell_id})"
        )

    if not pre_traced:
        return False, "Not implemented: pre_traced=False is currently not supported"

    # Important: use the extracted cell ID rather than the complete selected
    # G-Cut SWC stem.  For example, the following SWC:
    #   image_100007_0000_gcut_soma_1_TARGET.swc
    # must read:
    #   mask/image_100007.tif
    #   img/image_100007_0000.tif
    mask_stem = f"image_{cell_id}"
    raw_stem = f"image_{cell_id}_0000"
    input_imgfile = resolve_image_by_stem(mask_dir, mask_stem)
    if binary_input:
        raw_imgfile = resolve_image_by_stem(raw_image_dir, raw_stem)
    else:
        raw_imgfile = input_imgfile

    if not os.path.exists(input_imgfile):
        return False, f"Mask input missing: {input_imgfile}"
    if not os.path.exists(raw_imgfile):
        return False, f"Raw image missing: {raw_imgfile}"

    return True, (input_imgfile, gcut_swc_file, raw_imgfile)
def worker_task(gcut_swc_file, config):
    selected_swc_dir = config["selected_swc_dir"]
    mask_dir = config["mask_dir"]
    raw_image_dir = config["raw_image_dir"]
    out_swc_dir = config["out_swc_dir"]
    mip_dir, meta, sphere_zradius = (
        config["mip_dir"],
        config["meta"],
        config["sphere_zradius"],
    )
    downsample_scale = config["downsample_scale"]
    binary_input, pre_traced = config["binary_input"], config["pre_traced"]
    debug, verbose = config["debug"], config["verbose"]
    skip_existing_outputs = config.get("skip_existing_outputs", True)
    soma_img_dir = config.get("soma_img_dir")
    soma_mask_dir = config.get("soma_mask_dir")

    filename = os.path.basename(gcut_swc_file)
    with suppress_stdout(suppress=not verbose):
        try:
            # selected_swc_dir is kept in config as an explicit audit field:
            # every submitted task must originate from this directory.
            if os.path.dirname(os.path.abspath(gcut_swc_file)) != os.path.abspath(selected_swc_dir):
                return {
                    "status": "error",
                    "file": filename,
                    "category": "outside_selected_swc_dir",
                    "reason": "SWC 文件不在配置的 selected_swc_dir 中",
                    "msg": (
                        "SWC is outside selected_swc_dir: "
                        f"{gcut_swc_file}"
                    ),
                }

            is_valid, result_data = check_exist(
                gcut_swc_file,
                mask_dir,
                raw_image_dir,
                out_swc_dir,
                binary_input=binary_input,
                pre_traced=pre_traced,
                skip_existing_outputs=skip_existing_outputs,
            )
            if not is_valid:
                status, category = classify_precheck_result(result_data)
                return {
                    "status": status,
                    "file": filename,
                    "category": category,
                    "reason": str(result_data),
                    "msg": str(result_data),
                }

            input_imgfile, swc_neu, raw_imgfile = result_data
            tree_final, _ = process_neuron_morphology(
                input_imgfile,
                swc_neu,
                raw_imgfile,
                out_swc_dir,
                mip_dir,
                meta,
                sphere_zradius=sphere_zradius,
                downsample_scale=downsample_scale,
                binary_input=binary_input,
                pre_traced=pre_traced,
                debug=debug,
                show_steps=True,
                soma_img_dir=soma_img_dir,
                soma_mask_dir=soma_mask_dir,
            )
            if not tree_final:
                return {
                    "status": "error",
                    "file": filename,
                    "category": "no_salient_tree",
                    "reason": "所有候选分支均被过滤，最终没有可输出的显著树结构",
                    "msg": "No salient tree found after pruning and connectivity checks.",
                }
            return {
                "status": "success",
                "file": filename,
                "category": "success",
                "reason": "处理完成并生成非空 SWC 输出",
                "msg": "",
            }
        except Exception as exc:
            category, reason = classify_processing_exception(exc)
            return {
                "status": "error",
                "file": filename,
                "category": category,
                "reason": reason,
                "msg": traceback.format_exc(),
            }

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    config = {
        "binary_input": True,
        "tgamma": True,
        "debug": True,
        "sphere_zradius": 10,
        "downsample_scale": np.array([1, 1, 1]),
        "meta_file": PATHS["meta_file"],
        "pre_traced": True,

        # Resume mode: on every new run, skip a cell once its output SWC
        # already exists and contains at least one valid SWC node.
        # Set this to False only when you deliberately want to overwrite outputs.
        "skip_existing_outputs": True,

        # Task source: only selected G-Cut SWCs are scanned and submitted.
        "selected_swc_dir": PATHS["gcut_selected_swc_dir"],

        # Per-SWC companion files still needed by pruning.
        "mask_dir": PATHS["merged_mask_dir"],
        "raw_image_dir": PATHS["image_1um_dir"],
        "mip_dir": PATHS["prune_mip_dir"],
        "out_swc_dir": PATHS["prune_swc_dir"],
        "soma_img_dir": PATHS["soma_crop_dir"],
        "soma_mask_dir": PATHS["soma_seg_dir"],
        "num_workers": 8,
        "verbose": True,
    }

    log_file = PATHS["prune_error_log"]
    os.makedirs(config["out_swc_dir"], exist_ok=True)
    os.makedirs(config["mip_dir"], exist_ok=True)
    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    statistics_dir = os.path.commonpath([
        os.path.abspath(config["out_swc_dir"]),
        os.path.abspath(config["mip_dir"]),
    ])
    if statistics_dir == os.path.sep:
        statistics_dir = os.path.abspath(config["out_swc_dir"])
    statistics_csv = os.path.join(
        statistics_dir,
        f"pruning_run_statistics_{run_id}.csv",
    )
    run_records = []

    print("Loading Metadata...")
    config["meta"] = pd.read_csv(
        config["meta_file"],
        index_col="cell_id",
        low_memory=False,
        encoding="latin1",
    )

    print(f"Scanning selected G-Cut SWCs in {config['selected_swc_dir']} ...")
    input_files = []
    invalid_swc_files = []
    already_processed = 0
    already_processed_files = []

    with os.scandir(config["selected_swc_dir"]) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith(".swc"):
                continue

            if not swc_has_nodes(entry.path):
                invalid_swc_files.append(entry.name)
                continue

            # Output name is based on the selected SWC stem (image_<cell_id>).
            # Filter completed cells before creating any multiprocessing task.
            if (
                config.get("skip_existing_outputs", True)
                and swc_has_nodes(
                    prune_output_swc_path(entry.path, config["out_swc_dir"])
                )
            ):
                already_processed += 1
                already_processed_files.append(entry.name)
                continue

            input_files.append(entry.path)

    input_files.sort()
    run_records.extend({
        "file": name,
        "status": "failed",
        "category": "empty_or_invalid_swc",
        "reason": "Selected G-Cut SWC 为空或不包含合法的七列节点记录",
    } for name in invalid_swc_files)
    run_records.extend({
        "file": name,
        "status": "skipped",
        "category": "already_processed",
        "reason": "输出目录中已存在非空 SWC，启动扫描时跳过",
    } for name in already_processed_files)
    total_discovered = (
        len(input_files) + len(invalid_swc_files) + already_processed
    )
    print(
        f"Found {len(input_files)} pending selected SWCs "
        f"({already_processed} already processed, "
        f"{len(invalid_swc_files)} empty/invalid). "
        f"Starting processing with {config['num_workers']} workers..."
    )

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            f"Processing started at {run_started_at}\n"
            + "=" * 50
            + "\n"
        )
        if invalid_swc_files:
            f.write("[INVALID SELECTED SWCs]\n")
            for name in invalid_swc_files:
                f.write(f"{name}\n")
            f.write("-" * 50 + "\n")
        if already_processed_files:
            f.write("[ALREADY PROCESSED - SKIPPED AT STARTUP]\n")
            for name in already_processed_files:
                f.write(f"{name}\n")
            f.write("-" * 50 + "\n")

    with ProcessPoolExecutor(max_workers=config["num_workers"]) as executor:
        task_func = partial(worker_task, config=config)
        futures = {
            executor.submit(task_func, swc_file): os.path.basename(swc_file)
            for swc_file in input_files
        }

        try:
            for future in tqdm(
                as_completed(futures),
                total=len(input_files),
                desc="Pruning selected G-Cut SWCs",
            ):
                filename = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    category, reason = classify_processing_exception(exc)
                    result = {
                        "status": "error",
                        "file": filename,
                        "category": category,
                        "reason": reason,
                        "msg": traceback.format_exc(),
                    }

                result_category = result.get(
                    "category",
                    "success" if result["status"] == "success" else "unclassified",
                )
                result_reason = result.get("reason", result.get("msg", ""))
                run_records.append({
                    "file": result["file"],
                    "status": result["status"],
                    "category": result_category,
                    "reason": result_reason,
                })

                if result["status"] == "error":
                    print(
                        f"!! Error in {result['file']} "
                        f"[{result_category}]: {result_reason}"
                    )
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"\n[ERROR] File: {result['file']}\n"
                            f"Category: {result_category}\n"
                            f"Reason: {result_reason}\n"
                            + (
                                f"Traceback:\n{result.get('msg', '')}\n"
                                if result.get("msg") else ""
                            )
                            + "-" * 50
                            + "\n"
                        )

                elif result["status"] == "skipped":
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"\n[SKIPPED] File: {result['file']}\n"
                            f"Category: {result_category}\n"
                            f"Reason: {result_reason}\n"
                            + "-" * 50
                            + "\n"
                        )

        except Exception as e:
            print(f"Critical error handling future: {e}")
            category, reason = classify_processing_exception(e)
            run_records.append({
                "file": "<executor>",
                "status": "error",
                "category": f"executor_{category}",
                "reason": reason,
            })

    run_finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    statistics = write_run_statistics_csv(
        statistics_csv,
        run_started_at,
        run_finished_at,
        run_records,
        total_discovered=total_discovered,
        pending_submitted=len(input_files),
    )
    print("\nAll processing complete.")
    print(
        "Run statistics: "
        f"success={statistics['success_count']}, "
        f"failed={statistics['failure_count']}, "
        f"skipped={statistics['skipped_count']}"
    )
    if statistics["failure_category_counts"]:
        print(
            "Failure categories: "
            + ", ".join(
                f"{category}={count}"
                for category, count in sorted(
                    statistics["failure_category_counts"].items()
                )
            )
        )
    print(f"Statistics CSV: {statistics['csv_path']}")
