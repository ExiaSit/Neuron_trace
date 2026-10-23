"""*================================================================
*   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
*   
*   Filename    : swc_handler.py
*   Author      : Yufeng Liu
*   Date        : 2021-03-15
*   Description : 
*
================================================================*"""
import os
import re
import numpy as np
from copy import deepcopy
from collections import defaultdict
from skimage.draw import line_nd
import subprocess
import pandas as pd


NEURITE_TYPES = {
    'soma': [1],
    'axon': [2],
    'basal dendrite': [3],
    'apical dendrite': [4],
    'dendrite': [3,4],
}


def load_spacings(spacing_file, zxy_order=False):
    """
    Load the spacing information for each brain. The spacing here refers to
    the resolution along x,y,z axes.
    """
    spacing_dict = {}
    with open(spacing_file, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            ctxts = line.split(',')
            brain_id = ctxts[0]
            if not brain_id.isdigit(): 
                continue # the brain is encoded as digits

            brain_id = int(brain_id)
            spacing = tuple(map(float, ctxts[1:]))
            if zxy_order:
                spacing = (spacing[2],spacing[0],spacing[1])
            spacing_dict[brain_id] = spacing
    
    return spacing_dict


def parse_swc(swc_file):
    tree = []
    with open(swc_file) as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            if line[0] == '#': continue
            idx, type_, x, y, z, r, p = line.split()[:7]
            idx = int(idx)
            type_ = int(type_)
            x = float(x)
            y = float(y)
            z = float(z)
            r = float(r)
            p = int(p)
            tree.append((idx, type_, x, y, z, r, p))
    
    return tree


def write_swc(tree, swc_file, header=tuple()):
    if header is None:
        header = []
    with open(swc_file, 'w') as fp:
        for s in header:
            if not s.startswith("#"):
                s = "#" + s
            if not s.endswith("\n") or not s.endswith("\r"):
                s += "\n"
            fp.write(s)
        fp.write(f'##n type x y z r parent\n')
        for leaf in tree:
            idx, type_, x, y, z, r, p = leaf
            fp.write(f'{idx:d} {type_:d} {x:.5f} {y:.5f} {z:.5f} {r:.1f} {p:d}\n')


def find_soma_node(tree, p_soma=-1, p_idx_in_leaf=6):
    for leaf in tree:
        if leaf[p_idx_in_leaf] == p_soma:
            #print('Soma: ', leaf)
            return leaf[0]
    #raise ValueError("Could not find the soma node!")
    return -99


def find_soma_index(tree, p_soma=-1):
    for i, leaf in enumerate(tree):
        if leaf[6] == p_soma:
            return i
    #raise ValueError("find_soma_index: Could not find the somma node!")
    return -99


def get_child_dict(tree, p_idx_in_leaf=6):
    child_dict = {}
    for leaf in tree:
        p_idx = leaf[p_idx_in_leaf]
        if p_idx in child_dict:
            child_dict[p_idx].append(leaf[0])
        else:
            child_dict[p_idx] = [leaf[0]]
    return child_dict


def get_index_dict(tree):
    index_dict = {}
    for i, leaf in enumerate(tree):
        idx = leaf[0]
        index_dict[idx] = i
    return index_dict


def is_in_box(x, y, z, imgshape):
    """
    imgshape must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
        x > imgshape[2] - 1 or \
        y > imgshape[1] - 1 or \
        z > imgshape[0] - 1:
        return False
    return True

def is_in_bbox(x, y, z, zyxzyx):
    """ 
    zyxzyx is bbox in format of [(zmin, ymin, xmin), (zmax, ymax, xmax)]
    """
    (zmin, ymin, xmin), (zmax, ymax, xmax) = zyxzyx
    if x < xmin or y < ymin or z < zmin or \
        x > xmax or \
        y > ymax or \
        z > zmax:
        return False
    return True

def prune(tree: list, ind_set: set):
    """
    prune all nodes given by ind_set in morph
    """
    child_dict = get_child_dict(tree)
    index_dict = get_index_dict(tree)
    tree = deepcopy(tree)
    for i in ind_set:
        q = []
        ind = index_dict[i]
        if tree[ind] is None:
            continue
        tree[ind] = None
        if i in child_dict:
            q.extend(child_dict[i])
        while len(q) > 0:
            head = q.pop(0)
            ind = index_dict[head]
            if tree[ind] is None:
                continue
            tree[ind] = None
            if head in child_dict:
                q.extend(child_dict[head])
    return [t for t in tree if t is not None]


def trim_swc(tree_orig, imgshape, keep_candidate_points=True, bfs=True):
    """
    Trim the out-of-box and non_connecting leaves
    """
    if bfs:
        ib = set(t[0] for t in tree_orig if is_in_box(*t[2:5], imgshape))
        if keep_candidate_points:
            child_dict = get_child_dict(tree_orig)
            ib = ib.union(*(child_dict[i] for i in ib if i in child_dict))
        return prune(tree_orig, set(t[0] for t in tree_orig) - ib)

    def traverse_leaves(idx, child_dict, good_points, cand_pints, pos_dict):
        leaf = pos_dict[idx]
        p_idx, ib = leaf[-2:]

        if (p_idx in good_points) or (p_idx == -1):
            if ib:
                good_points.add(idx)    # current node
            else:
                cand_points.add(idx)
                return

        if idx not in child_dict:
            return

        for new_idx in child_dict[idx]:
            traverse_leaves(new_idx, child_dict, good_points, cand_pints, pos_dict)

    # execute trimming
    pos_dict = {}
    tree = deepcopy(tree_orig)
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        leaf = (idx, type_, x, y, z, r, p, is_in_box(x,y,z,imgshape))
        pos_dict[idx] = leaf
        tree[i] = leaf

    good_points = set() # points and all its upstream parents are in-box
    cand_points = set() # all upstream parents are in-box, itself not
    # initialize the visited set with soma, whose parent index is -1
    soma_idx = None
    for leaf in tree:
        if leaf[-2] == -1:
            soma_idx = leaf[0]
            break
    #print(soma_idx)

    child_dict = {}
    for leaf in tree:
        if leaf[-2] in child_dict:
            child_dict[leaf[-2]].append(leaf[0])
        else:
            child_dict[leaf[-2]] = [leaf[0]]
    # do DFS searching
    #print(soma_idx)
    traverse_leaves(soma_idx, child_dict, good_points, cand_points, pos_dict)
    #print("#good/#cand/#total:", len(good_points), len(cand_points), len(pos_dict))  

    # return the tree, (NOTE: without order)
    tree_trim = []
    if keep_candidate_points:
        keep_points = good_points | cand_points
    else:
        keep_points = good_points

    for i, leaf in enumerate(tree):
        idx = leaf[0]
        if idx in keep_points:
            tree_trim.append(leaf[:-1])

    return tree_trim


def trim_out_of_box(tree_orig, imgshape, keep_candidate_points=True):
    """ 
    Trim the out-of-box leaves
    """
    # execute trimming
    child_dict = {}
    for leaf in tree_orig:
        if leaf[-1] in child_dict:
            child_dict[leaf[-1]].append(leaf[0])
        else:
            child_dict[leaf[-1]] = [leaf[0]]
    
    pos_dict = {}
    for i, leaf in enumerate(tree_orig):
        pos_dict[leaf[0]] = leaf

    tree = []
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        ib = is_in_box(x,y,z,imgshape)
        if ib: 
            tree.append(leaf)
        elif keep_candidate_points:
            if p in pos_dict and is_in_box(*pos_dict[p][2:5], imgshape):
                tree.append(leaf)
            elif idx in child_dict:
                for ch_leaf in child_dict[idx]:
                    if is_in_box(*pos_dict[ch_leaf][2:5], imgshape):
                        tree.append(leaf)
                        break
    return tree


def get_specific_neurite(tree, type_id):
    if (not isinstance(type_id, list)) and (not isinstance(type_id, tuple)):
        type_id = (type_id,)
    
    new_tree = []
    for leaf in tree:
        if leaf[1] in type_id:
            new_tree.append(leaf)
    return new_tree


def shift_swc(swc_file, sx, sy, sz):
    if type(swc_file) == list:
        tree = swc_file
    else:
        tree = parse_swc(swc_file)
    new_tree = []
    for node in tree:
        idx, type_, x, y, z, r, p = node
        x = x - sx
        y = y - sy
        z = z - sz
        node = (idx, type_, x, y, z, r, p)
        new_tree.append(node)
    return new_tree


def scale_swc(swc_file, scale):
    if type(swc_file) == list:
        tree = swc_file
    else:
        tree = parse_swc(swc_file)
    if isinstance(scale, (int, float)):
        scale_x, scale_y, scale_z = scale, scale, scale
    elif isinstance(scale, tuple) or isinstance(scale, list):
        scale_x, scale_y, scale_z = scale
    else:
        raise NotImplementedError(f"Type of parameter scale {type(scale)} is not supported!")

    new_tree = []
    for node in tree:
        idx, type_, x, y, z, r, p = node
        x *= scale_x
        y *= scale_y
        z *= scale_z
        node = (idx, type_, x, y, z, r, p)
        new_tree.append(node)
    return new_tree

def flip_swc(swc_file, axis='y', dim=None):
    if type(swc_file) == list:
        tree = swc_file
    else:
        tree = parse_swc(swc_file)
    
    new_tree = []
    for node in tree:
        idx, type_, x, y, z, r, p = node
        if axis == 'x':
            x = dim - x
        elif axis == 'y':
            y = dim - y
        elif axis == 'z':
            z = dim - z
        node = (idx, type_, x, y, z, r, p)
        new_tree.append(node)
    return new_tree

def crop_tree_by_bbox(morph, bbox, keep_candidate_points=True):
    """ 
    Crop swc by trim all nodes out-of-bbox. This function differs from `trim_out_of_box` it does
    not assume center cropping
    """
    if isinstance(morph, list):
        mtree = morph
    else:
        mtree = morph.tree

    tree = []
    for i, leaf in enumerate(morph.tree):
        idx, type_, x, y, z, r, p = leaf[:7]
        ib = is_in_bbox(x,y,z,bbox) 
        if ib: 
            tree.append(leaf)
            if keep_candidate_points and (idx in morph.child_dict):
                for ch_leaf in morph.child_dict[idx]:
                    if not is_in_bbox(*morph.pos_dict[ch_leaf][2:5], bbox):
                        tree.append(morph.pos_dict[ch_leaf])
    return tree

def crop_spheric_from_soma(df_tree, radius): # radius in micrometer
    # NOTE that in this function we use the the pandas.DataFrame as tree object, instead of list
    # dataframe: index, cololumns = (type, x, y, z, r, pid)
    # the dataframe is parsed using: pd.read_csv(in_swc, sep=' ', names=('type', 'x', 'y', 'z', 'r', 'pid'), comment='#', index_col=0)

    ###### helper functions #######
    def dfs(irow, child_dict, good_points, tree):
        #print(irow)
        p_idx = tree.loc[irow, 'pid']
        if (p_idx in good_points) or (p_idx == -1):
            good_points.add(irow)

        if irow not in child_dict:
            return

        for new_idx in child_dict[irow]:
            dfs(new_idx, child_dict, good_points, tree)

    ###### end of helper functions ######


    # remove out-of-sphere points
    sidx = np.nonzero(df_tree.pid == -1)[0][0]
    sid = df_tree.iloc[sidx].name
    dxyz = df_tree[['x', 'y', 'z']] - df_tree.iloc[sidx][['x', 'y', 'z']]
    dR = np.sqrt((dxyz**2).sum(axis=1))
    tree_in = df_tree[dR < radius]
    
    # remove disconnected nodes
    # first, get the child dict
    child_dict = {}
    for irow, row in tree_in.iterrows():
        if row.pid in child_dict:
            child_dict[row.pid].append(irow)
        else:
            child_dict[row.pid] = [irow]
    # second, get the to-delete nodes iteratively
    good_points = set([])
    dfs(sid, child_dict, good_points, tree_in)

    # third, extract the good points
    tree_good = tree_in[tree_in.index.isin(good_points)]
    return tree_good


def tree_to_voxels(tree, crop_box):
    # crop_box in (z,y,x) order
    # initialize position dict
    pos_dict = {}
    new_tree = []
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        leaf_new = (*leaf, is_in_box(x,y,z,crop_box))
        pos_dict[leaf[0]] = leaf_new
        new_tree.append(leaf_new)
    tree = new_tree

    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if p == -1: continue # soma

        if p not in pos_dict:
            continue

        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError

        # draw line connecting each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)
        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    voxels = []
    for (xi,yi,zi) in zip(xl,yl,zl):
        if is_in_box(xi,yi,zi,crop_box):
            voxels.append((xi,yi,zi))
    # remove duplicate points
    voxels = np.array(list(set(voxels)), dtype=np.float32)
    return voxels


def rm_disconnected(tree: list, anchor: int):
    roots = [t[0] for t in tree if t[6] == -1]
    ch = get_child_dict(tree)
    idx = get_index_dict(tree)
    flag = np.zeros(len(tree), dtype=int)
    for r in roots:
        q = [r]
        while len(q) > 0:
            head = q.pop(0)
            flag[idx[head]] = r
            if head in ch:
                q.extend(ch[head])
    ind = flag[idx[anchor]]
    return prune(tree, set(t[0] for t, f in zip(tree, flag) if f != ind))

def get_soma_from_swc(swcfile, match_str='.* -1\n'):
    # fast parse swc information
    # only for swc, not eswc
    with open(swcfile) as fp:
        soma_str = re.search(match_str, fp.read()).group()
    soma = soma_str.split()
    return soma

def resample_swc(swc_in, swc_out, step=2, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x', correction=True):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x resample_swc -f resample_swc -i {swc_in} -o {swc_out} -p {step}'
    print(swc_in, swc_out)
    p = subprocess.check_output(cmd_str, shell=True)
    if correction:
        # The built-in resample_swc has a bug: the first node is commented out, and there are two additional columns
        subprocess.run(f"sed -i 's/pid1/pid\\n1/g' {swc_out}; sed -i 's/ -1 -1//g' {swc_out}", shell=True)
    return True

def sort_swc(swc_in, swc_out=None, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x', retype=True):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x sort_neuron_swc -f sort_swc -i {swc_in} -o {swc_out}'
    p = subprocess.check_output(cmd_str, shell=True)

    if retype:
        # retype
        df = pd.read_csv(swc_out, sep=' ', names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'), comment='#', index_col=False)
        df['type'] = 3
        df.loc[0, 'type'] = 1
        df.to_csv(swc_out, sep=' ', index=False)

    return True

def resample_sort_swc(swc_in, swc_out, retype=True):
    resample_swc(swc_in, swc_out)
    sort_swc(swc_out, swc_out, retype=retype)


def resample_swc_customized_core(swc_lines, step_length=1.0):
    nodes = {}       # 存储节点数据: node_id -> [n, type, x, y, z, r, parent]
    children = defaultdict(list)
    comments = []    # 用于原封不动保留头部的注释信息
    
    # 2. 极速解析非注释行
    for line in swc_lines:
        line_s = line.strip()
        if not line_s:
            continue
        if line_s.startswith('#'):
            comments.append(line_s + '\n')
            continue
            
        parts = line_s.split()
        if len(parts) >= 7:
            n, t, x, y, z, r, p = parts
            nodes[n] = [int(n), int(t), float(x), float(y), float(z), float(r), p]
            if p != '-1':
                children[p].append(n)
                
    # 3. 寻找关键拓扑节点 (Key Nodes: 胞体、分叉点、末端点)
    key_nodes = set(['-1'])
    roots = []  # 记录所有的根节点
    
    for n, data in nodes.items():
        p = data[6]
        c_list = children[n]
        if p == '-1':
            roots.append(n)
            key_nodes.add(n)
        elif len(c_list) != 1:  # 分叉点 (>1) 或 叶子节点 (0)
            key_nodes.add(n)

    # 4. 严格自顶向下 (Top-down BFS) 提取分支，保证拓扑排序 (Parent ID < Child ID)
    branches = []
    queue = roots.copy()
    
    while queue:
        start_node = queue.pop(0) # 取出一个已知的上游关键节点
        
        # 遍历它发出的所有直接下游子分支
        for child in children[start_node]:
            branch_path = [start_node]
            curr = child
            
            # 顺藤摸瓜，直到碰见下一个关键节点
            while curr not in key_nodes:
                branch_path.append(curr)
                # 容错：如果形态断裂没有子节点了，强行中断
                if not children[curr]: 
                    break
                curr = children[curr][0] # 非关键节点必然只有唯一一个子节点
                
            branch_path.append(curr) # 把游历到的终点(下一个关键节点)加入
            branches.append(branch_path)
            
            # 如果终点是分叉点，加入队列，等待下一次向下探索
            if len(children[curr]) > 0:
                queue.append(curr)

    # 5. Numpy 向量化插值重采样
    new_nodes = []
    new_id_counter = 1
    key_node_map = {} # 存储旧关键节点到新节点 ID 的映射
    
    for branch in branches:
        # 提取当前分支的 XYZR 坐标矩阵
        coords = np.array([[nodes[n][2], nodes[n][3], nodes[n][4], nodes[n][5]] for n in branch])
        types = [nodes[n][1] for n in branch] # 类别
        
        # 计算线段之间的欧氏距离和累积弧长
        diffs = np.diff(coords[:, :3], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cum_length = np.insert(np.cumsum(segment_lengths), 0, 0) # [0, L1, L1+L2, ...]
        
        total_length = cum_length[-1]
        
        # 根据步长生成新的采样点位置 (保证包含起点和终点)
        if total_length <= step_length:
            new_cum = np.array([0, total_length])
        else:
            new_cum = np.arange(0, total_length, step_length)
            if total_length - new_cum[-1] > 1e-5: # 确保终点被加入
                new_cum = np.append(new_cum, total_length)
                
        # 使用 Numpy 沿弧长线性插值 XYZ 和 R
        new_x = np.interp(new_cum, cum_length, coords[:, 0])
        new_y = np.interp(new_cum, cum_length, coords[:, 1])
        new_z = np.interp(new_cum, cum_length, coords[:, 2])
        new_r = np.interp(new_cum, cum_length, coords[:, 3])
        
        # 6. 重新分配 ID 和重建拓扑连接
        parent_node_original = branch[0]
        child_node_original = branch[-1]
        
        # 确定起点的新 ID
        if parent_node_original in key_node_map:
            current_parent_id = key_node_map[parent_node_original]
        else:
            # 根节点初始化
            current_parent_id = new_id_counter
            key_node_map[parent_node_original] = current_parent_id
            new_nodes.append(f"{current_parent_id} {types[0]} {new_x[0]:.3f} {new_y[0]:.3f} {new_z[0]:.3f} {new_r[0]:.3f} -1\n")
            new_id_counter += 1

        # 生成中间节点和终点
        for i in range(1, len(new_cum)):
            this_id = new_id_counter
            new_id_counter += 1
            
            # 最后一个点继承原始的类型，中间点使用起点类型
            t = types[-1] if i == len(new_cum) - 1 else types[0] 
            
            new_nodes.append(f"{this_id} {t} {new_x[i]:.3f} {new_y[i]:.3f} {new_z[i]:.3f} {new_r[i]:.3f} {current_parent_id}\n")
            current_parent_id = this_id
            
        # 记录终点（关键节点）的新 ID 映射，供下游分支衔接使用
        key_node_map[child_node_original] = current_parent_id

    return new_nodes, comments

def resample_swc_customized(in_swc, out_swc, step_length=1.0):
    """
    高性能 SWC 重采样算法
    :param in_swc: 输入的 SWC 文件路径
    :param step_length: 重采样步长 (默认 1.0 um)
    :return: 包含重采样后所有行(含注释)的字符串列表，可直接 writelines() 写入文件
    """
    if not os.path.exists(in_swc):
        raise FileNotFoundError(f"找不到文件: {in_swc}")
        
    # 1. 极速读取文件到内存
    with open(in_swc, 'r') as f:
        swc_lines = f.readlines()

    new_nodes, comments = resample_swc_customized_core(swc_lines, step_length)
    
    new_swc_lines = comments + new_nodes

    if out_swc and isinstance(out_swc, str):
        with open(out_swc, 'w') as fp:
            fp.writelines(new_swc_lines)

    return new_swc_lines

