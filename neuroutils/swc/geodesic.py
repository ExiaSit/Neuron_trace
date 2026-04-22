import numpy as np
import networkx as nx
from collections import deque


def compute_geodesic_distance(graph, root_id):
    """
    使用 BFS (队列) 计算从 root 到全图所有节点的路径距离 (Geodesic Distance)。

    Args:
        graph (nx.Graph): 包含 x, y, z 坐标的图
        root_id (int): 起始根节点 ID

    Returns:
        dict: {node_id: distance_value}
    """
    # 初始化距离字典：所有点设为无穷大，Root 设为 0
    distances = {n: float('inf') for n in graph.nodes}

    # 容错：如果 root 已经在 merge 步骤被删掉了（变成了别的 ID），
    # 调用者应该传入新的 root ID。这里做个防御性检查。
    if root_id not in graph.nodes:
        # 如果找不到 root，返回全 0 或者报错，这里选择返回全 0 并打印警告
        print(f"Warning: Root {root_id} not found in graph! Returning 0 distances.")
        return {n: 0.0 for n in graph.nodes}

    distances[root_id] = 0.0

    # BFS 队列
    queue = deque([root_id])

    while queue:
        u = queue.popleft() # 取出队首 (BFS)

        u_node = graph.nodes[u]
        u_pos = np.array([u_node['x'], u_node['y'], u_node['z']])

        for v in graph.neighbors(u):
            # 计算边权重 (欧氏距离)
            v_node = graph.nodes[v]
            v_pos = np.array([v_node['x'], v_node['y'], v_node['z']])
            dist_uv = np.linalg.norm(u_pos - v_pos)

            new_dist = distances[u] + dist_uv

            # 松弛操作：如果发现了更短的路径 (或者 v 还没被访问过)
            if new_dist < distances[v]:
                distances[v] = new_dist
                queue.append(v)

    return distances


def compute_branch_angles(graph, root_id, path_distances=None):
    """
    计算分支夹角。

    针对图结构（存在环/多父节点）的解决方案：
    利用 path_distances 构建逻辑上的 Shortest Path Tree (SPT)。
    对于任意节点 u，其唯一的逻辑父节点 p 定义为：
    p = argmin(dist(v)) for v in neighbors(u) if dist(v) < dist(u)

    Args:
        graph: NetworkX graph
        root_id: Root ID
        path_distances: dict, {node_id: distance_to_root}

    Returns:
        dict: {node_id: angle_in_radians}
    """

    # --- 1. 构建逻辑父子关系 (Predecessor Map) ---
    # key: current_node, value: best_parent_node
    if(path_distances is None):
        path_distances = compute_geodesic_distance(graph, root_id)
    predecessor_map = {root_id: None}

    for node in graph.nodes:
        if node == root_id:
            continue

        # 获取当前节点的距离
        curr_dist = path_distances.get(node, float('inf'))
        if curr_dist == float('inf'):
            predecessor_map[node] = None
            continue

        # 寻找最佳父亲
        best_parent = None
        min_parent_dist = float('inf')

        # 遍历邻居
        for neighbor in graph.neighbors(node):
            neighbor_dist = path_distances.get(neighbor, float('inf'))

            # 关键逻辑：父亲必须比儿子更靠近根
            # 使用 <= 是为了容错，但严格来说应该是 <。
            # 考虑到 float 精度，我们要求至少小一点点，或者是单纯比较大小
            if neighbor_dist < curr_dist:

                # 策略：找距离最小的父亲（即最短路径上的上游）
                if neighbor_dist < min_parent_dist:
                    min_parent_dist = neighbor_dist
                    best_parent = neighbor
                # Tie-breaking: 如果距离相等（菱形回路），选 ID 小的，保证确定性
                elif neighbor_dist == min_parent_dist:
                    if best_parent is None or neighbor < best_parent:
                        best_parent = neighbor

        predecessor_map[node] = best_parent

    # --- 2. 计算夹角 ---
    angles = {}

    def get_pos(nid):
        node = graph.nodes[nid]
        return np.array([node['x'], node['y'], node['z']])

    for node in graph.nodes:
        # 获取逻辑父亲
        p = predecessor_map.get(node)

        # 如果没有父亲（根节点或断连点），角度为 0
        if p is None:
            angles[node] = 0.0
            continue

        # 获取逻辑爷爷
        pp = predecessor_map.get(p)

        # 如果没有爷爷（根的第一层子节点），认为顺滑，角度为 0
        if pp is None:
            angles[node] = 0.0
            continue

        # --- 计算向量与夹角 ---
        # Vec 1 (In): 爷爷 -> 父亲
        vec_in = get_pos(p) - get_pos(pp)
        # Vec 2 (Out): 父亲 -> 自己
        vec_out = get_pos(node) - get_pos(p)

        norm_in = np.linalg.norm(vec_in)
        norm_out = np.linalg.norm(vec_out)

        # 避免重合点导致的 NaN
        if norm_in < 1e-6 or norm_out < 1e-6:
            angles[node] = 0.0
        else:
            dot_val = np.dot(vec_in, vec_out)
            cos_val = dot_val / (norm_in * norm_out)
            cos_val = np.clip(cos_val, -1.0, 1.0)
            angles[node] = np.arccos(cos_val)

    return angles