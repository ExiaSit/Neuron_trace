import numpy as np


def translate_tree(G, shift):
    """对干扰树整体平移"""
    for node in G.nodes:
        G.nodes[node]['x'] += shift[0]
        G.nodes[node]['y'] += shift[1]
        G.nodes[node]['z'] += shift[2]


def apply_random_rotation(graph, center_node_id, rot_matrix):
    """
    将图中的所有节点绕着指定的中心节点 (center_node_id) 进行随机旋转。
    """
    # 1. 获取旋转中心坐标 (即节点 a 的坐标)
    center_node = graph.nodes[center_node_id]
    center_pos = np.array([center_node['x'], center_node['y'], center_node['z']])

    # 3. 遍历所有节点进行坐标变换
    for n in graph.nodes:
        node = graph.nodes[n]
        # 提取坐标
        pos = np.array([node['x'], node['y'], node['z']])

        # 核心变换公式： v' = R * (v - center) + center
        # 先移回原点 -> 旋转 -> 移回中心
        new_pos = np.dot(rot_matrix, (pos - center_pos)) + center_pos

        # 更新回图节点
        node['x'], node['y'], node['z'] = new_pos