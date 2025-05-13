import itertools
import math
import traceback
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib
import networkx as nx
from networkx import isomorphism
from pandas import array
from pypinyin import lazy_pinyin
from networkx.algorithms.isomorphism import GraphMatcher


def build_graph_with_positions(layout, positions):
    """
    根据户型图的分间布局和位置信息构建图。
    """
    graph = nx.Graph()

    # 如果layout为空，确保至少添加一个节点
    if not layout and positions:
        for node in positions.keys():
            graph.add_node(node)

    for edge in layout:
        graph.add_edge(edge[0], edge[1])
    nx.set_node_attributes(graph, positions, 'pos')
    nx.set_node_attributes(graph, {node: node for node in graph.nodes()}, 'id')
    return graph


def maximum_common_subgraph(graph1, graph2, threshold=0.2):
    """
    寻找两个图的最大公共子图。
    """

    def node_match(n1, n2):
        # 打印节点位置和距离
        pos1 = n1['pos']
        pos2 = n2['pos']
        id1 = n1['id'].split('_')[0]
        id2 = n2['id'].split('_')[0]
        distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        # print(f"Comparing nodes with positions {id1}:{pos1} and {id2}:{pos2}, distance: {distance}")
        return distance < threshold and id1 == id2

    subgraph_nodes = []
    subgraph_edges = []
    subgraph_nodes_pos = {}
    subgraph_nodes_id = {}
    for node in graph1.nodes():
        for node2 in graph2.nodes():
            if node_match(graph1.nodes[node], graph2.nodes[node2]):
                subgraph_nodes.append(node)
                subgraph_nodes_pos[node] = graph1.nodes[node]['pos']
                subgraph_nodes_id[node] = graph1.nodes[node]['id']
                # print(f"Found isomorphic node: {node} and {node2}")
    for edge in graph1.edges():
        for edge2 in graph2.edges():
            if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes and edge2[0] in subgraph_nodes and edge2[
                1] in subgraph_nodes:
                subgraph_edges.append(edge)
                # print(f"Found isomorphic edge: {edge} and {edge2}")
    subgraph_edges = list(set(subgraph_edges))
    subgraph = nx.Graph()
    if not subgraph_nodes and not subgraph_edges:
        return subgraph
    subgraph.add_nodes_from(subgraph_nodes)
    nx.set_node_attributes(subgraph, subgraph_nodes_pos, 'pos')
    nx.set_node_attributes(subgraph, subgraph_nodes_id, 'id')
    subgraph.add_edges_from(subgraph_edges)
    return subgraph

    # 寻找同构子图
    # graph2_nodes = list(graph2.nodes)
    # found_isomorphic_subgraph = False

    # for sub_nodes in itertools.combinations(graph2_nodes, len(graph1.nodes)):
    #     subgraph = graph2.subgraph(sub_nodes)
    #     gm = GraphMatcher(subgraph, graph1, node_match=node_match)
    #     if gm.is_isomorphic():
    #         print(f"Found isomorphic subgraph: {subgraph.nodes}")
    #         found_isomorphic_subgraph = True
    #         return subgraph
    #         # plot_graph(subgraph, 'Isomorphic Subgraph')
    #         break

    # if not found_isomorphic_subgraph:
    #     print("No isomorphic subgraph found")
    #     return nx.Graph()

    # # 使用NetworkX的最大公共子图算法
    # gm = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2, node_match=node_match)
    # max_common_subgraphs = []

    # for subgraph in gm.subgraph_isomorphisms_iter():
    #     print(f"Found subgraph: {subgraph}")
    #     common_subgraph = graph1.subgraph(subgraph.keys())
    #     max_common_subgraphs.append(common_subgraph)

    # return max_common_subgraphs
    # return isomorphism.GraphMatcher(graph1, graph2, node_match=node_match).subgraph_isomorphisms_iter()


def translate_graph(graph1, graph2):
    # 获取两个图的公共节点
    common_node = list()
    for node in graph1.nodes():
        if node in graph2.nodes():
            common_node.append(node)

    if not common_node:
        return None

    # 找最近的节点
    common_node_distance = {}
    min_distance = 1000000
    for node in common_node:
        pos1 = graph1.nodes[node]['pos']
        pos2 = graph2.nodes[node]['pos']
        distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        common_node_distance[node] = distance
        if distance < min_distance:
            min_distance = distance

    common_node_pos = []
    for node, distance in common_node_distance.items():
        if distance == min_distance:
            common_node_pos.append(graph1.nodes[node]['pos'])
            common_node_pos.append(graph2.nodes[node]['pos'])
            break

    # graph2的副本
    graph2_copy = graph2.copy()

    # 计算平移距离
    x1, y1 = common_node_pos[0]
    x2, y2 = common_node_pos[1]
    dx = x1 - x2
    dy = y1 - y2

    # 平移graph2
    for node in graph2_copy.nodes():
        if node not in common_node:
            continue
        pos = graph2_copy.nodes[node]['pos']
        graph2_copy.nodes[node]['pos'] = (pos[0] + dx, pos[1] + dy)

    return graph1, graph2_copy


def draw_graph(graph1, graph2, subgraph, title):
    fig, ax = plt.subplots()
    pos = {}
    # 绘制第一个图，使用蓝色
    for key, node in graph1.nodes().items():
        pos[key] = node['pos']

    nx.draw(graph1, pos, labels={node: ''.join(lazy_pinyin(node)) for node in graph1.nodes()},
            with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, ax=ax)

    # 绘制第二个图，使用绿色
    pos = {}
    # # 绘制第一个图，使用蓝色
    for key, node in graph2.nodes().items():
        pos[key] = node['pos']

    nx.draw(graph2, pos, labels={node: ''.join(lazy_pinyin(node)) for node in graph2.nodes()},
            with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500, font_size=10, ax=ax)

    # 绘制子图，使用红色
    if subgraph.number_of_nodes() > 0:
        nx.draw(subgraph, pos, labels={node: ''.join(lazy_pinyin(node)) for node in subgraph.nodes()},
                with_labels=True, node_color='lightcoral', edge_color='gray', node_size=500, font_size=10, ax=ax)

    plt.savefig(f"{title}.png")


def cal_isomorphism_metric(graph1, graph2, graph1_num, graph2_num, res_path=None, json_id=None):
    """
    计算两个图的同构性分数。
    """

    if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
        return 0, 0, nx.Graph()

    # 平移图2，使得两个图的公共节点位置最接近
    graph1, graph2 = translate_graph(graph1, graph2)

    # 寻找最大公共子图
    try:
        max_common_subgraph = maximum_common_subgraph(graph1, graph2)
        # draw_graph(graph1, graph2, max_common_subgraph, f"{res_path}/png/{json_id}_layout")
        # if max_common_subgraph:
        #     # max_common_subgraph = max(max_common_subgraphs, key=lambda g: g.number_of_nodes())
        #     # 绘制最大公共子图
        #     if res_path:
        #         draw_graph(graph1, graph2, max_common_subgraph, f"{res_path}/png/{json_id}_layout")
        # else:
        #     max_common_subgraph = nx.Graph()  # 如果没有公共子图，返回一个空图
        #     if res_path:
        #         draw_graph(graph1, graph2, max_common_subgraph, f"{res_path}/png/{json_id}_layout")
        # max_common_subgraphs = maximum_common_subgraph(graph1, graph2)
        # max_common_subgraph = max(max_common_subgraphs, key=len, default=[])

    except Exception as e:
        traceback.print_exc()
        print(f"Error during finding maximum common subgraph: {e}")
        return 0, 0, nx.Graph()
    # 计算分数
    # num_common_nodes = len(max_common_subgraph)
    num_common_nodes = max_common_subgraph.number_of_nodes()
    total_nodes = max(graph1_num, graph2_num)
    # total_nodes = max(graph1.number_of_nodes(), graph2.number_of_nodes())
    # 准召
    # precision = num_common_nodes / graph1.number_of_nodes()
    # recall = num_common_nodes / graph2.number_of_nodes()
    precision = round(num_common_nodes / graph1_num, 4)
    recall = round(num_common_nodes / graph2_num, 4)

    if total_nodes == 0:
        return 0, 0, maximum_common_subgraph

    return precision, recall, max_common_subgraph

