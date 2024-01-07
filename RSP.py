from dataclasses import dataclass, field
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix

from utils import calculate_cost


@dataclass
class RSP:
	points: np.ndarray  # 点集，每个点由纬度和经度组成，存储在 NumPy 数组中
	alpha: float        # assignment cost 的权重
	p: int              # 选择作为车站的点的数量
	distance_matrix: np.ndarray = field(init=False)  # 距离矩阵，初始时不在构造函数中设置
	assignments: List[int] = field(default_factory=list)  # 非车站点到车站点的分配
	tsp_path: List[int] = field(default_factory=list)  # 旅行商问题路径
	cost: float = field(init=False)  # RSP 的总成本

	def __post_init__(self):
		self.distance_matrix = distance_matrix(self.points, self.points)

	def create_graph(self):
		G = nx.Graph()
		# 添加节点
		for i in range(len(self.points)):
			G.add_node(i, pos=(self.points[i][0], self.points[i][1]))

		# 使用 TSP 路径添加车站之间的边
		for i in range(len(self.tsp_path)):
			if i < len(self.tsp_path) - 1:
				G.add_edge(self.tsp_path[i], self.tsp_path[i+1], type= 'tsp')
			# 将路径的最后一个点连接回第一个点，形成闭环
			else:
				G.add_edge(self.tsp_path[i], self.tsp_path[0], type='tsp')

		# 添加非车站点到车站点的边
		for i, center in enumerate(self.assignments):
			if i not in self.tsp_path:
				G.add_edge(i, center, type='assignment')

		return G

	def savefig(self, filename):
		G = self.create_graph()
		pos = nx.get_node_attributes(G, 'pos')
		nx.draw_networkx_nodes(G, pos, node_size=5)
		# draw centers
		nx.draw_networkx_nodes(G, pos, nodelist=self.tsp_path, node_size=10, node_color='r')
		# draw assignments
		nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'assignment'], width=1, alpha=0.5, edge_color='black')
		# draw tsp path
		nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'tsp'], width=1, alpha=1, edge_color='red')
		# write cost
		plt.text(0, 0, f"Cost: {self.cost}", fontsize=12)
		plt.savefig(filename)

	def evaluate(self):
		self.cost = calculate_cost(self.distance_matrix, self.tsp_path, self.assignments, self.alpha)