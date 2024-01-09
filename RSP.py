from dataclasses import dataclass, field
from typing import List

# import networkx as nx
from graph_tool import Graph
from graph_tool.draw import graph_draw
import numpy as np

from scipy.spatial import distance_matrix

from utils import calculate_assignment_cost, calculate_tsp_cost, compute_time_and_ratio


@dataclass
class RSP:
	points: np.ndarray  # 点集，每个点由纬度和经度组成，存储在 NumPy 数组中
	alpha: float        # assignment cost 的权重
	p: int              # 选择作为车站的点的数量
	distance_matrix: np.ndarray = field(init=False)  # 距离矩阵，初始时不在构造函数中设置
	assignments: List[int] = field(default_factory=list)  # 非车站点到车站点的分配
	tsp_path: List[int] = field(default_factory=list)  # 旅行商问题路径
	cost: float = field(init=False)  # objective function value
	time: float = field(init=False)  # total time
	tsp_cost: float = field(init=False)  # tsp cost
	ratio: float = field(init=False)  # time ratio of assignment to tsp
	G: Graph | None = None  # graph

	def __post_init__(self):
		self.distance_matrix = distance_matrix(self.points, self.points)

	# def create_graph(self):
	# 	G = nx.Graph()
	# 	# 添加节点
	# 	for i in range(len(self.points)):
	# 		G.add_node(i, pos=(self.points[i][0], self.points[i][1]))

	# 	# 使用 TSP 路径添加车站之间的边
	# 	for i in range(len(self.tsp_path)):
	# 		if i < len(self.tsp_path) - 1:
	# 			G.add_edge(self.tsp_path[i], self.tsp_path[i+1], type= 'tsp')
	# 		# 将路径的最后一个点连接回第一个点，形成闭环
	# 		else:
	# 			G.add_edge(self.tsp_path[i], self.tsp_path[0], type='tsp')

	# 	# 添加非车站点到车站点的边
	# 	for i, center in enumerate(self.assignments):
	# 		if i not in self.tsp_path:
	# 			G.add_edge(i, center, type='assignment')

	# 	return G
	def create_graph(self):
	
		G = Graph(directed=False)
		pos = G.new_vertex_property('vector<double>')
		edge_type = G.new_edge_property('string')  
		edge_color = G.new_edge_property('vector<double>')  
		edge_weight = G.new_edge_property('double')  
		v_color = G.new_vertex_property('vector<double>')  
		G.vp["pos"] = pos
		G.vp["color"] = v_color
		G.ep["type"] = edge_type
		G.ep["color"] = edge_color
		G.ep["weight"] = edge_weight
		# 添加节点
		for point in self.points:
			v = G.add_vertex()
			pos[v] = point
			v_color[v] = [0, 0, 0, 1]

		# 添加边
		for i in range(len(self.tsp_path)):
			v_color[G.vertex(self.tsp_path[i])] = [1, 0, 0, 1]
			e = G.add_edge(G.vertex(self.tsp_path[i]), G.vertex(self.tsp_path[(i+1) % len(self.tsp_path)]))
			edge_type[e] = 'tsp'
			edge_color[e] = [1, 0, 0, 1]
			edge_weight[e] = self.distance_matrix[self.tsp_path[i], self.tsp_path[(i+1) % len(self.tsp_path)]]

		for i, center in enumerate(self.assignments):
			if i not in self.tsp_path:
				e = G.add_edge(G.vertex(i), G.vertex(center))
				edge_type[e] = 'assignment'
				edge_color[e] = [0, 0, 0, 1]
				edge_weight[e] = self.distance_matrix[i, center] * self.alpha

		self.G = G

		return G

	def savefig(self, filename):
		if self.G is None:
			self.create_graph()
		# 设置绘图选项
		graph_draw(self.G, pos=self.G.vp["pos"], vertex_size = 4 , vertex_color = self.G.vp['color'] ,vertex_fill_color = self.G.vp['color'],edge_color=self.G.ep["color"], bg_color='white',output=filename)
		


	# def savefig(self, filename, runtime = None):
	# 	G = self.create_graph()
	# 	pos = nx.get_node_attributes(G, 'pos')
	# 	nx.draw_networkx_nodes(G, pos, node_size=5)
	# 	# draw centers
	# 	nx.draw_networkx_nodes(G, pos, nodelist=self.tsp_path, node_size=10, node_color='r')
	# 	# draw assignments
	# 	nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'assignment'], width=1, alpha=0.5, edge_color='black')
	# 	# draw tsp path
	# 	nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'tsp'], width=1, alpha=1, edge_color='red')
	# 	# write cost
	# 	plt.text(0, 0, f"Cost: {self.cost}", fontsize=12)
	# 	if runtime:
	# 		plt.text(0.6, 0, f"Runtime: {runtime}", fontsize=12)
	# 	plt.savefig(filename)

	def evaluate(self):
		if self.G is None:
			self.G = self.create_graph()
		
		self.tsp_cost = calculate_tsp_cost(self.distance_matrix, self.tsp_path)
		assignment_cost = calculate_assignment_cost(self.distance_matrix, self.assignments)
		self.cost = self.tsp_cost + self.alpha * assignment_cost
		self.ratio, self.time = compute_time_and_ratio(self.G)