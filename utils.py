import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def read_points(filename):
	points = []

	with open(filename, 'r') as file:
		lines = file.readlines()
		start = False
		
		for line in lines:
			if line.startswith('NODE_COORD_SECTION'):
				start = True
				continue
			elif line.startswith('EOF'):
				break
			
			if start:
				parts = line.strip().split()
				points.append((float(parts[1]), float(parts[2])))

	return np.array(points)


def calculate_tsp_cost(d, path):
	path = np.hstack([path, path[0]])
	total_cost = np.sum(d[path[:-1], path[1:]])
	return total_cost


def calculate_cost(d, path, assignments, rsp_alpha):
	tsp_cost = calculate_tsp_cost(d, path)
	assignment_cost = np.sum(d[np.arange(len(assignments)), assignments])
	return tsp_cost + rsp_alpha * assignment_cost

def find_subtours(edges):
	"""Given a list of edges, return the shortest subtour (as a list of nodes)
	found by following those edges. It is assumed there is exactly one 'in'
	edge and one 'out' edge for every node represented in the edge list."""

	# Create a mapping from each node to its neighbours
	node_neighbors = defaultdict(list)
	for i, j in edges:
		node_neighbors[i].append(j)
	assert all(len(neighbors) == 2 for neighbors in node_neighbors.values())

	# Follow edges to find cycles. Each time a new cycle is found, keep track
	# of the shortest cycle found so far and restart from an unvisited node.
	unvisited = set(node_neighbors)
	cycles = []
	while unvisited:
		cycle = []
		neighbors = list(unvisited)
		while neighbors:
			current = neighbors.pop()
			cycle.append(current)
			unvisited.remove(current)
			neighbors = [j for j in node_neighbors[current] if j in unvisited]
		cycles.append(cycle)

	return cycles

def dump_solution(x):
    # Print tsp path
	print("TSP path:")
	edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
	tour = find_subtours(edges)
	print(f"Optimal tour: {tour}")

def draw_solution(points, x, y):
	edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
	N = len(points)

	
	G = nx.Graph()
	for i in range(N):
		G.add_node(i, pos = points[i])

	# add assignments edges to graph
	for i in range(N):
		for j in range(N):
			if y[i, j].X > 0.5 and i != j:
				G.add_edge(i, j, type = 'assignment')

	# add tsp edges to graph
	for edge in edges:
		G.add_edge(edge[0], edge[1], type = 'tsp')

	# draw graph
	pos = nx.get_node_attributes(G, 'pos')
	nx.draw_networkx_nodes(G, pos, node_size=5)

	# draw assignments
	nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'assignment'], width=1, alpha=0.5, edge_color='black')

	# draw tsp path
	nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'tsp'], width=1, alpha=1, edge_color='red')

	plt.show()