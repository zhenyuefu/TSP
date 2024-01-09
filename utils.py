import argparse
from collections import defaultdict
import os
from graph_tool import Graph

import numpy as np
from graph_tool.topology import shortest_path


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


def calculate_assignment_cost(d, assignments):
	assignment_cost = np.sum(d[np.arange(len(assignments)), assignments])
	return assignment_cost


def calculate_cost(d, path, assignments, rsp_alpha):
	tsp_cost = calculate_tsp_cost(d, path)
	assignment_cost = calculate_assignment_cost(d, assignments)
	return tsp_cost + rsp_alpha * assignment_cost


def compute_time_and_ratio(G: Graph):
	N = G.num_vertices()

	ratio = 0
	time = 0
	walk = 0
	metro = 0
	for i in range(N):
		for j in range(i + 1, N):
			_, edges = shortest_path(G, i, j, weights=G.ep['weight'], negative_weights=False)
			for x in edges:
				if G.ep.type[x] == "assignment":
					walk += G.ep.weight[x]
				else:
					metro += G.ep.weight[x]
				time += G.ep.weight[x]
		# ratio += walk / metro if metro != 0 else 0
	ratio = walk / metro if metro != 0 else 0
	time /= (N * (N - 1) / 2)

	return ratio, time


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


def dump_result(result, rsp, name):
	directory = "result"
	base_filename = f"{directory}/" + name
	run_num = 0

	# Check if the directory exists, if not, create it
	if not os.path.exists(directory):
		os.makedirs(directory)
	else:
		# Check existing files to determine the next run_num
		existing_files = [f for f in os.listdir(directory) if f.startswith(name) and f.endswith('.txt')]
		if existing_files:
			run_num = len(existing_files)

	filename = f"{base_filename}_{run_num}.txt"

	with open(filename, "w") as file:
		file.write(str(result))
	print(f"Result dumped to {filename}")

	if rsp:
		rsp.savefig(f"{base_filename}_{run_num}.svg")


def arg_parser():
	parser = argparse.ArgumentParser(description="Description of your program")
	parser.add_argument('--option', '-o', help='Description of the option')
	parser.add_argument('-f', '--filename', help='Filename')
	parser.add_argument('-a', '--alpha', type=float, default=10, help='Alpha')
	parser.add_argument('-e', '--epsilon', type=float, default=0.0001, help='Epsilon')
	parser.add_argument('-p', '--prop', type=float, default=0.25, help='Proportion of stations')
	args = parser.parse_args()
	return args


def get_solution(points, x, y):
	# Print tsp path
	print("TSP path:")
	edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
	tsp_path = find_subtours(edges)[0]
	print(f"Optimal tour: {tsp_path}")

	N = len(points)

	assignments = [i for i in range(N)]
	for i in range(N):
		for j in range(N):
			if y[i, j].X > 0.5:
				assignments[i] = j
				break

	return tsp_path, assignments
