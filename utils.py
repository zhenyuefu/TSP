import numpy as np


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
