from math import ceil, sqrt
import time

import elkai
import numpy as np
from scipy.spatial import distance_matrix

from RSP import RSP
from utils import dump_result


class RSPHeuristics:
	def __init__(self, rsp_instance: RSP):
		self.rsp = rsp_instance

	def solve(self):
		start = time.time()
		centers_indices, assignments = self._solve_p_center_problem()
		tsp_path = self._solve_tsp(centers_indices)
		end = time.time()
		rsp.assignments = assignments
		rsp.tsp_path = tsp_path
		print(f"assignments: {assignments}")
		print(f"tsp_path: {tsp_path}")
		rsp.evaluate()
		return end - start

	def _solve_tsp(self, centers_indices):
		# Create a distance matrix for the centers
		distances = self.rsp.distance_matrix[centers_indices][:, centers_indices]
		c = elkai.DistanceMatrix(distances.tolist())
		path = c.solve_tsp()
		path = path[:-1]
		solution = [centers_indices[i] for i in path]

		return solution

	def _solve_p_center_problem(self):
		p = self.rsp.p
		q = ceil(sqrt(p))
		points = self.rsp.points
		x_min, y_min = np.min(points, axis=0)
		x_max, y_max = np.max(points, axis=0)

		# Divide the area into q x q rectangles
		x_step = (x_max - x_min) / q
		y_step = (y_max - y_min) / q
		x_coords, y_coords = np.meshgrid(np.arange(x_min, x_max, x_step), np.arange(y_min, y_max, y_step))
		rect_centers = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

		# Find the closest point to each rectangle center
		distances = distance_matrix(rect_centers, points)
		closest_indices = np.argmin(distances, axis=1)
		closest_indices = np.unique(closest_indices)

		# Ensure we have exactly p centers
		if len(closest_indices) > p:
			centers = points[np.random.choice(closest_indices, p, replace=False)]
		elif len(closest_indices) < p:
			additional_indices = np.random.choice(points.shape[0], p - len(closest_indices), replace=False)
			centers = np.vstack([points[closest_indices], points[additional_indices]])
		else:
			centers = points[closest_indices]

		# Find the center indices in the points array
		centers_indices = [np.where((points == center).all(axis=1))[0][0] for center in centers]

		# Assign non-center points to the nearest center
		assignments_indices = np.argmin(distance_matrix(points, centers), axis=1)

		assignments = [centers_indices[i] for i in assignments_indices]

		return centers_indices, assignments


if __name__ == '__main__':
	from utils import read_points, arg_parser
	args = arg_parser()
	alpha = args.alpha
	data = read_points(args.filename)
	p = int(len(data) * args.prop)
	instance_name = args.filename.split('/')[-1].split('.')[0]

	rsp = RSP(points=data, alpha=alpha, p=p)
	solver = RSPHeuristics(rsp)
	t = solver.solve()
	results = {
		"Objective Value": rsp.cost,
		"Cost": rsp.tsp_cost,
		"MeanTime": rsp.time,
		"Ratio": rsp.ratio,
		"Runtime": t,
		"path": rsp.tsp_path,
		"assignments": rsp.assignments,
	}
	dump_result(results, rsp, f"heuristics_{instance_name}_p{p}_alpha{alpha}")
