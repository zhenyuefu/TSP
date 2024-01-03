from math import ceil, sqrt
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from python_tsp.heuristics import solve_tsp_simulated_annealing

from NuagePoints import NuagePoints
import copy

def _solve_tsp(centers):
	# Create a distance matrix for the centers
	distances = distance_matrix(centers, centers)
	permutation, distance = solve_tsp_simulated_annealing(distances)
	return permutation, distance

def _solve_p_center_problem(points, p):
	p = p
	q = ceil(sqrt(p))
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

	# Assign non-center points to the nearest center
	assignments = np.argmin(distance_matrix(points, centers), axis=1)

	return centers, assignments

def _evaluate(points, centers, assignments, tsp_distance, alpha):
	# Compute the sum of the distances between the points and their assigned center
	assignment_distance = np.sum(np.linalg.norm(points - centers[assignments], axis=1))
	
	# Apply the weight to the tsp distance
	weighted_tsp_distance = alpha * tsp_distance
	
	# Total evaluation
	total_distance = weighted_tsp_distance + assignment_distance
	
	return total_distance


class RingStarProblem:
	def __init__(self, nuage_points:NuagePoints, alpha=5):
		self.nuage_points = nuage_points
		self.p = 0
		self.alpha = alpha
		self.centers = None
		self.assignments = None
		self.tsp_path = None
		self.tsp_distance = 0

	def get_solution(self):
		return self.centers, self.assignments, self.tsp_path

	def _solve_p(self, p):
		self.centers, self.assignments = _solve_p_center_problem(self.nuage_points.points, p)
		self.tsp_path, self.tsp_distance = _solve_tsp(self.centers)
		return _evaluate(self.nuage_points.points, self.centers, self.assignments, self.tsp_distance, self.alpha)
	
	# def solve(self, p_min, p_max):
	# 	p_values = np.arange(p_min, p_max + 1)
	# 	results = np.array([self._solve_p(p) for p in p_values])
	# 	best_p = p_values[np.argmin(results)]
	# 	self._solve_p_center_problem(best_p)
	# 	self._solve_tsp(self.centers)
	# 	return best_p

	def draw(self):
		if self.centers is not None and self.assignments is not None:
			plt.scatter(self.nuage_points.points[:,0], self.nuage_points.points[:,1], c=self.assignments)
			plt.scatter(self.centers[:,0], self.centers[:,1], c='red')
			# plot assignment lines
			for i in range(self.nuage_points.num_cities):
				plt.plot([self.nuage_points.points[i,0], self.centers[self.assignments[i],0]], [self.nuage_points.points[i,1], self.centers[self.assignments[i],1]], c='black', alpha=0.1)
			# write p
			plt.annotate('p = {}'.format(self.p), xy=(0.05, 0.85), xycoords='axes fraction')
			# plot the tsp path
			if self.tsp_path is not None:
				# Add the first point to the end of the path
				path = np.hstack([self.tsp_path, self.tsp_path[0]])
				plt.plot(self.centers[path,0], self.centers[path,1], c='black')
				# write the total distance
				tot_dist = _evaluate(self.nuage_points.points, self.centers, self.assignments, self.tsp_distance, self.alpha)
				plt.annotate('Total distance: {:.2f}'.format(tot_dist), xy=(0.05, 0.95), xycoords='axes fraction')
			plt.show()
		else:
			raise Exception('The problem has not been solved yet')
		
	def metha_heuristic(self, iteration, p):
		c = self._solve_p(p)

		for _ in range(iteration):
			# Choose a random center and a random point
			swap_index = np.random.choice(self.centers)

			for i in range(self.nuage_points.num_cities):
				if i not in self.centers:
					new_centers = copy.deepcopy(self.centers)
					new_centers[swap_index] = i
					assignments = np.argmin(distance_matrix(self.nuage_points.points, new_centers), axis=1)
					_solve_tsp(new_centers)


if __name__ == '__main__':
	nuage_points = NuagePoints('./Instances_TSP/att48.tsp')
	rsp = RingStarProblem(nuage_points)
	rsp.solve(3, 20)
	rsp.draw()