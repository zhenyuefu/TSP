import random
from dataclasses import dataclass, field
import time
from typing import List

import numpy as np
from scipy.spatial import distance_matrix
import elkai

from RSP import RSP
from christofides_tsp import christofides_tsp
from three_opt import tsp_3_opt
from utils import dump_result, read_points, calculate_tsp_cost, calculate_cost, arg_parser


@dataclass
class Ant:
	path: List[int]
	cost: float = float('inf')
	assignments: List[int] = field(default_factory=list)


class HybridACS:
	def __init__(self, rsp_instance: RSP, num_ants: int = 50, Q: int = 100, alpha: float = 1, beta: float = 3,
	             rho: float = 0.9, theta: float = 0.1, xi: float = 0.5, sigma: float = 0.1, q0: float = 1,
	             max_iterations: int = 200, no_improve_iterations: int = 2):
		self.rsp = rsp_instance
		self.num_ants = num_ants
		self.max_iterations = max_iterations
		self.no_improve_iterations = no_improve_iterations
		self.alpha = alpha
		self.beta = beta
		self.rho = rho
		self.theta = theta
		self.xi = xi
		self.sigma = sigma
		self.q0 = q0
		self.pheromone = None
		self.assignment_pheromone = None
		self.tau_0 = 1
		self.Q = Q
		self.best_solution = None
		self.iterations_since_improvement = 0

	def solve(self):
		init_path = christofides_tsp(self.rsp.distance_matrix)
		self.best_solution = Ant(path=init_path, cost=np.inf)
		self.initialize_pheromone(self.rsp.distance_matrix, init_path)

		iteration = 0
		while not self.termination_condition(iteration):
			ants = []
			for _ in range(self.num_ants):
				Sm = self.ant_construction()
				self.assign_non_visited_nodes(Sm)
				self.local_pheromone_update(Sm)
				ants.append(Sm)

			for Sm in ants:
				self.vnd(Sm)

			Slocal = min(ants, key=lambda x: x.cost)
			print(f"Local best cost: {Slocal.cost}")

			if Slocal.cost < self.best_solution.cost:
				# update the best solution
				self.best_solution = Slocal
				self._lin_kernighan(self.best_solution)
				self.iterations_since_improvement = 0
			else:
				self.iterations_since_improvement += 1

			self.global_pheromone_update()

			iteration += 1
			print(f"Iteration {iteration} best cost: {self.best_solution.cost}")

		return self.best_solution

	def termination_condition(self, iteration):
		# Check if the maximum number of iterations has been reached
		if iteration > self.max_iterations:
			return True

		# Check if there has been no improvement for a given number of iterations
		if self.iterations_since_improvement >= self.no_improve_iterations:
			return True

		return False

	def initialize_pheromone(self, d, init_solution):
		n = d.shape[0]
		total_cost = calculate_tsp_cost(d, init_solution)
		tau_0 = 1 / (n * total_cost) if total_cost != 0 else 1
		self.pheromone = np.full(d.shape, tau_0)
		self.assignment_pheromone = np.full(n, tau_0)
		self.tau_0 = tau_0

	def ant_construction(self):
		# n = len(self.rsp.points)
		# ant = Ant(path=[random.choice(range(n))])  # Assuming depot node is 0
		ant = Ant(path=[0])
		tabu_list = set(ant.path)

		while len(ant.path) < self.rsp.p:
			last_node = ant.path[-1]
			candidate_nodes = [node for node in range(len(self.rsp.points)) if node not in tabu_list]

			if not candidate_nodes:
				break  # No more nodes to visit, cycle construction is complete

			attractiveness = self.calculate_attractiveness(last_node, candidate_nodes)
			next_node = self.select_next_node(candidate_nodes, attractiveness)

			ant.path.append(next_node)
			tabu_list.add(next_node)

		return ant

	def calculate_attractiveness(self, last_node, candidate_nodes):
		attractiveness = []
		for j in candidate_nodes:
			tau_prime_j = self.assignment_pheromone[j]
			tau_ij = self.pheromone[last_node, j]
			eta_ij = 1 / self.rsp.distance_matrix[last_node, j]  # Heuristic information
			a_j = (tau_prime_j * tau_ij) ** self.alpha * eta_ij ** self.beta
			attractiveness.append(a_j)
		return attractiveness

	def select_next_node(self, candidate_nodes, attractiveness):
		if random.random() < self.q0:
			# Choose the node with the highest attractiveness
			max_index = np.argmax(attractiveness)
			return candidate_nodes[max_index]
		else:
			# Probabilistic selection of the next node
			probabilities = np.array(attractiveness) / sum(attractiveness)
			return np.random.choice(candidate_nodes, p=probabilities)

	def assign_non_visited_nodes(self, ant: Ant):
		centers = self.rsp.points[ant.path]
		assignments_indices = np.argmin(distance_matrix(self.rsp.points, centers), axis=1)
		assignments = [ant.path[i] for i in assignments_indices]
		ant.assignments = assignments

	def local_pheromone_update(self, ant: Ant):
		for i in range(len(ant.path) - 1):
			j = ant.path[i]
			k = ant.path[i + 1]

			# Update routing pheromone
			self.pheromone[j][k] = (1 - self.xi) * self.pheromone[j][k] + self.xi * self.tau_0
			self.pheromone[k][j] = self.pheromone[j][k]  # assuming symmetry

			# Update assignment pheromone
			for l in ant.assignments:
				self.assignment_pheromone[l] = (1 - self.sigma) * self.assignment_pheromone[l] + self.sigma * self.tau_0

	def global_pheromone_update(self):
		best_path = self.best_solution.path
		best_cost = self.best_solution.cost
		Q = self.Q

		for i in range(len(best_path) - 1):
			j = best_path[i]
			k = best_path[i + 1]

			# Update routing pheromone
			self.pheromone[j][k] = (1 - self.rho) * self.pheromone[j][k] + self.rho * Q / best_cost
			self.pheromone[k][j] = self.pheromone[j][k]  # assuming symmetry

		# Update assignment pheromone
		for l in range(len(self.assignment_pheromone)):
			self.assignment_pheromone[l] = (1 - self.theta) * self.assignment_pheromone[l] + self.theta * Q / best_cost

	def _lin_kernighan(self, ant: Ant):
		original_path = ant.path
		new_d = self.rsp.distance_matrix[original_path, :][:, original_path]
		c = elkai.DistanceMatrix(new_d.tolist())
		path = c.solve_tsp()
		path = path[:-1]
		path_in_original = [original_path[i] for i in path]
		ant.path = path_in_original
		ant.cost = calculate_cost(self.rsp.distance_matrix, path_in_original, ant.assignments, self.rsp.alpha)

	def _3_opt(self, ant):
		orginal_path = ant.path
		new_d = self.rsp.distance_matrix[orginal_path, :][:, orginal_path]
		path = tsp_3_opt(new_d, list(range(len(orginal_path))))
		path_in_original = [orginal_path[i] for i in path]
		ant.path = path_in_original

	def vnd(self, Si: Ant):
		k = 1
		k_max = 2
		while k <= k_max:
			improved = False
			if k == 1:
				self._3_opt(Si)
			# print("3-opt done")
			elif k == 2:
				improved = self.exchange(Si)
			# print("exchange done")

			if improved:
				k = 1
			else:
				k += 1

		return Si

	def exchange(self, ant: Ant):
		improved = False
		current_cost = calculate_cost(self.rsp.distance_matrix, ant.path, ant.assignments, self.rsp.alpha)
		ant.cost = current_cost

		# i in a random order
		for i in random.sample(range(1, len(ant.path)), len(ant.path) - 1):  # Exclude depot from swapping
			# for i in range(1, len(ant.path)):  # Exclude depot from swapping
			for j in range(len(self.rsp.points)):
				if j not in ant.path:
					new_path = ant.path[:]
					new_path[i] = j  # Swap node

					centers = self.rsp.points[new_path]
					assignments_indices = np.argmin(distance_matrix(self.rsp.points, centers), axis=1)
					assignments = [new_path[i] for i in assignments_indices]
					new_total_cost = calculate_cost(self.rsp.distance_matrix, new_path, assignments, self.rsp.alpha)

					if new_total_cost < current_cost:
						ant.path = new_path
						current_cost = new_total_cost
						ant.cost = new_total_cost
						ant.assignments = assignments
						improved = True
						return improved

		return improved


if __name__ == '__main__':
	args = arg_parser()
	alpha = args.alpha
	points = read_points(args.filename)
	p = int(len(points) * args.prop)
	instance_name = args.filename.split('/')[-1].split('.')[0]
	rsp = RSP(points=points, alpha=alpha, p=p)
	solver = HybridACS(rsp, num_ants=4)
	start_time = time.time()
	solution = solver.solve()
	end_time = time.time()
	print(f"Runtime: {end_time - start_time}")
	rsp.tsp_path = solution.path
	rsp.assignments = solution.assignments
	rsp.cost = solution.cost
	print(solution.cost)
	print(solution.path)
	print(solution.assignments)
	rsp.evaluate()
	results = {
		"Objective Value": solution.cost,
		"Cost": rsp.tsp_cost,
		"MeanTime": rsp.time,
		"Ratio": rsp.ratio,
		"Runtime": end_time - start_time,
	}
	dump_result(results, rsp, f"hacs_{instance_name}_p{p}_alpha{alpha}")