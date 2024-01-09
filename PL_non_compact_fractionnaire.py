from itertools import combinations
import time

import gurobipy as gp
from gurobipy import GRB
from igraph import Graph

from utils import arg_parser, find_subtours, get_solution, dump_result, read_points
from RSP import RSP
import numpy as np


class RingStarCallback:
	def __init__(self, points, x, y, p, V, epsilon):
		self.points = points
		self.n = len(points)
		self.x = x
		self.y = y
		self.p = p
		self.V = V
		self.epsilon = epsilon

	def __call__(self, model, where):
		# lazy separation
		if where == GRB.Callback.MIPSOL:
			x_sol = model.cbGetSolution(self.x)

			edges = [(i, j) for (i, j), v in x_sol.items() if v > 0.5]
			cycles = find_subtours(edges)
			if len(cycles) > 1:
				for S in cycles:
					nS = [i for i in self.V if i not in S]
					if 1 not in S:
						for i in S:
							model.cbLazy(gp.quicksum(self.x[x, y] for x in S for y in nS)
							             >= 2 * self.y[i, i])

		# heuristic separation
		elif where == GRB.Callback.MIPNODE:
			if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
				xrel = model.cbGetNodeRel(self.x)
				yrel = model.cbGetNodeRel(self.y)

				for i in range(1, self.n):
					if yrel[i, i] > 0:
						S = [i]
						nS = [j for j in self.V if j != i]
						for j in range(1, self.n):
							if j != i:
								S.append(j)
								nS.remove(j)
							if np.sum([xrel[x, y] for x in S for y in nS]) < 2 * yrel[i, i]:
								for k in S:
									model.cbCut(gp.quicksum(self.x[x, y] for x in S for y in nS)
									            >= 2 * self.y[k, k])
								break

	# exact separation
	# elif where == GRB.Callback.MIPNODE:
	# 	if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
	# 		xrel = model.cbGetNodeRel(self.x)
	# 		yrel = model.cbGetNodeRel(self.y)
	# 		edges = [(i, j) for (i, j), v in xrel.items() if v > 0]

	# 		# 使用 igraph 构建一个无向图
	# 		G = Graph()
	# 		G.add_vertices(self.V)  # 添加顶点
	# 		G.add_edges(edges)  # 添加边
	# 		G.es['capacity'] = [xrel[edge] for edge in edges]  # 添加边的权重

	# 		for i in range(1, self.n):
	# 			# 使用 igraph 计算最小割
	# 			mc = G.mincut(0, i, capacity='capacity')
	# 			min_cut = mc.value
	# 			S, nS = mc.partition

	# 			if min_cut < 2 * yrel[i, i] - self.epsilon:
	# 				for k in S:
	# 					model.cbCut(gp.quicksum(self.x[x, y] for x in S for y in nS if (x, y) in self.x)
	# 								>= 2 * self.y[k, k])


def pl_non_compact_fractionnaire(args):
	# Get the distance matrix
	points = read_points(args.filename)
	n = len(points)  # Number of nodes
	p = int(n * args.prop)  # Number of stations
	rsp = RSP(points, args.alpha, p)
	d = rsp.distance_matrix

	start_time = time.time()

	# Create a Gurobi model
	m = gp.Model("RSP")

	# Create variables
	x = {}  # Edge of the TSP path
	y = {}  # Assignment of nodes to stations

	V = range(n)
	A = gp.tuplelist([(i, j) for i in V for j in V])

	distances = {
		(i, j): d[i][j]
		for i, j in combinations(V, 2)
	}

	x = m.addVars(distances.keys(), vtype=GRB.BINARY, name="x")
	x.update({(j, i): v for (i, j), v in x.items()})
	y = m.addVars(A, vtype=GRB.BINARY, name="y")

	# set objective
	obj = (gp.quicksum(d[i, j] * x[i, j] for i in V for j in V if i != j)
	       + gp.quicksum(args.alpha * d[i, j] * y[i, j] for i in V for j in V if i != j))
	m.setObjective(obj, GRB.MINIMIZE)

	# Add constraints
	# Exactly p stations (1)
	m.addConstr(gp.quicksum(y[i, i] for i in V) == p)

	# Each node that is not a station is assigned to only one station (2)
	m.addConstrs(gp.quicksum(y[i, j] for j in V) == 1
	             for i in V)

	# Each station is assigned to itself (3)
	m.addConstrs(y[i, j] <= y[j, j]
	             for i in V
	             for j in V if i != j)

	# Each station has 2 edges (in and out) in the TSP path (4)
	m.addConstrs(gp.quicksum(x[i, j] for j in V if i != j) == 2 * y[i, i]
	             for i in V)

	# (9)
	m.addConstrs(y[j, j] >= x[i, j] for i in range(1, n) for j in V if i != j)

	# 0 is the first node in the TSP path
	m.addConstr(y[0, 0] == 1)
	m.addConstrs(y[0, i] == 0 for i in range(1, n))

	# Optimize the model
	m.Params.LazyConstraints = 1
	m.Params.PreCrush = 1
	cb = RingStarCallback(points, x, y, p, V, args.epsilon)
	m.optimize(cb)

	end_time = time.time()
	instance_name = args.filename.split("/")[2].split(".")[0]

	# Check if the optimization was successful
	if m.status == GRB.OPTIMAL:
		objective_value = m.objVal

		# Calculate the runtime
		runtime = end_time - start_time

		rsp.tsp_path, rsp.assignments = get_solution(points, x, y)
		rsp.evaluate()

		# Construct the result to dump
		result_to_dump = {
			"Objective Value": rsp.cost,
			"Cost": rsp.tsp_cost,
			"MeanTime": rsp.time,
			"Ration": rsp.ratio,
			"Runtime": runtime
		}

		# Dump the result
		dump_result(result_to_dump, rsp, "PL_ncf_" + instance_name)
	else:
		dump_result("Failed", None, "PL_ncf_" + instance_name)
		print("Optimization did not find an optimal solution.")


if __name__ == "__main__":
	args = arg_parser()
	pl_non_compact_fractionnaire(args)
