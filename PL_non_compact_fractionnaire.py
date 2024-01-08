from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance_matrix
import networkx as nx

import numpy as np
from utils import find_subtours, draw_solution, dump_solution, read_points

class RingStarCallback:
	def __init__(self, points, x, y, p, V, epsilon):
		self.points = points
		self.n = len(points)
		self.x = x
		self.y = y
		self.p = p
		self.V = V
		self.epsilon = epsilon

	def __call__(self,model,where):
		# lazy separation
		if where == GRB.Callback.MIPSOL:
			x_sol = model.cbGetSolution(self.x)
			
			edges = [(i, j) for (i, j), v in x_sol.items() if v > 0.5]
			cycles = find_subtours(edges)
			if len(cycles)>1:
				for S in cycles:
					nS = [i for i in self.V if i not in S]
					if 1 not in S:
						for i in S:
							model.cbLazy(gp.quicksum(self.x[x,y] for x in S for y in nS) 
                    					>= 2*self.y[i, i])

				print("Added lazy constraints for violated cycles")

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
							if np.sum([xrel[x,y] for x in S for y in nS]) < 2*yrel[i, i]:
								print("Added user cut")
								for k in S:
									model.cbCut(gp.quicksum(self.x[x,y] for x in S for y in nS) 
                    					>= 2*self.y[k, k])
								break

		# exact separation
		elif where == GRB.Callback.MIPNODE:
			if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
				xrel = model.cbGetNodeRel(self.x)
				yrel = model.cbGetNodeRel(self.y)
				edges = [(i, j) for (i, j), v in xrel.items() if v > 0]	
 
				# Construct an undirected graph with weighted edges
				G = nx.Graph()
				# add nodes
				for i in self.V:
					G.add_node(i, pos = self.points[i])
				# add edges
				for (x,y) in edges:
					G.add_edge(x, y, capacity=xrel[x,y])
				
				for i in range(1, self.n):
					min_cut, (S, nS) = nx.minimum_cut(G, 0, i)
					if min_cut < 2*yrel[i, i] - self.epsilon:
						print("Added user cut")
						for k in S:
							model.cbCut(gp.quicksum(self.x[x,y] for x in S for y in nS) 
                    					>= 2*self.y[k, k])


def pl_non_compact_fractionnaire(file, prop, alpha, epsilon):
	# Get the distance matrix
	points = read_points(file)
	d = distance_matrix(points, points)

	# Create a Gurobi model
	m = gp.Model("RSP")

	# Create variables
	n = d.shape[0]  	# Number of nodes
	p = n*prop  		# Number of stations
	x = {} 				# Edge of the TSP path
	y = {} 				# Assignment of nodes to stations

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
	obj = (gp.quicksum( d[i,j] * x[i,j] for i in V for j in V if i != j) 
		+  gp.quicksum(alpha * d[i,j] * y[i,j] for i in V for j in V if i != j))
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
	m.addConstrs(y[j, j] >= x[i, j] for i in range(1, n) for j in V if i!=j)

	# 0 is the first node in the TSP path
	m.addConstr(y[0, 0] == 1)
	m.addConstrs(y[0, i] == 0 for i in range(1, n))

	# Optimize the model
	m.Params.LazyConstraints = 1
	m.Params.PreCrush = 1
	cb = RingStarCallback(points, x, y, p, V, epsilon)
	m.optimize(cb)

	# show solution
	dump_solution(x)
	draw_solution(points, x, y)

if __name__ == "__main__":
	file = "data/instances/instance_1.txt"
	prop = 0.20
	alpha = 10
	epsilon = 0.0001
	pl_non_compact_fractionnaire(file, prop, alpha, epsilon)