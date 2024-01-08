from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance_matrix

from utils import find_subtours, draw_solution, dump_solution, read_points

class RingStarCallback:
	def __init__(self, x, y, p, V):
		self.x = x
		self.y = y
		self.p = p
		self.V = V

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


def pl_non_compact_entier(file, prop, alpha):
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
	cb = RingStarCallback(x, y, p, V)
	m.optimize(cb)


	# show solution
	dump_solution(x)
	draw_solution(points, x, y)

if __name__ == "__main__":
    file = "./Instances_TSP/ch150.tsp"
    prop = 0.20
    alpha = 10
    pl_non_compact_entier(file, prop, alpha)