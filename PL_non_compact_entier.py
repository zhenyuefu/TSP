from itertools import combinations
import time

import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance_matrix

from utils import arg_parser, find_subtours, draw_solution, dump_result, read_points

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


def pl_non_compact_entier(args):
	start_time = time.time()
 
	# Get the distance matrix
	points = read_points(args.filename)
	d = distance_matrix(points, points)

	# Create a Gurobi model
	m = gp.Model("RSP")

	# Create variables
	n = d.shape[0]  	# Number of nodes
	p = n*args.prop  	# Number of stations
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
		+  gp.quicksum(args.alpha * d[i,j] * y[i,j] for i in V for j in V if i != j))
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
 
	end_time = time.time()

	# Check if the optimization was successful
	if m.status == GRB.OPTIMAL:
		objective_value = m.objVal
		
		# Calculate the runtime
		runtime = end_time - start_time
		
		# Construct the result to dump
		result_to_dump = {
			"Objective Value": objective_value,
			"Runtime": runtime
		}
		
		# Dump the result
		dump_result(result_to_dump, "PL_nce_"+args.filename)
		draw_solution(points, x, y)
	else:
		dump_result("Failed", "PL_nce_"+args.filename)
		print("Optimization did not find an optimal solution.")



if __name__ == "__main__":
	args = arg_parser()
	pl_non_compact_entier(args)