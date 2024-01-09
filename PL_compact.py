from itertools import combinations
import time

import gurobipy as gp
from gurobipy import GRB

from utils import get_solution, dump_result, read_points, arg_parser
from RSP import RSP


def pl_compact(args):
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
	z = {}  # Flow variable

	V = range(n)
	E = gp.tuplelist([(i, j) for i in V for j in V if i != j])
	A = gp.tuplelist([(i, j) for i in V for j in V])

	distances = {
		(i, j): d[i][j]
		for i, j in combinations(V, 2)
	}

	x = m.addVars(distances.keys(), vtype=GRB.BINARY, name="x")
	x.update({(j, i): v for (i, j), v in x.items()})
	y = m.addVars(A, vtype=GRB.BINARY, name="y")
	z = m.addVars(E, vtype=GRB.CONTINUOUS, lb=0, ub=p - 1, name="z")

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

	# Flow constraints
	# Flow out of the first node is p-1 (5)
	m.addConstr(gp.quicksum(z[0, j] for j in range(1, n)) == p - 1)

	# Flow out is equal to flow in - 1 (6)
	m.addConstrs(
		(gp.quicksum(z[j, i] for j in V if j != i)) == (gp.quicksum(z[i, j] for j in range(1, n) if j != i) + y[i, i])
		for i in range(1, n))

	# Flow is on edges of the TSP path (7)
	m.addConstrs(z[i, j] + z[j, i] <= (p - 1) * x[i, j]
	             for i in V
	             for j in range(1, n) if j != i)

	# 0 is the first node in the TSP path
	m.addConstr(y[0, 0] == 1)
	m.addConstrs(y[0, i] == 0 for i in range(1, n))

	# Optimize the model
	m.optimize()

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
		dump_result(result_to_dump, rsp, "PL_c_" + instance_name)
	else:
		dump_result("Failed", None, "PL_c_" + instance_name)
		print("Optimization did not find an optimal solution.")


if __name__ == "__main__":
	args = arg_parser()
	pl_compact(args)
