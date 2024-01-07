from collections import defaultdict
from itertools import chain, combinations

import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance_matrix
import networkx as nx
import matplotlib.pyplot as plt

from NuagePoints import NuagePoints
import numpy as np

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

class RingStarCallback:
	def __init__(self, points, x, y, p, V, epsilon, alpha):
		self.points = points
		self.x = x
		self.y = y
		self.p = p
		self.V = V
		self.distances_matrix = distance_matrix(points, points)
		self.epsilon = epsilon
		self.alpha = alpha

	def __call__(self,model,where):
		# lazy separation
		if where == GRB.Callback.MIPSOL:
			x_sol = model.cbGetSolution(self.x)
			
			edges = [(i, j) for (i, j), v in x_sol.items() if v > 0.5]
			cycles = find_subtours(edges)
			if len(cycles)>1:
				for S in cycles:
					if 1 not in S:
						for i in S:
							model.cbLazy(gp.quicksum(self.x[x,y] for x in S for y in V if not(y in S)) 
                    					>= 2*self.y[i, i])

				print("Added lazy constraints for violated cycles")

		# heuristic separation
		elif where == GRB.Callback.MIPNODE:
			status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
			if status == GRB.OPTIMAL:
				xrel = model.cbGetNodeRel(self.x)
				yrel = model.cbGetNodeRel(self.y)

				for i in range(1, n):
					if yrel[i, i] > 0:
						S = [i]
						for j in range(1,n):
							if j != i:
								S.append(j)
							if np.sum([xrel[x,y] for x in S for y in V if not(y in S)]) < 2*yrel[i, i]:
								print("Added user cut")
								for k in S:
									model.cbLazy(gp.quicksum(self.x[x,y] for x in S for y in V if not(y in S)) 
                    					>= 2*self.y[k, k])
								break

		# exact separation
		# elif where == GRB.Callback.MIPNODE:
		# 	status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
		# 	if status == GRB.OPTIMAL:
		# 		xrel = model.cbGetNodeRel(self.x)
		# 		yrel = model.cbGetNodeRel(self.y)
		# 		edges = [(i, j) for (i, j), v in xrel.items() if v > 0]	
 
		# 		# Construct an undirected graph with weighted edges
		# 		G = nx.Graph()
		# 		# add nodes
		# 		for i in range(n):
		# 			G.add_node(i, pos = self.points[i])
		# 		# add ring edges
		# 		for (x,y) in edges:
		# 			G.add_edge(x, y, capacity=xrel[x,y])
		# 		# add star edges
		# 		for i in range(n):
		# 			for j in range(i, n):
		# 				if i!=j:
		# 					if yrel[i, j] > 0:
		# 						G.add_edge(i, j, capacity=yrel[i, j])
				
		# 		for i in range(1, n):
		# 			min_cut, cuts = nx.minimum_cut(G, 0, i)
		# 			S = cuts[0]	if i in cuts[0] else cuts[1]
		# 			if min_cut < 2*yrel[i, i] - self.epsilon:
		# 				print("Added user cut")
		# 				for k in S:
		# 					model.cbLazy(gp.quicksum(self.x[x,y] for x in S for y in V if not(y in S)) 
        #             					>= 2*self.y[k, k])



# Get the distance matrix
nuage = NuagePoints('./Instances_TSP/ch150.tsp')
d = distance_matrix(nuage.points, nuage.points)

# Create a Gurobi model
m = gp.Model("RSP")

# Create variables
n = d.shape[0]  # Number of nodes
p = n*0.20  # Number of stations
alpha = 10
x = {} # Edge of the TSP path
y = {} # Assignment of nodes to stations

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
cb = RingStarCallback(nuage.points, x, y, p, V, 0.0001, alpha)
m.optimize(cb)

# Print the optimal solution
print("Optimal solution:")
for i in range(n):
	if y[i, i].X > 0.5:
		print(f"Node {i} is selected")


# Print tsp path
print("TSP path:")
edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
tour = find_subtours(edges)
print(f"Optimal tour: {tour}")

# add all assignments to graph
G = nx.Graph()
for i in range(n):
	G.add_node(i, pos = nuage.points[i])

for i in range(n):
	for j in range(n):
		if y[i, j].X > 0.5 and i != j:
			G.add_edge(i, j, type = 'assignment')

# add tsp path to graph
for edge in edges:
	G.add_edge(edge[0], edge[1], type = 'tsp')

# draw graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G, pos, node_size=5)

# draw assignments
nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'assignment'], width=1, alpha=0.5, edge_color='black')

# draw tsp path
nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'tsp'], width=1, alpha=1, edge_color='red')

plt.show()
