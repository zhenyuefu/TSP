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
    def __init__(self, x, y, p, V):
        self.x = x
        self.y = y
        self.p = p
        self.V = V

    def __call__(self,model,where):
        # 检查是否是整数解的回调
        if where == GRB.Callback.MIPSOL:
            # 获取当前解
            x_sol = model.cbGetSolution(self.x)
            
            edges = [(i, j) for (i, j), v in x_sol.items() if v > 0.5]
            cycles = find_subtours(edges)
            if len(cycles)>1:
                for s in cycles:
                    if 1 not in s:
                        for i in s:
                            model.cbLazy(gp.quicksum(self.x[i,j] for j in V if not(j in s)) >= 2*self.y[i, i])

                print("Added lazy constraints for violated cycles")


# Get the distance matrix
nuage = NuagePoints('./Instances_TSP/att48.tsp')
d = distance_matrix(nuage.points, nuage.points)

# Create a Gurobi model
m = gp.Model("RSP")

# Create variables
n = d.shape[0]  # Number of nodes
p = 10  # Number of stations
alpha = 10
x = {} # Assign node i to station j
y = {} # Edge of the TSP path

V = range(n)
E = gp.tuplelist([(i, j) for i in V for j in V if i != j])
A = gp.tuplelist([(i, j) for i in V for j in V])

distances = {
    (i, j): d[i][j]
    for i, j in combinations(V, 2)
}

# distances times 10 for every element
weighted_tsp_distance = {key: value * alpha for (key, value) in distances.items()}

x = m.addVars(distances.keys(), vtype=GRB.BINARY, name="x")
x.update({(j, i): v for (i, j), v in x.items()})
y = m.addVars(A, vtype=GRB.BINARY, name="y")

# set objective
obj = (gp.quicksum( d[i,j] * x[i,j] for i in V for j in V if i != j) 
       +  gp.quicksum(alpha * d[i,j] * y[i,j] for i in V for j in V if i != j))
m.setObjective(obj, GRB.MINIMIZE)

# Add constraints
# Exactly p stations
m.addConstr(gp.quicksum(y[i, i] for i in V) == p)

# Each node that is not a station is assigned to only one station
m.addConstrs(gp.quicksum(y[i, j] for j in V) == 1
             for i in V)

# Each station is assigned to itself
m.addConstrs(y[i, j] <= y[j, j]
             for i in V
             for j in V if i != j)

# Each station has 2 edges (in and out) in the TSP path
m.addConstrs(gp.quicksum(x[i, j] for j in V if i != j) == 2 * y[i, i] 
             for i in V)

# 0 is the first node in the TSP path
m.addConstr(y[0, 0] == 1)
m.addConstrs(y[0, i] == 0 for i in range(1, n))

# Optimize the model
m.Params.LazyConstraints = 1
cb = RingStarCallback(x, y, p, V)
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
nx.draw_networkx_nodes(G, pos, node_size=50)

# draw assignments
nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'assignment'], width=1, alpha=0.5, edge_color='black')

# draw tsp path
nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if G.edges[edge]['type'] == 'tsp'], width=1, alpha=1, edge_color='red')

plt.show()
