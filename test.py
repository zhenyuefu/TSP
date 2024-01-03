from turtle import st
import gurobipy as gp
from gurobipy import GRB
from math import sqrt
from scipy.spatial import distance_matrix
from NuagePoints import NuagePoints
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Get the distance matrix
nuage = NuagePoints('./Instances_TSP/att48.tsp')
d = distance_matrix(nuage.points, nuage.points)

# Create a Gurobi model
m = gp.Model("RSP")

# Create variables
n = d.shape[0]  # Number of nodes
p = 10  # Number of stations
alpha = 10 
x = {}
y = {}
z = {} 

V = range(n)
E = gp.tuplelist([(i, j) for i in V for j in V if i != j])
A = gp.tuplelist([(i, j) for i in V for j in V])
distances = {
        (i, j): d[i][j]
        for i, j in combinations(V, 2)
    }

# distances times 10 for every element
weighted_tsp_distance = {key: value * alpha for (key, value) in distances.items()}

x = m.addVars(distances.keys(), vtype=GRB.BINARY, name="x", obj=weighted_tsp_distance)
x.update({(j, i): v for (i, j), v in x.items()})
y = m.addVars(A, vtype=GRB.BINARY, name="y", obj=distances)
z = m.addVars(E, vtype=GRB.CONTINUOUS, lb=0, ub=p-1, name="z")


# Add constraints
# Exactly p stations
m.addConstr(gp.quicksum(y[i, i] for i in range(n)) == p) 

# Each node that is not a station is assigned to only one station
m.addConstrs(gp.quicksum(y[i, j] for j in range(n)) == 1 
             for i in range(n))

# 
m.addConstrs(y[i, j] <= y[j, j] 
             for i in range(n) 
             for j in range(n) if i != j)

# Each station has 2 
for i in V:
    m.addConstr(gp.quicksum(x[i, j] for j in V if i != j) == 2 * y[i, i])

m.addConstr(gp.quicksum(z[0, j] for j in range(1, n)) == p - 1)

m.addConstrs((gp.quicksum(z[j, i] for j in range(n) if j != i))
             == (gp.quicksum(z[i, j] for j in range(1, n) if j != i) + y[i, i])
             for i in range(1, n))

m.addConstrs(z[i, j] + z[j, i] <= (p - 1) * x[i, j] 
             for i in range(1, n)
             for j in range(1, n) if j != i )

# Set initial values for y variables
m.addConstr(y[0, 0] == 1)
m.addConstrs(y[0, i] == 0 for i in range(1,n))

# Optimize the model
m.optimize()

# Print the optimal solution
print("Optimal solution:")
for i in range(n):
    if y[i, i].X > 0.5:
        print(f"Node {i} is selected")


# Print tsp path
print("TSP path:")
for i in range(n):
    for j in range(n):
        if i !=j and x[i, j].X > 0.5:
            print(f"Edge {i} -> {j} is selected")
