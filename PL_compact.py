from turtle import st
import gurobipy as gp
from gurobipy import GRB
from math import sqrt
from scipy.spatial import distance_matrix
from NuagePoints import NuagePoints
import networkx as nx
import matplotlib.pyplot as plt

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

for i in range(n):
    for j in range(n):
        y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") # Assign node i to station j
        if i != j:
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") # Edge of the TSP path
        if j != 0 and i != j:
            z[i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=p-1, name=f"z_{i}_{j}") # Flow variable

# Set objective function
obj = (gp.quicksum(alpha * d[i][j]*x[i,j] for i in range(n) for j in range(n) if i != j) + 
       gp.quicksum(d[i][j]*y[i,j] for i in range(n) for j in range(n)))

m.setObjective(obj, GRB.MINIMIZE)



# Add constraints
# Exactly p stations
m.addConstr(gp.quicksum(y[i, i] for i in range(n)) == p) 

# Each node that is not a station is assigned to only one station
m.addConstrs(gp.quicksum(y[i, j] for j in range(n)) == 1 
             for i in range(n))

# Each station is assigned to itself
m.addConstrs(y[i, j] <= y[j, j] 
             for i in range(n) 
             for j in range(n) if i != j)

# Each station has 2 edges (in and out) in the TSP path
m.addConstrs(gp.quicksum(x[i, j] + x[j,i] for j in range(n) if j != i) == 2 * y[i, i] 
             for i in range(n))

# Flow constraints
# Flow out of the first node is p-1
m.addConstr(gp.quicksum(z[0, j] for j in range(1, n)) == p - 1)

# Flow out is equal to flow in - 1
m.addConstrs((gp.quicksum(z[j, i] for j in range(n) if j != i))
             == (gp.quicksum(z[i, j] for j in range(1, n) if j != i) + y[i, i])
             for i in range(1, n))

# Flow is on edges of the TSP path
m.addConstrs(z[i, j] + z[j, i] <= (p - 1) * x[i, j] 
             for i in range(1, n)
             for j in range(1, n) if j != i )

# 0 is the first node in the TSP path
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
