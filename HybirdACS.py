import random
import numpy as np
import networkx as nx

def calculate_total_cost(distance_matrix, routes):
    total_cost = 0
    for route in routes:
        indices = np.array(route)
        pairwise_costs = distance_matrix[indices[:-1], indices[1:]]
        total_cost += np.sum(pairwise_costs)
    return total_cost

def initialize_pheromone(G, init_solution):
    n = len(G.nodes())
    distance_matrix = nx.to_numpy_matrix(G)
    total_cost = calculate_total_cost(distance_matrix, init_solution)
    tau_0 = 1 / (n * total_cost) if total_cost != 0 else 1

    pheromone_matrix = np.full(distance_matrix.shape, tau_0)

    return pheromone_matrix


def calculate_savings(G):
    # Extract nodes and number of nodes
    nodes = list(G.nodes())[1:]  # Assuming the first node is the depot
    n = len(nodes)

    # Create a NumPy array for distances
    distances = np.array(nx.to_numpy_matrix(G, nodelist=[0] + nodes))

    # Distance from depot to each node and vice versa (excluding the depot itself)
    depot_to_nodes = distances[0, 1:]
    nodes_to_depot = distances[1:, 0]

    # Calculate savings matrix
    savings_matrix = depot_to_nodes[:, np.newaxis] + nodes_to_depot - distances[1:, 1:]

    # Extract upper triangular part as savings (excluding diagonal)
    i, j = np.triu_indices(n, k=1)
    savings = savings_matrix[i, j]

    # Create a structured array for easy sorting
    dtype = [('savings', float), ('i', int), ('j', int)]
    structured_savings = np.array(list(zip(savings, np.array(nodes)[i], np.array(nodes)[j])), dtype=dtype)

    # Sort savings in descending order
    structured_savings.sort(order='savings')[::-1]

    return structured_savings

def clarke_wright_savings(G):
    nodes = list(G.nodes())[1:]  # Excluding the depot
    routes = [[node] for node in nodes]
    savings = calculate_savings(G)

    for saving in savings:
        _, i, j = saving
        route1 = next(route for route in routes if i in route)
        route2 = next(route for route in routes if j in route)

        if route1 != route2:
            if (route1[0] == i and route2[-1] == j) or (route1[-1] == i and route2[0] == j):
                routes.remove(route1)
                routes.remove(route2)
                routes.append(route1 + route2 if route1[-1] == i else route2 + route1)

    return [[0] + route + [0] for route in routes]  # Assuming 0 is the depot


def initialize_solution(graph):
    return clarke_wright_savings(graph)


def ant_construction(graph, pheromone_matrix, alpha, beta, q0, max_nodes):
    n = len(graph.nodes())
    path = [0]  # Assuming 0 is the depot
    tabu_list = set(path)
    distance_matrix = nx.to_numpy_matrix(graph)

    while len(path) < max_nodes + 1:  # Including the depot
        i = path[-1]
        available_nodes = [node for node in range(n) if node not in tabu_list]

        if not available_nodes:
            break  # If there are no available nodes to visit

        probabilities = []
        for j in available_nodes:
            tau_ij = pheromone_matrix[i, j]
            eta_ij = 1 / distance_matrix[i, j] if distance_matrix[i, j] != 0 else 0
            probabilities.append((tau_ij ** alpha) * (eta_ij ** beta))

        # Normalizing probabilities
        probabilities_sum = sum(probabilities)
        probabilities = [p / probabilities_sum for p in probabilities]

        # Choose the next node
        if random.random() < q0:
            # Choose the best node based on probability
            max_prob_index = probabilities.index(max(probabilities))
            next_node = available_nodes[max_prob_index]
        else:
            # Choose a node based on roulette wheel selection
            next_node = np.random.choice(available_nodes, p=probabilities)

        path.append(next_node)
        tabu_list.add(next_node)

    if path[-1] != 0:  # Add depot to end if not present
        path.append(0)

    return path

def assign_non_visited_nodes(path, graph):
    # 将图转换为距离矩阵
    distance_matrix = nx.to_numpy_matrix(graph)
    n = len(graph.nodes())

    # 找到未访问的节点
    visited = set(path)
    non_visited = set(range(n)) - visited

    # 分配每个未访问节点到最近的已访问节点
    assignment = {}
    for node in non_visited:
        closest_node = None
        min_distance = float('inf')

        for visited_node in visited:
            if distance_matrix[node, visited_node] < min_distance:
                closest_node = visited_node
                min_distance = distance_matrix[node, visited_node]

        assignment[node] = closest_node

    return assignment


def local_pheromone_update(pheromone_matrix, solution, decay_rate):
    # 局部信息素更新
    pass

def vnd(solution):
    # 可变邻域下降
    pass

def calculate_cost(solution):
    # 计算解决方案的成本
    pass

def lin_kernighan(solution):
    # Lin-Kernighan 局部搜索
    pass

def global_pheromone_update(pheromone_matrix, best_solution, decay_rate):
    # 全局信息素更新
    pass

def hybrid_acs(graph, num_ants, num_iterations, alpha, beta, decay_rate):
    pheromone_matrix = initialize_pheromone(graph)
    Sbest = initialize_solution(graph)

    for _ in range(num_iterations):
        ant_solutions = []

        # 蚂蚁构建路径和分配未访问节点
        for _ in range(num_ants):
            Sm = ant_construction(graph, pheromone_matrix, alpha, beta)
            assign_non_visited_nodes(Sm, graph)
            local_pheromone_update(pheromone_matrix, Sm, decay_rate)
            ant_solutions.append(Sm)

        # 应用 VND
        for Sm in ant_solutions:
            Sm = vnd(Sm)

        # 寻找最佳局部解
        Slocal = min(ant_solutions, key=calculate_cost)

        # 检查并更新全局最优解
        if calculate_cost(Slocal) < calculate_cost(Sbest):
            Sbest = Slocal
            Sbest = lin_kernighan(Sbest)

        # 全局信息素更新
        global_pheromone_update(pheromone_matrix, Sbest, decay_rate)

    return Sbest
