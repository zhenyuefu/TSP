using LinearAlgebra
using Plots

include("TSP_IO.jl")

function merge_closest_points(points, p)
    while length(points) > p
        min_distance = Inf
        merge_index = (0, 0)
        for i in 1:length(points)
            for j in (i+1):length(points)
                distance = norm(points[i] - points[j])
                if distance < min_distance
                    min_distance = distance
                    merge_index = (i, j)
                end
            end
        end

        deleteat!(points, merge_index[2])

    end
    return points
end

function p_median_greedy(V, p)
    q = ceil(Int, sqrt(p))
    min_x, max_x = minimum(V.X), maximum(V.X)
    min_y, max_y = minimum(V.Y), maximum(V.Y)
    # points a vector dim1 = x, dim2 = y, not tuple
    points = [vcat(x, y) for (x, y) in zip(V.X, V.Y)]
    dx = (max_x - min_x) / q
    dy = (max_y - min_y) / q

    centers = []
    for i in 1:q
        for j in 1:q
            rect_min_x = min_x + (i - 1) * dx
            rect_max_x = min_x + i * dx
            rect_min_y = min_y + (j - 1) * dy
            rect_max_y = min_y + j * dy

            rect_points = filter(p -> rect_min_x <= p[1] <= rect_max_x && rect_min_y <= p[2] <= rect_max_y, points)
            if !isempty(rect_points)
                center = rect_points[argmin(norm(p - [rect_min_x + dx / 2, rect_min_y + dy / 2]) for p in rect_points)]
                push!(centers, center)
            end
        end
    end

    centers = merge_closest_points(centers, p)

    return centers
end

# function p_median_randomized(V, p, num_iterations)
#     best_centers = []
#     best_cost = Inf

#     for _ in 1:num_iterations
#         centers = sample(V, p, replace=false)
#         cost = sum(minimum(norm(p - c) for c in centers) for p in V)

#         if cost < best_cost
#             best_centers = centers
#             best_cost = cost
#         end
#     end

#     return best_centers
# end

filename = "./Instances_TSP/att48.tsp"
I = Read_undirected_TSP(filename)

filename_inst = replace(filename, ".tsp" => "_inst")
centers = p_median_greedy(I, 11)
println(centers)
p = plot(I.X, I.Y, seriestype=:scatter, legend=false)
# plot centers
for c in centers
    plot!([c[1]], [c[2]], seriestype=:scatter, markersize=10, color=:red, legend=false)
end
filename_with_pdf_as_extension = filename_inst * ".pdf"
# save to pdf
savefig(p, filename_with_pdf_as_extension)