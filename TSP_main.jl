using JuMP
using CPLEX
using CPUTime

include("TSP_IO.jl")
include("TSP_heuristique.jl")
include("TSP_meta_heuristique.jl")
include("TSP_PLNE_MTZ.jl")
include("TSP_BandC.jl")

function Resoud_TSP(filename)

    I = Read_undirected_TSP(filename)

    filename_inst = replace(filename, ".tsp" => "_inst")
    WritePdf_visualization_TSP(I, filename_inst)

    # Heuristique gloutonne
    S_ppv = plus_proche_voisin(I)

    val_ppv = Compute_value_TSP(I, S_ppv)
    println("Solution Heuristique S=", S_ppv)
    println("Valeur: ", val_ppv)
    println()

    filename_ppv = replace(filename, ".tsp" => "_ppv")

    WritePdf_visualization_solution_ordre(I, S_ppv, filename_ppv)


    # Meta-heuristique

    S_meta = desc_grad_stock_2opt(I, S_ppv)

    val_meta = Compute_value_TSP(I, S_meta)
    println("Solution Meta :S=", S_meta)
    println("Valeur: ", val_meta)
    println()

    filename_meta = replace(filename, ".tsp" => "_meta")

    WritePdf_visualization_solution_ordre(I, S_meta, filename_meta)


    # Launch compact PLNE and visualization for the stable set problem

    @time @CPUtime S_MTZ = PLNE_TSP_MTZ(I) # on le résout

    val_MTZ = Compute_value_TSP(I, S_MTZ)
    println("Solution MTZ :S=", S_MTZ)
    println("Valeur: ", val_MTZ)
    println()

    filename_MTZ = replace(filename, ".tsp" => "_MTZ")

    WritePdf_visualization_solution_ordre(I, S_MTZ, filename_MTZ) # on fait le pdf



    # Launch compact PLNE and visualization for the stable set problem

    @time @CPUtime S_BC = BandC_TSP(I) # on le résout

    val_BC = Compute_value_TSP(I, S_BC)
    println("Solution BandC :S=", S_BC)
    println("Valeur: ", val_BC)
    println()


    filename_BC = replace(filename, ".tsp" => "_BandC")

    WritePdf_visualization_solution_ordre(I, S_BC, filename_BC) # on fait le pdf



end



