# Code largement inspiré du travail de Florian Belhassen-Dubois
using Plots

# Sructure de données "Nuage de points"
struct NuagePoints
    nb_points    # nombre de points
    X            # tableau des coordonnées x des points
    Y            # tableau des coordonnées y des points    
    max_distance # plus longue distance entre deux points
end

# Lecture d'un fichier de la TSPLIB .tsp (atttention uniquement les instances symétriques données par des coordonnées)
# Renvoie l'instance du TSP c
function Read_undirected_TSP(filename)

    I = NuagePoints(0, [], [], 0)

    open(filename) do f

        node_coord_section = 0 # repère dans quelle section du fichier .tsp on est en train de lire
        nbp = 0
        X = Array{Float64}(undef, 0)
        Y = Array{Float64}(undef, 0)

        # on lit une par une nos ligne du fichier	
        for (i, line) in enumerate(eachline(f))

            # on sépare cette ligne en mots
            x = split(line, " ") # For each line of the file, splitted using space as separator

            # on supprime les mots vides, en effet une espace suivi d'un autre espace renvoie le mot vide
            deleteat!(x, findall(e -> e == "", x))


            if (node_coord_section == 0)       # If it's not a node section

                # si on est dans les coordonnées on va s'arrêter et remplir notre instance sinon il nous reste des labels à lire
                if (x[1] == "NODE_COORD_SECTION")
                    node_coord_section = 1
                    # si on est dans le label dimension, on le récupère
                elseif (x[1] == "DIMENSION")
                    nbp = parse(Int, x[3])
                end

                # on est enfin dans nos coordonnées ! On les lit et on remplit notre instance avec
            elseif (node_coord_section == 1 && x[1] != "EOF")

                push!(X, parse(Float64, x[2]))
                push!(Y, parse(Float64, x[3]))

            else

                node_coord_section = 2

            end
        end


        # Calcule la plus longue distance entre deux points
        max_distance = 0
        for i in 1:nbp
            for j in 1:nbp
                if (max_distance < ((X[i] - X[j])^2 + (Y[i] - Y[j])^2))
                    max_distance = (X[i] - X[j])^2 + (Y[i] - Y[j])^2
                end
            end
        end

        # on construit notre nuage de points
        I = NuagePoints(nbp, X, Y, max_distance)


    end
    return I
end

# Visualisation d'une instance comme un nuage de points dans un fichier pdf dont le nom est passé en paramètre
function WritePdf_visualization_TSP(I, filename)

    filename_splitted_in_two_parts = split(filename, ".") # split to remove the file extension
    filename_with_pdf_as_extension = filename_splitted_in_two_parts[1] * ".pdf"
    # save to pdf

    # un simple plot en fonction des coordonnées 
    p = plot(I.X, I.Y, seriestype=:scatter)
    savefig(p, filename_with_pdf_as_extension)

end


# Renvoie la distance euclidienne entre deux points du nuage
function dist(I, i, j)

    return ((I.X[i] - I.X[j])^2 + (I.Y[i] - I.Y[j])^2)^(0.5)

end

# Crée une matrice de toutes les distances point à point
function calcul_dist(I)

    c = Array{Float64}(undef, (I.nb_points, I.nb_points))

    for i in 1:I.nb_points

        for j in 1:I.nb_points

            c[i, j] = dist(I, i, j)

        end

    end

    return c

end


# calcule la somme des coûts de notre arête solution
function Compute_value_TSP(I, S)

    res = ((I.X[S[1]] - I.X[S[I.nb_points]])^2 + (I.Y[S[1]] - I.Y[S[I.nb_points]])^2)^(0.5)
    for i = 1:(I.nb_points-1)
        res = res + ((I.X[S[i]] - I.X[S[i+1]])^2 + (I.Y[S[i]] - I.Y[S[i+1]])^2)^(0.5)
    end

    return res

end


# permet de visualiser notre solution (un circuit / cycle) dans un fichier pdf dont le nom est spécifié en paramètres
# La solution est donnée par la liste ordonné des points à visiter commençant par 1
function WritePdf_visualization_solution_ordre(I, S, filename)

    filename_splitted_in_two_parts = split(filename, ".") # split to remove the file extension
    filename_with_pdf_as_extension = filename_splitted_in_two_parts[1] * ".pdf"
    # save to pdf

    tabX = Float16[]
    tabY = Float16[]

    for i in S
        push!(tabX, I.X[i])
        push!(tabY, I.Y[i])
    end

    # on ajoute le premier point pour le plot, c'est important sinon il manque l'arête entre 1 et n...
    push!(tabX, I.X[1])
    push!(tabY, I.Y[1])

    p = plot(I.X, I.Y, seriestype=:scatter, legend=false)
    plot!(p, tabX, tabY, legend=false)
    savefig(p, filename_with_pdf_as_extension)

end
