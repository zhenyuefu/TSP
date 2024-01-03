using JuMP
using CPLEX
using CPUTime

include("TSP_IO.jl")


# fonction principale pour notre PLNE MTZ, graphe orienté ici
function PLNE_TSP_MTZ(G)

	c = calcul_dist(G)
	
	println("Création du PLNE")

	m = Model(CPLEX.Optimizer)

	@variable(m, x[1:G.nb_points, 1:G.nb_points], Bin)
	@variable(m, u[1:G.nb_points], Int)

	@objective(m, Min, sum((sum(x[i, j] * c[i, j]) for j = 1:G.nb_points) for i = 1:G.nb_points ) )

	# contraite pour dire "une arête entrante dans un sommet"
	for i in 1:G.nb_points
		@constraint(m, (sum(x[i, j] for j in 1:G.nb_points))== 1)
	end
	
	# contraite pour dire "une arête sortante dans un sommet"
	for j in 1:G.nb_points
		@constraint(m, (sum(x[i, j] for i in 1:G.nb_points))== 1)
	end
	
	# pas d'arêtes qui commencent et finissent dans le même point !
	for i in 1:G.nb_points
		@constraint(m, x[i, i] == 0)
	end
	
	# il faut une certaine symétrie...
	for i in 1:G.nb_points
		for j in 1:G.nb_points
			@constraint(m, x[i, j] + x[j, i] <= 1 )
		end
	end
	
	# première contrainte mtz
	@constraint(m, u[1] == 1)
	
	# deuxième contrainte mtz
	for i in 2:G.nb_points
		@constraint(m, 2 <= u[i] <= G.nb_points)
	end
	
	# troisième contrainte mtz
	for i in 2:G.nb_points
		for j in 2:G.nb_points
			if i != j
				@constraint(m, u[i] - u[j] +1 <= G.nb_points * (1 - x[i, j]))
			end
		end
	end
		

	print(m)
	println()
	
	println("Résolution du PLNE par le solveur")
	optimize!(m)
   	println("Fin de la résolution du PLNE par le solveur")
   	
	#println(solution_summary(m, verbose=true))

	status = termination_status(m)

	# un petit affichage sympathique
	if status == JuMP.MathOptInterface.OPTIMAL
		println("Valeur optimale = ", objective_value(m))
		println("Solution primale optimale :")
		
		S = Int64[]
        i=1
        j=2
        while (value(x[i, j]) < 0.999) 
     	  j=j+1
        end        
        push!(S,1)
        push!(S,j)
        i=j
        while (i!=1)
          j=1
          while  ( j==i || value(x[i,j]) < 0.999 ) 
             j=j+1
          end
          push!(S,j)
          i=j
		end
		 println("Temps de résolution :", solve_time(m))
		 return S
	else
		 println("Problème lors de la résolution")
	end

end


