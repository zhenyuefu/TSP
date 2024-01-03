using Plots
using Shuffle

include("TSP_heuristique.jl")



# utile pour la descente de gradient
# permet de retourner une chaine / un chemin même si c'est une chaine de 7 à 2 par exemple (coupée par la fin et le début)
function Switch(arr, deb, fin, size)

	# si on est dans le cas classique 
	if deb <= fin
		
		arr[deb:fin] = reverse(arr[deb:fin])
	
	# sinon
	else
	
		a1 = arr[deb:size]
		a2 = arr[1:fin]
		
		a1 = reverse(a1)
		a2 = reverse(a2)
		
		#a3 = vcat(a1, a2)
		a3 = [a1 ; a2]
		
		a3 = reverse(a3)
		
		arr[deb:size] = a3[1:(size - deb +1)]
		arr[1:fin] = a3[(size - deb +2) : (fin + size - deb +1)]
	
	end

end

# calcule la distance entre deux points dont l'indice est spécifié en paramètres 
function distance(a, b, X, Y) 
	
	# petite subtilité, étant donné que le dernier point est aussi le premier, on traite n+1 comme étant 1 
	if(a > length(X))
		a = 1
	end
	
	if(b > length(Y))
		b = 1
	end

	return ((X[a] - X[b])^2 + (Y[a] - Y[b])^2)^(0.5)
end

# tout est dit dans le nom... retourne la solution fournie 
function desc_grad_stock_2opt(I,sol)

	X = copy(I.X)
	Y = copy(I.Y)
	
	#sol2 = copy(sol) #Vector(1:(I.nb_points))
	
	# en pratique le shuffle devrait être fait avant les deux boucles for mais c'est un petit détail
	A = Shuffle.shuffle(Vector(1:(I.nb_points)))
	
	push!(A, A[1])
	
	amelioration = true
		
	# tant qu'on trouve une solution améliorante
	while amelioration

		amelioration = false
		
		# on cherche une combinaison de 2 arêtes à switcher
		for i = 1:(length(A) -2)
		
			# ici on cherche la deuxième
			for j = 2:(length(A) -1)
				
				# on ne veut pas switch d'une facon étrange et qui ne fonctionnerait pas
				if((j != (i -1)) && (j != i) && (j != (i +1)))
					
					# si ca améliore notre solution alors let's switch !
					if(distance(A[i], A[(i +1)], X, Y) + distance(A[j], A[(j +1)], X, Y) > distance(A[i], A[j], X, Y) + distance(A[(i +1)], A[(j +1)], X, Y))
						
						if(i > j)
							Switch(A, j, i+1, length(A))
						else
							Switch(A, i+1, j, length(A))
						end
						
						amelioration = true
						
						break
					
					end
				
				end
			end
			
			# si on a amélioré, on revient au début pour que le shuffle soit appliqué et garder notre "aléatoire"
			if(amelioration)
				break
			end
			
		end
	
	end
	
	return A
	
end


