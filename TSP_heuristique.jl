using Plots
using Shuffle

include("TSP_IO.jl")


# AlIorithme Ilouton "au plus proche voisin" pour le TSP
function plus_proche_voisin(I) 

	X = copy(I.X)
	Y = copy(I.Y)
	
#	# permer de se souvenir du nom du point dont on parle 
#	V = Vector(1:(I.nb_points -1))
#	V = V .+ 1  #Ajoute 

       Added=zero(1:I.nb_points)
       
	# arbitrairement on part du point 1 
	
	res = Int64[]
	push!(res, 1)
	Added[1]=1
	
	cx=X[1]
	cy=Y[1]
	
	# on itère n-1 fois
	for i in 1:I.nb_points-1
		
		ppv = 1
		pcd = I.max_distance
		# on cherche son plus proche voisin
		for j in 1:I.nb_points
		
		  if (Added[j]==0)
			dist = (X[j] - cx)^2 + (Y[j] - cy)^2
			if(pcd > dist)
				pcd = dist
				ppv = j
			end
		  end
		end
		
		# puis on ajoute ce fameux plus proche voisin
		cx = X[ppv]
		cy = Y[ppv]
		
		push!(res, ppv)
		Added[ppv]=1
	
	end
	
	return res
	
end




