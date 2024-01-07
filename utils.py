import numpy as np


def read_points(filename):
	points = []

	with open(filename, 'r') as file:
		lines = file.readlines()
		start = False
		
		for line in lines:
			if line.startswith('NODE_COORD_SECTION'):
				start = True
				continue
			elif line.startswith('EOF'):
				break
			
			if start:
				parts = line.strip().split()
				points.append((float(parts[1]), float(parts[2])))

	return np.array(points)	