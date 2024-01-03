import numpy as np
import matplotlib.pyplot as plt

class NuagePoints:
	def __init__(self, filename):
		self.points = None
		self.num_cities = 0
		self.max_distance = 0
		self._read_points(filename)


	def _read_points(self, filename):
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

		self.points = np.array(points)
		self.num_cities = len(points)
		self.max_distance = np.max(np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1)))

	def draw(self):
		if self.points is not None:
			plt.scatter(self.points[:,0], self.points[:,1])




