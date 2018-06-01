import numpy as np
import cv2
from multiprocessing import Pool

class LocalExpStereo(ndisp=280):
	def __init__():
		# initialize parameters as paper states
		# optimization parameters
		self.cellSize = [5, 15, 25] # 3 grid structures: 5x5, 15x15, 25x25
		self.Kprop = [1, 2, 2] # iteration numbers for propagation for 3 grid structures
		self.Krand = [7, 0, 0] # for randomization
		self.iter = 10 # for main loop
		self.ndisp = ndisp # dispmax: max number of disparity 

		# MRF data term parameters
		self.e = 1e-4
		self.taoCol = 10
		self.taoGrad = 2
		self.alpha = 0.9
		self.WpSize = 41
		self.WkSize = 21

		# MRF smoothness term parameters
		self.lambda_ = 1
		self.taoDis = 1
		self.eps = 0.01
		self.gamma = 10

	def generateDisparityMap(leftImg, rightImg):
		"""
		Generate disparity maps for both left and right images.
		It follows the optimization procedure (Algorithm 2 in paper).
		--------------------------------------------------------
		Inputs:
		- leftImg: Left image
		- rightImg: right image
		Outputs:
		- leftDis: processed Left disparity
		- rightDis: processed right disparity
		"""
		leftDis, rightDis = self.optimize(leftImg, rightImg)
		leftDis, rightDis = self.postprocessing(leftDis, rightDis)

		return leftDis, rightDis

	def optimize(leftImg, rightImg):
		"""
		Optimization involves iterative alpha-expansion in 16 disjoint groups
		for each grid structure (3 by default).
		--------------------------------------------------------
		Inputs:
		- leftImg: Left image
		- rightImg: right image
		Outputs:
		- leftDis: Left disparity
		- rightDis: right disparity
		"""
		# initialize f randomly
		f = np.zeros_like(leftImg) # f: (x, y, fp), fp = (ap, bp, cp)
		z0 = np.random.uniform(0, self.ndisp, f.shape[:2])
		n = self.randomUnitVector(f.shape)
		f[...,0] = -n[0]/n[2] # ap = -nx/nz
		f[...,1] = -n[1]/n[2] # bp = -ny/nz
		f[...,2] = -(n[0]*np.arange(f.shape[0])[...,np.newaxis] + \
					 n[1]*np.arange(f.shape[1]) + n[2]*z0)/n[2] # cp = -(nxpu + nypv + nzz0)/nz

		# initialize perturbation size
		rd = self.ndisp/2; rn = 1

		# loop for 'iter' times
		for _ in range(self.iter):
			# for each grid structure
			for cellSize in self.cellSize:
				# define cell grid size
				cellHeight = leftImg.shape[0]/cellSize
				cellWidth = leftImg.shape[1]/cellSize
				# for each disjoint group (0, ..., 15)
				groupIdx = np.meshgrid(range(4), range(4))
				for i, j in zip(groupIdx[1].flatten(), groupIdx[0].flatten()):
					# compute center index for each disjoint expansion region
					y, x = np.meshgrid(np.arange(j, cellWidth, 4), np.arange(i, cellHeight, 4))
					# create cell grid
					cellGrid = np.zeros((cellHeight, cellWidth))
					cellGrid[x,y] = 1
					cellGrid = cv2.dilate(cellGrid, np.ones((3,3)), iterations=1)
					# for each cell (in parallel)
					for center_i, center_j in x, y:


	def postprocessing():


	def randomUnitVector(dim=3, size):
		"""
		Generate random unit vector for f initialization.
		--------------------------------------------------------
		Inputs:
		- dim: dimension of the vector (3 by default)
		- size: number of the vector (normally the f shape)
		Outputs:
		- vec: random unit vector in the shape of 'size'
		"""
		vec = np.random.normal(0, 1, size)
		mag = np.sum(vec**2, axis=-1, keepdims=True)**0.5
		return vec/mag

	def disparity(f, u, v):
		"""
		Compute disparity for a pixel: dp = au + bv + c.
		--------------------------------------------------------
		Inputs:
		- f: disparity plane, f = (a, b, c)
		- u, v: location of the pixel (invert order of numpy array)
		Outputs:
		- d: disparity for a pixel
		"""
		a, b, c = f
		return a*u + b*v + c