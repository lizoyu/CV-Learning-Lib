import numpy as np
import cv2
from multiprocessing import Pool
import PyMaxflow 

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
		f_left = np.zeros_like(leftImg) # f: (x, y, fp), fp = (ap, bp, cp)
		z0 = np.random.uniform(0, self.ndisp, f_left.shape[:2])
		n = self.randomUnitVector(f_left.shape)
		f_left[...,0] = -n[0]/n[2] # ap = -nx/nz
		f_left[...,1] = -n[1]/n[2] # bp = -ny/nz
		f_left[...,2] = -(n[0]*np.arange(f_left.shape[0])[...,np.newaxis] + \
					 n[1]*np.arange(f_left.shape[1]) + n[2]*z0)/n[2] # cp = -(nxpu + nypv + nzz0)/nz
		f_right = np.zeros_like(rightImg) # f: (x, y, fp), fp = (ap, bp, cp)
		z0 = np.random.uniform(0, self.ndisp, _right.shape[:2])
		n = self.randomUnitVector(f_right.shape)
		f_right[...,0] = -n[0]/n[2] # ap = -nx/nz
		f_right[...,1] = -n[1]/n[2] # bp = -ny/nz
		f_right[...,2] = -(n[0]*np.arange(f_right.shape[0])[...,np.newaxis] + \
					 n[1]*np.arange(f_right.shape[1]) + n[2]*z0)/n[2] # cp = -(nxpu + nypv + nzz0)/nz

		# initialize perturbation size
		rd = self.ndisp/2; rn = 1

		# loop for 'iter' times
		for _ in range(self.iter):
			## for each grid structure
			for cellSize in self.cellSize:
				# define cell grid size
				cellHeight = leftImg.shape[0]/cellSize
				cellWidth = leftImg.shape[1]/cellSize
				## for each disjoint group (0, ..., 15)
				groupIdx = np.meshgrid(range(4), range(4))
				for i, j in zip(groupIdx[1].flatten(), groupIdx[0].flatten()):
					# compute center index for each disjoint expansion region
					y, x = np.meshgrid(np.arange(j, cellWidth, 4), np.arange(i, cellHeight, 4))
					# create cell grid
					cellGrid = np.zeros((cellHeight, cellWidth))
					cellGrid[x,y] = 1
					cellGrid = cv2.dilate(cellGrid, np.ones((3,3)), iterations=1)
					## for each cell (in parallel)
					for center_i, center_j in x, y:
						## propagation
						fr = f_left[np.random.randint(center_i*cellSize, (center_i+1)*cellSize), 
									np.random.randint(center_j*cellSize, (center_j+1)*cellSize)]
						# define expansion region (pixel level)
						topleftIdx = (max(0, (center_i-1)*cellSize), max(0, (center_j-1)*cellSize)) # inclusive
						bottomrightIdx = (min(leftImg.shape[0], (center_i+1)*cellSize), 
										  min(leftImg.shape[1], (center_j+1)*cellSize)) # exclusive

						# create the graph
						g = maxflow.Graph[float]()
						nodeids = g.add_grid_nodes(leftImg.shape[:2])

						# add unary cost (data term) as terminal edges
						# original f add to source(0), new f (fr) add to sink(1)
						f_new = np.tile(fr.reshape(1,1,-1), (f_left.shape[0], f_left.shape[1],1))
						g.add_grid_tedges(nodeids, self.unaryCost(f_left, leftImg, rightImg, topleftIdx, bottomrightIdx), 
										  self.unaryCost(f_new, leftImg, rightImg, topleftIdx, bottomrightIdx))

						# add pairwise cost (smoothness term)
						# E_smooth: (H, W, 4)
						# contains smoothness between center and 4 neighbors (another 4 are symmetric)
						# (right, bottom-right, bottom, bottom-left)


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
		Compute disparity map using disparity planes: d = au + bv + c.
		--------------------------------------------------------
		Inputs:
		- f: disparity plane: (H, W, 3)
		- u, v: pixel locations: (H, W)(invert order of numpy array)
		Outputs:
		- d: disparity for a pixel: (H, W)
		"""
		f[...,0] *= u
		f[...,1] *= v
		return np.sum(f, axis=-1)

	def unaryCost(f, refImg, matchImg, topleftIdx, bottomrightIdx):
		"""
		Computer the unary cost/data term of MRF energy function.
		--------------------------------------------------------
		Inputs:
		- f: disparity plane: (H, W, 3)
		- refImg, matchImg: refImg matches to matchImg (e.g. left disparity: left matches right)
		- topleftIdx, bottomrightIdx: define the expansion region: (x,y), (x,y)
		Outputs:
		- E_data: Energy of data term
		"""
		# phi = guidedfilter(expansion region in img, rou)
		# compute rou
		s_y, s_x = np.meshgrid(range(topleftIdx[0],bottomrightIdx[0]),
							   range(topleftIdx[1],bottomrightIdx[1]))
		s_match_y, s_match_x = np.round(s_y - self.disparity(f[s_x, s_y], s_y, s_x)), s_x # use rounding for convinence
		rou = (1-self.alpha)*np.minimum(self.taoCol, np.absolute(refImg[s_x,s_y]-matchImg[s_match_x,s_match_y])) + \
			  self.alpha*np.minimum(self.taoGrad, np.absolute(cv2.Sobel(refImg[s_x,s_y],-1,1,0,ksize=3)-
			  												  cv2.Sobel(matchImg[s_x,s_y],-1,1,0,ksize=3)))

		# guided filtering
		GuidedFilter = cv2.createGuidedFilter(refImg[s_x,s_y], self.WpSize, self.e)
		E_data = None; GuidedFilter.filter(rou, E_data)
		return E_data

	def pairwiseCost(f, refImg, topleftIdx, bottomrightIdx):
		"""
		Computer the pairwise cost/smoothness term of MRF energy function.
		--------------------------------------------------------
		Inputs:
		- f: disparity plane: (H, W, 3)
		- refImg: refImg matches to matchImg (e.g. left disparity: left matches right)
		- topleftIdx, bottomrightIdx: define the expansion region: (x,y), (x,y)
		Outputs:
		- E_smooth: Energy of smoothness term: (H, W, 4) (in 4 neighbors)
		"""
		E_smooth = np.zeros((f.shape[0],f,shape[1],4))

		# add right edges (symmetric)
		w = np.exp(-np.absolute(refImg-np.append(refImg[:,1:], np.zeros((refImg.shape[0],1)), axis=1))/self.gamma)
		fq = np.append(f[:,1:], np.zeros((f.shape[0],1)), axis=1)
		psi = None
