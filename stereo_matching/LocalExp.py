import numpy as np
import cv2
from multiprocessing import Pool
import maxflow 

class LocalExpStereo():
	"""
	Local expansion moves from Taniai T., et al, PAMI 2017.
	"""
	def __init__(self, ndisp=280):
		# initialize parameters as paper states
		# optimization parameters
		self.cellSize = [5, 15, 25] # 3 grid structures: 5x5, 15x15, 25x25
		self.Kprop = {5:1, 15:2, 25:2} # iteration numbers for propagation for 3 grid structures
		self.Krand = {5:7, 15:0, 25:0} # for randomization
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

	def generateDisparityMap(self, leftImg, rightImg):
		"""
		Generate disparity maps for both left and right images.
		It follows the optimization procedure (Algorithm 2 in paper).
		--------------------------------------------------------
		Inputs:
		- leftImg: Left image
		- rightImg: right image
		Outputs:
		- leftDis: Left disparity
		- rightDis: right disparity
		"""
		f_left, f_right = self.optimize(leftImg, rightImg)
		leftDis, rightDis = self.postprocessing(f_left, f_right)

		return leftDis, rightDis

	def optimize(self, leftImg, rightImg):
		"""
		Optimization involves iterative alpha-expansion in 16 disjoint groups
		for each grid structure (3 by default).
		--------------------------------------------------------
		Inputs:
		- leftImg: Left image
		- rightImg: right image
		Outputs:
		- f_left: Left disparity plane
		- f_right: right disparity plane
		"""
		# initialize f randomly
		print('Initialize f randomly')
		f_left = np.zeros_like(leftImg) # f: (x, y, fp), fp = (ap, bp, cp)
		z0 = np.random.uniform(0, self.ndisp, f_left.shape[:2])
		n_left = self.randomUnitVector(f_left.shape)
		f_left[...,0] = -n_left[...,0]/n_left[...,2] # ap = -nx/nz
		f_left[...,1] = -n_left[...,1]/n_left[...,2] # bp = -ny/nz
		f_left[...,2] = -(n_left[...,0]*np.arange(f_left.shape[0])[...,np.newaxis] + \
					 n_left[...,1]*np.arange(f_left.shape[1]) + n_left[...,2]*z0)/n_left[...,2] # cp = -(nxpu + nypv + nzz0)/nz

		f_right = np.zeros_like(rightImg) # f: (x, y, fp), fp = (ap, bp, cp)
		z0 = np.random.uniform(0, self.ndisp, f_right.shape[:2])
		n_right = self.randomUnitVector(f_right.shape)
		f_right[...,0] = -n_right[...,0]/n_right[...,2] # ap = -nx/nz
		f_right[...,1] = -n_right[...,1]/n_right[...,2] # bp = -ny/nz
		f_right[...,2] = -(n_right[...,0]*np.arange(f_right.shape[0])[...,np.newaxis] + \
					 n_right[...,1]*np.arange(f_right.shape[1]) + n_right[...,2]*z0)/n_right[...,2] # cp = -(nxpu + nypv + nzz0)/nz
		print('f shape:', f_left.shape, f_right.shape)

		# initialize perturbation size
		print('Initialize perturbation size')
		rd = self.ndisp/2; rn = 1

		# loop for 'iter' times
		for a in range(self.iter):
			print('Main loop iteration:', a+1, '/', self.iter)
			## for each grid structure
			for cellSize in self.cellSize:
				print('Grid structure:', cellSize)
				# define cell grid size
				cellHeight = leftImg.shape[0]/cellSize
				cellWidth = leftImg.shape[1]/cellSize

				## for each disjoint group (0, ..., 15)
				groupIdx = np.meshgrid(range(4), range(4))
				for i, j in zip(groupIdx[1].flatten(), groupIdx[0].flatten()):
					print('Disjoint group:', i, j)
					# compute center index for each disjoint expansion region
					y, x = np.meshgrid(np.arange(j, cellWidth, 4), np.arange(i, cellHeight, 4))
					y = y.flatten(); x = x.flatten()

					## for each cell (in parallel)
					for center_i, center_j in zip(x, y):
						print('For cell:', center_i, center_j)
						# define expansion region (pixel level)
						topleftIdx = (max(0, (center_i-1)*cellSize), max(0, (center_j-1)*cellSize)) # inclusive
						bottomrightIdx = (min(leftImg.shape[0], (center_i+1)*cellSize), 
										  min(leftImg.shape[1], (center_j+1)*cellSize)) # exclusive

						## propagation
						for b in range(self.Kprop[cellSize]):
							print('Propagation:', b+1, '/', self.Kprop[cellSize])
							# randomly choose an f from center region
							fr = f_left[np.random.randint(center_i*cellSize, (center_i+1)*cellSize), 
										np.random.randint(center_j*cellSize, (center_j+1)*cellSize)]

							# alpha expansion
							f_left = self.alphaExp(f_left, leftImg, rightImg, topleftIdx, bottomrightIdx, fr)

							# repeat for right disparity
							fr = f_right[np.random.randint(center_i*cellSize, (center_i+1)*cellSize), 
										np.random.randint(center_j*cellSize, (center_j+1)*cellSize)]
							f_right = self.alphaExp(f_right, rightImg, leftImg, topleftIdx, bottomrightIdx, fr)

						## refinement
						rd_ = rd; rn_ = rn
						for c in range(self.Krand[cellSize]):
							print('Refinement:', c+1, '/', self.Krand[cellSize])
							# randomly choose an f from center region
							v, u = np.random.randint(center_i*cellSize, (center_i+1)*cellSize), np.random.randint(center_j*cellSize, (center_j+1)*cellSize)
							fr = f_left[v,u]

							# perturb the chosen f
							z0 = self.disparity(fr, u, v) + np.random.uniform(-rd_, rd_)
							n = np.array([-f[0]/np.sqrt(f[0]**2+f[1]**2+1), -f[1]/np.sqrt(f[0]**2+f[1]**2+1), 
								 		  1/np.sqrt(f[0]**2+f[1]**2+1)])
							theta = np.random.uniform(-np.pi/2, np.pi/2); phi = np.random.uniform(0, 2*np.pi)
							n += np.array([rn_*np.cos(theta)*np.cos(phi), rn_*np.cos(theta)*np.sin(phi), rn_*np.sin(theta)])
							n = n / np.sum(n**2)**0.5
							fr[0] = -n[0]/n[2] # ap = -nx/nz
							fr[1] = -n[1]/n[2] # bp = -ny/nz
							fr[2] = -(n[0]*u + n[1]*v + n[2]*z0)/n[2] # cp = -(nxpu + nypv + nzz0)/nz

							# alpha expansion
							f_left = self.alphaExp(f, leftImg, rightImg, topleftIdx, bottomrightIdx, fr)

							# repeat for right disparity
							v, u = np.random.randint(center_i*cellSize, (center_i+1)*cellSize), np.random.randint(center_j*cellSize, (center_j+1)*cellSize)
							fr = f_right[v,u]

							z0 = self.disparity(fr, u, v) + np.random.uniform(-rd_, rd_)
							n = np.array([-f[0]/np.sqrt(f[0]**2+f[1]**2+1), -f[1]/np.sqrt(f[0]**2+f[1]**2+1), 
								 		  1/np.sqrt(f[0]**2+f[1]**2+1)])
							theta = np.random.uniform(-np.pi/2, np.pi/2); phi = np.random.uniform(0, 2*np.pi)
							n += np.array([rn_*np.cos(theta)*np.cos(phi), rn_*np.cos(theta)*np.sin(phi), rn_*np.sin(theta)])
							n = n / np.sum(n**2)**0.5
							fr[0] = -n[0]/n[2] # ap = -nx/nz
							fr[1] = -n[1]/n[2] # bp = -ny/nz
							fr[2] = -(n[0]*u + n[1]*v + n[2]*z0)/n[2] # cp = -(nxpu + nypv + nzz0)/nz

							f_right = self.alphaExp(f, rightImg, leftImg, topleftIdx, bottomrightIdx, fr)

							rd_ = rd_ / 2; rn_ = rn_ / 2
			rd = rd / 2; rn = rn / 2

		return f_left, f_right

	def postprocessing(self, f_left, f_right, leftImg, rightImg):
		"""
		Left/right consistency check and weighted median filtering.
		--------------------------------------------------------
		Inputs:
		- f_left: Left disparity plane
		- f_right: right disparity plane
		Outputs:
		- leftDis: Left disparity
		- rightDis: right disparity
		"""
		# find invalidated pixels
		s_y, s_x = np.meshgrid(range(f_left.shape[1]), range(f_left.shape[0]))
		match_y, match_x = s_y - np.ceil(f_left), s_x
		checkLeft = np.absolute(f[s_x, s_y] - f[match_x, match_y]) > 1
		match_y, match_x = s_y - np.ceil(f_right), s_x
		checkright = np.absolute(f[s_x, s_y] - f[match_x, match_y]) > 1

		# fill in the invalidated pixels
		for i, j in np.where(checkLeft):
			leftValid = [i, j-1]
			while leftValid[1] > 0 and not check[leftValid]:
				leftValid[1] -= 1
			rightValid = [i, j+1]
			while rightValid[1] < leftImg.shape[1] and not check[rightValid]:
				rightValid[1] += 1

			leftValid[1] = max(0, leftValid[1]); rightValid[1] = min(rightValid[1], leftImg.shape[1]-1)
			d = {tuple(f_left[leftValid]): self.disparity(f_left, leftValid[1], leftValid[0]), 
				 tuple(f_left[rightValid]): self.disparity(f_left, rightValid[1], rightValid[0])}
			f_left[i,j] = min(d, key=lambda x: d[x])

		for i, j in np.where(checkright):
			leftValid = [i, j-1]
			while leftValid[1] > 0 and not check[leftValid]:
				leftValid[1] -= 1
			rightValid = [i, j+1]
			while rightValid[1] < leftImg.shape[1] and not check[rightValid]:
				rightValid[1] += 1

			leftValid[1] = max(0, leftValid[1]); rightValid[1] = min(rightValid[1], rightImg.shape[1]-1) 
			d = {tuple(f_right[leftValid]): self.disparity(f_right, leftValid[1], leftValid[0]), 
				 tuple(f_right[rightValid]): self.disparity(f_right, rightValid[1], rightValid[0])}
			f_right[i,j] = min(d, key=lambda x: d[x])

		# disparity plane f -> disparity d
		leftDis = self.disparity(f_left, s_y, s_x)
		rightDis = self.disparity(f_right, s_y, s_x)

		# apply weighted median filtering
		r = int(self.WpSize / 2)
		for i, j in np.where(checkLeft):
			beta = leftImg[max(0,i-r):min(i+r+1,leftImg.shape[0]),max(0,j-r):min(j+r+1,leftImg.shape[1])]
			w = np.tile(np.exp(-np.absolute(leftImg[i,j]-beta)/self.gamma)[np.newaxis,...], (beta.size,1,1))
			X = np.tile(beta[np.newaxis,...], (beta.size,1,1))
			beta = beta.flatten().reshape(-1,1,1)
			idx = np.argmin(np.sum(w*np.absolute(X-beta), axis=(1,2)))
			leftDis[i,j] = beta[idx,0,0]

		for i, j in np.where(checkright):
			beta = rightImg[max(0,i-r):min(i+r+1,rightImg.shape[0]),max(0,j-r):min(j+r+1,rightImg.shape[1])]
			w = np.tile(np.exp(-np.absolute(rightImg[i,j]-beta)/self.gamma)[np.newaxis,...], (beta.size,1,1))
			X = np.tile(beta[np.newaxis,...], (beta.size,1,1))
			beta = beta.flatten().reshape(-1,1,1)
			idx = np.argmin(np.sum(w*np.absolute(X-beta), axis=(1,2)))
			rightDis[i,j] = beta[idx,0,0]

		return leftDis, rightDis		

	def randomUnitVector(self, size):
		"""
		Generate random unit vector for f initialization.
		--------------------------------------------------------
		Inputs:
		- size: number of the vector (normally the f shape)
		Outputs:
		- vec: random unit vector in the shape of 'size'
		"""
		vec = np.random.normal(0, 1, size)
		mag = np.sum(vec**2, axis=-1, keepdims=True)**0.5
		return vec/mag

	def disparity(self, f, u, v):
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

	def alphaExp(self, f, refImg, matchImg, topleftIdx, bottomrightIdx, alpha):
		"""
		Alpha expansion.
		--------------------------------------------------------
		Inputs:
		- f: disparity plane: (H, W, 3)
		- refImg, matchImg: refImg matches to matchImg (e.g. left disparity: left matches right)
		- topleftIdx, bottomrightIdx: define the expansion region: (x,y), (x,y)
		- alpha: alternative label
		Outputs:
		- f: alpha-expanded disparity plane
		"""
		# reduce re-computation efforts
		f_new = np.tile(alpha.reshape(1,1,-1), (f.shape[0], f.shape[1],1))
		struct = {'right': np.array([[0,0,0],[0,0,1],[0,0,0]]), 
			 'bottom-right': np.array([[0,0,0],[0,0,0],[0,0,1]]),
			 'bottom': np.array([[0,0,0],[0,0,0],[0,1,0]]), 
			 'bottom-left': np.array([[0,0,0],[0,0,0],[1,0,0]])}

		# loop until convergence
		energy = 0; prev_energy = np.inf
		while energy < prev_energy:
			# create the graph
			g = maxflow.Graph[float]()
			nodeids = g.add_grid_nodes(refImg.shape[:2])

			# add unary cost (data term) as terminal edges
			# new f (f_new) add to source(0), original f add to sink(1)
			sinkedges = self.unaryCost(f, refImg, matchImg, topleftIdx, bottomrightIdx)
			# if equal to 'alpha'(f_new), set sink edges to inf
			comp = np.sum(np.isclose(np.around(f_new, 2), np.around(f, 2)), axis=-1)
			sinkedges[comp==3] = np.inf
			g.add_grid_tedges(nodeids, self.unaryCost(f_new, refImg, matchImg, topleftIdx, bottomrightIdx),
							  sinkedges)

			# add pairwise cost (smoothness term)
			# smoothness between center and 4 neighbors (another 4 are symmetric)
			# (right, bottom-right, bottom, bottom-left)
			g.add_grid_edges(nodeids, self.pairwiseCost(f, refImg, topleftIdx, bottomrightIdx, 'right'), struct['right'], True)
			g.add_grid_edges(nodeids, self.pairwiseCost(f, refImg, topleftIdx, bottomrightIdx, 'bottom-right'), struct['bottom-right'], True)
			g.add_grid_edges(nodeids, self.pairwiseCost(f, refImg, topleftIdx, bottomrightIdx, 'bottom'), struct['bottom'], True)
			g.add_grid_edges(nodeids, self.pairwiseCost(f, refImg, topleftIdx, bottomrightIdx, 'bottom-left'), struct['bottom-left'], True)

			# maxflow
			prev_energy = energy
			energy = g.maxflow()

			# update new label
			seg = g.get_grid_segments(nodeids)
			f[seg==True] = alpha

		return f


	def unaryCost(self, f, refImg, matchImg, topleftIdx, bottomrightIdx):
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
		GuidedFilter = cv2.createGuidedFilter(refImg[s_x,s_y], self.WkSize, self.e)
		E_data = None; GuidedFilter.filter(rou, E_data)
		return E_data

	def pairwiseCost(self, f, refImg, topleftIdx, bottomrightIdx):
		"""
		Compute the pairwise cost/smoothness term of MRF energy function.
		--------------------------------------------------------
		Inputs:
		- f: disparity plane: (H, W, 3)
		- refImg: refImg matches to matchImg (e.g. left disparity: left matches right)
		- topleftIdx, bottomrightIdx: define the expansion region: (x,y), (x,y)
		Outputs:
		- E_smooth: Energy of smoothness term: (H, W, 4) (in 4 neighbors)
		"""
		E_smooth = np.zeros((f.shape[0],f,shape[1],4))
		for i, direction in enumerate(['right', 'bottom-right', 'bottom', 'bottom-left']):
			E_smooth[...,i] = self.add_edge(f, refImg, topleftIdx, bottomrightIdx, direction)

		return E_smooth

	def add_edge(self, f, refImg, topleftIdx, bottomrightIdx, direction):
		"""
		Used in pairwiseCost function to add edges.
		--------------------------------------------------------
		Input:
		- f: disparity plane: (H, W, 3)
		- refImg: refImg matches to matchImg (e.g. left disparity: left matches right)
		- topleftIdx, bottomrightIdx: define the expansion region: (x,y), (x,y)
		- direction: one of 'right', 'bottom-right', 'bottom', 'bottom-left'
		Outputs:
		- edges: smoothness term specific to 'direction'
		"""
		d = {'right': (0, f.shape[0], 1, f.shape[1], 0, 1, 0, 0, 0, 1), 
			 'bottom-right': (1, f.shape[0], 1, f.shape[1], 1, 1, 0, 1, 0, 1),
			 'bottom': (1, f.shape[0], 0, f.shape[1], 1, 0, 0, 1, 0, 0), 
			 'bottom-left': (1, f.shape[0], 0, f.shape[1]-1, 1, -1, 0, 1, 1, 0)}
		i_start, i_end, j_start, j_end, i_offset, j_offset, up, down, left, right = d[direction]

		w = np.exp(-np.absolute(refImg-np.pad(refImg[i_start:i_end,j_start:j_end], ((up,down),(left,right)), 'constant')/self.gamma))
		fq = np.pad(f[i_start:i_end,j_start:j_end], ((up,down),(left,right)), 'constant')
		p_u, p_v = np.meshgrid(range(topleftIdx[1],bottomrightIdx[1]),
							   range(topleftIdx[0],bottomrightIdx[0]))
		q_u, q_v = np.meshgrid(range(topleftIdx[1]+j_offset,bottomrightIdx[1]+j_offset),
							   range(topleftIdx[0]+i_offset,bottomrightIdx[0]+i_offset))
		psi = np.absolute(self.disparity(f, p_u, p_v)-self.disparity(fq, p_u, p_v)) + \
			  np.absolute(self.disparity(fq, q_u, q_v)-self.disparity(f, q_u, q_v))

		return np.maximum(self.eps, w)*np.minimum(self.taoDis, psi)

def tester():
	leftImg = np.float32(cv2.imread('data/left.jpg'))/255
	rightImg = np.float32(cv2.imread('data/right.jpg'))/255
	stereo = LocalExpStereo()
	leftDis, rightDis = stereo.generateDisparityMap(leftImg, rightImg)

tester()