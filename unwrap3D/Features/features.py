
from ..Utility_Functions import file_io as fio
import numpy as np 
from scipy import signal


class DsiftExtractor:
	r"""
	The class that does dense sift feature extractor. See https://github.com/Yangqing/dsift-python
	Sample Usage:
		extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
		feaArr,positions = extractor.process_image(Image)
	
	Reference : 
		Y. Jia and T. Darrell. "Heavy-tailed Distances for Gradient Based Image Descriptors". NIPS 2011.
		
		Lowe, David G. "Object recognition from local scale-invariant features." Proceedings of the seventh IEEE international conference on computer vision. Vol. 2. Ieee, 1999.
	"""
	def __init__(self, gridSpacing, patchSize,
				 nrml_thres = 1.0,\
				 sigma_edge = 0.8,\
				 sift_thres = 0.2,
				 Nangles = 8,
				 Nbins = 4,
				 alpha = 9.0):
		'''
		gridSpacing: 
			the spacing for sampling dense descriptors
		patchSize: int
			the size of each sift patch
		nrml_thres: scalar
			low contrast normalization threshold
		sigma_edge: scalar
			the standard deviation for the gaussian smoothing before computing the gradient
		sift_thres: scalar
			sift thresholding (0.2 works well based on Lowe's SIFT paper)
		'''
		self.Nangles = Nangles
		self.Nbins = Nbins
		self.alpha = alpha
		self.Nsamples = Nbins**2
		self.angles = np.array(range(Nangles))*2.0*np.pi/Nangles # the thresholds of the angle histogram [0,2pi]
		
		self.gS = gridSpacing
		self.pS = patchSize
		self.nrml_thres = nrml_thres
		self.sigma = sigma_edge
		self.sift_thres = sift_thres
		# compute the weight contribution map
		sample_res = self.pS / np.double(Nbins) # spatial resolution. 
		sample_p = np.array(range(self.pS))
		sample_ph, sample_pw = np.meshgrid(sample_p,sample_p) # this is 32 x 32 (image squared?)
		sample_ph.resize(sample_ph.size)
		sample_pw.resize(sample_pw.size)
		bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5 
#        print(bincenter)
		bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
#        print(bincenter_h)
#        print(bincenter_w)
		bincenter_h.resize((bincenter_h.size,1))
		bincenter_w.resize((bincenter_w.size,1))
		dist_ph = abs(sample_ph - bincenter_h)
		dist_pw = abs(sample_pw - bincenter_w)
		weights_h = dist_ph / sample_res
		weights_w = dist_pw / sample_res
		weights_h = (1-weights_h) * (weights_h <= 1)
		weights_w = (1-weights_w) * (weights_w <= 1)
		# weights is the contribution of each pixel to the corresponding bin center
		self.weights = weights_h * weights_w

	def gen_dgauss(self,sigma):
		r'''
		generates a derivative of Gaussian filter with the same specified :math:`\sigma` in both the X and Y
		directions.
		'''
		fwid = int(2*np.ceil(sigma))
		G = np.array(range(-fwid,fwid+1))**2
		G = G.reshape((G.size,1)) + G
		G = np.exp(- G / 2.0 / sigma / sigma)
		G /= np.sum(G)
		GH,GW = np.gradient(G)
		GH *= 2.0/np.sum(np.abs(GH))
		GW *= 2.0/np.sum(np.abs(GW))
		return GH,GW
		
	def process_image(self, image, positionNormalize = True,\
					   verbose = True):
		'''
		processes a single image, return the locations
		and the values of detected SIFT features.
		image: a M*N image which is a numpy 2D array. If you pass a color image, it will automatically be convertedto a grayscale image.
		positionNormalize: whether to normalize the positions to [0,1]. If False, the pixel-based positions of the top-right position of the patches is returned.
		
		Return values:
		feaArr: the feature array, each row is a feature
		positions: the positions of the features
		'''

		image = image.astype(np.double)
		if image.ndim == 3:
			# we do not deal with color images.
			image = np.mean(image,axis=2)
		# compute the grids
		H,W = image.shape
		gS = self.gS
		pS = self.pS
		remH = np.mod(H-pS, gS)
		remW = np.mod(W-pS, gS)
		offsetH = remH//2
		offsetW = remW//2
		gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
		gridH = gridH.flatten()
		gridW = gridW.flatten()
		if verbose:
			print ('Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
					format(W,H,gS,pS,gridH.size))
		feaArr = self.calculate_sift_grid(image,gridH,gridW) # this is the heavy lifting. 
		feaArr = self.normalize_sift(feaArr)
		if positionNormalize:
			positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
		else:
			positions = np.vstack((gridH, gridW))
		return feaArr, positions

	def calculate_sift_grid(self,image,gridH,gridW):
		'''
		This function calculates the unnormalized sift features at equidistantly spaced control points in the image as specified by the number in height (gridH) and in width (gridW)
		It is called by process_image().
		'''
		from scipy import signal

		H,W = image.shape
		Npatches = gridH.size
		feaArr = np.zeros((Npatches, self.Nsamples*self.Nangles)) # Nsamples is the number of grid positions of the image being taken. # number of angles 
		
		# calculate gradient
		GH,GW = self.gen_dgauss(self.sigma) # this is the gradient filter for the image. 
		
		IH = signal.convolve2d(image,GH,mode='same')
		IW = signal.convolve2d(image,GW,mode='same')
		Imag = np.sqrt(IH**2+IW**2)
		Itheta = np.arctan2(IH,IW)
		Iorient = np.zeros((self.Nangles,H,W))
		
		for i in range(self.Nangles):
			Iorient[i] = Imag * np.maximum(np.cos(Itheta - self.angles[i])**self.alpha,0) # either increment the count or not. 
			#pyplot.imshow(Iorient[i])
			#pyplot.show()
		for i in range(Npatches):
			currFeature = np.zeros((self.Nangles,self.Nsamples))
			for j in range(self.Nangles):
				# this is the gaussian spatial weights in each cell. 
				currFeature[j] = np.dot(self.weights,\
						Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
			feaArr[i] = currFeature.flatten()
		return feaArr

	def normalize_sift(self,feaArr):
		'''
		This function does sift feature normalization
		following David Lowe's definition (normalize length ->
		thresholding at 0.2 -> renormalize length)
		'''
		siftlen = np.sqrt(np.sum(feaArr**2,axis=1)) # this is the L2
		hcontrast = (siftlen >= self.nrml_thres)
		siftlen[siftlen < self.nrml_thres] = self.nrml_thres
		# normalize with contrast thresholding
		feaArr /= siftlen.reshape((siftlen.size,1))
		# suppress large gradients
		feaArr[feaArr>self.sift_thres] = self.sift_thres
		# renormalize high-contrast ones
		feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
				reshape((feaArr[hcontrast].shape[0],1))
		return feaArr

class SingleSiftExtractor(DsiftExtractor):
	'''
	The simple wrapper class that does feature extraction, treating
	the whole image as a local image patch.
	'''
	def __init__(self, patchSize,
				 nrml_thres = 1.0,\
				 sigma_edge = 0.8,\
				 sift_thres = 0.2,
				 Nangles = 8,
				 Nbins = 4,
				 alpha = 9.0):
		# simply call the super class __init__ with a large gridSpace
		DsiftExtractor.__init__(self, patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)   
	
	def process_image(self, image):
		return DsiftExtractor.process_image(self, image, False, False)[0]

	