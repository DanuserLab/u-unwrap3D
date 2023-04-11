

import numpy as np 

def pad_vol_2_size(im, shape, fill_method='constant', fill_value=0):
	r""" Utility function to center and copy the given volumetric image into a larger volume.  
	
	Parameters
	----------
	im : (MxNxL) array
		input volume to copy
	shape : (m,n,l) array
		new volume shape
	fill_method : str
		method of specifying the missing pixels, one of 

		'constant' : str
			fill with a constant value specifed by the additional argument ``fill_value``
		'mean' : str
			fill with the mean value of ``im``
		'median' : str
			fill with the median value of ``im``
	
	fill_value : scalar
		the constant value to initialise the new array if ``fill_method=='constant'``

	Returns
	-------
	new_im : (MxNxL) array
		output volume the same shape as that specified by ``shape``
	"""
	import numpy as np 

	shift_1 = (shape[0] - im.shape[0]) // 2
	shift_2 = (shape[1] - im.shape[1]) // 2
	shift_3 = (shape[2] - im.shape[2]) // 2
	
	new_im = np.zeros(shape, dtype=np.uint8)
	new_im[shift_1:shift_1+im.shape[0],
		   shift_2:shift_2+im.shape[1],
		   shift_3:shift_3+im.shape[2]] = im.copy()

	return new_im


def mean_vol_img(vol_img_list, target_shape, fill_method='constant', fill_value=0):
	r""" Utility function to find the mean volume image given a list of volumetric image files. The volumes will automatically be padded to the specified target shape. 
	
	Parameters
	----------
	vol_img_list : array of filepaths
		list of volumetric image filepaths to compute the mean image for 
	target_shape : (m,n,l) array
		desired volume shape, must be at least the size of the largest volume shape. 
	fill_method : str
		method of specifying the missing pixels, one of 

		'constant' : str
			fill with a constant value specifed by the additional argument ``fill_value``
		'mean' : str
			fill with the mean value of ``im``
		'median' : str
			fill with the median value of ``im``
	
	fill_value : scalar
		the constant value to initialise the new array if ``fill_method=='constant'``

	Returns
	-------
	mean_vol : (MxNxL) array
		mean volume with the same datatype as that of the input volumes
	"""
	import skimage.io as skio 
	# work in float but convert to ubytes for allocation? 
	mean_vol = np.zeros(target_shape) 
	n_imgs = len(vol_img_list)

	for v in vol_img_list:
		im = skio.imread(v)
		im = pad_vol_2_size(im, target_shape, fill_method=fill_method, fill_value=fill_value) 
		mean_vol += im/float(n_imgs)
	
	mean_vol = mean_vol.astype(im.dtype)
	return mean_vol # cast to the same type


def max_vol_img(vol_img_list, target_shape, fill_method='constant', fill_value=0):
	r""" Utility function to find the maximum intensty volume image given a list of volumetric image files. The volumes will automatically be padded to the specified target shape. 
	
	Parameters
	----------
	vol_img_list : array of filepaths
		list of volumetric image filepaths to compute the mean image for 
	target_shape : (m,n,l) array
		desired volume shape, must be at least the size of the largest volume shape. 
	fill_method : str
		method of specifying the missing pixels, one of 

		'constant' : str
			fill with a constant value specifed by the additional argument ``fill_value``
		'mean' : str
			fill with the mean value of ``im``
		'median' : str
			fill with the median value of ``im``
	
	fill_value : scalar
		the constant value to initialise the new array if ``fill_method=='constant'``

	Returns
	-------
	max_vol : (MxNxL) array
		maximum intensity volume with the same datatype as that of the input volumes
	"""
	import skimage.io as skio  
	max_vol = np.zeros(target_shape) 

	for v in vol_img_list:
		im = skimage.imread(v)
		im = pad_vol_2_size(im, target_shape, fill_method=fill_method, fill_value=fill_value) 
		max_vol = np.maximum(max_vol, im)       
		
	return max_vol


def imadjust(vol, p1, p2): 
	r""" A crucial preprocessing for many applications is to contrast stretch the original microscopy volume intensities. This is done here through percentiles similar to Matlabs imadjust function.

	Parameters
	----------
	vol : nd array
		arbitrary n-dimensional image
	p1 : scalar
		lower percentile of intensity, specified as a number in the range 0-100. All intensities in a lower percentile than p1 will be mapped to 0.
	p2 : scalar
		upper percentile of intensity, specified as a number in the range 0-100. All intensities in the upper percentile than p2 will be mapped to the maximum value of the data type.
	
	Returns
	-------
	vol_rescale : nd array
		image of the same dimensions as the input with contrast enchanced

	"""
	import numpy as np 
	from skimage.exposure import rescale_intensity
	# this is based on contrast stretching and is used by many of the biological image processing algorithms.
	p1_, p2_ = np.percentile(vol, (p1,p2))
	vol_rescale = rescale_intensity(vol, in_range=(p1_,p2_))
	return vol_rescale

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
	r""" Linear image intensity rescaling based on the given minimum and maximum intensities, with optional clipping
	
	The new intensities will be 

	.. math::
		x_{new} = \frac{x-\text{mi}}{\text{ma}-\text{mi} + \text{eps}}
		
	Parameters
	----------
	x : nd array
		arbitrary n-dimensional image
	mi : scalar
		minimum intensity value 
	ma : scalar
		maximum intensity value 
	clip : bool
		whether to clip the transformed image intensities to [0,1] 
	eps : scalar
		small number for numerical stability 
	dtype : np.dtype
		if not None, casts the input x into the specified dtype 

	Returns
	-------
	x : nd array
		normalized output image of the same dimensionality as the input image

	"""
	if dtype is not None:
		x   = x.astype(dtype,copy=False)
		mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
		ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
		eps = dtype(eps)
	try:
		import numexpr
		x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
	except ImportError:
		x =                   (x - mi) / ( ma - mi + eps )
	if clip:
		x = np.clip(x,0,1)
	return x

def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
	r""" Percentile-based image normalization, same as :func:`unwrap3D.Image_Functions.image.imadust` but exposes more flexibility in terms of axis selection and optional clipping. 

	Parameters
	----------
	x : nd array
		arbitrary n-dimensional image
	pmin : scalar
		lower percentile of intensity, specified as a number in the range 0-100. All intensities in a lower percentile than p1 will be mapped to 0.
	pmax : scalar
		upper percentile of intensity, specified as a number in the range 0-100. All intensities in the upper percentile than p2 will be mapped to the maximum value of the data type.
	axis : int
		if not None, compute the percentiles restricted to the specified axis, otherwise they will be based on the statistics of the whole image.
	clip : bool
		whether to clip to the maximum and minimum of [0,1] for float32 or float64 data types.    
	eps : scalar
		small number for numerical stability 
	dtype : np.dtype
		if not None, casts the input x into the specified dtype 

	Returns
	-------
	x_out : nd array
		normalized output image of the same dimensionality as the input image

	"""
	mi = np.percentile(x,pmin,axis=axis,keepdims=True)
	ma = np.percentile(x,pmax,axis=axis,keepdims=True)

	x_out = normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)
	return x_out


def std_norm(im, mean=None, factor=1, cutoff=4, eps=1e-8):
	r""" Contrast normalization using the mean and standard deviation with optional clipping. 

	The intensities are transformed relative to mean :math:`\mu` and standard deviation :math:`\sigma`. The transformed pixel intensities can then be effectively gated based on properties of a Gaussian distribution. Compared to percentile normalisation this methods is easier to tune.

	.. math::
		I_{new} = \frac{I-\mu}{\text{factor}\cdot\sigma + \text{eps}}
		
	The extent of enhancement is effectively adjusted through ``factor``. If clipping is specified through ``cutoff`` the positive intensities up to ``cutoff`` is retained

	.. math::
		I_{final} &= \text{clip}(I_{new}, 0, \text{cutoff}) \\
		I_{final} &= \frac{I_{final}-\min(I_{final})}{\max(I_{final})-\min(I_{final}) + \text{eps}}


	Parameters
	----------
	im : nd array
		arbitrary n-dimensional image
	mean : scalar
		The mean intensity to subtract off. if None, it is computed as the global mean of `im`. 
	factor : scalar
		adjusts the extent of contrast enhancment. 
	cutoff : int
		if not None, clip the transformed intensities to the range [0, cutoff] and then rescale the result to [0,1]
	eps : scalar
		small number for numerical stability 
	
	Returns
	-------
	im_ : nd array
		normalized output image of the same dimensionality as the input image

	"""
	if mean is None:
		im_ = im - im.mean()
	else:
		im_ = im - mean
	im_ = im_/(factor*im.std() + eps)

	if cutoff is not None:
		im_ = np.clip(im_, 0, cutoff) # implement clipping. 
		im_ = (im_ - im_.min()) / (im_.max() - im_.min() + eps) # rescale. 
	
	return im_ 


def map_intensity_interp2(query_pts, 
						  grid_shape, 
						  I_ref, 
						  method='linear', 
						  cast_uint8=False, 
						  s=0,
						  fill_value=0):
	r""" Interpolate a 2D image at specified coordinate points. 

	Parameters
	----------
	query_pts : (n_points,2) array
		array of (y,x) image coordinates to interpolate intensites at from ``I_ref``
	grid_shape : (M,N) tuple
		shape of the image to get intensities from 
	I_ref : (M,N) array
		2D image to interpolate intensities from 
	method : str
		Interpolating algorithm. Either 'spline' to use scipy.interpolate.RectBivariateSpline or any other e.g. 'linear' from scipy.interpolate.RegularGridInterpolator
	cast_uint8 : bool
		Whether to return the result as a uint8 image. 
	s : scalar
		if method='spline', s=0 is a interpolating spline else s>0 is a smoothing spline.
	fill_value : scalar
		specifies the imputation value if any of query_pts needs to extrapolated. 
	
	Returns
	-------
	I_query : (n_points,) array
		the intensities of ``I_ref`` interpolated at the specified ``query_pts`` 

	"""
	import numpy as np 
	from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator 
	
	if method == 'spline':
		spl = RectBivariateSpline(np.arange(grid_shape[0]), 
								  np.arange(grid_shape[1]), 
								  I_ref,
								  s=s)
		I_query = spl.ev(query_pts[...,0], 
						 query_pts[...,1])
	else:
		spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
									   np.arange(grid_shape[1])), 
									   I_ref, method=method, bounds_error=False, 
									   fill_value=fill_value)
		I_query = spl((query_pts[...,0], 
					   query_pts[...,1]))

	if cast_uint8:
		I_query = np.uint8(I_query)
	
	return I_query

def scattered_interp2d(pts2D, pts2Dvals, gridsize, method='linear', cast_uint8=False, fill_value=None):
	r""" Given an unordered set of 2D points with associated intensities, use Delaunay triangulation to interpolate a 2D image of the given size. Grid points outside of the convex hull of the points will be set to the given ``fill_value`` 

	Parameters
	----------
	pts2D : (n_points,2)
		array of (y,x) image coordinates where intensites are known 
	pts2Dvals : (n_points,)
		array of corresponding image intensities at ``pts2D``
	gridsize : (M,N) tupe
		shape of the image to get intensities
	method : str
		one of the interpolatiion methods in ``scipy.interpolate.griddata``
	cast_uint8 : bool
		Whether to return the result as a uint8 image. 
	fill_value :
		specifies the imputation value if any of query_pts needs to extrapolated. 

	Returns
	-------
	img : (M,N)
		the interpolated dense image intensities of the given unordered 2D points
	
	"""
	import scipy.interpolate as scinterpolate
	import numpy as np 

	if fill_value is None:
		fill_value = np.nan

	YY, XX = np.indices(gridsize)

	img = scinterpolate.griddata(pts2D, 
								 pts2Dvals, 
								 (YY, XX), method=method, fill_value=fill_value)
	
	return img

def map_intensity_interp3(query_pts, 
							grid_shape, 
							I_ref, 
							method='linear', 
							cast_uint8=False,
							fill_value=0):
	r""" Interpolate a 3D volume image at specified coordinate points. 

	Parameters
	----------
	query_pts : (n_points,3) array
		array of (x,y,z) image coordinates to interpolate intensites at from ``I_ref``
	grid_shape : (M,N,L) tuple
		shape of the volume image to get intensities from 
	I_ref : (M,N,L) array
		3D image to interpolate intensities from 
	method : str
		Interpolating order. One of those e.g. 'linear' from scipy.interpolate.RegularGridInterpolator
	cast_uint8 : bool
		Whether to return the result as a uint8 image. 
	fill_value : scalar
		specifies the imputation value if any of query_pts needs to extrapolated. 
	
	Returns
	-------
	I_query : (n_points,) array
		the intensities of ``I_ref`` interpolated at the specified ``query_pts`` 

	"""
	# interpolate instead of discretising to avoid artifact.
	from scipy.interpolate import RegularGridInterpolator #, RBFInterpolator
	# from scipy.interpolate import griddata
	from scipy import ndimage
	import numpy as np 
	
	spl_3 = RegularGridInterpolator((np.arange(grid_shape[0]), 
									 np.arange(grid_shape[1]), 
									 np.arange(grid_shape[2])), 
									 I_ref, method=method, bounds_error=False, 
									 fill_value=fill_value)

	I_query = spl_3((query_pts[...,0], 
					  query_pts[...,1],
					  query_pts[...,2]))
	
	if cast_uint8:
		I_query = np.uint8(I_query)
	
	return I_query

