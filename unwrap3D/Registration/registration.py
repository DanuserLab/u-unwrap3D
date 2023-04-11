"""

This module implements various different registration algorithms

"""
	
# these that are commented out should now be incorporated into individual functions below 
# from ..Utility_Functions import file_io as fio
# from ..Visualisation import imshowpair as imshowpair
# from ..Geometry import geometry as geom
# from transforms3d.affines import decompose44, compose
import numpy as np 

def matlab_affine_register(fixed_file, moving_file, save_file, reg_config, multiscale=False):
	r""" Use Python-Matlab interface to conduct affine registration. The transformed image is saved to file to bypass speed bottleneck in transferring Matlab to Python

	Parameters
	----------
	fixed_file : filepath
		the filepath of the image to be used as reference for registration.
	moving_file : filepath
		the filepath of the image to be registered to the fixed image. 
	save_file : filepath
		the output filepath to save the registered image of the moving_file
	reg_config : dict
		Python dict specifying the parameters for running the registration. See :func:`unwrap3D.Parameters.params.affine_register_matlab` for the parameters required to be set. 
	multiscale : bool
		if True, do multiscale registration given by the ``register3D_intensity_multiscale`` script, otherwise run the ``register3D_rigid_faster`` script.

	Returns
	-------
	transform : 4x4 array
		the forward homogeneous transformation matrix to map fixed to moving. 
	im1 : array
		return the fixed image as numpy array from fixed_file if ``reg_config['return_img']!=1``. if ``reg_config['return_img']==1``, the transform is directly applied within Matlab and the result saved to save_file
	im2_ : array
		return the transformed moving image after applying the transform as a numpy array if ``reg_config['return_img']!=1``. if ``reg_config['return_img']==1``, the transform is directly applied within Matlab and the result saved to save_file
	"""
	import matlab.engine
	import scipy.io as spio 
	import skimage.io as skio 
	from transforms3d.affines import decompose44, compose
	from ..Geometry import geometry as geom

	eng = matlab.engine.start_matlab() 
	
	if reg_config['view'] is not None:

		# use initialisation. 
		initial_tf = reg_config['initial_tf']

		# decompose the initialisation to comply with Matlab checks. # might be better ways to do this?
		T,R,Z,S = decompose44(initial_tf) # S is shear!
		
		if reg_config['type'] == 'similarity':
			# initialise with just rigid instead -> checks on this normally are too stringent.
			affine = compose(T, R, np.ones(3), np.zeros(3)) # the similarity criteria too difficult for matlab 
		if reg_config['type'] == 'rigid':
			affine = compose(T, R, np.ones(3), np.zeros(3))
		if reg_config['type'] == 'translation':
			affine = compose(T, np.eye(3), np.ones(3), np.zeros(3))
		if reg_config['type'] == 'affine':
			affine = initial_tf.copy()
		
		# save the tform as temporary for matlab to read. 
		spio.savemat('tform.mat', {'tform':affine}) # transpose for matlab 
		
		if multiscale == False:
			transform = eng.register3D_rigid_faster(str(fixed_file), str(moving_file), str(save_file), 
													'tform.mat', 1, reg_config['downsample'], 
													reg_config['modality'], reg_config['max_iter'], 
													reg_config['type'], 
													reg_config['return_img'], 
													nargout=1) 
		else:
			transform = eng.register3D_intensity_multiscale(str(fixed_file), str(moving_file), str(save_file), 
													'tform.mat', 1, reg_config['downsample'], 
													reg_config['modality'], reg_config['max_iter'], 
													reg_config['type'], 
													reg_config['return_img'], 
													nargout=1)
	else:
		if multiscale == False:
			transform = eng.register3D_rigid_faster(str(fixed_file), str(moving_file), str(save_file), 
													'tform.mat', 0, reg_config['downsample'], 
													reg_config['modality'], reg_config['max_iter'], 
													reg_config['type'], 
													reg_config['return_img'], 
													nargout=1)    
		else:
			# print('multiscale')
			# convert to matlab arrays. 
			levels = matlab.double(reg_config['downsample'])
			warps = matlab.double(reg_config['max_iter'])
			transform = eng.register3D_intensity_multiscale(str(fixed_file), str(moving_file), str(save_file), 
													'tform.mat', 0, levels, 
													reg_config['modality'], warps, 
													reg_config['type'], 
													reg_config['return_img'], 
													nargout=1)
	transform = np.asarray(transform)
	
	if reg_config['return_img']!= 1:
		# then the transform is applied within python 
		im1 = skio.imread(fixed_file)
		im2 = skio.imread(moving_file)
		im2_ = np.uint8(geom.apply_affine_tform(im2.transpose(2,1,0), 
						np.linalg.inv(transform), 
						np.array(im1.shape)[[2,1,0]]))
		im2_ = im2_.transpose(2,1,0) # flip back
	
		return transform, im1, im2_
	else:
		return transform 
		

# add multiscale capability 
def matlab_group_register_batch(dataset_files, ref_file, in_folder, out_folder, reg_config, reset_ref_steps=0, multiscale=True, debug=False):
	
	r""" Temporal affine registration of an entire video using Matlab call.

	Parameters
	----------
	dataset_files : list of files
		list of img files arranged in increasing temporal order
	ref_file : str
		filepath of the initial reference file for registration. 
	in_folder : str
		specifies the parent folder of the dataset_files. This string is replaced with the out_folder string in order to generate the save paths for the registered timepoints.
	reg_config : dict
		Python dict specifying the parameters for running the registration. See :func:`unwrap3D.Parameters.params.affine_register_matlab` for the parameters required to be set. 
	reset_ref_steps : int
		if reset_ref_steps > 0, uses every nth registered file as the reference. default of 0 = fixed reference specified by ref_file, whilst 1 = sequential    
	multiscale : bool
		if True, do multiscale registration given by the ``register3D_intensity_multiscale`` script, otherwise run the ``register3D_rigid_faster`` script.
	debug : bool
		if True, will plot for every timepoint the 3 mid-slices in x-y, y-z, x-z to qualitatively check registration. Highly recommended for checking parameters.  
	
	Returns
	-------
	tforms : array
		the compiled 4x4 transformation matrices between all successive timepoints
	tformfile : str
		the filepath to the saved transforms 

	"""
	import matlab.engine
	import scipy.io as spio 
	import os
	import shutil
	import pylab as plt 
	from tqdm import tqdm
	import skimage.io as skio 
	from ..Visualisation import imshowpair as imshowpair
	from ..Geometry import geometry as geom
	from ..Utility_Functions import file_io as fio

	# start matlab engine. 
	eng = matlab.engine.start_matlab() 
	fio.mkdir(out_folder) # check output folder exists. 
	
	tforms = []
	ref_files = []
	
	# difference is now fixed_file is always the same one. 
	fixed_file = ref_file
	all_save_files = np.hstack([f.replace(in_folder, out_folder) for f in dataset_files])
	
	for i in tqdm(range(len(dataset_files))):
	
		moving_file = dataset_files[i]
		save_file = all_save_files[i]
		
		if multiscale:
			# print('multiscale')
			levels = matlab.double(reg_config['downsample'])
			warps = matlab.double(reg_config['max_iter'])
			transform = eng.register3D_intensity_multiscale(str(fixed_file), str(moving_file), str(save_file), 
													'tform.mat', 0, levels, 
													reg_config['modality'], warps, 
													reg_config['type'], 
													reg_config['return_img'], 
													nargout=1)
		else:
			transform = eng.register3D_rigid_faster(str(fixed_file), str(moving_file), str(save_file), 
											'tform.mat', 0, reg_config['downsample'], 
											reg_config['modality'], reg_config['max_iter'], 
											reg_config['type'], 
											reg_config['return_img'], 
											nargout=1)        
		
		transform = np.asarray(transform) # (z,y,x) 
		tforms.append(transform)
		ref_files.append(fixed_file) # which is the reference used. 

#        if reg_config['return_img'] != 1: # this is too slow and disabled. 
		im1 = skio.imread(fixed_file)
		im2 = skio.imread(moving_file)
		
		im2_ = np.uint8(geom.apply_affine_tform(im2.transpose(2,1,0), np.linalg.inv(transform), sampling_grid_shape=np.array(im1.shape)[[2,1,0]]))
		im2_ = im2_.transpose(2,1,0) # flip back

		# fio.save_multipage_tiff(im2_, save_file)
		skio.imsave(save_file, im2_)

		if debug:

			# visualize all three possible cross sections for reference and checking.
			fig, ax = plt.subplots(nrows=1, ncols=2)
			imshowpair(ax[0], im1[im1.shape[0]//2], im2[im2.shape[0]//2]) 
			imshowpair(ax[1], im1[im1.shape[0]//2], im2_[im2_.shape[0]//2])
			
			fig, ax = plt.subplots(nrows=1, ncols=2)
			imshowpair(ax[0], im1[:,im1.shape[1]//2], im2[:,im2.shape[1]//2])
			imshowpair(ax[1], im1[:,im1.shape[1]//2], im2_[:,im2_.shape[1]//2])

			fig, ax = plt.subplots(nrows=1, ncols=2)
			imshowpair(ax[0], im1[:,:,im1.shape[2]//2], im2[:,:,im2.shape[2]//2])
			imshowpair(ax[1], im1[:,:,im1.shape[2]//2], im2_[:,:,im2_.shape[2]//2])

			plt.show()
		
		if reset_ref_steps > 0:
			# change the ref file.
			if np.mod(i+1, reset_ref_steps) == 0: 
				fixed_file = save_file # change to the file you just saved. 
		
	# save out tforms into a .mat file.
	tformfile = os.path.join(out_folder, 'tforms-matlab.mat')
	tforms = np.array(tforms)
	ref_files = np.hstack(ref_files)
	
	spio.savemat(tformfile, {'tforms':tforms,
							 'ref_files':ref_files,
							 'in_files':dataset_files,
							 'out_files':all_save_files,
							 'reg_config':reg_config})
			
	return tforms, tformfile
	
		
def COM_2d(im1, im2):
	r""" Compute the center of mass between two images, im1 and im2 based solely on binary Otsu Thresholding. This can be used to compute the rough translational displacement.
	
	Parameters
	----------
	im1 : array
		n-dimension image 1
	im2 : array
		n-dimension image 2
	
	Returns
	-------
	com1 : array
		n-dimension center of mass coordinates of image 1
	com2 : 
		n-dimension center of mass coordinates of image 2

	"""
	from scipy.ndimage.measurements import center_of_mass
	from scipy.ndimage.morphology import binary_fill_holes
	from skimage.filters import threshold_otsu
	
	mask1 = binary_fill_holes(im1>=threshold_otsu(im1))
	mask2 = binary_fill_holes(im2>=threshold_otsu(im2))

	com1 = center_of_mass(mask1)
	com2 = center_of_mass(mask2)

	return com1, com2 
	
	
# function to join two volume stacks 
def simple_join(stack1, stack2, cut_off=None, blend=True, offset=10, weights=[0.7,0.3]):
	r""" Joins two volumetric images, stack1 and stack2 in the first dimension by direct joining or linear weighting constrained to +/- offset to the desired ``cut_off`` slice number.
	
	The constrained linear blending within the interval :math:`[x_c-\Delta x, x_c + \Delta x]` is given by 

	.. math::
		I = 
		\Biggl \lbrace 
		{ 
		w_2\cdot I_1 + w_1\cdot I_2 ,\text{ if } 
		  {x\in [x_c, x_c + \Delta x]}
		\atop 
		w_1\cdot I_1 + w_2\cdot I_2, \text{ if } 
		  {x\in [x_c-\Delta x, x_c]}
		}
		
	where :math:`x_c` denotes the ``cut_off`` point. The weight reversal enables asymmetric weights to be define once but used twice. 

	Parameters
	----------
	stack1 : (MxNxL) array
		volume image whose intensities will dominant in the first dimension when < ``cut_off``
	stack2 : (MxNxL) array
		volume image whose intensities will dominant in the first dimension when > ``cut_off``
	cut_off : int
		the slice number of the first dimension where the transition from stack2 to stack1 will occur
	blend : bool
		if blend==True, the join occurs directly at the cut_off point with simple array copying and there is no blending. If True then the linear blending as described occurs around the cut_off point. 
	offset : int
		the +/- slice numbers around the cut_off to linearly blend the intensities of stack1 and stack2 for a smoother effect.
	weights : 2 vector array
		:math:`w_1=` ``weights[0]`` and :math:`w_2=` ``weights[1]``. The sum of weights should be 1. 

	Returns
	-------
	the blended image of the same image dimension  

	"""
	if cut_off is None:
		cut_off = len(stack1) // 2

	if blend:
		combined_stack = np.zeros_like(stack1)
		combined_stack[:cut_off-offset] = stack1[:cut_off-offset]
		combined_stack[cut_off-offset:cut_off] = weights[0]*stack1[cut_off-offset:cut_off]+weights[1]*stack2[cut_off-offset:cut_off]
		combined_stack[cut_off:cut_off+offset] = weights[1]*stack1[cut_off:cut_off+offset]+weights[0]*stack2[cut_off:cut_off+offset]
		combined_stack[cut_off+offset:] = stack2[cut_off+offset:]         
	else:
		combined_stack = np.zeros_like(stack1)
		combined_stack[:cut_off] = stack1[:cut_off] 
		combined_stack[cut_off:] = stack2[cut_off:] 
	
	return combined_stack


def sigmoid_join(stack1, stack2, cut_off=None, gradient=200, shape=1, debug=False):
	r""" Joins two volumetric images, stack1 and stack2 in the first dimension using a sigmoid type blending function. Both stacks are assumed to be the same size. 
	
	The sigmoid has the following mathematical expression 

	.. math::
		w_2 &= \frac{1}{\left( 1+e^{-\text{grad} (x - x_c)} \right)^{1/\text{shape}}} \\
		w_1 &= 1 - w_2
		
	where :math:`x_c` denoting the ``cut_off``.
	
	The blended image is then the weighted sum 

	.. math::
		I_{blend} = w_1\cdot I_1 + w_2\cdot I_2

	see https://en.wikipedia.org/wiki/Generalised_logistic_function

	Parameters
	----------
	stack1 : (MxNxL) array
		volume image whose intensities will dominant in the first dimension when < ``cut_off``
	stack2 : (MxNxL) array
		volume image whose intensities will dominant in the first dimension when > ``cut_off``
	cut_off : int
		the slice number of the first dimension where the transition from stack2 to stack1 will occur
	gradient : scalar
		controls the sharpness of the transition between the two stacks 
	shape : scalar
		controls the shape of the sigmoid, in particular introduces asymmetry in the shape
	debug : bool
		if True, will use Matplotlib to plot the computed sigmoid weights for checking.   

	Returns
	-------
	the blended image of the same image dimension  

	"""
	if cut_off is None:
		cut_off = len(stack1) // 2

	def generalised_sigmoid( stack1, cut_off=cut_off, shape=shape, grad=gradient):
		
		x = np.arange(0,stack1.shape[0])
		weights2 = 1./((1+np.exp(-grad*(x - cut_off)))**(1./shape))
		weights1 = 1. - weights2
		
		return weights1, weights2
	
	weights1, weights2 = generalised_sigmoid( stack1, cut_off=cut_off, shape=shape, grad=gradient)
	if debug:
		import pylab as plt 
		plt.figure()
		plt.plot(weights1, label='1')
		plt.plot(weights2, label='2')
		plt.legend()
		plt.show()
		
	return stack1*weights1[:,None,None] + stack2*weights2[:,None,None]


def nonregister_3D_demons_matlab(infile1, infile2, savefile, savetransformfile, reg_config):
	r""" This function uses Matlab's imregdemons function to run Demon's registration. The result is directly saved to disk.
	
	Parameters
	----------
	infile1 : str
		filepath of the reference static image
	infile2 : str
		filepath of the moving image to register to the fixed image
	savefile : str
		filepath to save the transformed moving image to 
	savetransformfile : str
		filepath to save the displacement field to 
	reg_config : dict
		parameters to pass to imregdemons. see :func:`unwrap3D.Parameters.params.demons_register_matlab` 

	Returns
	-------
	return_val : int
		return value of 1 if Matlab execution was successful and completed
	"""
	import matlab.engine
	eng = matlab.engine.start_matlab()
	
	print(reg_config['level'], reg_config['warps'])
	
#    alpha = matlab.single(reg_config['alpha'])
#    level = matlab.double(reg_config['level'])
	level = float(reg_config['level'])
	warps = matlab.double(reg_config['warps'])
	smoothing = float(reg_config['alpha'])
	
	# add in additional options for modifying. 
	return_val = eng.nonrigid_register3D_demons(str(infile1), str(infile2), str(savefile), 
										str(savetransformfile),
										level, warps, smoothing)
		
	return return_val  


def warp_3D_demons_tfm(infile, savefile, transformfile, downsample, direction=1):
	r""" This function warps the input image file according to the deformation field specified in transformfile and saves the result in savefile. 
	If direction == 1 warp in the same direction else if direction == -1 in the reverse direction.
	
	Parameters
	----------
	infile : str 
		filepath of the image to transform 
	savefile : str
		filepath of transformed result
	transformfile : str
		filepath to the displacement field transform 
	downsample : int
		this parameter specifies the highest level of downsampling the displacement field came from i.e. if the displacement field is the same size as the image then downsample=1. If the displacement field was generated at a scale 4x smaller than the input image then specify downsample=4 here.
		The displacment field will be rescaled by this factor before applying to the input image.
	direction : 1 or -1 
		if 1, warps the image in the same direction of the diplacement field. If -1 warps the image in the reverse direction of the displacement field

	Returns
	-------
	return_val : int
		if 1, the process completed successfully in Matlab 
	"""
	import matlab.engine
	eng = matlab.engine.start_matlab()

	return_val = eng.warp_3D_demons(str(infile), str(savefile), str(transformfile), int(downsample), direction)

	return return_val


def warp_3D_demons_matlab_tform_scipy(im, tformfile, direction='F', pad=20):
	r""" Python-based warping of Matlab imregdemons displacement fields adjusting for the difference in coordinate convention.

	Parameters
	----------
	im : (MxNxL) array
		input 3D volumetric image to warp
	tformfile : str 
		filepath to the displacement field transform 
	direction : 'F' or 'B'
		if 'F' warps im in the forwards direction specified by the displacement field or if 'B' warps im in the backwards direction.
	pad : int
		optional isotropic padding applied to every image dimension before warping

	Returns
	-------
	im_interp : (MxNxL) array
		the warped 3D volumetric image

	"""
	import numpy as np 
	from scipy.ndimage import map_coordinates
	from ..Utility_Functions import file_io as fio
	
	dx,dy,dz = fio.read_demons_matlab_tform(tformfile, im.shape)
	# print(dx.max(), dx.min())
	im_ = im.copy()
	if pad is not None:
		im_ = np.pad(im_, [[pad,pad] for i in range(len(im_.shape))], mode='constant') # universally pad the volume by the edge values. 
		dx = np.pad(dx, [[pad,pad] for i in range(len(dx.shape))], mode='constant') # universally pad the volume by the edge values. 
		dy = np.pad(dy, [[pad,pad] for i in range(len(dy.shape))], mode='constant') # universally pad the volume by the edge values. 
		dz = np.pad(dz, [[pad,pad] for i in range(len(dz.shape))], mode='constant') # universally pad the volume by the edge values. 

	im_interp = warp_3D_displacements_xyz(im_, dx, dy, dz, direction=direction)
	
	return im_interp


def warp_3D_displacements_xyz(im, dx, dy, dz, direction='F', ret_xyz=False, mode='nearest'):
	r""" Warps an image with a 3D displacement field specified by arrays, optionally returning the corresponding coordinates where the image intensities were interpolated from in the image

	Parameters
	----------
	im : (MxNxL) array
		the input 3D volume image
	dx : (MxNxL) array
		the displacements in x-direction (first image axis)
	dy : (MxNxL) array
		the displacements in y-direction (second image axis)
	dz : (MxNxL) array
		the displacements in z-direction (third image axis)
	direction : 'F' or 'B'
		if 'F' warp image in the same direction as specified by (dx,dy,dz) or if 'B' warp image in the opposite direction of (-dx,-dy,-dz)
	ret_xyz : bool
		if True, return as a second output the x-, y- and z- coordinates in ``im`` where ``im_interp`` intensities were interpolated at
	mode : str
		 one of the boundary modes specified as ``mode`` argument in `scipy.map_coordinates <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html>`_

	Returns
	-------
	im_interp : (MxNxL) array
		the transformed image 
	[XX, YY, ZZ] : list of arrays
		the x-, y- and z- coordinates of same array shape as im where ``im_interp`` intensities were taken from ``im`` 

	"""    
	from scipy.ndimage import map_coordinates
	import numpy as np 
	
	XX, YY, ZZ = np.indices(im.shape) # set up the interpolation grid.
	
	if direction == 'F':
		XX = XX + dx
		YY = XX + dy 
		ZZ = ZZ + dz
	if direction == 'B':
		XX = XX - dx
		YY = YY - dy
		ZZ = ZZ - dz
	
	im_interp = map_coordinates(im, 
								np.vstack([(XX).ravel().astype(np.float32), 
										   (YY).ravel().astype(np.float32), 
										   (ZZ).ravel().astype(np.float32)]), prefilter=False, order=1, mode=mode) # note the mode here refers to the boundary handling.
	im_interp = im_interp.reshape(im.shape)
	
	if ret_xyz:
		return im_interp, [XX,YY,ZZ]
	else:
		return im_interp


def warp_3D_displacements_xyz_pts(pts3D, dxyz, direction='F'):
	r""" Convenience function for warping forward or backward a 3D point cloud given the displacement field 

	If direction == 'F', the new points, :math:`(x',y',z')` are

	.. math::
		(x',y',z') = (x+dx, y+dy, z+dz)

	If direction == 'B', the new points, :math:`(x',y',z')` are

	.. math::
		(x',y',z') = (x-dx, y-dy, z-dz)

	Parameters
	----------
	pts3D : array
		n-dimensional array with last dimension of size 3 for (x,y,z)
	dxyz : array 
		n-dimensional array with last dimension of size 3 for (dx,dy,dz)
	direction : 'F' or 'B'
		if 'F' warp points in the same direction as specified by (dx,dy,dz) or if 'B' warp image in the opposite direction of (-dx,-dy,-dz)
	
	Returns
	-------
	pts3D_warp : array
		n-dimensional array with last dimension of size 3 for the new (x,y,z) coordinates after warping

	"""    
	if direction == 'F':
		pts3D_warp = pts3D + dxyz
	if direction == 'B':
		pts3D_warp = pts3D - dxyz
	
	return pts3D_warp

def warp_3D_transforms_xyz(im, tmatrix, direction='F', ret_xyz=False):
	r""" Warps a volume image with the given affine transformation matrix forward or reverse with trilinear interpolation, returning if specified the corresponding (x,y,z) grid points

	Parameters
	----------
	im : (MxNxL) array
		the input 3D volume image
	tmatrix : (4x4) array 
		the homogeneous affine 3D transformation matrix 
	direction : 'F' or 'B'
		if 'F' warp image with tmatrix or if 'B' warp image with the matrix inverse of tmatrix
	ret_xyz : bool
		if True, return as a second output the xyz- coordinates in ``im`` where ``im_interp`` intensities were interpolated at
	
	Returns
	-------
	im_interp : (MxNxL) array
		the transformed image 
	xyz_ : list of arrays
		the xyz- coordinates of same array shape as im where ``im_interp`` intensities were taken from ``im`` 

	"""    
	import numpy as np 
	from scipy.ndimage import map_coordinates

	XX, YY, ZZ = np.indices(im.shape) # set up the interpolation grid.
	
	xyz = np.vstack([(XX).ravel().astype(np.float32), 
					 (YY).ravel().astype(np.float32), 
					 (ZZ).ravel().astype(np.float32),
					 np.ones(len(ZZ.ravel()), dtype=np.float32)])
	
	if direction == 'F':
		xyz_ = tmatrix.dot(xyz)
	if direction == 'B':
		xyz_ = (np.linalg.inv(tmatrix)).dot(xyz)
	
	im_interp = map_coordinates(im, 
								xyz_[:3], prefilter=False, order=1, mode='nearest')
	im_interp = im_interp.reshape(im.shape)

	# return also the coordinates that it maps to 
	if ret_xyz:
		return im_interp, xyz_
	else:
		return im_interp

def warp_3D_transforms_xyz_pts(pts3D, tmatrix, direction='F'):
	r""" Convenience function for warping forward or backward a 3D point cloud given an affine transform matrix 

	Parameters
	----------
	pts3D : array
		n-dimensional array with last dimension of size 3 for (x,y,z)
	tmatrix : (4x4) array 
		the homogeneous affine 3D transformation matrix 
	direction : 'F' or 'B'
		if 'F' warp points with ``tmatrix`` or if 'B' warp points with the reverse transformation given by``np.linalg.inv(tmatrix)``, the matrix inverse
	
	Returns
	-------
	pts3D_warp : array
		n-dimensional array with last dimension of size 3 for the new (x,y,z) coordinates after warping

	"""    
	import numpy as np 
	# first make homogeneous coordinates.
	xyz = np.vstack([(pts3D[...,0]).ravel().astype(np.float32), 
					 (pts3D[...,1]).ravel().astype(np.float32), 
					 (pts3D[...,2]).ravel().astype(np.float32),
					 np.ones(len(pts3D[...,0].ravel()), dtype=np.float32)])
	
	if direction == 'F':
		xyz_ = tmatrix.dot(xyz)
	if direction == 'B':
		xyz_ = (np.linalg.inv(tmatrix)).dot(xyz)
	
	pts3D_warp = (xyz_[:3].T).reshape(pts3D.shape)

	return pts3D_warp


def warp_3D_transforms_xyz_similarity(im, translation=[0,0,0], rotation=np.eye(3), zoom=[1,1,1], shear=[0,0,0], center_tform=True, direction='F', ret_xyz=False, pad=50):
	r""" Warps a volume image forward or reverse  where the transformation matrix is explicitly given by the desired translation, rotation, zoom and shear matrices, returning if specified the corresponding (x,y,z) grid points and optional padding

	Parameters
	----------
	im : (MxNxL) array
		the input 3D volume image
	translation : (3,) array 
		the global (dx,dy,dz) translation vector
	rotation : (3x3) array 
		the desired rotations given as a rotation matrix
	zoom : (3,) array
		the independent scaling factor in x-, y-, z- directions 
	shear : (3,) array
		the shearing factor in x-, y-, z- directions
	center_tform : bool
		if true, the transformation will be applied preserving ``im_center`` 
	direction : 'F' or 'B'
		if 'F' warp points with the specified transformation or if 'B' warp points by the reverse transformation
	ret_xyz : bool
		if True, return as a second output the xyz- coordinates in ``im`` where ``im_interp`` intensities were interpolated at
	pad : int 
		if not None, the input image is prepadded with the same number of pixels given by ``pad`` in all directions before transforming 
	
	Returns
	-------
	im_interp : (MxNxL) array
		the transformed image 
	xyz_ : list of arrays
		the xyz- coordinates of same array shape as im where ``im_interp`` intensities were taken from ``im`` 

	"""    
	"""
	This function is mainly to test how to combine the coordinates with transforms. 

	pad enables an optional padding of the image. 
	"""
	from scipy.ndimage import map_coordinates
	from transforms3d.affines import compose
	import numpy as np 

	# compose the 4 x 4 homogeneous matrix. 
	tmatrix = compose(translation, rotation, zoom, shear)

	im_ = im.copy()

	if pad is not None:
		im_ = np.pad(im_, [[pad,pad] for i in range(len(im.shape))], mode='constant') # universally pad the volume by the edge values. 

	if center_tform:
		im_center = np.array(im_.shape)//2
		tmatrix[:-1,-1] = tmatrix[:-1,-1] + np.array(im_center)
		decenter = np.eye(4); decenter[:-1,-1] = -np.array(im_center)
		tmatrix = tmatrix.dot(decenter)

	XX, YY, ZZ = np.indices(im_.shape) # set up the interpolation grid.
	
	xyz = np.vstack([(XX).ravel().astype(np.float32), 
					 (YY).ravel().astype(np.float32), 
					 (ZZ).ravel().astype(np.float32),
					 np.ones(len(ZZ.ravel()), dtype=np.float32)])
	
	if direction == 'F':
		xyz_ = tmatrix.dot(xyz)
	if direction == 'B':
		xyz_ = (np.linalg.inv(tmatrix)).dot(xyz)
	
	# needs more memory % does this actually work? 
	im_interp = map_coordinates(im_, 
								xyz_[:3], prefilter=False, order=1, mode='nearest')
	im_interp = im_interp.reshape(im_.shape)
	
	if ret_xyz:
		# additional return of the target coordinates. 
		return im_interp, xyz_
	else:
		return im_interp


def warp_3D_transforms_xyz_similarity_pts(pts3D, translation=[0,0,0], rotation=np.eye(3), zoom=[1,1,1], shear=[0,0,0], center_tform=True, im_center=None, direction='F'):
	r""" Convenience function for warping forward or backward a 3D point cloud where  the transformation matrix is explicitly given by the desired translation, rotation, zoom and shear matrices  

	Parameters
	----------
	pts3D : array
		n-dimensional array with last dimension of size 3 for (x,y,z)
	translation : (3,) array 
		the global (dx,dy,dz) translation vector
	rotation : (3x3) array 
		the desired rotations given as a rotation matrix
	zoom : (3,) array
		the independent scaling factor in x-, y-, z- directions 
	shear : (3,) array
		the shearing factor in x-, y-, z- directions
	center_tform : bool
		if true, the transformation will be applied preserving ``im_center`` 
	im_center : (3,) array
		if None, the mean of ``pt3D`` in x-, y-, z- is taken as ``im_center`` and takes effect only if ``center_tform=True``
	direction : 'F' or 'B'
		if 'F' warp points with the specified transformation or if 'B' warp points by the reverse transformation
	
	Returns
	-------
	pts3D_warp : array
		n-dimensional array with last dimension of size 3 for the new (x,y,z) coordinates after warping

	"""    
	from transforms3d.affines import compose 
	
	# compose the 4 x 4 homogeneous matrix. 
	tmatrix = compose(translation, rotation, zoom, shear)

	if center_tform:
		if im_center is None:
			im_center = np.hstack([np.nanmean(pts3D[...,ch]) for ch in pts3D.shape[-1]])
		# im_center = np.array(im.shape)//2
		tmatrix[:-1,-1] = tmatrix[:-1,-1] + np.array(im_center)
		decenter = np.eye(4); decenter[:-1,-1] = -np.array(im_center)
		tmatrix = tmatrix.dot(decenter)

	# first make homogeneous coordinates.
	xyz = np.vstack([(pts3D[...,0]).ravel().astype(np.float32), 
					 (pts3D[...,1]).ravel().astype(np.float32), 
					 (pts3D[...,2]).ravel().astype(np.float32),
					 np.ones(len(pts3D[...,0].ravel()), dtype=np.float32)])
	
	if direction == 'F':
		xyz_ = tmatrix.dot(xyz)
	if direction == 'B':
		xyz_ = (np.linalg.inv(tmatrix)).dot(xyz)
	
	pts3D_warp = (xyz_[:3].T).reshape(pts3D.shape)

	return pts3D_warp

	
def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
	""" Utility function used in :func:`unwrap3D.Registration.registration.multiscale_demons` for generating multiscale image pyramids for registration based on the SimpleITK library

	see SimpleITK notebooks, https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html

	Parameters
	----------
	image : SimpleITK image
		The image we want to resample.
	shrink_factors : scalar or array
		Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
	smoothing_sigma(s): scalar or array
		Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.

	Returns
	-------
	image_resample : SimpleITK image
		Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
	"""
	import SimpleITK as sitk
	import numpy as np 

	if np.isscalar(shrink_factors):
		shrink_factors = [shrink_factors]*image.GetDimension()
	if np.isscalar(smoothing_sigmas):
		smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()

	smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
	
	original_spacing = image.GetSpacing()
	original_size = image.GetSize()
	new_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(shrink_factors,original_size)]
	new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
				   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]

	image_resample = sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
						 sitk.sitkLinear, image.GetOrigin(),
						 new_spacing, image.GetDirection(), 0.0, 
						 image.GetPixelID())
	
	return image_resample
	

def multiscale_demons(registration_algorithm,
					  fixed_image, moving_image, initial_transform = None, 
					  shrink_factors=None, smoothing_sigmas=None):
	r""" Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
	original images are implicitly incorporated as the base of the pyramid.
	
	We use the algorithm here to run demons registration in multiscale hence the name.

	see SimpleITK notebooks, https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html

	Parameters
	----------
	registration_algorithm : SimpleITK algorithm instance
		Any registration algorithm in `SimpleITK <https://simpleitk.readthedocs.io/en/master/filters.html>`_ that has an Execute(fixed_image, moving_image, displacement_field_image) method.
	fixed_image: SimpleITK image
		This is the reference image. Resulting transformation maps points from this image's spatial domain to the moving image spatial domain. 
	moving_image : SimpleITK image
		This is the image to register to the fixed image. Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
	initial_transform : SimpleITK transform
		Any SimpleITK transform, used to initialize the displacement field.
	shrink_factors : list of lists or scalars 
		Shrink factors relative to the original image's size. When the list entry, shrink_factors[i], is a scalar the same factor is applied to all axes.
		When the list entry is a list, shrink_factors[i][j] is applied to axis j. This allows us to specify different shrink factors per axis. This is useful
		in the context of microscopy images where it is not uncommon to have unbalanced sampling such as a 512x512x8 image. In this case we would only want to 
		sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
	smoothing_sigmas : list of lists or scalars  
		Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These are in physical (image spacing) units.

	Returns
	-------
	SimpleITK.DisplacementFieldTransform

	"""
	import SimpleITK as sitk
	import numpy as np 

	# Create image pyramid in a memory efficient manner using a generator function.
	# The whole pyramid never exists in memory, each level is created when iterating over
	# the generator.
	def image_pair_generator(fixed_image, moving_image, shrink_factors, smoothing_sigmas):
		end_level = 0
		start_level = 0
		if shrink_factors is not None:
			end_level = len(shrink_factors)
		for level in range(start_level, end_level):
			f_image = smooth_and_resample(fixed_image, shrink_factors[level], smoothing_sigmas[level])
			m_image = smooth_and_resample(moving_image, shrink_factors[level], smoothing_sigmas[level])
			yield(f_image, m_image)
		yield(fixed_image, moving_image)
	
	# Create initial displacement field at lowest resolution. 
	# Currently, the pixel type is required to be sitkVectorFloat64 because 
	# of a constraint imposed by the Demons filters.
	if shrink_factors is not None:
		original_size = fixed_image.GetSize()
		original_spacing = fixed_image.GetSpacing()
		s_factors =  [shrink_factors[0]]*len(original_size) if np.isscalar(shrink_factors[0]) else shrink_factors[0]
		df_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(s_factors,original_size)]
		df_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
					  for original_sz, original_spc, new_sz in zip(original_size, original_spacing, df_size)]
	else:
		df_size = fixed_image.GetSize()
		df_spacing = fixed_image.GetSpacing()
 
	if initial_transform:
		initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
																	   sitk.sitkVectorFloat64,
																	   df_size,
																	   fixed_image.GetOrigin(),
																	   df_spacing,
																	   fixed_image.GetDirection())
	else:
		initial_displacement_field = sitk.Image(df_size, sitk.sitkVectorFloat64, fixed_image.GetDimension())
		initial_displacement_field.SetSpacing(df_spacing)
		initial_displacement_field.SetOrigin(fixed_image.GetOrigin())
 
	# Run the registration.            
	# Start at the top of the pyramid and work our way down.    
	for f_image, m_image in image_pair_generator(fixed_image, moving_image, shrink_factors, smoothing_sigmas):
		initial_displacement_field = sitk.Resample (initial_displacement_field, f_image)
		initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
	return sitk.DisplacementFieldTransform(initial_displacement_field)


def SITK_multiscale_affine_registration(vol1, vol2, 
										imtype=16,
										p12=None,
										rescale_intensity=True,  
										centre_tfm_model='geometry', 
										tfm_type = 'rigid', 
										metric='Matte',  
										metric_numberOfHistogramBins=16,
										sampling_intensity_method='random',
										MetricSamplingPercentage=0.1,
										shrink_factors = [1], 
										smoothing_sigmas = [0],
										optimizer='gradient', 
										optimizer_params=None, 
										eps=1e-12):
	r""" Main function to affine register two input volumes using SimpleITK library in a multiscale manner. 
	
	Affine registration includes the following transformations

	- rigid (rotation + translation)
	- iso_similarity (isotropic scale + rigid)
	- aniso_similarity (anisotropic scale + rigid)
	- affine (skew + aniso_similarity)

	see SimpleITK notebooks, https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/61_Registration_Introduction_Continued.html

	Parameters
	----------
	vol1 : (MxNxL) array
		reference volume given as a Numpy array
	vol2 : (MxNxL) array
		volume to register to vol2, given as a Numpy array
	imtype : int
		specifies the -bit of the image e.g. 8-bit, 16-bit etc. Used to normalize the intensity if ``rescale_intensity=False``
	p12 : (p1,p2) tuple
		if provided, vol1 and vol2 are constract enhanced using percentage intensity normalization 
	rescale_intensity : bool
		if True, does min-max scaling of image intensities in the two volumes independently before registering. This follows percentage intensity normalization if specified. 
	centre_tfm_model : str
		specifies the initial centering model. This is one of two options. For microscopy, we find 'geometry' has less jitter and is best. 
	tfm_type : str
		Type of affine transform from 4. The more complex, the longer the registration. 

		'rigid' :
			rotation + translation
		'iso_similarity' : 
			isotropic scale + rotation + translation 
		'aniso_similarity' :
		 	anisotropic scale + rotation + translation 
		'affine' : 
			skew + scale + rotation + translation 
	
	metric : str 
		This parameter is currently not used. Matte's mutual information ('Matte') is used by default as it offers the best metric for fluorescence imaging and multimodal registration.  
	metric_numberOfHistogramBins : int,
		The number of histogram bins to compute mutual information. Higher bins may allow finer registration but also adds more intensity noise. Smaller bins is more robust to intensity variations but may not be as discriminative.
	sampling_intensity_method : str
		This parameter is currently not used. 'random' is on by default. Evaluating the metric of registration is too expensive for the full image, thus we need to sample. Random is good since it avoids consistently evaluating the same grid points. 
	MetricSamplingPercentage : scalar,
		Number betwen 0-1 specifying the fraction of image voxels to evaluate the cost per registration iteration. Usually want to keep this fairly low for speed.  
	shrink_factors : list of lists or scalars 
		Shrink factors relative to the original image's size. When the list entry, shrink_factors[i], is a scalar the same factor is applied to all axes.
		When the list entry is a list, shrink_factors[i][j] is applied to axis j. This allows us to specify different shrink factors per axis. This is useful
		in the context of microscopy images where it is not uncommon to have unbalanced sampling such as a 512x512x8 image. In this case we would only want to 
		sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
	smoothing_sigmas : list of lists or scalars  
		Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These are in pixel units.
	optimizer : str 
		The optimization algorithm used to find the parameters. 

		'gradient' : 
			uses regular gradient descent. registration_method.SetOptimizerAsGradientDescent
		'1+1_evolutionary' : 
			uses 1+1 evolutionary which is default in Matlab. registration_method.SetOptimizerAsOnePlusOneEvolutionary

	optimizer_params : dict
		Python dictionary-like specification of the optimization parameters. See :func:`unwrap3D.Parameters.params.gradient_descent_affine_reg` for setting up gradient descent (Default) and 
		:func:`unwrap3D.Parameters.params.evolutionary_affine_reg` for setting up the evolutionary optimizer
	eps : scalar
		small number for numerical precision
	
	Returns
	-------
	SimpleITK.Transform

	"""
	import SimpleITK as sitk

	def imadjust(vol, p1, p2): 
		import numpy as np 
		from skimage.exposure import rescale_intensity
		# this is based on contrast stretching and is used by many of the biological image processing algorithms.
		p1_, p2_ = np.percentile(vol, (p1,p2))
		vol_rescale = rescale_intensity(vol, in_range=(p1_,p2_))
		return vol_rescale

	import numpy as np 

	if p12 is None:
		im1_ = vol1.copy()
		im2_ = vol2.copy()
		# no contrast stretching 
		if rescale_intensity == False:
			im1_ = im1_ / (2**imtype - 1)
			im2_ = im2_ / (2**imtype - 1)
		else:
			im1_ = (im1_ - im1_.min()) / (im1_.max() - im1_.min() + eps)
			im2_ = (im2_ - im2_.min()) / (im2_.max() - im2_.min() + eps)
	else:
		# contrast stretching 
		im1_ = imadjust(vol1, p12[0], p12[1])
		im2_ = imadjust(vol2, p12[0], p12[1])

		if rescale_intensity == False:
			im1_ = im1_ / (2**imtype - 1)
			im2_ = im2_ / (2**imtype - 1)
		else:
			im1_ = (im1_ - im1_.min()) / (im1_.max() - im1_.min() + eps)
			im2_ = (im2_ - im2_.min()) / (im2_.max() - im2_.min() + eps)

	### main analysis scripts. 
	v1 = sitk.GetImageFromArray(im1_, isVector=False)
	v2 = sitk.GetImageFromArray(im2_, isVector=False)
		
	# a) initial transform 
	# translation.
	if centre_tfm_model=='geometry':
		translation_mode = sitk.CenteredTransformInitializerFilter.GEOMETRY
	if centre_tfm_model=='moments':
		translation_mode = sitk.CenteredTransformInitializerFilter.MOMENTS

	# if tfm_type == 'translation':
	#     transform_mode = sitk.TranslationTransform(3)    
	# these are the built in transforms without additional modification. 
	if tfm_type == 'rigid':
		transform_mode = sitk.Euler3DTransform()
	if tfm_type == 'iso_similarity':
		transform_mode = sitk.Similarity3DTransform()
	if tfm_type == 'aniso_similarity':
		transform_mode = sitk.ScaleVersor3DTransform() # this version allows anisotropic scaling 
	if tfm_type == 'affine': 
		# transform_mode = sitk.AffineTransform(3) # this is a generic that requires setting. 
		transform_mode = sitk.ScaleSkewVersor3DTransform() # this is anisotropic + skew

	initial_transform = sitk.CenteredTransformInitializer(v1, 
														  v2, 
														  transform_mode, # this is where we change the transform
														  translation_mode)

	# a) Affine registration transform  
	# Select a different type of affine registration
	# multiscale rigid.
	registration_method = sitk.ImageRegistrationMethod()

	# Similarity metric settings.
	registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=metric_numberOfHistogramBins) 
	# key for making it fast. (subsampling, don't use all the pixels)
	if sampling_intensity_method == 'random':
		registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
	registration_method.SetMetricSamplingPercentage(MetricSamplingPercentage)
	registration_method.SetInterpolator(sitk.sitkLinear) # use this to help resampling the intensity at each iteration. 

	# Optimizer settings. # put these settings in a dedicated params file? 
	if optimizer == 'gradient':
		registration_method.SetOptimizerAsGradientDescent(learningRate=optimizer_params['learningRate'], 
														  numberOfIterations=optimizer_params['numberOfIterations'], # increase this.  
														  convergenceMinimumValue=optimizer_params['convergenceMinimumValue'], 
														  convergenceWindowSize=optimizer_params['convergenceWindowSize'],
														  estimateLearningRate=registration_method.EachIteration)
	if optimizer == '1+1_evolutionary':
		registration_method.SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations=optimizer_params['numberOfIterations'], 
																 epsilon=optimizer_params['epsilon'], 
																 initialRadius= optimizer_params['initialRadius'],
																 growthFactor= optimizer_params['growthFactor'],
																 shrinkFactor= optimizer_params['shrinkFactor']
																 )
	registration_method.SetOptimizerScalesFromIndexShift()

	# set the multiscale registration parameters here. 
	registration_method.SetShrinkFactorsPerLevel(shrinkFactors = shrink_factors) # use just the one scale. 
	registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas) # don't filter. 
	registration_method.SetInitialTransform(initial_transform, inPlace=False)
	
	tfm = registration_method.Execute(sitk.Cast(v1, sitk.sitkFloat32), 
									  sitk.Cast(v2, sitk.sitkFloat32))

	# if tfm_type == 'affine': 
	#     # refit with the above params. 
	#     tfm_affine = sitk.AffineTransform(3)
	#     tfm_affine.SetMatrix(tfm.GetMatrix())
	#     tfm_affine.SetTranslation(tfm.GetTranslation())
	#     tfm_affine.SetCenter(tfm.GetCenter())
	#     registration_method.SetInitialTransform(tfm_affine, inPlace=False)
	#     tfm = registration_method.Execute(sitk.Cast(v1, sitk.sitkFloat32), 
	#                                       sitk.Cast(v2, sitk.sitkFloat32))

	return tfm 
	

# to do: define helper script to wrap the main preprocessing steps for demons registration of two volumetric images.
def SITK_multiscale_demons_registration(vol1, vol2, 
										imtype=16,
										p12=(2,99.8),
										rescale_intensity=True,  
										centre_tfm_model='geometry', 
										demons_type='diffeomorphic', 
										n_iters = 25, 
										smooth_displacement_field = True,
										smooth_alpha=.8,
										shrink_factors = [2.,1.], 
										smoothing_sigmas = [1.,1.],
										eps=1e-12): 
	r""" Main function to register two input volumes using Demon's registration algorithm in the SimpleITK library in a multiscale manner. 
	
	see SimpleITK notebooks, https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html

	Parameters
	----------
	vol1 : (MxNxL) array
		reference volume given as a Numpy array
	vol2 : (MxNxL) array
		volume to register to vol2, given as a Numpy array
	imtype : int
		specifies the -bit of the image e.g. 8-bit, 16-bit etc. Used to normalize the intensity if ``rescale_intensity=False``
	p12 : (p1,p2) tuple
		if provided, vol1 and vol2 are constract enhanced using percentage intensity normalization 
	rescale_intensity : bool
		if True, does min-max scaling of image intensities in the two volumes independently before registering. This follows percentage intensity normalization if specified. 
	centre_tfm_model : str
		specifies the initial centering model. This is one of two options. For microscopy, we find 'geometry' has less jitter and is best. 
		
		'geometry': str
			uses sitk.CenteredTransformInitializerFilter.GEOMETRY. This computes the geometrical center independent of intensity.
		'moments': str
			uses sitk.CenteredTransformInitializerFilter.MOMENTS. This computes the geometrical center of mass of the volume based on using image intensity as a weighting. 

	demons_type : str
		One of two demon's filters in SimpleITK. 

		'diffeomorphic' : str
			sitk.DiffeomorphicDemonsRegistrationFilter(). This implements the diffeomorphic demons approach of [1]_ to penalise foldovers in the warp field and is more biologically plausible
		'symmetric' : str
			sitk.FastSymmetricForcesDemonsRegistrationFilter(). This implements the symmetric forces of [2]_. The idea is that in general the warp field of registering vol2 to vol1 and vice versa is not quite the same. in Symmetric forces demon the learning takes the average of these two warp fields to ensure symmetry preservation.

	shrink_factors : list of lists or scalars 
		Shrink factors relative to the original image's size. When the list entry, shrink_factors[i], is a scalar the same factor is applied to all axes.
		When the list entry is a list, shrink_factors[i][j] is applied to axis j. This allows us to specify different shrink factors per axis. This is useful
		in the context of microscopy images where it is not uncommon to have unbalanced sampling such as a 512x512x8 image. In this case we would only want to 
		sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
	smoothing_sigmas : list of lists or scalars  
		Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These are in pixel units.
	eps : scalar
		small number for numerical precision
	
	Returns
	-------
	SimpleITK.DisplacementFieldTransform

	References
	----------
	.. [1] Vercauteren et al. "Diffeomorphic demons: Efficient non-parametric image registration." NeuroImage 45.1 (2009): S61-S72.
	.. [2] Avants et al. "Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain." Medical image analysis 12.1 (2008): 26-41.

	"""
	import SimpleITK as sitk

	def imadjust(vol, p1, p2): 
		import numpy as np 
		from skimage.exposure import rescale_intensity
		# this is based on contrast stretching and is used by many of the biological image processing algorithms.
		p1_, p2_ = np.percentile(vol, (p1,p2))
		vol_rescale = rescale_intensity(vol, in_range=(p1_,p2_))
		return vol_rescale

	import numpy as np 

	if p12 is None:
		im1_ = vol1.copy()
		im2_ = vol2.copy()
		# no contrast stretching 
		if rescale_intensity == False:
			im1_ = im1_ / (2**imtype - 1)
			im2_ = im2_ / (2**imtype - 1)
		else:
			im1_ = (im1_ - im1_.min()) / (im1_.max() - im1_.min() + eps)
			im2_ = (im2_ - im2_.min()) / (im2_.max() - im2_.min() + eps)
	else:
		# contrast stretching 
		im1_ = imadjust(vol1, p12[0], p12[1])
		im2_ = imadjust(vol2, p12[0], p12[1])

		if rescale_intensity == False:
			im1_ = im1_ / (2**imtype - 1)
			im2_ = im2_ / (2**imtype - 1)
		else:
			im1_ = (im1_ - im1_.min()) / (im1_.max() - im1_.min() + eps)
			im2_ = (im2_ - im2_.min()) / (im2_.max() - im2_.min() + eps)

	### main analysis scripts. 
	v1 = sitk.GetImageFromArray(im1_, isVector=False)
	v2 = sitk.GetImageFromArray(im2_, isVector=False)
		
	# a) initial transform 
	# translation.
	if centre_tfm_model=='geometry':
		translation_mode = sitk.CenteredTransformInitializerFilter.GEOMETRY
	if centre_tfm_model=='moments':
		translation_mode = sitk.CenteredTransformInitializerFilter.MOMENTS
	initial_transform = sitk.CenteredTransformInitializer(v1, 
														  v2, 
														  sitk.Euler3DTransform(),
														  translation_mode)

	# a) demons transform (best to have corrected out any rigid transforms a priori) 
	 # Select a Demons filter and configure it.
	if demons_type == 'diffeomorphic': 
		demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter() # we should use this version.
	if demons_type == 'symmetric':
		demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
	# set the number of iterations
	demons_filter.SetNumberOfIterations(n_iters) # 5 for less. # long time for 20? 
	# Regularization (update field - viscous, total field - elastic).
	demons_filter.SetSmoothDisplacementField(smooth_displacement_field)
	demons_filter.SetStandardDeviations(smooth_alpha)
	
	# run the registration and return the final transform parameters
	final_tfm = multiscale_demons(registration_algorithm=demons_filter, 
								  fixed_image = v1,
								  moving_image = v2,
								  initial_transform = initial_transform,
								  shrink_factors = shrink_factors, # did have 2 here. -> test, can we separate the  # do at the same scale. 
								  smoothing_sigmas = smoothing_sigmas) # set smoothing very low, since we want it to zone in on interesting features. 
	# check again how this is parsed .
	return final_tfm

# define helper scripts to transform new volumes given a deformation.
def transform_img_sitk(vol, tfm):
	r""" One-stop function for applying any SimpleITK transform to an input image. Linear interpolation is used.  
	
	Parameters
	----------
	vol: array
		input image as a numpy array 	
	tfm: SimpleITK.Transform
		A simpleITK transform instance such as that resulting from using the simpleITK registration functions in this module.
	
	Returns
	-------
	v_transformed : array
		resulting image after applying the given transform to the input, returned as a numpy array 

	"""
	import SimpleITK as sitk

	v = sitk.GetImageFromArray(vol, isVector=False)
	v_transformed = sitk.Resample(v, 
								  v, 
								  tfm, # this should work with all types of transforms.
								  sitk.sitkLinear, 
								  0.0, 
								  v.GetPixelID())
	v_transformed = sitk.GetArrayFromImage(v_transformed) # back to numpy format. 
	
	return v_transformed


# here we need to add functions for reading and writing SITK transforms, and converting a numpy array to diplacemeent field to take into advantage of SITK image resampling. 