
import numpy as np 
from ..Analysis_Functions import topography as topotools

        

def voxelize_unwrap_params(unwrap_params, 
                           vol_shape, 
                           preupsample=None, 
                           ksize=3, 
                           smooth_sigma=3, 
                           erode_ksize=1):
    r""" Voxelizes an unwrapped surface mesh described in (u,v) in (x,y,z) original space for propagation.

    Parameters
    ----------
    unwrap_params : (UxVx3) array 
        the unwrapped 2D lookup table implicitly associating 2D (u,v) to corresponding (x,y,z) coordinates
    vol_shape : (3,) array
        the shape of the original volume image.
    preupsample : scalar
        if not none, unwrap_params is resized to preupsample times the current image size. This effectively increases the mesh sampling and is recommended to get a closed volume voxelization. 
    ksize : int
        morphological closing radius of the Ball kernel to close small holes after voxelization
    smooth_sigma : scalar
        if not None, a Gaussian smooth of the indicated sigma is applied to smooth the voxelized binary, then rebinariesd by thresholding >= 0.5. This process helps to diffuse and close over larger holes than binary closing.
    erode_ksize : int
        if not None, the voxelized binary is eroded by erode_ksize. This is done when the voxelized binary is larger than the original shape.

    Returns
    ------- 
    surf_unwrap_vol : (MxNxL) array
        voxelized mesh in original Cartesian (x,y,z) space. 

    """
    import skimage.transform as sktform 
    import skimage.morphology as skmorph 
    # from skimage.filters import gaussian 
    import scipy.ndimage as ndimage
    import numpy as np 
    from scipy.ndimage.morphology import binary_fill_holes


    unwrap_params_ = unwrap_params.copy()
    try:
        surf_unwrap_vol = np.zeros(vol_shape, dtype=np.bool)
    except:
        surf_unwrap_vol = np.zeros(vol_shape, dtype=np.bool_)

    if preupsample is not None:
        unwrap_params_ = np.dstack([sktform.resize(unwrap_params_[...,ch], output_shape=(np.hstack(unwrap_params.shape[:2])*preupsample).astype(np.int32), preserve_range=True) for ch in np.arange(unwrap_params.shape[-1])])

    # discretize. 
    unwrap_params_ = (unwrap_params_+0.5).astype(np.int32) # 6-connectivity

    surf_unwrap_vol[unwrap_params_[...,0],
                    unwrap_params_[...,1],
                    unwrap_params_[...,2]] = 1

    surf_unwrap_vol = skmorph.binary_closing(surf_unwrap_vol>0, skmorph.ball(ksize)); 
    surf_unwrap_vol = binary_fill_holes(surf_unwrap_vol)

    if smooth_sigma is not None: 
        # surf_unwrap_vol = gaussian(surf_unwrap_vol*1., sigma=smooth_sigma, preserve_range=True)
        surf_unwrap_vol = ndimage.gaussian_filter(surf_unwrap_vol*1., sigma=smooth_sigma)
        surf_unwrap_vol = surf_unwrap_vol/surf_unwrap_vol.max()
        surf_unwrap_vol = surf_unwrap_vol >= 0.5

    if erode_ksize is not None:
        surf_unwrap_vol = skmorph.binary_erosion(surf_unwrap_vol, skmorph.ball(erode_ksize))

    return surf_unwrap_vol


def build_topography_space_from_Suv(Sref_uv, 
                                    Sref_binary,
                                    external_gradient_field = None, # build if none. 
                                    pad_size=50, 
                                    alpha_step=None,
                                    outer_surf_pts = None,
                                    outer_n_dists = None,
                                    outer_pad_dist=20,
                                    # outer_sdf_method = 'edt', always use edt. (for speed)
                                    outer_method='forward_euler',
                                    outer_smooth_win_factor=16,
                                    # inner_d_step=None,
                                    inner_n_dists = 10,
                                    inner_sdf_method = 'harmonic',
                                    inner_source_mask = None, 
                                    inner_harmonic_ds = 4., # 4x downsample by default. 
                                    inner_method='active_contours'): # convenience function for the standard way we put together these functions. -> options for gradient_outer, gradient_inner. 
    

    r""" Provides a convenience api for building topography space with an input reference mesh, with options to customize sdf specification.

    propagate surface outwards using forward euler + smoothing regularization. 
    propagate surface inwards using active contours 
    
    """
    import numpy as np 
    from ..Segmentation.segmentation import sdf_distance_transform, poisson_dist_tform_3D
    import scipy.ndimage as ndimage 
    
    vol_lims = Sref_binary.shape[:3]
    
    
    if alpha_step is None:
        
        # infer this and set the propagation step. 
        delta_xx = np.diff(Sref_uv, axis=1) # column-wise difference in (x,y,z)
        delta_yy = np.diff(Sref_uv, axis=0) # row-wise difference in (x,y,z)
        # convert the difference to magnitude. 
        delta_xx_abs = np.linalg.norm(delta_xx, axis=-1)
        delta_yy_abs = np.linalg.norm(delta_yy, axis=-1)
        # take the mean of the x-, y- direction magnitude 
        alpha_step = .5*(np.mean(delta_xx_abs) + np.mean(delta_yy_abs)) 
        
    ####
    #   0. Check and construct the distance transform. 
    ####  
    # create a pad. 
    
    outer_surf_pts_bound = None
    pad_image = False # check for whether we padded or not. 

    if external_gradient_field is None:
        pad_image = True # set as padding. 
        Sref_binary_pad = np.pad(Sref_binary, pad_width=[[pad_size,pad_size], 
                                                         [pad_size,pad_size],
                                                         [pad_size,pad_size]])
        Sref_uv_pad = Sref_uv + pad_size # offset. 
        EDT_sdf = sdf_distance_transform(Sref_binary_pad>0)
        external_gradient_field = np.array(np.gradient(EDT_sdf))
        external_gradient_field = external_gradient_field/(np.linalg.norm(external_gradient_field, axis=0)[None,...] + 1e-20).astype(np.float32)
        
        # smooth
        external_gradient_field = np.array([ndimage.gaussian_filter(grad, sigma=1) for grad in external_gradient_field])
        external_gradient_field = external_gradient_field/(np.linalg.norm(external_gradient_field,axis=0)[None,...]+1e-20)
        
        if outer_surf_pts is not None:
            outer_surf_pts_bound = outer_surf_pts + pad_size
    else:
        Sref_binary_pad = Sref_binary.copy()
        Sref_uv_pad = Sref_uv.copy()
        if outer_surf_pts is not None:
            outer_surf_pts_bound = outer_surf_pts.copy()
        
    ####
    #   1. Perform the outer propagation. 
    ####           
    
    if outer_method == 'forward_euler':
        smooth_winsize = np.min(Sref_uv.shape[:2]) // int(outer_smooth_win_factor)

        uv_unwrap_params_depth_out = prop_ref_surface(Sref_uv_pad, 
                                                      vol_binary = Sref_binary_pad,
                                                      binary_vol_grad = external_gradient_field.transpose(1,2,3,0),
                                                      vol_size=Sref_binary_pad.shape,
                                                      d_step=-1*alpha_step, # .5 pixel intervals.  ? 
                                                      n_dist=outer_n_dists, 
                                                      surf_pts_ref=outer_surf_pts_bound,
                                                      pad_dist=outer_pad_dist/float(alpha_step), # on top of the inference add some padding. 
                                                      smooth_method='uniform', 
                                                      smooth_win=smooth_winsize)
        
    if outer_method == 'active_contours':
        # determine the max bound if n_dist not specified. 
        from ..Mesh.meshtools import parametric_uv_unwrap_mesh_constant_img_flow
        uv_unwrap_params_depth_out = parametric_uv_unwrap_mesh_constant_img_flow(Sref_uv_pad, 
                                                                external_img_gradient=external_gradient_field.transpose(1,2,3,0),  
                                                                niters = outer_n_dists,
                                                                deltaL=5e-2, 
                                                                surf_pts_ref=outer_surf_pts_bound, 
                                                                step_size= -1*alpha_step,
                                                                pad_dist=outer_pad_dist/float(alpha_step),
                                                                method='implicit', 
                                                                robust_L=True, 
                                                                mollify_factor=1e-5,
                                                                conformalize=False, # not so good .  
                                                                gamma=1, 
                                                                alpha=0.1, 
                                                                beta=0.5, 
                                                                eps=1e-20)
        uv_unwrap_params_depth_out = uv_unwrap_params_depth_out.transpose(3,0,1,2)
        
        inner_sdf_method = 'harmonic',
        inner_harmonic_ds = 4., # 4x downsample by default. 
        inner_method='active_contours'
        
    if inner_sdf_method == 'harmonic':
        # we need to compute the poisson distance transform and switch this out. 
        import skimage.transform as sktform
        ds = float(inner_harmonic_ds)
        
        Sref_binary_pad_ds = ndimage.zoom(Sref_binary_pad*255., 
                                            zoom =[1./ds, 1./ds, 1./ds],
                                            order=1)/255. > 0.5 # perform linear down. 
        if inner_source_mask is not None:
            if pad_image:
                # then pad. 
                inner_source_mask_pad = np.pad(inner_source_mask, pad_width=[[pad_size,pad_size], [pad_size, pad_size], [pad_size, pad_size]])
            else:
                inner_source_mask_pad = inner_source_mask.copy() # assume the shapes are already correct. 
            inner_source_mask_pad_ds = ndimage.zoom( inner_source_mask_pad*255., 
                                                    zoom =[1./ds, 1./ds, 1./ds],
                                                    order=1)/255. > 0.5 # perform linear down. 
            poisson_dtform = poisson_dist_tform_3D(Sref_binary_pad_ds,
                                                   pts = np.argwhere(inner_source_mask_pad_ds>0)) # with source points. 
        else:
            poisson_dtform = poisson_dist_tform_3D(Sref_binary_pad_ds,
                                                   pts = None) # no source points. 
            
        poisson_dtform = sktform.resize(poisson_dtform, 
                                        Sref_binary_pad.shape,
                                        order=1, preserve_range=True)
                                                      
        external_gradient_field = np.array(np.gradient(poisson_dtform))
        external_gradient_field = external_gradient_field/(np.linalg.norm(external_gradient_field,axis=0)[None,...]+1e-20).astype(np.float32)

        external_gradient_field = np.array([ndimage.gaussian_filter(external_gradient_field[iii], sigma=3) for iii in np.arange(3)])
        external_gradient_field = external_gradient_field/(np.linalg.norm(external_gradient_field,axis=0)[None,...]+1e-20)


    if inner_method == 'forward_euler':
        smooth_winsize = np.min(Sref_uv.shape[:2]) // int(outer_smooth_win_factor)
        
        inner_d_step_size = -1*alpha_step
        if inner_sdf_method == 'harmonic':
            inner_d_step_size = alpha_step
        
        uv_unwrap_params_depth_in = prop_ref_surface(Sref_uv_pad, 
                                                      vol_binary = None,
                                                      binary_vol_grad = external_gradient_field.transpose(1,2,3,0),
                                                      vol_size=Sref_binary_pad.shape,
                                                      d_step=inner_d_step_size, # .5 pixel intervals.  ? 
                                                      n_dist=inner_n_dists, 
                                                      surf_pts_ref=None, # to do include this too ( we can do this by distance transform and voxelization)
                                                      pad_dist=0, # on top of the inference add some padding. 
                                                      smooth_method='uniform', 
                                                      smooth_win=smooth_winsize)
        
    if inner_method == 'active_contours':
        # determine the max bound if n_dist not specified. 
        from ..Mesh.meshtools import parametric_uv_unwrap_mesh_constant_img_flow
        
        uv_unwrap_params_depth_in = parametric_uv_unwrap_mesh_constant_img_flow(Sref_uv_pad, 
                                                                        external_img_gradient=external_gradient_field.transpose(1,2,3,0),  
                                                                        niters = inner_n_dists,
                                                                        deltaL=5e-2, 
                                                                        surf_pts_ref=None, 
                                                                        step_size= alpha_step,
                                                                        pad_dist=0,
                                                                        method='implicit', 
                                                                        robust_L=True, # must be switched on for this. 
                                                                        mollify_factor=1e-5,
                                                                        conformalize=False, # not so good .  
                                                                        gamma=1, 
                                                                        alpha=0.1, 
                                                                        beta=0.5, 
                                                                        eps=1e-20)
        uv_unwrap_params_depth_in = uv_unwrap_params_depth_in.transpose(3,0,1,2)


    # combine both.
    N_out = len(uv_unwrap_params_depth_out) - 1
    N_in = len(uv_unwrap_params_depth_in) - 1 
    
    
    uv_unwrap_params_depth_all = np.vstack([uv_unwrap_params_depth_in[::-1], 
                                            uv_unwrap_params_depth_out[1:]])
    
    # reverse the pad_size.
    if pad_image:
        # this is the only time padding was added. 
        uv_unwrap_params_depth_all = uv_unwrap_params_depth_all - pad_size
    else: # nothing to do, if false. 
        pass
    
    uv_unwrap_params_depth_all[...,0] = np.clip(uv_unwrap_params_depth_all[...,0], 0, vol_lims[0]-1)
    uv_unwrap_params_depth_all[...,1] = np.clip(uv_unwrap_params_depth_all[...,1], 0, vol_lims[1]-1)
    uv_unwrap_params_depth_all[...,2] = np.clip(uv_unwrap_params_depth_all[...,2], 0, vol_lims[2]-1)
    
    return uv_unwrap_params_depth_all, (N_in, N_out), alpha_step


def prop_ref_surface(unwrap_params_ref, 
                         vol_size,  
                         preupsample=3, 
                         vol_binary=None,
                         binary_vol_grad = None, 
                         ksize=1, 
                         smooth_sigma=None, 
                         d_step=1, 
                         n_dist=None, 
                         surf_pts_ref=None,
                         pad_dist=5, 
                         norm_vectors=False,
                         use_GVF=False, 
                         GVF_mu=0.01, 
                         GVF_iterations=15,
                         smoothgradient=100, 
                         smoothrobust=False,
                         smooth_method='uniform', 
                         smooth_win=15,
                         invertorder=False,
                         squarenorm=False):

    r""" Propagate a (u,v) indexed (x,y,z) surface such as that from unwrapping normally into or outwards at a uniform stepsize of ``d_step`` for ``n_dist`` total number of steps. This function is used to create the topography space to map a whole volume given an unwrapping of a single surface.   
    
    1. if vol_binary is not provided, a voxelization is first computed 
    2. based on the vol_binary, the signed distance function and the gradient field for propagation is computed
    3. initial points are then integrated along the gradient field iteratively using explicit Euler with new position x_next = x + d_step * gradient
    4. smoothing is used after each iteration to regularise the advected points. this is crucial for outward stepping when the interspacing between mesh points are increasing 

    Parameters
    ----------
    unwrap_params_ref : (UxVx3) array
        the unwrapped (u,v) image-indexed (x,y,z) surface to propagate
    vol_size : (m,n,l) 3-tuple
        the size of the original volume image
    preupsample : scalar
        if not none, unwrap_params is resized to preupsample times the current image size. This effectively increases the mesh sampling and is recommended to get a closed volume voxelization. 
    smooth_sigma : scalar
        if not None, a Gaussian smooth of the indicated sigma is applied to smooth the voxelized binary, then rebinariesd by thresholding >= 0.5. This process helps to diffuse and close over larger holes than binary closing.
    d_step : scalar
        the step size in voxels to move in the normal direction. A d_step of negative sign reverses the direction of advection
    n_dist : int
        if not None, the total number of steps to take in the normal direction. 
    surf_pts_ref : (N,3) array 
        if provided, this is a reference surface with which to automatically determine the n_dist when propagating the surface outwards to ensure the full reference shape is sampled in topography space.
    pad_dist : int 
        an additional fudge factor added to the automatically determined n_dist when surf_pts_ref is provided and n_dist is not user provided
    smoothgradient : scalar
        if given, this is the parameter that controls the amount of smoothing in the `smoothN <https://www.biomecardio.com/matlab/smoothn_doc.html>`_ algorithm
    smoothrobust : bool 
        if True, use the robust smoothing mode of smoothN algorithm 
    smooth_method : str
        specifies the gradient smoothing mode per iteration to regularise the points. 

        'smoothN' : str
            applies a spline-based smoothing of `Garcia <https://www.biomecardio.com/matlab/smoothn_doc.html>`_. This algorithm can be pretty slow. See :func:`unwrap3D.Unzipping.unzip.smooth_unwrap_params_3D_spherical_SmoothN`
        'uniform' : str
            treats the spacing of mesh points as uniform and applies fast separable 1D uniform filter smoothing along each axis of the 2D parameterization. See :func:`unwrap3D.Unzipping.unzip.smooth_unwrap_params_3D_spherical`

    smooth_win : int
        the smoothing window to smooth the advected points used in the uniform method. 
    invertorder : bool
        if invertorder, the final concatenated stack of points of (D,U,V,3) where D is the total number of steps is inverted in the first dimension. 
    squarenorm : bool 
        if True, propagate using the gradient field, V normalised by magnitude squared i.e. :math:`V/(|V|^2)` instead of the unit magnitude normalization :math:`V/|V|`.

    Returns
    -------
    unwrapped_coord_depth : (DxUxVx3) array
        the final unwrapped topography space of (d,u,v) image-indexed (x,y,z) surface which map an entire volume space to topography space.

    """
    import numpy as np 
    from tqdm import tqdm 
    from hausdorff import hausdorff_distance
    # import point_cloud_utils as pcu
    # import pylab as plt 
    from ..Segmentation.segmentation import surf_normal_sdf
    from ..Image_Functions.image import map_intensity_interp3

    
    # if gradient is not supplied ... 
    if binary_vol_grad is None:
        
        if vol_binary is None:
            # construct a binary of the input mesh/points. # todo change to the better voxelizer.... 
            vol_binary = voxelize_unwrap_params(unwrap_params_ref, 
                                                 vol_shape=vol_size, 
                                                 preupsample=preupsample, 
                                                 ksize=ksize, 
                                                 smooth_sigma=smooth_sigma)
            
            # print(np.sum(vol_binary))
            # plt.figure()
            # plt.imshow(vol_binary[vol_binary.shape[0]//2])
            # plt.show()
            
            # get the surface normal gradient outwards.  
            # binary_vol_grad = vol_gradient(vol_binary, smooth_sigma=smooth_sigma, normalize=True, invert=True, squarenorm=squarenorm) # always outwards. # i see. # we should get the sdf !.
            # binary_sdf_vol = sdf_distance_transform(binary, rev_sign=True, method='edt')
            binary_vol_grad, binary_vol_sdf = surf_normal_sdf(vol_binary, 
                                                              smooth_gradient=smooth_sigma, 
                                                              eps=1e-12, 
                                                              norm_vectors=norm_vectors, 
                                                              use_GVF=use_GVF, 
                                                              GVF_mu=GVF_mu, 
                                                              GVF_iterations=GVF_iterations)
            binary_vol_grad = -1*binary_vol_grad.transpose(1,2,3,0)
        
            # print(binary_vol_grad.shape)
            # print(np.min(binary_vol_sdf), np.max(binary_vol_sdf))
            # plt.figure()
            # plt.imshow(binary_vol_grad[vol_binary.shape[0]//2,...,0])
            # plt.show()
            
            # plt.figure()
            # plt.imshow(binary_vol_sdf[vol_binary.shape[0]//2])
            # plt.show()
            
        else:
            # print(np.sum(vol_binary))
            # plt.figure()
            # plt.imshow(vol_binary[vol_binary.shape[0]//2])
            # plt.show()
            
            # get the surface normal gradient outwards.  
            # binary_vol_grad = vol_gradient(vol_binary, smooth_sigma=smooth_sigma, normalize=True, invert=True, squarenorm=squarenorm) # always outwards. # i see. # we should get the sdf !.
            # binary_sdf_vol = sdf_distance_transform(binary, rev_sign=True, method='edt')
            binary_vol_grad, binary_vol_sdf = surf_normal_sdf(vol_binary, 
                                                              smooth_gradient=smooth_sigma, 
                                                              eps=1e-12, 
                                                              norm_vectors=norm_vectors, # set to True? 
                                                              use_GVF=use_GVF, 
                                                              GVF_mu=GVF_mu, 
                                                              GVF_iterations=GVF_iterations)
            binary_vol_grad = -1*binary_vol_grad.transpose(1,2,3,0)
    
        # print(binary_vol_grad.shape)
        # print(np.min(binary_vol_sdf), np.max(binary_vol_sdf))
        # plt.figure()
        # plt.imshow(binary_vol_grad[vol_binary.shape[0]//2,...,0])
        # plt.show()
        
        # plt.figure()
        # plt.imshow(binary_vol_sdf[vol_binary.shape[0]//2])
        # plt.show()
        
    # infer the number of dists to step for from the reference if not prespecified. 
    if n_dist is None :

        unwrap_params_ref_flat = unwrap_params_ref.reshape(-1, unwrap_params_ref.shape[-1])
        # hausdorff_dist = pcu.hausdorff_distance(unwrap_params_ref, surf_pts_ref)
        hausdorff_dist = hausdorff_distance(unwrap_params_ref_flat, surf_pts_ref) # much faster. 
        n_dist = np.int64(np.ceil(hausdorff_dist / np.abs(float(d_step))) + pad_dist) # convert to discrete number of steps

        # # Following is deprecated. 
        # unwrap_params_ref_flat = unwrap_params_ref.reshape(-1, unwrap_params_ref.shape[-1])
        # # infer the maximum step size so as to cover the initial otsu surface.
        # mean_pt = np.nanmean(unwrap_params_ref_flat, axis=0)
        # # # more robust to do an ellipse fit. ? => doesn't seem so... seems best to take the extremal point -> since we should have a self-similar shape. 
        # # unwrap_params_fit_major_len = np.max(np.linalg.eigvalsh(np.cov((unwrap_params_ref_flat-mean_pt[None,:]).T))); unwrap_params_fit_major_len=np.sqrt(unwrap_params_fit_major_len)
        # # surf_ref_major_len = np.max(np.linalg.eigvalsh(np.cov((surf_pts_ref-mean_pt[None,:]).T))); surf_ref_major_len = np.sqrt(surf_ref_major_len)

        # mean_dist_unwrap_params_ref = np.linalg.norm(unwrap_params_ref_flat-mean_pt[None,:], axis=-1).max()
        # mean_surf_pts_ref = np.linalg.norm(surf_pts_ref-mean_pt[None,:], axis=-1).max() # strictly should do an ellipse fit... 

        # n_dist = np.int64(np.ceil(mean_surf_pts_ref-mean_dist_unwrap_params_ref))
        # n_dist = n_dist + pad_dist # this is in pixels
        # n_dist = np.int64(np.ceil(n_dist / float(d_step))) # so if we take 1./2 step then we should step 2*

        # print(n_dist)
        # print('----')

    unwrapped_coord_depth = [unwrap_params_ref] # initialise with the reference surface. 

    for d_ii in tqdm(np.arange(n_dist)):

        pts_next = unwrapped_coord_depth[-1].copy(); pts_next = pts_next.reshape(-1,pts_next.shape[-1]) # pull the last and flatten. 

        # get the gradient at the next point. 
        grad_next_x = map_intensity_interp3(pts_next, 
                                            grid_shape=binary_vol_grad.shape[:-1], I_ref=binary_vol_grad[...,0], method='linear', cast_uint8=False)
        grad_next_y = map_intensity_interp3(pts_next, 
                                            grid_shape=binary_vol_grad.shape[:-1], I_ref=binary_vol_grad[...,1], method='linear', cast_uint8=False)
        grad_next_z = map_intensity_interp3(pts_next, 
                                            grid_shape=binary_vol_grad.shape[:-1], I_ref=binary_vol_grad[...,2], method='linear', cast_uint8=False)
        grad_next = (np.vstack([grad_next_x, grad_next_y, grad_next_z]).T).reshape(unwrapped_coord_depth[-1].shape)

        # smooth the gradients. 
        if smooth_method is not None:
            if smooth_method == 'smoothN':
                grad_next = smooth_unwrap_params_3D_spherical_SmoothN(grad_next, 
                                                                        S=smoothgradient, 
                                                                        isrobust=smoothrobust, 
                                                                        pad_size=None) # what is none?
            if smooth_method == 'uniform':
                grad_next = smooth_unwrap_params_3D_spherical(grad_next, 
                                                              sigma_window=smooth_win, 
                                                              filter_func=None, 
                                                              filter1d=True,  
                                                              filter2d=False, 
                                                                filterALS=False, 
                                                               ALS_lam=None, 
                                                               ALS_p=None, 
                                                               ALS_iter=None)
            # if pad_size is None:
            # #     # pad the mesh..... 
            # #     grad_next = np.hstack([grad_next, grad_next, grad_next])
            # #     grad_next = np.vstack([grad_next[1:][::-1,::-1],
            # #                             grad_next,
            # #                             grad_next[:-1][::-1,::-1]])
            # # # else:
            #     grad_next = smooth_unwrap_params_3D_spherical(grad_next, 
            #                                                 sigma_window=35, 
            #                                                 isrobust=smoothrobust, 
            #                                                 pad_size=None)
            #     # unwrap_xyz, sigma_window=15, filter_func=None, 
            #     #                        filter1d=True,  filter2d=False, 
            #     #                        filterALS=True, 
            #     #                        ALS_lam=None, 
            #     #                        ALS_p=None, 
            #     #                        ALS_iter=None
            # # renormalize. 
            if squarenorm:
                grad_next = grad_next / (np.linalg.norm(grad_next, axis=-1)[...,None]**2 + 1e-12) # renormalize! or we do square of the norm ? 
            else:
                grad_next = grad_next / (np.linalg.norm(grad_next, axis=-1)[...,None] + 1e-12) # renormalize! or we do square of the norm ? 
        else:
            # renormalize. 
            if squarenorm:
                grad_next = grad_next / (np.linalg.norm(grad_next, axis=-1)[...,None]**2 + 1e-12) # renormalize! or we do square of the norm ? 
            else:
                grad_next = grad_next / (np.linalg.norm(grad_next, axis=-1)[...,None] + 1e-12) # renormalize! or we do square of the norm ? 

        # get the next point and clip.
        pts_next_ = unwrapped_coord_depth[-1] + d_step * grad_next

        ## clip the pts to the range of the image. 
        #pts_next_[...,0] = np.clip(pts_next_[...,0], 0, vol_binary.shape[0]-1)
        #pts_next_[...,1] = np.clip(pts_next_[...,1], 0, vol_binary.shape[1]-1)
        #pts_next_[...,2] = np.clip(pts_next_[...,2], 0, vol_binary.shape[2]-1)
        pts_next_[...,0] = np.clip(pts_next_[...,0], 0, vol_size[0]-1)
        pts_next_[...,1] = np.clip(pts_next_[...,1], 0, vol_size[1]-1)
        pts_next_[...,2] = np.clip(pts_next_[...,2], 0, vol_size[2]-1)

        unwrapped_coord_depth.append(pts_next_)

    unwrapped_coord_depth = np.array(unwrapped_coord_depth)

    if invertorder:
        unwrapped_coord_depth = unwrapped_coord_depth[::-1]

    return unwrapped_coord_depth


# function to fix the bounds by reinterpolation based on griddata. 
def fix_unwrap_params_boundaries_spherical(unwrap_xyz, pad=10, rescale=False):
    r""" Given a UV unwrapping of a closed spherical-like surface, this function enforces that the first and last rows map to single coordinate representing the North and South poles and conducts scattered interpolation to impute missing values.  
    
    Parameters
    ----------
    unwrap_xyz : (UxVx3) array 
        the unwrapped (u,v) image-indexed (x,y,z) surface to propagate
    pad : int 
        the internal padding of unwrap_xyz to allow interpolation of missed edge coordinates and to have better continuity in internal regions. 
    rescale : bool 
        if True, rescales points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude. see `griddata <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_

    Returns 
    -------
    unwrap_xyz_fix : (UxVx3) array 
        the fixed unwrapped (u,v) image-indexed (x,y,z) of the same size as the input

    """
    from scipy.interpolate import griddata
    import numpy as np 

    # assuming a spherical geometry. 
    unwrap_xyz_fix = unwrap_xyz.copy()

    # use this to define the bound
    min_x = np.nanmin(unwrap_xyz_fix[...,0])
    max_x = np.nanmax(unwrap_xyz_fix[...,0])
    min_y = np.nanmin(unwrap_xyz_fix[...,1])
    max_y = np.nanmax(unwrap_xyz_fix[...,1])
    min_z = np.nanmin(unwrap_xyz_fix[...,2])
    max_z = np.nanmax(unwrap_xyz_fix[...,2])

    # first and last row map to singular points. 
    unwrap_xyz_fix[0,:] = np.nanmean(unwrap_xyz[0,:], axis=0)
    unwrap_xyz_fix[-1,:] = np.nanmean(unwrap_xyz[-1,:], axis=0)

    # wrap around to get a convex hull. 
    unwrap_xyz_fix = np.hstack([unwrap_xyz_fix[:,-pad:], unwrap_xyz_fix, unwrap_xyz_fix[:,:pad]])
    valid_mask = np.logical_not(np.max(np.isnan(unwrap_xyz_fix), axis=-1))

    # now create a griddata. 
    rows, cols = np.indices((unwrap_xyz_fix[:,:,0]).shape)
    rows_ = rows[valid_mask]
    cols_ = cols[valid_mask]
    interp_x = griddata(np.hstack([cols_[:,None], rows_[:,None]]), unwrap_xyz_fix[valid_mask,0],(cols, rows), method='linear', fill_value=np.nan, rescale=rescale)
    interp_y = griddata(np.hstack([cols_[:,None], rows_[:,None]]), unwrap_xyz_fix[valid_mask,1],(cols, rows), method='linear', fill_value=np.nan, rescale=rescale)
    interp_z = griddata(np.hstack([cols_[:,None], rows_[:,None]]), unwrap_xyz_fix[valid_mask,2],(cols, rows), method='linear', fill_value=np.nan, rescale=rescale)
    
    unwrap_xyz_fix = np.dstack([interp_x, 
                                interp_y, 
                                interp_z])
    unwrap_xyz_fix = unwrap_xyz_fix[:,pad:-pad]

    unwrap_xyz_fix[unwrap_xyz_fix[...,0]<min_x-1,0] = min_x
    unwrap_xyz_fix[unwrap_xyz_fix[...,0]>max_x+1,0] = max_x
    unwrap_xyz_fix[unwrap_xyz_fix[...,1]<min_y-1,1] = min_y
    unwrap_xyz_fix[unwrap_xyz_fix[...,1]>max_y+1,1] = max_y
    unwrap_xyz_fix[unwrap_xyz_fix[...,2]<min_z-1,2] = min_z
    unwrap_xyz_fix[unwrap_xyz_fix[...,2]<min_z-1,2] = max_z     

    return unwrap_xyz_fix


def pad_unwrap_params_3D_spherical(unwrap_xyz, pad=10):
    r""" Function to pad a single or multichannel 2D image in a spherical manner topologically such that the left/right sides are periodic extensions. The top is extended with a 180 degree flip, and the bottom is also extended with a 180 degree flip

    Parameters
    ----------
    unwrap_xyz : 
        an input (MxN) or (MxNxd) image to pad

    pad : int 
        The uniform padding in the width and height 

    Returns
    -------
    unwrap_xyz_pad : 
        The padded image of dimensionality (M+2*pad x N+2*pad) or (M+2*pad x N+2*pad x d)

    """
    import numpy as np 
    unwrap_xyz_pad = unwrap_xyz.copy()
    # periodic in x
    unwrap_xyz_pad = np.hstack([unwrap_xyz[:,-pad:], 
                                unwrap_xyz, 
                                unwrap_xyz[:,:pad]])

    # flip and pad in x # but don't take the first and last!!!! else we have a discontinuity.... 
    unwrap_xyz_pad = np.vstack([unwrap_xyz_pad[1:pad+1][:,::-1][::-1], 
                                unwrap_xyz_pad, 
                                unwrap_xyz_pad[-pad-1:-1][:,::-1][::-1]])

    return unwrap_xyz_pad


def pad_unwrap_params_3D_spherical_depth(unwrap_depth_vals, pad=10, pad_depth=False, pad_depth_mode='edge'):
    r""" Function to pad a topographic space coordinate set in a spherical manner topologically such that the left/right sides are periodic extensions. The top is extended with a 180 degree flip, and the bottom is also extended with a 180 degree flip
    Optionally one can extend the depth (first channel) with the specified edge pad mode. 

    Parameters
    ----------
    unwrap_depth_vals : array 
        an input (DxMxNxd) image to pad
    pad : int 
        The uniform padding in the width and height i.e. M, N axis
    pad_depth : bool
        If True, pad the depth dimension (D) also by ``pad`` 
    pad_depth_mode : str
        if pad_depth is True, this specifies the handling of the padding. It should be one of the options in numpy.pad

    Returns
    -------
    unwrap_depth_vals_pad : 
        The padded image of dimensionality (D x M+2*pad x N+2*pad x d) for pad_depth=False or (D+2*pad x M+2*pad x N+2*pad x d) for pad_depth=True

    """
    # depth x m x n x ndim. 
    import numpy as np

    # first pad in the horizontal then vertical ...
    # pad horizontal. 
    unwrap_depth_vals_pad = np.concatenate([unwrap_depth_vals[:,:,-pad:], 
                                            unwrap_depth_vals, 
                                            unwrap_depth_vals[:,:,:pad]], axis=2)
    # pad vertical.
    unwrap_depth_vals_pad = np.concatenate([unwrap_depth_vals_pad[:,1:pad+1][:,::-1,::-1],
                                            unwrap_depth_vals_pad,
                                            unwrap_depth_vals_pad[:,-pad-1:-1][:,::-1,::-1]], axis=1)
    
    if pad_depth:
        # we must also pad in depth!. 
        unwrap_depth_vals_pad = np.pad(unwrap_depth_vals_pad, [[pad, pad], [0,0], [0,0], [0,0]], mode=pad_depth_mode) 
    
    return unwrap_depth_vals_pad


def compute_unwrap_params_normal_curvature(unwrap_depth_binary,
                                           pad=10,  
                                           compute_curvature=True,
                                           smooth_gradient=3,
                                           smooth_curvature=3,
                                           mask_gradient=False,
                                           eps=1e-12):
    r""" Function to pad a topographic space coordinate set in a spherical manner topologically such that the left/right sides are periodic extensions. The top is extended with a 180 degree flip, and the bottom is also extended with a 180 degree flip
    Optionally one can extend the depth (first channel) with the specified edge pad mode. 

    Parameters
    ----------
    unwrap_depth_vals : array 
        an input (DxMxNxd) image to pad
    pad : int 
        The uniform padding in the width and height i.e. M, N axis
    pad_depth : bool
        If True, pad the depth dimension (D) also by ``pad`` 
    pad_depth_mode : str
        if pad_depth is True, this specifies the handling of the padding. It should be one of the options in numpy.pad

    Returns
    -------
    unwrap_depth_vals_pad : 
        The padded image of dimensionality (D x M+2*pad x N+2*pad x d) for pad_depth=False or (D+2*pad x M+2*pad x N+2*pad x d) for pad_depth=True

    """
    from ..Segmentation import segmentation as segmentation
    if pad > 0: 
        unwrap_depth_binary_pad = pad_unwrap_params_3D_spherical_depth(unwrap_depth_binary[...,None], 
                                                                        pad=pad, pad_depth=True, pad_depth_mode='edge')
        unwrap_depth_binary_pad = unwrap_depth_binary_pad[...,0]
    else:
        unwrap_depth_binary_pad = unwrap_depth_binary.copy()

    if compute_curvature:
        H_normal, sdf_vol_normal, sdf_vol = segmentation.mean_curvature_binary(unwrap_depth_binary_pad, 
                                                                                smooth=smooth_curvature, 
                                                                                mask=mask_gradient, 
                                                                                smooth_gradient=smooth_gradient, 
                                                                                eps=eps)
        return unwrap_depth_binary_pad, H_normal, sdf_vol_normal, sdf_vol
    else:
        sdf_vol_normal, sdf_vol = segmentation.surf_normal_sdf(unwrap_depth_binary_pad, 
                                                                return_sdf=True, 
                                                                smooth_gradient=smooth_gradient, 
                                                                eps=eps)
        return unwrap_depth_binary_pad, sdf_vol_normal, sdf_vol 

### add in the weighted filtering variant taking into account local mapping errors. 
def smooth_unwrap_params_3D_spherical(unwrap_xyz, 
                                      sigma_window=15, 
                                      filter_func=None, 
                                      filter1d=True,  
                                      filter2d=False, 
                                      filterALS=True, 
                                      ALS_lam=None, 
                                      ALS_p=None, 
                                      ALS_iter=None):
    r""" Applies image based smoothing techniques to smooth a (u,v) image parameterized (x,y,z) surface

    Options include
    - uniform 1d smoothing along x- and y- separately (filter1d=True)
    - Gaussian 2D smoothing (filter2d=True)
    - spline based smoothing (filterALS=True)   
    
    Only one of filter1d, filter2d and filterALS should be True

    Parameters
    ----------
    unwrap_xyz : array 
        an input (MxNxd) image to smooth
    sigma_window : int
        smoothing window in pixels 
    filter_func : array
        specifies the ``weight`` in scipy.ndimage.filters.convolve1d for ``filter1d=True``. If filter_func=None, defaults to the uniform filter equivalent to taking the mean over the window ``2*sigma_window+1``
    filter1d : bool
        if True use scipy.ndimage.filters.convolve1d with the given weights, filter_func to filter the unwrapping coordinates. 
    filter2d : bool
        if True, apply skimage.filters.gaussian with sigma=sigma_window
    filterALS : bool
        if True, asymmetric least squares is applied along independent x- and y- directions for smoothing with parameters given by ``ALS_lam, ALS_p and ALS_iter``. See :func:`unwrap3D.Analysis_Functions.timeseries.baseline_als`
    ALS_lam : scalar
        Controls the degree of smoothness in the baseline. The higher the smoother.
    ALS_p : scalar
        Controls the degree of smoothness in the baseline. The smaller the more asymmetric, the more the fitting biases towards the minimum value.
    ALS_iter : int
        The number of iterations to run the algorithm. Only a few iterations is required generally. This can generally be fixed.

    Returns
    -------
    unwrap_xyz_filt : 
        the smoothed output (MxNxd) image 
    """
    from scipy.ndimage.filters import convolve1d
    from skimage.filters import gaussian
    import scipy.ndimage as ndimage

    def baseline_als(y, lam, p, niter=10):
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
        
    if filter_func is None:
        # assume uniform
        filter_func = np.ones(2*sigma_window+1); 
        filter_func = filter_func/float(np.sum(filter_func)) # normalize
        # # # assume default is gaussian
        # filter_func = np.zeros(2*sigma_window+1);filter_func[sigma_window] = 1.
        # filter_func = ndimage.gaussian_filter1d(filter_func, sigma=sigma_window/np.sqrt(2))
        # filter_func = filter_func/float(np.sum(filter_func)) # normalize
        

    unwrap_xyz_pad = pad_unwrap_params_3D_spherical(unwrap_xyz, pad=3*sigma_window) # we need to pad this more... to prevent edge effects...  
    
    if filter1d: 
        unwrap_xyz_x = convolve1d(unwrap_xyz_pad, weights=filter_func, axis=0)
        unwrap_xyz_filt = convolve1d(unwrap_xyz_x, weights=filter_func, axis=1)

    if filterALS:
        # this idea works without fundamentally changing things -> so we should be able to run proxTV ....
        # then we are going to filter.... if we set this too high ... we get moving average filter ( as the support of it is too high! )
        unwrap_xyz_x = np.array([ np.array([baseline_als(unwrap_xyz_pad[ii,:,ch], lam=ALS_lam, p=ALS_p, niter=ALS_iter) for ii in range(len(unwrap_xyz_pad))])  for ch in range(unwrap_xyz_pad.shape[-1])]) # rows. 
        unwrap_xyz_x = unwrap_xyz_x.transpose(1,2,0)
        unwrap_xyz_filt = np.array([ np.array([baseline_als(unwrap_xyz_x[:,ii,ch], lam=ALS_lam, p=ALS_p, niter=ALS_iter) for ii in range(unwrap_xyz_pad.shape[1])])  for ch in range(unwrap_xyz_pad.shape[-1])])
        unwrap_xyz_filt = unwrap_xyz_filt.transpose(2,1,0)

    if filter2d:
        unwrap_xyz_filt = np.array([gaussian(unwrap_xyz_pad[...,ch], sigma=sigma_window, preserve_range=True) for ch in np.arange(unwrap_xyz_pad.shape[-1])])
        unwrap_xyz_filt = unwrap_xyz_filt.transpose(1,2,0)

    unwrap_xyz_filt = unwrap_xyz_filt[3*sigma_window:-3*sigma_window].copy() # because of extra padding. 
    unwrap_xyz_filt = unwrap_xyz_filt[:,3*sigma_window:-3*sigma_window].copy()

    return unwrap_xyz_filt


def smooth_unwrap_params_3D_spherical_SmoothN(unwrap_xyz, 
                                              S=100, 
                                              isrobust=True, 
                                              pad_size=None):
    r""" Applies the Whittaker smoother, smoothN of David Garcia to smooth a (u,v) image parameterized (x,y,z) surface. This is effectively a spline based smoothing

    Parameters
    ----------
    unwrap_xyz : (UxVx3) array 
        the input image to smooth
    S : scalar
        controls the extent of smoothing, the higher the more smoothing.    
    isrobust : bool
        if True, runs the filtering in robust mode 
    pad_size : int
        if specified, pads the image by the given size else pads the image by replicating itself. 

    Returns
    -------
    mesh_smooth : (UxVx3) array
        the smoothed output image
    """
    from ..Image_Functions.smoothn import smoothn

    if pad_size is None:
        # pad the mesh..... 
        unwrap_xyz_pad = np.hstack([unwrap_xyz, unwrap_xyz, unwrap_xyz]) # this pad is correct. 
        unwrap_xyz_pad = np.vstack([unwrap_xyz_pad[1:][::-1,::-1],
                                    unwrap_xyz_pad,
                                    unwrap_xyz_pad[:-1][::-1,::-1]])
    else:
        unwrap_xyz_pad = pad_unwrap_params_3D_spherical(unwrap_xyz, pad=pad_size)

    # mesh_smooth_x,s,exitflag,Wtot = smoothn(unwrap_xyz_pad[...,0], s=S, isrobust=isrobust)
    # mesh_smooth_y,s,exitflag,Wtot = smoothn(unwrap_xyz_pad[...,1], s=S, isrobust=isrobust)
    # mesh_smooth_z,s,exitflag,Wtot = smoothn(unwrap_xyz_pad[...,2], s=S, isrobust=isrobust)
    mesh_smooth,s,exitflag,Wtot = smoothn(unwrap_xyz_pad, s=S, isrobust=isrobust)
    # mesh_smooth = np.dstack([mesh_smooth_x, 
    #                          mesh_smooth_y, 
    #                          mesh_smooth_z])

    if pad_size is None:
        mesh_smooth = mesh_smooth[len(unwrap_xyz)-1:-len(unwrap_xyz)+1]
        mesh_smooth = mesh_smooth[:,unwrap_xyz.shape[1]:-unwrap_xyz.shape[1]]
    else:
        mesh_smooth = mesh_smooth[pad_size:-pad_size].copy()
        mesh_smooth = mesh_smooth[:,pad_size:-pad_size].copy()

    return mesh_smooth


def _check_border_uv_label_overlaps(uv_labels, 
                                    labels1, 
                                    labels2, 
                                    dilate_ksize=1,
                                    min_size=10,
                                    S_uv=None,
                                    use_real_size=False):
    
    import skimage.morphology as skmorph
    import numpy as np 
    
    if (use_real_size == True) and (S_uv is not None):
        dA = surface_area_uv(S_uv, eps=1e-12, pad=True) # compute per pixel area. 
        dA[np.isinf(dA)>0] = 0 # replace infs. 
        
    pairings = []
    
    for lab_ii in np.arange(len(labels1)):
        lab = labels1[lab_ii]
        mask_label = uv_labels == lab
        mask_label = skmorph.binary_dilation(mask_label, skmorph.disk(dilate_ksize)) 
        
        # what does expanded cover. 
        covered_ids = uv_labels[mask_label>0]
        # valid exclude bg,itself and any other labels in labels1 i.e. must intersect labels2
        valid_covered_ids = np.intersect1d(np.setdiff1d(covered_ids, [0]+list(labels1)), labels2)
    
        if len(valid_covered_ids)>0:
            
            # check for the pixel coverage. in valid ids. 
            if (use_real_size == True) and (S_uv is not None):
                covered_number = np.hstack([np.nansum((covered_ids==idd)*1.*dA) for idd in valid_covered_ids]) # weight by the differential area.
            else:
                covered_number = np.hstack([np.sum(covered_ids==idd) for idd in valid_covered_ids]) # just raw pixel number
            
            # size check 
            select = covered_number>=min_size
            
            if np.sum(select)>0:
                valid_covered_ids = valid_covered_ids[select] 
                covered_number = covered_number[select]
        
                pairing = np.hstack([lab, valid_covered_ids]) # these should be joined as one item
                pairings.append(pairing)
    
    return pairings
        
    
def correct_border_uv_segmentations(uv_labels, 
                                    border_size=10,
                                    dilate_ksize=1, 
                                    min_size=10, 
                                    S_uv=None, 
                                    use_real_size=False):
    r""" Given a label 2D image, this function implements assumes this is from a spherical unwrap and checks the four borders to see if any labels should be merged. Return is a new label image.
    
    Parameters
    ----------
    uv_labels : (UxV) array 
        the input label image where 0 is background, each unique positive integer is an instance segmentation
    border_size : int
        the border size to consider. Segmentations within uv_labels[border_size:-border_size,border_size:-border_size] are not considered as merging candidates
    dilate_ksize : int
        the label dilation amount to test for overlap with labels to merge   
    min_size : int
        the minimum size of overlap with another label for merging to occur. If S_uv is provided and use_real_size is True, then the overlap is measured in terms of the 3D surface area. If not, it is the number of shared pixels in the image.
    S_uv : (UxVx3) array
        The mapping giving the corresponding 3D xyz coordinates (last channel) in correspondence with each pixel. Used to compute the 3D surface area each pixel corresponds to 
    use_real_size: bool
        If True, interpret min_size as the minimum surface area of overlap between two segmentations for a merge. 

    Returns
    -------
    uv_labels_corrected : (UxV) array
        the corrected label image with possibly the labels at the boundaries corrected to refer to the same 3D object, instead of being split into multiple due to the unwrapping.
    """
    import numpy as np 
    
    V, U = uv_labels.shape[:2]
    uv_labels_pad = pad_unwrap_params_3D_spherical(uv_labels, pad=border_size) 
    
    if S_uv is not None:
        S_uv_pad = pad_unwrap_params_3D_spherical(S_uv, pad=border_size) 
    else:
        S_uv_pad = None # set as None
    
    """
    Construct the border points to check
    """
    fg_points = np.argwhere(uv_labels>0) # this is shared. 
    all_pairs = []
    
    """
    # 1a. check left or right (for periodic)
    """
    # check left-border with u-coordinate
    border_left_pts=fg_points[fg_points[:,1]<=border_size]
    border_left_labels = np.setdiff1d(np.unique(uv_labels[border_left_pts[:,0], border_left_pts[:,1]]),0)
    
    # find border_right_labels because they must choose from here
    border_right_pts=fg_points[fg_points[:,1]>=U-border_size]
    border_right_labels = np.setdiff1d(np.unique(uv_labels[border_right_pts[:,0], border_right_pts[:,1]]),0)
    
    # try pairing
    pairings_left_right = _check_border_uv_label_overlaps(uv_labels_pad,
                                                          border_left_labels,
                                                          border_right_labels,
                                                          dilate_ksize=dilate_ksize,
                                                          min_size=min_size,
                                                          S_uv=S_uv_pad,
                                                          use_real_size=use_real_size)
    all_pairs+=pairings_left_right
    
    """
    # 1b. top left / right 
    """
    ### check top and further divide into halves.
    border_top_pts_left=fg_points[np.logical_and(fg_points[:,0]<=border_size, fg_points[:,1]<=U//2)]
    border_top_pts_right=fg_points[np.logical_and(fg_points[:,0]<=border_size, fg_points[:,1]>U//2)]
    
    border_top_labels_left = np.setdiff1d(np.unique(uv_labels[border_top_pts_left[:,0], border_top_pts_left[:,0]]),0)
    border_top_labels_right = np.setdiff1d(np.unique(uv_labels[border_top_pts_right[:,0], border_top_pts_right[:,0]]),[0]+list(border_top_labels_left))
    
    # try pairing
    pairings_top_left_right = _check_border_uv_label_overlaps(uv_labels_pad,
                                                                border_top_labels_left,
                                                                border_top_labels_right,
                                                                dilate_ksize=dilate_ksize,
                                                                min_size=min_size,
                                                                S_uv=S_uv,
                                                                use_real_size=use_real_size)
    all_pairs+=pairings_top_left_right
    
    """
    # 1c. bottom left / right 
    """
    ### check bottom and further divide into halves.
    border_bottom_pts_left=fg_points[np.logical_and(fg_points[:,0]>=V-border_size, fg_points[:,1]<=U//2)]
    border_bottom_pts_right=fg_points[np.logical_and(fg_points[:,0]>=V-border_size, fg_points[:,1]>U//2)]
    
    border_bottom_labels_left = np.setdiff1d(np.unique(uv_labels[border_bottom_pts_left[:,0], border_bottom_pts_left[:,0]]),0)
    border_bottom_labels_right = np.setdiff1d(np.unique(uv_labels[border_bottom_pts_right[:,0], border_bottom_pts_right[:,0]]),[0]+list(border_bottom_labels_left))
    
    # try pairing
    pairings_bottom_left_right = _check_border_uv_label_overlaps(uv_labels_pad,
                                                                border_bottom_labels_left,
                                                                border_bottom_labels_right,
                                                                dilate_ksize=dilate_ksize,
                                                                min_size=min_size,
                                                                S_uv=S_uv,
                                                                use_real_size=use_real_size)
    all_pairs+=pairings_bottom_left_right
    
    # print(all_pairs) # only for checking.
    uv_labels_corrected = uv_labels.copy() # make a copy for saving
    
    if len(all_pairs)>0:
        """
        2. find clique -> assign all of those regions to new id 
        """
        uniq_cliques = []

        for pair in all_pairs:
            if len(uniq_cliques) == 0: # first iter
                uniq_cliques.append(pair)
            else:
                # check for intersection 
                add_cliq = True
                for cliq_ii in np.arange(len(uniq_cliques)):
                    cliq=uniq_cliques[cliq_ii]
                    intersect = np.intersect1d(cliq, pair) # does it intersect
                    if len(intersect)>0:
                        uniq_cliques[cliq_ii] = np.unique(np.hstack([cliq, pair]))
                        add_cliq = False
                        break
                if add_cliq:
                    uniq_cliques.append(pair) # start anew

        # print('found unique cliques')
        # print(uniq_cliques)
        """
        3. implement joining in clique, without loss assign rest to first label in a clique
        """
        for cliq in uniq_cliques:
            primary = cliq[0]
            rest = cliq[1:]
            
            for rr in rest:
                uv_labels_corrected[uv_labels==rr] = primary # set to primary's label
        
    return uv_labels_corrected

def impute_2D_spherical(img, method='linear', pad_size=None, blank_pixels=None):
    r""" Imputes an image with spherical boundary conditions using interpolation 

    Parameters
    ----------
    img : (UxV) array 
        the input image to impute with missing pixels indicated by np.isnan (by default)
    method : str
        interpolation method, one of the possible option specified in scipy.interpolate.griddata. Linear interpolation is used by default 
    pad_size : int
        if specified, pads the image by the given size else pads the image by replicating itself fully
    blank : scalar
        if specified, the alternative pixel intensity used to label the pixels to be imputed. By default we use np.isnan

    Returns
    -------
    fill_interp : (UxV) array
        the imputed image
    """
    import scipy.interpolate as scinterpolate
    import numpy as np 

    # prepad. 
    if pad_size is None:
        # pad the mesh..... 
        img_pad = np.hstack([img, img, img]) # this pad is correct. 
        img_pad = np.vstack([img_pad[1:][::-1,::-1],
                             img_pad,
                             img_pad[:-1][::-1,::-1]])
    else:
        img_pad = pad_unwrap_params_3D_spherical(img, pad=pad_size)

    YY, XX = np.indices(img_pad.shape)
    
    if blank_pixels is None:
        valid_mask = np.logical_not(np.isnan(img_pad))
    else:
        valid_mask = np.logical_not(img_pad==blank_pixels) 
    
    YYXX = np.vstack([YY[valid_mask], 
                      XX[valid_mask]]).T
    
    fill_interp = scinterpolate.griddata(YYXX, 
                                         img_pad[valid_mask], 
                                         (YY, XX), method=method, fill_value=np.nan)
    
    if pad_size is None:
        M, N = img.shape[:2]
        fill_interp = fill_interp[M:-M].copy() # crop out the row
        fill_interp = fill_interp[:,N:-N].copy() # crop out the col
    else:
        fill_interp = fill_interp[pad_size:-pad_size].copy()
        fill_interp = fill_interp[:,pad_size:-pad_size].copy()
        
    return fill_interp 


def find_principal_axes_uv_surface(uv_coords, pts_weights, map_to_sphere=True, sort='ascending', return_pts=False):
    """ Find the principal axes of a uv parametrized surface with given pixel weights.  
    
    Parameters 
    ----------
    uv_coords : (UxVx3) array 
        the input image specifying the uv unwrapped (x,y,z) surface to find the principal axis of in original Cartesian (x,y,z) space
    pts_weights : (UxV) array
        the positive weights at each pixel, specifying its relative importance in the computation of the principal axes 
    map_to_sphere : bool
        if True, the unit sphere coordinate i.e. spherical parametrization of the uv grid is used instead of the actual (x,y,z) coordinate positions to compute principal axes. This enables geometry-independent computation useful for e.g. getting directional alignment based only on surface intensity
    sort : 'ascending' or 'descending'
        the sorting order of the eigenvectors in terms of the absolute value of the respective eigenvalues
    return_pts : bool
        if True, return the demeaned points used to compute the principal axes

    Returns 
    -------
    w : (3,) array
        the sorted eigenvalues of the three principal eigenvectors
    v : (3,3) array    
        the sorted eigenvectors of the corresponding eigenvalues
    pts_demean : (UxVx3) array 
        the demeaned points used for doing the principal components analysis

    See Also
    --------
    :func:`unwrap3D.Mesh.meshtools.find_principal_axes_surface_heterogeneity` : 
        Equivalent for finding the principal eigenvectors when give a surface mesh of unordered points. 

    """
    import numpy as np 
    from ..Geometry import geometry as geometry

    pts = uv_coords.reshape(-1, uv_coords.shape[-1])
    pts_weights = pts_weights.ravel()

    if map_to_sphere:
        # we replace the pts with that of the unit sphere (x,y,z) 
        m, n = uv_coords.shape[:2] 
        psi_theta_grid_ref = geometry.img_2_angles(m,n)
        pts_demean = geometry.sphere_from_img_angles(psi_theta_grid_ref.reshape(-1,psi_theta_grid_ref.shape[-1]))
    else:
        # demean 
        pts_mean = np.mean(pts, axis=0)
        pts_demean = pts - pts_mean[None,:]

    # unweighted version. 
    pts_demean_ = pts_demean * pts_weights[:,None] / float(np.sum(pts_weights))
    pts_cov = np.cov(pts_demean_.T) # 3x3 matrix #-> expect symmetric. 

    w, v = np.linalg.eigh(pts_cov)
    # sort large to small. 
    if sort=='descending':
        w_sort = np.argsort(np.abs(w))[::-1]
    if sort=='ascending':
        w_sort = np.argsort(np.abs(w))

    w = w[w_sort]
    v = v[:,w_sort]

    if return_pts:
        pts_demean = pts_demean.reshape(uv_coords.shape)
        return w, v, pts_demean
    else:
        return w, v 



def uv_map_sphere_surface_parameterization(sphere_mesh, 
                                           surface_mesh,
                                           surface_measurements_scalars=None,
                                           surface_measurements_labels=None,
                                           uv_grid_size = 256, 
                                           optimize_aspect_ratio=True, 
                                           aspect_ratio_method='beltrami',
                                           length_ratio_avg_fn=np.nanmean,
                                           h_opt=None):
    
    r""" Given a (u,v) parametrization of an (x,y,z) surface. This function derives the corresponding (u,v) parametrization for an arbitrary rotation of the (x,y,z) surface by rotating the spherical parametrization and interpolating without needing to completely do a new spherical parametrization. 
   
    Parameters 
    ----------
    sphere_mesh : trimesh.Trimesh
        spherical parameterization of surface_mesh. Must have same number of vertices and faces. 
    surface_mesh : trimesh.Trimesh
        corresponding surface_mesh. Must have same number of vertices and faces as the sphere mesh.  
    surface_measurements_scalars : (n_vertices, D) array
        D number of real-valued scalar measurements such as curvature which is desired to be resampled onto (u,v) grid
    surface_measurements_labels : (n_vertices, D) array
        D number of integer discrete labels which to be mapped into (u,v) space using statistical mode and not to be interpolated. e.g. protrusion segmentation.
    uv_grid_size : int
        vertical height of (u,v) grid, i.e. the height of the image. This is fixed. The width may be optimized if  optimize_aspect_ratio=True. Otherwise the width is double the height.
    optimize_aspect_ratio : bool
        if True, optimize the width of the image to minimize metric distortion, based on the chosen method. 
    aspect_ratio_method : str 
        optimization method for aspect ratio. Either 'length_ratio', see :func:`unwrap3D.Unzipping.unzip.length_ratio_uv_opt` or 'Beltrami' to use the Beltrami coefficient, see :func:`unwrap3D.Unzipping.unzip.beltrami_coeff_uv_opt` 
    length_ratio_avg_fn : numpy function
        function used to compute an average length for u and v separately to determine h_opt when 'length_ratio' method is specified. Default is mean: np.nanmean. However, you might find np.nanmedian to get better looking results due to mode-seeking behavior of median vs mean. 
    h_opt : float
        a user set ratio for uv image width. uv image size will be (uv_grid_size, 2*((int(h_opt*M)+1)//2))
    

    Returns 
    -------
    S_uv : (M,N,3) array
        the uv parameterized (x,y,z) surface coordinates 
    match_params : [(UxV,) array, (UxV,3) array]
        list of the triangle on sphere each (u,v) pixel is matched to and second array gives the barycentric weight of sphere vertices.
    h_opt : scalar
        the optimal aspect ratio found. The output S_uv has shape: M = grid_size and N = 2*((int(h_opt*M)+1)//2)
    uv_surface_measurements_scalars : (M,N,D) array
        empty list if no input, otherwise the remapped D scalar measurements into (u,v) space
    uv_surface_measurements_labels : (M,N,D) array
        empty list if no input, otherwise the remapped D label measurements into (u,v) space. Labels unlike scalars are not interpolated but mapped by mode.
    """
    
    from ..Geometry import geometry as unwrap3D_geom
    from ..Mesh import meshtools as unwrap3D_meshtools
    import scipy.stats as spstats
    
    if h_opt is None:
        v_uv_grid_size = uv_grid_size
        h_uv_grid_size = 2*uv_grid_size
        ang_grid = unwrap3D_geom.img_2_angles(v_uv_grid_size, h_uv_grid_size)  
    else:
        # make this even!. 
        v_uv_grid_size = uv_grid_size
        h_uv_grid_size = 2*((int(h_opt*uv_grid_size)+1)//2) # guarantee a multiple of 2.
        ang_grid = unwrap3D_geom.img_2_angles(uv_grid_size, h_uv_grid_size)  
    
    unit_sphere_xyz = unwrap3D_geom.sphere_from_img_angles(ang_grid.reshape(-1,ang_grid.shape[-1]))
    unit_sphere_xyz = unit_sphere_xyz.reshape((v_uv_grid_size,h_uv_grid_size,3))

    match_params = unwrap3D_meshtools.match_and_interpolate_uv_surface_to_mesh(unit_sphere_xyz.reshape(-1,3), 
                                                                                     sphere_mesh, 
                                                                                     match_method='cross')

    # Barycentric interpolation to pull down S_ref with matching equiareal
    S_uv = unwrap3D_meshtools.mesh_vertex_interpolate_scalar(sphere_mesh, 
                                                    match_params[0], 
                                                    match_params[1], 
                                                    surface_mesh.vertices)
    S_uv = S_uv.reshape((v_uv_grid_size,h_uv_grid_size,-1)) 
    
    
    # optimize aspect ratio. 
    if optimize_aspect_ratio and h_opt is None:
        
        if aspect_ratio_method=='beltrami':
            h_opt = beltrami_coeff_uv_opt(S_uv, 
                                          eps=1e-12, 
                                          apply_opt=False)
        elif aspect_ratio_method == 'length_ratio':
            h_opt = length_ratio_uv_opt(S_uv, 
                                        avg_fn=np.nanmedian,
                                        eps=1e-12, 
                                        apply_opt=False)
        else:
            print('invalid method')
            
        # recreate the unwrap params with this new and rematch. 
        v_uv_grid_size = uv_grid_size
        h_uv_grid_size = 2*((int(h_opt*uv_grid_size)+1)//2) # guarantee a multiple of 2.
        ang_grid = unwrap3D_geom.img_2_angles(uv_grid_size, h_uv_grid_size)  
    
        unit_sphere_xyz = unwrap3D_geom.sphere_from_img_angles(ang_grid.reshape(-1,ang_grid.shape[-1]))
        unit_sphere_xyz = unit_sphere_xyz.reshape((v_uv_grid_size,h_uv_grid_size,3))
    
        match_params = unwrap3D_meshtools.match_and_interpolate_uv_surface_to_mesh(unit_sphere_xyz.reshape(-1,3), 
                                                                                         sphere_mesh, 
                                                                                         match_method='cross')
    
        # Barycentric interpolation to pull down S_ref with matching equiareal
        S_uv = unwrap3D_meshtools.mesh_vertex_interpolate_scalar(sphere_mesh, 
                                                        match_params[0], 
                                                        match_params[1], 
                                                        surface_mesh.vertices)
        S_uv = S_uv.reshape((v_uv_grid_size,h_uv_grid_size,-1)) 
    
    uv_surface_measurements_scalars=[]
    uv_surface_measurements_labels=[]
    
    # if measurements is supplied, simulatenously map these too. 
    if surface_measurements_scalars is not None:
        uv_surface_measurements_scalars = unwrap3D_meshtools.mesh_vertex_interpolate_scalar(sphere_mesh, 
                                                                                            match_params[0], 
                                                                                            match_params[1], 
                                                                                            surface_measurements_scalars)
        uv_surface_measurements_scalars = uv_surface_measurements_scalars.reshape((v_uv_grid_size, h_uv_grid_size, -1))
    if surface_measurements_labels is not None:
        uv_surface_measurements_labels = []
        for label_ii in np.arange(surface_measurements_labels.shape[-1]):
            surf_label = surface_measurements_labels[...,label_ii].copy()
            
            out_labels = spstats.mode(surf_label[sphere_mesh.faces[match_params[0]]], axis=-1)[0]
            uv_surface_measurements_labels.append(np.squeeze(out_labels))
        uv_surface_measurements_labels = np.array(uv_surface_measurements_labels).T
        uv_surface_measurements_labels = uv_surface_measurements_labels.reshape((v_uv_grid_size, h_uv_grid_size, -1))
            
    return S_uv, match_params, h_opt, uv_surface_measurements_scalars, uv_surface_measurements_labels
        

##### unwrap_params_rotate, integrating the beltrami coefficient optimization. 
def unwrap_params_rotate_coords(unwrap_params, 
                                rot_matrix, 
                                invert_transform=True,
                                method='spline', 
                                cast_uint8=False,
                                optimize=False,
                                h_range=[0.1,5], 
                                eps=1e-12):

    r""" Given a (u,v) parametrization of an (x,y,z) surface. This function derives the corresponding (u,v) parametrization for an arbitrary rotation of the (x,y,z) surface by rotating the spherical parametrization and interpolating without needing to completely do a new spherical parametrization. 
   
    Parameters 
    ----------
    unwrap_params : (UxVx3) array 
        the input image specifying the uv unwrapped (x,y,z) surface 
    rot_matrix : (4x4) array
        the specified rotation given as a 4x4 homogeneous rotation matrix  
    invert_transform : bool
        if True, the rotation matrix is applied inverted to get the corresponding coordinates forming the new (u,v) parametrization. This should be set to true if the rotation_matrix is the forward transformation that rotates the current surface to the new surface.  
    method : str 
        the interpolation method. Either of 'spline' to use scipy.interpolate.RectBivariateSpline or one of those available for scipy.interpolate.RegularGridInterpolator
    optimize : bool
        if True, the aspect ratio of the image will be optimized to minimize metric distortion based on the Beltrami coefficient, see :func:`unwrap3D.Unzipping.unzip.beltrami_coeff_uv_opt` 
    h_range : 2-tuple
        a list of aspect ratios to search for the optimal aspect ratio when optimize=True
    eps : scalar
        a small number for numerical stability 

    Returns 
    -------
    xy_rot : (M,N,2)
        the (u,v) coordinates of the input unwrap_params that form the new (u,v) of the rotated surface. The size (M,N) will be the same as input if optimize=False.
    unwrap_params_new : (M,N,3)
        the new uv unwrapped (x,y,z) surface coordinates denoting the rotated surface. 
    h_opt : scalar
        the optimal aspect ratio found within the given ``h_range`` which minimizes the Beltrami coefficient, if ``optimize=True``. The new size is (U, int(h_opt*U), 3) 

    """
    import numpy as np 
    from ..Geometry import geometry as geometry
    from ..Image_Functions.image import map_intensity_interp2
    # from LightSheet_Analysis.Geometry import meshtools
    # from LightSheet_Analysis.Geometry.geometry import compute_surface_distance_ref 

    """
    1. define the uv grid of the reference sphere. 
    """
    m, n = unwrap_params.shape[:2] 
    psi_theta_grid_ref = geometry.img_2_angles(m,n)

    """
    2. define the reference xyz of the unit sphere for pullback. 
    """
    sphere_xyz = geometry.sphere_from_img_angles(psi_theta_grid_ref.reshape(-1,psi_theta_grid_ref.shape[-1]))
    sphere_xyz_grid = sphere_xyz.reshape((m,n,-1))

    """
    3. apply the inverse intended rotation matrix to the reference sphere. 
        # rot_matrix is 4x4, so is just the inverse
    """
    if invert_transform==True:
        T_matrix = np.linalg.inv(rot_matrix)
    else:
        T_matrix = rot_matrix.copy()

    sphere_xyz_grid_rot = T_matrix.dot(np.vstack([sphere_xyz_grid.reshape(-1,sphere_xyz_grid.shape[-1]).T, 
                                                  np.ones(len(sphere_xyz_grid.reshape(-1,sphere_xyz_grid.shape[-1])))]))[:sphere_xyz_grid.shape[-1]]
    sphere_xyz_grid_rot = sphere_xyz_grid_rot.T
    sphere_xyz_grid_rot = sphere_xyz_grid_rot.reshape(sphere_xyz_grid.shape) # this gives new (x,y,z) coordinates relating to the reference.


    # pull down now the transformed angles
    psi_theta_grid_ref_rot = geometry.img_angles_from_sphere(sphere_xyz_grid_rot.reshape(-1,sphere_xyz_grid_rot.shape[-1]))
    psi_theta_grid_ref_rot = psi_theta_grid_ref_rot.reshape((m, n, psi_theta_grid_ref.shape[-1]))
    
    # instead what we wdo is following https://github.com/henryseg/spherical_image_editing/blob/master/sphere_transforms_numpy.py
    # we turn the angles into x,y coordinates! then use normal rectangular interpolation !. 
    xy_rot = geometry.angles_2_img(psi_theta_grid_ref_rot, (m,n)) # this is the new grid. 

    """
    4. we now do the reinterpolation of scalars. 
    """
    # s is only used in the case of smoothing for spline based. 
    unwrap_params_new = np.array([map_intensity_interp2(xy_rot[...,::-1].reshape(-1, xy_rot.shape[-1]), grid_shape=(m,n), I_ref=unwrap_params[...,ch], method=method, cast_uint8=cast_uint8, s=0).reshape((m,n)) for ch in np.arange(unwrap_params.shape[-1])])
    unwrap_params_new = unwrap_params_new.transpose([1,2,0])

    """
    5. we should add in the beltrami coefficient optimization for the new view. 
    """
    if optimize:
        h_opt, unwrap_params_new = beltrami_coeff_uv_opt(unwrap_params_new, h_range=h_range, eps=eps, apply_opt=True) # optimize and return this. 

        return xy_rot, unwrap_params_new, h_opt
    else:
        return xy_rot, unwrap_params_new


def length_ratio_uv_opt(surface_uv_params, avg_fn=np.nanmean, eps=1e-12, apply_opt=True):
    r""" Find the optimal image aspect ratio to minimise the l_u / l_v where l_u is the mean length of traversing in u-direction in xyz, and l_v is the mean distance traversing in v-direction 

    This is based on generalizing uv-map which chooses aspect ratio to preserve arc lengths of sphere equator i.e. (M x 2*M). The result now is (M x h*M) 
    
    Parameters
    ----------
    surface_uv_params : (UxVx3) array
        the input image specifying the uv unwrapped (x,y,z) surface 
    avg_fn : numpy function
        the average function to average computed xyz space across all u- and v- lines.      
    eps : scalar
        small number for numerical stability 
    apply_opt : bool
        if True, additionally return the resized surface as a second output

    Returns
    -------
    h_opt : scalar
        the optimal scaling factor within the specified h_range
    surface_uv_params_resize_opt : (U x W x 3)
        if apply_opt=True, the found aspect ratio is applied to the input image where the new width W is int(h_opt)*V. 
    """
    import numpy as np 
    import skimage.transform as sktform
    
    mean_length_rows = avg_fn(np.nansum(np.linalg.norm(np.diff(surface_uv_params, axis=0), axis=-1), axis=0))
    mean_length_cols = avg_fn(np.nansum(np.linalg.norm(np.diff(surface_uv_params, axis=1), axis=-1), axis=1))

    h_opt = mean_length_cols/mean_length_rows

    m, n = surface_uv_params.shape[:2]

    if apply_opt:

        surface_uv_params_resize_opt = np.array([sktform.resize(surface_uv_params[...,ch], (m, int(h_opt*m)), preserve_range=True) for ch in range(surface_uv_params.shape[-1])])
        surface_uv_params_resize_opt = surface_uv_params_resize_opt.transpose(1,2,0)

        return h_opt, surface_uv_params_resize_opt
    else:
        return h_opt 
    
    

def beltrami_coeff_uv_opt(surface_uv_params, h_range=[0.1,5], eps=1e-12, apply_opt=True):
    r""" Find the optimal image aspect ratio to minimise the Beltrami coefficient which is a measure of metric distortion for a (u,v) parametrized (x,y,z) surface 

    See https://en.wikipedia.org/wiki/Beltrami_equation 
    
    Parameters
    ----------
    surface_uv_params : (UxVx3) array
        the input image specifying the uv unwrapped (x,y,z) surface 
    h_range : 2-tuple
        specifies the [min, max] scaling factor of image width relative to height to search for the minimal Beltrami Coefficient      
    eps : scalar
        small number for numerical stability 
    apply_opt : bool
        if True, additionally return the resized surface as a second output

    Returns
    -------
    h_opt : scalar
        the optimal scaling factor within the specified h_range
    surface_uv_params_resize_opt : (U x W x 3)
        if apply_opt=True, the found aspect ratio is applied to the input image where the new width W is int(h_opt)*V. 
    """
    from scipy.optimize import fminbound
    import skimage.transform as sktform 
    import numpy as np 

    m, n = surface_uv_params.shape[:2]

    def opt_score(h):
        # surface_unwrap_params_rot_resize = np.array([sktform.resize(surface_unwrap_params_rot[...,ch], (int(h*n),n), preserve_range=True) for ch in range(surface_unwrap_params_rot.shape[-1])])
        surface_uv_params_resize = np.array([sktform.resize(surface_uv_params[...,ch], (m, int(h*m)), preserve_range=True) for ch in range(surface_uv_params.shape[-1])])
        surface_uv_params_resize = surface_uv_params_resize.transpose(1,2,0)
        
        beltrami_coeff = beltrami_coeff_uv(surface_uv_params_resize, eps=eps)
        score = np.nanmean(np.abs(beltrami_coeff)**2) # using mean works. 

        return score

    h_opt = fminbound(opt_score, h_range[0], h_range[1]);

    if apply_opt:

        surface_uv_params_resize_opt = np.array([sktform.resize(surface_uv_params[...,ch], (m, int(h_opt*m)), preserve_range=True) for ch in range(surface_uv_params.shape[-1])])
        surface_uv_params_resize_opt = surface_uv_params_resize_opt.transpose(1,2,0)

        return h_opt, surface_uv_params_resize_opt
    else:
        return h_opt 


def beltrami_coeff_uv(surface_uv_params, eps=1e-12):
    r""" Computes the Beltrami coefficient, a measure of uniform metric distortion given a (u,v) parametrized (x,y,z) surface  

    See https://en.wikipedia.org/wiki/Beltrami_equation for mathematical definition

    Parameters
    ----------
    surface_uv_params : (U,V,3) array
        the input image specifying the uv unwrapped (x,y,z) surface 
    eps : scalar
        small number for numerical stability 

    Returns
    -------
    mu : (U,V) array 
        Complex array giving the Beltrami coefficient at each (u,v) pixel position

    """
    # https://en.wikipedia.org/wiki/Beltrami_equation, return np.sum(np.abs(beltrami_coefficient(np.hstack([square_x[:,None],h*square_y[:,None]]),f,v))**2)
    from skimage.transform import resize
    import numpy as np 

    # compute according to wikipedia.... 
    dS_du = np.gradient(surface_uv_params, axis=0); 
    dXdu = dS_du[...,0].copy()
    dYdu = dS_du[...,1].copy()
    dZdu = dS_du[...,2].copy()
    dS_dv = np.gradient(surface_uv_params, axis=1);
    dXdv = dS_dv[...,0].copy()
    dYdv = dS_dv[...,1].copy()
    dZdv = dS_dv[...,2].copy()

    E = dXdu**2 + dYdu**2 + dZdu**2;
    G = dXdv**2 + dYdv**2 + dZdv**2;
    F = dXdu*dXdv + dYdu*dYdv + dZdu*dZdv;
    mu = (E - G + 2 * 1j * F) / ((E + G + 2.*np.sqrt(E*G - F**2))+eps);

    return mu


def gradient_uv(surface_uv_params, eps=1e-12, pad=False):
    r""" Compute the Jacobian of the uv parametrized (x,y,z) surface i.e. :math:`\partial S/\partial u` and `\partial S/\partial v`

    Parameters
    ----------
    surface_uv_params : (UxVx3) array
        the input image giving the uv unwrapped (x,y,z) surface 
    eps : scalar 
        small numerical value for numerical stability 
    pad : bool 
        if True, spherically pads by 1 pixel top and right to compute 1st order finite differences, if False using np.gradient to compute central differences

    Returns
    -------
    dS_du : (UxVx3) array   
        :math:`\partial S/\partial u`, the change in the (x,y,z) surface coordinates in the direction of image u- axis (horizontal) 
    dS_dv : (UxVx3) array 
        :math:`\partial S/\partial v`, the change in the (x,y,z) surface coordinates in the direction of image v- axis (vertical) 
    """
    # if pad then wrap, else use the gradient. 
    if pad:
        surface_uv_params_ = np.hstack([surface_uv_params, 
                                        surface_uv_params[:,1][:,None]])
        surface_uv_params_ = np.vstack([surface_uv_params_, 
                                        surface_uv_params_[-2][None,::-1]])
        dS_du = surface_uv_params_[:-1,1:] - surface_uv_params_[:-1,:-1]
        dS_dv = surface_uv_params_[1:,:-1] - surface_uv_params_[:-1,:-1]
    else:
        dS_du = np.gradient(surface_uv_params, axis=1);
        dS_dv = np.gradient(surface_uv_params, axis=0); 

    return dS_du, dS_dv


def find_stop_inner_topography_and_resample(topography_space,
                                            stop_S_ref, 
                                            alpha_step,
                                            n_samples=None, 
                                            n_cpu=None,
                                            convergence_tol = 1,
                                            invert_direction=True,
                                            cutoff_ALS_smooth=1e2):


    import point_cloud_utils as pcu 
    import multiprocess as mp
    import numpy as np 
    import scipy.interpolate as sinterp
    import time 


    if invert_direction: 
        Suv_prop_inside = topography_space[::-1].copy()
    else:
        Suv_prop_inside = topography_space.copy()

    if n_samples is None:
        # used if n_samples = None
        dist_moved_total = np.sum(np.linalg.norm(np.diff(Suv_prop_inside, axis=0), axis=-1) , axis=0) 

    """
    #   criteria 1: find the cutoffs based on the deviation from the final point
    """
    # compute this final distance ( and find the first cutoff to hit within a tol)
    dist_to_final = np.linalg.norm(Suv_prop_inside - Suv_prop_inside[-1][None,...], axis=-1)
    dist_to_final_bool = dist_to_final > convergence_tol
    dist_to_final_zeros = np.sum(dist_to_final_bool==0, axis=0)
    
    cutoff_dist = np.argmin(dist_to_final_bool, axis=0)  #### this works. 
    
    # fix for the case when the condition is never satisfied. 
    if np.sum(dist_to_final_zeros==0) > 0: 
        cutoff_dist[dist_to_final_zeros==0] = len(Suv_prop_inside)-1
    
    """
    #   criteria 2: find the cutoffs to the ref 
    """
    dists_ref, _ = pcu.k_nearest_neighbors(Suv_prop_inside.reshape(-1,3), 
                                              stop_S_ref, k=1)
    dists_ref = dists_ref.reshape(Suv_prop_inside.shape[:3]) # get back in the same shape. 
    dist_ref_bool = dists_ref > convergence_tol
    dist_to_ref_zeros = np.sum(dist_ref_bool == 0, axis=0)
        
    cutoff_ref = np.argmin(dist_ref_bool, axis=0)
        
    if np.sum(dist_to_ref_zeros==0) > 0: 
        cutoff_ref[dist_to_ref_zeros==0] = len(Suv_prop_inside)-1

    """
    # combine both criteria taking whichever is first satisfied. 
    """    
    cutoff_actual = np.minimum(cutoff_dist, cutoff_ref)
        
    
    """
    implement smoothing -> this is ensure continuity
    """
    cutoffs_pad = pad_unwrap_params_3D_spherical(cutoff_actual, pad=cutoff_actual.shape[0]//2)
    
    # detect bad values.
    cutoff_gradients = np.array(np.gradient(cutoffs_pad))
    invalid_mask = np.linalg.norm(cutoff_gradients, axis=0)>1.
    valid_mask = np.logical_not(invalid_mask)
    
    from scipy.interpolate import griddata
    # build interpolation space. 
    # now create a griddata. 
    rows, cols = np.indices(invalid_mask.shape)
    rows_ = rows[valid_mask]
    cols_ = cols[valid_mask]
    
    interp_cutoffs_pad = sinterp.griddata(np.hstack([cols_[:,None], rows_[:,None]]), 
                                  cutoffs_pad[valid_mask],(cols, rows), method='linear', 
                                  fill_value=np.nan, rescale=False)
    
    interp_cutoffs = interp_cutoffs_pad[cutoff_actual.shape[0]//2:-cutoff_actual.shape[0]//2, cutoff_actual.shape[0]//2:-cutoff_actual.shape[0]//2]
    
    # instead try an ALS? 
    cutoff_actual = topotools.penalized_smooth_topography_uv_img(interp_cutoffs, 
                                                                    ds=4, 
                                                                    padding_multiplier=4, 
                                                                    method='ALS', 
                                                                    lam=cutoff_ALS_smooth,
                                                                    p=0.5, 
                                                                    niter=10,
                                                                    uv_params=None)
    cutoff_actual = cutoff_actual.astype(np.int32)
    cutoff_actual = np.clip(cutoff_actual,0, len(topography_space)-1)
    """
    interpolate - resample 
    """
    # flatten this so we can list everything. 
    Suv_prop_inside_flat = Suv_prop_inside.reshape(Suv_prop_inside.shape[0], -1,3)
    cutoffs_flat = cutoff_actual.ravel()
    
    t1 = time.time()
    # build the interpolators (fast as list comprehension)    
    all_fs = [sinterp.interp1d(np.arange(cutoffs_flat[iii]+1)/(cutoffs_flat[iii]), 
                                Suv_prop_inside_flat[:cutoffs_flat[iii]+1,iii],
                                kind='linear', 
                                axis=0,
                                copy=True,
                                bounds_error=None,
                                assume_sorted=True) for iii in np.arange(len(cutoffs_flat))]
    print('building interpolators..., ', time.time()-t1, ' s')
    
    # this is much slower .... 
    t1 = time.time()
    # def _sample_new_coords(idx):
    #     if n_samples is None:
    #         return all_fs[idx](np.linspace(0,1.,int(np.nanmean(dist_moved_total)/alpha_step))) 
    #     else:
    #         return all_fs[idx](np.linspace(0,1.,int(n_samples))) 

    # with mp.Pool(n_cpu) as pool:
    #     Suv_prop_inside_flat = pool.map(_sample_new_coords, range(0, len(all_fs)))
    if n_samples is None:
        Suv_prop_inside_flat = [all_fs[iii](np.linspace(0,1.,int(np.nanmean(dist_moved_total)/alpha_step))) for iii in np.arange(len(all_fs))]
    else:
        Suv_prop_inside_flat = [all_fs[iii](np.linspace(0,1.,int(n_samples))) for iii in np.arange(len(all_fs))]
    print('resampling unwrapping coordinates..., ', time.time()-t1, ' s')
    
    Suv_prop_inside_flat = np.array(Suv_prop_inside_flat)
    Suv_prop_inside_flat = Suv_prop_inside_flat.transpose(1,0,2).reshape(Suv_prop_inside_flat.shape[1], Suv_prop_inside.shape[1], Suv_prop_inside.shape[2],3)

    Suv_prop_inside = Suv_prop_inside_flat.copy()
    
    del Suv_prop_inside_flat

    if invert_direction:
        Suv_prop_inside = Suv_prop_inside[::-1]
        cutoff_actual = (len(Suv_prop_inside)-1) - cutoff_actual # invert 

    return cutoff_actual, Suv_prop_inside


def gradient_uv_depth(depth_uv_params, eps=1e-12, pad=False):
    r""" Compute the Jacobian of the topography (d,u,v) parametrized (x,y,z) volume i.e. :math:`\partial V/\partial d`, :math:`\partial V/\partial u` and :math:`\partial V/\partial v`

    Parameters
    ----------
    depth_uv_params : (DxUxVx3) array
        the input volume image giving the topography (d,u,v) unwrapped (x,y,z) volume 
    eps : scalar 
        small numerical value for numerical stability 
    pad : bool 
        if True, spherically pads by 1 pixel depth, top and right to compute 1st order finite differences, if False use np.gradient to compute central differences

    Returns
    -------
    dV_dd : (DxUxVx3) array
        :math:`\partial V/\partial d`, the change in the (x,y,z) volume coordinates in the direction of image d- axis (depth) 
    dV_du : (DxUxVx3) array   
        :math:`\partial V/\partial u`, the change in the (x,y,z) volume coordinates in the direction of image u- axis (horizontal) 
    dV_dv : (DxUxVx3) array 
        :math:`\partial V/\partial v`, the change in the (x,y,z) volume coordinates in the direction of image v- axis (vertical) 
    """
    # if pad then wrap, else use the gradient. 
    if pad:
        depth_uv_params_ = np.vstack([depth_uv_params, 
                                      depth_uv_params[-1][None,:]])
        dV_dd = depth_uv_params_[1:] - depth_uv_params_[:-1] # difference in d

        depth_uv_params_ = np.dstack([depth_uv_params, 
                                      depth_uv_params[:,:,-1][:,:,None,:]])
        dV_du = depth_uv_params_[:,:,1:] - depth_uv_params_[:,:,:-1] # difference in d

        depth_uv_params_ = np.hstack([depth_uv_params, 
                                      depth_uv_params[:,-2][:,None,::-1,:]])
        dV_dv = depth_uv_params_[:,1:] - depth_uv_params_[:,:-1]
    else:
        dV_dd = np.gradient(depth_uv_params, axis=0);
        dV_du = np.gradient(depth_uv_params, axis=2);
        dV_dv = np.gradient(depth_uv_params, axis=1); 

    return dV_dd, dV_du, dV_dv


def conformality_error_uv(surface_uv_params, eps=1e-12, pad=False):
    r""" Compute the Quasi-conformal error for the uv parametrized (x,y,z) surface which is defined by the ratio of the largest to the smallest singular values of the Jacobian (see :func:`unwrap3D.Unzipping.unzip.gradient_uv`)

    .. math :: 
        \mathcal{Q} = |\sigma_{max}| / |\sigma_{min}|

    where :math:`\sigma_{min}` and :math:`\sigma_{max}` denote the smaller and larger of the eigenvalues of :math:`J^T J`, where :math:`J` is the Jacobian matrix of the surface with respect to (u,v).
    
    Parameters
    ----------
    surface_uv_params : (UxVx3) array
        the input image giving the uv unwrapped (x,y,z) surface 
    eps : scalar 
        small numerical value for numerical stability 
    pad : bool 
        if True, spherically pads by 1 pixel top and right to compute 1st order finite differences, if False using np.gradient to compute central differences

    Returns
    -------
    stretch_factor : (UxV) array   
        The quasi-conformal error at each pixel 
    mean_stretch_factor : scalar
        The area weighted mean quasi-conformal error summarising the overall conformal error for the surface 
    """
    # this is just the gradient area. 
    import numpy as np 

    dS_du, dS_dv = gradient_uv(surface_uv_params, eps=eps, pad=pad)
    # compile the jacobian.
    Jac = np.concatenate([dS_du[None,...], 
                          dS_dv[None,...]], axis=0)
    Jac = Jac.transpose(1,2,3,0) # transpose. 
    
    Jac2 = np.matmul(Jac.transpose(0,1,3,2), Jac)
    stretch_eigenvalues, stretch_eigenvectors = np.linalg.eigh(Jac2) # compute J_T.dot(J) which is the first fundamental form.

    stretch_factor = np.sqrt(np.max(np.abs(stretch_eigenvalues), axis=-1) / np.min(np.abs(stretch_eigenvalues), axis=-1)) # since this was SVD decomposition. 

    # compute the overall distortion factor, mean conformality error 
    areas3D = np.linalg.norm(np.cross(dS_du, 
                                      dS_dv), axis=-1)# # use the cross product
    stretch_factor_no_inf = stretch_factor.copy(); stretch_factor_no_inf[np.isinf(stretch_factor_no_inf)] = np.nan
    mean_stretch_factor = np.nansum(areas3D*stretch_factor_no_inf / (float(np.nansum(areas3D))))

    return stretch_factor, mean_stretch_factor


def surface_area_uv(surface_uv_params, eps=1e-12, pad=False):
    r""" Compute the total surface area of the unwrapped (u,v) parametrized surface using differential calculus. 

    Assuming the parametrization is continuous, the differential element area for a pixel is the area of a differential rectangular element which can be written as a cross-product of the gradient vectors, 

    .. math :: 
        A_{pixel} &= \left\|\frac{\partial S}{\partial u}\right\|\left\|\frac{\partial S}{\partial v}\right\| \\
                  &= \left\|\frac{\partial S}{\partial u} \times \frac{\partial S}{\partial v}\right\|

    where :math:`\times` is the vector cross product

    Parameters
    ----------
    surface_uv_params : (UxVx3) array
        the input image giving the uv unwrapped (x,y,z) surface 
    eps : scalar 
        small numerical value for numerical stability 
    pad : bool 
        if True, spherically pads by 1 pixel top and right to compute 1st order finite differences, if False using np.gradient to compute central differences

    Returns
    -------
    dS_dudv : (UxV) array   
        The surface area of each pixel  
    total_dS_dudv : scalar
        the total summed surface area of all pixels
    """
    import numpy as np 

    dS_du, dS_dv = gradient_uv(surface_uv_params, eps=eps, pad=pad)

    # area of the original surface.
    dS_dudv = np.linalg.norm(np.cross(dS_du, 
                                      dS_dv), axis=-1)# # use the cross product
    
    total_dS_dudv = np.nansum(dS_dudv)

    return dS_dudv, total_dS_dudv

# add total volume for a topogarphy space. 
def volume_uv(depth_uv_params, eps=1e-12, pad=False):
    r""" Compute the total volume of the unwrapped (d,u,v) parametrized volume using differential calculus. 

    Assuming the parametrization is continuous, the differential element element for a pixel is the volume of a differential parallelpiped element which can be written as a triple-product of the gradient vectors, 

    .. math :: 
        V_{pixel} &= \left\|\frac{\partial V}{\partial d}\right\|\left\|\frac{\partial V}{\partial u}\right\|\left\|\frac{\partial V}{\partial v}\right\| \\
                  &= \left\|\frac{\partial V}{\partial d} \cdot \frac{\partial V}{\partial u} \times \frac{\partial V}{\partial v} \right\|

    where :math:`\cdot` is the vector dot product and :math:`\times` is the vector cross product

    Parameters
    ----------
    depth_uv_params : (DxUxVx3) array
        the input image giving the topography (d,u,v) unwrapped (x,y,z) surface 
    eps : scalar 
        small numerical value for numerical stability 
    pad : bool 
        if True, spherically pads by 1 pixel top and right to compute 1st order finite differences, if False using np.gradient to compute central differences

    Returns
    -------
    dV : (DxUxV) array   
        The volume of each pixel  
    Volume : scalar
        the total summed volume of all pixels
    """
    import numpy as np 

    dV_dd, dV_du, dV_dv = gradient_uv_depth(depth_uv_params, eps=eps, pad=pad)

    # volume of the original surface.
    dV = np.cross(dV_du, dV_dv, axis=-1)
    dV = np.nansum(dV_dd * dV, axis=-1)
    dV = np.linalg.norm(dV, axis=-1)# finally taking the magniude. 
    
    Volume = np.nansum(dV)

    return dV, Volume

def area_distortion_uv(surface_uv_params, eps=1e-12, pad=False):
    r""" Compute the area distortion given a (u,v) parameterized (x,y,z) surface, :math:`S`. The area distortion factor, :math:`\lambda` is defined as the normalised surface area measured in (u,v) divided by the normalised surface area in (x,y,z) 

    .. math :: 
        \lambda &= \frac{dudv/\sum_{uv}dudv}{dS/\sum dS} \\
    
    Parameters
    ----------
    surface_uv_params : (UxVx3) array
        the input image giving the uv unwrapped (x,y,z) surface 
    eps : scalar
        small number for numerical stability 
    pad : bool
        if True, spherically pads by 1 pixel top and right to compute 1st order finite differences, if False using np.gradient to compute central differences

    """
    # this is just the gradient area. 
    import numpy as np

    dS_du, dS_dv = gradient_uv(surface_uv_params, eps=eps, pad=pad)
    # area of the original surface.
    dS_dudv = np.linalg.norm(np.cross(dS_du, 
                                      dS_dv), axis=-1)# # use the cross product
    dS_dudv = dS_dudv / np.nansum(dS_dudv)

    # area of a dudv element of unit area divided evenly. 1/(UV)
    dudv = 1./float(dS_dudv.shape[0]*dS_dudv.shape[1])

    area_distortion = np.abs( dudv / dS_dudv )

    return area_distortion

# add total volume distortion for a topogarphy space. 
def volume_distortion_duv(depth_uv_params, eps=1e-12, pad=False):
    r""" Compute the volume distortion given a (d,u,v) parameterized (x,y,z) surface, :math:`V`. The volume distortion factor, :math:`\lambda_{V}` is defined as the normalised volume measured in (d,u,v) divided by the normalised volume in (x,y,z) 

    .. math :: 
        \lambda_{V} &= \frac{dd du dv/\sum_{duv}dd du dv}{dV/\sum dV} \\
    
    Parameters
    ----------
    depth_uv_params : (DxUxVx3) array
        the input image giving the (d,u,v) unwrapped (x,y,z) surface 
    eps : scalar
        small number for numerical stability 
    pad : bool
        if True, spherically pads by 1 pixel top and right to compute 1st order finite differences, if False using np.gradient to compute central differences

    """
    # this is just the gradient area. 
    import numpy as np

    dV_dd, dV_du, dV_dv = gradient_uv_depth(depth_uv_params, eps=eps, pad=pad)

    # volume of the original surface.
    dV_dddudv = np.cross(dV_du, dV_dv, axis=-1)
    dV_dddudv = np.nansum(dV_dd * dV_dddudv, axis=-1)
    dV_dddudv = np.linalg.norm(dV_dddudv, axis=-1)# finally taking the magniude. 

    dV_dddudv = dV_dddudv / np.nansum(dV_dddudv) # normalise the original surface volume. 
    # area of a dddudv element of unit volume divided evenly. 1/(DUV)
    dddudv = 1./float(dV_dddudv.shape[0]*dV_dddudv.shape[1]*dV_dddudv.shape[2]) # normalised current volume 

    volume_distortion = np.abs( dddudv / dV_dddudv )

    return volume_distortion
