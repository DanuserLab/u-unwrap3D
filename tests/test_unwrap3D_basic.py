#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:46:13 2024

@author: s205272
"""


if __name__=="__main__":
    
    import skimage.io as skio 
    import os 
    import numpy as np 
    import pylab as plt 
    import scipy.ndimage as ndimage
    import skimage.morphology as skmorph
    from matplotlib import cm # this is for specifying a matplotlib color palette
    import igl

    import unwrap3D.Utility_Functions.file_io as fio # for common IO functions
    import unwrap3D.Segmentation.segmentation as segmentation # import the segmentation submodule which wraps the Otsu method
    import unwrap3D.Mesh.meshtools as meshtools # load in the meshtools submodule
    import unwrap3D.Image_Functions.image as image_fn # for common image processing functions
    import unwrap3D.Visualisation.colors as vol_colors # this is for colormapping any np.array using a color palette 
    import unwrap3D.Visualisation.plotting as plotting
    import unwrap3D.Unzipping.unzip as uzip
    import unwrap3D.Analysis_Functions.topography as topo_tools
    
    debug_viz = False
    

    imgfile = '/home2/s205272/matlab/python-applications/Unwrap3D/example_data/img/bleb_example.tif'   
    img = skio.imread(imgfile)

    """
    extract binary. 
    """
    # returns the binary and the auto determined threshold. 
    img_binary, img_binary_thresh = segmentation.segment_vol_thresh(img)
    # erode by ball kernel radius = 1 to make a tighter binary
    img_binary = ndimage.binary_erosion(img_binary, 
                                        iterations=1, 
                                        structure=skmorph.ball(1))


    """
    measure mean curvature
    """
    # compute the continuous mean curvature definition and smooth slightly with a Gaussian of sigma=3.  
    H_binary, H_sdf_vol_normal, H_sdf_vol = segmentation.mean_curvature_binary(img_binary, 
                                                                               smooth=3, 
                                                                               mask=True) # if mask=True, only the curvature of a thin shell (+/-smooth) around the binary segmentation is returned. 
    
    # if mask=True above, the following line is necessary to enable interpolation.
    H_binary[np.isnan(H_binary)] = 0 


    """
    get mesh
    """
    img_binary_surf_mesh = meshtools.marching_cubes_mesh_binary(img_binary.transpose(2,1,0), # The transpose is to be consistent with ImageJ rendering and Matlab convention  
                                                                presmooth=1., # applies a presmooth
                                                                contourlevel=.5,
                                                                remesh=True,
                                                                remesh_method='CGAL', 
                                                                remesh_samples=.9, # remeshing with a target #vertices = 90% of original
                                                                predecimate=True, # this applies quadric mesh simplication to remove very small edges before remeshing
                                                                min_mesh_size=10000,
                                                                upsamplemethod='inplane')

    print(meshtools.measure_triangle_props(img_binary_surf_mesh))
    
    print('Euler characteristic of mesh is: ', img_binary_surf_mesh.euler_number) #should be 2 if genus is 0

    # we also provide a more comprehensive function for common mesh properties
    mesh_property = meshtools.measure_props_trimesh(img_binary_surf_mesh, main_component=True, clean=True) 

    print(mesh_property)
    
    
    # interpolation 
    surf_H = image_fn.map_intensity_interp3(img_binary_surf_mesh.vertices[:,::-1], # undo the transpose to be consistent with the volume
                                                grid_shape= H_binary.shape, 
                                                I_ref= H_binary, 
                                                method='linear', 
                                                cast_uint8=False)

    # we generate colors from the mean curvature 
    surf_H_colors = vol_colors.get_colors(surf_H/.104, # 0.104 is the voxel resolution -> this converts to um^-1 
                                          colormap=cm.Spectral_r, 
                                          vmin=-1., 
                                          vmax=1.) # colormap H with lower and upper limit of -1, 1 um^-1. 

    # set the vertex colors to the computed mean curvature color
    img_binary_surf_mesh.visual.vertex_colors = np.uint8(255*surf_H_colors[...,:3]) 


    
    """
    Sample intensity
    """
    n_samples = 1./ .104 # total number of steps
    stepsize = 0.5 # voxels
        
    # flip the mesh vertex coordinates so that it aligns with the volume size 
    img_binary_surf_mesh.vertices = img_binary_surf_mesh.vertices[:,::-1].copy()

    # run the active contour cMCF to get the coordinates at different depths into the cell according to the external image gradient given by the gradient of the signed distance function.
    v_depth = meshtools.parametric_mesh_constant_img_flow(img_binary_surf_mesh, 
                                                          external_img_gradient = H_sdf_vol_normal.transpose(1,2,3,0), 
                                                          niters=int(n_samples/stepsize), 
                                                          deltaL=5e-5, # delta which controls the stiffness of the mesh
                                                          step_size=stepsize, 
                                                          method='implicit', # this specifies the cMCF solver.
                                                          conformalize=True) # ensure we use the cMCF Laplacian

    # we can check the size of the array
    print(v_depth.shape)

    # we can plot the trajectory with matplotlib 
    # get the intensities at the sampled depth coordinates. 
    v_depth_I = image_fn.map_intensity_interp3(v_depth.transpose(0,2,1).reshape(-1,3), 
                                                img.shape, 
                                                I_ref=img)
    v_depth_I = v_depth_I.reshape(-1,v_depth.shape[-1]) # matrix reshaping into a nicer shape. 

    # postprocess to check the total distance from the surface does not exceed the desired and replace any nans.  
    dist_v_depth0 = np.linalg.norm(v_depth - v_depth[...,0][...,None], axis=1)
    valid_I = dist_v_depth0<=n_samples
    v_depth_I[valid_I == 0 ] = np.nan # replace with nans


    # compute the mean sampled intensity which will be taken as the surface intensity. 
    surf_intensity_img_raw = np.nanmean(v_depth_I, axis=1)
    surf_intensity_img_raw[np.isnan(surf_intensity_img_raw)] = 0

    # for visualization, we find the intensity range to be more pleasing if clipped to between the 1st and 99th percentile. 
    I_min = np.percentile(surf_intensity_img_raw,1)
    I_max = np.percentile(surf_intensity_img_raw,99)

    surf_intensity_img_raw_colors = vol_colors.get_colors(surf_intensity_img_raw, 
                                                          colormap=cm.RdYlBu_r,   
                                                          vmin=I_min, 
                                                          vmax=I_max)
    # create a new surface mesh, now with the PI3K molecular signal colors. 
    img_binary_surf_mesh_colors = meshtools.create_mesh(vertices=img_binary_surf_mesh.vertices[:,::-1],
                                                      faces=img_binary_surf_mesh.faces, 
                                                      vertex_colors=np.uint8(255*surf_intensity_img_raw_colors[...,:3])) 
    # tmp = img_binary_surf_mesh_colors.export(os.path.join(savefolder, 
    #                                                 'PI3K_binary_mesh_'+basefname+'.obj')) # tmp is used to prevent printing to screen.


    if debug_viz:
        # again we can quickly view the coloring in matplotlib
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(img_binary_surf_mesh_colors.vertices[...,2], 
                   img_binary_surf_mesh_colors.vertices[...,1],
                   img_binary_surf_mesh_colors.vertices[...,0], 
                   s=1, 
                   c=surf_intensity_img_raw, 
                   cmap='RdYlBu_r', 
                   vmin=-I_min,
                   vmax=I_max)
        ax.view_init(-60, 180)
        plotting.set_axes_equal(ax)
        plt.show()
    
    # from CGAL.CGAL_Polygon_mesh_processing import isotropic_remeshing
    
    
    """
    Step 1: cMCF
    """
    mesh = img_binary_surf_mesh_colors.copy() # make a copye just in case. 

    # run cMCF on the mesh for 50 iterations, delta=5e-4 using the standard cotangent Laplacian 
    Usteps_MCF_img, mesh_F, MCF_measures_dict = meshtools.conformalized_mean_curvature_flow(mesh, 
                                                                                        max_iter=50, 
                                                                                        delta=5e-4, 
                                                                                        rescale_output=True, # this makes sure we evolve the output points are in image space. 
                                                                                        conformalize=True, # set this flag to run cMCF, else it will run MCF
                                                                                        robust_L =False, # if set, runs MCF/cMCF using the robust Laplacian instead of cotangent Laplacian
                                                                                        mollify_factor=1e-5)  # this is a parameter used in the robust Laplacian

    
    """
    Perform an automatic reference determination based on Gaussian Curvature evolution.  
    """
    
    # load the mean absolute Gaussian curvature at vertices at each iteration of the cMCF
    gauss_curve_MCF = MCF_measures_dict['gauss_curvature_iter'].copy()


    # method a: rate of change based. 
    # this is a speed-based (may generate overly smooth surfaces. )
    threshold_cMCF = 5e-7
    # determine the cut off iteration number such that the change between the previous iteration drops below 5e-7
    ind = meshtools.find_curvature_cutoff_index( gauss_curve_MCF, 
                                                  thresh=threshold_cMCF,  # cutoff on the rate of change.
                                                  absval=True) 
    print('threshold-based ind:, ', ind)

    # method b: changepoint based. -> can give tighter references. 
    switch_point_inds = meshtools.find_all_curvature_cutoff_index( gauss_curve_MCF, 
                                                                  winsize=5,  # cutoff on the rate of change.
                                                                  absval=True, 
                                                                  min_peak_height=1e-10) 
    print('all switch_point_inds: ', switch_point_inds)
    ind = switch_point_inds[0]

    # plot the curve evolution. 

    """
    measure conformal and equiareal distortion errors
    """
    conformal_error_flow = [meshtools.quasi_conformal_error(Usteps_MCF_img[...,0], Usteps_MCF_img[...,steps], mesh.faces) for steps in np.arange(Usteps_MCF_img.shape[-1])]
    area_distortion_error_flow = [meshtools.area_distortion_measure(Usteps_MCF_img[...,0], Usteps_MCF_img[...,steps], mesh.faces) for steps in np.arange(Usteps_MCF_img.shape[-1])]

    # figure out how to save out these information.
    conformal_error_flow_arrays = [ np.array([cc[jj] for cc in conformal_error_flow]) for jj in np.arange(len(conformal_error_flow[0]))]
    area_distortion_error_flow = np.array(area_distortion_error_flow) 


    mean_conformal_error_curve = conformal_error_flow_arrays[2].copy()
    mean_area_errors_curve = area_distortion_error_flow.mean(axis=1).copy()

    # plot the errors as a function of iteration number 
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.set_xlabel('Iteration Number', fontsize=18, fontname='Arial')
    ax1.set_ylabel('Conformal error', fontsize=18, fontname='Arial', color='k')
    ax1.plot(np.arange(len(mean_conformal_error_curve)), mean_conformal_error_curve, 'ko-')
    ax1.vlines(ind,1,1.5, linestyles='dashed', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    plt.xticks(fontname='Arial', fontsize=14)
    plt.yticks(fontname='Arial', fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Final area / initial area', fontsize=18, color='r', fontname='Arial')  # we already handled the x-label with ax1
    ax2.plot(np.arange(len(mean_area_errors_curve)), 1./mean_area_errors_curve, 'ro-')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.xticks(fontname='Arial', fontsize=14)
    plt.yticks(fontname='Arial', fontsize=14)
    plt.tick_params(length=5)
    # plt.savefig(os.path.join(savefolder,  
    #                        'mean_MCF_errors_iterations_'+basefname+'.png'), bbox_inches='tight', dpi=300)
    plt.show()



    # first explicitly make a mesh of the intermediate S_ref(x,y,z) obtained direct from cMCF.  
    cMCF_Sref = meshtools.create_mesh(vertices = Usteps_MCF_img[...,ind],  
                                        faces = mesh.faces,
                                        vertex_colors=mesh.visual.vertex_colors)


    """
    Voxelization of mesh 
    """
    # vol_shape allows us when available to use the volume grid size of the original image. -> the advantage is that no postcorrection is required to maintain image coordinates.
    # If not available the maximum extent of the vertices are used to determine a bounding box which is then padded
    smooth_img_binary = meshtools.voxelize_image_mesh_pts(cMCF_Sref, # might be best here to use barycenter 
                                                          pad=50, 
                                                          dilate_ksize=3, # this is the size of the ball kernel to plug holes and handles. 
                                                          erode_ksize=3, # this undoes the dilation to keep the remesh close to the original surface.
                                                          vol_shape=img_binary.shape[::-1]) # transpose the shape because vertices are transposed from original.

    """
    remesh at isovalue of 0.5
    """
    cMCF_Sref_remesh = meshtools.marching_cubes_mesh_binary(smooth_img_binary, 
                                                                presmooth=1., 
                                                                contourlevel=.5, 
                                                                remesh=True, 
                                                                remesh_method='CGAL', 
                                                                remesh_samples=0.25, # lower downsampling will give a smoother mesh, see below on voxelization artifacts.
                                                                predecimate=True, 
                                                                min_mesh_size=10000) # don't need a lot   
    
    
    
    # check this remesh gives a genus-0 mesh
    mesh_property = meshtools.measure_props_trimesh(cMCF_Sref_remesh, main_component=True, clean=True) 
    print(mesh_property)


    # quantify the discrepancy
    # 1. mean squared error or Chamfer distance
    mse = meshtools.chamfer_distance_point_cloud(cMCF_Sref.vertices, cMCF_Sref_remesh.vertices)
    print('mean squared error: ', mse) # in terms of pixels 

    # 3. diff surface area
    diff_A = meshtools.diff_area_trimesh_trimesh(cMCF_Sref, cMCF_Sref_remesh)
    fraction_A = diff_A / (np.nansum(igl.doublearea(cMCF_Sref.vertices, cMCF_Sref.faces) /2.))
    print('fraction total surface area change: ', fraction_A) 
        
    
    """
    Transfer curvature values. 
    """
    match_params, remesh_S_ref_H, _ = meshtools.transfer_mesh_measurements(cMCF_Sref,
                                                                        cMCF_Sref_remesh.vertices, 
                                                                        source_mesh_vertex_scalars = surf_H[:,None])
    remesh_S_ref_H = remesh_S_ref_H[...,0]

    
    remesh_S_ref_H_color = vol_colors.get_colors(remesh_S_ref_H/.104, 
                                                 colormap=cm.Spectral_r, 
                                                 vmin=-1, 
                                                 vmax=1)
    
    # savefolder='.'
    # cMCF_Sref_remesh.visual.vertex_colors = np.uint8(255*remesh_S_ref_H_color[:,:3])
    # tmp = cMCF_Sref_remesh.export(os.path.join(savefolder, 
    #                                  'unwrap_cMCF_Sref_mesh_H_color.obj'))


    cMCF_Sref.visual.vertex_colors = np.uint8(255*remesh_S_ref_H_color[...,:3])
# =============================================================================
# =============================================================================
# #     Spherical mapping 
# =============================================================================
# =============================================================================

    """
    Option 1: direct quasiconformal mapping (lowest conformal errors, but poor triangle quality)
    """
    sphere_xyz_conformal = meshtools.direct_spherical_conformal_map(cMCF_Sref_remesh.vertices, 
                                                                      cMCF_Sref_remesh.faces,
                                                                      apply_mobius_correction=True)
    

    """
    Conformal and equiareal errors
    """
    conformal_error = meshtools.quasi_conformal_error(cMCF_Sref_remesh.vertices, 
                                                      sphere_xyz_conformal, 
                                                      cMCF_Sref_remesh.faces)
    mean_conformal_errors = conformal_error[2]

    area_distortion_error = meshtools.area_distortion_measure(cMCF_Sref_remesh.vertices, 
                                                              sphere_xyz_conformal, 
                                                              cMCF_Sref_remesh.faces)  
    mean_areal_errors = np.nanmean(area_distortion_error)
        
    sphere_mesh = cMCF_Sref_remesh.copy(); sphere_mesh.vertices = sphere_xyz_conformal.copy()
    print(meshtools.measure_triangle_props(sphere_mesh))
    
    """
    Option 2: iterative tutte-like relaxation. (uniform laplacian averaging after projection on sphere)
    """
    sphere_xyz_conformal_tutte, n_inverted = meshtools.iterative_tutte_spherical_map(cMCF_Sref_remesh.vertices, 
                                                                                     cMCF_Sref_remesh.faces,
                                                                                     deltaL=5e-3, 
                                                                                     min_iter=5,
                                                                                     max_iter=25,
                                                                                     mollify_factor=1e-5)
    print(n_inverted)

    """
    Conformal and equiareal errors
    """
    conformal_error_tutte = meshtools.quasi_conformal_error(cMCF_Sref_remesh.vertices, 
                                                      sphere_xyz_conformal_tutte, 
                                                      cMCF_Sref_remesh.faces)
    mean_conformal_errors_tutte = conformal_error_tutte[2]

    area_distortion_error_tutte = meshtools.area_distortion_measure(cMCF_Sref_remesh.vertices, 
                                                                    sphere_xyz_conformal_tutte, 
                                                                    cMCF_Sref_remesh.faces)  
    mean_areal_errors_tutte = np.nanmean(area_distortion_error_tutte)
    
    
    sphere_mesh_tutte = cMCF_Sref_remesh.copy(); sphere_mesh_tutte.vertices = sphere_xyz_conformal_tutte.copy()
    print(meshtools.measure_triangle_props(sphere_mesh_tutte))
    
    
    """
    Areal relaxation
    """
    relax_v, relax_f, area_distortion_iter= meshtools.area_distortion_flow_relax_sphere(sphere_mesh_tutte, 
                                                                                        cMCF_Sref_remesh, 
                                                                                        max_iter=100, # increase this for shapes deviating signficantly from sphere. 
                                                                                        delta=0.1, # controls the stiffness, if too high - there is no flow, if too low - bijectivity is lost and flow may breakdown 
                                                                                        stepsize=1, 
                                                                                        debugviz=False) # the flip delaunay is not very smooth...?  

    # compute the evolution of the mean area distortion. We use this to determine the first iteration when area distortion is minimized. 
    area_distortion_iter_curve = np.hstack([np.mean(mm) for mm in area_distortion_iter])

    # to determine when area distortion is minimised we look for the timepoint when the sign first changes. This is because once the minimum is reached, depending on stepsize, the mesh will start to oscilate. The larger the stepsize, the faster the convergence but the larger the oscillatory instability. 
    min_area_distort = (np.arange(len(area_distortion_iter_curve))[:-1])[np.sign(np.diff(area_distortion_iter_curve))>0][0] # falls below 0. 
    min_area_distort_sign = min_area_distort
    
    # alternatively and more consistently we can look for when the area distortion has been minimized below a certain measure for the first time.
    target_area_distort = 1.01
    min_area_distort = (np.arange(len(area_distortion_iter_curve)))[area_distortion_iter_curve<=target_area_distort][0]


    if debug_viz:
        plt.figure()
        plt.plot(np.log(area_distortion_iter_curve))
        plt.vlines(min_area_distort_sign, ymin = 0, ymax=np.nanmax(np.log(area_distortion_iter_curve)), color='k', linestyles='dashed')
        plt.vlines(min_area_distort, ymin = 0, ymax=np.nanmax(np.log(area_distortion_iter_curve)), color='g', linestyles='solid')
        plt.show()
    
    
    # create the equiareal mesh and save it out
    equiareal_sphere_mesh = sphere_mesh_tutte.copy()
    equiareal_sphere_mesh.vertices = relax_v[min_area_distort].copy()
    
    
    """
    uv mapping. 
    """
    # =============================================================================
    #     Example: optimizing w.r.t original surfaces' curvature 
    # =============================================================================
    # compute the optimal rotation parameters. 
    # for weights, we are going to the use the original surface mean curvature
    weights = np.abs(remesh_S_ref_H)
    weights_proj = remesh_S_ref_H # as the measurement of the original curvature was inverse to the mesh. 
    
    # also optimize wrt the ref shape. 
    # instead of computing from a volume binary, which would involve mesh voxelization etc. we are going to compute mean curvature directly from the mesh using the definition that it is the mean of the 2 principal curvatures
    
    # =============================================================================
    #     Example: optimizing w.r.t ref surfaces' curvature 
    # =============================================================================
    H_Sref = meshtools.compute_mean_curvature(cMCF_Sref_remesh)
    # define weights
    weights = np.abs(H_Sref)
    weights_proj = H_Sref
        

    """
    Weighted PCA to solve for the optimal rotation 
    """
    rot_matrix, extra_rotate_bool = meshtools.optimize_sphere_rotation_from_weights(equiareal_sphere_mesh, 
                                                                                    weights=weights, 
                                                                                    signed_weights_to_orient=weights_proj)
    
    equiareal_sphere_mesh_rot = meshtools.apply_optimized_sphere_rotation_from_weights(equiareal_sphere_mesh, 
                                                                                        rot_matrix, 
                                                                                        extra_rotate_bool,
                                                                                        additional_rotation_angle_degrees=0)
                    
    """
    4. sphere to uv-grid unwrap.
    """
    # append all measurements we would like to map additional to coordinates. 
    surface_measurements_scalars = np.concatenate([cMCF_Sref_remesh.visual.vertex_colors/255., 
                                                   remesh_S_ref_H[...,None]], axis=-1)
    
    S_uv_opt, uv_Sref_equiareal_match_rot, h_opt, uv_surface_measurements_scalars, uv_surface_measurements_labels = uzip.uv_map_sphere_surface_parameterization(sphere_mesh=equiareal_sphere_mesh_rot, 
                                                                                                                        surface_mesh=cMCF_Sref_remesh,
                                                                                                                        surface_measurements_scalars=surface_measurements_scalars,
                                                                                                                        surface_measurements_labels=None,
                                                                                                                        uv_grid_size = 512, 
                                                                                                                        optimize_aspect_ratio=False, 
                                                                                                                        aspect_ratio_method='length_ratio',
                                                                                                                        length_ratio_avg_fn=np.nanmedian,
                                                                                                                        h_opt=None)
    
    
    uv_mesh_color = uv_surface_measurements_scalars[...,:4]
    remesh_S_ref_H_uv = uv_surface_measurements_scalars[...,-1]
    
    
    remesh_S_ref_H_uv_color = vol_colors.get_colors(remesh_S_ref_H_uv/.104,
                                                    colormap=cm.Spectral_r, 
                                                    vmin=-1, vmax=1)
    
    if debug_viz:
        plt.figure(figsize=(10,10))
        plt.imshow(remesh_S_ref_H_uv_color)
        plt.show()
    
    
    """
    Step 5: build topography space., using euclidean distance for exterior and harmonic distance for interior 
    """
    topography_space, (N_in, N_out), alpha_step = uzip.build_topography_space_from_Suv(S_uv_opt, 
                                                                                    smooth_img_binary, # this must be in the same geometric space as S_uv.
                                                                                    external_gradient_field = None, # build if none. 
                                                                                    pad_size=50, 
                                                                                    alpha_step=None,
                                                                                    outer_surf_pts = img_binary_surf_mesh.vertices[:,::-1], # this is the input mesh, must also live in same space.  
                                                                                    outer_n_dists = None, # let this be auto determined. 
                                                                                    outer_pad_dist=10, # the leeway.
                                                                                    outer_method='forward_euler',
                                                                                    outer_smooth_win_factor=16, # we will smooth by minimum length of uv divide this factor. 
                                                                                    inner_n_dists = 25, # in voxels.
                                                                                    inner_sdf_method = 'harmonic',
                                                                                    inner_source_mask = None, 
                                                                                    inner_harmonic_ds = 4., # 4x downsample by default. 
                                                                                    inner_method='active_contours')
    
    
    """
    Step 6: derive topographic mesh and map information to it, after transforming back to cartesian space. 
    """
    
    # map the segmentation
    topography_binary = image_fn.map_intensity_interp3(topography_space[...,:].reshape(-1,3), # it should be in the same space as image
                                                        grid_shape=img_binary.shape[::-1], 
                                                        I_ref=img_binary.transpose(2,1,0)*255.,  
                                                        method='linear')
    topography_binary = topography_binary.reshape(topography_space.shape[:-1])/255. > 0.5



    topographic_mesh = meshtools.marching_cubes_mesh_binary(topography_binary,   
                                                        presmooth=1., # applies a presmooth
                                                        contourlevel=.5,
                                                        keep_largest_only=True, # we want the largest connected component 
                                                        remesh=True,
                                                        remesh_method='pyacvd', 
                                                        remesh_samples=.5, # remeshing with a target #vertices = 50% of original
                                                        predecimate=True, # this applies quadric mesh simplication to remove very small edges before remeshing
                                                        min_mesh_size=10000,
                                                        upsamplemethod='inplane') # upsample the mesh if after the simplification and remeshing < min_mesh_size  



    # look up the curvature values (we could also interpolate with the surface mesh)
    topography_verts_xyz = topo_tools.uv_depth_pts3D_to_xyz_pts3D( topographic_mesh.vertices[...,:], 
                                                                    topography_space)


    # interpolate the value onto the topographic mesh.
    topo_surf_H = image_fn.map_intensity_interp3(topography_verts_xyz[:,:], 
                                                grid_shape= H_binary.shape[::-1], 
                                                I_ref= H_binary.transpose(2,1,0), 
                                                method='linear', 
                                                cast_uint8=False)
    
    topo_surf_H_colors = vol_colors.get_colors(topo_surf_H/.104, # 0.104 is the voxel resolution -> this converts to um^-1 
                                          colormap=cm.Spectral_r, 
                                          vmin=-1., 
                                          vmax=1.) # colormap H with lower and upper limit of -1, 1 um^-1. 

    # set the vertex colors to the computed mean curvature color
    topographic_mesh.visual.vertex_colors = np.uint8(255*topo_surf_H_colors[...,:3]) 
        
    
    savefolder = '.'
    tmp = topographic_mesh.export(os.path.join(savefolder, 
                                           'curvature_topographic_mesh.obj')) # tmp is used to prevent printing to screen.
    
    topographic_3D_xyz_mesh = topographic_mesh.copy()
    topographic_3D_xyz_mesh.vertices = topography_verts_xyz
    
    tmp = topographic_3D_xyz_mesh.export(os.path.join(savefolder, 
                                       'curvature_topographic_mesh_remapped-xyz.obj')) # tmp is used to prevent printing to screen.

    
    
    