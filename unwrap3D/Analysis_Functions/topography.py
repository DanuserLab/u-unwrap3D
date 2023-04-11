
from ..Mesh import meshtools as meshtools
from ..Segmentation import segmentation as segmentation
from ..Unzipping import unzip as uzip 
from ..Image_Functions import image as image_fn


def uv_depth_pts3D_to_xyz_pts3D( uv_pts3D, uv_depth_params):
    r""" Linear Interpolation of the corresponding (x,y,z) coordinates given query (d,u,v) topography coordinates where the injective map (d,u,v) -> (x,y,z) is given by uv_depth_params.

    Parameters
    ----------
    uv_pts3D : (Nx3) array
        the topography, (d,u,v) coordinate at N positions for which original (x,y,z) coordinates is desired 
    uv_depth_params : (D,U,V,3) array
        the lookup table mapping the uniform (D,U,V) grid to (x,y,z) in original shape space 
    
    Returns
    -------
    xyz_pts3D : (Nx3) array
        The corresponding (x,y,z) coordinates into the space indexed by uv_depth_params.
    """

    import numpy as np 

    uv_pts3D_ = uv_pts3D.copy()
    uv_pts3D_[...,0] = np.clip(uv_pts3D[:,0], 0, uv_depth_params.shape[0]-1)
    uv_pts3D_[...,1] = np.clip(uv_pts3D[:,1], 0, uv_depth_params.shape[1]-1)
    uv_pts3D_[...,2] = np.clip(uv_pts3D[:,2], 0, uv_depth_params.shape[2]-1)

    xyz_pts3D = np.array([image_fn.map_intensity_interp3(uv_pts3D_, 
                                                    grid_shape=uv_depth_params.shape[:-1], 
                                                    I_ref=uv_depth_params[...,ch]) for ch in np.arange(uv_depth_params.shape[-1])])
    xyz_pts3D = xyz_pts3D.T.copy()
    return xyz_pts3D

def estimate_base_topography_uv_img(depth_binary):
    r""" Given a binary in topography space, this function attempts to derive a 1-to-1 height map of the basal surface by projecting straight lines in d at every (u,v) pixel position to find contiguous stretches of intracellular space. The basal surface is given by the highest d in the longest stretch of each (u,v) position 
    The resulting height map is intended to be used for downstream processing.

    Parameters
    ----------
    depth_binary : (DxUxV) array
        a binary volume in topography space where 1 indicates intracellular space and 0 is background ( extracellular ) 
    
    Returns
    -------
    heigh_func : (UxV) array
        a grayscale image with intensity = to the height coordinate in d i.e. we express height as a function of (u,v) position, height = f(u,v) 
    """

    import numpy as np 
    
    m, n = depth_binary.shape[1:]
    heigh_func = np.zeros((m,n))
    YY,XX = np.indices((m,n))

    for ii in np.arange(m)[:]:
        for jj in np.arange(n)[:]:
            # iterate over this and parse the column the longest contig. 
            data = depth_binary[:,ii,jj] > 0
            valid = np.arange(len(data))[data > 0]  # convert to index. 
            
            # break into contigs
            contigs = []
            # contigs_nearest_mesh_distance = []
            contig = []
            for jjj in np.arange(len(valid)): # iterate over the sequence. 
                if jjj == 0:
                    contig.append(valid[jjj])
                else:
                    if len(contig)>0:
                        diff = valid[jjj] - contig[-1]
                        if diff == 1:
                            contig.append(valid[jjj])
                        else:
                            # we should wrap this up. in a function now. 
                            # finish a contig. 
                            # print(contig)
                            contigs.append(contig)
                            # query_contig_pt = np.hstack([contig[-1], ii, jj])
                            # print(query_contig_pt)
                            # query_contig_pt_distance = np.min(np.linalg.norm(topography_mesh_pts-query_contig_pt[None,:], axis=-1))
                            # contigs_nearest_mesh_distance.append(query_contig_pt_distance)
                            contig=[valid[jjj]] # start a new one. 
                    else:
                        contig.append(valid[jjj]) # extend cuyrrent. 
            if len(contig) > 0 :
                contigs.append(contig)
                # query_contig_pt = np.hstack([contig[-1], ii, jj])
                # query_contig_pt_distance = np.min(np.linalg.norm(topography_mesh_pts-query_contig_pt[None,:], axis=-1))
                # contigs_nearest_mesh_distance.append(query_contig_pt_distance)
                contig = []
    

            # filter by distance then take the maximum!. 
            # contigs = [contigs[kkk] for kkk in np.arange(len(contigs)) if contigs_nearest_mesh_distance[kkk]<10.] # within a minimum threshold. 
            max_contig = contigs[np.argmax([len(cc) for cc in contigs])] # we assume we take the longest contig. 
            heigh_func[ii,jj] = max_contig[-1]

    heigh_func = np.dstack([heigh_func, YY, XX])
    return heigh_func
    

def penalized_smooth_topography_uv_img(height_func, 
                                        ds=4, 
                                        padding_multiplier=4, 
                                        method='ALS', 
                                        lam=1,
                                        p=0.25, 
                                        niter=10,
                                        uv_params=None):
    r""" Applies extended 2D asymmetric least squares regression to smooth a topography surface given as a height image, that is where the surface has been parameetrized 1-to-1 with (u,v), :math:`d=f(u,v)` 

    Parameters
    ----------
    height_func : (UxV) array
        an input topography surface given as a height image such that d=f(u,v)
    ds : int
        isotropic downsampling factor of the original image, used for imposing additional smoothness + computational efficiency  
    padding_multiplier : scalar
        this specifies the padding size as a scalar multiple, 1/padding_multiplier of the downsampled image. It is used to soft enforce spherical bounds. 
    method : str
        one of 

        'ALS' : str
            Basic asymmetric least squares algorithm, see :func:`unwrap3D.Analysis_Functions.topography.baseline_als_Laplacian`   
        'airPLS' : str
            adaptive iteratively reweighted Penalized Least Squares algorithm, see :func:`unwrap3D.Analysis_Functions.topography.baseline_airPLS2D` 
    lam : scalar
        Controls the degree of smoothness in the baseline
    p : scalar
        Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
    niter : int
        The number of iterations to run the algorithm. Only a few iterations is required generally. 
    uv_params : (DxUxV,3) array
        the lookup table mapping the uniform (D,U,V) grid to (x,y,z) in original shape space. If provided the smoothness regularization will take into account the metric distortion. 
    
    Returns
    -------
    out : (UxV) array
        a smoothened output topography surface given as a height image such that d=f(u,v)
    """

    import skimage.transform as sktform
    import numpy as np 
    from sklearn.feature_extraction.image import grid_to_graph
    import scipy.sparse as spsparse

    output_shape = np.hstack(height_func.shape)
    height_binary_ds = sktform.resize(height_func, 
                                      output_shape=(output_shape//ds).astype(np.int), 
                                      preserve_range=True); 
    # this will be used to rescale. 
    height_binary_ds_max = height_binary_ds.max()
    height_binary_ds = height_binary_ds / float(height_binary_ds_max)

    if uv_params is not None:
        uv_params_ds = sktform.resize(uv_params, 
                                      output_shape=np.hstack([(output_shape//ds).astype(np.int), uv_params.shape[-1]]), 
                                      preserve_range=True); 
    
    # use padding to help regularize. 
    padding = (height_binary_ds.shape[0]//padding_multiplier)//2 * 2+1 # this is just to make it odd!. 

    # circular padding!. 
    height_binary_ds = np.hstack([height_binary_ds[:,-padding:], 
                                  height_binary_ds,
                                  height_binary_ds[:,:padding]])
    height_binary_ds = np.vstack([np.rot90(height_binary_ds[1:padding+1],2),
                                  height_binary_ds, 
                                  np.rot90(height_binary_ds[-1-padding:-1],2)])
    
    if uv_params is not None:
        uv_params_ds = np.hstack([uv_params_ds[:,-padding:], 
                                  uv_params_ds,
                                  uv_params_ds[:,:padding]])
        uv_params_ds = np.vstack([np.rot90(uv_params_ds[1:padding+1],2),
                                      uv_params_ds, 
                                      np.rot90(uv_params_ds[-1-padding:-1],2)])
    """
    build a graph to use the laplacian - this assumes equal weights. 
    """
    if uv_params is None:
        # build the equal weight Laplacian matrix. 
        img_graph = grid_to_graph(height_binary_ds.shape[0], height_binary_ds.shape[1])
        L = spsparse.spdiags(np.squeeze(img_graph.sum(axis=1)), 0, img_graph.shape[0], img_graph.shape[1])  - img_graph # degree - adjacency matrix. 
        L = L.tocsc()
    else:
        # build the weighted Laplacian matrix using the inverse edge length weights. 
        L = meshtools.get_inverse_distance_weight_grid_laplacian(uv_params_ds, grid_pts=uv_params_ds)
        L = L.tocsc()
    
    if method == 'ALS':
        out = baseline_als_Laplacian(height_binary_ds.ravel(), L, lam=lam, p=p, niter=niter) # because there was an issue  # seems like the best thing would be to do some smooth estimated fitting through...!
        out = out.reshape(height_binary_ds.shape)
        out = out[padding:-padding,padding:-padding].copy()
        out = sktform.resize(out, output_shape=height_func.shape) * height_binary_ds_max

    if method == 'airPLS':
        out = baseline_airPLS2D(height_binary_ds.ravel(), L, lam=lam, p=p, niter=niter)
        out = out.reshape(height_binary_ds.shape)
        out = out[padding:-padding,padding:-padding].copy()
        out = sktform.resize(out, output_shape=height_func.shape) * height_binary_ds_max

    return out 


def baseline_als_Laplacian(y, D, lam, p, niter=10):
    r""" Estimates a 1D baseline signal :math:`z=g(x_1,x_2,...,x_n)` to a 1D input signal :math:`y=f(x_1,x_2,...,x_n)` parametrized by :math:`n` dimensions  using asymmetric least squares. It can also be used for generic applications where a multidimensional image requires smoothing.
    Specifically the baseline signal, :math:`z` is the solution to the following optimization problem 

    .. math::
        z = arg\,min_z \{w(y-z)^2 + \lambda\sum(\Delta z)^2\}

    where :math:`y` is the input signal, :math:`\Delta z` is the 2nd derivative or Laplacian operator, :math:`\lambda` is the smoothing regularizer and :math:`w` is an asymmetric weighting

    .. math::
        w_i = 
        \Biggl \lbrace 
        { 
        p ,\text{ if } 
          {y_i>z_i}
        \atop 
        1-p, \text{ otherwise } 
        }

    Parameters
    ----------
    signal : 1D signal 
        The 1D signal to estimate a baseline signal. 
    D :  (NxN) sparse Laplacian matrix 
        the Laplacian matrix for the signal which captures the multidimensional structure of the signal e.g. the grid graph for a 2D or 3D image or the cotangent laplacian for a mesh. 
    lam : scalar
        Controls the degree of smoothness in the baseline
    p :  scalar
        Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
    niter: int
        The number of iterations to run the algorithm. Only a few iterations is required generally. 

    Returns
    -------
    z : 1D numpy array
        the estimated 1D baseline signal

    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    import numpy as np 
    
    L = len(y)
    # D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def baseline_airPLS2D(y, D, lam=100, p=1, niter=15):
    r""" Estimates a 1D baseline signal :math:`z=g(x_1,x_2,...,x_n)` to a 1D input signal :math:`y=f(x_1,x_2,...,x_n)` parametrized by :math:`n` dimensions  using Adaptive iteratively reweighted penalized least squares for baseline fitting. It can also be used for generic applications where a multidimensional image requires smoothing.
    Specifically the baseline signal, :math:`z` is the solution to the following optimization problem 

    .. math::
        z = arg\,min_z \{w(y-z)^2 + \lambda\sum(\Delta z)^2\}

    where :math:`y` is the input signal, :math:`\Delta z` is the 2nd derivative or Laplacian operator, :math:`\lambda` is the smoothing regularizer and :math:`w` is an asymmetric weighting

    .. math::
        w_i = 
        \Biggl \lbrace 
        { 
        0 ,\text{ if } 
          {y_i\ge z_i}
        \atop 
        e^{t(y_i-z_i)/|\textbf{d}|}, \text{ otherwise } 
        }

    where the vector :math:`\textbf{d}` consists of negative elements of the subtraction, :math:`y - z` and :math:`t` is the iteration number. 
    
    Instead of constant weights in airPLS, the weight :math:`w` is adaptively weighted for faster convergence. 

    Parameters
    ----------
    signal : 1D signal 
        The 1D signal to estimate a baseline signal. 
    D :  (NxN) sparse Laplacian matrix 
        the Laplacian matrix for the signal which captures the multidimensional structure of the signal e.g. the grid graph for a 2D or 3D image or the cotangent laplacian for a mesh. 
    lam : scalar
        Controls the degree of smoothness in the baseline
    p :  scalar
        Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
    niter: int
        The number of iterations to run the algorithm. Only a few iterations is required generally. 

    Returns
    -------
    z : 1D numpy array
        the estimated 1D baseline signal

    """
    import numpy as np
    import scipy.sparse as spsparse
    from scipy.sparse import csc_matrix, eye, diags
    from scipy.sparse.linalg import spsolve

    L = len(y)
    w=np.ones(L) # equal weights initially 
    for i in range(1,niter+1):
        W = spsparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        d=y-z # difference between original and the estimated baseline
        # w = p * (y > z) + (1-p) * (y < z) update in the original 
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(y)).sum() or i==niter):
            if(i==niter): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z


def segment_topography_vol_curvature_surface(vol_curvature, 
                                             vol_binary_mask,
                                             depth_ksize=1,
                                             smooth_curvature_sigma=[1],
                                             seg_method='kmeans',
                                             n_samples=10000,
                                             n_classes=3,
                                             random_state=0,
                                             scale_feats=False):
    
    r""" Segment protrusions on the unwrapped topography using multiscale mean curvature features. 

    The multiscale comes from extracting the `mean curvature <https://en.wikipedia.org/wiki/Mean_curvature>`_, :math:`H` computed as the divergence of the normalised gradient vectors of the signed distance function, :math:`\Phi`

    .. math::
        H = -\frac{1}{2}\nabla\cdot\left( \frac{\nabla \Phi}{|\nabla \Phi|}\right)
    
    and creating a multi-feature vector concatenating the smoothed :math:`H` after Gaussian smoothing with different :math:`\sigma` as specified by smooth_curvature_sigma  

    Parameters
    ----------
    vol_curvature : (MxNxL) numpy array
        the curvature computed using the definition above using the normalised gradient vector of the signed distance transform of the binary volume 
    vol_binary_mask : (MxNxL) numpy array
        the topography binary volume from which the vol_curvature was determined form 
    depth_ksize :  scalar
        the width of a ball morphological operator for extracting a binary mask of a thin shell of thickness 2 x depth_ksize capturing the topographic volume surface to be segmented. 
    smooth_curvature_sigma: list of scalars
        The list of N :math:`\sigma`'s which the vol_curvature is smoothed with to generate a N-vector :math:`[H_{\sigma_1},H_{\sigma_2}\cdots H_{\sigma_N}]` to describe the local surface topography of a voxel  
    seg_method : str
        one of two clustering methods

        kmeans : 
            `K-Means <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ clustering algorithm
        gmm : 
            `Gaussian mixture model <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_ algorithm with full covariances 
    n_samples : int
        the number of random sampled point to fit clustering on for computational efficiency
    n_classes :int 
        the number of clusters desired
    random_state : int
        a number to fix the random seed for reproducible clustering. 
    scale_feats : bool
        if set, the features are standard scaled prior to clustering. For mean curvature feats we find setting this seemed to make the clustering worse. 

    Returns
    -------
    depth_binary_mask, H_binary_depth_clusters : (MxNxL), (MxNxL) numpy array

        depth_binary_mask : (MxNxL) numpy array
            Binary mask of a shell of the topographic surface 
        H_binary_depth_clusters : (MxNxL) numpy array
            Integer array where background is 0 and, integers refer to the different clusters and arranged such that they reflect increasing mean curvature. 

    """
    import numpy as np 
    import skimage.morphology as skmorph
    import scipy.ndimage as ndimage
    
    # curvature only makes sense relative to a binary surface. 
    depth_binary_mask = np.logical_and(skmorph.binary_dilation(vol_binary_mask>0, skmorph.ball(depth_ksize)), 
                                       np.logical_not(skmorph.binary_erosion(vol_binary_mask>0, skmorph.ball(depth_ksize))))
    if len(smooth_curvature_sigma) == 1:
        vol_curvature_smooth = depth_binary_mask*ndimage.gaussian_filter(vol_curvature, sigma=smooth_curvature_sigma[0])

        if vol_height is not None:
            vol_curvature_smooth = np.concatenate([vol_curvature_smooth[...,None], 
                                                   vol_height[...,None]], axis=-1)
        else:
            vol_curvature_smooth = vol_curvature_smooth[...,None] # augment. 
        # if seg_method == 'kmeans':
        #     H_binary_depth_all_clusters = segmentation.multi_level_kmeans_thresh((vol_curvature_smooth[depth_binary_mask>0])[None,None,:,None],
        #                                                                        n_classes=n_classes, n_samples=n_samples, random_state=random_state, scale=scale_feats)
        # if seg_method == 'gmm':
        #     H_binary_depth_all_clusters = segmentation.multi_level_gaussian_thresh((vol_curvature_smooth[depth_binary_mask>0])[None,None,:,None],
        #                                                                        n_classes=n_classes, n_samples=n_samples, random_state=random_state, scale=scale_feats)
    else:
        vol_curvature_smooth = np.array([depth_binary_mask*ndimage.gaussian_filter(vol_curvature, sigma=sigma) for sigma in smooth_curvature_sigma])
        vol_curvature_smooth = vol_curvature_smooth.transpose(1,2,3,0) # put this in the last dimension!. 
    
        if vol_height is not None:
            vol_curvature_smooth = np.concatenate([vol_curvature_smooth, vol_height[...,None]], axis=-1)

    if seg_method == 'kmeans':
        H_binary_depth_all_clusters = segmentation.multi_level_kmeans_thresh((vol_curvature_smooth[depth_binary_mask>0])[None,None,:,:],
                                                                           n_classes=n_classes, n_samples=n_samples, random_state=random_state, scale=scale_feats)
    if seg_method == 'gmm':
        H_binary_depth_all_clusters = segmentation.multi_level_gaussian_thresh((vol_curvature_smooth[depth_binary_mask>0])[None,None,:,:],
                                                                           n_classes=n_classes, n_samples=n_samples, random_state=random_state, scale=scale_feats)
    # put these back into the volume!
    H_binary_depth_all_clusters = np.squeeze(H_binary_depth_all_clusters)

    # relabel the cluster labels in increasing curvature. 
    mean_curvature_cls = np.hstack([np.nanmean((vol_curvature_smooth[depth_binary_mask>0])[H_binary_depth_all_clusters==lab,0]) for lab in np.unique(H_binary_depth_all_clusters)])
    # conduct relabelling in the order of mean_curvature_cls.
    new_cluster_labels = np.argsort(mean_curvature_cls) 

    H_binary_depth_all_clusters_ = np.zeros_like(H_binary_depth_all_clusters)
    for new_lab, old_lab in enumerate(new_cluster_labels):
        H_binary_depth_all_clusters_[H_binary_depth_all_clusters==old_lab] = new_lab
    H_binary_depth_all_clusters = H_binary_depth_all_clusters_.copy()

    H_binary_depth_clusters = np.zeros(vol_curvature.shape, dtype=np.int32) # use int!. 
    H_binary_depth_clusters[depth_binary_mask==1] = H_binary_depth_all_clusters + 1 # add 1 as these curvature clusters are not background!. 
    
    return depth_binary_mask, H_binary_depth_clusters
    
    
def remove_topography_segment_objects_binary( vol_clusters, minsize=100, uv_params_depth=None):
    r""" Removes small connected components in the binary image input either by the number of voxels or if provided the mapping to the original space, on the apparent number of voxels after geometric correction.    
    
    Parameters
    ----------
    vol_clusters : (D,U,V) array
        a binary volume
    minsize : scalar
        the minimum size, any connected components less than this is removed
    uv_params_depth :  (D,U,V,3) array
        the lookup table mapping the uniform (D,U,V) grid to (x,y,z) in original shape space, if provided the minsize is computed for (x,y,z) space not the current (d,u,v) space 
    
    Returns
    -------
    vol_clusters_new : (D,U,V) array
        a binary volume where size of connected components with > minsize removed. 
        
    """
    import numpy as np 
    import skimage.measure as skmeasure
    
    vol_clusters_new = vol_clusters.copy()
    vol_clusters_label = skmeasure.label(vol_clusters) # run connected component analysis 
    
    # measure the properties 
    props = skmeasure.regionprops(vol_clusters_label)
    
    if uv_params_depth is None:
        props_size = np.hstack([re.area for re in props]) # min_size = 100
        remove_labels = np.setdiff1d(vol_clusters_label, 0)[props_size<minsize] # should this be applied in 3D ? or just in the image? 
        
        for lab in remove_labels:
            vol_clusters_new[vol_clusters_label==lab] = 0 # set to 0 / bg
    else:
        unique_labels = np.setdiff1d(np.unique(vol_clusters_label), 0)
        if len(unique_labels) > 0: 
                                     
            # we are going to compute the actual volume using differential calculus. 
            dV_dd = np.gradient(uv_params_depth, axis=0)
            dV_dy = np.gradient(uv_params_depth, axis=1)
            dV_dx = np.gradient(uv_params_depth, axis=2)
            
            dVol = np.abs(np.nansum(np.cross(dV_dx, dV_dy, axis=-1) * dV_dd, axis=-1))
            print(dVol.shape)
            
            remove_labels = []
            for lab in unique_labels:
                vol_mask = vol_clusters_label==lab
                vol_mask_area = np.nansum(dVol[vol_mask>0])
                if vol_mask_area < minsize:
                    remove_labels.append(lab)
        
            for lab in remove_labels:
                vol_clusters_new[vol_clusters_label==lab] = 0 # set to 0 / bg
                
    return vol_clusters_new
    
    
def prop_labels_watershed_depth_slices(topo_depth_clusters, depth_binary, rev_order=True, expand_labels_2d=2):
    r""" Propagate semantic labels volumetrically from surface labels in topography space using marker-seeded watershed slice-by-slice with markers seeded from surface and from top to bottom (or bottom to top).

    This function is used to propagate surface labels into the volume so as to obtain realistic volumizations of protrusion instance segmentations when mapped back to (x.y.z) space.

    Parameters
    ----------
    topo_depth_clusters : (MxNxL) numpy array
        integer labelled volume where background voxels = 0 and unique integers > 0 represent unique connected objects.  
    depth_binary : (MxNxL) numpy array
        the topography binary volume defining the voxels that needs semantic labeling
    rev_order :  bool
        if True, reverses the scan direction to go from top to bottom instead of bottom to top (default) 
    expand_labels_2d: int
        a preprocessing expansion of input topo_depth_clusters, to better guide the in-plane watershed propagation.            
    
    Returns
    -------
    topo_depth_clusters_ : (MxNxL) numpy array
        new integer labelled volume where background voxels = 0 and unique integers > 0 represent unique connected objects. 
    """

    import numpy as np 
    import scipy.ndimage as ndimage 
    import skimage.segmentation as sksegmentation

    D = len(topo_depth_clusters)

    if rev_order == True:
        topo_depth_clusters_ = topo_depth_clusters[::-1].copy()
        depth_binary_ = depth_binary[::-1].copy() # might want to keep as 0 labels? # how to prevent flow? # we can use piecewise cuts? to approx? 
    else:
        topo_depth_clusters_ = topo_depth_clusters.copy()
        depth_binary_ = depth_binary.copy() # might want to keep as 0 labels? # how to prevent flow? # we can use piecewise cuts? to approx? 
    
    for dd in np.arange(D):
        dtform_2D = ndimage.distance_transform_edt(depth_binary_[dd]>0)
        if dd == 0: 
            label_dd = topo_depth_clusters_[dd].copy()
            label_dd_watershed = sksegmentation.watershed(-dtform_2D, 
                                                          markers=sksegmentation.expand_labels(label_dd, distance=expand_labels_2d), 
                                                          mask=depth_binary_[dd]>0)  # labels = watershed(-distance, markers, mask=image)
            topo_depth_clusters_[dd] = label_dd_watershed.copy()
        # if dd>0:
        else:
            label_dd = topo_depth_clusters_[dd].copy()
            joint_mask = np.logical_and(depth_binary_[dd]>0, depth_binary_[dd-1]>0) # get the join
            # same time there has to be a value.
            joint_mask = np.logical_and(joint_mask, topo_depth_clusters_[dd-1]>0)
            
            if np.sum(joint_mask) > 0: 
                label_dd[joint_mask] = (topo_depth_clusters_[dd-1][joint_mask]).copy() # then copy the vals. 
            
            label_dd_watershed = sksegmentation.watershed(-dtform_2D, 
                                                          markers=sksegmentation.expand_labels(label_dd, distance=expand_labels_2d), 
                                                          mask=depth_binary_[dd]>0)  # labels = watershed(-distance, markers, mask=image)
            topo_depth_clusters_[dd] = label_dd_watershed.copy()
    
    if rev_order:
        topo_depth_clusters_ = topo_depth_clusters_[::-1].copy()

    return topo_depth_clusters_
    

def inpaint_topographical_height_image(vol_labels_binary, 
                                       pre_smooth_ksize=1,
                                       post_smooth_ksize=3,
                                       background_height_thresh=None,
                                       inpaint_method='Telea',
                                       inpaint_radius=1,
                                       spherical_pad=True):
    r""" Given a topographical binary where 1 at (d,u,v) denotes the cell describe the cell surface as a height map where :math:`d=f(u,v)` and 'holes' are represented with :math:`d\le h_{thresh}`, where :math:`h_{thresh}` is a minimal height threshold, use image inpainting to 'infill' the holes to obtain a complete surface.

    Parameters
    ----------
    vol_labels_binary : (MxNxL) numpy array
        unwrapped topographic binary volume  
    pre_smooth_ksize : (MxNxL) numpy array
        gaussian :math:`\sigma` for presmoothing the height image 
    post_smooth_ksize :  scalar
        gaussian :math:`\sigma` for postsmoothing the inpainted height image 
    background_height_thresh: scalar
        all (u,v) pixels with height less than the specified threshold (:math:`d\le h_{thresh}`) is marked as 'holes' for inpainting 
    inpaint_method : str
        one of two classical inpainting methods implemented in `OpenCV <https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html>`_. 

        'Telea' : 
            Uses Fast Marching method of `Telea et al. <https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gga8c5f15883bd34d2537cb56526df2b5d6a892824c38e258feb5e72f308a358d52e>`_.
        'NS' : 
            Uses Navier-Stokes method of `Bertalmio et al. <https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gga8c5f15883bd34d2537cb56526df2b5d6a892824c38e258feb5e72f308a358d52e>`_. 
    spherical_pad : int
        the number of pixels to spherically pad, to soft-mimic the closed spherical boundary conditions in original (x,y,z) space 
   
    Returns
    -------
    infill_height : (UxV) numpy array
        Inpainted height image representing the hole-completed topographic surface

    (background_height, infill_mask) : 

        background_height : (UxV) numpy array
            Height image of the input surface 
        infill_mask : (UxV) numpy array
            Binary image where 1 denotes the region to be infilled

    """

    import numpy as np 
    import cv2
    import skimage.morphology as skmorph 
    import scipy.ndimage as ndimage
    
    background_height_coords = np.argwhere(vol_labels_binary>0)
    background_height = np.zeros(vol_labels_binary.shape[1:]) # build the image. 
    background_height[background_height_coords[:,1], 
                      background_height_coords[:,2]] = background_height_coords[:,0] # assuming the 1st = height. 
    
    # mark the infill mask. 
    if pre_smooth_ksize is not None:
        if background_height_thresh is None:
            infill_mask = skmorph.binary_dilation(background_height<1, skmorph.disk(pre_smooth_ksize))
        else:
            infill_mask = skmorph.binary_dilation(background_height<background_height_thresh, skmorph.disk(pre_smooth_ksize))
    
    # for rescaling purposes. 
    background_height_max = background_height.max()
    
    if inpaint_method == 'Telea':
        infill_height = cv2.inpaint(np.uint8(255*background_height/background_height_max), np.uint8(255*infill_mask), inpaint_radius, cv2.INPAINT_TELEA)
    if inpaint_method == 'NS':
        infill_height = cv2.inpaint(np.uint8(255*background_height/background_height_max), np.uint8(255*infill_mask), inpaint_radius, cv2.INPAINT_NS)
    
    infill_height = infill_height/255. * background_height_max

    if spherical_pad:
        # we need to ensure that things connect linearly # a quick fix is to set left equal to right and the top and bottom to the mean with minimal changes.  
        infill_height_ = infill_height.copy()
        infill_height_[0,:] = np.nanmean(infill_height[0,:])
        infill_height_[-1,:] =  np.nanmean(infill_height[-1,:])
        infill_height_[:,-1] = infill_height_[:,0].copy()
        infill_height = infill_height_.copy()

    if post_smooth_ksize is not None:
        infill_height = ndimage.gaussian_filter(infill_height, sigma=post_smooth_ksize) # maybe a regulariser is a better smoother... 

    return infill_height, (background_height, infill_mask) 


def mask_volume_dense_propped_labels_with_height_image(vol_labels, height_map, 
                                                        ksize=1, 
                                                        min_size=500, 
                                                        connectivity=2,
                                                        keep_largest=False):
    r""" This function uses a hole-completed basal surface such as that from :func:`unwrap3D.Analysis_Functions.topography.inpaint_topographical_height_image` to isolate only the protrusions from a volume-dense volumization such as the output from :func:`unwrap3D.Analysis_Functions.topography.prop_labels_watershed_depth_slices`

    Parameters
    ----------
    vol_labels : (MxNxL) numpy array
        integer labelled volume where background voxels = 0 and unique integers > 0 represent unique connected objects. 
    height_map : (MxN) numpy array
        height image at every (u,v) pixel position specifying the basal surface :math:`d=f(u,v)` such that all voxels in vol_labels with coordinates :math:`d_{uv}\le d` will be set to a background label of 0
    ksize :  int
        Optional morphological dilation of the binary surface specified by height map with a ball kernel of radius ksize. Can be used to control the extent of exclusion of the basal surface. 
        Set this parameter to None to not do morphological processing.
    min_size: scalar
        The minimum size of connected components to keep after masking
    connectivity : int
        specifies the connectivity of voxel neighbors. If 1, the 6-connected neighborhood, if 2, the full 26-connected neighborhood including the diagonals 
    keep_largest : bool
        if True, for labels that end up disconnected after the masking, retain only the largest connected region. 

    Returns
    -------
    vol_labels_new : (MxNxL) numpy array
        the non excluded integer labelled volume where background voxels = 0 and unique integers > 0 represent unique connected objects. 
    infill_height_mask : (MxN) numpy array
        the exclusion volume specified by height_map after optional ksize morphological dilation.

    """

    import numpy as np 
    import skimage.morphology as skmorph 

    # use the height to mask out all invalid volumes... 
    # depth_height = depth_mesh.vertices[:,0].reshape(H_binary_.shape[1:])
    ZZ, _, _ = np.indices(vol_labels.shape)
    if ksize is not None:
        infill_height_mask = np.logical_not(skmorph.binary_dilation(np.logical_not(ZZ >= height_map), skmorph.ball(ksize)))
    else:
        infill_height_mask = ZZ >= height_map
    
    # apply mask. 
    vol_labels_new = vol_labels*infill_height_mask
    # postprocess - keep only the largest component in each cluster label!   
    keep_mask = skmorph.remove_small_objects(vol_labels_new>0, min_size=min_size, connectivity=connectivity)
    vol_labels_new = vol_labels_new*keep_mask # apply the mask. 

    if keep_largest:
        # suppressing this should be faster. 
        # for each label we should only have 1 connected component! 
        vol_labels_new = segmentation.largest_component_vol_labels(vol_labels_new, connectivity=connectivity)

    return vol_labels_new, infill_height_mask
    

