
import numpy as np 

def smooth_vol(vol, ds=4, smooth=5, method='gaussian'):
    r""" Smoothing particularly a 3D volume image with large Gaussian kernels or Median filters is extremely slow. This function combines downsampling of the original volume image with smaller kernel smoothing on the downsampled image before upsampling to do significantly faster smoothing for large arrays.

    Parameters
    ----------
    vol : array
        input image 
    ds : int
        the downsampling factor, the downsampled shape will have size ``vol.shape//ds``
    smooth : scalar
        the size of smoothing; ``sigma`` for scipy.ndimage.gaussian_filter or ``size`` for scipy.ndimage.median_filter

    Returns
    ------- 
    smoothed image, the same size as the input

    """
    from skimage.filters import gaussian
    from scipy.ndimage import gaussian_filter, median_filter
    import skimage.transform as sktform
    import numpy as np 
    
    small = sktform.resize(vol, np.array(vol.shape)//ds, preserve_range=True)
    if method == 'gaussian':
        small = gaussian_filter(small, sigma=smooth)
    if method == 'median':
        small = median_filter(small, size=smooth)

    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)

def largest_component_vol(vol_binary, connectivity=1):
    r""" Given a binary segmentation, return only the largest connected component of the given connectivity

    Parameters
    ----------
    vol : array
        input binary image 
    connectivity : 1 or 2
        if 1, the local 4-neighbors for 2D or 6-neighbors for 3D. 
        if 2, the local 8-neighbors for 2D or 26-neighbors for 3D. 
   
    Returns
    ------- 
    vol_binary : array
        output binary image same size as input retaining only the largest connected component

    """
    from skimage.measure import label, regionprops
    import numpy as np 
    
    vol_binary_labelled = label(vol_binary, connectivity=connectivity)
    # largest component.
    vol_binary_props = regionprops(vol_binary_labelled)
    vol_binary_vols = [re.area for re in vol_binary_props]
    vol_binary = vol_binary_labelled == (np.unique(vol_binary_labelled)[1:][np.argmax(vol_binary_vols)])
    
    return vol_binary

def largest_component_vol_labels(vol_labels, connectivity=1, bg_label=0):
    r""" Given a multi-label integer image, return for each unique label, the largest connected component such that 1 label, 1 connected area. Useful to enforce the spatial uniqueness of a label.

    Parameters
    ----------
    vol_labels : array
        input multi-label integer image  
    connectivity : 1 or 2
        if 1, the local 4-neighbors for 2D or 6-neighbors for 3D. 
        if 2, the local 8-neighbors for 2D or 26-neighbors for 3D. 
    bg_label : 0 
        the integer label of background non-object areas
   
    Returns
    ------- 
    vol_labels_new : array
        output multi-label integer image where every unique label is associated with only one connected region. 

    """
    import numpy as np 
    
    # now do a round well each only keeps the largest component. 
    uniq_labels = np.setdiff1d(np.unique(vol_labels), bg_label)
    vol_labels_new = vol_labels.copy()

    for lab in uniq_labels:
        mask = vol_labels == lab 
        mask = largest_component_vol(mask, connectivity=connectivity)
        vol_labels_new[mask>0] = lab # put this. 
    
    return vol_labels_new


def get_bbox_binary_2D(mask, feature_img=None, prop_nan=True):
    r""" Given a 2D binary image, return the largest bounding box described in terms of its top left and bottom right coordinate positions. If given the corresponding feature_img, compute the average feature vector describing the contents inside the bounding box and concatenate this with the bounding box coordinates for downstream applications.
    This function is primarily useful for describing a region of interest with a bounding box for downstream tracking, classification and clustering applications.

    Parameters
    ----------
    mask : array
        input binary image  
    feature_img : array 
        if not None, should be a ``mask.shape + (F,)`` array, where F is the number of features for which we would like the average (using mean) over any detected bounding box to append to the bounding box coordinates to be returned     
    prop_nan : bool 
        if True, when no valid bounding box can be detected e.g. for a 1 pixel only binary area, the bounding box coordinates and associated features (if specified) are subsituted for by ``np.nan``. Otherwise, the empty list [] is returned. This flag is useful for downstream applications where a regular sized array may be required.  
   
    Returns
    ------- 
    bbox : (N,) array
        the bounding box described by its top-left (x1,y1) and bottom right (x2,y2) coordinates given as a 1d array of x1,y1,x2,y2 concatenated if specified by the mean feature vector
    
    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.get_bbox_labels_2D` : 
        this function does the same for multi-label segmentation images

    """

    import numpy as np 
    yyxx = np.argwhere(mask>0) 
    yy = yyxx[:,0]
    xx = yyxx[:,1]
    
    x1 = np.min(xx)
    x2 = np.max(xx)
    y1 = np.min(yy)
    y2 = np.max(yy)

    if x2>x1 and y2>y1: 
        if feature_img is not None:
            features_box = np.hstack([np.nanmean(ff[mask>0]) for ff in feature_img])
            # features_box = np.nanmean(feature_img[mask>0], axis=-1) # average over the last dimension. 
            bbox = np.hstack([x1,y1,x2,y2, features_box])
        else:
            bbox = np.hstack([x1,y1,x2,y2])
    else:
        if prop_nan:
            if feature_img is not None:
                bbox = np.hstack([np.nan, np.nan, np.nan, np.nan, np.nan*np.ones(feature_img.shape[-1])])
            else:
                bbox = np.hstack([np.nan, np.nan, np.nan, np.nan])
        else:
            bbox = []
        
    return bbox
    

def get_bbox_labels_2D(label_img, feature_img=None, prop_nan=True, bg_label=0, split_multi_regions=False):
    r""" Given a 2D multi-label image, iterate over each unique foreground labelled regions and return the bounding boxes described in terms of its top left and bottom right coordinate positions and label. If given the corresponding feature_img, compute the average feature vector describing the contents inside the bounding box and concatenate this with the bounding box coordinates for downstream applications.
    This function is useful for describing regions of interest with a bounding box for downstream tracking, classification and clustering applications.

    Parameters
    ----------
    label_img : array
        input multi-labeled image  
    feature_img : array 
        if not None, should be a ``label_img.shape + (F,)`` array, where F is the number of features for which we would like the average (using mean) over any detected bounding box to append to the bounding box coordinates to be returned     
    prop_nan : bool 
        if True, when no valid bounding box can be detected e.g. for a 1 pixel only area, the bounding box coordinates and associated features (if specified) are subsituted for by ``np.nan``. Otherwise, the empty list [] is returned. This flag is useful for downstream applications where a regular sized array may be required.  
    bg_label : int 
        the integer label marking background regions
    split_multi_regions : bool
        if True, this function will derive generate a bounding box for each disconnected region with the same label

    Returns
    ------- 
    bboxes : list(array) of (N,) arrays
        all the detected bounding boxes for all labelled regions where each bounding box is described by the region label and its top-left (x1,y1) and bottom right (x2,y2) coordinates given as a 1d array of label, x1,y1,x2,y2 concatenated if specified by the mean feature vector. The return will be a regular 2-d numpy array if prop_nan is True, otherwise if one region did not detect a valid bounding box the result would be a list of arrays

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.get_bbox_binary_2D` : 
        this function does the same for binary segmentation images

    """

    import numpy as np 
    import skimage.measure as skmeasure 
    
    bboxes = []
    
    for lab in np.setdiff1d(np.unique(label_img), bg_label):
        mask = label_img==lab 
        
        if split_multi_regions==True:
            # if split then we do a separate connected components 
            labelled_mask = skmeasure.label(mask, connectivity=2) # 
            uniq_regions = np.setdiff1d(np.unique(labelled_mask),0)
            
            for region in uniq_regions:
                mask_region = labelled_mask==region 
                bbox = get_bbox_binary_2D(mask_region, feature_img=feature_img)
                if len(bbox) > 0: 
                    bboxes.append(np.hstack([lab, bbox]))
        else:
            bbox = get_bbox_binary_2D(mask, feature_img=feature_img)
            if len(bbox) > 0: 
                bboxes.append(np.hstack([lab, bbox]))
                
    if len(bboxes)>0:
        bboxes = np.vstack(bboxes)

    return bboxes


# crops an image box given an image or given a binary
def crop_box_3D(im, thresh=None, pad=50):
    r""" Derive the 3D bounding box given a volume intensity image or a volume binary image with optional additional padding. The input image is only specified if a constant scalar threshold, ``thresh`` is provided.

    Parameters
    ----------
    im : array
        input image  
    thresh : scalar 
        if None, the input image will be assumed binary and the bounding box will be determined by the largest connected component. If not None, the image is first binarised with ``im>=thresh``.
    pad : int 
        the isotropic padding to expand the found bounding box in all xyz-directions 

    Returns
    ------- 
    bbox : (6,) array
        the bounding box described by its top-left (x1,y1,z1) and bottom right (x2,y2,z2) coordinates concatenated as a vector [x1,y1,z1,x2,y2,z2]

    """
    import numpy as np 
   
    l,m,n = im.shape

    if thresh is not None:
        binary = im>=thresh
    else:
        binary = im.copy() # the input is already binarised. 
    binary = largest_component_vol(binary)
    
    # min_zz, min_yy, min_xx, max_zz, max_yy, max_xx = bounding_box(binary)
    ZZ, YY, XX = np.indices(binary.shape)
    
    min_zz = np.min(ZZ[binary])
    max_zz = np.max(ZZ[binary])
    min_yy = np.min(YY[binary])
    max_yy = np.max(YY[binary])
    min_xx = np.min(XX[binary])
    max_xx = np.max(XX[binary])
    
    min_zz = np.clip(min_zz - pad, 0, l-1)
    max_zz = np.clip(max_zz + pad, 0, l-1)
    min_yy = np.clip(min_yy - pad, 0, m-1)
    max_yy = np.clip(max_yy + pad, 0, m-1)
    min_xx = np.clip(min_xx - pad, 0, n-1)
    max_xx = np.clip(max_xx + pad, 0, n-1)

    bbox = np.hstack([min_zz, min_yy, min_xx, max_zz, max_yy, max_xx])
    
    return bbox

def segment_vol_thresh( vol, thresh=None, postprocess=True, post_ksize=3):
    r""" Basic image segmentation based on automatic binary Otsu thresholding or a specified constant threshold with simple morphological postprocessing

    Parameters
    ----------
    vol : array
        the input image to segment on intensity
    thresh : scalar
        if None, determine the constant threshold using Otsu binary thresholding else the binary is given by ``vol >= thresh``
    postprocess : bool
        if True, the largest connected component is retained, small holes are closed with a disk (2D) or ball kernel (3D) of radius given by ``post_ksize`` and finally the resulting binary is binary filled.
    post_ksize : int
        the size of the kernel to morphologically close small holes of ``postprocess=True`` 

    Returns
    -------
    im_binary : array
        the final binary segmentation image 
    im_thresh : scalar
        the intensity threshold used 

    """
    from skimage.filters import threshold_otsu
    import skimage.morphology as skmorph
    from scipy.ndimage.morphology import binary_fill_holes

    if thresh is None:
        im_thresh = threshold_otsu(vol.ravel())
    else:
        im_thresh = thresh

    im_binary = vol >= im_thresh

    if postprocess: 
        im_binary = largest_component_vol(im_binary) # ok. here we keep only the largest component. -> this is crucial to create a watertight segmentation.  
        if len(vol.shape) == 3: 
            im_binary = skmorph.binary_closing(im_binary, skmorph.ball(post_ksize))
        if len(vol.shape) == 2: 
            im_binary = skmorph.binary_closing(im_binary, skmorph.disk(post_ksize))
        im_binary = binary_fill_holes(im_binary) # check there is no holes!

    # return the volume and the threshold. 
    return im_binary, im_thresh


# also create a multi-level threshold? algorithm.
def multi_level_gaussian_thresh(vol, n_classes=3, n_samples=10000, random_state=None, scale=False):
    r""" Segments an input volume image into n_classes using bootstrapped Gaussian mixture model (GMM) clustering. This allows multi-dimensional features and not just intensity to be used for segmentation. The final clustering will result in larger/smoother clusters than K-means.

    Parameters
    ----------
    vol : array
        the 3D input image or 4D feature image to segment
    n_classes : int
        the number of desired clusters
    n_samples : int
        the number of randomly sampled pixels to fit the GMM 
    random_state : int
        if not None, uses this number as the fixed random seed
    scale : bool
        if True, standard scales input features before GMM fitting, see `scipy.preprocessing.StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_  

    Returns
    -------
    labels_ : array
        the final clustered image as a multi-label volume image, clusters are sorted in increasing order of the 1st feature by default 

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.multi_level_kmeans_thresh` :
        The K-Means clustering equivalent 
    """
    from sklearn.mixture import GaussianMixture
    import numpy as np 

    model = GaussianMixture(n_components=n_classes, random_state=random_state)
    
    volshape = vol.shape[:3]
    
    if len(vol.shape) > 3:
        vals = vol.reshape(-1,vol.shape[-1])
    else:
        vals = vol.ravel()[:,None]
        
    if random_state is not None:
        np.random.seed(random_state) # make this deterministic!. 
    random_select = np.arange(len(vals))
    np.random.shuffle(random_select)
    
    # if applying scale we need to do standard scaling of features. 
    if scale: 
        from sklearn.preprocessing import StandardScaler
        vals = StandardScaler().fit_transform(vals) # apply the standard scaling of features prior to 

    X = vals[random_select[:n_samples]] # random sample to improve inference time. 
    model.fit(X)
    
    if len(vol.shape) > 3:
        labels_means = model.means_[:,0].ravel()
    else:
        labels_means = model.means_.ravel()

    labels_order = np.argsort(labels_means)
    
    labels = model.predict(vals)
    labels_ = np.zeros_like(labels)
    
    for ii, lab in enumerate(labels_order):
        labels_[labels==lab] = ii
    
    labels_ = labels_.reshape(volshape)
    
    return labels_

def multi_level_kmeans_thresh(vol, n_classes=3, n_samples=10000, random_state=None, scale=False):
    r""" Segments an input volume image into n_classes using bootstrapped K-Means clustering. This allows multi-dimensional features and not just intensity to be used for segmentation. The final clustering will result in smaller/higher-frequency clusters than Gaussian mixture models.

    Parameters
    ----------
    vol : array
        the 3D input image or 4D feature image to segment
    n_classes : int
        the number of desired clusters
    n_samples : int
        the number of randomly sampled pixels to fit the K-Means 
    random_state : int
        if not None, uses this number as the fixed random seed
    scale : bool
        if True, standard scales input features before GMM fitting, see `scipy.preprocessing.StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_  

    Returns
    -------
    labels_ : array
        the final clustered image as a multi-label volume image, clusters are sorted in increasing order of the 1st feature by default 

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.multi_level_gaussian_thresh` :
        The Gaussian Mixture Model clustering equivalent 
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np 
    
    model = KMeans(n_clusters=n_classes, init='k-means++', random_state=random_state)
    
    volshape = vol.shape[:3]
    
    if len(vol.shape) > 3:
        vals = vol.reshape(-1,vol.shape[-1])
    else:
        vals = vol.ravel()[:,None]
        
    if random_state is not None:
        np.random.seed(random_state) # make this deterministic!. 
#    vals = StandardScaler().fit_transform(vals)
    random_select = np.arange(len(vals))
    np.random.shuffle(random_select)
    
    if scale: 
        from sklearn.preprocessing import StandardScaler
        vals = StandardScaler().fit_transform(vals)
    X = vals[random_select[:n_samples]]
    model.fit(X)
    
    if len(vol.shape) > 3:
        labels_means = model.cluster_centers_[:,0].ravel()
    else:
        labels_means = model.cluster_centers_.ravel()
    labels_order = np.argsort(labels_means)
    
    labels = model.predict(vals)
    labels_ = np.zeros_like(labels)
    
    for ii, lab in enumerate(labels_order):
        labels_[labels==lab] = ii
    
    labels_ = labels_.reshape(volshape)
    
    return labels_
    
def sdf_distance_transform(binary, rev_sign=True, method='edt'): 
    r""" Compute the signed distance function (SDF) of the shape specified by the input n-dimensional binary image. Signed distance function enables shape to be captured as a continuous function which is highly advantageous for shape arithmetic and machine learning applications.

    see https://en.wikipedia.org/wiki/Signed_distance_function for more details on the SDF. 
    
    Parameters
    ----------
    binary : array
        input n-dimensional binary image 
    rev_sign : bool
        if True, reverses the sign of the computed signed distance function. When this is True, the inside of the shape is +ve distances and -ve distances is the outside of the shape
    method : str
        specifies the method used to compute the distance transform

        'edt' : str
            This is the Euclidean distance transform computed with scipy.ndimage.distance_transform_edt
        'fmm' : str
            This is the geodesic distance transform computed with `scikit-fmm <https://github.com/scikit-fmm/scikit-fmm>`_ library

    Returns
    -------
    res : array
        the signed distance function, the same size as the input where the contours of the input binary has a distance of 0. 

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.get_label_distance_transform` :
        The multi-label equivalent but here we compute just the one-sided interior distance transform.  

    """
    import numpy as np 
    
    pos_binary = binary.copy()
    neg_binary = np.logical_not(pos_binary)
    
    if method == 'edt':
        from scipy.ndimage import distance_transform_edt
        res = distance_transform_edt(neg_binary) * neg_binary - (distance_transform_edt(pos_binary) - 1) * pos_binary
        
    if method =='fmm':
        import skmm
        res = skmm.distance(neg_binary) * neg_binary - (skmm.distance(pos_binary) - 1) * pos_binary

    if rev_sign:
        res = res * -1
    
    return res

def get_label_distance_transform(labelled, bg_label=0, normalise=False, rev_sign=False, method='edt'):
    r""" Compute the distance function for each label of the input multi-labelled image. The distance function captures the shape as a continuous function which is used to express for example multiple cell instances in a single image for deep learning instance-level segmenters.

    see https://en.wikipedia.org/wiki/Distance_transform for more details on the distance transform
    see https://en.wikipedia.org/wiki/Signed_distance_function for more details on the signed distance function 
    
    Parameters
    ----------
    labelled : array
        input n-dimensional multi-labelled image 
    bg_label : int 
        the integer label of the background areas
    normalise : bool
        if True, normalizes the distance transform of each label by dividing by the maximum distance
    rev_sign : bool
        if True, reverses the sign of the computed signed distance function. When this is True, the inside of the shape is +ve distances and -ve distances is the outside of the shape
    method : str
        specifies the method used to compute the distance transform

        'edt' : str
            This is the Euclidean distance transform computed with scipy.ndimage.distance_transform_edt
        'fmm' : str
            This is the geodesic distance transform computed with `scikit-fmm <https://github.com/scikit-fmm/scikit-fmm>`_ library

    Returns
    -------
    dist_tform : array
        the distance transform of each labelled region with the same size as the input

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.sdf_distance_transform` :
        The binary equivalent computing the double-sided interior and exterior signed distance transform.  

    """
    import numpy as np 

    if method == 'edt':
        from scipy.ndimage import distance_transform_edt
        dist_fnc = distance_transform_edt
    if method == 'fmm':
        import skfmm 
        dist_fnc = skfmm.distance

        # iterate over all unique_labels. 
        uniq_labels = np.setdiff1d(np.unique(labelled), bg_label)
        dist_tform = np.zeros(labelled.shape, dtype=np.float64)

        for lab in uniq_labels:
            binary_mask = labelled == lab 
            dist = dist_fnc(binary_mask)

            if rev_sign:
                dist = dist * -1
            if normalise:
                dist = dist / np.nanmax(dist)
            dist_tform[binary_mask>0] = dist[binary_mask>0].copy()

    return dist_tform


def gradient_watershed2D_binary(binary, 
                                gradient_img=None, 
                                smooth_sigma=1, 
                                smooth_gradient=1, 
                                delta=.5, 
                                n_iter=10, 
                                min_area=5, 
                                eps=1e-12, 
                                thresh_factor=None, 
                                mask=None,
                                return_tracks=False,
                                interp_bool=False):
    
    r""" Parses the instance level segmentation implicitly given as an input binary or a vector field. 
    The algorithm works as an inverse watershed.
    
    Step 1: a grid of points is seeds on the image 

    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
    result is an integer image the same size as binary. 

    Parameters
    ----------
    binary : (MxN) numpy array
        input binary image defining the voxels that need labeling
    gradient_img :  (MxNx2) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,0) where img is a potential such as a distance transform or probability map. 
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
    smooth_gradient : scalar
        the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
        the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
    n_iter: int 
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed. 
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
    mask: (MxN) numpy array
        optional binary mask to gate the region to parse labels for.
    return_tracks : bool
        if True, return the grid point trajectories 
    interp_bool : bool
        if True, interpolate the gradient field when advecting at the cost of speed. If False, point positions are clipped and this is much faster. 

    Returns
    -------
    cell_seg_connected_original : (MxN)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
    tracks : Nx2
        if return_tracks=True, returns as a second argument, the tracks of the initial seeded grid points to its final position

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.gradient_watershed3D_binary` :
        Equivalent for 3D images 
        
    """
    import scipy.ndimage as ndimage
    import numpy as np 
    import skimage.morphology as skmorph
    import pylab as plt 
    import skimage.measure as skmeasure 
    import skimage.segmentation as sksegmentation 
    from tqdm import tqdm 
    
    def interp2(query_pts, grid_shape, I_ref, method='linear', cast_uint8=False):
    
        import numpy as np 
        from scipy.interpolate import RegularGridInterpolator 
        
        spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                       np.arange(grid_shape[1])), 
                                       I_ref, method=method, bounds_error=False, fill_value=0)
        I_query = spl((query_pts[...,0], 
                       query_pts[...,1]))

        if cast_uint8:
            I_query = np.uint8(I_query)
        
        return I_query

    # compute the signed distance transform
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(2,0,1) # use the supplied gradients! 
        sdf_normals = sdf_normals * binary[None,...]
    else:
        sdf_normals, sdf_binary = surf_normal_sdf(binary, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        sdf_normals = sdf_normals * binary[None,...]
        
    # print(sdf_normals.shape)
    grid =  np.zeros(binary.shape, dtype=np.int32)
    pts = np.argwhere(binary>0) # (N,ndim)
    # print(pts.shape)
    # plt.figure(figsize=(10,10))
    # plt.imshow(binary)
    # plt.plot(pts[:,1], pts[:,0], 'r.')
    # plt.show()
    
    tracks = [pts]
    
    for ii in tqdm(np.arange(n_iter)):
        pt_ii = tracks[-1].copy()
        
        if interp_bool:
            pts_vect_ii = np.array([interp2(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
        else:
            pts_vect_ii = np.array([sdf_normals[ch][pt_ii[:,0].astype(np.int), pt_ii[:,1].astype(np.int)] for ch in np.arange(len(sdf_normals))]).T
        
        pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + 1e-12)
        pt_ii_next = pt_ii + delta*pts_vect_ii
            
        pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
        pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
        
        tracks.append(pt_ii_next)
        # plt.figure(figsize=(10,10))
        # plt.imshow(binary)
        # plt.plot(pt_ii_next[:,1], pt_ii_next[:,0], 'r.')
        # plt.show()

    tracks = np.array(tracks)
    """
    a radius neighbor graph or kNN graph here may be much more optimal here.... or use hdbscan? 
    """
    # parse ... 
    votes_grid_acc = np.zeros(binary.shape)
    votes_grid_acc[(tracks[-1][:,0]).astype(np.int), 
                   (tracks[-1][:,1]).astype(np.int)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)  
    
    if thresh_factor is not None:
        if mask is not None:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=1) # use the full conditional 
    cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    if len(cell_uniq_regions)>0:
        props = skmeasure.regionprops(cell_seg_connected)
        areas = np.hstack([re.area for re in props])
        invalid_areas = cell_uniq_regions[areas<=min_area]
    
        for invalid in invalid_areas:
            cell_seg_connected[cell_seg_connected==invalid] = 0
        
    if cell_seg_connected.max() > 0:
        cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[(pts[:,0]).astype(np.int), 
                                (pts[:,1]).astype(np.int)] = cell_seg_connected[(tracks[-1][:,0]).astype(np.int), 
                                                                                (tracks[-1][:,1]).astype(np.int)]
    
    if mask is not None:
        cell_seg_connected[mask == 0] = 0
        
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected)
    # plt.show()
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected_original)
    # plt.show()
    if return_tracks:
        return cell_seg_connected_original, tracks
    else:
        return cell_seg_connected_original    


def gradient_watershed3D_binary(binary, 
                                gradient_img=None, 
                                smooth_sigma=1, 
                                smooth_gradient=1, 
                                delta=1, 
                                n_iter=100, 
                                min_area=5, 
                                eps=1e-12, 
                                thresh_factor=None, 
                                mask=None,
                                return_tracks=False,
                                interp_bool=False):
    
    r""" Parses the instance level segmentation implicitly given as an input binary or a vector field. 
    The algorithm works as an inverse watershed.
    
    Step 1: a grid of points is seeds on the image 

    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
    result is an integer image the same size as binary. 

    Parameters
    ----------
    binary : (MxNxL) numpy array
        input binary image defining the voxels that need labeling
    gradient_img :  (MxNxLx3) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map. 
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
    smooth_gradient : scalar
        the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
        the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
    n_iter: int 
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed. 
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
    mask: (MxNxL) numpy array
        optional binary mask to gate the region to parse labels for.
    return_tracks : bool
        if True, return the grid point trajectories 
    interp_bool : bool
        if True, interpolate the gradient field when advecting at the cost of speed. If False, point positions are clipped and this is much faster. 

    Returns
    -------
    cell_seg_connected_original : (MxNxL)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
    tracks : Nx3
        if return_tracks=True, returns as a second argument, the tracks of the initial seeded grid points to its final position

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.gradient_watershed2D_binary` :
        Equivalent for 2D images 
        
    """
    import scipy.ndimage as ndimage
    import numpy as np 
    import skimage.morphology as skmorph
    import pylab as plt 
    import skimage.measure as skmeasure 
    import skimage.segmentation as sksegmentation 
    from tqdm import tqdm 

    def interp3(query_pts, grid_shape, I_ref, method='linear', cast_uint8=False):
    
        from scipy.interpolate import RegularGridInterpolator
        from scipy import ndimage
        
        spl_3 = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                             np.arange(grid_shape[1]), 
                                             np.arange(grid_shape[2])), 
                                             I_ref, method=method, bounds_error=False, fill_value=0)
        I_query = spl_3((query_pts[...,0], 
                          query_pts[...,1],
                          query_pts[...,2]))
        if cast_uint8:
            I_query = np.uint8(I_query)
        
        return I_query
    
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(3,0,1,2) # use the supplied gradients! 
        sdf_normals = sdf_normals * binary[None,...]
    else:
        # compute the signed distance transform
        sdf_normals, sdf_binary = surf_normal_sdf(binary, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        sdf_normals = sdf_normals * binary[None,...]
    
    grid =  np.zeros(binary.shape, dtype=np.int32)
    pts = np.argwhere(binary>0) # (N,ndim)
    
    tracks = [pts]
    
    for ii in tqdm(np.arange(n_iter)):
        pt_ii = tracks[-1].copy()
        
        if interp_bool:
            pts_vect_ii = np.array([interp3(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
        else:
            pts_vect_ii = np.array([sdf_normals[ch][pt_ii[:,0].astype(np.int), pt_ii[:,1].astype(np.int), pt_ii[:,2].astype(np.int)] for ch in np.arange(len(sdf_normals))]).T
        pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + 1e-12)
        
        pt_ii_next = pt_ii + delta*pts_vect_ii            
        # clip to volume bounds
        pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
        pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
        pt_ii_next[:,2] = np.clip(pt_ii_next[:,2], 0, binary.shape[2]-1)
        
        tracks[-1] = pt_ii_next # overwrite 
        # plt.figure(figsize=(10,10))
        # plt.imshow(binary.max(axis=0))
        # plt.plot(pt_ii_next[:,2], 
        #          pt_ii_next[:,1], 'r.')
        # plt.show()
    tracks = np.array(tracks)
    
    # parse ... 
    votes_grid_acc = np.zeros(binary.shape)
    votes_grid_acc[(tracks[-1][:,0]).astype(np.int), 
                   (tracks[-1][:,1]).astype(np.int),
                   (tracks[-1][:,2]).astype(np.int)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)  
    
    if thresh_factor is not None:
        if mask is not None:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=2)
    cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    if len(cell_uniq_regions)>0:
        props = skmeasure.regionprops(cell_seg_connected)
        areas = np.hstack([re.area for re in props])
        invalid_areas = cell_uniq_regions[areas<=min_area]
    
        for invalid in invalid_areas:
            cell_seg_connected[cell_seg_connected==invalid] = 0
        
    if cell_seg_connected.max() > 0:
        cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    
    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[(pts[:,0]).astype(np.int), 
                                (pts[:,1]).astype(np.int),
                                (pts[:,2]).astype(np.int)] = cell_seg_connected[(tracks[-1][:,0]).astype(np.int), 
                                                                                (tracks[-1][:,1]).astype(np.int),
                                                                                (tracks[-1][:,2]).astype(np.int)]
                                                 
    if mask is not None:
        cell_seg_connected[mask == 0] = 0
        
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected.max(axis=0))
    # plt.show()
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected_original.max(axis=0))
    # plt.show()

    if return_tracks:
        return cell_seg_connected_original, tracks
    else:
        return cell_seg_connected_original    


def surf_normal_sdf(binary, smooth_gradient=None, eps=1e-12, norm_vectors=True):
    r""" Given an input binary compute the signed distance function with positive distances for the shape interior and the gradient vector field of the signed distance function. The gradient vector field passes through the boundaries of the binary at normal angles.

    Parameters
    ----------
    binary : array
        input n-dimensional binary image
    smooth_gradient : scalar
        if not None, Gaussian smoothes the gradient vector field with ``sigma=smooth_gradient``
    eps : scalar 
        small value for numerical stabilty
    norm_vectors : bool
        if True, normalise the gradient vector field such that all vectors have unit magnitude

    Returns
    -------
    sdf_vol_normal : array
        the gradient vector field of the signed distance function of the binary
    sdf_vol : array 
        the signed distance function of the binary, with positive distances denoting the interior and negative distances the exterior

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.sdf_distance_transform` :
        For computing just the signed distance function

    """
    import numpy as np 
    import scipy.ndimage as ndimage

    sdf_vol = sdf_distance_transform(binary, rev_sign=True) # so that we have it pointing outwards!. 
    
    # compute surface normal of the signed distance function. 
    sdf_vol_normal = np.array(np.gradient(sdf_vol))
    # smooth gradient
    if smooth_gradient is not None: # smoothing needs to be done before normalization of magnitude. 
        sdf_vol_normal = np.array([ndimage.gaussian_filter(sdf, sigma=smooth_gradient) for sdf in sdf_vol_normal])

    if norm_vectors:
        sdf_vol_normal = sdf_vol_normal / (np.linalg.norm(sdf_vol_normal, axis=0)[None,:]+eps)

    return sdf_vol_normal, sdf_vol


def edge_attract_gradient_vector(binary, 
                                 return_sdf=True, 
                                 smooth_gradient=None, 
                                 eps=1e-12, 
                                 norm_vectors=True,
                                 rev_sign=False):
    r""" Given an input binary compute an edge aware signed distance function which pulls all points in the volume towards the boundary edge of the binary. The construction is based on computing the signed distance function. 

    Parameters
    ----------
    binary : array
        input n-dimensional binary image
    return_sdf : bool
        if True, return the signed distance function 
    smooth_gradient : scalar
        if not None, Gaussian smoothes the gradient vector field with ``sigma=smooth_gradient``
    eps : scalar 
        small value for numerical stabilty
    norm_vectors : bool
        if True, normalise the gradient vector field such that all vectors have unit magnitude
    rev_sign : bool
        if True, create the opposite edge repelling field 

    Returns
    -------
    sdf_vol_vector : array
        the edge attracting or edge repelling gradient vector field of the signed distance function of the binary
    sdf_vol : array 
        the signed distance function of the binary, with positive distances denoting the interior and negative distances the exterior

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.surf_normal_sdf` :
        Similar gradient vector field, but pulls every point in the volume towards a central point within the binary

    """
    import numpy as np 
    import scipy.ndimage as ndimage

    pos_binary = binary.copy()
    neg_binary = np.logical_not(pos_binary)

    sdf_vol_vector, sdf_vol = surf_normal_sdf(binary, 
                                            smooth_gradient=smooth_gradient, 
                                            eps=eps, 
                                            norm_vectors=norm_vectors)

    # now we can invert the vectors.... 
    sdf_vol_vector = (-sdf_vol_vector) * pos_binary[None,...] + (sdf_vol_vector) * neg_binary[None,...]

    return sdf_vol_vector, sdf_vol


def mean_curvature_field(sdf_normal):
    r""" Compute the mean curvature given a vector field, :math:`V`. This is defined as 

    .. math::
        H = -\frac{1}{2}\nabla\cdot\left( V\right)
    
    The output is a scalar field. The vector dimension is the first axis. 

    Parameters
    ----------
    sdf_normal : array
        input (d,) + n-dimensional gradient field 

    Returns
    -------
    H : array
        output n-dimensional divergence 
       
    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.mean_curvature_binary` :
        Function wraps this function to compute the mean curvature given a binary 

    """
    def divergence(f):
        import numpy as np 
        """
        Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
        :param f: List of ndarrays, where every item of the list is one dimension of the vector field
        :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
        """
        num_dims = len(f)
        return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])
        
    # add the negative so that it is consistent with a positive distance tranform for the internal! 
    H = -.5*(divergence(sdf_normal))# total curvature is the divergence of the normal. 
    
    return H 

def mean_curvature_binary(binary, smooth=3, mask=True, smooth_gradient=None, eps=1e-12):
    r""" All in one function to compute the signed distance function, :math:`\Phi`, its normalised gradient field, :math:`\nabla\Phi /|\nabla\Phi|` and the mean curvature defined as 

    .. math::
        H = -\frac{1}{2}\nabla\cdot\left( \frac{\nabla \Phi}{|\nabla \Phi|}\right)
    
    The output is a scalar field. 

    Parameters
    ----------
    binary : array
        input n-dimensional binary image  
    smooth : scalar
        the sigma of the Gaussian smoother for smoothing the computed mean curvature field and the width of the derived surface mask if ``mask=True``
    mask : bool
        if True, the mean curvature is restricted to the surface captured by binary derived internally as a thin shell of width ``smooth``. Non surface values will then be returned as np.nan
    smooth_gradient : scalar
        if not None, Gaussian smoothes the gradient vector field with the provided sigma value before computing mean curvature
    eps : scalar
        small number for numerics

    Returns
    -------
    H_normal : array
        output n-dimensional mean curvature
    sdf_vol_normal : array 
        (n,) + n-dimensional gradient of the signed distance function. The first axis is the vectors.
    sdf_vol : array
        n-dimensional signed distance function
       
    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.mean_curvature_field` :
        for the computation of mean curvature where the input vector field is the normalised gradient vector the binary signed distance function.

    """
    import skimage.morphology as skmorph 
    import scipy.ndimage
    import numpy as np 

    sdf_vol_normal, sdf_vol = surf_normal_sdf(binary, smooth_gradient=smooth_gradient, eps=eps)
    H_normal = mean_curvature_field(sdf_vol_normal)

    if smooth is not None:
        H_normal = scipy.ndimage.gaussian_filter(H_normal, sigma=smooth)
    if mask:
        # do we really need to pass an additional mask here? 
        mask_outer = skmorph.binary_dilation(binary, skmorph.ball(smooth))
        mask_inner = skmorph.binary_erosion(binary, skmorph.ball(smooth))
        mask_binary = np.logical_and(mask_outer, np.logical_not(mask_inner))
        H_normal[mask_binary==0] = np.nan

    return H_normal, sdf_vol_normal, sdf_vol


def extract_2D_contours_img_and_curvature(binary, presmooth=None, k=4, error=0.1):
    r""" Given an input binary image, extract the largest closed 2D contour and compute the curvature at every point of the curve using splines.
    
    It is assumed there is only 1 region, 1 contour. 

    Parameters
    ----------
    binary : (MxN) image
        input 2D binary image
    presmooth : scalar
        if not None, smoothes the binary before running ``skimage.measure.find_contours`` with an isolevel of 0.5 to extract the 2D contour otherwise no smoothing is applied and we extract at isolevel of 0.
    k : int
        the polynomial order of the interpolating spline used for computing line curvature
    error : scalar
        The allowed error in the spline fitting. Controls the smoothness of the fitted spline. The larger the error, the more smooth the fitting and curvature value variation. 

    Returns
    -------
    contour : (N,2) array
        array of xy coordinates of the contour line 
    contour_prime : (N,2) array
        array of the 1st derivative of the contour line 
    contour_prime_prime : (N,2) array
        array of the 2nd derivative of the contour line 
    kappa :  (N,) array
        array of the line curvature values at each point on the contour
    orientation : angle in radians
        the orientation of the region 

    See Also
    --------
    :func:`unwrap3D.Segmentation.segmentation.curvature_splines` : 
        Function used to compute line curvature 

    """

    import skimage.measure as skmeasure
    import scipy.ndimage as ndimage

    if presmooth:
        binary_smooth = ndimage.gaussian_filter(binary*1, sigma=presmooth)
        contour = skmeasure.find_contours(binary_smooth, 0.5)[0]
        # contour = contour[np.argmax([len(cc) for cc in contour])] # largest
    else:
        binary_smooth = binary.copy()
        contour = skmeasure.find_contours(binary_smooth, 0)[0]
        # contour = contour[np.argmax([len(cc) for cc in contour])] # largest 

    orientation = skmeasure.regionprops(binary*1)[0].orientation # this is the complementary angle
    orientation = np.pi/2. - orientation
    contour = contour[:,::-1].copy() # to convert to x-y convention 

    # refine this with errors. 
    (x,y), (x_prime, y_prime), (x_prime_prime, y_prime_prime), kappa = curvature_splines(contour[:,0], y=contour[:,1], k=k, error=error)
    contour = np.vstack([x,y]).T
    contour_prime = np.vstack([x_prime,y_prime]).T
    contour_prime_prime = np.vstack([x_prime_prime,y_prime_prime]).T
 
    return contour, contour_prime, contour_prime_prime, kappa, orientation


# curvature of line using spline fitting. 
def curvature_splines(x, y=None, k=4, error=0.1):
    r"""Calculate the signed curvature of a 2D curve at each point using interpolating splines.
    
    Parameters
    ----------
    x,y : numpy.array(dtype=float) shape (n_points,)
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    k : int 
        The order of the interpolating spline 
    error : float
        The admisible error when interpolating the splines
    
    Returns
    -------
    [x_, y_] : list of [(n_points,), (n_points,)] numpy.array(dtype=float)
        the x, y coordinates of the interpolating spline where curvature was evaluated. 
    [x_prime, y_prime] : list of [(n_points,), (n_points,)] numpy.array(dtype=float)
        the x, y first derivatives of the interpolating spline 
    [x_prime_prime,y_prime_prime] : list of [(n_points,), (n_points,)] numpy.array(dtype=float)
        the x, y second derivatives of the interpolating spline 
    curvature: numpy.array shape (n_points,)
        The line curvature at each (x,y) point
    """
    from scipy.interpolate import UnivariateSpline
    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=k, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=k, w=1 / np.sqrt(std))
    # fx = UnivariateSpline(t, x, k=k, s=smooth)
    # fy = UnivariateSpline(t, y, k=k, s=smooth)
    
    x_ = fx(t)
    y_ = fy(t)

    x_prime = fx.derivative(1)(t)
    x_prime_prime = fx.derivative(2)(t)
    y_prime = fy.derivative(1)(t)
    y_prime_prime = fy.derivative(2)(t)
    curvature = (x_prime * y_prime_prime - y_prime* x_prime_prime) / np.power(x_prime** 2 + y_prime** 2, 3. / 2)
#    return [x_, y_], [xˈ, yˈ], curvature
    return [x_, y_], [x_prime, y_prime], [x_prime_prime,y_prime_prime], curvature

def reorient_line(xy):
    r""" Convenience function to reorient a given xy line to be anticlockwise orientation using the sign of the vector area
    
    Parameters 
    ----------
    xy : (n_points,2) array
        The input contour in xy coordinates 

    Returns
    -------
    xy_reorient : (n_points,2) array
        The reoriented input contour in xy coordinates 
    """
    # mean_xy = np.mean(xy, axis=0)
    # xy_ = xy - mean_xy[None,:]
    
    # angle = np.arctan2(xy_[:,1], xy_[:,0])
    # # angle_diff = np.mean(angle[1:] - angle[:-1]) # which way to go. # is this the correct way to average? 
    # angle_diff = np.mean(np.sign(angle[1:] - angle[:-1])) # correct assumption -> this average was key. 
    from ..Geometry import geometry as geom

    area = geom.polyarea_2D(xy)

    if np.sign(area) < 0:
        xy_orient = xy[::-1].copy()
        return xy_orient
    else:
        xy_orient = xy.copy()
        return xy_orient