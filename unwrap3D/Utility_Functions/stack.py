        
def bounding_box(mask3D):

    r""" Given a binary mask in 3D, locate the xyz corners of the tightest bounding box without geometric transformation.  
    
    Parameters
    ----------
    mask3D : (M,N,L) binary np.bool array 
        3D binary image to compute the bounding box coordinates for
    
    Returns
    -------
    bbox : [x1,y1,z1,x2,y2,z2] 1d numpy array 
        3D bounding box of the given object specified by the 'top left' (x1,y1,z1) and 'bottom right' (x2,y2,z2) corners in 3D
    
    """
    import numpy as np 
    
    coords = np.argwhere(mask3D>0).T
    
    min_x, max_x = np.min(coords[:,0]), np.max(coords[:,0])
    min_y, max_y = np.min(coords[:,1]), np.max(coords[:,1])
    min_z, max_z = np.min(coords[:,2]), np.max(coords[:,2])

    bbox = np.hstack([min_x, min_y, min_z, max_x, max_y, max_z])
    
    return bbox
    
    
def expand_bounding_box(bbox3D, clip_limits, border=50, border_x=None, border_y=None, border_z=None):
    
    r""" Given a bounding box specified with 'top left' (x1,y1,z1) and 'bottom right' (x2,y2,z2) corners in 3D, return another such bounding box with asymmetrically expanded limits
    
    Parameters
    ----------
    bbox3D : [x1,y1,z1,x2,y2,z2] 1d numpy array 
        original 3D bounding box specified by the 'top left' (x1,y1,z1) and 'bottom right' (x2,y2,z2) corners in 3D
    clip_limits : (M,N,L) integer tuple
        the maximum bounds corresponding to the volumetric image size
    border : int
        the default single scalar for expanding a bounding box, it is overridden in select directions by setting border_x, border_y, border_z the 1st, 2nd, 3rd axes respectively.   
    border_x : int
        the expansion in the 1st axis
    border_y : int 
        the expansion in the 2nd axis 
    border_z : int 
        the expansion in the 3rd axis 

    Returns
    -------
    bbox : [x1,y1,z1,x2,y2,z2] 1d numpy array 
        the coordinates of the expanded 3D bounding box specified by the 'top left' (x1,y1,z1) and 'bottom right' (x2,y2,z2) corners in 3D
    
    """
    import numpy as np 
    
    clip_x, clip_y, clip_z = clip_limits
    
    new_bounds = np.zeros_like(bbox3D)
    
    for i in range(len(new_bounds)):
        if i==0:
            if border_x is not None:
                new_bounds[i] = np.clip(bbox3D[i]-border_x, clip_x[0], clip_x[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]-border, clip_x[0], clip_x[1])
        if i==3:
            if border_x is not None:
                new_bounds[i] = np.clip(bbox3D[i]+border_x, clip_x[0], clip_x[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]+border, clip_x[0], clip_x[1])
        if i==1:
            if border_y is not None:
                new_bounds[i] = np.clip(bbox3D[i]-border_y, clip_y[0], clip_y[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]-border, clip_y[0], clip_y[1])
        if i==4:
            if border_y is not None:
                new_bounds[i] = np.clip(bbox3D[i]+border_y, clip_y[0], clip_y[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]+border, clip_y[0], clip_y[1])
        if i==2:
            if border_z is not None:
                new_bounds[i] = np.clip(bbox3D[i]-border_z, clip_z[0], clip_z[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]-border, clip_z[0], clip_z[1])
        if i==5:
            if border_z is not None:
                new_bounds[i] = np.clip(bbox3D[i]+border_z, clip_z[0], clip_z[1])
            else:
                new_bounds[i] = np.clip(bbox3D[i]+border, clip_z[0], clip_z[1])
    
    return new_bounds
    

def crop_img_2_box(volume, bbox3D):
    
    r""" crop a given 3D volumetric image given a 3D cropping bounding box. Bounding boxes are clipped internally to the size of the volume. 
    
    Parameters
    ----------
    volume : (M,N,L) numpy array
        3D image to crop
    bbox3D : [x1,y1,z1,x2,y2,z2] 1d numpy array 
        3D cropping bounding box specified by the 'top left' (x1,y1,z1) and 'bottom right' (x2,y2,z2) corners in 3D
    
    Returns
    -------
    cropped : cropped numpy array
        volume[x1:x2,y1:y2,z1:z2]
    
    """
    import numpy as np 

    x1,y1,z1,x2,y2,z2 = bbox3D
    
    M, N, L = volume.shape
    x1 = np.clip(x1, 0, M-1)
    x2 = np.clip(x2, 0, M-1)
    y1 = np.clip(y1, 0, N-1)
    y2 = np.clip(y2, 0, N-1)
    z1 = np.clip(z1, 0, L-1)
    z2 = np.clip(z2, 0, L-1)

    cropped = volume[x1:x2,y1:y2,z1:z2]
    
    return cropped
    
    
