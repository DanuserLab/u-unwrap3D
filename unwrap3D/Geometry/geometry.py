
import numpy as np 


def squicircle(uv, _epsilon = 0.0000000001):
    r""" Compute the squicircle mapping unit disk coordinates to the square on [-1,1]x[-1,1]

    see 'Analytical Methods for Squaring the Disc' - https://arxiv.org/abs/1509.06344

    Parameters
    ----------
    uv_ : (n_vertices,2) array
        the coordinates of the the disk to square
    _epsilon : scalar
        small scalar for numerical stability 

    Returns
    -------
    square : (n_vertices,2) array
        the coordinates of the corresponding square on [-1,1]x[-1,1]

    """
    import numpy as np 
    #_fgs_disc_to_square
    u = uv[:,0].copy()
    v = uv[:,1].copy()
    x = u.copy()
    y = v.copy()

    u2 = u * u
    v2 = v * v
    r2 = u2 + v2

    uv = u * v
    fouru2v2 = 4.0 * uv * uv
    rad = r2 * (r2 - fouru2v2)
    sgnuv = np.sign(uv)
    sgnuv[uv==0.0] = 0.0
    sqrto = np.sqrt(0.5 * (r2 - np.sqrt(rad)))

    y[np.abs(u) > _epsilon] = (sgnuv / u * sqrto)[np.abs(u) > _epsilon]
    x[np.abs(v) > _epsilon] = (sgnuv / v * sqrto)[np.abs(v) > _epsilon]

    square = np.vstack([x,y]).T
    
    return square


def elliptical_nowell(uv):
    r""" Compute the elliptical of Nowell mapping unit disk coordinates to the square on [-1,1]x[-1,1]

    see 'Analytical Methods for Squaring the Disc' - https://arxiv.org/abs/1509.06344

    Parameters
    ----------
    uv : (n_vertices,2) array
        the coordinates of the the disk to square
    
    Returns
    -------
    square : (n_vertices,2) array
        the coordinates of the corresponding square on [-1,1]x[-1,1]

    """
    import numpy as np 

    u = uv[:,0].copy()
    v = uv[:,1].copy()

    x = .5*np.sqrt(2.+2.*u*np.sqrt(2) + u**2 - v**2) - .5*np.sqrt(2.-2.*u*np.sqrt(2) + u**2 - v**2)
    y = .5*np.sqrt(2.+2.*v*np.sqrt(2) - u**2 + v**2) - .5*np.sqrt(2.-2.*v*np.sqrt(2) - u**2 + v**2)
    
    # there is one nan. 
    nan_select = np.isnan(x)
    # check the nan in the original 
    x[nan_select] = np.sign(u[nan_select]) # map to 1 or -1
    y[nan_select] = np.sign(v[nan_select])

    square = np.vstack([x,y]).T

    return square


def isotropic_scale_pts(pts, scale):
    r""" Isotropic scaling of points with respect to the centroid.

    Parameters
    ----------
    pts : (nxd) array
        an nxd matrix of d-dimension coordinates where d=2 for 2D and d=3 for 3D 

    Returns 
    -------
    pts_ : (nxd) array
        the rescaled coordinates

    """
    pts_ = pts.reshape(-1,pts.shape[-1])
    centroid = np.nanmean(pts_[pts_[:,0]>0], axis=0)
    
    pts_diff = pts_ - centroid[None,:]
    pts_ = centroid[None,:] + pts_diff*scale
    pts_[pts_[:,0]==0] = 0 # don't get it? 
    
    pts_ = pts_.reshape(pts.shape)
    
    return pts_


def rotmatrix3D_to_4D(rot_matrix3D):
    r""" Convert a 3x3 transformation matrix on [x,y,z] to 4x4 transformation matrix in homogeneous coordinates operating on [x,y,z,1]

    Parameters
    ----------
    rot_matrix3D: (3x3) array 
        a general 3D transformation matrix

        .. math::
            T = \begin{bmatrix}
                a_{11} & a_{12} & a_{13} \\
                a_{21} & a_{22} & a_{23} \\
                a_{31} & a_{32} & a_{33}
            \end{bmatrix}

    Returns 
    -------
    rot_matrix: (4x4) array
        the corresponding 4x4 homogeneous transformation matrix 

        .. math::
            T' = \begin{bmatrix}
                a_{11} & a_{12} & a_{13} & 0\\
                a_{21} & a_{22} & a_{23} & 0\\
                a_{31} & a_{32} & a_{33} & 0\\
                0 & 0 & 0 & 1
            \end{bmatrix}

    """
    import numpy as np 

    rot_matrix = np.eye(4,4)
    rot_matrix[:3,:3] = rot_matrix3D.copy()

    return rot_matrix


def rot_pts_xyz(pts, rot_matrix, mean_pts=None, demean_pts=True):
    r""" Rotate 3D points Given homogeneous 4x4 rotation matrix 

    Parameters
    ----------
    pts : (nx3) array
        an nx3 matrix of 3D (x,y,z) coordinates
    rot_matrix : (4x4) array
        a homogeneous 4x4 matrix which captures affine + translation transformation specified by A (3x3 top left submatrix) and (3x1 rightmost submatrix) respectively.
        
        .. math::
            \textbf{x}' = A\textbf{x} + T
    mean_pts : (3,) array
        a 3D vector specifying the designed centroid of affine transformation 
    demean_pts : bool
        if True, the rotation matrix is applied with respect to the given mean_pts. If mean_pts=None, it is inferred as the mean (x,y,z) coordinate of the input points.

    Returns 
    -------
    rot_pts : (nx3) array
        an nx3 matrix of the rotated 3D (x,y,z) coordinates

    """
    if demean_pts:
        if mean_pts is not None:
            pts_ = pts - mean_pts[None,:]
        else:
            mean_pts = np.nanmean(pts, axis=0)
            pts_ = pts - mean_pts[None,:]
    else:
        pts_ = pts.copy() 

    # homogeneous
    pts_ = np.hstack([pts_, 
                      np.ones(len(pts_))[:,None]])
    rot_pts = rot_matrix.dot(pts_.T)[:3].T

    if demean_pts:
        rot_pts = rot_pts + mean_pts[None,:] # add back the mean. 

    return rot_pts


# functions to map uv image grid to sphere grid. 
def img_2_angles(m,n):
    r""" Given an (m,n) image compute the corresponding polar, :math:`\theta` and azimuthal, :math:`\phi` angles of the unit sphere for each pixel position such that :math:`(0,\lfloor n/2 \rfloor)` maps to the South pole and :math:`(m-1,\lfloor n/2 \rfloor)` maps to the North pole of the unit sphere

    Given an image specifying the cartesian space, :math:`U\times V: [0,n]\times[0,m]`, 
    the corresponding angle space is, :math:`\phi\times \theta: [-\pi,\pi]\times[-\pi,0]`
    
    Parameters
    ----------
    m : int
        the height of the image
    n : int
        the width of the image

    Returns 
    -------
    psi_theta_grid : (mxnx2) array
        the corresponding :math:`(\phi,\theta)` coordinates for each :math:`(u,v)` image coordinate. ``psi_theta_grid[:,:,0]`` is :math:`\phi`, ``psi_theta_grid[:,:,1]`` is :math:`\theta`. 

    """
    import numpy as np 

    psi, theta = np.meshgrid(np.linspace(-np.pi, np.pi, n),
                             np.linspace(-np.pi, 0, m)) # elevation, rotation. 
    psi_theta_grid = np.dstack([psi, theta])

    return psi_theta_grid 

def img_pts_2_angles(pts2D, shape):
    r""" Given select :math:`(u,v)` image coordinates of an :math:`m\times n` pixel image compute the corresponding (azimuthal, polar),  :math:`(\phi,\theta)` coordinates with the convention that :math:`(0,\lfloor n/2 \rfloor)` maps to the South pole and :math:`(m-1,\lfloor n/2 \rfloor)` maps to the North pole of the unit sphere 

    .. math::
        \phi &= -\pi + \frac{u}{n-1}\cdot 2\pi\\ 
        \theta &= -\pi + \frac{v}{m-1} \cdot \pi\\ 

    Parameters
    ----------
    m : int
        the height of the image
    n : int
        the width of the image

    Returns 
    -------
    psi_theta_angles : (nx2) array
        the array of :math:`(\theta,\phi)` coordinates

    """
    import numpy as np 
    
    m, n = shape
    psi = pts2D[...,0]/float(n-1)*2*np.pi + (-np.pi) # transforms the x coordinate 
    theta = pts2D[...,1]/float(m-1)*np.pi + (-np.pi) # transforms the y coordinate 
    psi_theta_angles = np.vstack([psi, theta]).T 

    return psi_theta_angles 


def angles_2_img(psi_theta_angles, shape):
    r""" Given select (azimuthal, polar),  :math:`(\phi,\theta)` coordinates find the corresponding :math:`(u,v)` image coordinates of an :math:`m\times n` pixel image with the convention that :math:`(0,\lfloor n/2 \rfloor)` maps to the South pole and :math:`(m-1,\lfloor n/2 \rfloor)` maps to the North pole of the unit sphere 

    .. math::
        u &= \frac{\phi-(-\pi)}{2\pi}\cdot (n-1)\\ 
        v &= \frac{\theta-(-\pi)}{\pi}\cdot (m-1)\\ 

    Parameters
    ----------
    psi_theta_angles : array
        last dimension is 2 where ``psi_theta_grid[:,:,0]`` is :math:`\phi`, ``psi_theta_grid[:,:,1]`` is :math:`\theta`. 
    shape : (m,n) tuple
        the height, width of the image grid

    Returns 
    -------
    xy_out : array
        last dimension is 2 where ``xy_out[:,:,0]`` is :math:`u`, ``xy_out[:,:,1]`` is :math:`v`. 

    """
    import numpy as np

    m, n = shape
    x = (psi_theta_angles[...,0] - (-np.pi)) / (2*np.pi)*(n-1) #compute fraction 
    y = (psi_theta_angles[...,1] - (-np.pi)) / (np.pi)*(m-1)
    
    xy_out = np.zeros(psi_theta_angles.shape)
    xy_out[...,0] = x.copy()
    xy_out[...,1] = y.copy()

    return xy_out


def sphere_from_img_angles(psi_theta_angles):

    r""" Given select (azimuthal, polar),  :math:`(\phi,\theta)` coordinates compute the corresponding :math:`(x,y,z)` cartesian coordinates of the unit sphere.

    .. math::
        x &= \sin\theta\cos\phi\\ 
        y &= \sin\theta\sin\phi\\ 
        z &= \cos\theta

    Parameters
    ----------
    psi_theta_angles : (n,2) array
        last dimension is 2 where ``psi_theta_grid[:,:,0]`` is :math:`\phi`, ``psi_theta_grid[:,:,1]`` is :math:`\theta`. 

    Returns 
    -------
    xyz : (n,3) array
        the :math:`(x,y,z)` cartesian coordinates on the unit sphere.

    """
    import numpy as np 
    
    x = psi_theta_angles[:,0] # psi
    y = psi_theta_angles[:,1] # theta
    
    xyz = np.vstack([np.sin(y)*np.cos(x), np.sin(y)*np.sin(x), np.cos(y)]).T

    return xyz


def img_angles_from_sphere(xyz3D):

    r""" Given select :math:`(x,y,z)` cartesian coordinates of the unit sphere find the (azimuthal, polar), :math:`(\phi,\theta)` coordinates

    .. math::
        \phi &= -\cos^{-1}{z}\\ 
        \theta &= -\tan^{-1}{-\frac{y}{x}}

    with appropriate transformation so that 

    .. math::
        \phi &\in [-\pi,\pi]\\ 
        \theta &\in [-\pi,0]

    Parameters
    ----------
    xyz3D : (n,3) array
        (x,y,z) coordinates

    Returns 
    -------
    psitheta : (n,2) array
        the :math:`(\phi,\theta)` azimuthal, polar coordinates 

    """
    import numpy as np 

    theta = np.arccos(xyz3D[...,2]) * -1
    theta[theta==0] = 0
    psi = np.arctan2(xyz3D[...,1], -xyz3D[...,0]) * -1 
    psi[psi==0] = 0
    
    # restrict into the interval [-pi,0] x [-pi,pi]
    theta[theta<np.pi] = theta[theta<np.pi] + np.pi
    theta[theta>0] = theta[theta>0] - np.pi
    
    psi[psi<np.pi] = psi[psi<np.pi] + 2*np.pi
    psi[psi>np.pi] = psi[psi>np.pi] - 2*np.pi

    psitheta = np.vstack([psi,theta]).T
    
    return psitheta


# this is for rotating the volume -> not actually needed. 
def rot_pts_z(pts, angle, mean_pts=None, demean_pts=True):
    r""" Rotate 3D points around the z-axis given angle in radians 

    Parameters
    ----------
    pts : (nx3) array
        an nx3 matrix of 3D (x,y,z) coordinates
    angle : scalar
        rotation angle around the z-axis given in radians
    mean_pts : (3,) array
        a 3D vector specifying the designed centroid of affine transformation 
    demean_pts : bool
        if True, the rotation matrix is applied with respect to the given mean_pts. If mean_pts=None, it is inferred as the mean (x,y,z) coordinate of the input points.

    Returns 
    -------
    pts_new : (nx3) array
        an nx3 matrix of the rotated 3D (x,y,z) coordinates

    """
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0], 
                       [np.sin(angle), np.cos(angle),  0], 
                       [0, 0, 1]])

    if demean_pts:
        if mean_pts is not None:
            mean_pts = np.nanmean(pts, axis=0)
        pts_ = pts - mean_pts[None,:]
    else:
        pts_ = pts.copy() 

    pts_new = matrix.dot(pts.T)

    if demean_pts:
        pts_new = pts_new + mean_pts[None,:] # add back the mean. 

    pts_new = pts_new.T
    
    return pts_new

def rot_pts_y(pts, angle, mean_pts=None, demean_pts=True):
    r""" Rotate 3D points around the y-axis given angle in radians 

    Parameters
    ----------
    pts : (nx3) array
        an nx3 matrix of 3D (x,y,z) coordinates
    angle : scalar
        rotation angle around the y-axis given in radians
    mean_pts : (3,) array
        a 3D vector specifying the designed centroid of affine transformation 
    demean_pts : bool
        if True, the rotation matrix is applied with respect to the given mean_pts. If mean_pts=None, it is inferred as the mean (x,y,z) coordinate of the input points.

    Returns 
    -------
    pts_new : (nx3) array
        an nx3 matrix of the rotated 3D (x,y,z) coordinates

    """
    matrix = np.array([[np.cos(angle), 0, np.sin(angle)], 
                       [0, 1,  0], 
                       [-np.sin(angle), 0, np.cos(angle)]])

    if demean_pts:
        if mean_pts is not None:
            mean_pts = np.nanmean(pts, axis=0)
        pts_ = pts - mean_pts[None,:]
    else:
        pts_ = pts.copy() 

    pts_new = matrix.dot(pts.T)

    if demean_pts:
        pts_new = pts_new + mean_pts[None,:] # add back the mean. 
    
    pts_new = pts_new.T
    
    return pts_new

def rot_pts_x(pts, angle, mean_pts=None, demean_pts=True):
    r""" Rotate 3D points around the x-axis given angle in radians 

    Parameters
    ----------
    pts : (nx3) array
        an nx3 matrix of 3D (x,y,z) coordinates
    angle : scalar
        rotation angle around the x-axis given in radians
    mean_pts : (3,) array
        a 3D vector specifying the designed centroid of affine transformation 
    demean_pts : bool
        if True, the rotation matrix is applied with respect to the given mean_pts. If mean_pts=None, it is inferred as the mean (x,y,z) coordinate of the input points.

    Returns 
    -------
    pts_new : (nx3) array
        an nx3 matrix of the rotated 3D (x,y,z) coordinates

    """
    matrix = np.array([[1, 0, 0], 
                       [0, np.cos(angle), -np.sin(angle)], 
                       [0, np.sin(angle), np.cos(angle)]])
    
    if demean_pts:
        if mean_pts is not None:
            mean_pts = np.nanmean(pts, axis=0)
        pts_ = pts - mean_pts[None,:]
    else:
        pts_ = pts.copy() 

    pts_new = matrix.dot(pts.T)

    if demean_pts:
        pts_new = pts_new + mean_pts[None,:] # add back the mean. 

    pts_new = pts_new.T
    
    return pts_new


def get_rotation_x(theta):
    r""" Construct the homogeneous 4x4 rotation matrix to rotate :math:`\theta` radians around the x-axis

    .. math::
        R_x(\theta) = \begin{bmatrix}
                1 & 0 & 0 & 0\\
                0 & \cos\theta & -\sin\theta & 0 \\
                0 & \sin\theta & \cos\theta & 0 \\
                0 & 0 & 0 & 1 
            \end{bmatrix}

    Parameters
    ----------
    theta : scalar
        rotatiion angle in radians
    
    Returns 
    -------
    R_x : (4x4) array
        the 4x4 homogeneous rotation matrix around the x-axis

    """
    R_x = np.zeros((4,4))
    R_x[-1:] = np.array([0,0,0,1])
    
    R_x[:-1,:-1] = np.array([[1,0,0],
                   [0, np.cos(theta), -np.sin(theta)], 
                   [0, np.sin(theta), np.cos(theta)]])
    
    return R_x
    
def get_rotation_y(theta):
    r""" Construct the homogeneous 4x4 rotation matrix to rotate :math:`\theta` radians around the y-axis

    .. math::
        R_y(\theta) = \begin{bmatrix}
                \cos\theta & 0 & \sin\theta & 0\\
                0 & 1 & 0 & 0 \\
                -\sin\theta & 0 & \cos\theta & 0 \\
                0 & 0 & 0 & 1 
            \end{bmatrix}

    Parameters
    ----------
    theta : scalar
        rotatiion angle in radians
    
    Returns 
    -------
    R_y : (4x4) array
        the 4x4 homogeneous rotation matrix around the y-axis

    """
    R_y = np.zeros((4,4))
    R_y[-1:] = np.array([0,0,0,1])
    
    R_y[:-1,:-1] = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0], 
                   [-np.sin(theta), 0, np.cos(theta)]])
    
    return R_y
    
def get_rotation_z(theta):
    r""" Construct the homogeneous 4x4 rotation matrix to rotate :math:`\theta` radians around the z-axis

    .. math::
        R_z(\theta) = \begin{bmatrix}
                \cos\theta & -\sin\theta & 0 & 0\\
                \sin\theta & \cos\theta & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1 
            \end{bmatrix}

    Parameters
    ----------
    theta : scalar
        rotation angle in radians
    
    Returns 
    -------
    R_z : (4x4) array
        the 4x4 homogeneous rotation matrix around the z-axis

    """
    R_z = np.zeros((4,4))
    R_z[-1:] = np.array([0,0,0,1])
    
    R_z[:-1,:-1] = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0], 
                   [0, 0, 1]])
    
    return R_z


def rotation_matrix_xyz(angle_x,angle_y, angle_z, center=None, imshape=None):
    r""" Construct the composite homogeneous 4x4 rotation matrix to rotate :math:`\theta_x` radians around the x-axis, :math:`\theta_y` radians around the y-axis, 
    :math:`\theta_z` radians around the z-axis, with optional specification of rotation center. One of ``center`` or ``imshape`` must be specified to fix the center of rotation.

    The individual rotations are composited functionally in the order of x- then y- then z- rotations. 

    .. math::
        R(\theta_x,\theta_y,\theta_z) = R_z(R_y(R_x)))
    
    We then add the translation to put the output coordinates to the specified center :math:`T=(\bar{x},\bar{y},\bar{z})` after rotation. If ``center`` not specified then :math:`T=(0,0,0)`.
    
    .. math::
        A = \begin{pmatrix}
                R &\bigm \| & T \\
                \textbf{0} &\bigm \|& 1 
            \end{pmatrix}

    For rotating a volumetric image we must first decenter before applying :math:`A`. The decentering translation matrix is 
    
    .. math::
        A' = \begin{pmatrix}
                \textbf{I} &\bigm \| & T' \\
                \textbf{0} &\bigm \|& 1 
            \end{pmatrix}

    where :math:`T'=(-\bar{x}',-\bar{y}',-\bar{z}')` and  :math:`(\bar{x}',\bar{y}',\bar{z})` is either ``center`` or ``imshape``/2.

    The final one-step transformation matrix is given by left matrix multiplication and this is what this function returns

    .. math::
        T_{final} = AA'

    Parameters
    ----------
    angle_x : scalar
        x-axis rotation angle in radians
    angle_x : scalar
        x-axis rotation angle in radians
    angle_x : scalar
        x-axis rotation angle in radians
    center : (x,y,z) array
        centroid of the final rotated volume
    imshape : (m,n,l) array
        size of the volume transform will be applied to.  

    Returns 
    -------
    T : (4x4) array
        the composite 4x4 homogeneous rotation matrix

    """
    import numpy as np 
    affine_x = get_rotation_x(angle_x)
    affine_y = get_rotation_y(angle_y) # the top has coordinates (r, np.pi/2. 0)
    affine_z = get_rotation_z(angle_z)
    affine = affine_z.dot(affine_x.dot(affine_y))
    
    if center is not None:
        affine[:-1,-1] = center 
    if imshape is not None:
        decenter = np.eye(4); decenter[:-1,-1] = [-imshape[0]//2, -imshape[1]//2, -imshape[2]//2]
    else:
        decenter = np.eye(4); decenter[:-1,-1] = -center
    T = affine.dot(decenter)
    
    return T


def shuffle_Tmatrix_axis_3D(Tmatrix, new_axis):
    r""" Utility function to shift a 4x4 transformation matrix to conform to transposition of [x,y,z] coordinates. This is useful for example when converting between Python and Matlab where Matlab adopts a different convention in the specification of their transformation matrix. 

    Parameters
    ----------
    Tmatrix : 4x4 array
        4x4 homogeneous transformation matrix 
    new_axis : array-like
        an array specifying the permutation of the (x,y,z) coordinate axis e.g. [1,0,2], [2,0,1]
    
    Returns 
    -------
    Tmatrix_new : 4x4 array
        the corresponding 4x4 homogeneous transformation matrix matching the permutation of (x,y,z) specified by `new_axis`

    """
    Tmatrix_new = Tmatrix[:3].copy()
    Tmatrix_new = Tmatrix_new[new_axis,:] # flip rows first. (to flip the translation.)
    Tmatrix_new[:,:3] = Tmatrix_new[:,:3][:,new_axis] #flip columns (ignoring translations)
    Tmatrix_new = np.vstack([Tmatrix_new, [0,0,0,1]]) # make it homogeneous 4x4 transformation. 
    
    return Tmatrix_new


def rotate_vol(vol, angle, centroid, axis, check_bounds=True):
    
    r""" Utility function to rotate a volume image around a given centroid, around a given axis specified by one of 'x', 'y', 'z' with the angle specified in degrees. If check_bounds is True, the output volume size will be appropriately determined to ensure the content is not clipped else it will return the rotated volume with the same size as the input 
    
    The rotation currently uses Dipy library.

    Parameters
    ----------
    vol : (MxNxL) array
        input volume image
    angle : scalar
        rotation angle specified in degrees
    centroid : (x,y,z) array
        centroid of rotation specified as a vector 
    axis : str
        one of x-, y-, z- independent axis of rotation 
        
        'x' : 
            x-axis rotation 
        'y' :
            y-axis rotation 
        'z' :
            z-axis rotation 
    check_bounds : bool
        if True, the output volume will be resized to fit image contents
    
    Returns 
    -------
    vol_out : (MxNxL) array
        the rotated volume image 

    """
    rads = angle/180.*np.pi
    if axis == 'x':
        rot_matrix = get_rotation_x(rads)
    if axis == 'y':
        rot_matrix = get_rotation_y(rads)
    if axis == 'z':
        rot_matrix = get_rotation_z(rads)
    
    im_center = np.array(vol.shape)//2
    rot_matrix[:-1,-1] = np.array(centroid)
    decenter = np.eye(4); decenter[:-1,-1] = -np.array(im_center)
    
    T = rot_matrix.dot(decenter)
    # print(T)

    if check_bounds:
        vol_out = apply_affine_tform(vol, T,
                                        sampling_grid_shape=None,
                                        check_bounds=True,
                                        contain_all=True,
                                        codomain_grid_shape=None,
                                        domain_grid2world=None,
                                        codomain_grid2world=None,
                                        sampling_grid2world=decenter)
    else:
        vol_out = apply_affine_tform(vol, T,
                                        sampling_grid_shape=vol.shape)
        
    return vol_out
    


def rotate_volume_xyz(vol, vol_center, angle_x, angle_y, angle_z, use_im_center=True, out_shape=None):

    r""" Utility function to rotate a volume image around a given centroid, and different angles specified in degrees around x-, y-, z- axis. This function doesn't check bounds but allows a specification of the final shape. 
    
    The rotation currently uses Dipy library. The rotation will be applied in the order of x-, y-, then z-

    Parameters
    ----------
    vol : (MxNxL) array
        input volume image
    vol_center : (x,y,z) array
        centroid of rotation specified as a vector 
    angle_x : scalar
        x-axis rotation in degrees
    angle_y : scalar
        y-axis rotation in degrees
    angle_z : scalar
        z-axis rotation in degrees
    use_im_center : bool
        if True, rotate about the image center instead of the specified vol_center
    out_shape : (m,n,l) tuple
        shape of the final output volume which can be different from vol.shape

    Returns 
    -------
    rot_mat : 4x4 array
        the rotation matrix applied 
    vol_out : (MxNxL) array
        the final rotated volume 

    """
    import numpy as np 
    
    angle_x = angle_x/180. * np.pi
    angle_y = angle_y/180. * np.pi
    angle_z = angle_z/180. * np.pi
       
    # construct the correction matrix and transform the image. 
    imshape = vol.shape
    if use_im_center:
        rot_mat = rotation_matrix_xyz(-angle_x, -angle_y, -angle_z, vol_center, imshape)
    else:
        rot_mat = rotation_matrix_xyz(-angle_x, -angle_y, -angle_z, vol_center) # negative angle is to match the pullback nature of getting intensities. 
        
    if out_shape is None:
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=imshape)
    else:
        vol_out = apply_affine_tform(vol, rot_mat, sampling_grid_shape=out_shape)
        
    return rot_mat, vol_out


def apply_affine_tform(volume, matrix, sampling_grid_shape=None, check_bounds=False, contain_all=False, domain_grid_shape=None, codomain_grid_shape=None, domain_grid2world=None, codomain_grid2world=None, sampling_grid2world=None):
    
    r""" Given a homogeneous transformation matrix, creates an affine transform matrix and use Dipy library to apply the transformation.
    
    see https://dipy.org/documentation/1.1.1./reference/dipy.align/ for more info on parameters.

    Parameters
    ----------
    volume : (MxNxL) array
        input volume image
    matrix : 4x4 array
        the homogeneous transformation matrix 
    sampling_grid_shape : (m,n,l) tuple
        the shape of the input volume
    check_bounds : bool
        if true check if the final shape needs to be modified to avoid clipping the pixels from the input
    contain_all : bool
        if true the final shape is specified avoid clipping of any of the pixels from the input
    domain_grid_shape : bool
        if True, rotate about the image center instead of the specified vol_center
    codomain_grid_shape : (m,n,l) tuple
        the shape of the default co-domain sampling grid. When transform_inverse is called to transform an image, the resulting image will have this shape, unless a different sampling information is provided. If None (the default), then the sampling grid shape must be specified each time the transform_inverse method is called.
    domain_grid2world : 4x4 array
        the grid-to-world transform associated with the domain grid. If None (the default), then the grid-to-world transform is assumed to be the identity.
    codomain_grid2world : 4x4 array
        the grid-to-world transform associated with the co-domain grid. If None (the default), then the grid-to-world transform is assumed to be the identity.
    sampling_grid2world : 4x4 array
        the grid-to-world transform associated with the sampling grid (specified by sampling_grid_shape, or by default self.codomain_shape). If None (the default), then the grid-to-world transform is assumed to be the identity.
    
    Returns 
    -------
    out : (MxNxL) array
        the final rotated volume 

    Notes
    -----
    The various domain, codomain etc. are mostly relevant in MRI applications where there is discrepancy between world coordinates and image coordinates. For all practical purposes, user can just pass ``volume``, ``matrix``, ``sampling_shape``, ``check_bounds`` and ``contain_all``.

    """    
    import numpy as np 
    from dipy.align.imaffine import AffineMap
    
    if domain_grid_shape is None:
        domain_grid_shape = volume.shape
    if codomain_grid_shape is None:
        codomain_grid_shape = volume.shape
        
    if check_bounds:
        if contain_all:
            in_out_corners, out_shape, tilt_tf_ = compute_transform_bounds(domain_grid_shape, matrix, contain_all=True)
        else:
            in_out_corners, out_shape = compute_transform_bounds(domain_grid_shape, matrix, contain_all=False)
            tilt_tf_ = None
#        print out_shape
    affine_map = AffineMap(matrix,
                               domain_grid_shape=domain_grid_shape, domain_grid2world=domain_grid2world,
                               codomain_grid_shape=codomain_grid_shape, codomain_grid2world=codomain_grid2world)
        
    if check_bounds:
        out = affine_map.transform(volume, sampling_grid_shape=out_shape, sampling_grid2world=tilt_tf_)
    else:
        out = affine_map.transform(volume, sampling_grid_shape=sampling_grid_shape, sampling_grid2world=sampling_grid2world)
        
    return out


def compute_transform_bounds(im_shape, tf, contain_all=True):
    r""" This helper function is used by :func:`MicroscopySuite.Geometry.geometry.apply_affine_tform` to determine the minimum output size of the final volume image to avoid clipping pixels in the original after the application of a given transformation matrix 
    
    Parameters
    ----------
    im_shape : (M,N,L) array
        shape of the volume image
    tf : 4x4 array
        the homogeneous transformation matrix 
    contain_all : bool
        specifies whether all of the original input volume pixels must be captured.
    
    Returns 
    -------
    in_out_corners : 8x3 array
        the 8 corner points of the box containing the transformed input shape under tf  
    out_shape
        the new size of the image. 

    tf_mod : 4x4 array
        only returned if contain_all = True. This is a hack translation shift correction to make dipy play nice when the transformed coordinates is negative but we want to keep the full input volume in the final output.  

    """
    def general_cartesian_prod(objs):
    
        import itertools
        import numpy as np 
        
        out = []
        for element in itertools.product(*objs):
            out.append(np.array(element))
            
        out = np.vstack(out)
        
        return out

    import numpy as np 
    obj = [[0, ii-1] for ii in list(im_shape)] # remember to be -1 
    in_box_corners = general_cartesian_prod(obj)
        
    # unsure whether its left or right multiplication ! - i believe this should be the left multiplcation i,e, pts x tf. 
    out_box_corners = tf.dot(np.vstack([in_box_corners.T, np.ones(in_box_corners.shape[0])[None,:]]))[:-1].T
  
    in_out_corners = (in_box_corners, out_box_corners)
    
    if contain_all:
        out_shape = np.max(out_box_corners, axis=0) - np.min(out_box_corners, axis=0)
        out_shape = (np.rint(out_shape)).astype(np.int32)
        
        # to shift the thing, we need to change the whole world grid!. 
        mod_tf = np.min(out_box_corners, axis=0)  # what is the required transformation parameters to get the shape in line? # double check? in terms of point clouds?
        tf_mod = np.eye(4)
        tf_mod[:-1,-1] = mod_tf 
        
        # this adjustment needs to be made in a left handed manner!. 
        # here we probably need to create an offset matrix then multiply this onto the tf_ which is a more complex case.... of transformation? # to avoid sampling issues. 
        return in_out_corners, out_shape, tf_mod
        
    else:
        out_shape = np.max(out_box_corners,axis=0) # this should touch the edges now right? #- np.min(out_box_corners,axis=0) # not quite sure what dipy tries to do ?
        out_shape = (np.rint(out_shape)).astype(np.int32)
    
        return in_out_corners, out_shape


def xyz_2_spherical(x,y,z, center=None):
    r""" General Cartesian, :math:`(x,y,z)` to Spherical, :math:`(r,\theta,\phi)` coordinate transformation where :math:`\theta` is the polar or elevation angle and :math:`\phi` the aziumthal or circular rotation angle.

    .. math::
        r &= \sqrt{\hat{x}^2+\hat{y}^2+\hat{z}^2}\\ 
        \theta &= \cos^{-1}\frac{\hat{z}}{r} \\
        \phi &= \tan^{-1}\frac{\hat{y}}{\hat{x}}

    where :math:`\hat{x}=x-\bar{x}`, :math:`\hat{y}=y-\bar{y}`, :math:`\hat{z}=z-\bar{z}` are the demeaned :math:`(x,y,z)` coordinates.
    
    Parameters
    ----------
    x : array
        x-coordinate
    y : array
        y-coordinate
    z : array
        z-coordinate
    center : (x,y,z) vector
        the centroid coordinate, :math:`(\bar{x}, \bar{y}, \bar{z})`, if center=None, the mean of the input :math:`(x, y, z)` is computed

    Returns 
    -------
    r : array
        radial distance
    polar : array
        polar angle, :math:`\theta`
    azimuthal : array
        azimuthal angle, :math:`\phi`

    """
    if center is None:
        center = np.hstack([np.mean(x), np.mean(y), np.mean(z)])
        
    x_ = x - center[0]
    y_ = y - center[1]
    z_ = z - center[2]
    
    r = np.sqrt(x_**2+y_**2+z_**2)
    polar = np.arccos(z_/r) # normal circular rotation (longitude)
    azimuthal = np.arctan2(y_,x_) # elevation (latitude)
    
    return r, polar, azimuthal
    
def spherical_2_xyz(r, polar, azimuthal, center=None):
    r""" General Cartesian, :math:`(x,y,z)` to Spherical, :math:`(r,\theta,\phi)` coordinate transformation where :math:`\theta` is the polar or elevation angle and :math:`\phi` the aziumthal or circular rotation angle.

    .. math::
        x &= r\sin\theta\cos\phi + \bar{x}\\ 
        y &= r\sin\theta\sin\phi + \bar{y}\\ 
        z &= r\cos\theta + \bar{z}

    where :math:`(\bar{x}, \bar{y}, \bar{z})` is the provided mean of :math:`(x,y,z)` coordinates. If ``center`` is not specified the origin :math:`(0, 0, 0)` is used. 
    
    Parameters
    ----------
    r : array
        radial distance
    polar : array
        polar angle, :math:`\theta`
    azimuthal : array
        azimuthal angle, :math:`\phi`
    center : (x,y,z) vector
        the centroid coordinate, :math:`(\bar{x}, \bar{y}, \bar{z})`, if center=None, the centroid is taken to be the origin :math:`(0, 0, 0)`

    Returns 
    -------
    x : array
        x-coordinate
    y : array
        y-coordinate
    z : array
        z-coordinate
    """
    
    if center is None:
        center = np.hstack([0, 0, 0])
        
    x = r*np.sin(polar)*np.cos(azimuthal)
    y = r*np.sin(polar)*np.sin(azimuthal)
    z = r*np.sin(polar)
        
    x = x + center[0]
    y = y + center[1]
    z = z + center[2]
    
    return x,y,z
    
def xyz_2_longlat(x,y,z, center=None):
    r""" General Cartesian, :math:`(x,y,z)` to geographical longitude latitude, :math:`(r,\phi,\lambda)` coordinate transformation used in cartographic projections

    .. math::
        r &= \sqrt{\hat{x}^2+\hat{y}^2+\hat{z}^2}\\ 
        \phi &= \sin^{-1}\frac{\hat{z}}{r} \\
        \lambda &= \tan^{-1}\frac{\hat{y}}{\hat{x}}

    where :math:`\hat{x}=x-\bar{x}`, :math:`\hat{y}=y-\bar{y}`, :math:`\hat{z}=z-\bar{z}` are the demeaned :math:`(x,y,z)` coordinates.
    
    Parameters
    ----------
    x : array
        x-coordinate
    y : array
        y-coordinate
    z : array
        z-coordinate
    center : (x,y,z) vector
        the centroid coordinate, :math:`(\bar{x}, \bar{y}, \bar{z})`, if center=None, the mean of the input :math:`(x, y, z)` is computed

    Returns 
    -------
    r : array
        radial distance
    latitude : array
        latitude angle, :math:`\lambda`
    longitude : array
        longitude angle, :math:`\phi`

    """
    if center is None:
        center = np.hstack([np.mean(x), np.mean(y), np.mean(z)])
        
    x_ = x - center[0]
    y_ = y - center[1]
    z_ = z - center[2]
    
    r = np.sqrt(x_**2+y_**2+z_**2)
    latitude = np.arcsin(z_/r) # normal circular rotation (longitude)
    longitude = np.arctan2(y_,x_) # elevation (latitude)
    
    return r, latitude, longitude

def latlong_2_spherical(r,lat,long):
    r""" Conversion from geographical longitude latitude, :math:`(r,\lambda,\phi)` to Spherical, :math:`(r,\theta,\phi)` coordinates

    .. math::
        r &= r\\ 
        \theta &= \phi + \pi/2\\
        \phi &= \lambda

    see:
        https://vvvv.org/blog/polar-spherical-and-geographic-coordinates
    
    Parameters
    ----------
    r : array
        radial distance
    long : array
        longitude angle, :math:`\lambda`
    lat : array
        latitude angle, :math:`\phi`

    Returns 
    -------
    r : array
        radial distance
    polar : array
        polar angle, :math:`\theta`
    azimuthal : array
        azimuthal angle, :math:`\phi`

    """
    polar = lat + np.pi/2.
    azimuthal = long
    
    return r, polar, azimuthal
    
def spherical_2_latlong(r, polar, azimuthal):
    r""" Conversion from Spherical, :math:`(r,\theta,\phi)` to geographical longitude latitude, :math:`(r,\lambda,\phi)` coordinates

    .. math::
        r &= r\\ 
        \phi &= \theta - \pi/2\\
        \lambda &= \phi

    see:
        https://vvvv.org/blog/polar-spherical-and-geographic-coordinates
    
    Parameters
    ----------
    r : array
        radial distance
    polar : array
        polar angle, :math:`\theta`
    azimuthal : array
        azimuthal angle, :math:`\phi`


    Returns 
    -------
    r : array
        radial distance
    lat : array
        latitude angle, :math:`\lambda`
    long : array
        longitude angle, :math:`\phi`

    """
    lat = polar - np.pi/2.
    long = azimuthal
    
    return r, lat, long
    
    
def map_gnomonic(r, lon, lat, lon0=0, lat0=0):
    r""" Gnomonic map projection of geographical longitude latitude coordinates to :math:`(x,y)` image coordinates

    The gnomonic projection is a nonconformal map projection obtained by projecting points :math:`P_1` on the surface of a sphere from a sphere's center O to a point :math:`P` in a plane that is tangent to another point :math:`S` on the sphere surface. In a gnomonic projection, great circles are mapped to straight lines. The gnomonic projection represents the image formed by a spherical lens, and is sometimes known as the rectilinear projection. 
    
    The transformation equations for the plane tangent at the point :math:`S` having latitude phi and longitude lambda for a projection with central longitude :math:`\lambda_0` and central latitude :math:`\phi_0` are given by 
    
    .. math::
        x &= r\cdot\frac{\cos\phi\sin(\lambda-\lambda_0)}{\cos c}\\ 
        y &= r\cdot\frac{\cos\phi_0\sin\theta - \sin\phi_0\cos\phi\cos(\lambda-\lambda_0)}{\cos c}

    where :math:`c` is the angular distance of the point :math:`(x,y)` from the center of the projection given by

    .. math::
        \cos c= \sin\phi_0\sin_\phi + \cos\phi_0\cos\phi\cos(\lambda-\lambda_0)

    see :
        https://mathworld.wolfram.com/GnomonicProjection.html
    
    Parameters
    ----------
    r : array
        radial distance
    lon : array
        longitude angle, :math:`\lambda`
    lat : array
        latitude angle, :math:`\phi`
    lon0 : scalar
        central longitude
    lat0 : scalar
        central latitude


    Returns 
    -------
    x : array
        x-coordinate in the plane
    y : array
        y-coordinate in the plane
    """

    cos_c = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0)
    x = r*np.cos(lat)*np.sin(lon-lon0) / cos_c
    y = r*(np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0))/ cos_c
    
    return x, y
    
def azimuthal_ortho_proj3D(latlong, pole='pos'):
    r"""Orthographic azimuthal or perspective projection of sphere onto a tangent plane. This is just the straight line projection in spherical coordinates to the plane and strictly projects a hemisphere. Clipping must be applied to prevent overlap specified by the `pole` parameter.
    
    .. math::
        x &= r\cos\phi\cos\lambda\\ 
        y &= r\cos\phi\sin\lambda

    see :
        https://en.wikipedia.org/wiki/Orthographic_map_projection
    
    Parameters
    ----------
    latlong : list of arrays
        list of `[r, lat, long]` coordinates 
    pole : str
        if 'pos', the hemisphere specified by `lat <= 0` is projected else the hemisphere with `lat >= 0` is mapped

    Returns 
    -------
    x_select : array
        the x-coordinate of input points from the chosen hemisphere
    y_select : array
        the y-coordinate of input points from the chosen hemisphere
    select : array
        binary mask specifying which of the input points were mapped to the plane 

    """
    r, lat, long = latlong
    
    x = r*np.cos(lat)*np.cos(long)
    y = r*np.cos(lat)*np.sin(long) 
    
    # determine clipping points
    if pole == 'pos':
        select = lat <= 0
    else:
        select = lat >= 0 

    x_select = x[select]
    y_select = y[select]

    return x_select, y_select, select 


def azimuthal_equidistant(latlong, lat0=0, lon0=0):
    r""" The azimuthal equidistance projection preserves radial distance emanating from the central point of the projection. It is neither equal-area nor conformal. 

    Let :math:`\phi_0` and :math:`\lambda_0` be the latitude and longitude of the center of the projection, then the transformation equations are given by

    .. math::
        x &= k'\cos\phi\sin(\lambda-\lambda_0)\\
        y &= k'[\cos\phi_0\sin\phi-\sin\phi_0\cos\phi\cos(\lambda-\lambda_0)]

    where,

    .. math::
        k' = \frac{c}{\sin c}

    and :math:`c` is the angular distance of the point :math:`(x,y)` from the center of the projection given by 

    .. math::
        \cos c= \sin\phi_0\sin_\phi + \cos\phi_0\cos\phi\cos(\lambda-\lambda_0)

    see :
        https://mathworld.wolfram.com/AzimuthalEquidistantProjection.html

    Parameters
    ----------
    latlong : list of arrays
        list of `[r, lat, long]` coordinates 
    lat0 : scalar
        central latitude
    lon0 : scalar
        central longitude

    Returns 
    -------
    x : array
        x-coordinate in the plane
    y : array
        y-coordinate in the plane
    """
    r, lat, lon = latlong

    cos_c = np.sin(lat1)*np.sin(lat) + np.cos(lat1)*np.cos(lat)*np.cos(lon-lon0)
    c = np.arccos(cos_c)
    k_ = c/np.sin(c)
    
    x_p = k_ * np.cos(lat) * np.sin(lon-lon0)
    y_p = k_ * (np.cos(lat1)*np.sin(lat) - np.sin(lat1)*np.cos(lat)*np.cos(lon-lon0))

    x = r*x_p
    y = r*y_p 

    return x, y



def stereographic(u):
    r""" Function which both stereographic projection of 3D unit sphere points to the 2D plane and from the 2D plane back to the sphere depending on the dimensionality of the input.  

    Mathematically the transformations are

    From 2D plane to 3D sphere

    .. math::
        (u,v) \rightarrow \left( \frac{2u}{1+u^2+v^2}, \frac{2v}{1+u^2+v^2}, \frac{-1+u^2+v^2}{1+u^2+v^2}\right)

    From 3D sphere to 2D plane

    .. math::
        (x,y,z) \rightarrow \left( \frac{x}{1-z}, \frac{y}{1-z}\right)

    Parameters
    ----------
    u : (N,) complex array or (N,2) array for plane or (N,3) for sphere points
        the plane or sphere points. If coordinate dimensions is 3, assumes the 3D sphere points and the return will be the 2D plane points after stereographic projection 

    Returns
    -------
    v : (N,2) array of plane or (N,3) array of sphere points
        

    """

    # % STEREOGRAPHIC  Stereographic projection.
    # %   v = STEREOGRAPHIC(u), for N-by-2 matrix, projects points in plane to sphere
    # %                       ; for N-by-3 matrix, projects points on sphere to plane
    import numpy as np 
    if u.shape[-1] == 1:
        u = np.vstack([np.real(u), np.imag(u)])
    x = u[:,0].copy()
    y = u[:,1].copy()

    if u.shape[-1] < 3: 
        z = 1 + x**2 + y**2;
        v = np.vstack([2*x / z, 2*y / z, (-1 + x**2 + y**2) / z]).T;
    else:
        z = u[:,2].copy()
        v = np.vstack([x/(1.-z), y/(1.-z)]).T

    return v
        
    
def fix_improper_rot_matrix_3D(rot_matrix):
    r""" All rotational matrices are orthonormal, however only those with determinant = +1 constitute pure rotation matrices. Matrices with determinant = -1 involve a flip. This function is used to test a given 3x3 rotation matrix and if determinant = -1 make it determinant = + 1 by appropriately flipping the columns. 

    Parameters
    ----------
    rot_matrix : 3x3 array
        input rotation matrix of determinant = -1 or +1

    Returns 
    -------
    rot_matrix_fix : 3x3 matrix
        a proper rotation matrix with determinant = +1

    Notes
    -----
    Without loss of generality, we can turn an improper 3x3 rotation matrix into its proper equivalent by comparing sign with the +x, +y, +z axis. 

    Given a 3x3 rotation matrix, the function searches the 1st column vector and checks the x-component is positive i.e. in the same direction as the +x axis, if not it flips all the first column components by multiplying by -1. It then checks the second column component which should be alignment with the +y axis. The third vector by definition must be orthogonal to the first two therefore it does not need to be checked and is directly found by the cross-product.  
    
    """
    import numpy as np 
    
    if np.sign(np.linalg.det(rot_matrix)) < 0:
        rot_matrix_fix = rot_matrix.copy()
        for ii in np.arange(rot_matrix.shape[1]-1):
            if np.sign(rot_matrix[ii,ii]) >= 0:
                # nothing 
                pass
            else:
                rot_matrix_fix[:,ii] = -1*rot_matrix[:,ii]
        last = np.cross(rot_matrix_fix[:,0], rot_matrix_fix[:,1])
        last = last / (np.linalg.norm(last) + 1e-12)
        rot_matrix_fix[:,-1] = last.copy()
        return rot_matrix_fix        
    else:
        rot_matrix_fix = rot_matrix.copy()
        return rot_matrix_fix
        
    
def resample_vol_intensity_rotation(img, rot_matrix_fwd, 
                                    mean_ref=None, 
                                    mean_ref_out=None, 
                                    pts_ref=None, 
                                    pad=None, 
                                    out_shape=None,
                                    max_lims_out=None,
                                    min_lims_out=None):

    r""" Utility function to rotate a volume image given a general affine transformation matrix. Options are provided to specify the center of rotation of the input and volume center in the output which may be differ to prevent clipping.
    Unlike rotate_vol, this function does not require Dipy and is more flexible to tune. 

    Parameters
    ----------
    img : (MxNxL) array
        input volume image
    rot_matrix_fwd : 4x4 array
        the desired transformation matrix
    mean_ref : (x,y,z) array
        if specified, the centroid of rotation specified as a vector, to rotate the initial volume
    mean_ref_out : (x,y,z) array
        if specified, the centroid of rotation specified as a vector, to center the final output volume 
    pad : int 
        if pad is not None, this function will then compute the exact bounds of rotation to enable padding. If pad=None a volume the same size as input is returned. 
    out_shape : (M,N,L) tuple
        specifies the desired output shape of the rotated volume, this will only be taken into account if pad is not None. 
    max_lims_out : (M,N,L) tuple
        the maximum coordinate in each axis in the final rotated volume, if not specified it will be auto-inferred as the minimum bounding box
    min_lims_out : (M,N,L) tuple
        the minimum coordinate in each axis in the final rotated volume, if not specified it will be auto-inferred as the minimum bounding box
    
    Returns 
    -------
    img_pullback : (MxNxL) array
        the final transformed volume image
    mean_pts_img : (x,y,z) array
        the centroid of the final transformed volume image
    (min_lims, max_lims) : (3-tuple, 3-tuple)
        returned only if pad is not None. These specify the real minimum and maximum of the coordinate bounding box when the forward transform is applied. The final volume is of shape equal to difference in these if the `out_shape` was not specified by the user. 

    """
    def map_intensity_interp3(query_pts, grid_shape, I_ref, method='linear', cast_uint8=False):
    
        # interpolate instead of discretising to avoid artifact.
        from scipy.interpolate import RegularGridInterpolator
        
        #ZZ,XX,YY = np.indices(im_array.shape)
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

    
    import numpy as np 

    if pts_ref is None:
        XX, YY, ZZ = np.indices(img.shape) # get the full coordinates. 
        pts_ref = np.vstack([XX.ravel(), 
                             YY.ravel(),
                             ZZ.ravel()]).T
        
    if mean_ref is None:
        mean_ref = np.nanmean(pts_ref,axis=0)
    
    
    # # compute the forward transform to determine the bounds. 
    # forward_pts_tform = rot_matrix_fwd.dot((pts_ref-mean_ref[None,:]).T).T + mean_ref[None,:]
    
    if pad is None:
        # then we use the same volume size as the img. 
        rot_matrix_pulldown = np.linalg.inv(rot_matrix_fwd)
        
        XX, YY, ZZ = np.indices(img.shape)
        pts_img_orig = np.vstack([XX.ravel(), 
                                 YY.ravel(),
                                 ZZ.ravel()]).T
        if mean_ref_out is None:
            mean_pts_img = np.nanmean(pts_img_orig, axis=0)
        else:
            mean_pts_img = mean_ref_out.copy()
        
        pts_img_orig = rot_matrix_pulldown.dot((pts_img_orig-mean_pts_img[None,:]).T).T + mean_ref[None,:] # demean with respect to own coordinates, but mean_ref... 
        
        img_pullback = map_intensity_interp3(pts_img_orig, 
                                             img.shape, 
                                             I_ref=img, method='linear', 
                                             cast_uint8=False)
        img_pullback = img_pullback.reshape(img.shape) # reshape to the correct shape. 
        # interpolate the original image.! 
        return img_pullback, mean_pts_img
    else:
        
        rot_matrix_pulldown = np.linalg.inv(rot_matrix_fwd)
        
        # compute the forward transform to determine the bounds. 
        forward_pts_tform = rot_matrix_fwd.dot((pts_ref-mean_ref[None,:]).T).T + mean_ref[None,:]
        
        if min_lims_out is None:
            min_lims = np.nanmin(forward_pts_tform, axis=0)
        else:
            min_lims = min_lims_out.copy()
            
        if max_lims_out is None:
            max_lims = np.nanmax(forward_pts_tform, axis=0)
        else:
            max_lims = max_lims_out.copy()
            
        # the range of the max-min gives the size of the volume + the padsize. (symmetric padding.)
        # points then linearly establish correspondence. 
        # this vectorial represntation allows asymmetric handling. 
        print(min_lims, max_lims)
        min_lims = min_lims.astype(np.int32) - np.hstack(pad)
        max_lims = (np.ceil(max_lims)).astype(np.int32) + np.hstack(pad)
        
        # we need to create new cartesian combination of datapoints. 
        if out_shape is None:
            l, m, n = max_lims - min_lims 
        else:
            l, m, n = out_shape
        ll, mm, nn = np.indices((l,m,n))
        
        # map this to 0-1 scale and reverse (xx - a) / (a-b) for general (a,b) intervals.
        XX = ll/float(l) * (max_lims[0] - min_lims[0]) + min_lims[0]
        YY = mm/float(m) * (max_lims[1] - min_lims[1]) + min_lims[1]
        ZZ = nn/float(n) * (max_lims[2] - min_lims[2]) + min_lims[2]
        
        pts_img_orig = np.vstack([XX.ravel(), 
                                  YY.ravel(),
                                  ZZ.ravel()]).T
        
        if mean_ref_out is None:
            mean_pts_img = np.nanmean(forward_pts_tform, axis=0)
        else:
            mean_pts_img = mean_ref_out.copy()
            
        pts_img_orig = rot_matrix_pulldown.dot((pts_img_orig-mean_pts_img[None,:]).T).T + mean_ref[None,:] # demean with respect to own coordinates, but mean_ref... 
        
        img_pullback = map_intensity_interp3(pts_img_orig, 
                                             img.shape, 
                                             I_ref=img, method='linear', 
                                             cast_uint8=False)
        img_pullback = img_pullback.reshape((l,m,n)) # reshape to the correct shape. 
        
        return img_pullback, mean_pts_img, (min_lims, max_lims)
        

def minimum_bounding_rectangle(points):
    r"""
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    code taken from https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    
    Parameters
    ----------
    points: (nx2) array
        an nx2 matrix of coordinates

    Returns 
    -------
    rval: (4x2) array
        a 4x2 matrix of the coordinates specify the corners of the bounding box. 

    """
    from scipy.ndimage.interpolation import rotate
    import numpy as np
    from scipy.spatial import ConvexHull

    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval
        
        
def polyarea_2D(xy):
    r""" compute the signed polygon area of a closed xy curve

    Parameters
    ----------
    xy : (N,2)
        coordinates of the closed curve. 
    
    Returns
    -------
    signed_area : scalar
        signed area, where a negative means the contour is clockwise instead of positive anticlockwise convention 
    """
    # check whether the last is the same as the first point. 
    check = sum([xy[0][ch]==xy[-1][ch] for ch in range(xy.shape[1])])
    if check == xy.shape[1]: 
        # then the first = last point. 
        p = xy[:-1].copy()
    else:
        p = xy.copy()

    signed_area = 0.5*(np.dot(p[:,0], np.roll(p[:,1],1)) - np.dot(p[:,1], np.roll(p[:,0],1))) # shoelace formula
    
    return signed_area


def fit_1d_spline_and_resample(pts, 
                               smoothing=1, 
                               k=3, 
                               n_samples=1000, 
                               periodic=False):
    r""" Fits a Parametric spline to given n-dimensional points using splines. 
    This function allows enforcement of periodicity in the points e.g. when the points are part of a closed contour. A common application is to fit a curve to spatial points coming from a geometric line object. 
    
    Parameters
    ----------
    pts : (N,d)
        d-dimension point cloud sampled from a line
    smoothing : scalar 
        controls the degree of smoothing. if 0, spline will interpolate. The higher the more smoothing
    k : int 
        order of the interpolating polynomial spline
    n_samples : int
        the number of samples points to equisample from the final fitted spline
    periodic : bool
        If True, enforce periodic or closure of the fitted spline. This is required for example when the input points come from a circle  
    
    Returns
    -------
    tck, u : tuple
        (t,c,k) a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline. see ``scipy.interpolate.splprep``. To interpolate with this spline we can use ``scipy.interpolate.splev`` such as 

    pts_out : (n_samples, d)
        the resampled points based on the fitted spline
    
        .. code-block:: python
            
            import scipy.interpolate as interpolate
            import numpy as np 

            u_fine = np.linspace(0, 1, n_samples) # sample from 0-1 of the parameter t
            pts_out = interpolate.splev(u_fine, tck)

    """
    from scipy import interpolate
    import numpy as np 
    
    # 1. fitting.
    if periodic == True:
        tck, u = interpolate.splprep(pts, per=1, k=k, s=smoothing)
    else:
        tck, u = interpolate.splprep(pts, k=k, s=smoothing)
    
    # 2 reinterpolation. 
    u_fine = np.linspace(0, 1, n_samples)
    pts_out = interpolate.splev(u_fine, tck)
    
    return (tck, u), pts_out 









    
    
    
    
    

