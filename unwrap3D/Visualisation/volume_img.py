
import numpy as np 

def montage_vol_proj(vol, proj_fn=np.max):
    r"""Given a 3D grayscale volumetric imageand a projection function such as np.max, generates projected views onto x-y, x-z, y-z and montages them into one view.  

    Parameters
    ----------
    vol : (M,N,L) numpy array
        input grayscale volumetric image
    proj_fn : Python function
        this is any suitable projection function that allows the call format, proj_fn(vol, axis=0) to generate a 2D projection image.
    
    Returns
    -------
    montage_img : (N+M, L+L) 2D numpy array
        montaged 2D image which puts [top left] view_12, [top right] view_02, [bottom left] view_01
    """
    mm, nn, ll = vol.shape

    proj12 = proj_fn(vol, axis=0)
    proj02 = proj_fn(vol, axis=1)
    proj01 = proj_fn(vol, axis=2)

    vol_new = np.zeros((proj12.shape[0]+proj01.shape[0], proj12.shape[1]+proj02.shape[1]))
    
    vol_new[:proj12.shape[0], :proj12.shape[1]] = proj12.copy()
    vol_new[:proj02.shape[0], proj12.shape[1]:] = proj02.copy()
    vol_new[proj12.shape[0]:, :proj01.shape[1]] = proj01.copy()
    
    return vol_new

