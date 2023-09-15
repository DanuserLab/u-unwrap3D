
def imshowpair(im1,im2, ax=None):

    r""" Combines two potentially different sized grayscale images into one image. 

    Parameters
    ----------
    im1 : (M1,N1) grayscale numpy array
        input image 1 
    im2 :  (M2,N2) grayscale numpy array 
        input image 2 
    ax : Matplotlib axes object
        optional axis to display the image
    
    Returns
    -------
    img : (M_max,N_max,3) numpy array
        the combined image of im1 and im2, where M_max=max(M_1, M_2), and N_max=(N_1,N_2) are the largest row and column dimensions of im1, im2. The images are automatically centered in the output. 
    """

    from skimage.exposure import rescale_intensity
    import pylab as plt
    import numpy as np 
    
    dtype = type(im1.ravel()[0]) # check the data type
    shape1 = np.array(im1.shape)
    shape2 = np.array(im2.shape)
    
    img_shape = np.max(np.vstack([shape1,shape2]), axis=0)
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype=dtype)
    
    offset1x = (img_shape[0] - shape1[0]) // 2; offset1y = (img_shape[1]-shape1[1]) // 2;
    offset2x = (img_shape[0] - shape2[0]) // 2; offset2y = (img_shape[1]-shape2[1]) // 2;
    
    # display centered images. 
    img[offset1x:offset1x+shape1[0],offset1y:offset1y+shape1[1],0] = im1
    img[offset2x:offset2x+shape2[0],offset2y:offset2y+shape2[1],1] = im2 
    
    if ax is not None:
        ax.imshow(rescale_intensity(img))
    
    return img
    
def checkerboard_imgs(im1, im2, grid=(10,10)):
    
    r""" generate a checkerboard montage of two 2D images, useful for checking misalignment when registering two images. 
    
    Parameters
    ----------
    im1 : (M,N) grayscale or (M,N,3) color image as numpy array
        input image 1
    im2 : (M,N) grayscale or (M,N,3) color image as numpy array
        input image 2
    grid : (int,int) tuple
        specifies the number of rows and number of columns in the gridding of the checkerboard. 

    Returns
    -------
    im : (M,N,3) numpy array
        the checkerboard combined image of im1 and im2. 
    """

    import numpy as np
    import pylab as plt

    # im1, im2 are grayscale or rgb images only. 
    if len(im1.shape) == 2:
        # grayscale image.
        rows, cols = im1.shape
    if len(im1.shape) == 3:
        # rgb image.
        rows, cols, _ = im1.shape
        
    # set up return image
    im = np.zeros((rows, cols, 3))
    
    # create the checkerboard mask.
    check_rows = np.linspace(0, rows, grid[0]+1).astype(np.int32)
    check_cols = np.linspace(0, cols, grid[1]+1).astype(np.int32)
    checkerboard = np.zeros((rows,cols))
    
    for i in range(grid[0]):
        
        if np.mod(i,2) == 0:
            even = 0
        else:
            even = 1
            
        for j in range(grid[1]):
            r = [check_rows[i], check_rows[i+1]]
            c = [check_cols[j], check_cols[j+1]]
            checkerboard[r[0]:r[1], c[0]:c[1]] = even 
            
            if even == 0:
                even = 1
            else:
                even = 0
                    
    ones = np.array(np.where(checkerboard==1)).T
    zeros = np.array(np.where(checkerboard==0)).T
        
    if len(im1.shape) == 2:
        # grayscale image.
        im[ones[:,0], ones[:,1], 0] = im1[ones[:,0], ones[:,1]]
        im[zeros[:,0], zeros[:,1], 1] = im2[zeros[:,0], zeros[:,1]]
        
    if len(im1.shape) == 3:
        # rgb image.
        im[ones[:,0], ones[:,1], :] = im1[ones[:,0], ones[:,1], :]
        im[zeros[:,0], zeros[:,1], :] = im2[zeros[:,0], zeros[:,1], :]
    
    return im 
