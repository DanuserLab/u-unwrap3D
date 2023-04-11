

def viz_grid(im_array, shape=None, cmap='gray', figsize=(10,10)):

    r""" given an array of images, plots them montage style in matplotlib 

    Parameters
    ----------
    im_array : n_img x n_rows x n_cols (x 3) grayscale or color numpy array
        array of n_img images to display
    shape :  (i,j) integer tuple 
        specifys the number of images to display row- and column- wise, if not set, will try np.sqrt, and make as square a panel as possible.
    cmap : string
        optional specification of a named matplotlib colormap to apply to view the image. 
    figsize : (int,int) tuple
        specify the figsize in matplotlib plot, implicitly controlling viewing resolution

    Returns
    -------
    (fig, ax): figure and ax handles 
        returned for convenience for downstream accessing of plot elements.
    """
    
    import pylab as plt 
    import numpy as np 
    
    n_imgs = len(im_array)
    
    if shape is not None:
        nrows, ncols = shape
    else:
        nrows = int(np.ceil(np.sqrt(n_imgs)))
        ncols = nrows
    
    color=True
    if len(im_array.shape) == 3:
        color=False
    if len(im_array.shape) == 4 and im_array.shape[-1]==1:
        color=False
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    for i in range(nrows):
        for j in range(ncols):
            im_index = i*ncols + j 
            if im_index<n_imgs:
                if color:
                    ax[i,j].imshow(im_array[im_index])
                else:
                    ax[i,j].imshow(im_array[im_index], cmap=cmap)
            # switch off all gridlines. 
            ax[i,j].axis('off')
            ax[i,j].grid('off')
           
    return (fig, ax) 
    
    

