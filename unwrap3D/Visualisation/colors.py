



def get_colors(inp, colormap, vmin=None, vmax=None, bg_label=None):

	r""" Maps a given numpy input array with the specified Matplotlib colormap with optional specified minimum and maximum values. For an array that is integer such as that from multi-label segmentation, bg_label helps specify the background class which will automatically be mapped to a background color of [0,0,0,0]

    Parameters
    ----------
    inp : numpy array
        input n-d array to color
  	colormap :  matplotlib.cm colormap object
        colorscheme to apply e.g. cm.Spectral, cm.Reds, cm.coolwarm_r
    vmin : int/float
        specify the optional value to map as the minimum boundary of the colormap
    vmax : int/float
    	specify the optional value to map as the maximum boundary of the colormap
    bg_label: int
    	for an input array that is integer such as a segmentation mask, specify which integer label to mark as background. These values will all map to [0,0,0,0]

    Returns
    -------
    colored : numpy array
        the colored version of input as RGBA, the 4th being the alpha. colors are specified as floats in range 0.-1.
    """
	import pylab as plt 
	norm = plt.Normalize(vmin, vmax)

	colored = colormap(norm(inp))
	if bg_label is not None:
		colored[inp==bg_label] = 0 # make these all black!

	return colored