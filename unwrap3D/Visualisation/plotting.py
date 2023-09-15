
import pylab as plt 

def set_axes_equal(ax: plt.Axes):
    r"""Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.

    Parameters
    ----------
    ax : Maptlolib axes object
        3D Matplotlib axis to adjust     
    
    Returns
    -------
    None
    """
    import numpy as np 

    try:
        ax.set_box_aspect(aspect = (1,1,1))
    except:
        pass

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

    return []

def _set_axes_radius(ax, origin, radius):

    r"""Set 3D plot axes limits to origin +/- radius.

    This helper is used in set_axes_equal to ensure correct aspect ratio of 3D plots. 

    Parameters
    ----------
    ax : Maptlolib axes object
        3D Matplotlib axis to adjust  
    origin: (x,y,z) tuple of position
        the center coordinate to set the axis limits 
    radius: scalar
        the isotropic distance around origin to limit the 3D plot view to.  
    
    Returns
    -------
    None
    """
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

    return []
    
    

