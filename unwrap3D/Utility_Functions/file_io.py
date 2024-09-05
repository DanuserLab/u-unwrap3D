

def read_czifile(czi_file, squeeze=True):
    
    r""" Reads the data of a simple .czi microscope stack into a numpy array using the lightweight czifile library 

    Parameters
    ----------
    czifile : filepath
        path of the .czi file to read
    squeeze : bool
        specify whether singleton dimensions should be collapsed out

    Returns
    -------
    image_arrays : numpy array
        image stored in the .czi

    """   
    import numpy as np
    from czifile import CziFile
    
    with CziFile(czi_file) as czi:
        image_arrays = czi.asarray()
    if squeeze:
        image_arrays = np.squeeze(image_arrays)
    
    return image_arrays
    
def mkdir(directory):
    
    r"""Recursively creates all directories if they do not exist to make the folder specifed by 'directory'

    Parameters
    ----------
    directory : folderpath
        path of the folder to create
   
    """   

    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return []

def get_basename(pathname, ext):
   
    r"""helper to return the base file name minus the extension given an absolute or relative filepath location

    e.g. This function will return from a filepath, '../long_file_location/filename.tif', the file name minus the extension, 'filename'
    
    Parameters
    ----------
    pathname : filepath
        filepath to parse the file name from
    ext : string
        extension of the file format e.g. .tif, .png, .docx
   
    Returns
    -------
    basename : string
        the name of the file minus extension and location information
    """   

    import os 

    basename = os.path.split(pathname)[-1].split(ext)[0]

    return basename
    

def read_demons_matlab_tform( tform_file, volshape, keys=['u', 'v', 'w']):
    
    r"""helper for reading Matlab generated xyz demons deformation fields accounting for the difference in array convention.
    
    Parameters
    ----------
    tform_file : .mat (Matlab v5.4 format)
        filepath to the .mat of the saved deformation field saved from Matlab 
    volshape : (M,N,L) tuple
        shape of the original volume, the deformation fields correspond to. This shape is used to rescale any downsampled deformation fields to the original size using linear interpolation. 
    keys :  list of strings
        the variable names in the saved .mat corresponding to the x-, y-, z- direction deformation within Matlab 

    Returns
    -------
    w : (M,N,L) numpy array
        the 'x' deformation in Python tiff image reading convention 
    v : (M,N,L) numpy array
        the 'y' deformation in Python tiff image reading convention 
    u : (M,N,L) numpy array
        the 'z' deformation in Python tiff image reading convention 
    """   
    import scipy.io as spio
    import skimage.transform as sktform 
    import numpy as np 
    
    tform_obj = spio.loadmat(tform_file) # this assumes matlab v5.4 format

    u = (tform_obj[keys[0]]).astype(np.float32).transpose(2,0,1)
    v = (tform_obj[keys[1]]).astype(np.float32).transpose(2,0,1)
    w = (tform_obj[keys[2]]).astype(np.float32).transpose(2,0,1)

    scaling_factor = np.hstack(volshape).astype(np.float32) / np.hstack(u.shape)
    
    # transform this remembering to cast to float32.
    u = sktform.resize((u*scaling_factor[0]).astype(np.float32), volshape, preserve_range=True).astype(np.float32)
    v = sktform.resize((v*scaling_factor[2]).astype(np.float32), volshape, preserve_range=True).astype(np.float32)
    w = sktform.resize((w*scaling_factor[1]).astype(np.float32), volshape, preserve_range=True).astype(np.float32)

    return w,v,u


    
def save_array_to_nifti(data, savefile):

    r""" Saves a given numpy array to a nifti format using the nibabel library. The main use is for exporting volumes to annotate in ITKSnap.

    Parameters
    ----------
    data : numpy array
        input volume image
    savefile :  string
        filepath to save the output to, user should include the extension in this e.g. .nii.gz for compressed nifty

    """

    import nibabel as nib 
    import numpy as np 

    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, savefile)  

    return []


def read_pickle(filename):
    r""" read python3 pickled .pkl files

    Parameters
    ----------
    filename : filepath
        absolute path of the file to read
   
    """   
    import pickle
    
    with open(filename, 'rb') as output:
        return pickle.load(output)


def write_pickle(savepicklefile, savedict):
    r""" write python3 pickled .pkl files - use this for objects > 4GB 

    Parameters
    ----------
    savepicklefile : filepath
        absolute path of the file to write to
    savedict : dictionary
        dictionary of variables to write 
   
    """   

    import pickle

    with open(savepicklefile, 'wb') as handle:
        pickle.dump(savedict, 
                    handle)
        
    return []
    
    
    
