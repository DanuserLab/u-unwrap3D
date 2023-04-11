"""
This module generates some general default parameter settings to help get started with certain functions with many parameters 
"""

def optimesh_relaxation_config():
    r""" parameters for doing CVT relaxation of meshes using the optimesh library 
    
    see https://pypi.org/project/optimesh/
    """
    params = {}
    params['relax_method'] = 'CVT (block-diagonal)'
    params['tol'] = 1.0e-5
    params['n_iters']=20

    return params 

# Farneback 2D optical flow
def farneback2D_optical_flow():
    r""" parameters for extracting optical flow using Farneback method  
    
    see :func:`unwrap3D.Tracking.tracking.Eval_dense_optic_flow`
    """
    params = {}
    params['pyr_scale'] = 0.5
    params['levels'] = 3
    params['winsize'] = 15
    params['iterations'] = 5
    params['poly_n'] = 3
    params['poly_sigma'] = 1.2
    params['flags'] = 0 

    return params


def affine_register_matlab():
    r""" specify parameters for running Matlab imregtform to do Affine registration 
    
    see :func:`unwrap3D.Registration.registration.matlab_affine_register` for registering two timepoints and 

    see :func:`unwrap3D.Registration.registration.matlab_group_register_batch` for registering a video with switching of reference images.
    
    """
    params = {}
    params['downsample'] = [16., 8., 4.], # the scales to do registration at. [16, 8, 4, 2, 1] will continue on 2x downsampled and original resolution.
    params['modality'] = 'multimodal' # str specifying we want the multimodal registration option of imregtform
    params['type'] = 'similarity' # type of transform 
    params['view'] = None
    params['return_img'] = 1

    return params

def demons_register_matlab():
    r""" specify parameters for running Matlab imregdemons to do Demon's registration 
    
    see :func:`unwrap3D.Registration.registration.nonregister_3D_demons_matlab`
    """
    params = {}
    params['alpha'] = 0.1
    params['levels'] = [8,4,2,1]
    params['warps'] = [4,2,0,0]
    
    return params

def gradient_descent_affine_reg():
    r""" parameters for affine registration using simpleITK using Gradient Descent optimizer 
    
    see :func:`unwrap3D.Registration.registration.SITK_multiscale_affine_registration`
    """
    params = {}
    params['learningRate'] = 1
    params['numberOfIterations'] = 500
    params['convergenceMinimumValue'] = 1e-6
    params['convergenceWindowSize'] = 10 

    return params 

def evolutionary_affine_reg():
    r""" parameters for affine registration using simpleITK using Evolutionary optimizer
    
    see :func:`unwrap3D.Registration.registration.SITK_multiscale_affine_registration`
    """
    params = {}
    params['numberOfIterations'] = 100
    params['epsilon'] = 1.5e-4
    params['initialRadius'] = 6.25e-3
    params['growthFactor'] = 1.01
    params['shrinkFactor'] = -1.0 

    return params 

# demons multiscale registration in simpleITK