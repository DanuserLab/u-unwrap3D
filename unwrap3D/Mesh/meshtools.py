

from ..Geometry import geometry as geom 
from ..Unzipping import unzip as uzip


def read_mesh(meshfile, 
              process=False, 
              validate=False, 
              keep_largest_only=False):
    r""" Wrapper around trimesh.load_mesh such that the mesh is read exactly with the same vertices and face indexes by default. Additionally we introduce a convenient flag to keep just the largest mesh component

    Parameters
    ----------
    meshfile : filepath
        input mesh of any common format, e.g. .obj, .ply, .dae, .stl, see https://trimsh.org/index.html
    process : bool
        If True, degenerate and duplicate faces will be removed immediately, and some functions will alter the mesh to ensure consistent results.
    validate : bool
        if True, Nan and Inf values will be removed immediately and vertices will be merged
    keep_largest_only : bool
        if True, keep only the largest connected component of the mesh, ignoring whether this is watertight or not 

    Returns
    -------
    mesh : trimesh.Trimesh or trimesh.Scene
        loaded mesh geometry

    """
    import trimesh 
    import numpy as np 

    mesh = trimesh.load_mesh(meshfile, 
                             validate=validate, 
                             process=process)
    if keep_largest_only:
        mesh_comps = mesh.split(only_watertight=False)
        mesh = mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]

    return mesh 

def create_mesh(vertices,faces,vertex_colors=None, face_colors=None):
    r""" Wrapper around trimesh.Trimesh to create a mesh given the vertices, faces and optionally vertex colors or face colors.

    Parameters
    ----------
    vertices : (n_vertices,3) array 
        the vertices of the mesh geometry 
    faces : (n_faces,3) array
        the 0-indexed integer indices indicating how vertices are joined together to form a triangle element
    vertex_colors : (n_vertices,3) array
        if provided, an array of the RGB color values per vertex 
    face_colors : (n_faces,3) array
        if provided, an array of the RGB color values per face 

    Returns
    -------
    mesh : trimesh.Trimesh or trimesh.Scene
        created mesh geometry with colors saved in mesh.visual.vertex_colors or mesh.visual.face_colors

    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices,
                            faces=faces, 
                            process=False,
                            validate=False, 
                            vertex_colors=vertex_colors, 
                            face_colors=face_colors)

    return mesh 

def submesh(mesh,
            faces_sequence,
            mesh_face_attributes,
            repair=True,
            only_watertight=False,
            min_faces=None,
            append=False,**kwargs):
        
    r""" Return a subset of a mesh. Function taken from the Trimesh library.

    Parameters
    ------------
    mesh : Trimesh
        Source mesh to take geometry from
    faces_sequence : sequence (p,) int
        Indexes of mesh.faces
    only_watertight : bool
        Only return submeshes which are watertight.
    append : bool
        Return a single mesh which has the faces appended, if this flag is set, only_watertight is ignored

    Returns
    ----------
    if append : Trimesh object
    else        list of Trimesh objects
    """

    import copy
    import numpy as np

    def type_bases(obj, depth=4):
        """
        Return the bases of the object passed.
        """
        import collections
        bases = collections.deque([list(obj.__class__.__bases__)])
        for i in range(depth):
            bases.append([i.__base__ for i in bases[-1] if i is not None])
        try:
            bases = np.hstack(bases)
        except IndexError:
            bases = []
        # we do the hasattr as None/NoneType can be in the list of bases
        bases = [i for i in bases if hasattr(i, '__name__')]
        return np.array(bases)

    
    def type_named(obj, name):
        """
        Similar to the type() builtin, but looks in class bases
        for named instance.
        Parameters
        ------------
        obj: object to look for class of
        name : str, name of class
        Returns
        ----------
        named class, or None
        """
        # if obj is a member of the named class, return True
        name = str(name)
        if obj.__class__.__name__ == name:
            return obj.__class__
        for base in type_bases(obj):
            if base.__name__ == name:
                return base
        raise ValueError('Unable to extract class of name ' + name)
    
    # evaluate generators so we can escape early
    faces_sequence = list(faces_sequence)

    if len(faces_sequence) == 0:
        return []

    # avoid nuking the cache on the original mesh
    original_faces = mesh.faces.view(np.ndarray)
    original_vertices = mesh.vertices.view(np.ndarray)

    faces = []
    vertices = []
    normals = []
    visuals = []
    attributes = []

    # for reindexing faces
    mask = np.arange(len(original_vertices))

    for index in faces_sequence:
        # sanitize indices in case they are coming in as a set or tuple
        index = np.asanyarray(index)
        if len(index) == 0:
            # regardless of type empty arrays are useless
            continue
        if index.dtype.kind == 'b':
            # if passed a bool with no true continue
            if not index.any():
                continue
            # if fewer faces than minimum
            if min_faces is not None and index.sum() < min_faces:
                continue
        elif min_faces is not None and len(index) < min_faces:
            continue

        current = original_faces[index]
        unique = np.unique(current.reshape(-1)) # unique points. 

        # redefine face indices from zero
        mask[unique] = np.arange(len(unique))
        normals.append(mesh.face_normals[index])
        faces.append(mask[current])
        vertices.append(original_vertices[unique])
        attributes.append(mesh_face_attributes[index])
        visuals.append(mesh.visual.face_subset(index))

    if len(vertices) == 0:
        return np.array([])

    # we use type(mesh) rather than importing Trimesh from base
    # to avoid a circular import
    trimesh_type = type_named(mesh, 'Trimesh')

    # generate a list of Trimesh objects
    result = [trimesh_type(
        vertices=v,
        faces=f,
        face_normals=n,
        visual=c,
        metadata=copy.deepcopy(mesh.metadata),
        process=False) for v, f, n, c in zip(vertices,
                                             faces,
                                             normals,
                                             visuals)]
    result = np.array(result)

    return result, attributes


def split_mesh(mesh,
                mesh_face_attributes, 
                adjacency=None, 
                only_watertight=False, 
                engine=None, **kwargs):

    r""" Split a mesh into multiple meshes from face connectivity taken from the Trimesh library. If only_watertight is true it will only return watertight meshes and will attempt to repair
    single triangle or quad holes.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    only_watertight: bool
        Only return watertight components
    adjacency : (n, 2) int
        Face adjacency to override full mesh
    engine : str or None
        Which graph engine to use

    Returns
    ----------
    meshes : (m,) trimesh.Trimesh
        Results of splitting
    meshes_attributes : (m,d) attributes. 
        associated splitted attributes. 
    """

    import trimesh 
    import numpy as np 
    
    # used instead of trimesh functions in order to keep it consistent with the splitting of mesh attributes. 
    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 4
    else:
        min_len = 1

    components = trimesh.graph.connected_components(
        edges=adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=min_len,
        engine=engine)

    meshes, meshes_attributes = submesh(mesh,
                                    components, 
                                    mesh_face_attributes,
                                    **kwargs)

    return meshes, meshes_attributes


def decimate_resample_mesh(mesh, remesh_samples, predecimate=True):
    r""" Downsample (decimate) and optionally resample the mesh to equilateral triangles. 

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh 
    remesh_samples : 0-1
        fraction of the number of vertex points to target in size of the output mesh
    predecimate : bool
        if True, small edges are first collapsed using igl.decimate in the ``igl`` library

    Returns
    -------
    mesh : trimesh.Trimesh
        output mesh

    """
    # this will for sure change the connectivity 
    import pyacvd
    import pyvista as pv
    import igl
    import trimesh

    if predecimate:
        _, V, F, _, _ = igl.decimate(mesh.vertices, mesh.faces, int(.9*len(mesh.vertices))) # there is bug? 
        if len(V) > 0: # have a check in here to prevent break down. 
            mesh = trimesh.Trimesh(V, F, validate=True) # why no good? 

    # print(len(mesh.vertices))
    mesh = pv.wrap(mesh) # convert to pyvista format. 
    clus = pyacvd.Clustering(mesh)
    clus.cluster(int(remesh_samples*len(mesh.points))) # this guarantees a remesh is possible. 
    mesh = clus.create_mesh()

    mesh = trimesh.Trimesh(mesh.points, mesh.faces.reshape((-1,4))[:, 1:4], validate=True) # we don't care. if change
    # print(mesh.is_watertight)
    return mesh


def upsample_mesh(mesh, method='inplane'):
    r""" Upsample a given mesh using simple barycentric splittng ('inplane') or using 'loop', which slightly smoothes the output 
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh
    method : str
        one of 'inplane' or 'loop' allowed in igl.upsample

    Returns
    -------
    mesh_out : trimesh.Trimesh
        output mesh

    """
    """
    inplane or loop
    """
    import igl 
    import trimesh 

    if method =='inplane': 
        uv, uf = igl.upsample(mesh.vertices, mesh.faces)
    if method == 'loop':
        uv, uf = igl.loop(mesh.vertices, mesh.faces)
   
    mesh_out = trimesh.Trimesh(uv, uf, validate=False, process=False)
    
    return mesh_out 


def upsample_mesh_and_vertex_vals(mesh, vals, method='inplane'):
    r""" Upsample a given mesh using simple barycentric splittng ('inplane') or using 'loop', which slightly smoothes the output and also reinterpolate any associated vertex values for the new mesh
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh
    vals : (n_vertices, n_features)
        vertex based values to also upsample
    method : str
        one of 'inplane' or 'loop' allowed in igl.upsample

    Returns
    -------
    mesh_out : trimesh.Trimesh
        output mesh

    """
    """
    inplane only... vals is the same length as mesh vertices. 
    """
    import igl 
    import trimesh 
    import numpy as np 

    if method =='inplane': 
        uv, uf = igl.upsample(mesh.vertices, mesh.faces) # get the new vertex and faces. 
    if method == 'loop':
        uv, uf = igl.loop(mesh.vertices, mesh.faces)
    vals_new = np.zeros((len(uv), vals.shape[-1])); vals_new[:] = np.nan
    max_ind_mesh_in = len(mesh.vertices)
    vals_new[:max_ind_mesh_in] = vals.copy()

    old_new_edge_list = igl.edges(uf) # use the new faces. 
    old_new_edge_list = old_new_edge_list[old_new_edge_list[:,0]<max_ind_mesh_in]

    vals_new[old_new_edge_list[::2,1]] = .5*(vals_new[old_new_edge_list[::2,0]]+vals_new[old_new_edge_list[1::2,0]])
    mesh_out = trimesh.Trimesh(uv, uf, validate=False, process=False)
    
    return mesh_out, vals_new


def marching_cubes_mesh_binary(vol, 
                                presmooth=1., 
                                contourlevel=.5, 
                                remesh=False, 
                                remesh_method='pyacvd', 
                                remesh_samples=.5, 
                                remesh_params=None, 
                                predecimate=True, 
                                min_mesh_size=40000, 
                                keep_largest_only=True, 
                                min_comp_size=20, 
                                split_mesh=True, 
                                upsamplemethod='inplane'):
    r""" Mesh an input binary volume using Marching Cubes algorithm with optional remeshing to improve mesh quality 

    Parameters
    ----------
    vol : trimesh.Trimesh
        input mesh
    presmooth : scalar
        pre Gaussian smoothing with the specified sigma to get a better marching cubes mesh. 
    contourlevel : 
        isolevel to extract the Marching cubes mesh
    remesh_method : str
        one of 'pyacvd' or 'optimesh'. 

        'pyacvd' : str
            pyacvd uses voronoidal clustering i.e. kmeans clustering to produce a uniformly remeshing, see https://github.com/pyvista/pyacvd 
        'optimesh' : str
            if selected, this method aims to relax the mesh vertices to a more uniform state, see https://github.com/meshpro/optimesh. This doesn't change the number of vertices and so effect of this is limited and there is some changing of the input shape

    remesh_samples : 0-1
        fraction of the number of vertex points to target in size of the output mesh
    remesh_params : dict
        only for remesh_method='optimesh'. See :func:`unwrap3D.Parameters.params.optimesh_relaxation_config` for template of parameter settings
    predecimate : bool
        if True, collapse the small edges in the Marching Cubes output before remeshing
    min_mesh_size : int
        minimum number of vertices in the output mesh 
    keep_largest_only : bool
        if True, check and keep only the largest mesh component 
    min_comp_size : int
        if keep_largest_only=False, remesh=True and split_mesh=True, individual connected components of the mesh is checked and only those > min_comp_size are kept. This is crucial if the mesh is to be remeshed. The remeshing fraction is applied to all components equally.  Without this check, there will be errors as some mesh components become zero.
    split_mesh : bool
        if True, runs connected component to filter out Marching cubes components that are too small (keep_largest_only=False) or keep only the largest (keep_largest_only=True) prior to remeshing        
    upsamplemethod : str
        one of 'inplane' or 'loop' allowed in igl.upsample. This is called to meet the minimum number of vertices in the final mesh as specified in ``min_mesh_size``

    Returns
    -------
    mesh : trimesh.Trimesh
        output mesh

    """
    from skimage.filters import gaussian
    import trimesh
    try:
        from skimage.measure import marching_cubes_lewiner
    except:
        from skimage.measure import marching_cubes
    import igl 
    import numpy as np 

    if presmooth is not None:
        img = gaussian(vol, sigma=presmooth, preserve_range=True)
        img = img / img.max() # do this. 
    else:
        img = vol.copy()

    try:
        V, F, _, _ = marching_cubes_lewiner(img, level=contourlevel, allow_degenerate=False)
    except:
        V, F, _, _ = marching_cubes(img, level=contourlevel, method='lewiner', allow_degenerate=False)

    mesh = trimesh.Trimesh(V,F, validate=True)
    
    if split_mesh:
        mesh_comps = mesh.split(only_watertight=False)
        
        if keep_largest_only:
            mesh = mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]
        else:
            mesh_comps = [mm for mm in mesh_comps if len(mm.faces)>=min_comp_size] # keep a min_size else the remeshing doesn't work 
            # combine_mesh_components
            mesh = trimesh.util.concatenate(mesh_comps)
        # we need to recombine this
        # mesh = mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]
    if remesh:
        if remesh_method == 'pyacvd':
            mesh = decimate_resample_mesh(mesh, remesh_samples, predecimate=predecimate)
            # other remesh is optimesh which allows us to reshift the vertices (change the connections)
        if remesh_method == 'optimesh':
            if predecimate:
                _, V, F, _, _ = igl.decimate(mesh.vertices,mesh.faces, int(.9*len(mesh.faces))) # decimates up to the desired amount of faces? 
                mesh = trimesh.Trimesh(V, F, validate=True)
            mesh, _, mean_quality = relax_mesh( mesh, relax_method=remesh_params['relax_method'], tol=remesh_params['tol'], n_iters=remesh_params['n_iters']) # don't need the quality parameters. 
            # print('mean mesh quality: ', mean_quality)

    mesh_check = len(mesh.vertices) >= min_mesh_size # mesh_min size is only applied here.!
    while(mesh_check==0):
        mesh = upsample_mesh(mesh, method=upsamplemethod)
        mesh_check = len(mesh.vertices) >= min_mesh_size

    return mesh


def measure_props_trimesh(mesh, main_component=True, clean=True):
    r""" Compute basic statistics and properties of a given mesh

    - is Convex : Yes/No
    - is Volume : Yes/No - is it closed such that a volume can be computed
    - is Watertight : Yes/No - is it closed such that a volume can be computed
    - orientability : Yes/No - can all faces be oriented the same way. Mobius strips and Klein bottles are non-orientable
    - Euler number : or Euler characteristic, :math:`\chi` #vertices - #edges + #faces
    - Genus : :math:`(2-2\chi)/2` if orientable or :math:`2-\chi` if nonorientable 

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh
    main_component : bool 
        if True, get the largest mesh component and compute statistics on this 
    clean : bool
        if True, removes NaN and infs and degenerate and duplicate faces which may affect the computation of some of these statistics

    Returns
    -------
    props : dict
        A dictionary containing the metrics

        'convex' : bool

        'volume' : bool

        'watertight' : bool

        'orientability' : bool

        'euler_number' : scalar

        'genus' : scalar

    """
    import trimesh
    import numpy as np 
    # check 
    # make sure we do a split
    if clean: 
        mesh_ = trimesh.Trimesh(mesh.vertices,
                                mesh.faces,
                                validate=True,
                                process=True)
    else:
        mesh_ = mesh.copy()
    mesh_comps = mesh_.split(only_watertight=False)
    main_mesh = mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]
    
    props = {}
    props['convex'] = main_mesh.is_convex
    props['volume'] = main_mesh.is_volume
    props['watertight'] = main_mesh.is_watertight
    props['orientability'] = main_mesh.is_winding_consistent
    props['euler_number'] = main_mesh.euler_number
    
    if main_mesh.is_winding_consistent:
        # if orientable we can use the euler_number computation. see wolfram mathworld!. 
        genus = (2.-props['euler_number'])/2.# euler = 2-2g
    else:
        genus = (2.-props['euler_number'])
    props['genus'] = genus
    
    return props


def measure_triangle_props(mesh_, clean=True):
    r""" Compute statistics regarding the quality of the triangle faces

    Parameters
    ----------
    mesh_ : trimesh.Trimesh
        input mesh
    clean : bool
        if True, removes NaN and infs and degenerate and duplicate faces which may affect the computation of some of these statistics

    Returns 
    -------
    props : dict
        A dictionary containing the metrics

        'min_angle' : scalar
            minimum internal triangle angle of faces in degrees
        'avg_angle' : scalar
            mean internal triangle angle of faces in degrees
        'max_angle' : 
            maximum internal triangle angle of faces in degrees
        'std_dev angle' : 
            standard devation of internal triangle angle of faces in degrees
        'min_quality' : 
            minimum triangle quality. triangle quality is measured as 2*inradius/circumradius
        'avg_quality' : 
            mean triangle quality. triangle quality is measured as 2*inradius/circumradius
        'max_quality' : 
            maximum triangle quality. triangle quality is measured as 2*inradius/circumradius
        'quality'
            per face quality. triangle quality is measured as 2*inradius/circumradius
        'angles' : 
            all internal face angles 
    """
    import numpy as np 
    import trimesh
    import igl
    
    if clean: 
        mesh = trimesh.Trimesh(mesh_.vertices,
                                mesh_.faces,
                                validate=True,
                                process=True)
    else:
        mesh = mesh_.copy()
        
    # if use_igl:
    angles = igl.internal_angles(mesh.vertices, mesh.faces)
    q =  2.* igl.inradius(mesh.vertices, mesh.faces) / igl.circumradius(mesh.vertices, mesh.faces)# 2 * inradius / circumradius
            
    # mesh_meshplex = meshplex.MeshTri(mesh.vertices, mesh.faces)
    # angles = mesh_meshplex.angles / np.pi * 180.
    # q = mesh_meshplex.q_radius_ratio
    props = {}
    props['min_angle'] = np.nanmin(angles) / np.pi * 180.
    props['avg_angle'] = np.nanmean(angles) / np.pi * 180.
    props['max_angle'] = np.nanmax(angles) / np.pi * 180.
    props['std_dev_angle'] = np.nanstd(angles) / np.pi * 180.
    props['min_quality'] = np.nanmin(q)
    props['avg_quality'] = np.nanmean(q)
    props['max_quality'] = np.nanmax(q)
    props['quality'] = q
    props['angles'] = angles
    
    return props

def PCA_rotate_mesh(binary, mesh=None, mesh_contour_level=.5):
    r""" Compute principal components of a given binary through extracting the surface mesh or a used specified surface mesh 

    Parameters
    ----------
    binary : array
        input binary image 
    mesh : trimesh.Trimesh
        a user-specified surface mesh
    mesh_contour_level : scalar
        if only a binary is provided Marching cubes is used to extract a surface mesh at the isolevel given by ``mesh_contour_level``

    Returns
    -------
    pca_model : scikit-learn PCA model instance
        a fitted princial components model for the mesh. see sklearn.decomposition.PCA for attributes
    mean_pts : (3,) array
        the centroid of the surface mesh with which points were demeaned prior to PCA

    """
    import numpy as np 
    from sklearn.decomposition import PCA
    from skimage.measure import marching_cubes_lewiner
    import igl 
    
    if mesh is not None:
        # we don't have a given surface, instead we need to segment. 
        v = mesh.vertices.copy()
        f = mesh.faces.copy()
    else:
        # if use_surface:
        v, f, _, _ = marching_cubes_lewiner(binary, level=mesh_contour_level)
    # else:
    #     pts = np.argwhere(binary>0)
    #     pts = np.vstack(pts)
    barycenter = igl.barycenter(v,f)
    weights = igl.doublearea(v, f)
    # print(weights.shape)
    # print(barycenter.shape)
    mean_pts = np.nansum( (weights / float(np.sum(weights)))[:,None] * barycenter, axis=0)
    
    pts_ = v - mean_pts[None,:]
    pca_model = PCA(n_components=pts_.shape[-1], random_state=0, whiten=False)
    pca_model.fit(pts_)

    return pca_model, mean_pts


def voxelize_image_mesh_pts(mesh, pad=50, dilate_ksize=3, erode_ksize=1, vol_shape=None, upsample_iters_max=10, pitch=2):
    r""" Given a surface mesh, voxelises the mesh to create a closed binary volume to enable for exampled signed distance function comparison and for repairing small holes 

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh 
    pad : int 
        integer isotropic pad to create a volume grid if vol_shape is not given 
    dilate_ksize : int
        optional dilation of the voxelized volume with a ball kernel of specified radius to fill holes so that scipy.ndimage.morphology.binary_fill_holes will allow a complete volume to be otained
    erode_ksize : int
        optional erosion of the voxelized volume with a ball kernel of specified radius
    vol_shape : (m,n,l) tuple
        the size of the volume image to voxelize onto  
    upsample_iters_max : int 
        the maximum number of recursive mesh subdivisions to achieve the target pitch 
    pitch : scalar
        target side length of each voxel, the mesh will be recursively subdivided up to the maximum number of iterations specified by ``upsample_iters_max`` until this is met.
    
    Returns 
    -------
    smooth_img_binary : (MxNxL) array 
        binary volume image 

    """
    # works only for meshs from images. so all coordinates are positive. 
    # this voxelizes the pts without need for an image. 
    import numpy as np 
    import skimage.morphology as skmorph
    from scipy.ndimage.morphology import binary_fill_holes
    import igl 
    
    vv = mesh.vertices.copy()
    ff = mesh.faces.copy()

    if vol_shape is None:
        # mesh_pts = mesh.vertices.copy() + 1
        longest_edge_length = igl.edge_lengths(vv,ff).max()
        factor = longest_edge_length / float(dilate_ksize) / 2.
        # print(factor)
        if factor >= pitch / 2. :
            # print('upsample')
            # # then we can't get a volume even if watertight. 
            upsample_iters = int(np.rint(np.log2(factor)))
            # print(upsample_iters)
            upsample_iters = np.min([upsample_iters, upsample_iters_max])
            vv, ff = igl.upsample(mesh.vertices, mesh.faces, upsample_iters)
        mesh_pts = igl.barycenter(vv,ff) + 1
        # determine the boundaries. 
        min_x, min_y, min_z = np.min(mesh_pts, axis=0)
        max_x, max_y, max_z = np.max(mesh_pts, axis=0)

        # pad = int(np.min([min_x, min_y, min_z])) # auto determine the padding based on this. 
        # new binary. 
        smooth_img_binary = np.zeros((int(max_x)+pad, int(max_y)+pad, int(max_z)+pad))
    else:
        # mesh_pts = mesh.vertices.copy() #+ .5
        # mesh_pts = mesh.vertices.copy() + 1
        longest_edge_length = igl.edge_lengths(vv,ff).max()
        factor = longest_edge_length / float(dilate_ksize) / 2.
        if factor >= pitch / 2. :
            # then we can't get a volume even if watertight. 
            upsample_iters = int(np.rint(np.log2(factor)))
            upsample_iters = np.min([upsample_iters, upsample_iters_max])
            vv, ff = igl.upsample(mesh.vertices, mesh.faces, upsample_iters)
        mesh_pts = igl.barycenter(vv,ff)
        smooth_img_binary = np.zeros(vol_shape)

    smooth_img_binary[mesh_pts[:,0].astype(np.int), 
                      mesh_pts[:,1].astype(np.int), 
                      mesh_pts[:,2].astype(np.int)] = 1
    if dilate_ksize is not None:
        smooth_img_binary = skmorph.binary_dilation(smooth_img_binary, skmorph.ball(dilate_ksize))
    smooth_img_binary = binary_fill_holes(smooth_img_binary) # since we dilated before to create a full mesh. we inturn must erode. 
    
    if erode_ksize is not None:
        smooth_img_binary = skmorph.binary_erosion(smooth_img_binary, skmorph.ball(erode_ksize))

    return smooth_img_binary 


def area_normalize_mesh(mesh, map_color=False, centroid='area'):
    r""" Normalize the mesh vertices by subtracting the centroid and dividing by the square root of the total surface area. 

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh
    map_color : bool 
        if True, copy across the vertex and face colors to the new normalised mesh 
    centroid : str
        specifies the method for computing the centroid of the mesh. If 'area' the face area weighted centroid is computed from triangle barycenter. If 'points' the centroid is computed from triangle barycenters with no weighting 
    
    Returns 
    ------- 
    mesh_out : trimesh.Trimesh
        output normalized mesh 
    (v_mean, v_out_scale) : ((3,) array, scalar)
        the computed centroid and scalar normalisation

    """
    import igl 
    import trimesh
    import numpy as np 
    
    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    
    if centroid == 'area':
        # this uses barycenters. 
        area_weights_v = (igl.doublearea(v,f)/2.)[:,None] 
        v_mean = np.nansum( area_weights_v/float(np.nansum(area_weights_v)) * igl.barycenter(v,f), axis=0)
    if centroid == 'points':
        v_mean = np.nanmean(igl.barycenter(v,f), axis=0) # 3*more barycenters
    v_out = v - v_mean[None,:]
    v_out_scale = float(np.sqrt(np.sum(igl.doublearea(v,f))/2.))
    v_out = v_out / v_out_scale
    
    if map_color:
        mesh_out = trimesh.Trimesh(v_out, f, vertex_colors=mesh.visual.vertex_colors, face_colors=mesh.visual.face_colors, process=False, validate=False)
    else:
        mesh_out = trimesh.Trimesh(v_out, f, process=False, validate=False)
    
    return mesh_out, (v_mean, v_out_scale)


def unit_sphere_normalize_mesh(mesh, map_color=False, centroid='area'):
    r""" Normalize the mesh vertices by direct projection onto the unit sphere by normalising the displacement vector relative to the centroid

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh
    map_color : bool 
        if True, copy across the vertex and face colors to the new normalised mesh 
    centroid : str
        specifies the method for computing the centroid of the mesh. If 'area' the face area weighted centroid is computed from triangle barycenter. If 'points' the centroid is computed from triangle barycenters with no weighting 
    
    Returns 
    ------- 
    mesh_out : trimesh.Trimesh
        output normalized mesh 
    (v_mean, v_out_scale) : ((3,) array, scalar)
        the computed centroid and scalar normalisation

    """
    import igl 
    import trimesh
    import numpy as np 
    
    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    
    if centroid == 'area':
        # this uses barycenters. 
        area_weights_v = (igl.doublearea(v,f)/2.)[:,None] 
        v_mean = np.nansum( area_weights_v/float(np.nansum(area_weights_v)) * igl.barycenter(v,f), axis=0)
    if centroid == 'points':
        v_mean = np.nanmean(igl.barycenter(v,f), axis=0) # 3*more barycenters
    v_out = v - v_mean[None,:]
    v_out_scale = np.linalg.norm(v_out, axis=-1)
    v_out = v_out / v_out_scale[:,None]
    
    if map_color:
        mesh_out = trimesh.Trimesh(v_out, f, vertex_colors=mesh.visual.vertex_colors, process=False, validate=False)
    else:
        mesh_out = trimesh.Trimesh(v_out, f, process=False, validate=False)
    
    return mesh_out, (v_mean, v_out_scale)


def get_uv_grid_tri_connectivity(grid):
    r""" Construct the vertex and faces indices to convert a (M,N,d) d-dimensional grid coordinates with spherical geometry to a triangle mesh where vertices=grid.ravel()[vertex_indices], faces=face_indices. 
    
    Parameters
    ----------
    grid : (M,N) or (M,N,d) array
        input (u,v) image used to construct vertex and face indices for
    
    Returns
    -------
    vertex_indices_all : (N_all,3) array
        specifies the flattened indices in the grid to form the vertices of the triangle mesh
    triangles_all : 
        specifies the flattened indices in the grid to form the faces of the triangle mesh 

    """
    """
    grid should be odd 
    the first row is same point and degenerate. 
    the last row is same point and degenerate. 

    easier to build the vertex and face connectivity from scratch. in a non-degenerate way!.  
    """
    import numpy as np 
    import trimesh

    m, n = grid.shape[:2]
    interior_grid = grid[1:-1, :-1].copy()
    img_grid_indices = np.arange(np.prod(grid.shape[:2])).reshape(grid.shape[:2])
    # img_grid_indices_interior = img_grid_indices[1:-1, :-1].copy()
    img_grid_indices_interior = np.arange(np.prod(grid[1:-1, :-1].shape[:2])).reshape(grid[1:-1, :-1].shape[:2]) # set these as the new indices.
    M, N = img_grid_indices_interior.shape[:2]
    # print(m,n)
    vertex_indices_interior  = (img_grid_indices[1:-1,:-1].ravel())
    vertex_indices_north = img_grid_indices[0,n//2]
    vertex_indices_south = img_grid_indices[-1,n//2]

    vertex_indices_all = np.hstack([vertex_indices_interior, 
                                    vertex_indices_north, 
                                    vertex_indices_south]) # get all the vertex indices. 
    
    # build the interior grid periodic triangle connectivity. 
    img_grid_indices_main = np.hstack([img_grid_indices_interior, 
                                       img_grid_indices_interior[:,0][:,None]])
    M, N = img_grid_indices_main.shape[:2]
    # squares_main = np.vstack([img_grid_indices_main[:m-2, :n-1].ravel(),
    #                           img_grid_indices_main[1:m-1, :n-1].ravel(), 
    #                           img_grid_indices_main[1:m-1, 1:n].ravel(),
    #                           img_grid_indices_main[:m-2, 1:n].ravel()]).T    
    squares_main = np.vstack([img_grid_indices_main[:M-1, :N-1].ravel(),
                              img_grid_indices_main[1:M, :N-1].ravel(), 
                              img_grid_indices_main[1:M, 1:N].ravel(),
                              img_grid_indices_main[:M-1, 1:N].ravel()]).T
    # trianglulate the square and this is indexed in terms of the uv grid. 
    squares_main_triangles = trimesh.geometry.triangulate_quads(squares_main)
    # now add the triangles that connect to the poles. 
    triangles_north_pole = np.vstack([ len(vertex_indices_interior)*np.ones(len(img_grid_indices_main[0][1:])), 
                                       img_grid_indices_main[0][1:], 
                                       img_grid_indices_main[0][:-1]]).T
    
    triangles_south_pole = np.vstack([  img_grid_indices_main[-1][:-1],
                                        img_grid_indices_main[-1][1:],
                                        (len(vertex_indices_interior)+1)*np.ones(len(img_grid_indices_main[0][1:]))]).T

    # compile all the triangles together. 
    triangles_all = np.vstack([squares_main_triangles[:,::-1],
                               triangles_north_pole,
                               triangles_south_pole]).astype(np.int)

    # can determine the sign orientation using vector area. 
    # implement triangle orientation check to check orientation consistency?
    return vertex_indices_all, triangles_all 


def build_img_2d_edges(grid):
    r"""  Extract the 4-neighbor edge connectivity for a (M,N) 2D image, returning an array of the list of edge connections 

    Parameters
    ----------
    grid : (M,N) image
        input image of the width and height to get the edge connectivity between pixels
    
    Returns
    -------
    e : (n_edges,2) array
        the list of unique edges specified in terms of the flattened indices in the grid. 
    """
    import numpy as np 

    m, n = grid.shape[:2]
    img_grid_indices = np.arange(np.prod(grid.shape[:2])).reshape(grid.shape[:2])

    e1 = np.vstack([img_grid_indices[:m-1, :n-1].ravel(), 
                    img_grid_indices[1:m, :n-1].ravel()]).T
    e2 = np.vstack([img_grid_indices[1:m, :n-1].ravel(), 
                    img_grid_indices[1:m, 1:n].ravel()]).T
    e3 = np.vstack([img_grid_indices[1:m, 1:n].ravel(), 
                    img_grid_indices[:m-1, 1:n].ravel()]).T
    e4 = np.vstack([img_grid_indices[:m-1, 1:n].ravel(), 
                    img_grid_indices[:m-1, :n-1].ravel()]).T
    
    e = np.vstack([e1,e2,e3,e4]) # these should be the upper triangular matrix. 
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)

    return e 

def get_inverse_distance_weight_grid_laplacian(grid, grid_pts, alpha=0.1):
    r""" Compute a sparse grid Laplacian matrix for 2D image based on inverse weighting of edge lengths. This allows to take into account the length distortion of grid points constructed from 2D unwrapping  
    
    Parameters
    ----------
    grid : (M,N) image
        input image of the width and height to get the edge connectivity between pixels
    grid_pts : (M,N,d) image
        input image with which to compute edge lengths based on the Euclidean distance of the d-features. e.g. this could be the bijective (u,v) <-> (x,y,z) unwrapping parameters where d=3. 
    alpha : scalar
        a shape factor that controls the inverse distance weights. In short, this is a small pseudo-distance added to measured distances to avoid division by zero or infs. 

    Returns
    -------
    L : (MxN,MxN) array
        the sparse grid Laplacian where edge connections factor into account the distance between ``grid_pts``
    """
    import numpy as np
    import scipy.sparse as spsparse
    from sklearn.preprocessing import normalize

    m, n = grid.shape[:2] 
    grid_pts_flat = grid_pts.reshape(-1, grid_pts.shape[-1])

    elist = build_img_2d_edges(grid)
    dist_edges = np.linalg.norm(grid_pts_flat[elist[:,0]] - grid_pts_flat[elist[:,1]], axis=-1)

    # make into a vertex edge distance matrix. 
    n = len(grid_pts_flat)
    D = spsparse.csr_matrix((dist_edges, (elist[:,0], elist[:,1])), 
                            shape=(n,n))
    
    D = D + D.transpose() # make symmetric! # this is still not correct? # this should make symmetric!. 
    # # D = spsparse.triu(D).tocsr()
    D = normalize(D, axis=1, norm='l1') # should be computing the weights... 
    D = .5*(D + D.transpose()) # make symmetric! 
    D.data = 1./(alpha+D.data) # c.f. https://math.stackexchange.com/questions/4264675/handle-zero-in-inverse-distance-weighting
    D = normalize(D, axis=1, norm='l1')
    D = .5*(D + D.transpose()) # make symmetric! 

    # to convert to laplacian we can simply do D - A. 
    L = spsparse.spdiags(np.squeeze(D.sum(axis=1)), 0, D.shape[0], D.shape[1])  - D # degree - adjacency matrix. 
    L = L.tocsc()

    return L


# this version doesn't support mesh laplacian if using the corresponding 3D coordinates because the first row and last row maps to the same point and generates the triangles. - use the above triangle version. 
def get_uv_grid_quad_connectivity(grid, return_triangles=False, bounds='spherical'): 
    r""" Compute the quad and the triangle connectivity between pixels in a 2d grid with either spherical or no boundary conditions

    Parameters
    ----------
    grid : (M,N,d) image
        input image of the width and height to get the pixel connectivity, N must be odd if bounds='spherical' due to the necessary rewrapping. 
    return_triangles : bool
        if True, return in addition the triangle connectivity based on triangulation of the quad grid connectivity
    bounds : str
        string specifying the boundary conditions of the grid, either of 'spherical' or 'none'

        'spherical' : str
            this wraps the left to right side, pinches together the top and pinches together the bottom of the grid 
        'none' : str
            this does no unwrapping and returns the grid connectivity of the image. This is the same as sklearn.feature_extraction.image.grid_to_graph

    Returns 
    -------
    all_squares : (N_squares,4) array
        the 4-neighbor quad connectivity of flattened image indices
    all_squares_to_triangles : (2*N_squares,3) array
        the triangle connectivity of the flattened image indices. Each square splits into 2 triangles.
    
    """
    import trimesh
    import numpy as np 
    
    m, n = grid.shape[:2]
    img_grid_indices = np.arange(np.prod(grid.shape[:2])).reshape(grid.shape[:2])
        
    if bounds == 'spherical':

        img_grid_indices_main = np.hstack([img_grid_indices, img_grid_indices[:,0][:,None]])
        squares_main = np.vstack([img_grid_indices_main[:m-1, :n].ravel(),
                                  img_grid_indices_main[1:m, :n].ravel(), 
                                  img_grid_indices_main[1:m, 1:n+1].ravel(),
                                  img_grid_indices_main[:m-1, 1:n+1].ravel()]).T

        # then handle the top and bottom strips separately ... 
        img_grid_indices_top = np.vstack([img_grid_indices[0,n//2:][::-1], 
                                          img_grid_indices[0,:n//2]])
        squares_top = np.vstack([img_grid_indices_top[0, :img_grid_indices_top.shape[1]-1].ravel(),
                                 img_grid_indices_top[1, :img_grid_indices_top.shape[1]-1].ravel(), 
                                 img_grid_indices_top[1, 1:img_grid_indices_top.shape[1]].ravel(),
                                 img_grid_indices_top[0, 1:img_grid_indices_top.shape[1]].ravel()]).T
        
        img_grid_indices_bottom = np.vstack([img_grid_indices[-1,:n//2], 
                                             img_grid_indices[-1,n//2:][::-1]])
        squares_bottom = np.vstack([img_grid_indices_bottom[0, :img_grid_indices_bottom.shape[1]-1].ravel(),
                                    img_grid_indices_bottom[1, :img_grid_indices_bottom.shape[1]-1].ravel(), 
                                    img_grid_indices_bottom[1, 1:img_grid_indices_bottom.shape[1]].ravel(),
                                    img_grid_indices_bottom[0, 1:img_grid_indices_bottom.shape[1]].ravel()]).T

        
        all_squares = np.vstack([squares_main, 
                                 squares_top,
                                 squares_bottom])
    if bounds == 'none':
        all_squares = np.vstack([img_grid_indices[:m-1, :n-1].ravel(),
                                 img_grid_indices[1:m, :n-1].ravel(), 
                                 img_grid_indices[1:m, 1:n].ravel(),
                                 img_grid_indices[:m-1, 1:n].ravel()]).T
    all_squares = all_squares[:,::-1]
    
    if return_triangles:
        all_squares_to_triangles = trimesh.geometry.triangulate_quads(all_squares)
        
        return all_squares, all_squares_to_triangles
    else:
        return all_squares


def parametric_mesh_constant_img_flow(mesh, external_img_gradient, 
                                        niters=1, 
                                        deltaL=5e-4, 
                                        step_size=1, 
                                        method='implicit', 
                                        robust_L=False, 
                                        mollify_factor=1e-5,
                                        conformalize=True, 
                                        gamma=1, 
                                        alpha=0.2, 
                                        beta=0.1, 
                                        eps=1e-12):

    r""" This function performs implicit Euler propagation of a 3D mesh with steps of constant ``step_size`` in the direction of an external image gradient specified by ``external_img_gradient``

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh 
    external_img_gradient : (MxNxLx3) array
        the 3D volumetric displacement field with x,y,z coordinates the last axis.
    niters : int
        the number of total steps
    deltaL : scalar
        a stiffness regularization constant for the conformalized mean curvature flow propagation 
    step_size : scalar
        the multiplicative factor the image gradient is multipled with per iteration
    method : str
        one of 'implicit' for implicit Euler or 'explicit' for explicit Euler. 'implicit' is slower but much more stable and results in mesh updates that minimize foldover and instabilities. 'explicit' is unstable but fast.
    robust_L : bool
        if True, uses the robust Laplacian construction of Sharpe et al. [1]_. If False, the standard cotan Laplacian is used. The robust Laplacian enables proper handling of degenerate and nonmanifold vertices such as that if using the triangle mesh constructed from a uv image grid. The normal 3D mesh if remeshed does not necessarily need this
    mollify_factor : scalar
        the mollification factor used in the robust Laplacian. see https://github.com/nmwsharp/robust-laplacians-py
    conformalize : bool
        if True, uses the simplified conformalized mean curvature variant mesh propagation derived in the paper [2]_. If False, uses the normal active contours update which uses ``gamma``, ``alpha`` and ``beta`` parameters.
    gamma : scalar
        stability regularization parameter in the active contour
    alpha : scalar
        stiffness regularization parameters in the active contour
    beta : scalar
        bending regularization parameters in the active contour 
    eps : scalar
        small constant for numerical stability 

    Returns
    -------
    Usteps : (n_vertices,3,n_iters+1) array
        the vertex positions of the mesh at every interation. The face connectivity of the mesh does not change. 

    References
    ----------

    .. [1] Sharp, Nicholas, and Keenan Crane. "A laplacian for nonmanifold triangle meshes." Computer Graphics Forum. Vol. 39. No. 5. 2020.
    .. [2] c.f. Unwrapping paper.  

    """
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from tqdm import tqdm 
    from ..Image_Functions import image as image_fn 

    vol_shape = external_img_gradient.shape[:-1]
    v = np.array(mesh.vertices.copy())
    f = np.array(mesh.faces.copy())

    if robust_L:
        import robust_laplacian
        L, M = robust_laplacian.mesh_laplacian(np.array(v), np.array(f), mollify_factor=mollify_factor)
        L = -L # need to invert sign to be same convention as igl. 
    else:
        L = igl.cotmatrix(v,f)

    """
    initialise
    """
    # initialise the save array. 
    Usteps = np.zeros(np.hstack([v.shape, niters+1]))
    Usteps[...,0] = v.copy()
    U = v.copy()

    """
    propagation
    """
    for ii in tqdm(range(niters)):

        U_prev = U.copy(); # make a copy. 
        # get the next image gradient
        U_grad = np.array([image_fn.map_intensity_interp3(U_prev, 
                                                      grid_shape=vol_shape, 
                                                      I_ref=external_img_gradient[...,ch]) for ch in np.arange(v.shape[-1])])
        U_grad = U_grad.T

        if method == 'explicit':
            U_grad = U_grad / (np.linalg.norm(U_grad, axis=-1)[:,None]**2 + eps) # square this. 
            U = U_prev + U_grad * step_size #update. 
            Usteps[...,ii+1] = U.copy()

        if method == 'implicit':
            U_grad = U_grad / (np.linalg.norm(U_grad, axis=-1)[:,None] + eps)

            if conformalize:
                # if ii ==0:
                if robust_L:
                    import robust_laplacian
                    _, M = robust_laplacian.mesh_laplacian(U_prev, f, mollify_factor=mollify_factor) 
                else:
                    M = igl.massmatrix(U_prev, f, igl.MASSMATRIX_TYPE_BARYCENTRIC) # -> this is the only matrix that doesn't degenerate.               
                # # implicit solve. 
                S = (M - deltaL*L) # what happens when we invert? 
                b = M.dot(U_prev + U_grad * step_size)
            else:
                # construct the active contour version.
                S = gamma * spsparse.eye(len(v), len(v)) - alpha * L  + beta * L.dot(L)
                b = U_prev + U_grad * step_size
            
            # get the next coordinate by solving 
            U = spsparse.linalg.spsolve(S,b)
            Usteps[...,ii+1] = U.copy()

    # return all the intermediate steps. 
    return Usteps


def parametric_uv_unwrap_mesh_constant_img_flow(uv_grid, 
                                        external_img_gradient, 
                                        niters=1, 
                                        deltaL=5e-4, 
                                        surf_pts_ref=None, 
                                        step_size=1,
                                        pad_dist=5,
                                        method='implicit', 
                                        robust_L=False, 
                                        mollify_factor=1e-5,
                                        conformalize=True, gamma=1, alpha=0.2, beta=0.1, eps=1e-12):
    r""" This convenience function performs implicit Euler propagation of an open uv-parametrized 3D mesh with steps of constant ``step_size`` in the direction of an external image gradient specified by ``external_img_gradient``

    the uv-parametrization is closed before propagating in 3D (x,y,z) space

    Parameters
    ----------
    uv_grid : (M,N,3) image
        input mesh as an image with xyz on the last dimension.
    external_img_gradient : (MxNxLx3) array
        the 3D volumetric displacement field with x,y,z coordinates the last axis.
    niters : int
        the number of total steps
    deltaL : scalar
        a stiffness regularization constant for the conformalized mean curvature flow propagation 
    surf_pts_ref : (N,3) array 
        if provided, this is a reference surface with which to automatically determine the niters when propagating the surface outwards to ensure the full reference shape is sampled in topography space.
    step_size : scalar
        the multiplicative factor the image gradient is multipled with per iteration
    pad_dist : int
        an additional fudge factor added to the automatically determined n_dist when surf_pts_ref is provided and n_dist is not user provided
    method : str
        one of 'implicit' for implicit Euler or 'explicit' for explicit Euler. 'implicit' is slower but much more stable and results in mesh updates that minimize foldover and instabilities. 'explicit' is unstable but fast.
    robust_L : bool
        if True, uses the robust Laplacian construction of Sharpe et al. [1]_. If False, the standard cotan Laplacian is used. The robust Laplacian enables proper handling of degenerate and nonmanifold vertices such as that if using the triangle mesh constructed from a uv image grid. The normal 3D mesh if remeshed does not necessarily need this
    mollify_factor : scalar
        the mollification factor used in the robust Laplacian. see https://github.com/nmwsharp/robust-laplacians-py
    conformalize : bool
        if True, uses the simplified conformalized mean curvature variant mesh propagation derived in the paper [2]_. If False, uses the normal active contours update which uses ``gamma``, ``alpha`` and ``beta`` parameters.
    gamma : scalar
        stability regularization parameter in the active contour
    alpha : scalar
        stiffness regularization parameters in the active contour
    beta : scalar
        bending regularization parameters in the active contour 
    eps : scalar
        small constant for numerical stability 

    Returns
    -------
    Usteps_out_rec : (niters+1,M,N,3) array
        the vertex positions of the mesh for every interation for every pixel position

    See Also 
    --------
    :func:`unwrap3D.Mesh.meshtools.parametric_mesh_constant_img_flow` : 
        the propagation for a general 3D mesh
    :func:`unwrap3D.Unzipping.unzip.prop_ref_surface` : 
        the equivalent for explicit Euler propagation using the uv-based image coordinates without closing the mesh 

    References
    ----------

    .. [1] Sharp, Nicholas, and Keenan Crane. "A laplacian for nonmanifold triangle meshes." Computer Graphics Forum. Vol. 39. No. 5. 2020.
    .. [2] c.f. Unwrapping paper.  

    """
    """
    convert the uv image into a trimesh object and prop this with some conformal regularization in one function.
    """
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from tqdm import tqdm 
    from ..Unzipping import unzip_new as uzip 
    import trimesh

    vol_shape = external_img_gradient.shape[:-1]

    """
    Build the UV mesh connectivity. 
    """
    uv_connectivity_verts, uv_connectivity_tri = get_uv_grid_tri_connectivity(uv_grid[...,0])
    v = (uv_grid.reshape(-1,uv_grid.shape[-1]))[uv_connectivity_verts]
    f = uv_connectivity_tri.copy()

    mesh_tri = trimesh.Trimesh(vertices=v,
                               faces=f, 
                               validate=False,
                               process=False)
    trimesh.repair.fix_winding(mesh_tri) # fix any orientation issues. 

    """
    determine the propagation distance.
    """
    # infer the number of dists to step for from the reference if not prespecified. 
    if niters is None :
        unwrap_params_ref_flat = uv_grid.reshape(-1, uv_grid.shape[-1])
        # infer the maximum step size so as to cover the initial otsu surface.
        mean_pt = np.nanmean(unwrap_params_ref_flat, axis=0)
        # # more robust to do an ellipse fit. ? => doesn't seem so... seems best to take the extremal point -> since we should have a self-similar shape. 
        # unwrap_params_fit_major_len = np.max(np.linalg.eigvalsh(np.cov((unwrap_params_ref_flat-mean_pt[None,:]).T))); unwrap_params_fit_major_len=np.sqrt(unwrap_params_fit_major_len)
        # surf_ref_major_len = np.max(np.linalg.eigvalsh(np.cov((surf_pts_ref-mean_pt[None,:]).T))); surf_ref_major_len = np.sqrt(surf_ref_major_len)
        mean_dist_unwrap_params_ref = np.linalg.norm(unwrap_params_ref_flat-mean_pt[None,:], axis=-1).max()
        mean_surf_pts_ref = np.linalg.norm(surf_pts_ref-mean_pt[None,:], axis=-1).max() # strictly should do an ellipse fit... 

        niters = np.int(np.ceil(mean_surf_pts_ref-mean_dist_unwrap_params_ref))
        niters = niters + pad_dist # this is in pixels
        niters = np.int(np.ceil(niters / np.abs(float(step_size)))) # so if we take 1./2 step then we should step 2*

        print('auto_infer_prop_distance', niters)
        print('----')

    """
    Do the propagation wholly with this watertight mesh. 
    """
    Usteps_out = parametric_mesh_constant_img_flow(mesh_tri, 
                                                    external_img_gradient=external_img_gradient, 
                                                    niters=niters, 
                                                    deltaL=deltaL, 
                                                    step_size=step_size, 
                                                    method=method,
                                                    robust_L=robust_L, 
                                                    conformalize=conformalize, 
                                                    gamma=gamma, 
                                                    alpha=alpha, 
                                                    beta=beta, 
                                                    eps=eps)

    Usteps_out_rec = Usteps_out[:-2].reshape((uv_grid.shape[0]-2, uv_grid.shape[1]-1, Usteps_out.shape[1], Usteps_out.shape[2]))
    Usteps_out_rec = np.vstack([Usteps_out_rec[-2][None,...], 
                               Usteps_out_rec, 
                               Usteps_out_rec[-1][None,...]])
    Usteps_out_rec = np.hstack([Usteps_out_rec, 
                               Usteps_out_rec[:,0][:,None,...]])

    # return all the intermediate steps. 
    return Usteps_out_rec # return the propagated out mesh. 


def area_distortion_flow_relax_sphere(mesh, 
                                      mesh_orig, 
                                      max_iter=50,
                                      smooth_iters=0,  
                                      delta=0.1, 
                                      stepsize=.1, 
                                      conformalize=False,
                                      flip_delaunay=False,
                                      robust_L=False,
                                      mollify_factor=1e-5,
                                      eps = 1e-12,
                                      debugviz=False):
    r""" This function relaxes the area distortion of a spherical mesh by advecting vertex coordinates whilst maintaining the spherical geometry. 

    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input unit spherical mesh to relax
    mesh_orig : trimesh.Trimesh 
        the input original geometric mesh whose vertices correspond 1 to 1 with vertices of the spherical mesh. This is used to compute the area distortion per iteration
    max_iter : int
        the number of iterations relaxation will occur. The function may exit early if the mesh becomes unable to support further relaxation. A collapsed mesh will return vertices that are all np.nan
    smooth_iters : int
        if > 0, the number of Laplacian smoothing to smooth the per vertex area distortion 
    delta : scalar
        a stiffness constant of the mesh. it used to ensure maintenance of relative topology during advection
    stepsize : scalar
        the stepsize in the direction of steepest descent of area distortion. smaller steps can improve stability and precision but with much slower convergence
    conformalize : bool
        if True, uses the initial Laplacian without recomputing the Laplacian. This is a very severe penalty and stops area relaxation flow without reducing ``delta``. In general set this as False since relaxing area is in opposition to minimizing conformal error. 
    flip_delaunay : bool
        if True, flip triangles during advection. On the sphere we find this slows flow, affects barycentric interpolation and is generated not required. This option requires the ``meshplex`` library
    robust_L : bool
        if True, uses the robust Laplacian construction of Sharpe et al. [1]_. If False, the standard cotan Laplacian is used. The robust Laplacian enables proper handling of degenerate and nonmanifold vertices such as that if using the triangle mesh constructed from a uv image grid. The normal 3D mesh if remeshed does not necessarily need this
    mollify_factor : scalar
        the mollification factor used in the robust Laplacian. see https://github.com/nmwsharp/robust-laplacians-py
    eps : scalar
        small constant for numerical stability 
    debugviz : bool
        if True, a histogram of the area distortion is plotted per iteration to check if the flow is working properly. The area distortion is plotted as log(distortion) and so should move towards a peak of 0

    Returns
    -------
    v_steps : list of (n_vertices, 3) array
        the vertex position at every iteration 
    f_steps : list of (n_faces, 3) array
        the face connectivity at every iteration. This will be the same for all timepoints unless flip_delaunay=True
    area_distortion_iter : list
        the area distortion factor per face computed as area_original/area_sphere for every timepoint. 
    
    References
    ----------

    .. [1] Sharp, Nicholas, and Keenan Crane. "A laplacian for nonmanifold triangle meshes." Computer Graphics Forum. Vol. 39. No. 5. 2020.

    """
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from tqdm import tqdm 
    # import meshplex
    import pylab as plt
    if robust_L:
        import robust_laplacian

    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    
    if robust_L:
        L, M = robust_laplacian.mesh_laplacian(np.array(v), np.array(f), mollify_factor=mollify_factor)
        L = -L # need to invert sign to be same convention as igl. 
    else:
        L = igl.cotmatrix(v,f)

    area_distortion_iter = []
    v_steps = [v]
    f_steps = [f]

    for ii in tqdm(range(max_iter)):
        
        try:

            if conformalize == False:
                if robust_L:
                    L, m = robust_laplacian.mesh_laplacian(np.array(v), np.array(f),mollify_factor=mollify_factor); L = -L; # this must be computed. # if not... then no growth -> flow must change triangle shape!. 
                else:
                    L = igl.cotmatrix(v,f)
                    m = igl.massmatrix(v,f, igl.MASSMATRIX_TYPE_BARYCENTRIC)

            # # compute the area distortion of the face -> having normalized for surface area. -> this is because the sphere minimise the surface area. -> guaranteeing positive. 

            # which is correct? 
            # why we need to use the original connectivity? 
            # area_distortion_mesh = igl.doublearea(mesh_orig.vertices/np.sqrt(np.nansum(igl.doublearea(mesh_orig.vertices,mesh_orig.faces)*.5)), mesh_orig.faces) / igl.doublearea(v/np.sqrt(np.nansum(igl.doublearea(v,f)*.5)), f) # this is face measure!. and the connectivity is allowed to change! during evolution !. 
            area_distortion_mesh = igl.doublearea(mesh_orig.vertices/np.sqrt(np.nansum(igl.doublearea(mesh_orig.vertices,f)*.5)), f) / igl.doublearea(v/np.sqrt(np.nansum(igl.doublearea(v,f)*.5)), f) # this is face measure!. and the connectivity is allowed to change! during evolution !. 
            # area_distortion_mesh = area_distortion(mesh_orig.vertices,f, v)
            # area_distortion_mesh = area_distortion(mesh_orig.vertices)
            # area_distortion_iter.append(area_distortion_mesh) # append this. 
            # push to vertex. 
            area_distortion_mesh_vertex = igl.average_onto_vertices(v, 
                                                                    f, 
                                                                    np.vstack([area_distortion_mesh,area_distortion_mesh,area_distortion_mesh]).T)[:,0]
        
            # smooth ... 
            if debugviz:
                plt.figure()
                plt.hist(np.log10(area_distortion_mesh_vertex)) # why no change? 
                plt.show()

            if smooth_iters > 0:
                smooth_area_distortion_mesh_vertex = np.vstack([area_distortion_mesh_vertex,area_distortion_mesh_vertex,area_distortion_mesh_vertex]).T # smooth this instead of the gradient. 

                for iter_ii in range(smooth_iters):
                    smooth_area_distortion_mesh_vertex = igl.per_vertex_attribute_smoothing(smooth_area_distortion_mesh_vertex, f) # seems to work.
                area_distortion_mesh_vertex = smooth_area_distortion_mesh_vertex[:,0]
        
            # compute the gradient. 
            g = igl.grad(v, 
                         f)

            # if method == 'Kazhdan2019':
            # compute the vertex advection gradient.
            gA = g.dot(-np.log(area_distortion_mesh_vertex)).reshape(f.shape, order="F") 
            # scale by the edge length.
            gu_mag = np.linalg.norm(gA, axis=1)
            max_size = igl.avg_edge_length(v, f) / np.nanmedian(gu_mag) # if divide by median less good. 
            
            vA = max_size*gA # this is vector. 
            normal = igl.per_vertex_normals(v,f)
            # Vvertex = Vvertex - np.nansum(v*normal, axis=-1)[:,None]*normal # this projection actually makes it highly unstable? 
            vA_vertex = igl.average_onto_vertices(v, 
                                                  f, vA)
            vA_vertex = vA_vertex - np.nansum(vA_vertex*normal, axis=-1)[:,None]*normal # this seems necessary... 

            """
            advection step 
            """
            S = (m - delta*L)

            # if adaptive_step:
            #     v = spsparse.linalg.spsolve(S, m.dot(v + scale_factor * stepsize * vA_vertex))
            # else:
            v = spsparse.linalg.spsolve(S, m.dot(v + stepsize * vA_vertex))
            
            """
            rescale and reproject back to normal
            """
            area = np.nansum(igl.doublearea(v,f)*.5) #total area. 
            c = np.nansum((0.5*igl.doublearea(v,f)/area)[...,None] * igl.barycenter(v,f), axis=0) # this is just weighted centroid
            v = v - c[None,:] 
            # sphericalize 
            v = v/np.linalg.norm(v, axis=-1)[:,None] # forces sphere.... topology.... relaxation. 

            """
            flip delaunay ? or use optimesh refine? -> to improve triangle quality? 
            """
            if flip_delaunay:

                import meshplex
                # this clears out the overlapping. is this necessary
                mesh_out = meshplex.MeshTri(v, f)
                mesh_out.flip_until_delaunay()
            
                # update v and f!
                v = mesh_out.points.copy()
                f = mesh_out.cells('points').copy() 

            # update. 
            v_steps.append(v)
            f_steps.append(f)
            area_distortion_iter.append(area_distortion_mesh) # append this. 

        except:
            # if error then get out quickly. 
            return v_steps, f_steps, area_distortion_iter

    return v_steps, f_steps, area_distortion_iter



def area_distortion_flow_relax_disk(mesh, mesh_orig, 
                                    max_iter=50,
                                    # smooth_iters=5,  
                                    delta_h_bound=0.5, 
                                    stepsize=.1, 
                                    flip_delaunay=True, # do this in order to dramatically improve flow!. 
                                    robust_L=False, # use the robust laplacian instead of cotmatrix - slows flow. 
                                    mollify_factor=1e-5,
                                    eps = 1e-8,
                                    lam = 1e-4, 
                                    debugviz=False,
                                    debugviz_tri=True):
    
    r""" This function relaxes the area distortion of a mesh with disk topology i.e. a disk, square or rectangle mesh by advecting inner vertex coordinates to minimise area distortion. 
    
    The explicit Euler scheme of [1]_ is used. Due to the numerical instability of such a scheme, the density of mesh vertices and the stepsize constrains the full extent of relaxation.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input disk, square or rectangle mesh to relax. The first coordinate of all the vertices, i.e. mesh.vertices[:,0] should be uniformly set to a constant e.g. 0 to specify a 2D mesh
    mesh_orig : trimesh.Trimesh 
        the input original geometric mesh whose vertices correspond 1 to 1 with vertices of the input mesh. This is used to compute the area distortion per iteration    
    delta_h_bound : scalar
        the maximum value of the absolute area difference between original and the relaxing mesh. This constrains the maximum gradient difference, avoiding updating local areas too fast which will then destroy local topology. 
    stepsize : scalar
        the stepsize in the direction of steepest descent of area distortion. smaller steps can improve stability and precision but with much slower convergence
    flip_delaunay : bool
        if True, flip triangles during advection. We find this is important with the explicit Euler scheme adopted here to ensure correct topology and ensure fast relaxation.
    robust_L : bool
        if True, uses the robust Laplacian construction of Sharpe et al. [2]_. If False, the standard cotan Laplacian is used. The robust Laplacian enables proper handling of degenerate and nonmanifold vertices such as that if using the triangle mesh constructed from a uv image grid. The normal 3D mesh if remeshed does not necessarily need this
    mollify_factor : scalar
        the mollification factor used in the robust Laplacian. see https://github.com/nmwsharp/robust-laplacians-py
    eps : scalar
        small constant for numerical stability 
    debugviz : bool
        if True, a histogram of the area distortion is plotted per iteration to check if the flow is working properly. The area distortion is plotted as log(distortion) and so should move towards a peak of 0
    debugvis_tri : bool
        if True, plots the triangle mesh per iteration.

    Returns
    -------
    v_steps : list of (n_vertices, 3) array
        the vertex position at every iteration. The first coordinate of the vertices is set to 0. 
    f_steps : list of (n_faces, 3) array
        the face connectivity at every iteration. This will be the same for all timepoints unless flip_delaunay=True
    area_distortion_iter : list
        the area distortion factor per face computed as area_original/area_sphere for every timepoint. 

    References
    ----------
    .. [1] Zou, Guangyu, et al. "Authalic parameterization of general surfaces using Lie advection." IEEE Transactions on Visualization and Computer Graphics 17.12 (2011): 2005-2014.
    .. [2] Sharp, Nicholas, and Keenan Crane. "A laplacian for nonmanifold triangle meshes." Computer Graphics Forum. Vol. 39. No. 5. 2020.

    """
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from tqdm import tqdm 
    # import meshplex
    import pylab as plt
    if robust_L:
        import robust_laplacian

    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    
    f = igl.intrinsic_delaunay_triangulation(igl.edge_lengths(v,f), f)[1]

    if robust_L:
        L, M = robust_laplacian.mesh_laplacian(np.array(v), np.array(f), mollify_factor=mollify_factor)
    else:
        L = -igl.cotmatrix(v,f)

    area_distortion_iter = []
    v_steps = [v]
    f_steps = [f]

    for ii in tqdm(range(max_iter)):
        
        # try:

        # if conformalize == False:
        if robust_L:
            L, m = robust_laplacian.mesh_laplacian(np.array(v), np.array(f),mollify_factor=mollify_factor); # this must be computed. # if not... then no growth -> flow must change triangle shape!. 
        else:
            L = -igl.cotmatrix(v,f)

        v_bound = igl.boundary_loop(f)
                
        # # compute the area distortion of the face -> having normalized for surface area. -> this is because the sphere minimise the surface area. -> guaranteeing positive. 
        A2 = igl.doublearea(mesh_orig.vertices/np.sqrt(np.nansum(igl.doublearea(mesh_orig.vertices,f)*.5)), f)
        A1 = igl.doublearea(v/np.sqrt(np.nansum(igl.doublearea(v,f)*.5)), f)
        
        # B = np.log10(A1/(A2)) # - 1
        B = (A1+eps)/(A2+eps) - 1 # adding regularizer to top and bottom is better!. 

        area_distortion_mesh = (A2/(A1+eps)) # this is face measure!. and the connectivity is allowed to change! during evolution !. 
        area_distortion_mesh_vertex = igl.average_onto_vertices(v, 
                                                                f, 
                                                                np.vstack([area_distortion_mesh,area_distortion_mesh,area_distortion_mesh]).T)[:,0]

        # smooth ... 
        if debugviz:
            plt.figure()
            plt.hist(np.log10(area_distortion_mesh_vertex)) # why no change? 
            plt.show()

        # if smooth_iters > 0:
        #     smooth_area_distortion_mesh_vertex = np.vstack([area_distortion_mesh_vertex,area_distortion_mesh_vertex,area_distortion_mesh_vertex]).T # smooth this instead of the gradient. 

        #     for iter_ii in range(smooth_iters):
        #         smooth_area_distortion_mesh_vertex = igl.per_vertex_attribute_smoothing(smooth_area_distortion_mesh_vertex, f) # seems to work.
        #     area_distortion_mesh_vertex = smooth_area_distortion_mesh_vertex[:,0]

        B = np.clip(B, -delta_h_bound, delta_h_bound) # bound above and below. 
        # B_vertex = igl.average_onto_vertices(v, 
        #                                       f, 
        #                                       np.vstack([B,B,B]).T)[:,0] # more accurate? 
        B_vertex = f2v(v,f).dot(B)

        # from scipy.sparse.linalg import lsqr ---- this sometimes fails!... 
        I = spsparse.spdiags(lam*np.ones(len(v)), [0], len(v), len(v)) # tikholov regulariser. 
        g = spsparse.linalg.spsolve((L.T).dot(L) + I, (L.T).dot(B_vertex)) # solve for a smooth potential field.  # this is the least means square. 
        # g = spsparse.linalg.lsqr(L.T.dot(L), L.dot(B_vertex), iter_lim=100)[0] # is there a better way to solve this quadratic? 

        face_vertex = v[f].copy()
        face_normals = np.cross(face_vertex[:,1]-face_vertex[:,0], 
                                face_vertex[:,2]-face_vertex[:,0], axis=-1)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=-1)[:,None] + eps)
        # face_normals = np.vstack([np.ones(len(face_vertex)), 
        #                           np.zeros(len(face_vertex)),
        #                           np.zeros(len(face_vertex))]).T # should this be something else? 
        face_g = g[f].copy()
        
        # vertex_normals = igl.per_vertex_normals(v,f)

        # i,j,k = 1,2,3
        face_vertex_lhs = np.concatenate([(face_vertex[:,1]-face_vertex[:,0])[:,None,:],
                                          (face_vertex[:,2]-face_vertex[:,1])[:,None,:],
                                          face_normals[:,None,:]], axis=1) 
        face_g_rhs = np.vstack([(face_g[:,1]-face_g[:,0]),
                                (face_g[:,2]-face_g[:,1]),
                                 np.zeros(len(face_g))]).T


        # solve a simultaneous set of 3x3 problems
        dg_face = np.linalg.solve( face_vertex_lhs, face_g_rhs)

        gu_mag = np.linalg.norm(dg_face, axis=1) 
        max_size = igl.avg_edge_length(v, f) / np.nanmax(gu_mag) # stable if divide by nanmax # must be nanmax!. 
        
        # dg_face = stepsize*max_size*dg_face # this is vector. and is scaled by step size 
        # dg_face = max_size*dg_face
        
        # average onto the vertex. 
        dg_vertex = igl.average_onto_vertices(v, 
                                              f, 
                                              dg_face)
        dg_vertex = dg_vertex * max_size
        # dg_vertex = dg_vertex - np.nansum(dg_vertex*vertex_normals,axis=-1)[:,None]*vertex_normals
        
        # correct the flow at the boundary!. # this is good? ---> this is good for an explicit euler. 
        normal_vect = L.dot(v)
        normal_vect = normal_vect / (np.linalg.norm(normal_vect, axis=-1)[:,None] + 1e-8)

        # dg_vertex[v_bound] = dg_vertex[v_bound] - np.nansum(dg_vertex[v_bound] * v[v_bound], axis=-1)[:,None]*v[v_bound]

        """
        this is the gradient at the vertex. 
        """
        dg_vertex[v_bound] = dg_vertex[v_bound] - np.nansum(dg_vertex[v_bound] * normal_vect[v_bound], axis=-1)[:,None]*normal_vect[v_bound]
        # disps.append(dg_vertex)
        
        # print('no_adaptive')
        scale_factor=stepsize
        # print('scale_factor, ', scale_factor)
        """
        advection step 
        """
        # v = V[:,1:] + np.array(disps).sum(axis=0)[:,1:] # how to make this step stable? 
        v = v_steps[-1][:,1:] + scale_factor*dg_vertex[:,1:] # last one. 
        v = np.hstack([np.zeros(len(v))[:,None], v])
        
        if flip_delaunay: # we have to flip!. 
            # import meshplex
            # # this clears out the overlapping. is this necessary
            # mesh_out = meshplex.MeshTri(v, f)
            # mesh_out.flip_until_delaunay()
        
            # # update v and f!
            # v = mesh_out.points.copy()
            # f = mesh_out.cells('points').copy() 
            f = igl.intrinsic_delaunay_triangulation(igl.edge_lengths(v,f), f)[1]

        if debugviz_tri:
            plt.figure(figsize=(5,5))
            plt.triplot(v[:,1],
                        v[:,2], f, 'g-', lw=.1)
            plt.show()
            
        v_steps.append(v)
        f_steps.append(f)
        area_distortion_iter.append(area_distortion_mesh) # append this. 

        # except:
        #     # if error then break
        #     return v_steps, f_steps, area_distortion_iter

    return v_steps, f_steps, area_distortion_iter


def adjacency_edge_cost_matrix(V,E, n=None):
    r""" Build the Laplacian matrix for a line given the vertices and the undirected edge-edge connections 
    
    Parameters
    ----------
    V : (n_points,d) array
        the vertices of the d-dimensional line
    E : (n_edges,2) array
        the edge connections as integer vertex indices specifying how the vertices are joined together
    n : int
        if specified, the size of the Laplacian matrix, if not the same as the number of points in V. the returned Laplacian matrix will be of dimension ((n,n))

    Returns
    -------
    C : (n,n) sparse array
        the n x n symmetric vertex laplacian matrix 

    """
    import numpy as np 
    import scipy.sparse as spsparse
    # % compute edge norms
    edge_norms = np.linalg.norm(V[E[:,0]]-V[E[:,1]], axis=-1)
    # % number of vertices
    if n is None:
        n = len(V);
    #% build sparse adjacency matrix with non-zero entries indicated edge costs
    C = spsparse.csr_matrix((edge_norms, (E[:,0], E[:,1])), shape=(n, n))
    C = C + C.transpose() # to make undirected. 
    
    return C

def adjacency_matrix(E, n=None):
    r""" Build the Laplacian matrix for a line given the undirected edge-edge connections without taking into account distances between vertices
    
    Parameters
    ----------
    E : (n_edges,2) array
        the edge connections as integer vertex indices specifying how the vertices are joined together
    n : int
        if specified, the size of the Laplacian matrix, if not the same as the number of points in V. the returned Laplacian matrix will be of dimension ((n,n))

    Returns
    -------
    C : (n,n) sparse array
        the n x n symmetric vertex laplacian matrix 

    """
    import numpy as np 
    import scipy.sparse as spsparse
    # % number of vertices
    if n is None:
        n = len(E);
    #% build sparse adjacency matrix with non-zero entries indicated edge costs
    C = spsparse.csr_matrix((np.ones(len(E)), (E[:,0], E[:,1])), shape=(n, n))
    C = C + C.transpose() # to make undirected. 
    
    return C
        
def mass_matrix2D(A):
    r""" Build the Mass matrix for a given adjacency or Laplacian matrix. The mass matrix is a diagonal matrix of the row sums of the input matrix, A
    
    Parameters
    ----------
    A : (N,N) array or sparse array 
        the Adjacency or symmetric Laplacian matrix 
    
    Returns
    -------
    M : (N,N) sparse array
        a diagonal matrix whose entries are the row sums of the symmetric input matrix 

    """
    import scipy.sparse as spsparse
    import numpy as np 
    
    vals = np.squeeze(np.array(A.sum(axis=1)))
    M = spsparse.diags(vals, 0)
    # M = M / np.max(M.diagonal())
    
    return M 


def vertex_dihedral_angle_matrix(mesh, eps=1e-12):
    r""" Build the Dihedral angle matrix for vertices given an input mesh. The dihedral angles, is the angle between the normals of pairs of vertices measures the local mesh convexity. the dihedral angle is captured as a cosime distance
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input mesh
    eps : scalar
        a small constant scalar for numerical stability 
    
    Returns
    -------
    angles_edges_matrix : (n_vertices, n_vertices) sparse array
        a matrix capturing the dihedral angle between vertex_i to vertex_j between vertex neighbors

    """
    import numpy as np 
    import igl
    import scipy.sparse as spsparse

    # we use the dihdral angle formula... 
    vertex_edge_list = igl.edges(mesh.faces)
    normals1 = mesh.vertex_normals[vertex_edge_list[:,0]].copy() ; normals1 = normals1/(np.linalg.norm(normals1, axis=-1)[:,None]  + eps)
    normals2 = mesh.vertex_normals[vertex_edge_list[:,1]].copy() ; normals2 = normals2/(np.linalg.norm(normals2, axis=-1)[:,None]  + eps)
    
    # dot product. 
    angles_edges = np.nansum(normals1 * normals2, axis=-1) # this is signed cosine.. 
    # cosine distance matrix 
    angles_edges = (1.-angles_edges) / 2. # this makes it a proper distance!. # smaller should be closer...? 

    # make into adjacency matrix
    n = len(mesh.vertices)
    angles_edges_matrix = spsparse.csr_matrix((angles_edges, (vertex_edge_list[:,0], vertex_edge_list[:,1])), 
                            shape=(n,n))
    angles_edges_matrix = angles_edges_matrix + angles_edges_matrix.transpose() # symmetric

    return angles_edges_matrix

def vertex_edge_lengths_matrix(mesh):
    r""" Build the edge distance matrix between local vertex neighbors of the input mesh 
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input mesh
    
    Returns
    -------
    D : (n_vertices, n_vertices) sparse array
        a matrix capturing the euclidean edge distance between vertex_i to vertex_j of vertex neighbors

    """
    import numpy as np 
    import igl
    import scipy.sparse as spsparse

    vertex_edge_list = igl.edges(mesh.faces) # this is unique edges hence not undirected. (upper triangular)
    # get the distance matrix between the edges. 
    dist_edges = np.linalg.norm(mesh.vertices[vertex_edge_list[:,0]] - mesh.vertices[vertex_edge_list[:,1]], 
                                axis=-1)

    # make into adjacency matrix
    n = len(mesh.vertices)
    D = spsparse.csr_matrix((dist_edges, (vertex_edge_list[:,0], vertex_edge_list[:,1])), 
                            shape=(n,n))
    D = D + D.transpose() # make symmetric! 

    return D

def vertex_edge_affinity_matrix(mesh, gamma=None):
    r""" Compute an affinity distance matrix of the edge distances. This is done by computing the pairwise edge length distances between vertex neighbors and applying a heat kernel.

    .. math:: 
        A_{dist} = \exp^{\left(\frac{-D_{dist}^2}{2\sigma^2}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input mesh
    gamma : scalar
        a scalar normalisation of the distances in the distance matrix 
    
    Returns
    -------
    A : (n_vertices, n_vertices) sparse array
        a matrix capturing the euclidean edge affinity between vertex_i to vertex_j of vertex neighbors. This is a normalised measure of distances with values mainly in the scale of [0,1]

    """
    D = vertex_edge_lengths_matrix(mesh)
    A = distance_to_heat_affinity_matrix(D, gamma=gamma)

    return A 

def vertex_dihedral_angle_affinity_matrix(mesh, gamma=None, eps=1e-12):
    r""" Compute an affinity distance matrix of the vertex dihedral angles. This is done by computing the vertex dihedral angle distances and applying a heat kernel.

    .. math:: 
        A_{angle} = \exp^{\left(\frac{-D_{angle}^2}{2\sigma^2}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input mesh
    gamma : scalar
        a scalar normalisation of the distances in the distance matrix 
    eps : scalar
        a small constant for numerical stability 
    
    Returns
    -------
    A : (n_vertices, n_vertices) sparse array
        a matrix capturing the euclidean dihedral angle cosine distance affinity between vertex_i to vertex_j of vertex neighbors. This is a normalised measure of distances with values mainly in the scale of [0,1]

    """
    D = vertex_dihedral_angle_matrix(mesh, eps=eps)
    A = distance_to_heat_affinity_matrix(D, gamma=gamma)

    return A

def vertex_geometric_affinity_matrix(mesh, gamma=None, eps=1e-12, alpha=.5, normalize=True):
    r""" Compute an affinity matrix balancing geodesic distances and convexity by taking a weighted average of the edge distance affinity matrix and the vertex dihedral angle affinity matrix. 

    .. math:: 
        A = \alpha A_{dist} + (1-\alpha) A_{dihedral}

    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input mesh
    gamma : scalar
        a scalar normalisation of the distances in the distance matrix 
    eps : scalar
        a small constant for numerical stability of the dihedral angle distance matrix 
    alpha : 0-1
        the weight for averaging the two affinity matrices
    normalize : bool
        if True, apply left normalization to the averaged affinity matrix given by :math:`M^{-1}A` where :math:`M` is the mass matrix. 

    Returns
    -------
    W : (n_vertex, n_vertex) scipy sparse matrix
        the combined average affinity matrix

    See Also
    --------
    :func:`unwrap3D.Mesh.meshtools.vertex_edge_affinity_matrix` : 
        function used to compute the vertex edge distance affinity matrix
    :func:`unwrap3D.Mesh.meshtools.vertex_dihedral_angle_affinity_matrix` : 
        function used to compute the vertex dihedral angle distance affinity matrix

    """
    import scipy.sparse as spsparse
    import numpy as np 

    Distance_matrix = vertex_edge_affinity_matrix(mesh, gamma=gamma)
    Convexity_matrix = vertex_dihedral_angle_affinity_matrix(mesh, gamma=gamma, eps=eps)

    W = alpha * Distance_matrix + (1.-alpha) * Convexity_matrix

    if normalize:
        DD = 1./W.sum(axis=-1)
        DD = spsparse.spdiags(np.squeeze(DD), [0], DD.shape[0], DD.shape[0])
        W = DD.dot(W) # this is perfect normalization. 

    return W 

def distance_to_heat_affinity_matrix(Dmatrix, gamma=None):
    r""" Convert any distance matrix to an affinity matrix by applying a heat kernel.

    .. math:: 
        A = \exp^{\left(\frac{-D^2}{2\sigma^2}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.
    
    Parameters
    ----------
    Dmatrix : (N,N) sparse array
        a scipy.sparse input distance matrix
    gamma : scalar
        the normalisation scale factor of distances

    Returns 
    -------
    A : (N,N) sparse array
        a scipy.sparse output affinity distance matrix

    """
    import numpy as np 
    import igl
    import scipy.sparse as spsparse

    l = Dmatrix.shape[0]
    A = Dmatrix.copy()
    if gamma is None:
        sigma_D = np.mean(A.data)
    else:
        sigma_D = gamma
    den_D = 2 * (sigma_D ** 2)
    np.exp( -A.data**2/den_D, out=A.data )
    A = A + spsparse.diags(np.ones(l), 0)  # diagonal is 1 by definition. 

    return A 


def conformalized_mean_line_flow( contour_pts, E=None, close_contour=True, fixed_boundary = False, lambda_flow=1000, niters=10, topography_edge_fix=False, conformalize=True):
    r""" Conformalized mean curvature flow of a curve, also known as the isoperimetric flow.

    This function is adapted from the Matlab GPToolbox 

    Parameters
    ----------
    contour_pts : (n_points,d) array
        the list of coordinates of the line 
    E : (n_edges,2) array
        the edge connectivity of points on the line 
    close_contour : bool
        if True and E is None, construct the edge connectivity assuming the order of the given contour_pts and connecting the last point to the 1st point. If False and E is None, the order of the given contour_pts is still assumed but the last point to the 1st point is not connected by an edge
    fixed_boundary : bool
        if True, the ends of the contour_pts is not involved but is pinned to its original position. Only the interior points are updated
    lambda_flow : scalar
        controls the stepsize of the evolution per iteration. Smaller values given less movement
    niters : int
        the number of iterations to run 
    topography_edge_fix : bool
        this is only relevant for :func:`unwrap3D.Mesh.meshtools.conformalized_mean_curvature_flow_topography` or lines coming from topographic boundaries where we wish to remove all flow in other directions except that in the depth axis at the boundary. 
    conformalize : bool 
        if True, the Laplacian matrix is not recomputed at every iteration. 
    
    Returns
    -------
    contour_pts_flow : (n_points,d,niters+1)
        the list of coordinates of the line at each iteration including the initial position. The edge connectivity is the same as input 
    
    """
    # Fix boundary will find all degree = 1 nodes and make their laplacian 0 ---> inducing no flow. and therefore returning the identity  
    import numpy as np 
    import scipy.sparse as spsparse

    if E is None:
        if close_contour:
            E = [np.arange(len(contour_pts)), 
                 np.hstack([np.arange(len(contour_pts))[1:], 0])]
        else:
            E = [np.arange(len(contour_pts))[:-1], 
                 np.arange(len(contour_pts))[1:]]
        E = np.vstack(E).T

    A = adjacency_edge_cost_matrix(contour_pts, E)
    L = A-spsparse.diags(np.squeeze(np.array(A.sum(axis=1)))); # why is this so slow? 

    if fixed_boundary:
        boundary_nodes = np.arange(len(contour_pts))[A.sum(axis=1) == 1]
        L[boundary_nodes,:] = 0 # slice this in. 
    
    # so we need no flux boundary conditions to prevent flow in x,y at the boundary!....----> one way is to do mirror...( with rectangular grid this is easy... but with triangle is harder...)
    contour_pts_flow = [contour_pts]
    for iter_ii in np.arange(niters):
        A = adjacency_edge_cost_matrix(contour_pts_flow[-1], E)
        if conformalize ==False:
            L = A-spsparse.diags(np.squeeze(np.array(A.sum(axis=1))));
            if fixed_boundary:
                boundary_nodes = np.arange(len(contour_pts))[A.sum(axis=1) == 1]
                L[boundary_nodes,:] = 0 # slice this in. 
        M = mass_matrix2D(A)
        # # unfixed version of the problem 
        # vvv = spsparse.linalg.spsolve(M-lambda_flow*L, M.dot(boundary_mesh_pos[-1]))
        if topography_edge_fix:
            rhs_term =  -lambda_flow*L.dot(contour_pts_flow[-1]) # this balances the flow. # the x-y plane has normal with z. # so we just the opposite.
            rhs_term[:,0] = 0 # ok this is correct - this blocks all into plane flow. but what if we relax this.... -> permit just not orthogonal....
            vvv = spsparse.linalg.spsolve(M-lambda_flow*L, M.dot(contour_pts_flow[-1])  + rhs_term)
        else:
            vvv = spsparse.linalg.spsolve(M-lambda_flow*L, M.dot(contour_pts_flow[-1]))
        
        contour_pts_flow.append(vvv)
    contour_pts_flow = np.array(contour_pts_flow)
    contour_pts_flow = contour_pts_flow.transpose(1,2,0)

    return contour_pts_flow


def conformalized_mean_curvature_flow(mesh, max_iter=50, delta=5e-4, rescale_output = True, min_diff = 1e-13, conformalize = True, robust_L =False, mollify_factor=1e-5):
    r""" Conformalized mean curvature flow of a mesh of Kazhdan et al. [1]_

    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input 3D mesh  
    max_iter : int
        the number of iterations
    delta : scalar
        controls the stepsize of the evolution per iteration. Smaller values gives less deformation
    rescale_output : bool
        if False will return a surface area normalised mesh instead. if True return the mean curvature flow surfaces at the same scale as the input mesh
    min_diff : 
        not used for now 
    conformalize : 
        if True, the Laplacian matrix is not recomputed at every iteration. 
    robust_L : bool
        if True, uses the robust Laplacian construction of Sharpe et al. [2]_. If False, the standard cotan Laplacian is used. The robust Laplacian enables proper handling of degenerate and nonmanifold vertices such as that if using the triangle mesh constructed from a uv image grid. The normal 3D mesh if remeshed does not necessarily need this
    mollify_factor : scalar
        the mollification factor used in the robust Laplacian. see https://github.com/nmwsharp/robust-laplacians-py    
    
    Returns
    -------
    Usteps : (n_points,3,niters+1) array
        an array of the vertex coordinates of the mesh at each iteration
    F : (n_faces,3) array
        the face connectivity of the mesh 
    flow_metrics_dict : dict
        a dict of various statistical measures of the flow 
        
        'mean_curvature_iter' : array
            mean of absolute values of mean curvature per face per iteration 
        'max_curvature_iter' : array
            maximum of absolute values of mean curvature per face per iteration 
        'gauss_curvature_iter' : array
            mean of absolute values of Gaussian curvature per face per iteration 
        'canonical_c_all' : array
            array of the computed face area weighted centroid per iteration with respect to an area normalised mesh
        'canonical_area_all' : array 
            array of the total surface area used for area normalising per iteration
        'flow_d_all' : array
            matrix norm difference between current and previous vertex coordinate positions
        'V0_max' : scalar
            the maximum scalar value over all coordinate values used to initially scale the mesh vertices

    References
    ----------
    .. [1] Kazhdan, Michael, Jake Solomon, and Mirela Ben‐Chen. "Can mean‐curvature flow be modified to be non‐singular?." Computer Graphics Forum. Vol. 31. No. 5. Oxford, UK: Blackwell Publishing Ltd, 2012.
    .. [2] Sharp, Nicholas, and Keenan Crane. "A laplacian for nonmanifold triangle meshes." Computer Graphics Forum. Vol. 39. No. 5. 2020.
    
    """
    """
    input is a trimesh mesh object with vertices, faces etc. 
        delta: the step size of diffusion. 
    """
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from tqdm import tqdm 
    if robust_L:
        import robust_laplacian

    V = mesh.vertices 
    # V = V - np.mean(V, axis=0) # center this 
    F = mesh.faces

    # or the robust version (tufted -> see Keenan Crane) # sign is inversed..... from IGL. 
    if robust_L:
        L, M = robust_laplacian.mesh_laplacian(np.array(V), np.array(F), mollify_factor=mollify_factor); 
        L = -L 
    else:
        # first build the laplacian.  
        L = igl.cotmatrix(V,F) # this is negative semi-definite 
    
    # make a copy of the initial vertex, face
    SF = F.copy() # make a copy that is the initial faces connectivity. 
    V0 = V.copy();
    # V0_max = float(np.abs(V).max())
    V0_max = float(V.max())
    # V0_max = 1

    # initialise the save array. 
    Usteps = np.zeros(np.hstack([V.shape, max_iter+1]))
    
    # first step. 
    U = V.copy();
    # pre-normalize for stability. 
    U = U / V0_max # pre-divided by U.max().... here so we are in [0,1] # this is best for image-derived meshes.. not for scanned meshes? 
    Usteps[...,0] = U.copy() # initialise. 
    
    # save various curvature measures of intermediate. 
    curvature_steps = [] 
    max_curvature_steps = [] 
    gauss_curvature_steps = []
    
    # these must be applied iteratively in order to reconstitute the actual size ? 
    c_all = []
    area_all = []
    d_all = []
    c0 = np.hstack([0,0,0])
    area0 = 1.
    
    for ii in tqdm(range(max_iter)):

        # iterate. 
        U_prev = U.copy(); # make a copy. 
        # % 'full' seems slight more stable than 'barycentric' which is more stable
        # % than 'voronoi'
        
        if not robust_L:
            M = igl.massmatrix(U, F, igl.MASSMATRIX_TYPE_BARYCENTRIC) # -> this is the only matrix that doesn't degenerate. 
        else:
            L_, M = robust_laplacian.mesh_laplacian(np.array(U), np.array(F), mollify_factor=mollify_factor); L_ = -L_
        # M = igl.massmatrix(U, F, igl.MASSMATRIX_TYPE_FULL) # what about this ? 
        # %     M = massmatrix(U,F,'full');
        if conformalize==False:
            if not robust_L:
                # L = laplacian(V,F);
                L = igl.cotmatrix(U,F) # should be recomputation. 
            else:
                L = L_.copy()
            
        # # implicit solve. 
        S = (M-delta*L) # what happens when we invert? 
        b = M.dot(U)
        # b = U.copy()

        # # Solve # compare with spsolve. # best way to solve? ---> S is symmetric therefore we can do splu then solve.---> which should be really fast!.
        # u1,xx = spsparse.linalg.bicgstab(S, b[:,0]) # not that bad... 
        # u2,yy = spsparse.linalg.bicgstab(S, b[:,1]) 
        # u3,zz = spsparse.linalg.bicgstab(S, b[:,2])
        # u1,xx = spsparse.linalg.cg(S, b[:,0], maxiter=100) # not that bad... 
        # u2,yy = spsparse.linalg.cg(S, b[:,1], maxiter=100) 
        # u3,zz = spsparse.linalg.cg(S, b[:,2], maxiter=100)
        # u1,xx = spsparse.linalg.bicg(S, b[:,0], maxiter=100) # not that bad... 
        # u2,yy = spsparse.linalg.bicg(S, b[:,1], maxiter=100) 
        # u3,zz = spsparse.linalg.bicg(S, b[:,2], maxiter=100)
        # U = np.vstack([u1,u2,u3]).T
        U = spsparse.linalg.spsolve(S,b)

        if np.sum(np.isnan(U[:,0])) > 0:
            # if we detect nan, no good. 
            break 
        else:
        
            # canonical centering is a must to stabilize -> essentially affects a scaling + translation.  
            # rescale by the area? # is this a mobius transformation? ( can)
            area = np.sum(igl.doublearea(U,SF)*.5) #total area. 
            c = np.sum((0.5*igl.doublearea(U,SF)/area)[...,None] * igl.barycenter(U,F), axis=0) # this is just weighted centroid
            U = U - c[None,:] 
            U = U/(np.sqrt(area)) # avoid zero area
            d = ((((U-U_prev).T).dot(M.dot(U-U_prev)))**2).diagonal().sum() # assessment of convergence. 

            d_all.append(d)
            # append key parameters. 
            c_all.append(c)
            area_all.append(area)
            
            """
            Compute assessments of smoothness.
            """
            # compute the mean curvature # use the principal curvature computation. 
            # ll = igl.cotmatrix(U, F)
            # mm = igl.massmatrix(U, F, igl.MASSMATRIX_TYPE_VORONOI)
            # minv = spsparse.diags(1 / mm.diagonal())
            # hn = -minv.dot(ll.dot(U))
            # h = np.linalg.norm(hn, axis=1)

            # more robust:
            _, _, k1, k2 = igl.principal_curvature(U, F)
            h2 = 0.5 * (k1 + k2)
            curvature_steps.append(np.nanmean(np.abs(h2)))
            max_curvature_steps.append(np.nanmax(np.abs(h2)))
            
            # this is the best. 
            kk = igl.gaussian_curvature(U, F)
            gauss_curvature_steps.append(np.nanmean(np.abs(kk)))
            
            if rescale_output: 
                # x area + c
                c0 = c0 + c
                area0 = area0*np.sqrt(area)
                Usteps[...,ii+1] = ((U * area0) + c0[None,:]).copy() # since it is just iterative... 
            else:
                Usteps[...,ii+1] = U.copy() # copy the current iteration into it. 

    c_all = np.vstack(c_all)
    area_all = np.hstack(area_all)
    Usteps = Usteps[..., :len(area_all)+1]
        
    if rescale_output == False:
        # area = np.sum(igl.doublearea(Usteps[...,0],SF)*0.5); # original area. 
        # c = np.sum((0.5*igl.doublearea(Usteps[...,0],SF)/area)[...,None] * igl.barycenter(Usteps[...,0],SF), axis=0);
        # below is more easy to get back 
        area = area_all[0] # this is wrong? # -> this might be the problem ... we should have the same as number of iterations. 
        c = c_all[0]
        # print(area, c)
        # if nargout > 1
        for iteration in range(Usteps.shape[-1]-1):
            Usteps[:,:,iteration+1] = Usteps[:,:,iteration+1]*np.sqrt(area);
            Usteps[:,:,iteration+1] = Usteps[:,:,iteration+1] + c[None,:]

    # finally multiply all Usteps by umax
    Usteps = Usteps * V0_max

    # to save the key parameters that we will make use of .... 
    flow_metrics_dict = {'mean_curvature_iter': curvature_steps, 
                         'max_curvature_iter': max_curvature_steps, 
                         'gauss_curvature_iter': gauss_curvature_steps, 
                         'canonical_c_all': c_all, 
                         'canonical_area_all': area_all, 
                         'flow_d_all': d_all,
                         'V0_max': V0_max}
    
    return Usteps, F, flow_metrics_dict


def conformalized_mean_curvature_flow_topography(mesh, 
                                                 max_iter=50, 
                                                 delta=5e-4, 
                                                 min_diff = 1e-13, 
                                                 conformalize = True, 
                                                 robust_L =False, 
                                                 mollify_factor=1e-5):

    r""" Adapted conformalized mean curvature flow of a mesh of Kazhdan et al. [1]_ to allow for topographic meshes such that iterative applications flattens the topography to the plane.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        the input topography mesh 
    max_iter : int
        the number of iterations
    delta : scalar
        controls the stepsize of the evolution per iteration. Smaller values gives less deformation
    min_diff : 
        not used for now 
    conformalize : 
        if True, the Laplacian matrix is not recomputed at every iteration. 
    robust_L : bool
        if True, uses the robust Laplacian construction of Sharpe et al. [2]_. If False, the standard cotan Laplacian is used. The robust Laplacian enables proper handling of degenerate and nonmanifold vertices such as that if using the triangle mesh constructed from a uv image grid. The normal 3D mesh if remeshed does not necessarily need this
    mollify_factor : scalar
        the mollification factor used in the robust Laplacian. see https://github.com/nmwsharp/robust-laplacians-py    
    
    Returns
    -------
    Usteps : (n_points,3,niters+1) array
        an array of the vertex coordinates of the mesh at each iteration
    F : (n_faces,3) array
        the face connectivity of the mesh 
    flow_metrics_dict : dict
        a dict of various statistical measures of the flow 
        
        'mean_curvature_iter' : array
            mean of absolute values of mean curvature per face per iteration 
        'max_curvature_iter' : array
            maximum of absolute values of mean curvature per face per iteration 
        'gauss_curvature_iter' : array
            mean of absolute values of Gaussian curvature per face per iteration 
        'canonical_c_all' : array
            array of the computed face area weighted centroid per iteration with respect to an area normalised mesh
        'canonical_area_all' : array 
            array of the total surface area used for area normalising per iteration
        'flow_d_all' : array
            matrix norm difference between current and previous vertex coordinate positions
        'V0_max' : scalar
            the maximum scalar value over all coordinate values used to initially scale the mesh vertices

    References
    ----------
    .. [1] Kazhdan, Michael, Jake Solomon, and Mirela Ben‐Chen. "Can mean‐curvature flow be modified to be non‐singular?." Computer Graphics Forum. Vol. 31. No. 5. Oxford, UK: Blackwell Publishing Ltd, 2012.
    .. [2] Sharp, Nicholas, and Keenan Crane. "A laplacian for nonmanifold triangle meshes." Computer Graphics Forum. Vol. 39. No. 5. 2020.
    
    """
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from tqdm import tqdm 
    if robust_L:
        import robust_laplacian

    V = mesh.vertices 
    F = mesh.faces

    # parsing the boundary loop -> we assume largely a simple surface with just the 1 major boundary
    b = igl.boundary_loop(F) # boundary 
    v_b = np.unique(b) # this is the unique vertex indices. 
    # build edge connectivity matrix in original index. 
    E = [b, 
         np.hstack([b[1:], b[0]])]
    E = np.vstack(E).T

    ## List of all vertex indices
    v_all = np.arange(V.shape[0])
    ## List of interior indices
    v_in = np.setdiff1d(v_all, v_b)

    # build the laplacian for the boundary loop. 
    A_b = adjacency_edge_cost_matrix( V, E, n=len(v_all))
    L_b = A_b-spsparse.diags(np.squeeze(np.array(A_b.sum(axis=1))));  

    # or the robust version (tufted -> see Keenan Crane) # sign is inversed..... from IGL. 
    if robust_L:
        L, M = robust_laplacian.mesh_laplacian(np.array(V), np.array(F), mollify_factor=mollify_factor); 
        L = -L 
    else:
        # first build the laplacian.  
        L = igl.cotmatrix(V,F) # this is negative semi-definite 
    L[v_b] = L_b[v_b] # slice in is allowed. wow.
    
    # make a copy of the initial vertex, face
    SF = F.copy() # make a copy that is the initial faces connectivity. 
    V0 = V.copy();
    # better to do area norm straight up!. 
    # V0_max = float(np.sum(igl.doublearea(V0,SF)*.5)) # rescale - assume this is image based
    # V0_max = float(V.max())
    V0_max = float(1.)
    # initialise the save array. 
    Usteps = np.zeros(np.hstack([V.shape, max_iter+1]))
    
    # first step. 
    U = V.copy();
    # pre-normalize for stability. # actually might not be the case but will help convergence. 
    # U = U / V0_max # pre-divided by U.max().... here so we are in [0,1] # this is best for image-derived meshes.. not for scanned meshes? 
    U[:,0] = U[:,0] - np.nanmean(V[:,0])
    Usteps[...,0] = U.copy() # initialise. 
    
    # save various curvature measures of intermediate. 
    curvature_steps = [] 
    max_curvature_steps = [] 
    gauss_curvature_steps = []
    
    # these must be applied iteratively in order to reconstitute the actual size ? 
    c_all = []
    area_all = []
    d_all = []
    c0 = np.hstack([0,0,0])
    area0 = 1.
    
    for ii in tqdm(range(max_iter)):

        # iterate. 
        U_prev = U.copy(); # make a copy. 
        # % 'full' seems slight more stable than 'barycentric' which is more stable
        # % than 'voronoi'
        
        if not robust_L:
            M = igl.massmatrix(U, F, igl.MASSMATRIX_TYPE_BARYCENTRIC) # -> this is the only matrix that doesn't degenerate. 
            M = M.tocsr()
        else:
            L_, M = robust_laplacian.mesh_laplacian(np.array(U), np.array(F), mollify_factor=mollify_factor); L_ = -L_
            M = M.tocsr()

        A_b = adjacency_edge_cost_matrix(U, E, n=len(v_all))
        M_b = mass_matrix2D(A_b); M_b = M_b.tocsr()
        M[v_b,:] = M_b[v_b,:]

        # M = igl.massmatrix(U, F, igl.MASSMATRIX_TYPE_FULL) # what about this ? 
        # %     M = massmatrix(U,F,'full');
        if conformalize==False:
            if not robust_L:
                # L = laplacian(V,F);
                L = igl.cotmatrix(U,F) # should be recomputation. 
            else:
                L = L_.copy()

            L_b = A_b-spsparse.diags(np.squeeze(np.array(A_b.sum(axis=1))));  
            L[v_b] = L_b[v_b] # slice in is allowed. wow.
            
        # # implicit solve. 
        S = (M-delta*L) # what happens when we invert? 
        b = M.dot(U)

        """
        add additional boundary term on right
        """
        b_bnd_all = -delta*L.dot(U)
        b_bnd = np.zeros(b_bnd_all.shape)
        b_bnd[v_b,1:] = b_bnd_all[v_b,1:] # this generically destroys all x-y plane deformation. 
        
        # # do prefactorisation? 
        # S = S.tocsc()
        # # S_ = spsparse.linalg.splu(S)
        U = spsparse.linalg.spsolve(S, b + b_bnd)
        # U = S_.solve(b + b_bnd)

        if np.sum(np.isnan(U[:,0])) > 0:
            # if we detect nan, no good. 
            break 
        else:
            # canonical centering is a must to stabilize -> essentially affects a scaling + translation.  
            # rescale by the area? # is this a mobius transformation? ( can)
            area = np.sum(igl.doublearea(U,SF)*.5) #total area. 
            # c = np.sum((0.5*igl.doublearea(U,SF)/area)[...,None] * igl.barycenter(U,F), axis=0) # this is just weighted centroid
            # U = U - c[None,:] 
            # U = U/(np.sqrt(area)) # avoid zero area
            d = ((((U-U_prev).T).dot(M.dot(U-U_prev)))**2).diagonal().sum() # assessment of convergence. 
            # U = U*(np.sqrt(area)) 

            d_all.append(d)
            # append key parameters. 
            # c_all.append(c)
            area_all.append(area)
            
            """
            Compute assessments of smoothness.
            """
            # # compute the mean curvature
            # ll = igl.cotmatrix(U, F)
            # mm = igl.massmatrix(U, F, igl.MASSMATRIX_TYPE_VORONOI)
            # minv = spsparse.diags(1 / mm.diagonal())
            # hn = -minv.dot(ll.dot(U))
            # h = np.linalg.norm(hn, axis=1)
            # curvature_steps.append(np.nanmean(np.abs(h)))
            # max_curvature_steps.append(np.nanmax(np.abs(h)))
            
            # more robust:
            _, _, k1, k2 = igl.principal_curvature(U, F)
            h2 = 0.5 * (k1 + k2)
            curvature_steps.append(np.nanmean(np.abs(h2)))
            max_curvature_steps.append(np.nanmax(np.abs(h2)))
            
            # this is the best. 
            kk = igl.gaussian_curvature(U, F)
            gauss_curvature_steps.append(np.nanmean(np.abs(kk)))
            
            # if rescale_output: 
            #     # # x area + c
            #     # c0 = c0 + c
            #     area0 = area0*np.sqrt(area)
            #     Usteps[...,ii+1] = U * area0 #+ c0[None,:]).copy() # since it is just iterative... 
            # else:
            #     Usteps[...,ii+1] = U.copy() # copy the current iteration into it. 
            Usteps[...,ii+1] = U.copy()

    # c_all = np.vstack(c_all)
    area_all = np.hstack(area_all)
    Usteps = Usteps[..., :len(area_all)+1]
        
    # if rescale_output == False:
    #     # area = np.sum(igl.doublearea(Usteps[...,0],SF)*0.5); # original area. 
    #     # c = np.sum((0.5*igl.doublearea(Usteps[...,0],SF)/area)[...,None] * igl.barycenter(Usteps[...,0],SF), axis=0);
    #     # below is more easy to get back 
    #     area = area_all[0] # this is wrong? # -> this might be the problem ... we should have the same as number of iterations. 
    #     # c = c_all[0]
    #     # print(area, c)
    #     # if nargout > 1
    #     for iteration in range(Usteps.shape[-1]-1):
    #         Usteps[:,:,iteration+1] = Usteps[:,:,iteration+1]*np.sqrt(area);
    #         # Usteps[:,:,iteration+1] = Usteps[:,:,iteration+1] + c[None,:]

    # # finally multiply all Usteps by umax
    # Usteps = Usteps * V0_max
    Usteps[:,0] = Usteps[:,0] + np.nanmean(V[:,0])

    # to save the key parameters that we will make use of .... 
    flow_metrics_dict = {'mean_curvature_iter': curvature_steps, 
                         'max_curvature_iter': max_curvature_steps, 
                         'gauss_curvature_iter': gauss_curvature_steps, 
                         'canonical_c_all': c_all, 
                         'canonical_area_all': area_all, 
                         'flow_d_all': d_all,
                         'V0_max': V0_max}
    
    return Usteps, F, flow_metrics_dict


def recover_img_coordinate_conformal_mean_flow(Usteps, flow_dict):
    r""" This function reverses the iterative, centroid computation, subtraction and area normalisation in the conformalized mean curvature flow of a general 3D mesh, :func:`unwrap3D.Mesh.meshtools.conformalized_mean_curvature_flow`
    
    Parameters
    ----------
    Usteps : (n_vertices,3,niters) array
        the area normalised vertex positions of a 3D mesh for every iteration of the conformalized mean curvature flow. 1st output of :func:`unwrap3D.Mesh.meshtools.conformalized_mean_curvature_flow` computed with rescale_output=False
    flow_dict : 
        the statistical measures of a mesh deformed under conformalized mean curvature flow. 3rd output of :func:`unwrap3D.Mesh.meshtools.conformalized_mean_curvature_flow`
        
        'mean_curvature_iter' : array
            mean of absolute values of mean curvature per face per iteration 
        'max_curvature_iter' : array
            maximum of absolute values of mean curvature per face per iteration 
        'gauss_curvature_iter' : array
            mean of absolute values of Gaussian curvature per face per iteration 
        'canonical_c_all' : array
            array of the computed face area weighted centroid per iteration with respect to an area normalised mesh
        'canonical_area_all' : array 
            array of the total surface area used for area normalising per iteration
        'flow_d_all' : array
            matrix norm difference between current and previous vertex coordinate positions
        'V0_max' : scalar
            the maximum scalar value over all coordinate values used to initially scale the mesh vertices

    Returns
    -------
    Usteps_ : (n_vertices,3,niters) array
        the reconstructed image vertex positions of a 3D mesh for every iteration of the conformalized mean curvature flow, identical to if :func:`unwrap3D.Mesh.meshtools.conformalized_mean_curvature_flow` had been run with rescale_output=True
    """
    import numpy as np 
    import pylab as plt 
    # in the default case, all the secondary points have recovered 
    sqrt_area_all = np.sqrt(np.hstack(flow_dict['canonical_area_all'])) # this is not such an easy scaling factor..... 
    c_all = np.vstack(flow_dict['canonical_c_all'])
    U_max = flow_dict['V0_max']

    cum_area_all = np.cumprod(sqrt_area_all) # this should be cumulative. 
    cum_c_all = np.cumsum(c_all, axis=0).T

    Usteps_ = Usteps / U_max # first divide this. 
    Usteps_[...,2:] = (Usteps_[...,2:] - cum_c_all[:,0][None,:,None]) / cum_area_all[0] # remove... 
    Usteps_[...,2:] = (Usteps_[...,2:] * (cum_area_all[1:][None,None,:])) + cum_c_all[None,:,1:] # multiply first.
    Usteps_ = Usteps_*U_max

    return Usteps_


def relax_mesh( mesh_in, relax_method='CVT (block-diagonal)', tol=1.0e-5, n_iters=20, omega=1.):
    r""" wraps the optimesh library to perform mesh relaxation with Delaunay edge flipping. This is done mainly to improve triangle quality to improve the convergence of linear algebra solvers 
    
    Parameters
    ----------
    mesh_in : trimesh.Trimesh
        input mesh to relax
    relax_method : str
        any of those compatible with optimesh.optimize()
    tol : scalar
        sets the accepted tolerance for convergence
    n_iters : int
        the maximum number of iterations
    omega : scalar
        controls the stepping size, smaller values will have better accuracy but slower convergence.

    Returns 
    -------
    mesh : trimesh.Trimesh
        output relaxed mesh with no change in the number of vertices but with changes to faces
    opt_result : optimization results
        returns tuple of number of iterations and difference 
    mean_mesh_quality : scalar
        mean triangle quanlity, where trangle quality is 2*inradius/circumradius

    """
    import optimesh
    import trimesh
    import meshplex 

    points = mesh_in.vertices.copy()
    cells = mesh_in.faces.copy()

    mesh = meshplex.MeshTri(points, cells)
    opt_result = optimesh.optimize(mesh, relax_method, tol=tol, max_num_steps=n_iters, omega=omega)
    mean_mesh_quality = mesh.q_radius_ratio.mean()

    # give back the mesh. 
    mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells('points'), validate=False, process=False) # suppress any other processing to keep the correct lookup.

    return mesh, opt_result, mean_mesh_quality


def smooth_scalar_function_mesh(mesh, scalar_fn, exact=False, n_iters=10, weights=None, return_weights=True, alpha=0):
    r""" wraps the optimesh library to perform mesh relaxation with Delaunay edge flipping. This is done mainly to improve triangle quality to improve the convergence of linear algebra solvers 
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh
    scalar_fn : (n_vertices,d) array
        the d-dimensional scalar values to spatially smooth over the mesh  
    exact : bool
        if True, evaluates the normalised :math:`M^-1 L` where :math:`M` is the mass and :math:`L` is the cotan Laplacian matrices respectively and uses this as the weights for spatial smoothing. The inversion is very slow. Setting to False, use the cotan Laplacian to smooth. 
    n_iters : int
        the maximum number of iterations
    weights : (n_vertices,n_vertices) sparse array
        user-specified vertex smoothing matrix. This overrides the options of ``exact``. 
    return_weights : bool
        return additionally the weights matrix used as a second optional output
    alpha : 0-1 scalar
        not used at present

    Returns 
    -------
    scalar0 : (n_vertices,d) array
        the spatially smoothed d-dimensional scalar values over the mesh 
    weights : (n_vertices,n_vertices) sparse array
        if return_weights=True, the weights used is returned for convenience if the user wishes to reuse the weights e.g. to avoid recomputing.

    """
    """
    If exact we use M-1^L else we use the cotangent laplacian which is super fast with no inversion required. 
    allow optional usage of weights. 
    """
    import igl 
    import time
    import numpy as np 
    import scipy.sparse as spsparse
    
    v = mesh.vertices.copy()
    f = mesh.faces.copy()

    scalar0 = scalar_fn.copy()

    # # if ii ==0:
    L = igl.cotmatrix(v,f)
    M = igl.massmatrix(v, 
                       f, 
                       igl.MASSMATRIX_TYPE_BARYCENTRIC) # -> this is the only matrix that doesn't degenerate.            
    
    if exact:
        if weights is None:
            # t1 = time.time()
            weights = spsparse.linalg.spsolve(M, L) # m_inv_l # this inversion is slow.....  the results for me is very similar....  but much slower!. 
            # print('matrix inversion for weights in: ', time.time() - t1,'s') # 65s? 
    else:
        if weights is None:
            weights = L
        
    for iteration in np.arange(n_iters):
        
        scalar0_next =  (np.squeeze((np.abs(weights)).dot(scalar0)) / (np.abs(weights).sum(axis=1))) 
        scalar0_next = np.array(scalar0_next).ravel()
        scalar0_next = scalar0_next.reshape(scalar_fn.shape)
        scalar0 = scalar0_next.copy()
        # print(kappa0_next.shape)
        
    scalar0 = np.array(scalar0)
    # case to numpy array and return
    if return_weights:
        return scalar0, weights
    else:
        return scalar0


def connected_components_mesh(mesh, original_face_indices=None, engine=None, min_len=1):
    r""" find the connected face components given a mesh or submesh
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh, 
    original_face_indices : (N,) array
        if provided these should be the original face indices that mesh.faces come from. The final connected components will be re-indexed with the provided original_face_indices. 
    engine : str
        Which graph engine to use ('scipy', 'networkx')
    min_len : int
        the minimum number of faces in a connected component. Those with a number of faces that these will be dropped

    Returns 
    -------
    components : list of arrays
        list of an array of face indices forming a connected component 

    """
    import trimesh
    import numpy as np
    
    f = mesh.faces.copy()
    components = trimesh.graph.connected_components(
                edges=mesh.face_adjacency,
                nodes=np.arange(len(f)),
                min_len=min_len, # this makes sure we don't exclude everything!. by setting this to 1!!!! c.f. trimesh definitions. 
                engine=engine)

    # if the original_face_indices is provided, then the mesh was formed as a subset of the triangles and we want to get this back!. 
    if original_face_indices is not None:
        components = [original_face_indices[cc] for cc in components] # remap to the original indexing

    return components


def get_k_neighbor_ring(mesh, K=1, stateful=False):
    r""" Find all vertex indices within a K-ring neighborhood of individual vertices of a mesh. For a vertex its K-Neighbors are all those of maximum length K edges away. The result is return as an adjacency list
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh, 
    K : int
        the maximum distance of a topological neighbor defined by the number of edge-hops away. 
    stateful : bool
        if True, returns list of adjacency lists for all neighborhoods k=1..to..K. if False, only the asked for K neighborhood is returned.
    
    Returns 
    -------
    k_rings : (n_vertex long adjacency list) or (list of n_vertex long adjacency list)
        if stateful=False, one adjacency list for the specified K of length n_vertex is returned else all list of adjacency lists for all neighborhoods k=1..to..K is provided

    """
    # by default we only compute the 1_ring
    # if return_sparse_matrix then map tp adjacency ..... 
    # if stateful return list if k>1 

    import networkx as nx 
    import numpy as np 
    
    g = nx.from_edgelist(mesh.edges_unique) # build edges .
    one_ring = [np.array(list(g[i].keys())) for i in range(len(mesh.vertices))] # one neighborhood.... 
    
    if K > 1:
        k_rings = [one_ring]
        iters = K-1
        for iter_ii in np.arange(iters):
            k_ring = [np.unique(np.hstack([one_ring[rrr] for rrr in (k_rings[-1])[rr]])) for rr in np.arange(len(one_ring))]    
            k_ring = [np.unique(np.hstack([(k_rings[-1])[rr_ii], k_ring[rr_ii]])) for rr_ii in np.arange(len(k_ring))] # make sure the previous was covered. 
            
            k_rings.append(k_ring)
            
        if stateful:
            return k_rings
        else:
            k_rings = list(k_rings[-1])
            return k_rings
    else:
        k_rings = list(one_ring)
        return k_rings


# # edge_list to sparse matrix. ?
# def edge_list_to_matrix(edge_list):
#     r""" Function to convert a given list or array of edge connections to an adjacency matrix where 1 denotes the presence of an edge between 
#     """
#     import scipy.sparse as spsparse    
#     import numpy as np 
    
#     row_inds = np.hstack([[ii]*len(edge_list[ii]) for ii in len(edge_list)])
#     col_inds = np.hstack(edge_list)
#     # crucial third array in python, which can be left out in r
#     ones = np.ones(len(row_inds), np.uint32)
#     matrix = spsparse.csr_matrix((ones, (row_inds, col_inds)))
    
#     return matrix # useful for clustering applications. and for saving .... 


def find_central_ind(v, vertex_components_indices):
    r""" Given a list of mesh patches in the form of vertex indices of the mesh, find the index on the mesh surface for each patch that is closest to the patch centroid. This assumes that each patch is local such that the patch centroid is covered by the convex hull.
        
    Parameters
    ----------
    v : (n_vertex,3)
        vertices of the full 3D mesh
    vertex_components_indices : list of arrays 
        list of patches, where each patch is given as vertex indices into ``v``

    Returns
    -------
    central_inds : (n_components,)
        an array the same number as the given vertex patches/components given the index into ``v`` closest to the geometrical centroid of the vertex component

    """
    import numpy as np 
    
    central_inds = []
    for component_inds in vertex_components_indices:
        verts = v[component_inds].copy()
        mean_verts = np.nanmean(verts, axis=0)
        central_inds.append(component_inds[np.argmin(np.linalg.norm(verts - mean_verts[None,:]))])
    central_inds = np.hstack(central_inds)
    
    return central_inds

def compute_geodesic_sources_distance_on_mesh(mesh, source_vertex_indices, t_diffuse=.5, return_solver=True, method='heat'):
    r""" Compute the geodesic distance of all vertex points on a mesh to given sources using the fast approximate vector heat method of Crane et al. [1]_, [2]_ or using exact Djikstras shortest path algorithm

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh 
    source_vertex_indices : list of arrays
        List of individual 'sources' specifying multiple sources to compute distance from. Sources are vertex points where the geodesic distance are 0. The vector heat method, method='heat' allows arrays to describe individual sources as non-point like. if method='exact', multipoint sources are reduced to a central single-point source   
    t_diffuse : scalar
        set the time used for short-time heat flow in method='heat'. Larger values may make the solution more stable at the cost of over-smoothing
    return_solver : bool
        if True, return the solver used in the vector heat method, see https://github.com/nmwsharp/potpourri3d
    method : str
        specifies the geodesic distance computation method. either of 'heat' for vector heat or 'exact' for Djikstra
    
    Returns
    -------
    if method == 'heat' :
        geodesic_distances_components : (n_sources, n_vertex) array
            the geodesic distance of each vertex point of the mesh to each of the sources specified in ``source_vertex_indices``
        solver : potpourri3d MeshHeatMethodDistanceSolver instance 
            if return_solver=True, return the vector heat solver used for geodesic distance computation 

    if method == 'exact' :
        distances : (n_vertex,) array
            an array of the shortest geodesic distance to any source for each vertex
        best_source : (n_vertex,) array
            the id of the closest source for each vertex

    References
    ----------
    .. [1] Crane, Keenan, Clarisse Weischedel, and Max Wardetzky. "The heat method for distance computation." Communications of the ACM 60.11 (2017): 90-99.
    .. [2] Sharp, Nicholas, Yousuf Soliman, and Keenan Crane. "The vector heat method." ACM Transactions on Graphics (TOG) 38.3 (2019): 1-19.

    """
    import igl # igl version is incredibly slow!. use nick sharps potpourri3D! 
    # import potpourri3d as pp3d
    import numpy as np 

    v = mesh.vertices.copy()
    f = mesh.faces.copy() 

    # geodesic_distances_components = [igl.heat_geodesic(v=v, 
    #                                                    f=f, 
    #                                                    t=t_diffuse, 
    #                                                    gamma=np.array(cc)) for cc in source_vertex_indices]
    if method == 'heat':
        
        import potpourri3d as pp3d
        solver = pp3d.MeshHeatMethodDistanceSolver(v,f,t_coef=t_diffuse)
        geodesic_distances_components = [solver.compute_distance_multisource(np.array(cc)) for cc in source_vertex_indices]
        geodesic_distances_components = np.array(geodesic_distances_components)
        
        if return_solver:
            return geodesic_distances_components, solver
        else:
            return geodesic_distances_components
        
    if method == 'exact':
        # use the very fast method in pygeodesic.
        import pygeodesic.geodesic as geodesic
        geoalg = geodesic.PyGeodesicAlgorithmExact(v, f)
        
        source_centre_vertex_indices = find_central_ind(v,source_vertex_indices)
        source_indices = np.array(source_centre_vertex_indices) # just one ind per component. 
        target_indices = np.array(np.arange(len(v)))
        distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)
        
        return distances, best_source


# using the above we can do watershed with seed regions and optional masks.... 
def mesh_watershed_segmentation_faces(mesh, face_components, t_diffuse=.5, mesh_face_mask=None, return_solver=True, method='heat'):
    r""" Do marker seeded watershed segmentation on the mesh by assigning individual faces to the user provided 'seeds' or sources by geodesic distance. Output is an integer label for each mesh face with -1 denoting a user masked out face.    
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh 
    face_components : list of arrays
        the seeds or 'sources' given as a list of face index components
    t_diffuse : scalar 
        set the time used for short-time heat flow in method='heat'. Larger values may make the solution more stable at the cost of over-smoothing
    mesh_face_mask : (n_faces,) array
        a binary array 0 or 1 specifying which faces should not be included in the final labels. Masked out faces are assigned a label of -1, whereas all valid labels are integer 0 or above. 
    return_solver : bool
        if True, return the solver used in the vector heat method, see https://github.com/nmwsharp/potpourri3d
    method : str
        specifies the geodesic distance computation method. either of 'heat' for vector heat or 'exact' for Djikstra

    Returns
    -------
    geodesic_distances_components : (n_faces,) array
        for each face, the geodesic distance to the assigned source
    geodesic_distances_label : (n_faces,) array
        for each faces, the id of the source assigned 
    """
    # also see. https://pypi.org/project/pygeodesic/
    import numpy as np 
    import igl 

    v = mesh.vertices.copy()
    f = mesh.faces.copy() 

    # convert the face to vertex components for processing
    vertex_components = [np.unique(f[ff_c].ravel()) for ff_c in face_components]
    
    if method=='heat':
        # give the face components directly in terms of the original mesh indices. 
        if return_solver:
            geodesic_distances_components, heat_solver = compute_geodesic_sources_distance_on_mesh(mesh, vertex_components, t_diffuse=t_diffuse)
        else:
            geodesic_distances_components = compute_geodesic_sources_distance_on_mesh(mesh, vertex_components, t_diffuse=t_diffuse)
        # average onto the faces. 
        # use the average onto faces to get n_components x n_faces. 
        geodesic_distances_components = igl.average_onto_faces(f, geodesic_distances_components.T).T

        # assign to nearest neighbor. 
        geodesic_distances_label = np.argmin(geodesic_distances_components, axis=0)
            
        # apply mask if exists.
        if mesh_face_mask is not None:
            geodesic_distances_label[mesh_face_mask>0] = -1 # assign this negative.
    
        # return the original distance computations as well as the nearest neighbor. 
        if return_solver:
            return geodesic_distances_components, geodesic_distances_label, heat_solver
        else:
            return geodesic_distances_components, geodesic_distances_label
        
    if method =='exact':
        import scipy.stats as spstats
        geodesic_distances_components, geodesic_distances_label = compute_geodesic_sources_distance_on_mesh(mesh, vertex_components, t_diffuse=t_diffuse, method=method)
        # project onto faces.
        geodesic_distances_components = igl.average_onto_faces(f, geodesic_distances_components[:,None]).T
        
        geodesic_distances_label = spstats.mode( geodesic_distances_label[f], axis=1, nan_policy='omit' )[0]
        geodesic_distances_label = np.squeeze(geodesic_distances_label)
    
        # apply mask if exists.
        if mesh_face_mask is not None:
            geodesic_distances_label[mesh_face_mask>0] = -1 # assign this negative.
            
        return geodesic_distances_components, geodesic_distances_label
    

def find_curvature_cutoff_index(values, thresh=None, absval=True):
    r""" For a given array of values, find the first index where the difference or absolute difference between two values falls below a threshold value. This function can be used to find the stopping iteration number for conformalized mean curvature smoothing based on the dcrease in absolute Gaussian curvature. If no such index can be found then np.nan is returned

    Parameters
    ----------
    values : (N,) array
        1d array of values 
    thresh : scalar
        the cutoff value threshold where we return the index where values <= thresh. If None, the median of np.diff(values) is used. 
    absval : bool 
        determine whether the cut-off is on the absolute differences in value or differences in value 

    Returns
    -------
    ind : int
        if an index is found return an int otherwise return np.nan
            
    """
    import numpy as np 

    diff_values = np.diff(values)
    if absval:
        diff_values = np.abs(diff_values)

    if thresh is None:
        thresh = np.median(diff_values)

    ind = np.arange(len(diff_values))[diff_values<=thresh]
    # print(ind, thresh)
    if len(ind) > 0: 
        ind = ind[0]
    else:
        ind = np.nan # couldn't find one -> means more runs of flow is required. 

    return ind 

def average_onto_barycenter(v,f,vals, return_barycenter=True):
    r""" Convert vertex values to face values by barycentric averaging of multi-dimensional vertex-associated values. This function does not work for segmentation labels.

    Parameters
    ----------
    v : array
        the vertices of the 3D mesh 
    f : array
        the faces of the 3D mesh specified by vertex indices
    vals : (n_vertex, d) array
        the multi-dimensional values associated with each vertex to convert to face values
    return_barycenter : bool
        if True, return the barycenter coordinates

    Returns
    -------
    barycenter_vals_f : (n_faces,d) array
        the face-associated resampling of the input vertex-associated ``vals``
    barycenter : (n_faces,3) array
        the barycenter coordinates of the mesh
    
    """
    import igl 

    if len(vals.shape) == 1:
        vals_ = vals[:,None].copy()
    else:
        vals_ = vals.copy()

    barycenter = igl.barycenter(v,f)
    vals_f = vals_[f].copy()
    barycenter_vals_f = 1./3*vals_f[:,0] + 1./3*vals_f[:,1] + 1./3*vals_f[:,2] 

    return barycenter_vals_f, barycenter 


def find_principal_axes_surface_heterogeneity(pts, pts_weights=None, map_to_sphere=False, sort='ascending'):
    r""" Find the principal axes of a point cloud given individual weights for each point. If weights are not given, every point is weighted equally  

    Parameters
    ----------
    pts : array
        the coordinates of a point cloud
    pts_weights : array
        the positive weights specifying the importance of each point in the principal axes computation 
    map_to_sphere : bool
        if True, the unit sphere coordinate by projecting each point by distance normalization to compute principal axes. This enables geometry-independent computation useful for e.g. getting directional alignment based only on surface intensity
    sort : 'ascending' or 'descending'
        the sorting order of the eigenvectors in terms of the absolute value of the respective eigenvalues

    Returns
    -------
    w : (d,) array
        the sorted eigenvalues of the principal eigenvectors of the d-dimensional point cloud
    v : (d,d) array
        the sorted eigenvectors of the corresponding eigenvalues
    
    See Also
    --------
    :func:`unwrap3D.Unzipping.unzip.find_principal_axes_uv_surface` : 
        Equivalent for finding the principal eigenvectors when give a uv parametrized surface of xyz coordinates.
    :func:`unwrap3D.Mesh.meshtools.find_principal_axes_surface_heterogeneity_mesh` : 
        Equivalent for finding the principal eigenvectors when give a 3D triangle mesh.         

    """
    """
    weights should be positive. map_to_sphere so as not to be affected by geometry / reduce this.  
    """
    import numpy as np 

    if pts_weights is None: 
        pts_weights = np.ones(len(pts))
    pts_mean = np.mean(pts, axis=0) # this shouldn't have weighting 
    # # compute the weighted centroid. 
    # pts_mean = np.nansum(pts * pts_weights[:,None]/ float(np.sum(pts_weights)), axis=0)
    pts_demean = pts - pts_mean[None,:]

    if map_to_sphere:
        # r =  np.linalg.norm(pts_demean, axis=-1)
        # theta = np.arctan2(pts_demean[:,1], pts_demean[:,0])
        # phi = np.arccos(pts_demean[:,2] / r)

        # construct a new version with 0 magnitude? 
        pts_demean = pts_demean / (np.linalg.norm(pts_demean, axis=-1)[:,None] + 1e-8)

    # apply the weighting 
    pts_demean = pts_demean * pts_weights[:,None] / float(np.sum(pts_weights))
    pts_cov = np.cov(pts_demean.T) # 3x3 matrix #-> expect symmetric. 

    w, v = np.linalg.eigh(pts_cov)
    # sort large to small. 
    if sort=='descending':
        w_sort = np.argsort(np.abs(w))[::-1]
    if sort=='ascending':
        w_sort = np.argsort(np.abs(w))

    w = w[w_sort]
    v = v[:,w_sort]

    return w, v 


def find_principal_axes_surface_heterogeneity_mesh(v, f, v_weights=None, map_to_sphere=False, sort='ascending'):
    r""" Find the principal axes of a mesh given individual weights for each vertex. If weights are not given, every vertex is weighted equally  

    Parameters
    ----------
    v : array
        the vertex cordinates of the 3D mesh
    f : array
        the faces of the 3D mesh specified by vertex indices
    v_weights : array
        the positive weights specifying the importance of each vertex in the principal axes computation 
    map_to_sphere : bool
        if True, the unit sphere coordinate by projecting each vertex by distance normalization to compute principal axes. This enables geometry-independent computation useful for e.g. getting directional alignment based only on surface intensity
    sort : 'ascending' or 'descending'
        the sorting order of the eigenvectors in terms of the absolute value of the respective eigenvalues

    Returns
    -------
    w : (d,) array
        the sorted eigenvalues of the principal eigenvectors of the d-dimensional point cloud
    v : (d,d) array
        the sorted eigenvectors of the corresponding eigenvalues
    
    See Also
    --------
    :func:`unwrap3D.Unzipping.unzip.find_principal_axes_uv_surface` : 
        Equivalent for finding the principal eigenvectors when give a uv parametrized surface of xyz coordinates.
    :func:`unwrap3D.Mesh.meshtools.find_principal_axes_surface_heterogeneity_mesh` : 
        Equivalent for finding the principal eigenvectors when give a 3D triangle mesh.         

    """
    """
    weights should be positive. map_to_sphere so as not to be affected by geometry / reduce this.  
    """
    import numpy as np 
    import igl 

    # pts = v.copy()
    pts = igl.barycenter(v,f)
    # pts_mean = np.mean(pts, axis=0) # use the barycenter? 
    pts_area_weights = igl.doublearea(v,f)
    pts_area_weights = pts_area_weights / float(np.nansum(pts_area_weights))
    pts_mean = np.mean(pts*pts_area_weights[:,None], axis=0)
    
    pts_demean = pts - pts_mean[None,:]

    if map_to_sphere:
        # r =  np.linalg.norm(pts_demean, axis=-1)
        # theta = np.arctan2(pts_demean[:,1], pts_demean[:,0])
        # phi = np.arccos(pts_demean[:,2] / r)

        # construct a new version with 0 magnitude? 
        pts_demean = pts_demean / (np.linalg.norm(pts_demean, axis=-1)[:,None] + 1e-8)

    # unweighted version. 
    v_weights_barycenter = igl.average_onto_faces(f, v_weights[:,None])
    v_weights_ = v_weights_barycenter*pts_area_weights
    pts_demean = pts_demean * v_weights_[:,None] / float(np.sum(v_weights_))
    pts_cov = np.cov(pts_demean.T) # 3x3 matrix #-> expect symmetric. 

    w, v = np.linalg.eigh(pts_cov)
    # sort large to small. 
    if sort=='descending':
        w_sort = np.argsort(np.abs(w))[::-1]
    if sort=='ascending':
        w_sort = np.argsort(np.abs(w))

    w = w[w_sort]
    v = v[:,w_sort]

    return w, v 


def rescale_mesh_points_to_grid_size(rect_mesh, grid=None, grid_shape=None):
    r""" Give a rectangular-like mesh where the last 2 coordinate axes is the 2D xy coordinates such as that from a rectangular conformal map, resize points coordinates along x- and y- axes onto a given image grid so that we can for example interpolate mesh values onto an image

    Parameters
    ----------
    rect_mesh : trimesh.Trimesh
        input 2D mesh where the first 2 coordinate axes is the 2D xy coordinates 
    grid : (M,N) or (M,N,d) single- or multi- channel image 
        input image to get the (M,N) shape
    grid_shape : (M,N) tuple
        the shape of the grid, only used if grid is not specified. Only one of grid or grid_shape needs to be passed 

    Returns
    -------
    mesh_ref : trimesh.Trimesh
        output 2D mesh with resized vertex coordinates where like input the last 2 coordinate axes is the 2D xy coordinates 
    grid_shape : (M,N) tuple
        the shape of the image grid, the output mesh indexes into 

    """
    import trimesh
    import pylab as plt 
    import numpy as np 

    # either give a predefined image grid coordinates or grid_shape. 
    if grid is not None:
        grid_shape = grid.shape[:2]
    else:
        grid = np.indices(grid_shape)
        grid = np.dstack(grid) # m x n x 2

    # grid_pts = grid.reshape(-1,grid.shape[-1]) # make flat. 
    v = rect_mesh.vertices.copy(); v = v[:,1:].copy() # take just the last two axes. 
    f = rect_mesh.faces.copy()
    
    scale_v_0 = (grid.shape[0]-1.) / (v[:,0].max() - v[:,0].min())
    scale_v_1 = (grid.shape[1]-1.) / (v[:,1].max() - v[:,1].min())
    v[:,0] = v[:,0] * scale_v_0
    v[:,1] = v[:,1] * scale_v_1 
    
    v = np.hstack([np.ones(len(v))[:,None], v])
    mesh_ref = trimesh.Trimesh(v, f, validate=False, process=False)

    return mesh_ref, grid_shape


def match_and_interpolate_img_surface_to_rect_mesh(rect_mesh, grid=None, grid_shape=None, rescale_mesh_pts=True, match_method='cross'):
    r""" Match the grid of an image to the faces of a reference rectangular-like triangle mesh where the last 2 coordinate axes is the 2D xy coordinates to allow mapping of mesh measurements to an image

    Parameters
    ----------
    rect_mesh : trimesh.Trimesh
        input 2D mesh where the first 2 coordinate axes is the 2D xy coordinates 
    grid : (M,N) or (M,N,d) single- or multi- channel image 
        input image to get the (M,N) shape
    grid_shape : (M,N) tuple
        the shape of the grid, only used if grid is not specified. Only one of grid or grid_shape needs to be passed 
    rescale_mesh_pts : bool
        if True, the rect_mesh vertices are first rescaled in order to maximally cover the size of the intended image
    match_method : str
        one of 'cross' implementing https://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf or 'cramer' implementing http://blackpawn.com/texts/pointinpoly for computing the barycentric coordinate after matching each image pixel to the rect_mesh
    
    Returns
    -------
    tri_id : (MxN,)
        1d array giving the face index each image pixel maps to in the rect_mesh 
    mesh_ref_closest_pt_barycentric : (MxN,3)
        the barycentric weights giving the position inside the matched triangle face, each image pixel maps to  
    grid_shape : (M,N) tuple
        the shape of the image grid
        
    """
    import trimesh
    # import pylab as plt 
    import numpy as np 

    # either give a predefined image grid coordinates or grid_shape. 
    if grid is not None:
        grid_shape = grid.shape[:2]
    else:
        grid = np.indices(grid_shape)
        grid = np.dstack(grid) # m x n x 2

    # grid_pts = grid.reshape(-1,grid.shape[-1]) # make flat. 
    v = rect_mesh.vertices.copy(); v = v[:,1:].copy() # take just the last two axes. 
    f = rect_mesh.faces.copy()
    # print(v.shape)

    if rescale_mesh_pts:
        scale_v_0 = (grid.shape[0]-1.) / (v[:,0].max() - v[:,0].min())
        scale_v_1 = (grid.shape[1]-1.) / (v[:,1].max() - v[:,1].min())
        v[:,0] = v[:,0] * scale_v_0
        v[:,1] = v[:,1] * scale_v_1 
        
        # fig, ax = plt.subplots()
        # ax.triplot(v[:,1], v[:,0], f, lw=.1)
        # ax.set_aspect(1)
        # plt.show()
    # we have to force this to make it 3D to leverage the functions.
    v = np.hstack([np.ones(len(v))[:,None], v])
    mesh_ref = trimesh.Trimesh(v, f, validate=False, process=False)
    # create nearest lookup object 
    prox_mesh_ref = trimesh.proximity.ProximityQuery(mesh_ref) # this is the lookup object.
    # get the closest point on the mesh to the unwrap_params surface  

    grid_query = grid.reshape(-1,grid.shape[-1])
    grid_query = np.hstack([np.ones(len(grid_query))[:,None], grid_query])
    closest_pt, dist_pt, tri_id = prox_mesh_ref.on_surface(grid_query)

    """
    get the barycentric for interpolation. 
    """
    # given then triangle id, we can retrieve the corresponding vertex point and the corresponding vertex values to interpolate. 
    # fetch the barycentric. 
    mesh_ref_tri_id_vertices = mesh_ref.vertices[mesh_ref.faces[tri_id]] # get the vertex points (N,3) 
    
    # convert to barycentric for each triangle which gives the interpolating weights. 
    mesh_ref_closest_pt_barycentric = trimesh.triangles.points_to_barycentric(mesh_ref_tri_id_vertices, 
                                                                              closest_pt, 
                                                                              method=match_method)

    return tri_id, mesh_ref_closest_pt_barycentric, grid_shape



def match_and_interpolate_uv_surface_to_mesh(unwrap_params, mesh_ref, match_method='cross'):
    r""" Match the grid of an image given by the shape of the uv unwrapped surface to a reference 3D triangle mesh based on nearest distance for reinterpolation of mesh measurements to unwrapped surface  
    
    Parameters
    ----------
    unwrap_params : (UxVx3) array 
        the input 2D image specifying the uv unwrapped (x,y,z) surface 
    mesh_ref : trimesh.Trimesh
        input 3D mesh we wish to match the coordinates of unwrap_params to, to allow barycentric interpolation 
    match_method : str
        one of 'cross' implementing https://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf or 'cramer' implementing http://blackpawn.com/texts/pointinpoly for computing the barycentric coordinate after matching each image pixel to the rect_mesh
    
    Returns
    -------
    tri_id : (UxV,)
        1d array giving the face index each (u,v) image pixel maps to in ``mesh_ref``
    mesh_ref_closest_pt_barycentric : (UxV,3)
        the barycentric weights giving the position inside the matched triangle face, each (u,v) image pixel maps to     
    """
    """
    we use the mesh as a reference 
    """
    import trimesh

    # create nearest lookup object 
    prox_mesh_ref = trimesh.proximity.ProximityQuery(mesh_ref) # this is the lookup object.
    # get the closest point on the mesh to the unwrap_params surface  
    closest_pt, dist_pt, tri_id = prox_mesh_ref.on_surface(unwrap_params.reshape(-1,unwrap_params.shape[-1]))

    # given then triangle id, we can retrieve the corresponding vertex point and the corresponding vertex values to interpolate. 
    # fetch the barycentric. 
    mesh_ref_tri_id_vertices = mesh_ref.vertices[mesh_ref.faces[tri_id]] # get the vertex points (N,3) 
    
    # convert to barycentric for each triangle which gives the interpolating weights. 
    mesh_ref_closest_pt_barycentric = trimesh.triangles.points_to_barycentric(mesh_ref_tri_id_vertices, 
                                                                              closest_pt, 
                                                                              method=match_method)

    return tri_id, mesh_ref_closest_pt_barycentric


def mesh_vertex_interpolate_scalar(mesh_ref, mesh_tri_id, mesh_tri_barycentric, scalar_vals):
    r""" Interpolate vertex associated values associated with the vertices of a reference mesh onto a different geometry specified implicitly by the face id and barycentric coordinates the new geometry maps to in the reference mesh  

    Parameters
    ----------
    mesh_ref : trimesh.Trimesh
        input 3D mesh we wish to match the coordinates of unwrap_params to, to allow barycentric interpolation 
    mesh_tri_id : (N,)
        1d array of the triangle face index in ``mesh_ref`` we want to interpolate ``scalar_vals`` on 
    mesh_tri_barycentric : (N,3)
        the barycentric weights specifying the linear weighting of the vertex scalar_vals associated with the vertices of ``mesh_ref`` to compute the new scalar values
    scalar_vals : (n_vertices,d) 
        the vertex associated measurements on the mesh_ref which is used to compute the new scalar values at the coordinate locations on the mesh specified by ``mesh_tri_id`` and ``mesh_tri_barycentric``

    Returns
    -------
    scalar_vals_interp : (N,d)
        the reinterpolated ``scalar_vals`` at the coordinate locations on the mesh specified by ``mesh_tri_id`` and ``mesh_tri_barycentric``
       
    """
    import numpy as np 

    scalar_vals_tri = scalar_vals[mesh_ref.faces[mesh_tri_id]].copy() # N_faces x 3 x d 
    # print(scalar_vals_tri.shape)
    scalar_vals_interp = np.sum(np.array([mesh_tri_barycentric[:,ch][:,None]*scalar_vals_tri[:,ch] for ch in np.arange(mesh_ref.vertices.shape[-1])]), axis=0)

    return scalar_vals_interp


def uv_surface_pulldown_mesh_surface_coords( unwrap_params, mesh_ref, Varray, match_method='cross', return_interp=False):
    r""" Main function to unwrap a list of 3D triangle meshes given by their vertices onto a 2D image grid based on matching the uv-unwrap of a joint 3D triangle mesh typically this is the uv-rectangle and the unit sphere.  
    
    Parameters
    ----------
    unwrap_params : (M,N,3)
        (u,v) parameterization of the 3D triangle mesh given by `mesh_ref`
    mesh_ref : trimesh.Trimesh
        the 3D triangle mesh equivalent of the geometry specified ``unwrap_params`` e.g. the unit sphere when ``unwrap_params`` is the UV-map 
    Varray : list of (n_vertices,3)
        a list of 3D meshes given only by their vertices whom are bijective to ``mesh_ref`` such that they share the same number of vertices and the same face connectivity
    match_method : str
        one of 'cross' implementing https://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf or 'cramer' implementing http://blackpawn.com/texts/pointinpoly for computing the barycentric coordinate after matching each image pixel to the rect_mesh
    return_interp : bool
        if True, also return the matching parameters between ``unwrap_params`` and ``mesh_ref`` for reuse 

    Returns
    -------
    pulldown_Varray_coords : (len(Varray),M,N,3) array
        The UV-unwrapping of all 3D surface meshes bijective to ``mesh_ref``
    (mesh_tri_id, mesh_ref_closest_pt_barycentric) : ( (M*N,), (M*N,3) ) tuple of arrays
        These are the matching parameters that establish correspondence between ``unwrap_params`` and ``mesh_ref`` in order to map all other 3D meshes given by ``Varray`` bijective to ``mesh_ref`` into the 2D grid specified by ``unwrap_params``

    """
    import numpy as np 

    unwrap_params_shape = unwrap_params.shape
    F = mesh_ref.faces.copy()

    mesh_tri_id, mesh_ref_closest_pt_barycentric = match_and_interpolate_uv_surface_to_mesh(unwrap_params, mesh_ref, match_method=match_method)

    # interpolate ...
    pulldown_Varray_coords = []
    for iii in np.arange(len(Varray)):
        Varray_tri_id_vertices_iii = Varray[iii][F[mesh_tri_id]].copy()
        # interpolate using the barycentric coordinates. to build the depth lookup array. 
        # Varray_interp_coords = closest_pt_barycentric[:,0][:,None] * tri_id_vertices_orig[:,0] + closest_pt_barycentric[:,1][:,None] * tri_id_vertices_orig[:,1] + closest_pt_barycentric[:,2][:,None] * tri_id_vertices_orig[:,2]
        Varray_interp_coords = np.sum(np.array([mesh_ref_closest_pt_barycentric[:,ch][:,None]*Varray_tri_id_vertices_iii[:,ch] for ch in np.arange(unwrap_params.shape[-1])]), axis=0)
        Varray_interp_coords = Varray_interp_coords.reshape(unwrap_params_shape)
        pulldown_Varray_coords.append(Varray_interp_coords)

    pulldown_Varray_coords = np.array(pulldown_Varray_coords)

    if return_interp:
        return pulldown_Varray_coords, (mesh_tri_id, mesh_ref_closest_pt_barycentric)
    else:
        return pulldown_Varray_coords


def xyz_surface_pulldown_mesh_surface_coords( unwrap_params, mesh_ref, Varray, mesh_tri_id=None, mesh_ref_closest_pt_barycentric=None, match_method='cross', return_interp=False):
    r""" Main function to unwrap a list of 3D triangle meshes given by their vertices bijective to a common 3D mesh onto another 3D surface through proximity-based matching 
    
    Parameters
    ----------
    unwrap_params : (N,3)
        3D triangle mesh to transfer coordinates to 
    mesh_ref : trimesh.Trimesh
        the 3D triangle mesh equivalent or similar to the geometry of ``unwrap_params`` bijective to geometrices given by `Varray`
    Varray : list of (n_vertices,3)
        a list of 3D meshes given only by their vertices whom are bijective to ``mesh_ref`` such that they share the same number of vertices and the same face connectivity
    match_method : str
        one of 'cross' implementing https://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf or 'cramer' implementing http://blackpawn.com/texts/pointinpoly for computing the barycentric coordinate after matching each image pixel to the rect_mesh
    return_interp : bool
        if True, also return the matching parameters between ``unwrap_params`` and ``mesh_ref`` for reuse 

    Returns
    -------
    pulldown_Varray_coords : (len(Varray),N,3) array
        The remapped coordinates of 3D surface meshes bijective to ``mesh_ref`` as vertex measurements on ``unwrap_params``
    (mesh_tri_id, mesh_ref_closest_pt_barycentric) : ( (M*N,), (M*N,3) ) tuple of arrays
        Returned if return_interp=True. These are the matching parameters that establish correspondence between ``unwrap_params`` and ``mesh_ref`` in order to map all other 3D meshes given by ``Varray`` bijective to ``mesh_ref`` into the 2D grid specified by ``unwrap_params``

    """
    import numpy as np 

    # unwrap_params_shape = unwrap_params.shape
    F = mesh_ref.faces.copy()

    if mesh_tri_id is None or mesh_ref_closest_pt_barycentric is None:
        mesh_tri_id, mesh_ref_closest_pt_barycentric = match_and_interpolate_uv_surface_to_mesh(unwrap_params, mesh_ref, match_method=match_method)

    # interpolate ...
    pulldown_Varray_coords = []
    for iii in np.arange(len(Varray)):
        Varray_tri_id_vertices_iii = Varray[iii][F[mesh_tri_id]].copy()
        # interpolate using the barycentric coordinates. to build the depth lookup array. 
        # Varray_interp_coords = closest_pt_barycentric[:,0][:,None] * tri_id_vertices_orig[:,0] + closest_pt_barycentric[:,1][:,None] * tri_id_vertices_orig[:,1] + closest_pt_barycentric[:,2][:,None] * tri_id_vertices_orig[:,2]
        Varray_interp_coords = np.sum(np.array([mesh_ref_closest_pt_barycentric[:,ch][:,None]*Varray_tri_id_vertices_iii[:,ch] for ch in np.arange(unwrap_params.shape[-1])]), axis=0)
        # Varray_interp_coords = Varray_interp_coords.reshape(unwrap_params_shape)
        pulldown_Varray_coords.append(Varray_interp_coords)
    pulldown_Varray_coords = np.array(pulldown_Varray_coords)

    if return_interp:
        return pulldown_Varray_coords, (mesh_tri_id, mesh_ref_closest_pt_barycentric)
    else:
        return pulldown_Varray_coords



def grid2D_surface_pulldown_mesh_surface_coords( rect_mesh, Varray, grid=None, grid_shape=None, rescale_mesh_pts=True, match_method='cross', return_interp=False, interp_method='linear'):
    r""" Main function to map a 2D rectangular mesh and list of 3D triangle meshes given by their vertices bijective to the 3D surface it describes to an image of a given grid_shape through proximity-based matching 
    
    Parameters
    ----------
    rect_mesh : trimesh.Trimesh
        A 2D triangle mesh specified as a 3D triangle mesh where the last 2 coordinate axes are taken to be the x-,y- coordinates of the output image. 
    Varray : list of (n_vertices,3)
        a list of 3D meshes given only by their vertices whom are bijective to ``rect_mesh`` such that they share the same number of vertices and the same face connectivity. We assume the order of the vertices in ``rect_mesh`` and the meshes in ``Varray`` are aligned.
    grid : (M,N) or (M,N,d) single- or multi- channel image 
        input image to get the (M,N) shape
    grid_shape : (M,N) tuple
        the shape of the grid, only used if grid is not specified. Only one of grid or grid_shape needs to be passed 
    rescale_mesh_pts : bool
        if True, the rect_mesh vertices are first rescaled in order to maximally cover the size of the intended image
    match_method : str
        one of 'cross' implementing https://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf or 'cramer' implementing http://blackpawn.com/texts/pointinpoly for computing the barycentric coordinate after matching each image pixel to the rect_mesh
    return_interp : bool
        if True, also return the matching parameters between the final image coordinates and ``rect_mesh`` for reuse 
    interp_method : str
        One of 'linear' for ``matplotlib.tri.LinearTriInterpolator`` interpolation, 
        'cubic_geom' for ``matplotlib.tri.CubicTriInterpolator`` with kind='geom' and 'cubic_min_E' for mtri.CubicTriInterpolator with kind='min_E'

    Returns
    -------
    pulldown_Varray_coords : (len(Varray),N,3) array
        The remapped coordinates of 3D surface meshes bijective to ``mesh_ref`` as vertex measurements on ``unwrap_params``
    triang : matplotlib.tri.Triangulation instance
        Optional return if ``return_interp=True``. The Matlplotlib triangulation of the rescaled ``rect_mesh`` used for interpolation with ``matplotlib.tri.LinearTriInterpolator`` internally

    """
    # since 2D we can directly exploit matplotlib. 
    import numpy as np 
    import matplotlib.tri as mtri 

    if rescale_mesh_pts:
        rect_mesh_rescale, grid_shape = rescale_mesh_points_to_grid_size(rect_mesh, grid=grid, grid_shape=grid_shape)
    else:
        rect_mesh_rescale = rect_mesh.copy()
        if grid_shape is None:
            grid_shape = grid.shape[:2]

    # print(rect_mesh_rescale.shape)
    triang = mtri.Triangulation(rect_mesh_rescale.vertices[:,1], 
                                rect_mesh_rescale.vertices[:,2], 
                                triangles=rect_mesh.faces)
    
    # specify the grid points 
    xi, yi = np.indices(grid_shape) 

    # interpolate ...
    pulldown_Varray_coords = []
    for iii in np.arange(len(Varray)):
        Varray_iii = Varray[iii].copy()

        if interp_method == 'linear': 
            # define the interpolation object for the Varray. 
            interp_objs = [mtri.LinearTriInterpolator(triang, Varray_iii[:,ch]) for ch in np.arange(Varray_iii.shape[-1])]
        if interp_method == 'cubic_geom':
            interp_objs = [mtri.CubicTriInterpolator(triang, Varray_iii[:,ch], kind='geom') for ch in np.arange(Varray_iii.shape[-1])]
        if interp_method == 'cubic_min_E':
            interp_objs = [mtri.CubicTriInterpolator(triang, Varray_iii[:,ch], kind='min_E') for ch in np.arange(Varray_iii.shape[-1])]
        interp_objs_scalars = np.dstack([interp_objs[ch](xi,yi) for ch in np.arange(Varray_iii.shape[-1])])

        pulldown_Varray_coords.append(interp_objs_scalars)

    pulldown_Varray_coords = np.array(pulldown_Varray_coords)

    if return_interp:
        return pulldown_Varray_coords, triang
    else:
        return pulldown_Varray_coords

# """
# include rectangular conformal map functions 
# """
def angle_distortion(v,f, param): 
    r""" Calculate the angle difference of triangles between two meshes of the same face connectivity 
    
    Parameters
    ----------
    v : (n_vertices, 3) array
        vertex coordinates of triangle mesh 1
    f : (n_faces, 3) array
        triangulations of both meshes given in terms of the vertex indices
    param : (n_vertices, 3) array
        vertex coordinates of triangle mesh 2  

    Returns
    -------
    distortion : (n_faces, 3) array
        array of angle differences in degrees at each triangle face 

    """
    import numpy as np 

    nv = len(v)
    nv2 = len(param)

    if nv!=nv2:
        print('Error: The two meshes are of different size.')
        return []
    else:

        # if input is not 3d coords... then make it 3D. 
        if v.shape[-1] == 1:
            # if 1d not 3d. 
            v = np.vstack([np.real(v), np.imag(v), np.zeros(len(v))]).T;
        if v.shape[-1] == 2: 
            v = np.hstack([v, np.zeros((len(v),1))]);
        

        if param.shape[-1] == 1:
            param = np.vstack([np.real(param), np.imag(param), np.zeros(len(param))]).T;
        if param.shape[-1] == 2:
            param = np.hstack([param, np.zeros((len(param), 1))])

        f1 = f[:,0].copy(); f2 = f[:,1].copy(); f3 = f[:,2].copy();

        # % calculate angles on v
        a3 = np.vstack([v[f1,0]-v[f3,0], v[f1,1]-v[f3,1], v[f1,2]-v[f3,2]]).T
        b3 = np.vstack([v[f2,0]-v[f3,0], v[f2,1]-v[f3,1], v[f2,2]-v[f3,2]]).T
        a1 = np.vstack([v[f2,0]-v[f1,0], v[f2,1]-v[f1,1], v[f2,2]-v[f1,2]]).T
        b1 = np.vstack([v[f3,0]-v[f1,0], v[f3,1]-v[f1,1], v[f3,2]-v[f1,2]]).T
        a2 = np.vstack([v[f3,0]-v[f2,0], v[f3,1]-v[f2,1], v[f3,2]-v[f2,2]]).T
        b2 = np.vstack([v[f1,0]-v[f2,0], v[f1,1]-v[f2,1], v[f1,2]-v[f2,2]]).T

        vcos1 = (a1[:,0]*b1[:,0] + a1[:,1]*b1[:,1] + a1[:,2]*b1[:,2]) / (np.sqrt(a1[:,0]**2+a1[:,1]**2+a1[:,2]**2) * np.sqrt(b1[:,0]**2+b1[:,1]**2+b1[:,2]**2))
        vcos2 = (a2[:,0]*b2[:,0] + a2[:,1]*b2[:,1] + a2[:,2]*b2[:,2]) / (np.sqrt(a2[:,0]**2+a2[:,1]**2+a2[:,2]**2) * np.sqrt(b2[:,0]**2+b2[:,1]**2+b2[:,2]**2))
        vcos3 = (a3[:,0]*b3[:,0] + a3[:,1]*b3[:,1] + a3[:,2]*b3[:,2]) / (np.sqrt(a3[:,0]**2+a3[:,1]**2+a3[:,2]**2) * np.sqrt(b3[:,0]**2+b3[:,1]**2+b3[:,2]**2))
            
        # % calculate angles on param
        c3 = np.vstack([param[f1,0]-param[f3,0], param[f1,1]-param[f3,1], param[f1,2]-param[f3,2]]).T
        d3 = np.vstack([param[f2,0]-param[f3,0], param[f2,1]-param[f3,1], param[f2,2]-param[f3,2]]).T
        c1 = np.vstack([param[f2,0]-param[f1,0], param[f2,1]-param[f1,1], param[f2,2]-param[f1,2]]).T
        d1 = np.vstack([param[f3,0]-param[f1,0], param[f3,1]-param[f1,1], param[f3,2]-param[f1,2]]).T
        c2 = np.vstack([param[f3,0]-param[f2,0], param[f3,1]-param[f2,1], param[f3,2]-param[f2,2]]).T
        d2 = np.vstack([param[f1,0]-param[f2,0], param[f1,1]-param[f2,1], param[f1,2]-param[f2,2]]).T

        paramcos1 = (c1[:,0]*d1[:,0] + c1[:,1]*d1[:,1] + c1[:,2]*d1[:,2]) / (np.sqrt(c1[:,0]**2+c1[:,1]**2+c1[:,2]**2) * np.sqrt(d1[:,0]**2+d1[:,1]**2+d1[:,2]**2))
        paramcos2 = (c2[:,0]*d2[:,0] + c2[:,1]*d2[:,1] + c2[:,2]*d2[:,2]) / (np.sqrt(c2[:,0]**2+c2[:,1]**2+c2[:,2]**2) * np.sqrt(d2[:,0]**2+d2[:,1]**2+d2[:,2]**2))
        paramcos3 = (c3[:,0]*d3[:,0] + c3[:,1]*d3[:,1] + c3[:,2]*d3[:,2]) / (np.sqrt(c3[:,0]**2+c3[:,1]**2+c3[:,2]**2) * np.sqrt(d3[:,0]**2+d3[:,1]**2+d3[:,2]**2))


        # % calculate the angle difference
        # distortion = (np.arccos(np.hstack([paramcos1, paramcos2, paramcos3])) - np.arccos(np.hstack([vcos1,vcos2,vcos3])))*180/np.pi;
        distortion = (np.arccos(np.vstack([paramcos1, paramcos2, paramcos3])) - np.arccos(np.vstack([vcos1,vcos2,vcos3])))*180/np.pi;
        distortion = distortion.T 

        return distortion # in terms of angles. ( the angles is the same number as triangles. )
# % histogram
# figure;
# hist(distortion,-180:1:180);
# xlim([-180 180])
# title('Angle Distortion');
# xlabel('Angle difference (degree)')
# ylabel('Number of angles')
# set(gca,'FontSize',12);


def _cotangent_laplacian(v,f):
    # % Compute the cotagent Laplacian of a mesh used in the rectangular conformal unwrapping 
    # 
    # See: 
    # % [1] T. W. Meng, G. P.-T. Choi and L. M. Lui, 
    # %     "TEMPO: Feature-Endowed Teichmüller Extremal Mappings of Point Clouds."
    # %     SIAM Journal on Imaging Sciences, 9(4), pp. 1922-1962, 2016.
    import numpy as np 
    import scipy.sparse as sparse
    
    nv = len(v);
    
    f1 = f[:,0]; 
    f2 = f[:,1]; 
    f3 = f[:,2];
    
    l1 = np.sqrt(np.sum((v[f2] - v[f3])**2,1));
    l2 = np.sqrt(np.sum((v[f3] - v[f1])**2,1));
    l3 = np.sqrt(np.sum((v[f1] - v[f2])**2,1));
    
    s = (l1 + l2 + l3)*0.5;
    area = np.sqrt( s*(s-l1)*(s-l2)*(s-l3));
     
    cot12 = (l1**2 + l2**2 - l3**2)/(area)/2.;
    cot23 = (l2**2 + l3**2 - l1**2)/(area)/2.; 
    cot31 = (l1**2 + l3**2 - l2**2)/(area)/2.; 
    diag1 = -cot12-cot31; 
    diag2 = -cot12-cot23; 
    diag3 = -cot31-cot23;
    
    # II = [f1; f2; f2; f3; f3; f1; f1; f2; f3];
    # JJ = [f2; f1; f3; f2; f1; f3; f1; f2; f3];
    # V = [cot12; cot12; cot23; cot23; cot31; cot31; diag1; diag2; diag3];
    II = np.hstack([f1,f2,f2,f3,f3,f1,f1,f2,f3])
    JJ = np.hstack([f2,f1,f3,f2,f1,f3,f1,f2,f3])
    V = np.hstack([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])
    
    L = sparse.csc_matrix((V, (II,JJ)), shape=(nv,nv));
    
    return L 


def beltrami_coefficient(v, f, map_):
    r""" Compute the Beltrami coefficient of a mapping between two triangle meshes. The lower the Beltrami coefficient the lower the metric distortion between the meshes

    Parameters
    ----------
    v : (n_vertices, 3) array
        vertex coordinates of triangle mesh 1
    f : (n_faces, 3) array
        triangulations of both meshes given in terms of the vertex indices
    map_ : (n_vertices, 3) array
        vertex coordinates of triangle mesh 2  

    Returns
    -------
    mu : (n_faces, ) complex array
        array of the beltrami coefficient for each triangle face  

    References
    ----------
    .. [1] T. W. Meng, G. P.-T. Choi and L. M. Lui, "TEMPO: Feature-Endowed Teichmüller Extremal Mappings of Point Clouds." SIAM Journal on Imaging Sciences, 9(4), pp. 1922-1962, 2016.

    """
    import numpy as np 
    import scipy.sparse as sparse

    nf = len(f);    
    Mi = np.reshape(np.vstack([np.arange(nf), 
                               np.arange(nf), 
                               np.arange(nf)]), ((1,3*nf)), order='F').copy();
    Mj = np.reshape(f.T, ((1,3*nf)), order='F').copy();
    
    e1 = v[f[:,2],:2] - v[f[:,1],:2];
    e2 = v[f[:,0],:2] - v[f[:,2],:2];
    e3 = v[f[:,1],:2] - v[f[:,0],:2];
    
    area = (-e2[:,0]*e1[:,1] + e1[:,0]*e2[:,1])/2.;
    area = np.vstack([area,area,area])

    # should this be F or C?
    # Mx = np.reshape(np.vstack([e1[:,1],e2[:,1],e3[:,1]])/area /2. , ((1, 3*nf)), order='C');
    # My = -np.reshape(np.vstack([e1[:,0],e2[:,0],e3[:,0]])/area /2. , ((1, 3*nf)), order='C');
    Mx = np.reshape(np.vstack([e1[:,1],e2[:,1],e3[:,1]])/area /2. , ((1, 3*nf)), order='F');
    My = -np.reshape(np.vstack([e1[:,0],e2[:,0],e3[:,0]])/area /2. , ((1, 3*nf)), order='F');
    
    Mi = Mi.ravel()
    Mj = Mj.ravel()
    Mx = Mx.ravel()
    My = My.ravel()
    # Dx = sparse(Mi,Mj,Mx);
    # Dy = sparse(Mi,Mj,My);
     # S = sparse(i,j,s) where m = max(i) and n = max(j).
    Dx = sparse.csr_matrix((Mx, (Mi,Mj)), shape=((np.max(Mi)+1, np.max(Mj)+1)))
    Dy = sparse.csr_matrix((My, (Mi,Mj)), shape=((np.max(Mi)+1, np.max(Mj)+1)))

    dXdu = Dx*map_[:,0];
    dXdv = Dy*map_[:,0];
    dYdu = Dx*map_[:,1];
    dYdv = Dy*map_[:,1];
    dZdu = Dx*map_[:,2];
    dZdv = Dy*map_[:,2];
    
    E = dXdu**2 + dYdu**2 + dZdu**2;
    G = dXdv**2 + dYdv**2 + dZdv**2;
    F = dXdu*dXdv + dYdu*dYdv + dZdu*dZdv;
    
    # this line? 
    mu = (E - G + 2 * 1j * F) / ((E + G + 2.*np.sqrt(E*G - F**2))+1e-12);

    return mu


def linear_beltrami_solver(v,f,mu,landmark,target):
    r""" Linear Beltrami solver to find the minimal quasiconformal distortion mapping for unwrapping an open 3D mesh to a 2D rectangular map  

    Parameters
    ----------
    v : (n_vertices, 3) array
        vertex coordinates of triangle mesh 1
    f : (n_faces, 3) array
        triangulations of both meshes given in terms of the vertex indices
    mu : (n_faces,) complex array
        the beltrami coefficient at each triangular face 
    landmark : (n,) array
        the vertex indices in the triangle mesh to enforce mapping to ``target`` 
    target : (n,) complex array
        the coordinates of ``landmark`` vertices in the 2D unwrapping 

    Returns
    -------
    param : (n_vertices, 2) array
        the 2D coordinates of the now 2D parametrized vertex coordinates of the input mesh 

    References
    ----------
    .. [1] P. T. Choi, K. C. Lam, and L. M. Lui, "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces." SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.
    
    """
    import numpy as np 
    import scipy.sparse as spsparse

    af = (1-2.*np.real(mu)+np.abs(mu)**2)/(1.0-np.abs(mu)**2);
    bf = -2*np.imag(mu)/(1.0-np.abs(mu)**2);
    gf = (1+2*np.real(mu)+np.abs(mu)**2)/(1.0-np.abs(mu)**2);

    f0 = f[:,0].copy(); f1 = f[:,1].copy(); f2 = f[:,2].copy();

    uxv0 = v[f1,1] - v[f2,1];
    uyv0 = v[f2,0] - v[f1,0];
    uxv1 = v[f2,1] - v[f0,1];
    uyv1 = v[f0,0] - v[f2,0]; 
    uxv2 = v[f0,1] - v[f1,1];
    uyv2 = v[f1,0] - v[f0,0];

    l = np.vstack([np.sqrt(uxv0**2 + uyv0**2), np.sqrt(uxv1**2 + uyv1**2), np.sqrt(uxv2**2 + uyv2**2)]).T;
    # l = np.vstack([np.sqrt(np.sum(uxv0**2 + uyv0**2, axis=-1)), np.sqrt(np.sum(uxv1**2 + uyv1**2,axis=-1)), np.sqrt(np.sum(uxv2**2 + uyv2**2,axis=-1))]).T; # this is just the lengths. 
    s = np.sum(l,axis=-1)*0.5;
    area = np.sqrt(s*(s-l[:,0])*(s-l[:,1])*(s-l[:,2])) + 1e-12; # heron's formula. 

    v00 = (af*uxv0*uxv0 + 2*bf*uxv0*uyv0 + gf*uyv0*uyv0)/area;
    v11 = (af*uxv1*uxv1 + 2*bf*uxv1*uyv1 + gf*uyv1*uyv1)/area;
    v22 = (af*uxv2*uxv2 + 2*bf*uxv2*uyv2 + gf*uyv2*uyv2)/area;
    v01 = (af*uxv1*uxv0 + bf*uxv1*uyv0 + bf*uxv0*uyv1 + gf*uyv1*uyv0)/area;
    v12 = (af*uxv2*uxv1 + bf*uxv2*uyv1 + bf*uxv1*uyv2 + gf*uyv2*uyv1)/area;
    v20 = (af*uxv0*uxv2 + bf*uxv0*uyv2 + bf*uxv2*uyv0 + gf*uyv0*uyv2)/area;

    I = np.hstack([f0,f1,f2,f0,f1,f1,f2,f2,f0]);
    J = np.hstack([f0,f1,f2,f1,f0,f2,f1,f0,f2]);
    V = np.hstack([v00,v11,v22,v01,v01,v12,v12,v20,v20])/2.;
    # # A = sparse(I,J,-V);
    A = spsparse.csr_matrix((-V, (I,J)), (len(v),len(v)), dtype=np.cfloat); A = spsparse.lil_matrix(A, dtype=np.cfloat)
   
    targetc = target[:,0] + 1j*target[:,1];
    b = -A[:,landmark].dot(targetc);
    
    b[landmark] = targetc;
    A[landmark,:] = 0; A[:,landmark] = 0;
    A = A + spsparse.csr_matrix((np.ones(len(landmark)), (landmark,landmark)), (A.shape[0], A.shape[1])); # size(A,1), size(A,2));
    param = spsparse.linalg.spsolve(A,b)
    param = np.vstack([np.real(param), np.imag(param)]).T
    # map = A\b;
    # map = [real(map),imag(map)];
    return param 


def direct_spherical_conformal_map(v,f):
    r""" A linear method for computing spherical conformal map of a genus-0 closed surface using quasiconformal mapping

    Parameters
    ----------
    v : (n_vertices, 3) array
        vertex coordinates of a genus-0 triangle mesh
    f : (n_faces, 3) array
        triangulations of a genus-0 triangle mesh
    
    Returns
    -------
    param : (n_vertices, 3) array 
        vertex coordinates of the spherical conformal parameterization which maps the input mesh to the unit sphere. 

    References
    ----------
    .. [1] P. T. Choi, K. C. Lam, and L. M. Lui, "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces." SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.
    
    """
    import numpy as np 
    import scipy.sparse as spsparse
    import scipy.sparse.linalg as spalg

    # # # %% Check whether the input mesh is genus-0
    # # if len(v)-3*len(f)/2+len(f) != 2:
    # #     print('The mesh is not a genus-0 closed surface.\n');
    # #     return []
    # # else:
    # print('spherical param')

    # %% Find the most regular triangle as the "big triangle"
    temp = v[f.ravel()].copy()
    e1 = np.sqrt(np.sum((temp[1::3] - temp[2::3])**2, axis=-1)) 
    e2 = np.sqrt(np.sum((temp[0::3] - temp[2::3])**2, axis=-1))
    e3 = np.sqrt(np.sum((temp[0::3] - temp[1::3])**2, axis=-1))
    regularity = np.abs(e1/(e1+e2+e3)-1./3) + np.abs(e2/(e1+e2+e3)-1./3) + np.abs(e3/(e1+e2+e3)-1./3) # this being the most equilateral. 
    bigtri = np.argmin(regularity) 


    # % In case the spherical parameterization result is not evenly distributed,
    # % try to change bigtri to the id of some other triangles with good quality

    # %% North pole step: Compute spherical map by solving laplace equation on a big triangle
    nv = len(v); 
    M = _cotangent_laplacian(v,f);

    # this becomes the fixed triangle.
    p1 = f[bigtri,0]#.copy();
    p2 = f[bigtri,1]#.copy();
    p3 = f[bigtri,2]#.copy();

    fixed = np.hstack([p1,p2,p3]);
    # [mrow,mcol,mval] = find(M(fixed,:));
    (mrow, mcol, mval) = spsparse.find(M[fixed])
    # print(mrow,mcol,mval)
    M = M - spsparse.csr_matrix((mval, (fixed[mrow],mcol)),(nv,nv)) + spsparse.csr_matrix((np.ones(3), (fixed,fixed)),(nv,nv));
    
    # % set the boundary condition for big triangle
    x1 = 0; y1 = 0; x2 = 1; y2 = 0; #% arbitrarily set the two points
    a = v[p2] - v[p1];
    b = v[p3] - v[p1];
    sin1 = (np.linalg.norm(np.cross(a,b), ord=2))/(np.linalg.norm(a,ord=2)*np.linalg.norm(b,ord=2));
    ori_h = np.linalg.norm(b,ord=2)*sin1;
    ratio = np.linalg.norm([x1-x2,y1-y2],ord=2)/np.linalg.norm(a,ord=2);
    y3 = ori_h*ratio; #% compute the coordinates of the third vertex
    x3 = np.sqrt(np.linalg.norm(b,ord=2)**2*ratio**2-y3**2);

    # print(x3,y3)
    # % Solve the Laplace equation to obtain a harmonic map
    c = np.zeros(nv); c[p1] = x1; c[p2] = x2; c[p3] = x3;
    d = np.zeros(nv); d[p1] = y1; d[p2] = y2; d[p3] = y3;
    z = spalg.spsolve(M , c+1j*d);
    z = z-np.mean(z); # o this.

    # print(np.mean(z))
    # print('z', z.shape)

    # % inverse stereographic projection
    S = np.vstack([2.*np.real(z)/(1+np.abs(z)**2), 2*np.imag(z)/(1.+np.abs(z)**2), (-1+np.abs(z)**2)/(1+np.abs(z)**2)]).T

    # %% Find optimal big triangle size
    w = S[:,0]/(1.+S[:,2]) + 1j*S[:,1]/(1.+S[:,2])

    # % find the index of the southernmost triangle
    index = np.argsort(np.abs(z[f[:,0]]) + np.abs(z[f[:,1]]) + np.abs(z[f[:,2]]), kind='stable') # this is absolutely KEY!
    inner = index[0];
    if inner == bigtri:
        inner = index[1]; # select the next one. # this is meant to be the northernmost.... 

    # print('bigtri', bigtri)
    # print('inner', index[:20])

    # % Compute the size of the northern most and the southern most triangles 
    NorthTriSide = (np.abs(z[f[bigtri,0]]-z[f[bigtri,1]]) + np.abs(z[f[bigtri,1]]-z[f[bigtri,2]]) + np.abs(z[f[bigtri,2]]-z[f[bigtri,0]]))/3.; # this is a number. 
    SouthTriSide = (np.abs(w[f[inner,0]]-w[f[inner,1]]) + np.abs(w[f[inner,1]]-w[f[inner,2]]) + np.abs(w[f[inner,2]]-w[f[inner,0]]))/3.;

    # % rescale to get the best distribution
    z = z*(np.sqrt(NorthTriSide*SouthTriSide))/(NorthTriSide); 

    # % inverse stereographic projection
    S = np.vstack([2.*np.real(z)/(1+np.abs(z)**2), 2*np.imag(z)/(1.+np.abs(z)**2), (-1+np.abs(z)**2)/(1+np.abs(z)**2)]).T

    if np.sum(np.isnan(S)) > 0: 
        # if harmonic map fails due to very bad triangulations, use tutte map
        print('implement tutte map')
        return []

    # %% South pole step
    I = np.argsort(S[:,2], kind='stable')

    # % number of points near the south pole to be fixed  
    # % simply set it to be 1/10 of the total number of vertices (can be changed)
    # % In case the spherical parameterization is not good, change 10 to
    # % something smaller (e.g. 2)
    fixnum = np.maximum(int(np.round(len(v)/10)), 3)
    fixed = I[:np.minimum(len(v), fixnum)]

    # % south pole stereographic projection
    P = np.vstack([S[:,0]/(1.+S[:,2]),  S[:,1]/(1.+S[:,2])]).T

    # % compute the Beltrami coefficient
    mu = beltrami_coefficient(P, f, v); 
    # problem is here. 
    # % compose the map with another quasi-conformal map to cancel the distortion
    param = linear_beltrami_solver(P,f,mu,fixed,P[fixed]); # fixed is index. 

    # print('num_nan: ', np.sum(np.isnan(param)))
    if np.sum(np.isnan(param)) > 0: # this is failing at the moment
        print('recomputing fixed elements')
        # % if the result has NaN entries, then most probably the number of
        # % boundary constraints is not large enough  
        # % increase the number of boundary constrains and run again
        fixnum = fixnum*5; #% again, this number can be changed
        fixed = I[:np.minimum(len(v),fixnum)]; 
        param = linear_beltrami_solver(P,f,mu,fixed,P[fixed]); 
        
        if np.sum(np.isnan(param)) > 0:
            param = P.copy(); #% use the old result
    z = param[:,0] + 1j*param[:,1]
    # z = complex(map(:,1),map(:,2));

    # % inverse south pole stereographic projection
    param = np.vstack([2*np.real(z)/(1.+np.abs(z)**2), 2*np.imag(z)/(1+np.abs(z)**2), -(np.abs(z)**2-1)/(1.+np.abs(z)**2)]).T
    # map = [2*real(z)./(1+abs(z).^2), 2*imag(z)./(1+abs(z).^2), -(abs(z).^2-1)./(1+abs(z).^2)];

    return param 


# implementation of extension functions to improve conformal mapping. 
def _face_area(v,f):
    """ Compute the area of every face of a triangle mesh.

    References
    ----------
    [1] P. T. Choi, K. C. Lam, and L. M. Lui, "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces." SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.

    """
    # % Compute the area of every face of a triangle mesh.
    # %
    # % If you use this code in your own work, please cite the following paper:
    # % [1] P. T. Choi, K. C. Lam, and L. M. Lui, 
    # %     "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces."
    # %     SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.
    # %
    # % Copyright (c) 2013-2018, Gary Pui-Tung Choi
    # % https://scholar.harvard.edu/choi
    import numpy as np 

    v12 = v[f[:,1]] - v[f[:,0]]
    v23 = v[f[:,2]] - v[f[:,1]]
    v31 = v[f[:,0]] - v[f[:,2]]

    a = np.sqrt(np.nansum( v12 * v12, axis=-1))
    b = np.sqrt(np.nansum( v23 * v23, axis=-1))
    c = np.sqrt(np.nansum( v31 * v31, axis=-1))

    s = (a+b+c)/2.;
    fa = np.sqrt(s*(s-a)*(s-b)*(s-c)); # heron's formula

    return fa 

# def area_distortion(v,f,param):

#     # % Calculate and visualize the area distortion log(area_map/area_v)
#     # % 
#     # % Input:
#     # % v: nv x 3 vertex coordinates of a genus-0 triangle mesh
#     # % f: nf x 3 triangulations of a genus-0 triangle mesh
#     # % param: nv x 2 or 3 vertex coordinates of the mapping result
#     # %
#     # % Output:
#     # % distortion: 3*nf x 1 area differences
#     # % 
#     # % If you use this code in your own work, please cite the following paper:
#     # % [1] G. P. T. Choi, Y. Leung-Liu, X. Gu, and L. M. Lui, 
#     # %     "Parallelizable global conformal parameterization of simply-connected surfaces via partial welding."
#     # %     SIAM Journal on Imaging Sciences, 2020.
#     # %
#     # % Copyright (c) 2018-2020, Gary Pui-Tung Choi
#     # % https://scholar.harvard.edu/choi

#     import numpy as np
#     nv = len(v);
#     nv2 = len(param);

#     if nv != nv2:
#         print('Error: The two meshes are of different size.');
#         return []

#     if v.shape[-1] == 1:
#         v = np.vstack([np.real(v), np.imag(v), np.zeros(len(v))]).T
#     if v.shape[-1] == 2:
#         v = np.hstack([v, np.zeros((len(v),1))])

#     if param.shape[-1] == 1:
#         param = np.vstack([np.real(param), np.imag(param), np.zeros(len(param))]).T
#     if param.shape[-1] == 2:
#         param = np.hstack([np.real(param), np.zeros((len(param),1))])

#     # % calculate area of v
#     area_v = _face_area(v,f);
#     # % calculate area of map
#     area_map = _face_area(param,f);
#     # % normalize the total area # have to normalize this.... 
#     v = v*np.sqrt(np.nansum(area_map)/np.nansum(area_v));
#     area_v = _face_area(v,f);
#     # % calculate the area ratio
#     # distortion = np.log(area_map/area_v);
#     distortion = np.log(area_map) - np.log(area_v) # might be more stable. 

#     return distortion 
#     # % histogram
#     # figure;
#     # histogram(distortion,30);
#     # xlim([-5 5])
#     # title('Area Distortion');

#     # xlabel('log(final area/initial area)')
#     # ylabel('Number of faces')
#     # set(gca,'FontSize',12);

def area_distortion_measure(v1,v2, f):
    r""" Compute the normalised area scaling factor as measure of area distortion between two triangle meshes
    
    .. math::
        \lambda = \frac{A_1}{A_2}

    where :math:`A_1, A_2` are the areas of the mesh after rescaling the meshes by the square root of the respective total surface areas, :math:`S_1,S_2` respectively.

    Parameters
    ----------
    v1 : (n_vertices, 3) array
        vertex coordinates of a triangle mesh 1
    v2 : (n_vertices, 3) array
        vertex coordinates of a triangle mesh 2
    f : (n_faces, 3) array
        the triangulation of both the first and second mesh, given in terms of the vertex indices 

    Returns
    -------  
    ratio : (n_faces, 3) array
        the area distortion factor per face between meshes ``v1`` and ``v2``

    """
    import numpy as np
    import igl 
    
    v1_ = v1/np.sqrt(np.nansum(igl.doublearea(v1,f)/2.))
    v2_ = v2/np.sqrt(np.nansum(igl.doublearea(v2,f)/2.))

    ratio = igl.doublearea(v1_,f)/igl.doublearea(v2_,f)

    return ratio 
    # # % calculate area of v
    # area_v = face_area(v,f);
    # # % calculate area of map
    # area_map = face_area(param,f);
    # # % normalize the total area # have to normalize this.... 
    # v = v*np.sqrt(np.nansum(area_map)/np.nansum(area_v));
    # area_v = face_area(v,f);
    # # % calculate the area ratio
    # # distortion = np.log(area_map/area_v);
    # distortion = np.log(area_map) - np.log(area_v) # might be more stable. 
    # return distortion 

def _finitemean(A):
    """ for avoiding the Inf values caused by division by a very small area
    """
    import numpy as np 
    # % for avoiding the Inf values caused by division by a very small area
    m = np.mean(A[~np.isnan(A)], axis=0) # return mean of columns. 
    return m 

def _stereographic(u):
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

def mobius_area_correction_spherical(v,f,param):
    r""" Find an optimal Mobius transformation for reducing the area distortion of a spherical conformal parameterization using the method in [1]_.

    Parameters
    ----------
    v : (n_vertices,3) array
        vertex coordinates of a genus-0 closed triangle mesh
    f : (n_faces,3) array
        triangulations of the genus-0 closed triangle mesh
    param : (n_vertices,3) array
        vertex coordinates of the spherical conformal parameterization of the mesh given by ``v`` and ``f`` 

    Returns
    -------
    map_mobius : (n_vertices,3) array
        vertex coordinates of the updated spherical conformal parameterization 

    References
    ----------
    .. [1] G. P. T. Choi, Y. Leung-Liu, X. Gu, and L. M. Lui, "Parallelizable global conformal parameterization of simply-connected surfaces via partial welding." SIAM Journal on Imaging Sciences, 2020.

    """
    # % Find an optimal Mobius transformation for reducing the area distortion of a spherical conformal parameterization using the method in [1].
    # %
    # % Input:
    # % v: nv x 3 vertex coordinates of a genus-0 closed triangle mesh
    # % f: nf x 3 triangulations of a genus-0 closed triangle mesh
    # % map: nv x 3 vertex coordinates of the spherical conformal parameterization
    # % 
    # % Output:
    # % map_mobius: nv x 3 vertex coordinates of the updated spherical conformal parameterization
    # % x: the optimal parameters for the Mobius transformation, where
    # %    f(z) = \frac{az+b}{cz+d}
    # %         = ((x(1)+x(2)*1i)*z+(x(3)+x(4)*1i))./((x(5)+x(6)*1i)*z+(x(7)+x(8)*1i))
    # %
    # % If you use this code in your own work, please cite the following paper:
    # % [1] G. P. T. Choi, Y. Leung-Liu, X. Gu, and L. M. Lui, 
    # %     "Parallelizable global conformal parameterization of simply-connected surfaces via partial welding."
    # %     SIAM Journal on Imaging Sciences, 2020.
    # %
    # % Copyright (c) 2019-2020, Gary Pui-Tung Choi
    # % https://scholar.harvard.edu/choi
    import numpy as np 
    import scipy.optimize as spotimize
    # %%
    # % Compute the area with normalization
    area_v = _face_area(v,f); area_v = area_v/np.nansum(area_v);

    # % Project the sphere onto the plane
    p = _stereographic(param);
    z = p[:,0] + 1j*p[:,1] # make into complex. 

    # % Function for calculating the area after the Mobius transformation 
    def area_map(x):

        mobius = ((x[0]+x[1]*1j)*z +(x[2]+x[3]*1j))/((x[4]+x[5]*1j)*z + (x[6]+x[7]*1j)) # this is the mobius transform
        area_mobius = _face_area(_stereographic(np.vstack([np.real(mobius), np.imag(mobius)]).T), f)

        return area_mobius / np.nansum(area_mobius) # normalised area.

    def d_area(x):
        return _finitemean(np.abs(np.log(area_map(x)/area_v)))

    # % Optimization setup
    x0 = np.hstack([1,0,0,0,0,0,1,0]); #% initial guess
    lb = np.hstack([-1,-1,-1,-1,-1,-1,-1,-1])*100; #% lower bound for the parameters
    ub = np.hstack([1,1,1,1,1,1,1,1])*100; #% upper bound for the parameters
    bounds = tuple((lb[ii],ub[ii]) for ii in np.arange(len(lb)))
    
    # options = optimoptions('fmincon','Display','iter');
    # % Optimization (may further supply gradients for better result, not yet implemented)
    # x = fmincon(d_area,x0,[],[],[],[],lb,ub,[],options);
    opt_res = spotimize.minimize(d_area, x0, bounds=bounds)
    x = opt_res.x

    # % obtain the conformal parameterization with area distortion corrected
    fz = ((x[0]+x[1]*1j)*z + (x[2]+x[3]*1j)) / ((x[4]+x[5]*1j)*z+(x[6]+x[7]*1j))
    map_mobius = _stereographic(np.vstack([np.real(fz), np.imag(fz)]).T);

    return map_mobius


def mobius_area_correction_disk(v,f,param):
    r""" Find an optimal Mobius transformation for reducing the area distortion of a disk conformal parameterization using the method in [1]_.

    Parameters
    ----------
    v : (n_vertices,3) array
        vertex coordinates of a simply-connected open triangle mesh
    f : (n_faces,3) array
        triangulations of a simply-connected open triangle mesh
    param : (n_vertices,2) array
        vertex coordinates of the disk conformal parameterization of the mesh given by ``v`` and ``f`` 

    Returns
    -------
    map_mobius_disk : (n_vertices,2) array
        vertex coordinates of the updated disk conformal parameterization 

    References
    ----------
    .. [1] G. P. T. Choi, Y. Leung-Liu, X. Gu, and L. M. Lui, "Parallelizable global conformal parameterization of simply-connected surfaces via partial welding." SIAM Journal on Imaging Sciences, 2020.
    
    """
    # % Find an optimal Mobius transformation for reducing the area distortion of a disk conformal parameterization using the method in [1].
    # %
    # % Input:
    # % v: nv x 3 vertex coordinates of a simply-connected open triangle mesh
    # % f: nf x 3 triangulations of a simply-connected open triangle mesh
    # % map: nv x 2 vertex coordinates of the disk conformal parameterization
    # % 
    # % Output:
    # % map_mobius_disk: nv x 2 vertex coordinates of the updated disk conformal parameterization
    # % x: the optimal parameters for the Mobius transformation, where
    # %    f(z) = \frac{z-a}{1-\bar{a} z}
    # %    x(1): |a| (0 ~ 1)        magnitude of a
    # %    x(2): arg(a) (0 ~ 2pi)   argument of a
    # %
    # % If you use this code in your own work, please cite the following paper:
    # % [1] G. P. T. Choi, Y. Leung-Liu, X. Gu, and L. M. Lui, 
    # %     "Parallelizable global conformal parameterization of simply-connected surfaces via partial welding."
    # %     SIAM Journal on Imaging Sciences, 2020.
    # %
    # % Copyright (c) 2019-2020, Gary Pui-Tung Choi
    # % https://scholar.harvard.edu/choi
    
    import numpy as np 
    import scipy.optimize as spotimize
    # % Compute the area with normalization
    area_v = _face_area(v,f); area_v = area_v/float(np.nansum(area_v));

    z = param[:,0] + 1j*param[:,1]

    # % Function for calculating the area after the Mobius transformation 
    def area_map(x):
        v_mobius = np.vstack([np.real((z-x[0]*np.exp(1j*x[1]))/(1.-np.conj(x[0]*np.exp(1j*x[1]))*z)), np.imag((z-x[0]*np.exp(1j*x[1]))/(1.-np.conj(x[0]*np.exp(1j*x[1]))*z))]).T
        area_mobius = _face_area( v_mobius, f )
        return area_mobius / np.nansum(area_mobius) # normalised area.

    # % objective function: mean(abs(log(area_map./area_v)))
    def d_area(x):
        return _finitemean(np.abs(np.log(area_map(x)/area_v)))

    # % Optimization setup
    x0 = np.hstack([0,0]); #% initial guess, try something diferent if the result is not good
    lb = np.hstack([0,0]); #% lower bound for the parameters
    ub = np.hstack([1,2*np.pi]); #% upper bound for the parameters
    bounds = tuple((lb[ii],ub[ii]) for ii in np.arange(len(lb)))

    opt_res = spotimize.minimize(d_area, x0, bounds=bounds)
    x = opt_res.x
    
    # % obtain the conformal parameterization with area distortion corrected
    fz = (z-x[1]*np.exp(1j*x[1]))/(1.-np.conj(x[0]*np.exp(1j*x[1]))*z);
    map_mobius_disk = np.vstack([np.real(fz), np.imag(fz)]).T;

    return map_mobius_disk


def _generalized_laplacian(v,f,mu):
    # function A = generalized_laplacian(v,f,mu)
    # % Compute the generalized Laplacian.
    # % 
    # % If you use this code in your own work, please cite the following paper:
    # % [1] T. W. Meng, G. P.-T. Choi and L. M. Lui, 
    # %     "TEMPO: Feature-Endowed Teichmüller Extremal Mappings of Point Clouds."
    # %     SIAM Journal on Imaging Sciences, 9(4), pp. 1922-1962, 2016.
    # %
    # % Copyright (c) 2015-2018, Gary Pui-Tung Choi
    # % https://scholar.harvard.edu/choi

    import numpy as np 
    import scipy.sparse as sparse
    
    af = (1.-2*mu.real+np.abs(mu)**2)/(1.0-np.abs(mu)**2);
    bf = -2.*mu.imag/(1.0-np.abs(mu)**2);
    gf = (1.+2*mu.real+np.abs(mu)**2)/(1.0-np.abs(mu)**2);

    f0 = f[:,0].copy(); 
    f1 = f[:,1].copy(); 
    f2 = f[:,2].copy();

    uxv0 = v[f1,1] - v[f2,1];
    uyv0 = v[f2,0] - v[f1,0];
    uxv1 = v[f2,1] - v[f0,1];
    uyv1 = v[f0,0] - v[f2,0]; 
    uxv2 = v[f0,1] - v[f1,1];
    uyv2 = v[f1,0] - v[f0,0];

    l = np.vstack([np.sqrt(uxv0**2 + uyv0**2), 
                   np.sqrt(uxv1**2 + uyv1**2), 
                   np.sqrt(uxv2**2 + uyv2**2)]).T;
    s = np.sum(l,1)*0.5;

    area = np.sqrt(s*(s-l[:,0])*(s-l[:,1])*(s-l[:,2]) + 1e-12) #+ 1e-12;
    
    v00 = (af*uxv0*uxv0 + 2*bf*uxv0*uyv0 + gf*uyv0*uyv0)/area;
    v11 = (af*uxv1*uxv1 + 2*bf*uxv1*uyv1 + gf*uyv1*uyv1)/area;
    v22 = (af*uxv2*uxv2 + 2*bf*uxv2*uyv2 + gf*uyv2*uyv2)/area;
    v01 = (af*uxv1*uxv0 + bf*uxv1*uyv0 + bf*uxv0*uyv1 + gf*uyv1*uyv0)/area;
    v12 = (af*uxv2*uxv1 + bf*uxv2*uyv1 + bf*uxv1*uyv2 + gf*uyv2*uyv1)/area;
    v20 = (af*uxv0*uxv2 + bf*uxv0*uyv2 + bf*uxv2*uyv0 + gf*uyv0*uyv2)/area;
    
    I = np.hstack([f0,f1,f2,f0,f1,f1,f2,f2,f0])
    J = np.hstack([f0,f1,f2,f1,f0,f2,f1,f0,f2])
    V = np.hstack([v00,v11,v22,v01,v01,v12,v12,v20,v20])/2.
    
    A = sparse.csr_matrix((-V, (I,J)), shape=((np.max(I)+1, np.max(J)+1)))
        
    return A 


def rectangular_conformal_map(v,f,corner=None, map2square=False, random_state=0, return_bdy_index=False):
    r""" Compute the rectangular conformal mapping using the fast method in [1]_. This first maps the open mesh to a disk then from a disk to the rectangle. 
    
    Parameters
    ----------
    v : (n_vertices, 3) array 
        vertex coordinates of a simply-connected open triangle mesh
    f : (n_faces, 3) array 
        triangulations of a simply-connected open triangle mesh
    corner : (4,) array
        optional input for specifying the exact 4 vertex indices for the four corners of the final rectangle, with anti-clockwise orientation
    map2square : bool 
        if True, do the rectangular conformal map, else if False, return the intermediate Harmonic disk parametrization which is much faster. 
    random_state : int
        if corner is None, this is a random seed that randomly picks the 4 corners of the final rectangle from the input vertices. 
    return_bdy_index : bool
        if True, returns additionally the indices of the vertex that form the boundary of the input triangle mesh 

    Returns
    -------
    map_ : (n_vertices,2) array
        vertex coordinates of the rectangular conformal parameterization if map2square=True of the harmonic disk conformal parametrization if map2square=False. 
    h_opt : scalar 
        if map2square=True, return the optimal y-coordinate scaling factor to have the lowest Beltrami coefficient in the rectangular conformal map 
    bdy_index : 
        if return_bdy_index=True, return as the last output the vertex indices that form the boundary of the input triangle mesh 
    
    References
    ----------
    .. [1] T. W. Meng, G. P.-T. Choi and L. M. Lui, "TEMPO: Feature-Endowed Teichmüller Extremal Mappings of Point Clouds." SIAM Journal on Imaging Sciences, 9(4), pp. 1922-1962, 2016.

    Notes
    -----
    1. Please make sure that the input mesh does not contain any unreferenced vertices/non-manifold vertices/non-manifold edges.
    2. Please remove all valence 1 boundary vertices (i.e. vertices with only 1 face attached to them) before running the program.
    3. Please make sure that the input triangulations f are with anti-clockwise orientation.
    4. The output rectangular domain will always have width = 1, while the height depends on the choice of the corners and may not be 1. (The Riemann mapping theorem guarantees that there exists a conformal map from any simple-connected open surface to the unit square, but if four vertices on the surface boundary are specified to be the four corners of the planar domain, the theorem is no longer applicable.)
    
    """
    # corner is given as an index of the original boundary. 
    import trimesh
    import igl 
    import numpy as np 
    import scipy.sparse as sparse
    from scipy.optimize import fminbound
    # import time 
    
    nv = len(v);
    bdy_index = igl.boundary_loop(f)
    
    if corner is None: 
        # just pick 4 regularly sampled indices on the boundary 
        if random_state is not None:
            np.random.seed(random_state)
        corner = bdy_index[(np.linspace(0, len(bdy_index)-1, 5)[:4]).astype(np.int)]
    
    # % rearrange the boundary indices to be correct anticlockwise. 
    id1 = np.arange(len(bdy_index))[bdy_index==corner[0]]
    if len(id1)>0:
        id1 = id1[0]
    # re-index. 
    bdy_index = np.hstack([bdy_index[id1:], bdy_index[:id1]]);
    # relabel 
    id1 = 0;
    id2 = np.arange(len(bdy_index))[bdy_index==corner[1]]
    id3 = np.arange(len(bdy_index))[bdy_index==corner[2]]
    id4 = np.arange(len(bdy_index))[bdy_index==corner[3]]
    
    id2 = id2[0]
    id3 = id3[0]
    id4 = id4[0]
    
    # print(id1,id2,id3,id4)
    # print('=====')
    # %% Step 1: Mapping the input mesh onto the unit disk
    bdy_length = np.sqrt((v[bdy_index,0] - v[np.hstack([bdy_index[1:], bdy_index[0]]),0])**2 + 
                         (v[bdy_index,1] - v[np.hstack([bdy_index[1:], bdy_index[0]]),1])**2 + 
                         (v[bdy_index,2] - v[np.hstack([bdy_index[1:], bdy_index[0]]),2])**2);
    # partial_edge_sum = np.zeros(len(bdy_length));
    partial_edge_sum = np.cumsum(bdy_length)
    
    # # % arc-length parameterization boundary constraint
    # theta = 2*np.pi*partial_edge_sum/np.sum(bdy_length); # theta. 
    # bdy = np.exp(theta*1j); # r
    
    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(v, bdy_index)

    # % 1. disk harmonic map
    disk = igl.harmonic_weights(v.astype(np.float),
                              f, 
                              bdy_index, 
                              bnd_uv, 
                              1); # if 2 then biharmonic  

    if map2square:
        # then do conformal mapping to square
        # return disk, [id1,id2,id3,id4], bdy_index
        # if sum(sum(isnan(disk))) ~= 0
        #     % use tutte embedding instead
        #     disk = tutte_map(v,f,bdy_index,bdy); 
        # end
        
        # the below is super slow -> try to find a faster way. 

        # %% Step 2: Mapping the unit disk to the unit square [ super slow ... construction...]
        # % compute the generalized Laplacian
        mu = beltrami_coefficient(disk, f, v)
        # mu = beltrami_coefficient(disk,f,v);
        Ax = _generalized_laplacian(disk,f,mu); # ok....-> this does look like just the degree matrix? 
        Ay = Ax.copy();
        
        # % set the boundary constraints
        bottom = bdy_index[id1:id2+1];
        right = bdy_index[id2:id3+1];
        top = bdy_index[id3:id4+1];
        left = np.hstack([bdy_index[id4:], bdy_index[0]])
        # print(bottom, right, top, left)
        
        Ax = sparse.lil_matrix(Ax); # convert to this first... 
        Ay = sparse.lil_matrix(Ay);
        
        bx = np.zeros(nv); by = bx.copy();
        Ax[np.hstack([left,right]),:] = 0;
        Ax[np.hstack([left,right]), np.hstack([left,right])[:,None]] = np.diag(np.ones(len(np.hstack([left,right]))));
        # this diag sounds like just putting ones... 
        # Ax[np.hstack([left,right]), np.hstack([left,right])] = 1
        
        bx[right] = 1;
        Ay[np.hstack([top,bottom]),:] = 0;
        Ay[np.hstack([top,bottom]), np.hstack([top,bottom])[:,None]] = np.diag(np.ones(len(np.hstack([top,bottom]))));
        # Ay[np.hstack([top,bottom]), np.hstack([top,bottom])] = 1
        by[top] = 1;
        
        Ax = sparse.csr_matrix(Ax);
        Ay = sparse.csr_matrix(Ay);
        
        # % solve the generalized Laplace equation
        square_x = sparse.linalg.spsolve(Ax, bx);
        square_y = sparse.linalg.spsolve(Ay, by);
        # square_x = sparse.linalg.cg(Ax, bx)[0]; # no solve? 
        # square_y = sparse.linalg.cg(Ay, by)[0];
        # print(square_x.max())
        # print(square_y.max())

        # %% Step 3: Optimize the height of the square to achieve a conformal map
        def func(h):
            # here. 
            return np.sum(np.abs(beltrami_coefficient(np.hstack([square_x[:,None],h*square_y[:,None]]),f,v))**2)
        
        h_opt = fminbound(func, 0,5);
        
        map_ = np.vstack([square_x, h_opt*square_y]).T;
        # map_ = np.vstack([square_x, square_y]).T;
        
        if return_bdy_index:
            return map_, h_opt, bdy_index
        else:
            return map_, h_opt

    else:
        map_ = disk.copy()

        if return_bdy_index:
            return map_, bdy_index
        else:
            # just return disk 
            return map_    


def f2v(v,f):
    r""" Compute the face to vertex interpolation matrix taking into account unequal lengths. 

    Parameters
    ----------
    v : (n_vertices,3) array
        vertex coordinates of a triangle mesh
    f : (n_faces, 3) array 
        triangulations of a triangle mesh

    Returns
    -------
    S : (n_vertices, n_faces) sparse array
        the face to vertex matrix such that S.dot(face_values), gives the interpolated vertex values equivalent 
    """
    """
    Compute the face to vertex interpolation matrix. of 
    % [1] P. T. Choi and L. M. Lui, 
    %     "Fast Disk Conformal Parameterization of Simply-Connected Open Surfaces."
    %     Journal of Scientific Computing, 65(3), pp. 1065-1090, 2015.
    """
    import trimesh
    import scipy.sparse as spsparse
    import numpy as np 
    
    mesh = trimesh.Trimesh(vertices=v,
                           faces=f, 
                           process=False,
                           validate=False)
    
    S = mesh.faces_sparse.copy() #.dot(rgba.astype(np.float64))
    degree = mesh.vertex_degree
    nonzero = degree > 0
    normalizer = np.zeros(degree.shape)
    normalizer[nonzero] = 1./degree[nonzero]
    D = spsparse.spdiags(np.squeeze(normalizer), [0], len(normalizer), len(normalizer))
    S = D.dot(S)

    return S 

def disk_conformal_map(v,f,corner=None, random_state=0, north=5, south=100, threshold=0.00001, max_iter=5):
    r""" Compute the disk conformal mapping using the method in [1].

    Parameters
    ----------
    v : (n_vertices,3) array 
        vertex coordinates of a simply-connected open triangle mesh
    f : (n_faces,3) array
        triangulations of a simply-connected open triangle mesh
    corner : (4,) array
        optional input for specifying the exact 4 vertex indices for rearranging the boundary index, with anti-clockwise orientation
    random_state : int
        if corner is None, this is a random seed that randomly picks the 4 corners for rearranging the boundary index. 
    north : int
        scalar for fixing the north pole iterations
    south : 
        scalar for fixing the south pole iterations
    threshold : scalar
        convergence threshold between the old and new energy cost per iteration in the Beltrami coefficient optimization. 
    max_iter : int
        the maximum number of Beltrami coefficient optimization 

    Returns
    -------
    disk_new : (n_vertices,2) array
        vertex coordinates of the updated disk conformal parameterization starting from an initial harmonic disk parametrization 
    
    References
    ----------
    .. [1] P. T. Choi and L. M. Lui, "Fast Disk Conformal Parameterization of Simply-Connected Open Surfaces." Journal of Scientific Computing, 65(3), pp. 1065-1090, 2015.

    Notes
    -----
    1. Please make sure that the input mesh does not contain any unreferenced vertices/non-manifold vertices/non-manifold edges.
    2. Please remove all valence 1 boundary vertices (i.e. vertices with only 1 face attached to them) before running the program.

    """
    # % Compute the disk conformal mapping using the method in [1].
    # %
    # % Input:
    # % v: nv x 3 vertex coordinates of a simply-connected open triangle mesh
    # % f: nf x 3 triangulations of a simply-connected open triangle mesh
    # % 
    # % Output:
    # % map: nv x 2 vertex coordinates of the disk conformal parameterization
    # % 
    # % Remark:
    # % 1. Please make sure that the input mesh does not contain any 
    # %    unreferenced vertices/non-manifold vertices/non-manifold edges.
    # % 2. Please remove all valence 1 boundary vertices (i.e. vertices with 
    # %    only 1 face attached to them) before running the program.
    # % 
    # % If you use this code in your own work, please cite the following paper:
    # % [1] P. T. Choi and L. M. Lui, 
    # %     "Fast Disk Conformal Parameterization of Simply-Connected Open Surfaces."
    # %     Journal of Scientific Computing, 65(3), pp. 1065-1090, 2015.
    # %
    # % Copyright (c) 2014-2018, Gary Pui-Tung Choi
    # % https://scholar.harvard.edu/choi

    # corner is given as an index of the original boundary. 
    import trimesh
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from scipy.optimize import fminbound
    import pylab as plt 
    # import time 
    

    """
    Stage 1: obtain a harmonic map initialization. 
    """
    nv = len(v);
    bdy_index = igl.boundary_loop(f)
    
    if corner is None: 
        # just pick 4 regularly sampled indices on the boundary 
        if random_state is not None:
            np.random.seed(random_state)
        corner = bdy_index[(np.linspace(0, len(bdy_index)-1, 5)[:4]).astype(np.int)]
    
    # % rearrange the boundary indices to be correct anticlockwise. 
    id1 = np.arange(len(bdy_index))[bdy_index==corner[0]]
    if len(id1)>0:
        id1 = id1[0]
    # re-index. 
    bdy_index = np.hstack([bdy_index[id1:], bdy_index[:id1]]);
    # relabel 
    id1 = 0;
    id2 = np.arange(len(bdy_index))[bdy_index==corner[1]]
    id3 = np.arange(len(bdy_index))[bdy_index==corner[2]]
    id4 = np.arange(len(bdy_index))[bdy_index==corner[3]]
    
    id2 = id2[0]
    id3 = id3[0]
    id4 = id4[0]
    
    # print(id1,id2,id3,id4)
    # print('=====')
    # %% Step 1: Mapping the input mesh onto the unit disk
    bdy_length = np.sqrt((v[bdy_index,0] - v[np.hstack([bdy_index[1:], bdy_index[0]]),0])**2 + 
                         (v[bdy_index,1] - v[np.hstack([bdy_index[1:], bdy_index[0]]),1])**2 + 
                         (v[bdy_index,2] - v[np.hstack([bdy_index[1:], bdy_index[0]]),2])**2);
    # partial_edge_sum = np.zeros(len(bdy_length));
    partial_edge_sum = np.cumsum(bdy_length)
    
    # # % arc-length parameterization boundary constraint
    # theta = 2*np.pi*partial_edge_sum/np.sum(bdy_length); # theta. 
    # bdy = np.exp(theta*1j); # r
    
    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(v, bdy_index)

    # % 1. disk harmonic map
    disk = igl.harmonic_weights(v.astype(np.float),
                              f, 
                              bdy_index, 
                              bnd_uv, 
                              1); # if 2 then biharmonic  

    z = disk[:,0]+disk[:,1]*1j
    ### the remainder is optimization. 
    # %% North Pole iteration
    # % Use the Cayley transform to map the disk to the upper half plane
    # % All boundary points will be mapped to the real line
    mu = beltrami_coefficient(disk, f, v); 
    mu_v = f2v(v,f).dot(mu); # map this from face to vertex. 
    bdy_index_temp = np.hstack([bdy_index[1:], bdy_index[0]]); 
    least = np.argmin(np.abs(mu_v[bdy_index])+np.abs(mu_v[bdy_index_temp]))
    z = z*np.exp(-1j*(np.angle(z[bdy_index[least]])+np.angle(z[bdy_index[np.mod(least,len(bdy_index))]])/2.)); # this is not giving the same as matlab? 
    g = 1j*(1. + z)/(1. - z); 

    #fix the points near the puncture, i.e. near z = 1
    ind = np.argsort(-np.real(z));
    fixed = np.setdiff1d(ind[:np.max([int(np.round(float(len(v))/north)),np.min([100,len(z)-1])])], bdy_index);
    # fixed = [fixed; find(real(g) == max(real(g))); find(real(g) == min(real(g)))];
    fixed = np.hstack([fixed, np.argmax(np.real(g)), np.argmin(np.real(g))])
    P = np.vstack([np.real(g), 
                   np.imag(g), 
                   np.ones(len(g))]).T
    mu = beltrami_coefficient(P, f, v); 
    
    #compute the updated x coordinates
    target = P[fixed,0];
    A = _generalized_laplacian(P,f,mu); # is this more efficient by changing to a column structure. 
    A = spsparse.lil_matrix(A)
    Ax = A.copy(); Ay = A.copy();

    b = -Ax[:,fixed].dot(target);
    b[fixed] = target;
    Ax[fixed,:] = 0; Ax[:,fixed] = 0;
    Ax = Ax.tocsr()
    Ax = Ax + spsparse.csr_matrix((np.ones(len(fixed)), (fixed,fixed)), shape=(A.shape[0], A.shape[1]));
    x = spsparse.linalg.spsolve(Ax,b)

    #compute the updated y coordinates
    target = P[fixed,1];
    fixed = np.hstack([fixed, bdy_index])
    target = np.hstack([target, np.zeros(len(bdy_index))]);
    b = -Ay[:,fixed].dot(target);
    b[fixed] = target;
    Ay[fixed,:] = 0; Ay[:,fixed] = 0;
    Ay = Ay.tocsr()
    Ay = Ay + spsparse.csr_matrix((np.ones(len(fixed)), (fixed,fixed)), shape=(A.shape[0], A.shape[1]));
    y = spsparse.linalg.spsolve(Ay,b)

    g_new = x+y*1j
    z_new = (g_new - 1j)/(g_new + 1j);
    disk_new = np.vstack([np.real(z_new), np.imag(z_new)]).T;

    if np.sum(np.isnan(disk_new)) != 0: 
        #% use the old result in case of getting NaN entries
        disk_new = disk.copy();
        z_new = disk_new[:,0] + 1j*disk_new[:,2];

    print('North pole step completed.\n')

    """
    reflection in the unit disk to a triangle. 
    """
    f_temp = f + len(v);
    a = np.sort(bdy_index + len(v)); # this is an actual sort!!! 
    for i in np.arange(len(a)-1, -1, -1):
        f_temp[f_temp == a[i]] = a[i] - len(v);
        f_temp = f_temp - (f_temp > a[i]);
    f_filled = np.vstack([f, np.fliplr(f_temp)]);

    z_filled = np.hstack([z_new, 1./np.conj(z_new)]);
    select = np.setdiff1d(np.arange(len(z_filled)), bdy_index + len(v))
    z_filled = z_filled[select] # = [];

    energy_old = 0;
    energy = np.mean(np.abs(beltrami_coefficient(np.vstack([np.real(z_new), np.imag(z_new), np.zeros(len(z_new))]).T, f, v)));

    iteration_count = 1; 
    map_opt = disk_new.copy();
    
    print('Reflection completed.\n')


    while np.abs(energy_old-energy) > threshold:

        energy_old = energy;

        mu = beltrami_coefficient(np.vstack([np.real(z_new), np.imag(z_new), np.zeros(len(z_new))]).T, f, v); # vector.         
        mu_filled = np.hstack([mu, 
                               1./3*((z_new[f[:,0]]/(np.conj(z_new[f[:,0]])))**2 + 
                                     (z_new[f[:,1]]/(np.conj(z_new[f[:,1]])))**2 + 
                                     (z_new[f[:,2]]/(np.conj(z_new[f[:,2]])))**2)*np.conj(mu)/ np.abs(((z_new[f[:,0]]/(np.conj(z_new[f[:,0]])))**2 + 
                                                                                                       (z_new[f[:,1]]/(np.conj(z_new[f[:,1]])))**2 + 
                                                                                                       (z_new[f[:,2]]/(np.conj(z_new[f[:,2]])))**2))]); # doubles in length. 
        
        # % fix the points near infinity
        ind = np.argsort(-np.abs(z_filled))
        fixed2 = ind[:np.max([int(np.round(float(len(v))/south)), np.min([100,len(z)-1])])];
        map_filled = linear_beltrami_solver(np.vstack([np.real(z_filled), np.imag(z_filled),  np.zeros(len(z_filled))]).T, 
                                            f_filled, mu_filled,
                                            fixed2, np.vstack([np.real(z_filled[fixed2]), np.imag(z_filled[fixed2])]).T);

        z_big = map_filled[:,0] + 1j*map_filled[:,1]
        z_final = z_big[:len(v)].copy();
        
        # % normalization
        z_final = z_final - np.mean(z_final); #% move centroid to zero
        if np.max(np.abs(z_final))>1:
            z_final = z_final/(np.max(np.abs(z_final))); #% map it into unit circle

        mu_temp = beltrami_coefficient(np.vstack([np.real(z_final), np.imag(z_final), np.zeros(len(z_final))]).T , f,v);
        map_temp = linear_beltrami_solver(np.vstack([np.real(z_final), np.imag(z_final), np.zeros(len(z_final))]).T, 
                                          f, 
                                          mu_temp,     
                                          bdy_index, 
                                          np.vstack([np.real(z_final[bdy_index]/np.abs(z_final[bdy_index])), 
                                                    np.imag(z_final[bdy_index]/np.abs(z_final[bdy_index]))]).T);

        z_new = map_temp[:,0] + 1j*map_temp[:,1];

        z_filled = np.hstack([z_new, 1./np.conj(z_new)]);
        select = np.setdiff1d(np.arange(len(z_filled)), bdy_index + len(v))
        z_filled = z_filled[select] # = [];


        disk_iter = np.vstack([np.real(z_new), np.imag(z_new)]).T;
        
        # # return disk_new
        # # return disk_new
        # plt.figure(figsize=(10,10))
        # plt.title('iter disk')
        # plt.plot(disk[:,0], disk[:,1], 'r.', ms=.1)
        # plt.plot(disk_new[:,0], disk_new[:,1], 'g.', ms=.1)
        # plt.show()
        
        if np.sum(np.isnan(disk_iter)) != 0: 
            # % use the previous result in case of getting NaN entries
            disk_iter = map_opt.copy(); 
            # disk_iter[:,0] = -disk_iter[:,0].copy(); # don't get this 

        energy = np.mean(np.abs(beltrami_coefficient(disk_iter, f, v)));
        map_opt = disk_iter.copy();
      
        print('Iteration %d: mean(|mu|) = %.4f\n' %(iteration_count, energy));
        iteration_count = iteration_count+1;
        
        if iteration_count > max_iter:
            # % it usually converges within 5 iterations so we set 5 here
            break;


    disk_new = map_opt.copy()
    # disk_new[:,0] = -disk_new[:,0].copy()

    # # return disk_new
    # plt.figure(figsize=(10,10))
    # plt.title('final disk')
    # plt.plot(disk[:,0], disk[:,1], 'r.', ms=.1)
    # plt.plot(disk_new[:,0], disk_new[:,1], 'g.', ms=.1)
    # plt.show()

    return disk_new


def _squicircle(uv, _epsilon = 0.0000000001):

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
    
    return np.vstack([x,y]).T


def _elliptical_nowell(uv):

    #https://squircular.blogspot.com/2015/09/mapping-circle-to-square.html
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

    return np.vstack([x,y]).T

def find_and_loc_corner_rect_open_surface(mesh, vol_shape, ref_depth=0, order='cc', curvature_flow=True, delta_flow=1e3, flow_iters=50):
    r""" Find and locate the 4 corners of a rectangular topography mesh. Curvature flow of the boundary is used to identify the corners fast and accurately

    Parameters
    ----------
    mesh : trimesh.Trimesh
        a simply-connected open triangle topography mesh
    vol_shape : (D,U,V) tuple
        the shape of the topography volume space the topography mesh comes from         
    ref_depth : int 
        if curvature_flow=False, this is the depth coordinate of the topography mesh used to locate corners (implicitly assuming all corners are of equal depth) 
    order : 'cc' or 'acc'    
        specifies whether the input mesh has faces oriented 'cc'-clockwise or 'acc'-anticlockwise
    curvature_flow : bool
        if True, uses curvature flow of the boundary to help locate the corners of the topography mesh. This is most accurate. If False, the corners will attempt to be found by idealistic matching to 4 corners constructed by ref_depth and the 4 corners of the image grid spanned by vertices of the input mesh
    delta_flow : scalar 
        specifies the speed of flow if curvature_flow=True. Higher flow gives faster convergence. 
    flow_iters : int
        specifies the number of iterations of curvature flow. Higher will give more flow 

    Returns
    -------
    bnd : (n,) array
        the vertex indices of the boundary of the topography mesh 
    corner_bnd_ind : (4,) array
        the indices of ``bnd`` specifying the 4 corners in anti-clockwise order
    corner_v_ind : (4,) array
        the vertex indices of the input mesh specifying the 4 corners in anti-clockwise order

    """
    import numpy as np 
    import igl 
    
    # make copy of the input mesh. 
    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    

    # =============================================================================
    #   3) Fix the boundary of the surface unwrapping 
    # =============================================================================

    d, m, n = vol_shape[:3]
    
    if order=='cc':
        # the 4 corners of the image. 
        corner1yx = np.hstack([0,0]) 
        corner2yx = np.hstack([m-1,0])
        corner3yx = np.hstack([m-1,n-1])
        corner4yx = np.hstack([0,n-1])

    if order =='acc':
        # the 4 corners of the image. 
        corner1yx = np.hstack([0,0]) 
        corner2yx = np.hstack([0,n-1])
        corner3yx = np.hstack([m-1,n-1])
        corner4yx = np.hstack([m-1,0])
    

    ## convert the coordinates to the indices of the open boundary
    bnd = igl.boundary_loop(f) # vertex index. 
    bnd_vertex = v[bnd].copy()
    bnd_index = np.arange(len(bnd))

    # initial algorithm which fails to properly constrain the x-y plane 
    # the problem here is because of the curvature of the curve... -> to be fully accurate we should do curvature flow of the bounary edge line!.
    if curvature_flow:
        bnd_vertex_evolve = conformalized_mean_line_flow( bnd_vertex, 
                                                             E=None, 
                                                             close_contour=True, 
                                                             fixed_boundary = False, 
                                                             lambda_flow=delta_flow, 
                                                             niters=flow_iters, 
                                                             topography_edge_fix=True, 
                                                             conformalize=True)
        # we then solve for the index on the evolved boundary!. 
        min1_ind = np.argmin(np.linalg.norm(bnd_vertex_evolve[...,-1] - np.hstack([ref_depth, corner1yx])[None,:], axis=-1))
        min2_ind = np.argmin(np.linalg.norm(bnd_vertex_evolve[...,-1] - np.hstack([ref_depth, corner2yx])[None,:], axis=-1))
        min3_ind = np.argmin(np.linalg.norm(bnd_vertex_evolve[...,-1] - np.hstack([ref_depth, corner3yx])[None,:], axis=-1))
        min4_ind = np.argmin(np.linalg.norm(bnd_vertex_evolve[...,-1] - np.hstack([ref_depth, corner4yx])[None,:], axis=-1))
    else: 
        min1_ind = np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner1yx])[None,:], axis=-1))
        min2_ind = np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner2yx])[None,:], axis=-1))
        min3_ind = np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner3yx])[None,:], axis=-1))
        min4_ind = np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner4yx])[None,:], axis=-1))

    pts1_ind_bnd = bnd_index[min1_ind]
    pts2_ind_bnd = bnd_index[min2_ind]
    pts3_ind_bnd = bnd_index[min3_ind]
    pts4_ind_bnd = bnd_index[min4_ind]

    pts1_ind_v = bnd[min1_ind]
    pts2_ind_v = bnd[min2_ind]
    pts3_ind_v = bnd[min3_ind]
    pts4_ind_v = bnd[min4_ind]
# =============================================================================
#    3) rectangular conformal map -> first maps to the disk  
# =============================================================================
    # stack the corners. 
    corner_bnd_ind = np.hstack([pts1_ind_bnd, pts2_ind_bnd, pts3_ind_bnd, pts4_ind_bnd])
    corner_v_ind = np.hstack([pts1_ind_v, pts2_ind_v, pts3_ind_v, pts4_ind_v])

    return bnd, corner_bnd_ind, corner_v_ind


def reconstruct_border_inds(all_border_inds, corner_inds):
    r""" Given an ordered list of corner indices within an array of boundary indices specifying a closed loop, construct the continuous line segments linking the corner points     

    Parameters
    ----------
    all_border_inds : (N,) array
        array of vertex indices specifying the boundary of a mesh
    corner_inds : (n_corners,) array
        array specifying which indices of ``all_border_inds`` are 'corners'. This should be ordered such that corner_inds[0]:corner_inds[1] constitute a continuous segment.

    Returns
    -------
    segs : list of n_corners+1 arrays 
        a list of all the continuous boundary segments between consecutive corners 

    """
    import numpy as np 

    corner_inds_ = np.hstack([corner_inds, corner_inds[0]])
    # print(corner_inds_)
    N = len(corner_inds_)
    segs = []

    for ii in np.arange(N-1):
        start = corner_inds_[ii] 
        end = corner_inds_[ii+1]
        # print(start,end)
        if end > start:
            inds = np.arange(start,end,1)
        else:
            inds = np.hstack([np.arange(start, len(all_border_inds),1), 
                              np.arange(0, end)])
        segs.append(inds)
    return segs


def flat_open_surface(mesh, vol_shape, map2square=False, square_method='elliptical', 
                        ref_depth=0, 
                        order='cc', 
                        curvature_flow=True, 
                        delta_flow=1e3, 
                        flow_iters=50,
                        optimize=True):
    r""" Main wrapping function to unwrap an open 3D mesh, primarily a topography into 2D disk, or 2D rectangle (continuing from the 2D disk)   

    Parameters
    ----------
    mesh : trimesh.Trimesh
        a simply-connected open triangle topography mesh
    vol_shape : (M,N,L) tuple
        the shape of the volume space the topography mesh comes from 
    map2square : bool
        If True, continue to map the disk to the square or conformal rectangle with options specified by ``square_method``. If False or if square_method=None, the intermediate disk parameterization is returned 
    square_method : str
        One of 'Teichmuller' for conformal rectangular mapping, 'squicircle' for squicircle squaring of disk to square, 'elliptical' for elliptical mapping of Nowell of disk to square. 'Teichmuller' is slow but conformal minimizing.   
    ref_depth : int
        if curvature_flow=False, this is the depth coordinate of the topography mesh used to locate corners (implicitly assuming all corners are of equal depth) 
    order : 'cc' or 'acc'
        specifies whether the input mesh has faces oriented 'cc'-clockwise or 'acc'-anticlockwise
    optimize : bool
        if True, applies Beltrami coefficient optimization to compute the rectangular aspect ratio to minimize distortion given the square_method='squicircle' and square_method='elliptical' options. Teichmuller by default will have this option enabled.

    Returns
    -------
    square : (n_vertices, 2)
        the disk or square parametrization of the input mesh 

    """
    import numpy as np 
    import igl 
    from scipy.optimize import fminbound

    # make copy of the input mesh. 
    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    

    # =============================================================================
    #   3) Fix the boundary of the surface unwrapping 
    # =============================================================================
    # d, m, n = vol_shape[:3]
    
    # if order=='cc':
    #     # the 4 corners of the image. 
    #     corner1yx = np.hstack([0,0]) 
    #     corner2yx = np.hstack([m-1,0])
    #     corner3yx = np.hstack([m-1,n-1])
    #     corner4yx = np.hstack([0,n-1])

    # if order =='acc':
    #     # the 4 corners of the image. 
    #     corner1yx = np.hstack([0,0]) 
    #     corner2yx = np.hstack([0,n-1])
    #     corner3yx = np.hstack([m-1,n-1])
    #     corner4yx = np.hstack([m-1,0])
    

    # ## convert the coordinates to the indices of the open boundary
    # bnd = igl.boundary_loop(f) # vertex index. 
    # bnd_vertex = v[bnd].copy()

    # """
    # to do: replace this with the actual boundary.... 
    # """
    # # match by distance rather than exact match. 
    # pts1_ind = bnd[np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner1yx])[None,:], axis=-1))]
    # pts2_ind = bnd[np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner2yx])[None,:], axis=-1))]
    # pts3_ind = bnd[np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner3yx])[None,:], axis=-1))]
    # pts4_ind = bnd[np.argmin(np.linalg.norm(bnd_vertex - np.hstack([ref_depth, corner4yx])[None,:], axis=-1))]
    bnd, corner_bnd_ind, _ = find_and_loc_corner_rect_open_surface(mesh, 
                                                                   vol_shape, 
                                                                   ref_depth=ref_depth, 
                                                                   order=order, 
                                                                   curvature_flow=curvature_flow, 
                                                                   delta_flow=delta_flow, 
                                                                   flow_iters=flow_iters)
    bnd_vertex = v[bnd].copy()
    pts1_ind, pts2_ind, pts3_ind, pts4_ind = corner_bnd_ind

    # # this needs proper sorting... 
    # print(pts1_ind, pts2_ind, pts3_ind, pts4_ind)
    
    # import pylab as plt 
    # plt.figure()
    # plt.plot([corner1yx[0], corner2yx[0], corner3yx[0], corner4yx[0]],
    #          [corner1yx[1], corner2yx[1], corner3yx[1], corner4yx[1]], 'r.-')
    # plt.show()
# =============================================================================
#    3) rectangular conformal map -> first maps to the disk  
# =============================================================================
    # stack the corners. 
    corner = np.hstack([pts1_ind, pts2_ind, pts3_ind, pts4_ind])

    if map2square == True:

        if square_method is not None:
            if square_method == 'Teichmuller':
                square, h_opt = rectangular_conformal_map(v, f, corner, map2square=map2square)
                # return square 
            else:
                # then the disk = the remesh. 
                disk = rectangular_conformal_map(v, f, corner, map2square=map2square)

                if square_method == 'squicircle':
                    print('squicircle')
                    square = _squicircle(disk)

                if square_method == 'elliptical':
                    print('elliptical')
                    square = _elliptical_nowell(disk)

                # then we try to optimize aspect ratio to get conformality. 
                def func(h):
                    return np.sum(np.abs(beltrami_coefficient(np.hstack([square[:,0][:,None],h*square[:,1][:,None]]), f, v))**2)

                if optimize:
                    h_opt = fminbound(func, 0, 5);
                    square = np.vstack([square[:,0], h_opt*square[:,1]]).T;
                # return square     
        else:
            disk = rectangular_conformal_map(v, f, corner, map2square=map2square)
            # print('direct return')
            square = disk.copy()
            # return square
    else:
        # print('Teichmuller')
        # then the disk = the remesh.
        # default to disk!.  
        disk = rectangular_conformal_map(v, f, corner, map2square=False)
        square = disk.copy()

    return square
    


"""
mesh quality metrics. 
"""
# not used. 
# def conformal_distortion_factor_trimesh(pts2D, pts3D, triangles, eps=1e-20):
#     """
#     """
#     # this metric is implemented from http://hhoppe.com/tmpm.pdf and which seems to be mainly used by all the graphical community.
#     # 1 = conformal, this is also just the stretch factor.
#     # measuring  the  quasi-conformal  error,  computed  as  the  area-weighted  average of  the  ratios  of  the  largest  to  smallest  singular  values  of the map’s Jacobian 
#     import igl 
#     import numpy as np 

#     tri2D = pts2D[triangles].copy() # N x 3 x 2
#     tri3D = pts3D[triangles].copy() # N x 3 x 3 

#     q1 = tri3D[:,0].copy()
#     q2 = tri3D[:,1].copy()
#     q3 = tri3D[:,2].copy()

#     # 2D coordinates.
#     s1 = tri2D[:,0,0].copy()
#     s2 = tri2D[:,1,0].copy()
#     s3 = tri2D[:,2,0].copy()

#     t1 = tri2D[:,0,1].copy()
#     t2 = tri2D[:,1,1].copy()
#     t3 = tri2D[:,2,1].copy()


#     # A = ((tri2D[:,1,0] - tri2D[:,0,0]) * (tri2D[:,2,1] - tri2D[:,0,1]) - (tri2D[:,2,0] - tri2D[:,0,0]) * (tri2D[:,1,1] - tri2D[:,0,1])) / 2. # area 2D triangles 
#     A = ((s2 - s1)*(t3-t1) - (s3 - s1)*(t2-t1)) / 2.
#     Ss = (q1*(t2-t3)[:,None] + q2*(t3-t1)[:,None] + q3*(t1-t2)[:,None]) / (2*A[:,None] + eps) # dS / ds
#     St = (q1*(s3-s2)[:,None] + q2*(s1-s3)[:,None] + q3*(s2-s1)[:,None]) / (2*A[:,None] + eps) # dS / dt

#     # get the largest and smaller single values of the Jacobian for each element... 
#     a = Ss.dot(Ss)
#     b = Ss.dot(St)
#     c = St.dot(St)

#     Gamma = np.sqrt((a+c + np.sqrt((a-c)**2 + 4*b**2))/2.)
#     gamma = np.sqrt((a+c - np.sqrt((a-c)**2 + 4*b**2))/2.)

#     stretch_ratios = Gamma/gamma 

#     area = igl.doublearea(pts3D,triangles) #total area. 
#     mean_stretch_ratio = np.nansum(area * stretch_ratios) / (np.nansum(area))

#     return mean_stretch_ratio, stretch_ratios


# # compute the 3D to 3D deformation analysis. 
# # def statistical_strain_rate_mesh(grid_squares_time, unwrap_params_3D):
# def statistical_strain_mesh3D(pts1, pts2, triangles):
    
#     # we do this in (x,y,z) coordinates. for polygonal mesh. 
#     # see, http://graner.net/francois/publis/graner_tools.pdf for an introduction of the mathematics. 
#     # both pts1 and pts2 are 3D ! 

#     import numpy as np 

#     # compute the differential change in links. 
#     triangles1 = pts1[triangles].copy() # N_tri x 3 x 3 
#     triangles2 = pts2[triangles].copy() 

#     # form the displacements -> i.e the edge vectors !. 
#     links_3D = triangles1 - triangles2 # displacements (x,y,z)


#     # build the covariance matrices. 
#     M_matrix_00 = np.mean( links_3D[...,0] ** 2, axis=1)
#     M_matrix_01 = np.mean( links_3D[...,0] * links_3D[...,1], axis=1)
#     M_matrix_02 = np.mean( links_3D[...,0] * links_3D[...,2], axis=1)
#     M_matrix_10 = np.mean( links_3D[...,1] * links_3D[...,0], axis=1)
#     M_matrix_11 = np.mean( links_3D[...,1] **2, axis=1)
#     M_matrix_12 = np.mean( links_3D[...,1] * links_3D[...,2], axis=1)
#     M_matrix_20 = np.mean( links_3D[...,2] * links_3D[...,0], axis=1)
#     M_matrix_21 = np.mean( links_3D[...,2] * links_3D[...,1], axis=1)
#     M_matrix_22 = np.mean( links_3D[...,2] **2, axis=-1)
    
#     # compute the inverse 3 x 3 matrix using fomulae.
#     M_matrix = np.array([[M_matrix_00, M_matrix_01, M_matrix_02], 
#                          [M_matrix_10, M_matrix_11, M_matrix_12], 
#                          [M_matrix_20, M_matrix_21, M_matrix_22]])
    
#     M_matrix = M_matrix.transpose(2,0,1) 

#     # from this we should be able to go ahead and extract the principal strains. 
    
#     return M_matrix



### need to check the following. 
# def mesh_strain_polygon(pts1, pts2, triangles):
#     r""" Compute the temporal polygonal mesh strain 3D deformation as described in reference [1]_
    
#     Parameters
#     ----------
#     pts1 : (n_time, n_vertices, 3) array
#         vertices of mesh 1 for all timepoints 
#     pts2 : (n_time, n_vertices, 3) array 
#         vertices of mesh 2 for all timepoints 
#     triangles : (n_time, n_faces, 3) array
#         triangulations of the mesh at all timepoints
        
#     Returns
#     -------
#     V : 
#         strain matrix     (symmetric component)
#     Omega : 
#         rotational strain matrix (antisymmetric component)
        
#     References
#     ----------
#     .. [1] Graner, François, et al. "Discrete rearranging disordered patterns, part I: Robust statistical tools in two or three dimensions." The European Physical Journal E 25.4 (2008): 349-369.
    
#     """
#     # we can do this in (x,y,z) coordinates.
#     # see, http://graner.net/francois/publis/graner_tools.pdf
#     import numpy as np 

#     # get the triangles and combine.  
#     triangles12 = np.array([pts1[triangles], 
#                             pts2[triangles]]) # combine to 2 x N_tri x 3 x 3
#     triangles12 = np.concatenate([triangles12, 
#                                   triangles12[:,:,0,:][:,:,None,:]], axis=2)

#     links_squares_time_3D = triangles12[:,:,1:] - triangles12[:,:,:-1] # compute the edge vectors. 
#     # links_squares_time_3D = unwrap_params_3D[grid_squares_time[:,:,1:,1].astype(np.int), 
#     #                                           grid_squares_time[:,:,1:,0].astype(np.int)] - unwrap_params_3D[grid_squares_time[:,:,:-1,1].astype(np.int), 
#     #                                           grid_squares_time[:,:,:-1,0].astype(np.int)]
    
#     # time differential. => here this is the evolution.   
#     d_links_squares_time_3D = links_squares_time_3D[1:] - links_squares_time_3D[:-1] # this is the stretch ...    
#     # links_squares_time_3D = links_squares_time_3D[1:] - links_squares_time_3D[:-1] # this is the stretch ...    

#     M_matrix_00 = np.mean( links_squares_time_3D[...,0] ** 2, axis=-1) # take the 2nd last to get the average of the polygons.
#     M_matrix_01 = np.mean( links_squares_time_3D[...,0] * links_squares_time_3D[...,1], axis=-1)
#     M_matrix_02 = np.mean( links_squares_time_3D[...,0] * links_squares_time_3D[...,2], axis=-1)
#     M_matrix_10 = np.mean( links_squares_time_3D[...,1] * links_squares_time_3D[...,0], axis=-1)
#     M_matrix_11 = np.mean( links_squares_time_3D[...,1] **2, axis=-1)
#     M_matrix_12 = np.mean( links_squares_time_3D[...,1] * links_squares_time_3D[...,2], axis=-1)
#     M_matrix_20 = np.mean( links_squares_time_3D[...,2] * links_squares_time_3D[...,0], axis=-1)
#     M_matrix_21 = np.mean( links_squares_time_3D[...,2] * links_squares_time_3D[...,1], axis=-1)
#     M_matrix_22 = np.mean( links_squares_time_3D[...,2] **2, axis=-1)
    
#     # compute the inverse 3 x 3 matrix using fomulae.
#     M_matrix = np.array([[M_matrix_00, M_matrix_01, M_matrix_02], 
#                          [M_matrix_10, M_matrix_11, M_matrix_12], 
#                          [M_matrix_20, M_matrix_21, M_matrix_22]])
    
#     M_matrix = M_matrix.transpose(2,3,0,1)
#     M_inv = np.linalg.pinv(M_matrix.reshape(-1,3,3)).reshape(M_matrix.shape)
    
#     # print(np.allclose(M_matrix[0,0], np.dot(M_matrix[0,0], np.dot(M_inv[0,0], M_matrix[0,0]))))
# # #    print(M_inv[0,0])
# # #    print(np.linalg.inv(M_matrix[0,0]))
# # #    print(np.dot(M_matrix[0,0], np.linalg.inv(M_matrix[0,0])))
    
#     C_matrix_00 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,0], axis=-1) # this one is inner product.... 
#     C_matrix_01 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,1], axis=-1)
#     C_matrix_02 = np.mean( links_squares_time_3D[:-1,...,0] * d_links_squares_time_3D[...,2], axis=-1)
#     C_matrix_10 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,0], axis=-1)
#     C_matrix_11 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,1], axis=-1)
#     C_matrix_12 = np.mean( links_squares_time_3D[:-1,...,1] * d_links_squares_time_3D[...,2], axis=-1)
#     C_matrix_20 = np.mean( links_squares_time_3D[:-1,...,2] * d_links_squares_time_3D[...,0], axis=-1)
#     C_matrix_21 = np.mean( links_squares_time_3D[:-1,...,2] * d_links_squares_time_3D[...,1], axis=-1)
#     C_matrix_22 = np.mean( links_squares_time_3D[:-1,...,2] * d_links_squares_time_3D[...,2], axis=-1)

#     C_matrix = np.array([[C_matrix_00, C_matrix_01, C_matrix_02], 
#                          [C_matrix_10, C_matrix_11, C_matrix_12], 
#                          [C_matrix_20, C_matrix_21, C_matrix_22]])
#     C_matrix = C_matrix.transpose(2,3,0,1)
#     C_matrix_T = C_matrix.transpose(0,1,3,2) # obtain the matrix transpose.
    
# #    V = 1./2 *( np.matmul(M_inv.reshape(-1,3,3), C_matrix.reshape()) + np.matmul(C_matrix_T, M_inv))
    
# #    MM = M_inv.reshape(-1,3,3)
# #    CC = C_matrix.reshape(-1,3,3)
# #    CC_T = C_matrix_T.reshape(-1,3,3)
#     V = 1./2 *( np.matmul(M_inv[:-1], C_matrix) + 
#                 np.matmul(C_matrix_T, M_inv[:-1]))

#     Omega = 1./2 * ( np.matmul(M_inv[:-1], C_matrix) - 
#                      np.matmul(C_matrix_T, M_inv[:-1]))
    
#     return V, Omega
#     # return M_matrix


def map_3D_to_2D_triangles(pts3D, triangles):
    r""" Isometric projection of 3D triangles to 2D coordinates This function implements the solution of [1]_. This function is similar to igl.project_isometrically_to_plane

    Given the vertices :math:`v_1, v_2, v_3` of a triangle in 3D, a 2D isometric projection can be constructed that preserves length and area with new vertex coordinate defined by 

    .. math::
        v^{2D}_1 &= (0, 0) \\
        v^{2D}_2 &= (|A|, 0) \\
        v^{2D}_3 &= (A.B/ |A|, |A \times B|/|A|)

    where :math:`A=v_2-v_1`, :math:`B=v_3-v_1`.

    Parameters
    ----------
    pts3D : (n_vertices,3) array
        the 3D vertices of the mesh 
    triangles : (n_faces,3) array
        the triangulation of pts3D given by vertex indices

    Returns
    -------
    pts_2D : (n_faces,2) array
        the vertices of the triangle in 2D 

    References
    ----------
    .. [1] https://stackoverflow.com/questions/8051220/flattening-a-3d-triangle
    .. [2] https://scicomp.stackexchange.com/questions/25327/finding-shape-functions-for-a-triangle-in-3d-coordinate-space
    
    """
    import numpy as np 

    pts_tri = pts3D[triangles].copy()

    A = pts_tri[:,1] - pts_tri[:,0]
    B = pts_tri[:,2] - pts_tri[:,0] 

    # set the first point to (0,0)
    pts_2D_0 = np.zeros((len(pts_tri), 2))  
    pts_2D_1 = np.hstack([np.linalg.norm(A, axis=-1)[:,None], np.zeros(len(pts_tri))[:,None]])
    pts_2D_2 = np.hstack([(np.sum(A*B, axis=-1))[:,None], (np.linalg.norm(np.cross(A,B), axis=-1))[:,None]]) / (np.linalg.norm(A, axis=-1))[:,None] 
    pts_2D = np.concatenate([pts_2D_0[:,None,:], 
                             pts_2D_1[:,None,:], 
                             pts_2D_2[:,None,:]], axis=1)

    return pts_2D



def quasi_conformal_error(pts1_3D, pts2_3D, triangles):
    r""" Computes the quasi-conformal error between two 3D triangle meshes as defined in [1]_ 
    
    Parameters
    ----------
    pts1_3D : (n_vertices,3) array
        vertices of mesh 1 
    pts2_3D : (n_vertices,3) array
        vertices of mesh 2 
    triangles : (n_faces,3) array
        the triangulation of pts1_3D and pts2_3D in terms of the vertex indices

    Returns 
    -------
    Jac_eigvals : (n_faces,)
        eigenvalues of the transformation matrix mapping the 2D isometric projections of pts1_3D to pts_2_3D
    stretch_factor : (n_faces,)
        the ratio of the square root of maximum singular value over square root of of the minimum singular value of the square form of the transformation matrix mapping the 2D isometric projections of pts1_3D to pts_2_3D
    mean_stretch_factor : scalar
        the area weighted mean stretch factor or quasi-conformal error  
    (areas3D, areas3D_2) : ((n_faces,), (n_faces,)) list of arrays 
        the triangle areas of the first and second mesh respectively

    References
    ----------
    .. [1] Hormann, K. & Greiner, G. MIPS: An efficient global parametrization method. (Erlangen-Nuernberg Univ (Germany) Computer Graphics Group, 2000)

    """
    """
    maps R^3 to R^2 triangle. Describes the transformation by linear means -> equivalent to the jacobian matrix.
        # we also take the opportunity to compute the area.          
    """
    import igl 
    import numpy as np 

    # map the 3D triangle to 2D triangle coordinates. 
    pts1_2D = map_3D_to_2D_triangles(pts1_3D, triangles)
    pts2_2D = map_3D_to_2D_triangles(pts2_3D, triangles)

    # convert 2D coordinates to homogeneous coordinates in order to solve uniquely.
    # having converted we can now get the jacobian by specifying Y = AX (homogeneous coordinates.)
    pts1_2D_hom = np.concatenate([pts1_2D, 
                                  np.ones((pts1_2D.shape[0], pts1_2D.shape[1]))[...,None]], axis=-1)
    pts2_2D_hom = np.concatenate([pts2_2D, 
                                  np.ones((pts2_2D.shape[0], pts2_2D.shape[1]))[...,None]], axis=-1)
    pts1_2D_hom = pts1_2D_hom.transpose(0,2,1).copy()
    pts2_2D_hom = pts2_2D_hom.transpose(0,2,1).copy()
    
    # solve exactly. 
    try:
        Tmatrix = np.matmul(pts2_2D_hom, np.linalg.inv(pts1_2D_hom)) # this really is just solving registration problem ? 
    except:
        Tmatrix = np.matmul(pts2_2D_hom, np.linalg.inv(pts1_2D_hom + np.finfo(float).eps)) # if fails we need a small eps to stabilise. 
    JacMatrix = Tmatrix[:,:2,:2].copy() # take the non jacobian component? 
    

    # The error is given as the ratio of the largest to smallest eigenvalue. -> since SVD, hence the singular values are squared. 
    u, s, v = np.linalg.svd(JacMatrix)
    
    # see http://graphics.stanford.edu/courses/cs468-10-fall/LectureSlides/13_Parameterization2.pdf to better understand. 
    Jac_eigvals = s.copy()

    # stretch_factor is gotten from eigenvalues of J^T J
    JacMatrix2 = np.matmul(JacMatrix.transpose(0,2,1), JacMatrix)
    stretch_eigenvalues, stretch_eigenvectors = np.linalg.eigh(JacMatrix2) # this should be square root.! 
    # stretch_eigenvalues = np.sqrt(stretch_eigenvalues) # we should do this !!!! since we squared the matrix form!. 

    # stretch_factor = np.sqrt(np.max(np.abs(s), axis=1) / np.min(np.abs(s), axis=1)) # since this was SVD decomposition. 
    stretch_factor = np.sqrt(np.max(np.abs(stretch_eigenvalues), axis=1) / np.min(np.abs(stretch_eigenvalues), axis=1)) # since this was SVD decomposition. 

    areas3D = igl.doublearea(pts1_3D, triangles) / 2.
    mean_stretch_factor = np.nansum(areas3D*stretch_factor / (float(np.sum(areas3D)))) # area weighted average.

    # we also compute the final areas to derive an area change factor. 
    areas3D_2 = igl.doublearea(pts2_3D, triangles) / 2. 

    return Jac_eigvals, stretch_factor, mean_stretch_factor, (areas3D, areas3D_2)


def MIPS_cost(pts1_3D, pts2_3D, triangles, area_mips_theta=1, norm_pts=True):
    r""" Compute the Most isometric parametrization (MIPs) and the Area-preserving MIPs cost defined in [1]_ and [2]_ respectively

    Parameters
    ----------
    pts1_3D : (n_vertices,3) array 
        vertices of mesh 1 
    pts2_3D : (n_vertices,3) array 
        vertices of mesh 2
    triangles : (n_faces,3) array
        the triangulation of pts1_3D and pts2_3D in terms of the vertex indices
    area_mips_theta : scalar
        the exponent of the area-preserving MIPs cost in [2]_. If area_mips_theta=1, the area-preserving MIPs measures the area uniformity of stretch distortion of the surface
    norm_pts : True
        normalize vertex points by the respective surface areas of the mesh before computing the cost

    Returns
    -------
    (MIPS, area_MIPS, MIPS_plus) : ((n_faces,), (n_faces,), (n_faces,)) list of array
        the MIPs, area preserving MIPs and the direct sum of stretch + area distortion  
    (mean_MIPS, mean_area_MIPS, mean_MIPS_plus) : (3,) tuple
        tuple of the mean MIPs, area preserving MIPs and the sum of stretch + area distortion 
    (sigma1,sigma2) : ((n_faces,), (n_faces,)) tuple 
        the square root of the maximum singular and square root of the minimum singular value of the square form of the transformation matrix mapping the 2D isometric projections of pts1_3D to pts_2_3D
    (stretch_eigenvalues, stretch_eigenvectors) : ((n_faces,), (n_faces,)) tuple 
        the singular value eigenvalue matrix and corresponding eigenvector matrix of the square form of the transformation matrix mapping the 2D isometric projections of pts1_3D to pts_2_3D

    References
    ----------
    .. [1] Hormann, K. & Greiner, G. MIPS: An efficient global parametrization method. (Erlangen-Nuernberg Univ (Germany) Computer Graphics Group, 2000)
    .. [2] Degener, P., Meseth, J. & Klein, R. An Adaptable Surface Parameterization Method. IMR 3, 201-213 (2003).

    """
    """
    maps R^3 to R^2 triangle. Describes the transformation by linear means -> equivalent to the jacobian matrix.
        # we also take the opportunity to compute the area.          
    """
    """
    use the Most isometric parametrization cost. 
    https://arxiv.org/pdf/1810.09031.pdf
    """
    import igl 
    import numpy as np 
    
    if norm_pts:
        pts1_3D_ = pts1_3D / np.nansum((igl.doublearea(pts1_3D, triangles) / 2.))
        pts2_3D_ = pts2_3D / np.nansum(igl.doublearea(pts2_3D, triangles) / 2.)
        pts1_2D = map_3D_to_2D_triangles(pts1_3D_, triangles)
        pts2_2D = map_3D_to_2D_triangles(pts2_3D_, triangles)
    else:
        # map the 3D triangle to 2D triangle coordinates. 
        pts1_2D = map_3D_to_2D_triangles(pts1_3D, triangles)
        pts2_2D = map_3D_to_2D_triangles(pts2_3D, triangles)

    # convert 2D coordinates to homogeneous coordinates in order to solve uniquely.
    # having converted we can now get the jacobian by specifying Y = AX (homogeneous coordinates.)
    pts1_2D_hom = np.concatenate([pts1_2D, 
                                  np.ones((pts1_2D.shape[0], pts1_2D.shape[1]))[...,None]], axis=-1)
    pts2_2D_hom = np.concatenate([pts2_2D, 
                                  np.ones((pts2_2D.shape[0], pts2_2D.shape[1]))[...,None]], axis=-1)
    pts1_2D_hom = pts1_2D_hom.transpose(0,2,1).copy()
    pts2_2D_hom = pts2_2D_hom.transpose(0,2,1).copy()
    
    # solve exactly. 
    try:
        Tmatrix = np.matmul(pts2_2D_hom, np.linalg.inv(pts1_2D_hom)) # this really is just solving registration problem ? 
    except:
        Tmatrix = np.matmul(pts2_2D_hom, np.linalg.inv(pts1_2D_hom + np.finfo(float).eps)) # if fails we need a small eps to stabilise. 
    # Tmatrix = np.matmul(pts2_2D_hom, np.linalg.inv(pts1_2D_hom)) # this really is just solving registration problem ? 
    JacMatrix = Tmatrix[:,:2,:2].copy() # take the non jacobian component? 
    

    # The error is given as the ratio of the largest to smallest eigenvalue. -> since SVD, hence the singular values are squared. 
    u, s, v = np.linalg.svd(JacMatrix)
    
    # see http://graphics.stanford.edu/courses/cs468-10-fall/LectureSlides/13_Parameterization2.pdf to better understand. 
    # Jac_eigvals = s.copy()

    # stretch_factor is gotten from eigenvalues of J^T J
    JacMatrix2 = np.matmul(JacMatrix.transpose(0,2,1), JacMatrix)
    stretch_eigenvalues, stretch_eigenvectors = np.linalg.eigh(JacMatrix2)
    
    sigma1 = np.sqrt(np.max(stretch_eigenvalues, axis=1)) # this is the ones. 
    sigma2 = np.sqrt(np.min(stretch_eigenvalues, axis=1))
    # sigma1 = np.max(np.abs(s), axis=1)
    # sigma2 = np.min(np.abs(s), axis=1)
    
    MIPS = sigma1/sigma2 + sigma2/sigma1
    area_MIPS = (sigma1/sigma2 + sigma2/sigma1)*(sigma1*sigma2+1./(sigma1*sigma2))**area_mips_theta
    MIPS_plus = sigma1/sigma2 + sigma1*sigma2
    
    areas3D = igl.doublearea(pts1_3D, triangles) / 2.
    mean_MIPS = np.nansum(areas3D*MIPS / (float(np.sum(areas3D)))) # area weighted average.
    mean_area_MIPS = np.nansum(areas3D*area_MIPS / (float(np.sum(areas3D)))) 
    mean_MIPS_plus = np.nansum(areas3D*MIPS_plus / (float(np.sum(areas3D)))) 
    
    return (MIPS, area_MIPS, MIPS_plus), (mean_MIPS, mean_area_MIPS, mean_MIPS_plus), (sigma1,sigma2), (stretch_eigenvalues, stretch_eigenvectors)
    
    
    

# def MIPS_cost(pts1_3D, pts2_3D, triangles):
    
#     """
#     use the Most isometric parametrization cost. 
#     https://arxiv.org/pdf/1810.09031.pdf
#     """
#     import igl 
#     import numpy as np 

#     # normalize the pts 
#     pts1_3D_ =  pts1_3D / np.sqrt( np.sum(igl.doublearea(pts1_3D, triangles)/2.) )
#     pts2_3D_ =  pts2_3D / np.sqrt( np.sum(igl.doublearea(pts2_3D, triangles)/2.) )
#     # pts1_3D_ = pts1_3D.copy()
#     # pts2_3D_ = pts2_3D.copy()

#     # map the 3D triangle to 2D triangle coordinates. 
#     [U1,UF1,I1] = igl.project_isometrically_to_plane(pts1_3D_, triangles) # V to U.    
#     [U2,UF2,I2] = igl.project_isometrically_to_plane(pts2_3D_, triangles)

#     pts1_2D = U1[UF1]
#     pts2_2D = U2[UF2]

#     # convert 2D coordinates to homogeneous coordinates in order to solve uniquely.
#     # having converted we can now get the jacobian by specifying Y = AX (homogeneous coordinates.)
#     pts1_2D_hom = np.concatenate([pts1_2D, 
#                                   np.ones((pts1_2D.shape[0], pts1_2D.shape[1]))[...,None]], axis=-1)
#     pts2_2D_hom = np.concatenate([pts2_2D, 
#                                   np.ones((pts2_2D.shape[0], pts2_2D.shape[1]))[...,None]], axis=-1)
#     pts1_2D_hom = pts1_2D_hom.transpose(0,2,1).copy()
#     pts2_2D_hom = pts2_2D_hom.transpose(0,2,1).copy()
    
#     # solve exactly. 
#     Tmatrix = np.matmul(pts2_2D_hom, np.linalg.inv(pts1_2D_hom)) # this really is just solving registration problem ? 
#     JacMatrix = Tmatrix[:,:2,:2].copy() # take the non jacobian component? 
    
#     # The error is given as the ratio of the largest to smallest eigenvalue. -> since SVD, hence the singular values are squared. 
#     u, s, v = np.linalg.svd(JacMatrix)

#     Jac_eigvals = s.copy()

#     # stretch_factor is gotten from eigenvalues of J^T J
#     JacMatrix2 = np.matmul(JacMatrix.transpose(0,2,1), JacMatrix)
#     stretch_eigenvalues, stretch_eigenvectors = np.linalg.eigh(JacMatrix2)

#     print(stretch_eigenvalues.shape)
#     # sigma1 = np.sqrt(np.max(stretch_eigenvalues, axis=1))
#     # sigma2 = np.sqrt(np.min(stretch_eigenvalues, axis=1))
#     sigma1 = np.max(np.abs(s), axis=1)
#     sigma2 = np.min(np.abs(s), axis=1)

#     # stretch_factor = np.sqrt(np.max(np.abs(s), axis=1) / np.min(np.abs(s), axis=1)) # since this was SVD decomposition. 
#     stretch_factor = np.sqrt(np.max(np.abs(stretch_eigenvalues), axis=1) / np.min(np.abs(stretch_eigenvalues), axis=1)) # since this was SVD decomposition. 

#     return sigma1/sigma2 + sigma2/sigma1, stretch_factor # this is the MIPS cost... as a balanced cost. 
#     # return sigma1/sigma2 + sigma2*sigma1, stretch_factor
#     # return sigma1*sigma2 + 1./(sigma2*sigma1), stretch_factor


# #### functions for mesh-based morphological operations.
def remove_small_mesh_components_binary(v,f,labels, vertex_labels_bool=True, physical_size=True, minsize=100): # assume by default vertex labels.
    r""" Remove small connected components of a binary labelled mesh. Connected components is run and regions with number of vertices/faces or covering an area less than the specified threshold is removed by returning a new binary label array where they have been set to 0 
    
    Parameters
    ----------
    v : (n_vertices,) array
        vertices of the triangle mesh 
    f : (n_faces,) array
        faces of the triangle mesh 
    labels : (n_vertices,) or (n_faces,) array
        the binary labels either specified for the vertex or face. Which is which is set by the parameter ``vertex_labels_bool``. 
    vertex_labels_bool : bool
        if True, process the ``labels`` as associated with vertices or if False, process the ``labels`` as associated with faces. 
    physical_size : bool
        if True, interpret ``minsize`` as the minimum surface area of each connected component. If False, interpret ``minsize`` as the minimum number of vertex/face elements within the connected component
    minsize : scalar 
        if physical_size=True, the minimum surface area of a connected component or if physical_size=False, the minimum number of vertex/face elements within the connected component
    
    Returns
    -------
    labels_clean : (n_vertices,) or (n_faces,) array
        The updated vertex (if vertex_labels_bool=True) or face (if vertex_labels_bool=False) binary label array

    """
    import numpy as np 
    import scipy.stats as spstats
    import trimesh
    import igl

    if vertex_labels_bool:
        face_labels = spstats.mode(labels[f], axis=1)[0]
        face_labels = np.squeeze(face_labels)
    else:
        face_labels = labels.copy()

    mesh_cc_label = connected_components_mesh(trimesh.Trimesh(vertices=v, 
                                                             faces=f[face_labels>0], 
                                                             process=False, 
                                                             validate=False),
                                                             original_face_indices=np.arange(len(f))[face_labels>0])
    if physical_size:
        mesh_cc_area = [np.nansum(igl.doublearea(v, f[cc])/2.) for cc in mesh_cc_label] # we need to double check this 
        mesh_cc_label = [mesh_cc_label[cc] for cc in np.arange(len(mesh_cc_area)) if mesh_cc_area[cc] > minsize]
    else:
        mesh_cc_label = [mesh_cc_label[cc] for cc in np.arange(len(mesh_cc_label)) if len(mesh_cc_label[cc]) > minsize] # just by number of faces.!
    
    # rebuild the output vertex/face labels
    labels_clean = np.zeros(labels.shape, dtype=np.int32)
    if vertex_labels_bool:
        for cc in mesh_cc_label:
            faces_cc = f[cc] 
            unique_verts_cc = np.unique(faces_cc.ravel())
            labels_clean[unique_verts_cc] = 1
    else:
        for cc in mesh_cc_label:
            labels_clean[cc] = 1

    return labels_clean


def remove_small_mesh_components_labels(v,f,labels, 
                                        bg_label=0, 
                                        vertex_labels_bool=True, 
                                        physical_size=True, 
                                        minsize=100, 
                                        keep_largest_only=True): # assume by default vertex labels.
    r""" Remove small connected components of a multi-labelled integer mesh. Connected components is run on each labelled region and disconnected regions with number of vertices/faces or covering an area less than the specified threshold is removed by returning a new multi label array where they have been set to the specified background label 
    
    Parameters
    ----------
    v : (n_vertices,) array
        vertices of the triangle mesh 
    f : (n_faces,) array
        faces of the triangle mesh 
    labels : (n_vertices,) or (n_faces,) array
        the integer labels specified for the vertex or face as determined by the boolean parameter ``vertex_labels_bool``.
    bg_label : int 
        the integer label of background regions 
    vertex_labels_bool : bool
        if True, process the ``labels`` as associated with vertices or if False, process the ``labels`` as associated with faces. 
    physical_size : bool
        if True, interpret ``minsize`` as the minimum surface area of each connected component. If False, interpret ``minsize`` as the minimum number of vertex/face elements within the connected component
    minsize : scalar 
        if physical_size=True, the minimum surface area of a connected component or if physical_size=False, the minimum number of vertex/face elements within the connected component
    keep_largest_only : bool 
        if True, keep only the largest connected region per label of size > minsize. if False, all connected regions per label of size > minsize is kept    

    Returns
    -------
    labels_clean : (n_vertices,) or (n_faces,) array
        The updated vertex (if vertex_labels_bool=True) or face (if vertex_labels_bool=False) multi label array

    """
    import numpy as np 
    import scipy.stats as spstats
    import trimesh
    import igl

    if vertex_labels_bool:
        face_labels = spstats.mode(labels[f], axis=1)[0]
        face_labels = np.squeeze(face_labels)
    else:
        face_labels = labels.copy()

    # create a vector to stick output to. 
    labels_clean = (np.ones(face_labels.shape, dtype=np.int32) * bg_label).astype(np.int32)

    uniq_labels = np.setdiff1d(np.unique(labels), bg_label)
    for lab in uniq_labels:
        # return a binary selector to place the labels. 
        mesh_cc_label = connected_components_mesh(trimesh.Trimesh(vertices=v, 
                                                                 faces=f[face_labels==lab], 
                                                                 process=False, 
                                                                 validate=False),
                                                                 original_face_indices=np.arange(len(f))[face_labels==lab])
        if physical_size:
            mesh_cc_area = [np.nansum(igl.doublearea(v, f[cc])/2.) for cc in mesh_cc_label] # we need to double check this 
            mesh_cc_label = [mesh_cc_label[cc] for cc in np.arange(len(mesh_cc_area)) if mesh_cc_area[cc] > minsize]
            mesh_cc_area = [mesh_cc_area[cc] for cc in np.arange(len(mesh_cc_area)) if mesh_cc_area[cc] > minsize] # also update this. 
        else:
            mesh_cc_label = [mesh_cc_label[cc] for cc in np.arange(len(mesh_cc_label)) if len(mesh_cc_label[cc]) > minsize] # just by number of faces.!
            mesh_cc_area = [len(cc) for cc in mesh_cc_area]
        if keep_largest_only:
            if len(mesh_cc_label) > 0:
                # print('Region label ', lab, len(mesh_cc_label), len(mesh_cc_area))
                labels_clean[mesh_cc_label[np.argmax(mesh_cc_area)]] = lab
        else:
            for cc in mesh_cc_label: # set all of them!. 
                labels_clean[cc] = lab 

    if vertex_labels_bool:
        # case to vertex labels 
        mesh = trimesh.Trimesh(vertices=v,
                            faces=f,
                            process=False,
                            validate=False)
        vertex_triangle_adj = trimesh.geometry.vertex_face_indices(len(v), 
                                                                   f,
                                                                   mesh.faces_sparse)

        vertex_triangle_labels = labels_clean[vertex_triangle_adj].astype(np.float16) # cast to a float
        vertex_triangle_labels[vertex_triangle_adj==-1] = np.nan # cast to nan. 
        vertex_triangle_labels = np.squeeze(spstats.mode(vertex_triangle_labels, axis=1, nan_policy='omit')[0].astype(np.int32)) # recast to int.
        labels_clean = vertex_triangle_labels.copy()
    
    return labels_clean


def remove_small_mesh_label_holes_binary(v, f, labels, vertex_labels_bool=True, physical_size=True, minsize=100): 
    r""" Remove small binary holes i.e. small islands of zeros within a region of 1s in a binary-labelled mesh. Connected components is run on 0's and regions with number of vertices/faces or covering an area less than the specified threshold is removed by returning a new binary label array where they have been set to 1 
    
    Parameters
    ----------
    v : (n_vertices,) array
        vertices of the triangle mesh 
    f : (n_faces,) array
        faces of the triangle mesh 
    labels : (n_vertices,) or (n_faces,) array
        the binary labels specified for the vertex or face as determined by the boolean parameter ``vertex_labels_bool``.
    vertex_labels_bool : bool
        if True, process the ``labels`` as associated with vertices or if False, process the ``labels`` as associated with faces. 
    physical_size : bool
        if True, interpret ``minsize`` as the minimum surface area of each connected component. If False, interpret ``minsize`` as the minimum number of vertex/face elements within the connected component
    minsize : scalar 
        if physical_size=True, the minimum surface area of a connected component or if physical_size=False, the minimum number of vertex/face elements within the connected component
    
    Returns
    -------
    labels_clean : (n_vertices,) or (n_faces,) array
        The updated vertex (if vertex_labels_bool=True) or face (if vertex_labels_bool=False) binary array

    """
    import numpy as np 
    import scipy.stats as spstats
    import trimesh
    import igl

    if vertex_labels_bool:
        face_labels = spstats.mode(labels[f], axis=1)[0]
        face_labels = np.squeeze(face_labels)
    else:
        face_labels = labels.copy()

    mesh_cc_label = connected_components_mesh(trimesh.Trimesh(vertices=v, 
                                                             faces=f[face_labels==0], # note the inverse!
                                                             process=False, 
                                                             validate=False),
                                                             original_face_indices=np.arange(len(f))[face_labels==0])
    if physical_size:
        mesh_cc_area = [np.nansum(igl.doublearea(v, f[cc])) for cc in mesh_cc_label]
        mesh_cc_label = [mesh_cc_label[cc] for cc in np.arange(len(mesh_cc_area)) if mesh_cc_area[cc] < minsize and mesh_cc_area[cc]>0]
    else:
        mesh_cc_label = [mesh_cc_label[cc] for cc in np.arange(len(mesh_cc_label)) if len(mesh_cc_label[cc]) < minsize and len(mesh_cc_label[cc])>1] # just by number of faces.!
    
    # rebuild the output vertex/face labels
    labels_clean = labels.copy()
    labels_clean = labels_clean.astype(np.int32) #copy the previous labels. 
    if vertex_labels_bool:
        for cc in mesh_cc_label:
            faces_cc = f[cc] 
            unique_verts_cc = np.unique(faces_cc.ravel())
            labels_clean[unique_verts_cc] = 1 # flip 0 -> 1 
    else:
        for cc in mesh_cc_label:
            labels_clean[cc] = 1

    return labels_clean


#### functions for label spreading
def labelspreading_mesh_binary(v,f,y,W=None, niters=10, return_proba=True, thresh=1):
    r""" Applies Laplacian 'local weighted' smoothing with a default or specified affinity matrix to diffuse vertex-based binary labels on a 3D triangular mesh  

    Parameters
    ----------
    v : (n_vertices,) array
        vertices of the triangle mesh 
    f : (n_faces,) array
        faces of the triangle mesh 
    y : (n_vertices,) array
        the initial binary labels to diffuse
    W : (n_vertex, n_vertex) sparse array
        if specified, the Laplacian-like affinity matrix used to diffuse binary labels. Defaults to the cotan Laplacian matrix 
    niters : int
        the number of iterations     
    return_proba : (n_vertices,) 
        if True, return the diffused probability matrix
    thresh : 0-1 scalar
        if less than 1, the probability matrix per iteration is binarised by thresholding > thresh. This allows faster diffusion of positive labels, equivalent of an inflation factor

    Returns
    -------
    labels_clean : (n_vertices,) array
        The updated vertex binary label array 

    """
    import scipy.sparse as spsparse
    import igl
    import numpy as np 

    if W is None:
        W_ = igl.cotmatrix(v, f)
    else:
        W_ = W.copy()

    # normalize. 
    # DD = 1./W_.sum(axis=-1)
    sumW_ = np.array(np.absolute(W_).sum(axis=1))
    sumW_ = np.squeeze(sumW_)
    DD = np.zeros(len(sumW_))
    DD[sumW_>0] = 1./np.sqrt(sumW_[sumW_>0])
    DD = np.nan_to_num(DD) # avoid infs.
    DD = spsparse.spdiags(np.squeeze(DD), [0], DD.shape[0], DD.shape[0])
    W_ = DD.dot(W_) # this is perfect normalization. 

    init_matrix = y.copy()

    for ii in np.arange(niters):
        init_matrix = W_.dot(init_matrix)

        if thresh<1:
            prob_matrix = init_matrix.copy()
            init_matrix = init_matrix > thresh
        else:
            prob_matrix = init_matrix.copy()

    if return_proba: 
        return init_matrix, prob_matrix
    else:
        return init_matrix


def labelspreading_mesh(v,f, x, y, W=None, niters=10, alpha_prop=.1, return_proba=False, renorm=False, convergence_iter=5):
    r""" Applies Label propagation of [1]_ with a default or specified affinity matrix, W to competitively diffuse vertex-based multi-labels on a 3D triangular mesh. Background labels are assumed to be 0  

    Parameters
    ----------
    v : (n_vertices,) array
        vertices of the triangle mesh 
    f : (n_faces,) array
        faces of the triangle mesh
    x : (N,) array
        the vertex indices that have been assigned integer labels > 0 
    y : (N,) array
        the matching assumed sequential integer labels from 1 to n_labels of the specified vertex indices in ``x``
    W : (n_vertex, n_vertex) sparse array
        if specified, the Laplacian-like affinity matrix used to diffuse binary labels. Defaults to the cotan Laplacian matrix 
    niters : int
        the number of iterations     
    alpha_prop : 0-1 scalar
        clamping factor. A value in (0, 1) that specifies the relative amount that a vertex should adopt the information from its neighbors as opposed to its initial label. alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information.
    return_proba : (n_vertices,) 
        if True, return the diffused probability matrix
    renorm : bool
        if True, at each iteration assign each vertex to the most probable label with probability = 1. 
    convergence_iter : int
        the number of iterations for which the diffused labels do not change for. After this number of iterations the function will early stop before ``n_iters``, otherwise the propagation occurs for at least ``n_iters``. 

    Returns
    -------
    z_label : (n_vertices,) array
        The updated vertex multi label array 
    z : (n_vertices, n_labels+1)
        The probabilistic vertex multi label assignment where rowsums = 1 

    References
    ----------
    .. [1] Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston, Bernhard Schoelkopf. Learning with local and global consistency (2004)

    """
    # we do it on vertex. 
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse

    n_samples = len(v)
    n_classes = int(np.nanmax(y)+1)

    init_matrix = np.zeros((n_samples,n_classes)) # one hot encode. 
    # print(init_matrix.shape)

    # this is the prior - labelled. 
    base_matrix = np.zeros((n_samples, n_classes)); 
    base_matrix[x.astype(np.int), y.astype(np.int)] = 1; 
    base_matrix = (1.-alpha_prop)*base_matrix # this is the moving average.     
    
    if W is None:
        W_ = igl.cotmatrix(v, f)
    else:
        W_ = W.copy()

    sumW_ = np.array(np.absolute(W_).sum(axis=1))
    sumW_ = np.squeeze(sumW_)
    D = np.zeros(len(sumW_))
    D[sumW_>0] = 1./np.sqrt(sumW_[sumW_>0])
    D = np.nan_to_num(D) # avoid infs.
    D = spsparse.spdiags(np.squeeze(D), [0], n_samples, n_samples)
    W_ = D.dot(W_.dot(D)) # apply the normalization!. 
    # print(W_[0].data)
    # print(W_[0].data)
    W_ = alpha_prop * W_
    # init_matrix = base_matrix.copy()

    convergence_count = 0
    n_comps = np.sum(base_matrix)
    # propagate this version is better diffusion with laplacian matrix. 
    for iter_ii in np.arange(niters): # no convergence... 
        # base_matrix should act as a clamp!. why is this getting smaller and smaller?  
        init_matrix = W_.dot(init_matrix) + base_matrix # this is just moving average # why is this not changing? # we should renormalize... # this is weird. 
        # init_matrix = spsparse.linalg.spsolve(spsparse.identity(base_matrix.shape[0])-W_, init_matrix)
        # init_matrix = init_matrix/init_matrix.max() # renormalize. 
        # init_matrix = (init_matrix-init_matrix.min())/ (init_matrix.max()-init_matrix.min())

        # convert to proba
        z = np.nansum(init_matrix, axis=1)
        z[z==0] += 1 # Avoid division by 0
        z = ((init_matrix.T)/z).T
        z_label = np.argmax(z, axis=1)
        
        n_comps2 = np.sum(z_label)
        # print(n_comps2)
        if np.abs(n_comps2 - n_comps) == 0:
            convergence_count+=1
            if convergence_count == convergence_iter:
                break
        else:
            n_comps = n_comps2

        if renorm:
            init_matrix = z.copy() # need to renormalize
            # # print(init_matrix.min(), init_matrix.max())
            init_matrix = np.zeros(init_matrix.shape)
            init_matrix[np.arange(len(init_matrix)), z_label] = 1
        else:
            init_matrix = z.copy()

    if return_proba:
        return z_label, z
    else:
        return z_label # this is the new. 


# assumes vertex labels
def labelspreading_fill_mesh(v,f, vertex_labels, niters=10, alpha_prop=0., minsize=0, bg_label=0):
    r""" Applies Constrained Label propagation of [1]_ with the uniform Laplacian matrix to diffuse vertex-based multi-labels on a 3D triangular mesh to infill small non-labelled background areas within the boundary of individual labelled regions

    Parameters
    ----------
    v : (n_vertices,) array
        vertices of the triangle mesh 
    f : (n_faces,) array
        faces of the triangle mesh
    vertex_labels : (n_vertices,) array
        the integer vertex-based multi labels  
    niters : int
        the number of iterations of infilling     
    alpha_prop : 0-1 scalar
        clamping factor. A value in (0, 1) that specifies the relative amount that a vertex should adopt the information from its neighbors as opposed to its initial label. alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information.
    minsize : int
        the minimum size of a labelled region to infill. Small labelled regions are not infilled as they themselves are assumed to be unstable
    bg_label : int
        the integer label denoting the background regions  

    Returns
    -------
    vertex_labels_final : (n_vertices,) array
        The updated vertex multi label array 

    References
    ----------
    .. [1] Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston, Bernhard Schoelkopf. Learning with local and global consistency (2004)
    
    """
    """restricts diffusion to the external boundary by disconnecting the boundary loop. from the rest of the mesh with optional designation of size.  
    """
    import igl 
    import scipy.stats as spstats 
    import numpy as np 
    import trimesh
    import scipy.sparse as spsparse

    unique_labels = np.setdiff1d(np.unique(vertex_labels), bg_label)

    # precompute the adjacency lists. 
    adj_mesh = igl.adjacency_list(f) # get the master list. 
    adj_matrix = igl.adjacency_matrix(f).tocsr()
    l = adj_matrix.shape[0]
    
    # transfer vertex to face labels. 
    face_labels = spstats.mode(vertex_labels[f], axis=1)[0] # this could be more efficient using one hot encoding. to do.  
    face_labels = np.squeeze(face_labels)
    
    # compute separately for each unique label
    vertex_labels_final = np.ones_like(vertex_labels) * bg_label # initialise all to background label!. 

    for unique_lab in unique_labels: 
        mesh_cc_label = connected_components_mesh(trimesh.Trimesh(vertices=v, 
                                                                  faces=f[face_labels==unique_lab], # connecting only the current lab
                                                                  process=False, validate=False),
                                                                original_face_indices=np.arange(len(face_labels))[face_labels==unique_lab])
        # only process a component of minsize
        mesh_cc_label = [cc for cc in mesh_cc_label if len(cc) >= minsize]
        adj_matrix_label = igl.adjacency_matrix(f).tocsr()
        
        # now we diffuse the labels inside here... ( we need to form the laplacian matrix. )
        for cc in mesh_cc_label[:]: 
            # get the unique verts
            unique_verts = np.unique(f[np.squeeze(cc)])
            boundary_verts = igl.boundary_loop(f[np.squeeze(cc)])
            
            #### for the unique_verts # maybe we can't diffuse in parallel? 
            for vv in boundary_verts: # iterate over the boundary only! and disconnect. 
                adj_v = adj_mesh[vv].copy()
                for adj_vv in adj_v:
                    if adj_vv not in unique_verts:
                        adj_matrix_label[vv,adj_vv] = 0. # set to 0 
                        adj_matrix_label[adj_vv,vv] = 0.

            adj_matrix_label.eliminate_zeros()

        # produce the propagation matrix. 
        Laplacian_adj_matrix = adj_matrix - spsparse.spdiags(np.squeeze(adj_matrix_label.sum(axis=1)), [0], l,l)
        Laplacian_adj_matrix = Laplacian_adj_matrix.tocsr()
            
        # print(Laplacian_adj_matrix[0].data)
        # # create the label_indices and labels of unique_lab 
        labels_ind = np.arange(len(vertex_labels))[vertex_labels==unique_lab]
        labels_labels = (np.ones(len(labels_ind)) * unique_lab).astype(np.int32)

        # diffusion has to occur now diffuse...
        prop_label = labelspreading_mesh(v,
                                         f, 
                                         x=labels_ind, 
                                         y=labels_labels, 
                                         W=Laplacian_adj_matrix, 
                                         niters=niters, 
                                          alpha_prop=alpha_prop, 
                                         return_proba=False)
        vertex_labels_final[prop_label==unique_lab] = unique_lab

    return vertex_labels_final



# implementing some measures of discrepancy between two meshes.
def chamfer_distance_point_cloud(pts1, pts2):
    r""" Compute the standard L2 chamfer distance (CD) between two points clouds. For each point in each cloud, CD finds the nearest point in the other point set, and finds the mean L2 distance.

    Given two point clouds, :math:`S_1, S_2`, the chamfer distance is defined as

    .. math::
        \text{CD}(S_1,S_2)=\frac{1}{|S_1|}\sum_{x\in S_1} {\min_{y\in S_2} ||x-y||_2} + \frac{1}{|S_2|}\sum_{x\in S_2} {\min_{y\in S_1} ||x-y||_2}

    Parameters
    ----------
    pts1 : (n_vertices_1,3) array
        the vertices of point cloud 1. The number of vertices can be different to that of ``pts2``
    pts2 : (n_vertices_2,3) array
        the vertices of point cloud 2. The number of vertices can be different to that of ``pts1``

    Returns
    -------
    chamfer_dist : scalar
        the chamfer distance between the two point clouds

    """
    import point_cloud_utils as pcu
    import numpy as np 

    pts1_ = np.array(pts1, order='C')
    pts2_ = np.array(pts2, order='C')
    chamfer_dist = pcu.chamfer_distance(pts1_.astype(np.float32), pts2_.astype(np.float32))
    
    return chamfer_dist


def hausdorff_distance_point_cloud(pts1, pts2, mode='two-sided', return_index=False):
    r""" Compute the Hausdorff distance (H) between two points clouds. The Hausdorff distance is the it is the greatest of all the distances from a point in one point cloud to the closest point in the other point cloud.
    
    The 'two-sided' Hausdorff distance takes the maximum of comparing the 1st point cloud to the 2nd point cloud and the 2nd point cloud to the 1st point cloud

    Parameters
    ----------
    pts1 : (n_vertices_1, 3) array
        the vertices of point cloud 1. The number of vertices can be different to that of ``pts2``
    pts2 : (n_vertices_2, 3) array
        the vertices of point cloud 2. The number of vertices can be different to that of ``pts1``
    mode : 'one-sided' or 'two-sided'
        compute either the one sided with the specified order of pts1 to pts2 or the 'two-sided' which compares both orders and returns the maximum 
    return_index : bool
        if True, return two additional optional outputs that specify the index of a point cloud and the index of its closest neighbor in the other point cloud

    Returns
    -------
    hausdorff_dist : scalar
        the chamfer distance between the two point clouds
    id_a : (N,) array
        the vertex id of the points in pts1 matched with maximum shortest distance to vertex ids ``id_b`` in pts2     
    id_b : (N,) 
        the vertex id of the points in pts2 matched with maximum shortest distance to vertex ids ``id_b`` in pts1     
    """
    import point_cloud_utils as pcu
    import numpy as np

    if mode == 'one-sided':
        # Compute one-sided squared Hausdorff distances
        if return_index:
            hausdorff_dist, id_a, id_b = pcu.one_sided_hausdorff_distance(np.array(pts1,order='C'), np.array(pts2, order='C'), return_index=True)
        else:
            hausdorff_dist = pcu.one_sided_hausdorff_distance(np.array(pts1,order='C'), np.array(pts2, order='C'))
        # hausdorff_b_to_a = pcu.one_sided_hausdorff_distance(b, a)
    if mode == 'two-sided':
        # Take a max of the one sided squared distances to get the two sided Hausdorff distance
        if return_index:
            hausdorff_dist, id_a, id_b = pcu.hausdorff_distance(np.array(pts1,order='C'), np.array(pts2, order='C'), return_index=True)
        else:
            hausdorff_dist = pcu.hausdorff_distance(np.array(pts1,order='C'), np.array(pts2, order='C'))

    if return_index:
        return hausdorff_dist, id_a, id_b 
    else:
        return hausdorff_dist


def wasserstein_distance_trimesh_trimesh(trimesh1,trimesh2,n_samples_1=1000,n_samples_2=1000):
    r""" Compute the Wasserstein distance between two triangle meshes using the Sinkhorn approximation and point cloud subsampling
    
    The triangle meshes are converted into a weighted point cloud or measure using the normalised triangle area

    Parameters
    ----------
    trimesh1 : trimesh.Trimesh
        a 3D triangle mesh
    trimesh2 : trimesh.Trimesh
        a 3D triangle mesh
    n_samples_1 : int
        the number of uniformly sampled random vertices from trimesh1
    n_samples_2 : int 
        the number of uniformly sampled random vertices from trimesh2

    Returns
    -------
    sinkhorn_dist : scalar
        the approximated wasserstein distance between the two meshes

    Notes
    -----
    https://github.com/fwilliams/point-cloud-utils for sinkhorn computation 

    """
    # use sampling to bring this down. random sampling. 
    # https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/plot_interpolation_3D.html#sphx-glr-auto-examples-optimal-transport-plot-interpolation-3d-py
    # we need to turn the mesh into a measure. i.e. load it with dirac atoms. 
    import igl 
    import numpy as np 
    import trimesh
    import point_cloud_utils as pcu
    
    area1 = igl.doublearea(trimesh1.vertices, trimesh1.faces) / 2.
    area1 = area1/np.nansum(area1)
    pts1 = igl.barycenter(trimesh1.vertices,trimesh1.faces)

    area2 = igl.doublearea(trimesh2.vertices, trimesh2.faces) / 2.
    area2 = area2/np.nansum(area2) # so it sums up to 1. 
    pts2 = igl.barycenter(trimesh2.vertices,trimesh2.faces)

    # subsample.
    if n_samples_1 is not None:
        select1 = np.random.choice(len(area1), n_samples_1)
        area1 = area1[select1].copy() # this is also the weights. 
        # area1 = area1 / float(np.nansum(area1)) # renormalize. to be a measure. 
        pts1 = pts1[select1].copy()

    if n_samples_2 is not None:
        select2 = np.random.choice(len(area2), n_samples_2)
        area2 = area2[select2].copy() # this is also the weights. 
        # area2 = area2 / float(np.nansum(area2)) # renormalize. to be a measure. 
        pts2 = pts2[select2].copy()

    M = pcu.pairwise_distances(pts1,pts2)
    P = pcu.sinkhorn(area1.astype(np.float32), 
                     area2.astype(np.float32), 
                     M.astype(np.float32), eps=1e-3, 
                     max_iters=500)

    # to get distance we compute the frobenius inner product <M, P>
    sinkhorn_dist = np.nansum(M*P)

    return sinkhorn_dist

def wasserstein_distance_trimesh_uv(trimesh1,uv2, eps=1e-12, pad=True, uv_to_trimesh=False, n_samples_1=1000, n_samples_2=1000):
    r""" Compute the Wasserstein distance between a triangle 3D mesh and a (u,v) image parameterized 3D mesh using the Sinkhorn approximation and point cloud subsampling
    
    The meshes are converted into a weighted point cloud or measure using the normalised areas

    Parameters
    ----------
    trimesh1 : trimesh.Trimesh
        a 3D triangle mesh
    uv2 : (U,V,3) array 
        a (u,v) image parameterized 3D surface
    eps : scalar
        a small constant for numerical stability 
    pad : bool
        if True, uses edge padding to compute the finite differences for evaluating the differential areas of ``uv2``
    uv_to_trimesh : bool
        if True, convert the (u,v) image parameterized 3D triangle mesh into a 3D triangle mesh before evaluating the difference. 
    n_samples_1 : int
        the number of uniformly sampled random vertices from trimesh1
    n_samples_2 : int 
        the number of uniformly sampled random vertices from trimesh2

    Returns
    -------
    sinkhorn_dist : scalar
        the approximated wasserstein distance between the two meshes

    Notes
    -----
    https://github.com/fwilliams/point-cloud-utils for sinkhorn computation 

    """
    # https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/plot_interpolation_3D.html#sphx-glr-auto-examples-optimal-transport-plot-interpolation-3d-py
    # we need to turn the mesh into a measure. i.e. load it with dirac atoms. 
    import igl 
    import numpy as np 
    import trimesh
    import point_cloud_utils as pcu

    if uv_to_trimesh:
        uv_vertex_indices_all, uv_triangles = get_uv_grid_tri_connectivity(np.zeros(uv2.shape[:2]))
        uv_pos_3D_v = uv2.reshape(-1,3)[uv_vertex_indices_all]
        trimesh2 = trimesh.Trimesh(vertices=uv_pos_3D_v, faces=uv_triangles, 
                                    process=False, validate=False)
        # reduces to the triangle case. 
        sinkhorn_dist = wasserstein_distance_trimesh_trimesh(trimesh1, trimesh2,
                                                            n_samples_1=n_samples_1, 
                                                            n_samples_2=n_samples_2)
    else:

        area1 = igl.doublearea(trimesh1.vertices, trimesh1.faces) / 2.
        area1 = area1/np.nansum(area1)
        pts1 = igl.barycenter(trimesh1.vertices,trimesh1.faces)

        dS_du, dS_dv = uzip.gradient_uv(uv2, eps=eps, pad=pad) # might be more accurate to convert ...
        # area of the original surface.
        area2 = np.linalg.norm(np.cross(dS_du, 
                                        dS_dv), axis=-1)# # use the cross product
        area2 = area2/np.nansum(area2)
        area2 = area2.ravel()
        pts2 = uv2.reshape(-1,uv2.shape[-1]).copy() # just make this a point cloud.

        # print(pts1.shape, area1.shape)
        # print(pts2.shape, area2.shape)
        # # subsample.
        if n_samples_1 is not None:
            select1 = np.random.choice(len(area1), n_samples_1)
            area1 = area1[select1].copy() # this is also the weights. 
            # area1 = area1 / float(np.nansum(area1)) # renormalize. to be a measure. 
            pts1 = pts1[select1].copy()

        if n_samples_2 is not None:
            select2 = np.random.choice(len(area2), n_samples_2)
            area2 = area2[select2].copy() # this is also the weights. 
            # area2 = area2 / float(np.nansum(area2)) # renormalize. to be a measure. 
            pts2 = pts2[select2].copy()

        M = pcu.pairwise_distances(pts1,pts2)
        P = pcu.sinkhorn(area1.astype(np.float32), 
                         area2.astype(np.float32), 
                         M.astype(np.float32), eps=1e-3)

        # to get distance we compute the frobenius inner product <M, P>
        sinkhorn_dist = np.nansum(M*P)
        # print(area1.sum(), area2.sum())

    return sinkhorn_dist


def sliced_wasserstein_distance_trimesh_trimesh(trimesh1,trimesh2,
                                                n_seeds=10,
                                                n_projections=50,
                                                p=1,
                                                mode='max',
                                                seed=None):

    r""" Compute the sliced Wasserstein distance approximation between two triangle meshes which gives a proxy of the Wasserstein distance using summed 1D random projections. 
    This method is advantageous in terms of speed and computational resources for very large meshes. 
    
    The triangle meshes are converted into a weighted point cloud or measure using the normalised triangle area

    Parameters
    ----------
    trimesh1 : trimesh.Trimesh
        a 3D triangle mesh
    trimesh2 : trimesh.Trimesh
        a 3D triangle mesh
    n_seeds : int
        the number of trials to average the distance over. Larger numbers give greater stability 
    n_projections : int 
        the number of 1D sliced random projections to sum over
    p : int
        the order of the Wasserstein distance. 1 is equivalent to the Earth Mover's distance
    mode : 'max' or any other string
        if mode='max' compute the maximum sliced wasserstein distance see ``ot.max_sliced_wasserstein_distance`` in the Python Optimal Transport library  
    seed : int 
        if specified, fix the random seed for reproducibility runs

    Returns
    -------
    sliced_W_dist : scalar
        the mean sliced wasserstein distance between the two meshes over `n_seeds` iterations

    Notes
    -----
    https://pythonot.github.io/ for sliced wassersten distance computation 

    """
    import igl 
    import numpy as np 
    import trimesh
    import ot
    
    area1 = igl.doublearea(trimesh1.vertices, trimesh1.faces) / 2.
    area1 = area1/np.nansum(area1)
    pts1 = igl.barycenter(trimesh1.vertices,trimesh1.faces)

    area2 = igl.doublearea(trimesh2.vertices, trimesh2.faces) / 2.
    area2 = area2/np.nansum(area2) # so it sums up to 1. 
    pts2 = igl.barycenter(trimesh2.vertices,trimesh2.faces)

    # subsample.
    sliced_W_dist_iters = []
    for iter_ii in np.arange(n_seeds):
        if mode=='max':
            sliced_W_dist = ot.max_sliced_wasserstein_distance(pts1, 
                                                           pts2, 
                                                           a=area1, 
                                                           b=area2, 
                                                           p=p,
                                                           n_projections=n_projections, 
                                                           seed=seed) # hm...
        else:
            sliced_W_dist = ot.sliced_wasserstein_distance(pts1, 
                                                           pts2, 
                                                           a=area1, 
                                                           b=area2, 
                                                           p=p,
                                                           n_projections=n_projections, 
                                                           seed=seed) # hm...
        sliced_W_dist_iters.append(sliced_W_dist)
    sliced_W_dist = np.nanmean(sliced_W_dist_iters) 

    return sliced_W_dist

def sliced_wasserstein_distance_trimesh_uv(trimesh1, 
                                            uv2, 
                                            eps=1e-12, 
                                            pad=True, 
                                            uv_to_trimesh=False,
                                            n_seeds=10,
                                            n_projections=50,
                                            p=1,
                                            mode='max',
                                            seed=None):
    r""" Compute the sliced Wasserstein distance approximation between a triangle mesh and a (u,v) image parameterized 3D mesh which gives a proxy of the Wasserstein distance using summed 1D random projections. 
    This method is advantageous in terms of speed and computational resources for very large meshes. 
    
    The triangle meshes are converted into a weighted point cloud or measure using the normalised triangle area

    Parameters
    ----------
    trimesh1 : trimesh.Trimesh
        a 3D triangle mesh
    uv2 : (U,V,3) array
        a (u,v) image parameterized 3D surface
    eps : scalar
        a small constant for numerical stability 
    pad : bool
        if True, uses edge padding to compute the finite differences for evaluating the differential areas of ``uv2``
    uv_to_trimesh : bool
        if True, convert the (u,v) image parameterized 3D triangle mesh into a 3D triangle mesh before evaluating the difference. 
    n_seeds : int
        the number of trials to average the distance over. Larger numbers give greater stability 
    n_projections : int 
        the number of 1D sliced random projections to sum over
    p : int
        the order of the Wasserstein distance. 1 is equivalent to the Earth Mover's distance
    mode : 'max' or any other string
        if mode='max' compute the maximum sliced wasserstein distance see ``ot.max_sliced_wasserstein_distance`` in the Python Optimal Transport library  
    seed : int 
        if specified, fix the random seed for reproducibility runs

    Returns
    -------
    sliced_W_dist : scalar
        the mean sliced wasserstein distance between the two meshes over `n_seeds` iterations

    Notes
    -----
    https://pythonot.github.io/ for sliced wassersten distance computation 

    """
    import igl 
    import numpy as np 
    import trimesh
    import ot # uses the Python POT library for optimal transport. 

    if uv_to_trimesh:
        uv_vertex_indices_all, uv_triangles = get_uv_grid_tri_connectivity(np.zeros(uv2.shape[:2]))
        uv_pos_3D_v = uv2.reshape(-1,3)[uv_vertex_indices_all]
        trimesh2 = trimesh.Trimesh(vertices=uv_pos_3D_v, faces=uv_triangles, 
                                    process=False, validate=False)

        sliced_W_dist = sliced_wasserstein_distance_trimesh_trimesh(trimesh1,trimesh2,
                                                                    n_seeds=n_seeds,
                                                                    n_projections=n_projections,
                                                                    p=p,
                                                                    seed=seed)
    
    else:

        area1 = igl.doublearea(trimesh1.vertices, trimesh1.faces) / 2.
        area1 = area1/np.nansum(area1)
        pts1 = igl.barycenter(trimesh1.vertices,trimesh1.faces)

        dS_du, dS_dv = uzip.gradient_uv(uv2, eps=eps, pad=pad) # might be more accurate to convert ...
        # area of the original surface.
        area2 = np.linalg.norm(np.cross(dS_du, 
                                        dS_dv), axis=-1)# # use the cross product
        area2 = area2/np.nansum(area2)
        pts2 = uv2.reshape(-1,uv2.shape[-1]).copy() # just make this a point cloud.
        area2 = area2.ravel().copy()

        # print(pts1.shape, pts2.shape)
        # print(area1.shape, area2.shape)
        sliced_W_dist_iters = []
        for iter_ii in np.arange(n_seeds):
            if mode=='max':
                sliced_W_dist = ot.max_sliced_wasserstein_distance(pts1, 
                                                           pts2, 
                                                           a=area1, 
                                                           b=area2, 
                                                           p=p,
                                                           n_projections=n_projections, 
                                                           seed=seed) # hm...
            else:
                sliced_W_dist = ot.sliced_wasserstein_distance(pts1, 
                                                               pts2, 
                                                               a=area1, 
                                                               b=area2, 
                                                               p=p,
                                                               n_projections=n_projections, 
                                                               seed=seed) # hm...
            sliced_W_dist_iters.append(sliced_W_dist)
        sliced_W_dist = np.nanmean(sliced_W_dist_iters) 

    return sliced_W_dist


def diff_area_trimesh_trimesh(trimesh1, trimesh2):
    r""" Difference in total surface area between two triangle meshes

    Parameters
    ----------
    trimesh1 : trimesh.Trimesh
        a 3D triangle mesh
    trimesh2 : trimesh.Trimesh 
        a 3D triangle mesh

    Returns 
    -------
    diff_area : scalar
        the signed difference in the total surface area between mesh 1 and mesh 2

    """
    import igl 
    import numpy as np 

    area1 = np.nansum(igl.doublearea(trimesh1.vertices, trimesh1.faces) / 2.)
    area2 = np.nansum(igl.doublearea(trimesh2.vertices, trimesh2.faces) / 2.)

    diff_area = area1-area2

    return diff_area

def diff_area_trimesh_uv(trimesh1, uv2, eps=1e-12, pad=False, uv_to_trimesh=False):
    r""" Difference in total surface area between a triangle mesh and a (u,v) image parameterized 3D mesh 

    Parameters
    ----------
    trimesh1 : trimesh.Trimesh
        a 3D triangle mesh
    uv2 : (U,V,3) array
        a (u,v) image parameterized 3D surface
    eps : scalar
        a small constant for numerical stability 
    pad : bool
        if True, uses edge padding to compute the finite differences for evaluating the differential areas of ``uv2``
    uv_to_trimesh : bool
        if True, convert the (u,v) image parameterized 3D triangle mesh into a 3D triangle mesh before evaluating the difference. 
        
    Returns 
    -------
    diff_area : scalar
        the signed difference in the total surface area between the triangle mesh and the (u,v) parameterized mesh 

    """
    import igl 
    import numpy as np 
    import trimesh

    area1 = np.nansum(igl.doublearea(trimesh1.vertices, trimesh1.faces) / 2.)

    if uv_to_trimesh:
        uv_vertex_indices_all, uv_triangles = get_uv_grid_tri_connectivity(np.zeros(uv2.shape[:2])) # oh... this is only valid uv unwrap not topography!. -> double check the ushape3D error analyses!. 
        uv_pos_3D_v = uv2.reshape(-1,3)[uv_vertex_indices_all]
        trimesh2 = trimesh.Trimesh(vertices=uv_pos_3D_v, faces=uv_triangles, 
                                    process=False, validate=False)
        area2 = np.nansum(igl.doublearea(trimesh2.vertices, trimesh2.faces) / 2.)
    else:
        dS_du, dS_dv = uzip.gradient_uv(uv2, eps=eps, pad=pad) # might be more accurate to convert ...
        # area of the original surface.
        area2 = np.linalg.norm(np.cross(dS_du, 
                                        dS_dv), axis=-1)# # use the cross product
        area2 = np.nansum(area2)

    # difference in area. 
    diff_area = np.abs(area1-area2)

    return diff_area
