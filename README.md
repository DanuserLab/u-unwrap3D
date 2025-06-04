# u-Unwrap3D
## Library for 3D Surface-guided computing
<p align="center">
  <img src="imgs/u_unwrap3D_overview.png" width="800"/>
</p>

<!-- TOC start -->
   * [Library Features](#library-features)
   * [Getting Started](#getting-started)
   * [Dependencies](#dependencies)
   * [Installation](#installation)
   * [New functionality](#new-functionality)
   * [Questions and Issues](#questions-and-issues)
   * [Danuser Lab Links](#danuser-lab-links)
<!-- TOC end -->

#### June 2025 
- installation through pypi: `pip install u-Unwrap3D`
- introducing uniformization of area as a proxy loss for robustly achieving minimal equiareal spherical and disk parameterization via `unwrap3D.Mesh.meshtools.uniform_distortion_flow_relax_sphere` and `unwrap3D.Mesh.meshtools.uniform_distortion_flow_relax_disk` functions

#### :star2: v1 (Oct 2024) :star2:

major updates to code and functionality, preprint to come, now u-Unwrap3D works for any mesh :D, including:
- **new docs:** [work-in-progress book](https://fyz11.github.io/u-Unwrap3D_notebooks/intro.html) for theory understanding and easy reading of code examples e.g. protrusion segmentation. This will also serve to showcase the interesting things you can do with u-Unwrap3D
- **harmonic distance transform** for full interior mapping of cells with high-curvature protrusions
- **genus-0 shrinkwrap** of arbitrary meshes to guarantee spherical parameterization
- **new changepoint detection** method of gaussian curvature for finding tighter reference surface shapes
- **aspect ratio optimization** of uv-map dimensions to best represent $S_\text{ref}(x,y,z)$ i.e. not just (1:2) but (1:h) height-to-width ratio
- **Botsch remeshing** from CGAL for more robust and faster isotropic remeshing, replacing pyacvd
- **transfer scalar and label measurements** between two surface meshes with one function
- **one-function** for rotation optimization, uv-map, build topography space
- **one-function** for curvature measurement
- **updated notebooks** to illustrate new functions
 

#### April 10, 2023
u-unwrap3D is a Python library of functions designed to map 3D surface and volume data into different representations which are more optimal for the desired computing task. For example it is far easier to track surface features on the 3D sphere or 2D plane. Similarly the 3D reference surface enables comparison of global shape across cells and the topography surface specifically highlight surface protrusions. We pay particular attention to minimize conformal and equiareal distortion errors and avoid surface-cutting and restitching. The representations were chosen to preserve the full surface and simplify downstream quantitative characterization of surface features with particular attention to single cell biology.

It is associated with the paper currently under review, [**Surface-guided computing to analyze subcellular morphology and membrane-associated signals in 3D**](https://doi.org/10.1101/2023.04.12.536640), *bioRxiv*, 2023, written by Felix Y. Zhou, Andrew Weems, Gabriel M. Gihana, Bingying Chen, Bo-Jui Chang, Meghan K. Driscoll and [Gaudenz Danuser](https://www.danuserlab-utsw.org/).


## Example of u-unwrap3D applied to complex cell surfaces
u-unwrap3D can handle a large variety of input surfaces. A key motivation of this work was to allow the mapping of high-genus cell surfaces that are common when meshing binary cell segmentations from microscopy. Here are some examples of the surfaces we can map. More detailed characterization is included in our paper.
<p align="center">
  <img src="imgs/u_unwrap3D_example_cell_panel.png" width="800"/>
</p>

## Library Features
u-unwrap3D is a library motivated by scipy / numpy / opencv that provides re-usable functions that can be used to build-up complex processing pipelines. Whilst its primary motivation is to provide mesh and image processing functions for handling 3D geometry and surfaces, it does also include a plethora of functions associated with preprocessing/postprocessing and analysis. All the functions are organized into submodules related to the high-level task they associated with e.g. mesh processing, registration, file handling, geometry etc. A brief summary is provided below. The detailed functions can be found from the mainpage of the documentation in this repository, docs/build/html/index.html or directly docs/build/html/py-modindex.html.

|Module                   |Functionality|
|-------------------------|-------------|
|Analysis_Functions       | Functions for analyzing topographic representation (mapping to Cartesian coordinates, watershed depth propagation) and timeseries (cross-correlation)|
|Features                 | 2D SIFT features extractor  |
|Geometry                 | Functions for computing 3D geometric transformations of points e.g. rotating points, steregraphic mapping 3D sphere to 2D plane |
|Image_Functions          | Functions for general image manipulation e.g. interpolation, intensity normalization|
|Mesh                     | Functions for mesh processing, e.g. conformalized mean curvature flow, mesh propagation, conformal, equiareal error computation, area-distortion relaxation, voxelization, reading/writing meshes|
|Parameters               | Defining default parameters for e.g. 3D demons registration, 2D optical flow|
|Registration             | Volumetric registration through SimpleITK and wrappers for Matlab|
|Segmentation             | Functions for working with segmentation e.g. morphological operations, threshold-based segmentation, signed distance transform, mean curvature|
|Tracking                 | 2D optical flow, 2D optical flow assisted bounding box tracker |
|Unzipping                | Functions for working with image-based uv-parameterized surfaces, e.g. surface propagation, conformal/equiareal distortion measures, spherical boundary padding |
|Utility_Functions        | Functions for file manipulation e.g. folder creation, reading image formats|
|Visualisation            | Functions for plotting e.g. colormapping numpy arrays, forcing equal aspect ratio for matplotlib 3D plotting |

## Getting Started
The simplest way to get started is to check out the included notebooks in this repository which walks through the steps described in the paper for obtaining all representation starting from step0: the extraction of surface from a binary cell segmentation. 

Alternatively, you may prefer to read the tutorials from the [u-Unwrap3D jupyter-book](https://fyz11.github.io/u-Unwrap3D_notebooks/03_basic_workflow/readme.html) which can be navigated more easily. 

## Dependencies
u-unwrap3D relies on the following packages for various functionalities. All can be readily installed using conda or pip. Not all needs to be installed. Feel free to install as needed / when an error is thrown. The key ones are below:
- [libigl](https://libigl.github.io/libigl-python-bindings/tut-chapter0/) - `pip install libigl` : for mesh processing
- [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils) - `pip install point-cloud-utils` : for computing point cloud metrics
- [trimesh](https://trimsh.org/) - `pip install trimesh` : for mesh io and processing
- [CGAL](https://pypi.org/project/pyacvd/) - `pip install cgal` : for isotropic remeshing, alpha wrap
- [transforms3d](https://pypi.org/project/transforms3d/) - `pip install transforms3d` : for 3D coordinate transforms 
- [numpy](https://numpy.org/) - `pip install numpy` : for general computing
- [scipy](https://www.scipy.org/) - `pip install scipy` : for image filtering, transformations and sparse linear algebra
- [scikit-image](https://scikit-image.org/) - `pip install scikit-image` : general image processing
- [matplotlib](https://matplotlib.org/) - `pip install matplotlib` : general plotting 
- [opencv](https://pypi.org/project/opencv-contrib-python/) - `pip install opencv-contrib-python` : for optical flow tracking, inpainting, various image processing
- [tqdm](https://tqdm.github.io/) - `pip install tqdm` : for progressbar

More optional / certain functions:
- [SimpleITK](https://simpleitk.org/) - `pip install SimpleITK` : for optional volumetric registration
- [robust-laplacian](https://pypi.org/project/robust-laplacian/) - `pip install robust-laplacian` : for using the robust laplacian of Sharpe et al. instead of cotangent Laplacian 
- [pygeodesic](https://pypi.org/project/pygeodesic/) - `pip install pygeodesic` : for fast Djikstras algorithm for computing exact geodesic distance on triangle meshes
- [potpourri3d](https://github.com/nmwsharp/potpourri3d) - `pip install potpourri3d` : for using heat method to compute approximate geodesic distance on triangle meshes with multiple sources

## Installation
The above dependencies and library can be installed by git cloning the repository and running pip in the cloned folder with python>=3.8. We have tested and recommend 3.9, 3.10, 3.11.
```
pip install .
```
You can also install directly from the github without cloning:
```
pip install u-unwrap3D@git+https://github.com/DanuserLab/u-unwrap3D.git
```
and also through pypi using pip:
```
pip install u-Unwrap3D
```

**encountered installation errors**:

`scikit-fmm` does not have precompiled wheels in pip, therefore you may get compilation errors, particularly in Windows. Either remove `scikit-fmm` from requirements.txt or install first before pip through conda-forge into your environment, `conda install -c conda-forge scikit-fmm`.

`libigl` may not yet have a wheel for your distibution, and your system fails to compile wheels. In this case try downgrading to an earlier version. We have encountered this on Linux.   

## New functionality
New tools will be constantly added to improve useability and applicability. You can help by opening a GitHub issue.  

## Questions and Issues
Feel free to open a GitHub issue or email me at felixzhou1@gmail.com.

## Danuser Lab Links
[Danuser Lab Website](https://www.danuserlab-utsw.org/)

[Software Links](https://github.com/DanuserLab/)
