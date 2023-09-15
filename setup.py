from setuptools import setup

import os
import re

HERE = os.path.abspath(os.path.dirname(__file__))

exc_folders = ['__pycache__', '__init__.py']
subpkgs = os.listdir(os.path.join(HERE,'unwrap3D'))
subpkgs = [pkg for pkg in subpkgs if pkg not in exc_folders]
print(subpkgs)

with open("requirements.txt", "r") as fp:
    install_requires = list(fp.read().splitlines())

setup(name='u-Unwrap3D',
	  version='0.1.0',
	  description='Surface-guided computing of arbitrary 3D surfaces and volumes by bidirectional mapping to lower dimensions',
	  author='Felix Y. Zhou',
	  packages=['unwrap3D'] + ['unwrap3D.'+ pkg for pkg in subpkgs],
	#   package_dir={"": "unwrap3D"}, # directory containing all the packages (e.g.  src/mypkg, src/mypkg/subpkg1, ...)
	  include_package_data=True,
	  install_requires=install_requires,
)

