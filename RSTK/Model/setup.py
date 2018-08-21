from Cython.Distutils import build_ext
from Cython.Build import cythonize
from distutils.core import setup, Extension
import numpy

setup(
    name='SVD',
    ext_modules=cythonize(Extension('SVD',['SVD.pyx'], include_dirs=[numpy.get_include()]))
)