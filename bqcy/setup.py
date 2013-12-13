from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(ext_modules=[Extension(
                   name="bqcy",
                   sources=["bqcy.pyx"],
                   language="c",
                   include_dirs = [np.get_include()],
                   )],
      cmdclass={'build_ext': build_ext})
