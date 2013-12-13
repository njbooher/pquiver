from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(ext_modules=[Extension(
                   name="bqfast",
                   sources=["bqfast.pyx"],
                   language="c",
                   include_dirs = [".", "/opt/cuda/include", np.get_include()],
                   library_dirs = [".", "/opt/cuda/lib64"],
                   libraries = ["cudart", "cuda"],
                   extra_objects = ["bqcuda.o"]
                   )],
      cmdclass={'build_ext': build_ext})
