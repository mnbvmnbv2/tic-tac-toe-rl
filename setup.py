from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="ttt.cython.c_tictactoe",
        sources=["python/ttt/cython/c_tictactoe.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="ttt.cython.c_tictactoe_pvp",
        sources=["python/ttt/cython/c_tictactoe_pvp.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="tic-tac-toe-rl",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)