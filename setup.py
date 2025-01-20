# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="c_tictactoe",  # This should match the name of the .pyx file without the extension
        sources=["envs/c_tictactoe.pyx"],  # The Cython file to be compiled
        include_dirs=[np.get_include()],  # Include directory for numpy
    ),
    Extension(
        name="c_tictactoe_pvp",
        sources=["pvp/c_tictactoe_pvp.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="C_TicTacToe",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    py_modules=["c_tictactoe", "c_tictactoe_pvp"],
)
