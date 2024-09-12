# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension for env.pyx
extensions = [
    Extension(
        name="c_tictactoe",  # This should match the name of the .pyx file without the extension
        sources=["c_tictactoe.pyx"],  # The Cython file to be compiled
        include_dirs=[np.get_include()],  # Include directory for numpy
    )
]

setup(
    name="C_TicTacToe",  # Name of your package
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    py_modules=["c_tictactoe"],  # Explicitly specify which modules to include
)
