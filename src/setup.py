"""
        \    /\
    _____)__( ')_________
   /     (  /  )        /|
  /       \(__)|       / |
 /____________________/ /|
|_____________________|/||
|| ||                || 
||                   || 

"""

from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils.extension import Extension
from distutils.command.clean import clean
import os
import numpy as np


class CleanButCleaner(clean):
    def run(self):
        super().run()

        if self.all is not None:
            print("Removing compiled modules")
            for root, _, files in os.walk("."):
                for f in files:
                    if f.endswith(".c") or f.endswith(".cpp") or f.endswith(".so") or f.endswith(".pyd") or f.endswith(".html"):
                        os.remove(os.path.join(root, f))

setup(
    name = "TheCatIsOnTheTablut",
    ext_modules = cythonize(
        "**/*.pyx", 
        annotate = True,
    ),
    include_dirs = [
        np.get_include()
    ],
    cmdclass={
        'clean': CleanButCleaner,
    },
)