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
from distutils.command.clean import clean
import os


class CleanButCleaner(clean):
    def run(self):
        super().run()

        if self.all is not None:
            print("Removing compiled modules")
            for f in [ f for f in os.listdir("./gametree") if f.endswith(".c") or f.endswith(".so") or f.endswith(".pyd") ]:
                os.remove(os.path.join("./gametree", f))

setup(
    name = "TheCatIsOnTheTablut",
    ext_modules = cythonize("gametree/*.py", exclude=["gametree/__init__.py"]),
    cmdclass={
        'clean': CleanButCleaner,
    },
)