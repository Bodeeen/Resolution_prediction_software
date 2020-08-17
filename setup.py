import sys

from cx_Freeze import setup, Executable

import frcpredict

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["frcpredict.ui"],
                     "include_files": ["data"]}

setup(name=frcpredict.__title__,
      version=frcpredict.__version__,
      description=frcpredict.__summary__,
      options={"build_exe": build_exe_options},
      executables=[Executable("frcpredict/__main__.py", targetName="frcpredict", base=None)])
