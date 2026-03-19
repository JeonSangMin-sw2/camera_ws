import os
import glob
from setuptools import setup, find_packages

# Find all .py files in the current directory except setup.py
py_files = [os.path.splitext(f)[0] for f in glob.glob("*.py") if f != "setup.py"]

setup(
    name="rby1_camera_utils",
    version="0.1.0",
    description="Python utilities for RBY1 Camera",
    py_modules=py_files,
    packages=find_packages(),
)
