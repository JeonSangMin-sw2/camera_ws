from setuptools import setup, find_packages

setup(
    name="camera_ws",
    version="0.1",
    package_dir={"": "core"},
    packages=find_packages(where="core"),
    py_modules=[
        "marker_detection",
        "robot_motion",
        "paths",
        "language",
        "i18n",
        "wizard_widget",
    ],
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    install_requires=[
        "numpy>=2.0.1",
        "opencv-contrib-python",
        "PySide6",
        "scipy",
        "pyyaml",
    ],
)