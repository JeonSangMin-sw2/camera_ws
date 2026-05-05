from setuptools import setup, find_packages

setup(
    name="camera_ws",
    version="0.1",
    # core 폴더 내부의 파일들을 루트 모듈로 사용하고, calibration 폴더도 개별 패키지로 유지
    package_dir={"": "core", "calibration": "calibration"},
    packages=find_packages(where="core") + ["calibration"],
    py_modules=["marker_detection", "calibration_core", "homeoffset_core"],
    # .yaml 파일들을 포함시키기 위한 설정
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    install_requires=[
        "numpy",
        "opencv-python",
        "PySide6",
        "scipy",
        "pyyaml",
    ],
)
