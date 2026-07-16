# J3 (Elbow) Correlation Analysis
As verified via the Jacobian SVD/Cosine Similarity analysis:
Right Elbow (J3):
- Camera rx (roll): 0.44
- Camera rz (yaw): 0.50
- Head Tilt: -0.45

Left Elbow (J3):
- Camera rx (roll): -0.35
- Camera rz (yaw): 0.54

This means that while J3 is NOT completely collinear with the Camera or Head (which are 0.996 correlated), there is still a ~50% correlation. If the Camera offset is forced to 0, J3 WILL absorb some of the error, making the J3 calibration slightly inaccurate.
