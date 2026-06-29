import numpy as np
from scipy.optimize import least_squares

# Inputs for Left Arm
radius_6 = 105.02
radius_5 = 230.24
L_5_ee = 300.0
x_nom = 0.0
y_nom = 77.48
z_nom = -71.62

opt_delta_5_rad = 0.0
opt_delta_6_rad = 0.0

def residuals_trans(params):
    xe, ye, ze = params
    r6_pred = np.sqrt(ye**2 + ze**2)
    Z_prime = ye * np.sin(opt_delta_6_rad) + ze * np.cos(opt_delta_6_rad) + L_5_ee
    r5_pred = np.sqrt(xe**2 + Z_prime**2)
    res = [
        r6_pred - radius_6,
        r5_pred - radius_5
    ]
    reg_weight = 1e-7
    res.append(reg_weight * (xe - x_nom))
    res.append(reg_weight * (ye - y_nom))
    res.append(reg_weight * (ze - z_nom))
    return res

initial_guess = [x_nom, y_nom, z_nom]
opt_res = least_squares(residuals_trans, initial_guess, loss='huber')
print("Solved:", opt_res.x)
print("Initial guess residuals:", residuals_trans(initial_guess))
print("Solved residuals:", residuals_trans(opt_res.x))
