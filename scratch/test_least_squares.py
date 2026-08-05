import numpy as np
from scipy.optimize import least_squares

# Parameters
radius_6 = 54.5094
radius_5 = 176.2973
radius_4 = 184.1545
L_5_ee = 126.1000
z_sign = -1.0

x_nom = 0.0
y_nom = -54.0
z_nom = -48.0

def residuals_trans(params):
    xe, ye, ze = params
    r6_pred = np.sqrt(xe**2 + ye**2)
    Z_prime = ze + z_sign * L_5_ee
    r5_pred = np.sqrt(xe**2 + Z_prime**2)
    # J4 sweep radius pred
    r4_pred = np.sqrt((ze + z_sign * L_5_ee)**2 + ye**2)
    res = [
        r6_pred - radius_6,
        r5_pred - radius_5,
        r4_pred - radius_4
    ]
    reg_weight = 1e-7
    res.append(reg_weight * (xe - x_nom))
    res.append(reg_weight * (ye - y_nom))
    res.append(reg_weight * (ze - z_nom))
    return res

initial_guess = [x_nom, y_nom, z_nom]
lower_bounds = [x_nom - 30.0, y_nom - 30.0, -250.0]
upper_bounds = [x_nom + 30.0, y_nom + 30.0, 10.0]

opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
print("Optimized xe, ye, ze:", opt_res.x)
print("Residuals:", residuals_trans(opt_res.x)[:3])
