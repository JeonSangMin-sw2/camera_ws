import numpy as np
from scipy.optimize import least_squares

radius_6 = 54.0363
radius_5 = 177.1794
radius_4 = 185.1176
x_nom = 0.0
y_nom = 54.0
z_nom = -48.0
z_sign = -1.0
L_5_ee = 126.1

def residuals_trans(params):
    xe, ye, ze = params
    r6_pred = np.sqrt(xe**2 + ye**2)
    Z_prime = ze + z_sign * L_5_ee
    r5_pred = np.sqrt(xe**2 + Z_prime**2)
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

initial_guess = [x_nom + 1.0, y_nom, z_nom]
lower_bounds = [x_nom - 30.0, y_nom - 30.0, -250.0]
upper_bounds = [x_nom + 30.0, y_nom + 30.0, 10.0]

opt_res = least_squares(residuals_trans, initial_guess, bounds=(lower_bounds, upper_bounds), loss='huber')
print("Solution with perturbation:", opt_res.x)
print("Cost:", opt_res.cost)
print("Residuals:", residuals_trans(opt_res.x))
