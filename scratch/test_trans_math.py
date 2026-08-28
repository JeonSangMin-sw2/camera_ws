import numpy as np
from scipy.optimize import least_squares

def test_trans(radius_6, radius_5, radius_4, x_nom=67.0, y_nom=0.0, z_nom=0.0, L_5_ee=125.0, z_sign=-1.0):
    def residuals_trans(params):
        xe, ye, ze = params
        r6_pred = np.sqrt(ye**2 + ze**2)
        Z_prime = ze + z_sign * L_5_ee
        r5_pred = np.sqrt(xe**2 + Z_prime**2)
        r4_pred = np.sqrt(xe**2 + ye**2)
        res = [
            (r6_pred - radius_6),
            (r5_pred - radius_5),
            (r4_pred - radius_4)
        ]
        reg = 1e-2
        res.append(reg * (xe - x_nom))
        res.append(reg * (ye - y_nom))
        res.append(reg * (ze - z_nom))
        return res

    x_init = [x_nom, y_nom, z_nom]
    opt_res = least_squares(residuals_trans, x_init, bounds=([30.0, -30.0, -30.0], [100.0, 30.0, 30.0]), loss='huber')
    xe, ye, ze = opt_res.x
    
    r6_err = abs(radius_6 - np.sqrt(ye**2 + ze**2))
    r5_err = abs(radius_5 - np.sqrt(xe**2 + (ze + z_sign * L_5_ee)**2))
    r4_err = abs(radius_4 - np.sqrt(xe**2 + ye**2))
    print(f"Solved (xe, ye, ze): [{xe:.2f}, {ye:.2f}, {ze:.2f}] mm")
    print(f"Residuals: r6_err={r6_err:.2f}mm, r5_err={r5_err:.2f}mm, r4_err={r4_err:.2f}mm")

# Left arm radii
test_trans(radius_6=21.59, radius_5=34.26, radius_4=13.86)
