import numpy as np
from qpsolvers import solve_qp

P = np.array([[1.0, 0.0], [0.0, 1.0]])
q = np.array([-1.0, -1.0])
lb = np.array([-1.0, -1.0])
ub = np.array([0.1, 0.1])

print("Testing with default tolerances...")
x1 = solve_qp(P, q, lb=lb, ub=ub, solver="osqp")
print(f"Result: {x1} (Violation: {np.max(x1 - 0.1):.2e})")

print("\nTesting with stricter tolerances (eps_abs=1e-8, eps_rel=1e-8)...")
x2 = solve_qp(P, q, lb=lb, ub=ub, solver="osqp", eps_abs=1e-8, eps_rel=1e-8)
print(f"Result: {x2} (Violation: {np.max(x2 - 0.1):.2e})")
