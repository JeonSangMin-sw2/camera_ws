import numpy as np
from qpsolvers import solve_qp, available_solvers

print(f"Available solvers: {available_solvers}")

P = np.array([[1.0, 0.0], [0.0, 1.0]])
q = np.array([-1.0, -1.0])

# Strict bounds
lb = np.array([-1.0, -1.0])
ub = np.array([0.1, 0.1])

print(f"\nTesting with solver='osqp' (ub=[0.1, 0.1])...")
x = solve_qp(P, q, lb=lb, ub=ub, solver="osqp")
print(f"Result: {x}")

# Try another solver if available (scipy is usually available via qpsolvers)
if "scipy" in available_solvers:
    print(f"\nTesting with solver='scipy'...")
    x_scipy = solve_qp(P, q, lb=lb, ub=ub, solver="scipy")
    print(f"Result: {x_scipy}")
