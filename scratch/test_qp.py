import numpy as np
from qpsolvers import solve_qp

# Test problem
# min 1/2 x^T P x + q^T x
# s.t. lb <= x <= ub

P = np.array([[1.0, 0.0], [0.0, 1.0]])
q = np.array([-1.0, -1.0]) # Unconstrained minimum at x=[1, 1]

# Case 1: Bounds should push x to [0.5, 0.5]
lb = np.array([-1.0, -1.0])
ub = np.array([0.5, 0.5])

print(f"Testing with solver='osqp'...")
try:
    x = solve_qp(P, q, lb=lb, ub=ub, solver="osqp")
    print(f"Result (Case 1: ub=[0.5, 0.5]): {x}")
    if x is not None:
        if np.all(x <= 0.500001):
            print("SUCCESS: Constraints respected.")
        else:
            print("FAILURE: Constraints IGNORED!")
    else:
        print("FAILURE: Solver returned None")
except Exception as e:
    print(f"ERROR: {e}")

# Case 2: No bounds (should be [1, 1])
x_none = solve_qp(P, q, solver="osqp")
print(f"Result (Case 2: No bounds): {x_none}")
