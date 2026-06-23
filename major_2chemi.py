import numpy as np
from scipy.optimize import linprog

# ==========================================
# Task 6: Initial Network Optimization
# ==========================================
print("--- Task 6: Base Optimization ---")

# 1. Define cost matrix (Flattened into a 1D array for linprog)
# Cost of transferring heat from H_i to C_j
# Matrix: 
# [3, 6, 7]
# [5, 4, 6]
# [8, 5, 3]
c = np.array([3, 6, 7, 5, 4, 6, 8, 5, 3])

# 2. Define constraint matrices
# Supply constraints (Ax <= b)
# Heat available: H1=80, H2=60, H3=40
A_supply = [
    [1, 1, 1, 0, 0, 0, 0, 0, 0],  # H1
    [0, 0, 0, 1, 1, 1, 0, 0, 0],  # H2
    [0, 0, 0, 0, 0, 0, 1, 1, 1]   # H3
]
b_supply = [80, 60, 40]

# Demand constraints (Originally Ax >= b, multiply by -1 for linprog: -Ax <= -b)
# Heat required: C1=50, C2=70, C3=40
A_demand = [
    [-1,  0,  0, -1,  0,  0, -1,  0,  0],  # C1
    [ 0, -1,  0,  0, -1,  0,  0, -1,  0],  # C2
    [ 0,  0, -1,  0,  0, -1,  0,  0, -1]   # C3
]
b_demand = [-50, -70, -40]

# Construct global constraint matrices
A_ub = np.vstack((A_supply, A_demand))
b_ub = np.concatenate((b_supply, b_demand))

# Non-negativity bounds (x_ij >= 0)
x_bounds = [(0, None) for _ in range(9)]

# 3. Solve using linprog (using 'highs' method which is fast and accurate)
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

if result.success:
    optimal_flows = result.x.reshape((3, 3))
    print(f"Minimum Base Cost: {result.fun}")
    print("\nOptimal Heat Transfers (Matrix Form):")
    print(np.round(optimal_flows, 2))
    
    print("\nNetwork Connection Flows:")
    for i in range(3):
        for j in range(3):
            flow = optimal_flows[i, j]
            if flow > 0:
                print(f"  H{i+1} -> C{j+1}: {flow:.2f} units")
else:
    print("Optimization failed:", result.message)


# ==========================================
# Task 7: Network Modification
# ==========================================
print("\n--- Task 7: Imposing Engineering Restrictions ---")
# Rule: Connections H1 -> C3 and H3 -> C1 are not allowed.
# We restrict them by modifying their variable bounds to 0.
# x13 is index 2, x31 is index 6.

bounds_mod = list(x_bounds)
bounds_mod[2] = (0, 0)  # Eliminate path H1 -> C3
bounds_mod[6] = (0, 0)  # Eliminate path H3 -> C1

result_mod = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_mod, method='highs')
if result_mod.success:
    print(f"Minimum Cost (Restricted): {result_mod.fun}")
else:
    print("Restricted optimization failed.")


# ==========================================
# Task 8: Infeasible Design Check
# ==========================================
print("\n--- Task 8: Infeasibility Check ---")
# Modify C2 demand to 120 units
b_demand_new = [-50, -120, -40]
b_ub_infeasible = np.concatenate((b_supply, b_demand_new))

result_inf = linprog(c, A_ub=A_ub, b_ub=b_ub_infeasible, bounds=x_bounds, method='highs')

if not result_inf.success:
    print(f"System status: INFEASIBLE")
    print(f"Reason provided by solver: {result_inf.message}")
else:
    print("Optimization Successful (Unexpectedly!)")