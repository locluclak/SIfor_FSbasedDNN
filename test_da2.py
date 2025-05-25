import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog

# Data
n_s, n_t, C = 100, 100, 3
X_s = np.random.rand(n_s, 2)  # Shape (100, 2)
X_t = np.random.rand(n_t, 2)  

# y_s = np.repeat(np.arange(C), n_s // C+1)[:n_s]  # Shape (100,), balanced classes
y_s = np.random.randint(0, C, n_s)
np.random.shuffle(y_s)  # Randomize order

# Validate inputs
assert isinstance(n_s, int) and n_s > 0, f"n_s must be a positive integer, got {n_s}"
assert isinstance(n_t, int) and n_t > 0, f"n_t must be a positive integer, got {n_t}"
assert isinstance(C, int) and C > 0, f"C must be a positive integer, got {C}"
assert X_s.shape == (n_s, 2), f"X_s shape {X_s.shape} != ({n_s}, 2)"
assert X_t.shape == (n_t, 2), f"X_t shape {X_t.shape} != ({n_t}, 2)"
assert y_s.shape == (n_s,), f"y_s shape {y_s.shape} != ({n_s},)"
assert np.all((y_s >= 0) & (y_s < C)), f"y_s contains invalid class labels: {np.unique(y_s)}"

# Source and target weights
a = np.ones(n_s) / n_s  # Shape (100,)
b = np.ones(n_t) / n_t  # Shape (100,)

# Source class proportions
pi_s = np.array([np.sum(y_s == c) / n_s for c in range(C)])  # Shape (3,)
I_c = [np.where(y_s == c)[0] for c in range(C)]  # List of index arrays
print("pi_s:", pi_s, "sum:", np.sum(pi_s))

# Check for empty classes
for c in range(C):
    if len(I_c[c]) == 0:
        raise ValueError(f"No source samples for class {c}")

# Cost matrix
C_matrix = np.sum((X_s[:, np.newaxis, :] - X_t[np.newaxis, :, :])**2, axis=2)  # Shape (100, 100)
C_matrix /= np.max(C_matrix)  # Normalize
assert C_matrix.shape == (n_s, n_t), f"C_matrix shape {C_matrix.shape} != ({n_s}, {n_t})"
assert np.all(np.isfinite(C_matrix)), "Cost matrix contains NaN or Inf"

# Vectorize variables
n_vars = n_s * n_t + n_t * C 
c_vec = np.zeros(n_vars)
c_vec[:n_s * n_t] = C_matrix.ravel()

# Constraints
n_constraints = n_s + n_t + C + n_t * C + n_t  # 100 + 100 + 3 + 100 * 3 + 100 = 600
A_eq = np.zeros((n_constraints, n_vars))
b_eq = np.zeros(n_constraints)

# Constraint indices
row = 0

# 1. Source marginal constraints
for i in range(n_s):
    A_eq[row, i * n_t:(i + 1) * n_t] = 1
    b_eq[row] = a[i]
    row += 1

# 2. Target marginal constraints
for j in range(n_t):
    A_eq[row, j::n_t] = 1
    b_eq[row] = b[j]
    row += 1

# 3. Class proportion constraints
for c in range(C):
    for i in I_c[c]:
        A_eq[row, i * n_t:(i + 1) * n_t] = 1
    b_eq[row] = pi_s[c]
    row += 1

# 4. Class coupling constraints
for j in range(n_t):
    for c in range(C):
        for i in I_c[c]:
            A_eq[row, i * n_t + j] = 1
        A_eq[row, n_s * n_t + j * C + c] = -b[j]
        row += 1

# 5. Probability constraints
for j in range(n_t):
    A_eq[row, n_s * n_t + j * C:n_s * n_t + (j + 1) * C] = 1
    b_eq[row] = 1
    row += 1

# Convert to sparse matrix
A_eq = csr_matrix(A_eq)

# Verify shapes
print("A_eq shape:", A_eq.shape, "Expected:", (n_constraints, n_vars))
print("c_vec shape:", c_vec.shape, "Expected:", (n_vars,))
assert A_eq.shape == (n_constraints, n_vars), f"A_eq shape {A_eq.shape} != ({n_constraints}, {n_vars})"
assert c_vec.shape == (n_vars,), f"c_vec shape {c_vec.shape} != ({n_vars},)"

# Bounds
bounds = [(0, None)] * n_vars
# res = linprog(c_vec, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': True})

# # Solve
# try:
#     res = linprog(c_vec, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': True})
# except Exception as e:
#     print(f"Solver exception: {e}")
res,basis = linprog(c_vec, A_eq=A_eq, b_eq=b_eq, method='simplex')

# Check result
if res.success:
    x_star = res.x
    Gamma_star = x_star[:n_s * n_t].reshape(n_s, n_t)
    Q_star = x_star[n_s * n_t:].reshape(n_t, C)
    pi_t = b @ Q_star
    print("Optimal Transport Plan shape:", Gamma_star.shape)
    print("Source Class Proportions:", pi_s, "sum:", np.sum(pi_s))
    print("Target Class Proportions:", pi_t, "sum:", np.sum(pi_t))
    print("Source marginal error:", np.sum(np.abs(np.sum(Gamma_star, axis=1) - a)))
    print("Target marginal error:", np.sum(np.abs(np.sum(Gamma_star, axis=0) - b)))
    print("Class proportion error:", [np.sum(Gamma_star[I_c[c], :]) - pi_s[c] for c in range(C)])
else:
    print(f"Solver failed: {res.message}")
    # Diagnose infeasibility
    print("Checking constraint consistency...")
    print("Sum of source marginals:", np.sum(a))
    print("Sum of target marginals:", np.sum(b))
    print("Sum of class proportions:", np.sum(pi_s))
    print("Class sizes:", [len(I_c[c]) for c in range(C)])