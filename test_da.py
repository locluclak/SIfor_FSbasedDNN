import numpy as np
import cvxpy as cp

# Data
n_s, n_t, C = 100, 100, 3
X_s = np.random.rand(n_s, 2)
X_t = np.random.rand(n_t, 2)
y_s = np.random.randint(0, C, n_s)
a = np.ones(n_s) / n_s
b = np.ones(n_t) / n_t
pi_s = np.array([np.sum(y_s == c) / n_s for c in range(C)])
I_c = [np.where(y_s == c)[0] for c in range(C)]



# Cost matrix (efficient computation)
Cost = np.sum((X_s[:, np.newaxis, :] - X_t[np.newaxis, :, :])**2, axis=2)


# Variables
Gamma = cp.Variable((n_s, n_t))  
Q = cp.Variable((n_t, C))  

# Objective
objective = cp.Minimize(cp.sum(cp.multiply(Gamma, Cost)))

constraints = [
    Gamma >= 0,
    cp.sum(Gamma, axis=1) == a,
    cp.sum(Gamma, axis=0) == b,
    Q >= 0,
    cp.sum(Q, axis=1) == 1
]

for c in range(C):
    constraints.append(cp.sum(Gamma[I_c[c], :]) == pi_s[c])
    for j in range(n_t):
        constraints.append(cp.sum(Gamma[I_c[c], j]) == b[j] * Q[j, c])

prob = cp.Problem(objective, constraints)
prob.solve(verbose=True)


Gamma_star = Gamma.value
Q_star = Q.value
pi_t = b @ Q_star  

print("Optimal Transport Plan shape:", Gamma_star.shape)
print("Target Class Proportions:", pi_t, "sum:", np.sum(pi_t))