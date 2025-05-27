import numpy as np 
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import OptimalTransport

def constructOMEGA(n, m):
    iden_vec1m = np.kron(np.eye(n), np.ones((m, 1)))  
    vecIm = -np.tile(np.eye(m), (n, 1))  
    return np.hstack((iden_vec1m, vecIm)) 

def construct_Gamma(T_k):
    """
    Construct the transformation matrix Gamma from the optimal transportation matrix T^(k).

    Parameters:
    T_k (numpy.ndarray): The optimal transportation matrix of shape (ns, nt).

    Returns:
    numpy.ndarray: The constructed Gamma matrix of shape (ns + nt, ns + nt).
    """
    ns, nt = T_k.shape  # Get source and target dimensions
    
    # Compute omega^(k) = diag(T^(k) 1_nt)^(-1) T^(k)
    row_sums = np.sum(T_k, axis=1)  # Sum along rows (shape: (ns,))
    diag_inv = np.diag(1 / row_sums)  # Inverse diagonal matrix (ns x ns)
    omega_k = diag_inv.dot(T_k)  # Compute transformation (ns x nt)

    # Construct Gamma matrix
    Gamma = np.block([
        [np.diag(np.diag(omega_k)), np.zeros((ns, nt))],  # Top-left (ns x nt)
        [np.zeros((nt, ns)), np.eye(nt)]  # Bottom-right (nt x nt identity)
    ])
    
    return Gamma



def convert(m, n):
    A = np.arange(m * n).reshape(m, n)
    B = np.zeros((m + n, m * n), dtype=int)
    B[np.arange(m)[:, None], A] = 1
    B[m + np.arange(n)[:, None], A.T] = 1
    return B

def constructGamma(ns, nt, T):
    top = np.hstack((np.zeros((ns, ns)), ns * T))            # shape: (ns, ns+nt)
    bottom = np.hstack((np.zeros((nt, ns)), np.eye(nt)))     # shape: (nt, ns+nt)
    return np.vstack((top, bottom))                          # shape: (ns+nt, ns+nt)


def solveOT(ns, nt, S_, h_, X_):
    #Cost vector
    cost = 0
    p = X_.shape[1] - 1 


    cost = cdist(X_[:ns], X_[ns:], metric='sqeuclidean').reshape(-1,1)

    # Solve wasserstein distance
    res = linprog(cost, A_ub=-np.identity(ns * nt), b_ub=np.zeros((ns * nt, 1)), 
                  A_eq=S_, b_eq=h_, method='simplex', 
                  options={'maxiter': 100000})
    # Transport Map
    Tobs = res.x.reshape((ns,nt))

    return {"T": Tobs, "basis": res.basis}
