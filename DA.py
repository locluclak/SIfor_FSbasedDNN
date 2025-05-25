import numpy as np
import OptimalTransport
import util
import solveinequalities.interval as bst_solving_eq
def interval_DA(ns, nt, p, B, S_, h_, a, b):
    Bc = np.delete(np.array(range(ns*nt)), B)

    OMEGA = OptimalTransport.constructOMEGA(ns,nt)
    
    w_tilde = 0
    r_tilde = 0
    o_tilde = 0

    for k in range(p):
        epk = np.zeros((p,1))
        epk[k][0] = 1
        OMEGA_kron_ej = np.kron(OMEGA, epk.T)
        OMEGA_kron_ej_dota = OMEGA_kron_ej.dot(a)
        OMEGA_kron_ej_dotb = OMEGA_kron_ej.dot(b)
        w_tilde += OMEGA_kron_ej_dota * OMEGA_kron_ej_dota
        r_tilde += 2*OMEGA_kron_ej_dota * OMEGA_kron_ej_dotb
        o_tilde += OMEGA_kron_ej_dotb * OMEGA_kron_ej_dotb

    # print(w_tilde + r_tilde*z + o_tilde*z**2)

    # Omega_a = OMEGA.dot(a)
    # Omega_b = OMEGA.dot(b)

  
    S_B_invS_Bc = np.linalg.inv(S_[:, B]).dot(S_[:, Bc])

    w = (w_tilde[Bc, :].T - w_tilde[B, :].T.dot(S_B_invS_Bc)).T
    r = (r_tilde[Bc, :].T - r_tilde[B, :].T.dot(S_B_invS_Bc)).T
    o = (o_tilde[Bc, :].T - o_tilde[B, :].T.dot(S_B_invS_Bc)).T
    # print(w.shape)
    list_intervals = []
    interval = bst_solving_eq.solve_system_of_quadratics(-o,-r,-w)
    # interval = [(-np.inf, np.inf)]
    # for i in range(w.shape[0]):
    #     g3 = - o[i][0]
    #     g2 = - r[i][0]
    #     g1 = - w[i][0]
    #     itv = util.solve_quadratic_inequality(g3,g2,g1)
    #     interval = util.interval_intersection(interval, itv)
    return interval
def construct_D2(y, C):
    """
    Constructs the D2 matrix based on the given formula.
    
    Parameters:
        y (list or array): List of class labels for n^(k) samples.
        C (list or array): List of unique class labels.
    
    Returns:
        np.ndarray: The constructed D2 matrix of shape (n_k, len(C)),
                    where C is the predefined list of label types.
    """
    n_k = len(y)
    class_counts = {cls: np.sum(np.array(y) == cls) for cls in C}
    
    D2 = np.zeros((n_k, len(C)))
    
    for i, label in enumerate(y):
        if label in C:
            class_idx = np.where(np.array(C) == label)[0][0]  # Find index of class in C
            D2[i, class_idx] = 1 / class_counts[label] if class_counts[label] > 0 else 0
    
    return D2

def SingleOTwithTargetshift(xs, xt, ys, h):
    ns, nt = xs.shape[0], xt.shape[0]
    C = range(h.shape[0])
    D2_s = construct_D2(ys, C)
    q = np.concatenate((D2_s.dot(h), np.ones((nt,1))/nt), axis = 0) 
    S = OptimalTransport.convert(ns, nt)

    S = S[:-1].copy()
    q = q[:-1].copy()
    
    X = np.vstack((xs, xt))

    T, basis_var = OptimalTransport.solveOT(ns, nt, S, q, X).values()
    row_sums = np.sum(T, axis=1) 
    col0 = np.zeros((ns + nt, ns))
    col1 = np.concatenate((np.linalg.inv(np.diag(row_sums)).dot(T), np.identity(nt)), axis=0)
    GAMMA = np.hstack((col0, col1))
    return GAMMA, S, q, basis_var


def SingleOT(xs, xt):
    ns, nt = xs.shape[0], xt.shape[0]

    q = np.vstack((
        np.full((ns, 1), 1/ns),
        np.full((nt, 1), 1/nt)
    ))

    S = OptimalTransport.convert(ns, nt)[:-1]
    q = q[:-1]

    X = np.vstack((xs, xt))

    # Solve OT
    result = OptimalTransport.solveOT(ns, nt, S, q, X)
    T = result["T"]
    basis_var = result["basis"]

    # Construct GAMMA
    col0 = np.zeros((ns + nt, ns))         # shape: (ns+nt, ns)
    col1 = np.vstack((ns * T, np.eye(nt))) # shape: (ns+nt, nt)
    GAMMA = np.hstack((col0, col1))        # shape: (ns+nt, ns+nt)

    return GAMMA, S, q, basis_var
