import torch
import numpy as np
import matplotlib.pyplot as plt
import integrated_gradients as ig
import model_train
import util
import construct_interval
from scipy.stats import kstest
from scipy.linalg import block_diag
import multiprocessing
global device
device = torch.device("cpu")

def FS_basedIGs(model, x_tensor, xbaseline_tensor,target, n_steps=30, percentile = 80, device = torch.device("cpu")):
    custom_attributions = (
        ig.custom_integrated_gradients(
            model, x_tensor, xbaseline_tensor, target=target, n_steps=30, device=device
        )
        .squeeze().cpu().detach().numpy()
    )
    # print("Custom Integrated Gradients Attributions:", np.mean(custom_attributions,axis=0))
    M = ig.get_threshold_attributions(custom_attributions, percentile=80)
    return M

def overconditioning(model, a, b, X, etaj, Sigma, M, target, threshold, n_steps, return_pvalue = False, zobs = 0):
    n, p = X.shape[0] //2, X.shape[1]
    
    I = [(-np.inf, np.inf)]
    O = [(-np.inf, np.inf)]
    T = [(-np.inf, np.inf)]
    for i in range(0,n*p,p):
        a_temp = np.vstack((a[i:i+p], a[i+n*p:i+n*p+p]))
        b_temp = np.vstack((b[i:i+p], b[i+n*p:i+n*p+p]))

        X_temp = np.hstack((X[i//p], X[i//p+n])).reshape(1,-1)

        I = util.interval_intersection(I, construct_interval.IGscondition(model, a_temp, b_temp, X_temp, n_steps=n_steps))
        O = util.interval_intersection(O, construct_interval.outputClasscondition(model, a_temp, b_temp, X_temp, c = target[i//p]))
    
    T = util.interval_intersection(I, construct_interval.thresholdcondition2(model, a, b, X, target, len(M), threshold = threshold, n_steps=n_steps,z=zobs))
    
    interval_oc = util.interval_intersection(
        I,
        util.interval_intersection(O, T)
    )

    if return_pvalue:
        Xvec = X.flatten().reshape((-1,1))

        etaX= (etaj.T.dot(Xvec)).item()
        etaT_Sigma_eta = (etaj.T.dot(Sigma.dot(etaj))).item()
        p_value = util.compute_p_value(interval_oc, etaX, etaT_Sigma_eta)
        return p_value
    else:
        return interval_oc

def parametric(model, a, b, X, etaj, Sigma, M, threshold, n_steps, zmin = -20, zmax = 20):
    p = X.shape[1]//2
    L1 = np.hstack((np.eye(p), np.zeros((p,p)))) 
    L2 = np.hstack((np.zeros((p,p)), np.eye(p)))
    
    TD = []
    detectedinter = []
    z =  zmin
    zmax = zmax
    countitv=0

    while z < zmax:
        z += 0.0001

        for i in range(len(detectedinter)):
            if detectedinter[i][0] <= z <= detectedinter[i][1]:
                z = detectedinter[i][1] + 0.0001
                detectedinter = detectedinter[i:]
                break
        if z > zmax:
            break
        print(z)
        Xdeltaz = (a + b*z).reshape((1,-1))

        x_deltaz = torch.from_numpy(Xdeltaz.dot(L1.T)).float()
        xbaseline_deltaz = torch.from_numpy(Xdeltaz.dot(L2.T)).float()

        target = model(x_deltaz).argmax(dim=1).item()

        Minloop = FS_basedIGs(model, x_deltaz, xbaseline_deltaz, target, n_steps=30, percentile=80)

        intervalinloop = overconditioning(model, a, b, Xdeltaz, etaj, Sigma, Minloop, target, threshold, n_steps)
        
        countitv += 1
        # print(f"intervalinloop: {intervalinloop}")
        detectedinter = util.interval_union(detectedinter, intervalinloop)

        if sorted(M) != sorted(Minloop):
            continue

        TD = util.interval_union(TD, intervalinloop)

    etaT_Sigma_eta = (etaj.T.dot(Sigma.dot(etaj))).item()
    etaX= (etaj.T.dot(X.reshape(-1,1))).item()
    
    p_value = util.compute_p_value(TD, etaX, etaT_Sigma_eta)
    return p_value

def main2(model, n, p):
    x = np.random.randn(n, p)
    xbaseline = np.random.randn(n, p)

    Sigma_x = np.eye(n*p)
    Sigma_xbaseline = np.eye(n*p)
    Sigma = block_diag(Sigma_x, Sigma_xbaseline)

    X = np.vstack((x,xbaseline))

    x_tensor = torch.from_numpy(x).float()
    xbaseline_tensor = torch.from_numpy(xbaseline).float()

    target = model(x_tensor.to(device)).argmax(dim=1)

    M = FS_basedIGs(model, x_tensor, xbaseline_tensor, target, n_steps=30, percentile=80)

    Xvec = X.flatten().reshape((-1,1))

    # Test statistic
    j = np.random.choice(M)
    ej = np.zeros((p,1))
    ej[j][0] = 1

    In_kron_ej = np.kron(np.eye(n), ej.T)
    etaj = np.dot((np.hstack((In_kron_ej, -1*In_kron_ej))).T, np.ones((n, 1)) / n )

    zobs = (etaj.T.dot(Xvec)).item()

    etaT_Sigma_eta = (etaj.T.dot(Sigma.dot(etaj))).item()
    b = (Sigma.dot(etaj)) / etaT_Sigma_eta
    a = (np.eye(2*n*p) - b.dot(etaj.T)).dot(Xvec)

    p_value = overconditioning(model, a, b, X, etaj, Sigma, M, target, threshold=80, n_steps=30, return_pvalue=True, zobs=zobs)
    return p_value

from functools import partial
def compute_pvalue(model, p, _=None):
    return main2(model,50, p)


# if __name__ == "__main__":
#     model = model_train.gendata_trainmodel(train=False, device=device)["model"]
#     p = 10
#     number_of_ins =50
#     iteration = 500
#     list_p_value = []

#     # import time
#     # st=time.time()
#     # print(main2(model, number_of_ins, p))
#     # print(f"Take {time.time() - st}s")
#     num_cores = multiprocessing.cpu_count() // 2

#     compute_pvalue_with_args = partial(compute_pvalue, model, p)
#     with multiprocessing.Pool(processes=num_cores) as pool:
#         list_p_value = pool.map(compute_pvalue_with_args, range(iteration))

#     with open("multidatapoint_p_values.txt", "a") as f:
#         for p_value in list_p_value:
#             f.write(f"{p_value}\n")

#     plt.hist(list_p_value)
#     plt.title("Histogram of p-values")
#     plt.xlabel("p-value")
#     plt.ylabel("Density")
#     plt.show()
#     print(kstest(list_p_value, 'uniform'))


# ----- load file to check uniform
with open("multidatapoint_p_values.txt", "r") as f:
    list_p_value = [float(line.strip()) for line in f]
plt.hist(list_p_value)
plt.title("Histogram of p-values")
plt.xlabel("p-value")
plt.ylabel("Density")
plt.show()
plt.savefig("Multi_obs_p_values_histogram.png")
print(kstest(list_p_value, 'uniform'))