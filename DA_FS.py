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
import DA
device = torch.device("cpu")
import random
import time

def FS_basedIGs(model, x_tensor, xbaseline_tensor,target, n_steps=30, percentile = 80, device = torch.device("cpu")):
    custom_attributions = (
        ig.custom_integrated_gradients(
            model, x_tensor, xbaseline_tensor, target=target, n_steps=n_steps, device=device
        )
        .squeeze().cpu().detach().numpy()
    )
    # print("Custom Integrated Gradients Attributions:", np.mean(custom_attributions,axis=0))
    M = ig.get_threshold_attributions(custom_attributions, percentile=percentile)
    return M, custom_attributions

def overconditioning(model, a, b, X, etaj, Sigma, M, ig, target, threshold, n_steps, z, return_pvalue = False):
    n, p = X.shape[0] //2, X.shape[1]
    # t1 = time.time()
    I,a_, b_ = construct_interval.IGscondition(model, a, b, X, n_steps=n_steps)
    # t2 = time.time()

    O = construct_interval.outputClasscondition(model, a_, b_, X, c = target)
    # t3 = time.time()
    
    T = construct_interval.thresholdcondition2(model, a, b, X, ig, target, len(M), threshold = threshold, n_steps=n_steps,z=z)
    # t4 = time.time()
    # print(f"Time I: {t2-t1}, O: {t3-t2}, T: {t4-t3}")
    print(f"I: {I} O: {O} T: {T}")
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
    n, p = X.shape[0] //2, X.shape[1]
    
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
        Xdeltaz = (a + b*z).reshape((2*n,p))

        x_deltaz = torch.from_numpy(Xdeltaz[:n].copy()).float()
        xbaseline_deltaz = torch.from_numpy(Xdeltaz[n:2*n].copy()).float()

        target = model(x_deltaz).argmax(dim=1)

        Minloop, iginloop = FS_basedIGs(model, x_deltaz, xbaseline_deltaz, target, n_steps=n_steps, percentile=threshold)

        intervalinloop = overconditioning(model, a, b, Xdeltaz, etaj, Sigma, Minloop, iginloop, target, threshold, n_steps, z= z)
        
        countitv += 1
        # print(f"Active set: {M}")
        # print(f"intervalinloop: {intervalinloop}")
        detectedinter = util.interval_union(detectedinter, intervalinloop)

        if sorted(M) != sorted(Minloop):
            continue

        TD = util.interval_union(TD, intervalinloop)

    etaT_Sigma_eta = (etaj.T.dot(Sigma.dot(etaj))).item()
    etaX= (etaj.T.dot(X.reshape(-1,1))).item()
    
    p_value = util.compute_p_value(TD, etaX, etaT_Sigma_eta)
    return p_value

def parametric2(model, ns, nt, a, b, X, etaj, Sigma, M, threshold, n_steps, zmin = -20, zmax = 20):
    p = X.shape[1]

    z =  zmin
    zmax = zmax
    countitv=0
    Z = []
    while z < zmax:
        # z += 0.0001
        print(z)
        
        Xdeltaz = (a + b*z).reshape((2*(ns+nt),p))

        xs_deltaz = Xdeltaz[:ns]
        xt_deltaz = Xdeltaz[ns:ns+nt]

        GAMMA, S, q, B = DA.SingleOT(xs_deltaz, xt_deltaz)

        intervalDA = DA.interval_DA(ns, nt ,p, B, S, q, a[:(ns+nt)*p], b[:(ns+nt)*p])
        
        for i in intervalDA:
            lu, ru = i
            if lu <= z <= ru:
                break
        zuv = max(lu,zmin)
        while zuv <= ru:
            if zuv>=zmax:
                break
            Xdeltaz = (a + b*zuv).reshape((2*(ns+nt),p))

            xs_deltaz = Xdeltaz[:ns]
            xt_deltaz = Xdeltaz[ns:ns+nt]
            xbaseline_deltaz = Xdeltaz[ns+nt:]
            Xtrans = GAMMA.dot(np.vstack((xs_deltaz, xt_deltaz)))

            x_tensor = torch.from_numpy(Xtrans).float()
            xbaseline_tensor = torch.from_numpy(xbaseline_deltaz).float()

            target = model(x_tensor.to(device)).argmax(dim=1)
            Xtrans_baseline = np.vstack((Xtrans, xbaseline_deltaz))
            M_z, ig_value = FS_basedIGs(model, x_tensor, xbaseline_tensor, target, n_steps=threshold, percentile=threshold)

            GAMMAvec = np.kron(GAMMA, np.eye(p))
            a_ = np.vstack((GAMMAvec.dot(a[:(ns+nt)*p]),a[(ns+nt)*p:]))
            b_ = np.vstack((GAMMAvec.dot(b[:(ns+nt)*p]),b[(ns+nt)*p:]))

            intervalFS = overconditioning(model, 
                                    a_,
                                    b_, 
                                    Xtrans_baseline, etaj, Sigma, M_z, ig_value, target, threshold=80, n_steps=30, return_pvalue=False, z=zuv)
            
            print(M)
            print(M_z)
            if sorted(M) == sorted(M_z):
                Z = util.interval_union(Z, util.interval_intersection(intervalFS,intervalDA))
            print(intervalFS)
            zuv = intervalFS[0][1] + 0.0001 # ruv
        z = zuv

    etaT_Sigma_eta = (etaj.T.dot(Sigma.dot(etaj))).item()
    etaX= (etaj.T.dot(X.reshape(-1,1))).item()
    
    p_value = util.compute_p_value(Z, etaX, etaT_Sigma_eta)
    return p_value

def main2(model, ns, nt, p):
    seed = random.randint(0, 2**32 - 1) 
    np.random.seed(seed) 
    print(seed)
    xs = np.random.randn(ns, p)+3
    ys = np.random.randint(0, 2, size=ns)
    xt = np.random.randn(nt, p)
    xbaseline = np.random.randn(ns+nt, p)

    X = np.vstack((xs, xt,xbaseline))

    # Sigma_x = np.eye((ns+nt)*p)
    # Sigma_xbaseline = np.eye((ns+nt)*p)
    # Sigma = block_diag(Sigma_x, Sigma_xbaseline)
    Sigma = np.eye((ns+nt)*2*p)
    # Optimal transport
    GAMMA, S, q, B = DA.SingleOT(xs, xt)
    Xtrans = GAMMA.dot(np.vstack((xs, xt)))


    x_tensor = torch.from_numpy(Xtrans).float()
    xbaseline_tensor = torch.from_numpy(xbaseline).float()
    target = model(x_tensor.to(device)).argmax(dim=1)

    Xtrans_baseline = np.vstack((Xtrans, xbaseline))

    M, ig_value = FS_basedIGs(model, x_tensor, xbaseline_tensor, target, n_steps=30, percentile=80)
    Xvec = X.reshape((-1,1))


    # Test statistic
    j = np.random.choice(M)
    ej = np.zeros((p,1))
    ej[j][0] = 1

    In_kron_ej = np.kron(np.eye(nt), ej.T)
    indices = list(range(ns*p, (ns+nt)*p)) + list(range((ns+nt+ns)*p, 2*(ns*p+nt*p)))
    I_cut = np.eye((ns+nt)*2*p)
    I_cut = I_cut[indices]
    etaj = np.dot(((np.hstack((In_kron_ej, -1*In_kron_ej))).dot(I_cut)).T, np.ones((nt, 1)) / nt)

    zobs = (etaj.T.dot(Xvec)).item()

    etaT_Sigma_eta = (etaj.T.dot(Sigma.dot(etaj))).item()
    b = (Sigma.dot(etaj)) / etaT_Sigma_eta
    a = (np.eye(2*(ns+nt)*p) - b.dot(etaj.T)).dot(Xvec)
    
    p_value = parametric2(model, ns, nt, a, b, X, etaj, Sigma, M, threshold=80, n_steps=30, zmin = -20, zmax = 20)

    # st = time.time()
    # intervalDA = DA.interval_DA(ns, nt ,p, B, S, q, a[:(ns+nt)*p], b[:(ns+nt)*p])
    # print("DA:", time.time() - st, intervalDA)
    # p_value = parametric(model, a, b, X, etaj, Sigma, M, threshold=80, n_steps=30, zmin = -20, zmax=20)
    
    # GAMMAvec = np.kron(GAMMA, np.eye(p))
    # a_ = np.vstack((GAMMAvec.dot(a[:(ns+nt)*p]),a[(ns+nt)*p:]))
    # b_ = np.vstack((GAMMAvec.dot(b[:(ns+nt)*p]),b[(ns+nt)*p:]))
    # st = time.time()
    
    # intervalFS = overconditioning(model, 
    #                                 a_,
    #                                 b_, 
    #                                 Xtrans_baseline, etaj, Sigma, M, ig_value, target, threshold=80, n_steps=30, return_pvalue=False, z=zobs)
    # print(f"Take {time.time() - st}s")



    # finalinterval = util.interval_intersection(intervalDA, intervalFS)
    # # print(f"Final interval: {finalinterval}, Length: {(finalinterval[0][1] - finalinterval[0][0])}")
    # etaT_Sigma_eta = (etaj.T.dot(Sigma.dot(etaj))).item()
    # try:
    #     p_value = util.compute_p_value(finalinterval, etaj.T.dot(Xvec).item(), etaT_Sigma_eta)
    # except:
    #     print("Seed", seed)
    #     return 0
    return p_value

from functools import partial
def compute_pvalue(model, ns, nt, p, _=None):
    return main2(model,ns, nt,p)


if __name__ == "__main__":
    model = model_train.gendata_trainmodel(train=False, device=device)["model"]
    p = 10
    number_of_ins =30
    iteration = 500
    list_p_value = []

    for _ in range(1):
        print(main2(model, 200, 10, p))
       

    
    # num_cores = multiprocessing.cpu_count() // 3*2

    # compute_pvalue_with_args = partial(compute_pvalue, model,200,10, p)
    # with multiprocessing.Pool(processes=num_cores) as pool:
    #     list_p_value = pool.map(compute_pvalue_with_args, range(iteration))

    # with open("multidatapoint_p_values.txt", "a") as f:
    #     for p_value in list_p_value:
    #         f.write(f"{p_value}\n")

    # plt.hist(list_p_value)
    # plt.title("Histogram of p-values")
    # plt.xlabel("p-value")
    # plt.ylabel("Density")
    # plt.show()
    # print(kstest(list_p_value, 'uniform'))


# # ----- load file to check uniform
# with open("multidatapoint_p_values.txt", "r") as f:
#     list_p_value = [float(line.strip()) for line in f]
# plt.hist(list_p_value)
# plt.title("Histogram of p-values")
# plt.xlabel("p-value")
# plt.ylabel("Density")
# plt.show()
# plt.savefig("Multi_obs_p_values_histogram.png")
# print(kstest(list_p_value, 'uniform'))