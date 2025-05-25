import torch
import numpy as np
import util
# import time
import solveinequalities.interval as bst_solving_eq

def ReLUcondition(model, a, b, X):
    # p = int(X.shape[0] / 2)
    layers = []
    for name, param in model.named_children():
        temp = dict(param._modules)
        
        for layer_name in temp.values():
            if ('Linear' in str(layer_name)):
                layers.append('Linear')
            elif ('ReLU' in str(layer_name)):
                layers.append('ReLU')
    ptr = 0 

    itv = [(-np.inf, np.inf)]
    weight = None
    bias = None
    for name, param in model.named_parameters():
        if (layers[ptr] == 'Linear'):
            if ('weight' in name):
                weight = np.asarray(param.data.cpu())
            elif ('bias' in name):
                bias = np.asarray(param.data.cpu()).reshape(-1, 1)
                bias = bias.dot(np.ones((1, X.shape[0]))).T
                ptr += 1
                X = X.dot(weight.T) + bias
                a = a.dot(weight.T) + bias
                b = b.dot(weight.T)
        # t2 = time.time()
        if (ptr < len(layers) and layers[ptr] == 'ReLU'):
            ptr += 1

            sign_X = np.sign(X)
            at = (a * -1*sign_X).flatten()
            bt = (b * -1*sign_X).flatten()
            itv = bst_solving_eq.interval_intersection(bt,-at)

            sign_X[sign_X < 0] = 0
            X = X*sign_X
            a = a*sign_X
            b = b*sign_X

            # sub_itv = [(-np.inf, np.inf)]

            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         if X[i][j] > 0:
            #             sub_itv = util.interval_intersection(
            #                 sub_itv, 
            #                 util.solve_quadratic_inequality(a=0, b=-b[i][j], c=-a[i][j])
            #                 )
            #         else:
            #             sub_itv = util.interval_intersection(
            #                 sub_itv, 
            #                 util.solve_quadratic_inequality(a=0, b=b[i][j], c = a[i][j])
            #                 )

            #             X[i][j] = 0
            #             a[i][j] = 0
            #             b[i][j] = 0
            
            # itv = util.interval_intersection(itv, sub_itv)
            # print(f"if 2: {time.time() - t2}")

    return itv, a, b

def IGscondition(model, a, b, X, n_steps):
    n,p = X.shape[0]//2, X.shape[1]
    x = X[:n].copy()
    xbaseline = X[n:].copy()
    A = a.reshape(2*n, p)
    a_ = A[:n].copy()
    a_baseline = A[n:].copy()
    
    B = b.reshape(2*n, p)
    b_ = B[:n].copy()
    b_baseline = B[n:].copy()
    
    alphas = torch.linspace(0, 1, n_steps)

    interval = [(-np.inf, np.inf)]
    for alpha in alphas:
        alpha = alpha.item()

        a_alpha =  a_baseline + alpha * (a_ - a_baseline)
        b_alpha =  b_baseline + alpha * (b_ - b_baseline)

        x_interpolated = xbaseline + alpha * (x - xbaseline)
        RELUinterval, a_out, b_out = ReLUcondition(model, a_alpha, b_alpha, x_interpolated)
        interval = util.interval_intersection(
            interval, 
            RELUinterval
        )
    return interval, a_out, b_out
def outputClasscondition(model, a_, b_, X, c,numberofclass=2):
    n, p = X.shape[0]//2, X.shape[1]

    a_output = a_.reshape(n,numberofclass)
    b_output = b_.reshape(n,numberofclass)
    
    C = numberofclass
    e = np.zeros((C,1))
    interval = [(-np.inf, np.inf)]

    for i in range(n):
        ci = c[i].item()
        e[ci][0] = -1
        for other_c in range(C):
            if other_c != ci:
                e[other_c][0] = 1
                interval = util.interval_intersection(
                    interval,
                    util.solve_quadratic_inequality(a=0, b=b_output[i].dot(e), c= a_output[i].dot(e))
                )
                e[other_c][0] = 0
        e[ci][0] = 0
    return interval
# def thresholdcondition(model, a, b, X, target, number_feas, threshold, n_steps):
#     p = int(X.shape[1]/2)
#     L1 = np.hstack((np.eye(p), np.zeros((p,p)))) 
#     L2 = np.hstack((np.zeros((p,p)), np.eye(p))) 

#     alphas = torch.linspace(0, 1, n_steps)
#     model.eval()

#     x_tensor = torch.from_numpy(X.dot(L1.T))
#     xbaseline_tensor = torch.from_numpy(X.dot(L2.T))

#     ig = torch.zeros_like(x_tensor)
#     for alpha in alphas:
#         # Interpolate between baseline and input
#         interpolated = xbaseline_tensor + alpha * (x_tensor - xbaseline_tensor)
#         interpolated = interpolated.to(dtype=torch.float32)
#         interpolated.requires_grad_(True)
#         # Compute model output
#         outputs = model(interpolated)

#         # Compute gradients with respect to the interpolated input
#         target_outputs = outputs[torch.arange(outputs.size(0)), target]
#         gradients = torch.autograd.grad(target_outputs.sum(), interpolated, create_graph=True)[0]
#         ig += gradients / n_steps

#     attributions = ig * (x_tensor - xbaseline_tensor)
#     attributions = attributions.detach().cpu().numpy()

#     ig = ig.detach().cpu().numpy()
#     L12 = L1 - L2
#     a_ = L12.dot(a)
#     b_ = L12.dot(b)

#     a_ = a_*ig.reshape((-1,1))
#     b_ = b_*ig.reshape((-1,1))
#     # attributions = a_ + b_*z
    
#     #sign condition
#     # s1 + s2z > 0
#     interval_1 = [(-np.inf, np.inf)]
    
#     s1 = np.sign(attributions).T * a_
#     s2 = np.sign(attributions).T * b_
#     # print(s1+s2*z)
#     for i in range(p):
#         interval_1 = util.interval_intersection(
#             interval_1,
#             util.solve_quadratic_inequality(a=0, b=-s2[i][0], c=-s1[i][0])
#         )
#     # print(interval_1)


#     #sort condition
#     interval_2 = [(-np.inf, np.inf)]
#     indexsort = np.argsort(np.abs(attributions[0]))[::-1]
#     # print("abs attr",np.abs(attributions))
#     # print(indexsort)
#     # a1 > a2 > a3 > a4
#     e = np.zeros((p,1))
#     for i in range(p-1):
#         e[indexsort[i]][0] = 1
#         for j in range(i+1,p):
#             e[indexsort[j]][0] = -1
#             interval_2 = util.interval_intersection(
#                 interval_2, 
#                 util.solve_quadratic_inequality(a=0, b=-(e.T.dot(s2)).item(), c=-(e.T.dot(s1)).item())
#             )
#             e[indexsort[j]][0] = 0
#         e[indexsort[i]][0] = 0
#     # print(interval_2)


#     #threshold condition
#     ep = threshold/100 * np.ones((p,1))
#     #s[i-1] < threshold*sum (sp) <= s[i]

#     ei_minus1 = np.zeros((p,1))
#     for i in range(number_feas-1):
#         ei_minus1[indexsort[i]][0] = 1
    
#     ei = ei_minus1.copy()
#     ei[indexsort[number_feas-1]][0] = 1

#     interval_3 = [(-np.inf, np.inf)]

#     epi = ep - ei

#     interval_3 = util.interval_intersection(
#         interval_3,
#         util.solve_quadratic_inequality(a=0, b=(epi.T.dot(s2)).item(), c=(epi.T.dot(s1)).item())
#         )
#     ei_minus1p = ei_minus1 - ep

#     interval_3 = util.interval_intersection(
#         interval_3,
#         util.solve_quadratic_inequality(a=0, b=(ei_minus1p.T.dot(s2)).item(), c=(ei_minus1p.T.dot(s1)).item())
#         )

#     interval = util.interval_intersection(
#         interval_1,
#         util.interval_intersection(interval_2, interval_3)
#     )    
#     # print(interval)
#     return interval


def thresholdcondition2(model, a, b, X, ig_value, target, number_feas, threshold, n_steps,z):
    n, p = X.shape[0] //2, X.shape[1]

    x_tensor = torch.from_numpy(X[:n]).float()
    xbaseline_tensor = torch.from_numpy(X[n:2*n]).float()

    ig_value = ig_value / (x_tensor - xbaseline_tensor).detach().cpu().numpy()
    ig = ig_value.reshape((n*p,1))
    # print(ig)
    a_ = (a[:n*p] - a[n*p:2*n*p]).copy()
    b_ = (b[:n*p] - b[n*p:2*n*p]).copy()

    a_ = a_*ig
    b_ = b_*ig

    a_ = np.mean((a_).reshape((n,p)), axis=0)
    b_ = np.mean((b_).reshape((n,p)), axis=0)

    a_ = a_.reshape((p,1))
    b_ = b_.reshape((p,1))

    attributions = a_ + b_*z

    #sign condition
    # s1 + s2z > 0
    interval_1   = [(-np.inf, np.inf)]
    
    s1 = np.sign(attributions) * a_
    s2 = np.sign(attributions) * b_

    for i in range(p):
        interval_1 = util.interval_intersection(
            interval_1,
            util.solve_quadratic_inequality(a=0, b=-s2[i][0], c=-s1[i][0])
        )


    #sort condition
    interval_2 = [(-np.inf, np.inf)]
    indexsort = np.argsort(np.abs(attributions.reshape(-1)))[::-1]
    # print("abs attr",np.abs(attributions))
    # print(indexsort)
    # a1 > a2 > a3 > a4
    e = np.zeros((p,1))
    for i in range(p-1):
        e[indexsort[i]][0] = 1
        for j in range(i+1,p):
            e[indexsort[j]][0] = -1
            interval_2 = util.interval_intersection(
                interval_2, 
                util.solve_quadratic_inequality(a=0, b=-(e.T.dot(s2)).item(), c=-(e.T.dot(s1)).item())
            )
            e[indexsort[j]][0] = 0
        e[indexsort[i]][0] = 0

    #threshold condition
    ep = threshold/100 * np.ones((p,1))
    #s[i-1] < threshold*sum (sp) <= s[i]

    ei_minus1 = np.zeros((p,1))
    for i in range(number_feas-1):
        ei_minus1[indexsort[i]][0] = 1

    ei = ei_minus1.copy()
    ei[indexsort[number_feas-1]][0] = 1

    interval_3 = [(-np.inf, np.inf)]

    epi = ep - ei

    interval_3 = util.interval_intersection(
        interval_3,
        util.solve_quadratic_inequality(a=0, b=(epi.T.dot(s2)).item(), c=(epi.T.dot(s1)).item())
        )
    ei_minus1p = ei_minus1 - ep

    interval_3 = util.interval_intersection(
        interval_3,
        util.solve_quadratic_inequality(a=0, b=(ei_minus1p.T.dot(s2)).item(), c=(ei_minus1p.T.dot(s1)).item())
        )

    interval = util.interval_intersection(
        interval_1,
        util.interval_intersection(interval_2, interval_3)
    )    
    # print(interval)
    return interval