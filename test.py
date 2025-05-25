import numpy as np 
import solveinequalities.interval as itv

def convert(m, n):
    A = np.arange(m * n).reshape(m, n)
    B = np.zeros((m + n, m * n), dtype=int)
    
    # Process rows
    B[np.arange(m)[:, None], A] = 1
    
    # Process columns
    B[m + np.arange(n)[:, None], A.T] = 1
    
    return B

A = np.array([1, -1])
B = np.array([1, -2])

result = itv.interval_intersection(A, B)
print("Resulting interval(s):", result)

# a = [1, -1]
# b = [-3, 2]
# c = [2, 3]

# intervals = itv.solve_system_of_quadratics(a, b, c)
# print(intervals)