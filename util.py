import numpy as np
from mpmath import mp
mp.dps = 500

def compute_p_value(intervals, test_stat, etaT_Sigma_eta):
    denominator = 0
    numerator = 0

    for i in intervals:
        leftside, rightside = i
        if leftside <= test_stat <= rightside:
            numerator = denominator + mp.ncdf(test_stat / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
        denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)
    return 2 * min(cdf, 1 - cdf)

def interval_intersection(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]

        # Calculate the potential intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        # If the interval is valid, add to results
        if start < end:
            result.append((start, end))

        # Move the pointer which ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result


def interval_union(a, b):
    # Merge the two sorted interval lists into one sorted list
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][0]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    # Add any remaining intervals from a or b
    merged.extend(a[i:])
    merged.extend(b[j:])

    # Merge overlapping intervals
    if not merged:
        return []

    result = [merged[0]]
    for current in merged[1:]:
        last = result[-1]
        if current[0] < last[1]:
            # Overlapping or adjacent, merge them
            new_start = last[0]
            new_end = max(last[1], current[1])
            result[-1] = (new_start, new_end)
        else:
            result.append(current)
    return result

def solve_quadratic_inequality(a, b, c):
    """ ax^2 + bx +c <= 0 """
    a, b, c = float(a), float(b), float(c)
    if abs(a) < 1e-10:
        a = 0
    if abs(b) < 1e-10:
        b = 0
    if abs(c) < 1e-10:
        c = 0
    if a == 0:
        if b > 0:
            return [(-np.inf, np.around(-c / b, 8))]
        elif b == 0:
            if c <= 0:
                return [(-np.inf, np.inf)]
            else:
                print('Error bx + c')
                return 
        else:
            return [(np.around(-c / b, 8), np.inf)]

    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return [(-np.inf, np.inf)]
        else:
            print("Error to find interval. ")

    x1 = (- b - np.sqrt(delta)) / (2.0*a)
    x2 = (- b + np.sqrt(delta)) / (2.0*a)

    x1 = np.around(x1, 8)
    x2 = np.around(x2, 8)
    if a < 0:
        return [(-np.inf, x2),(x1, np.inf)]
    return [(x1,x2)]