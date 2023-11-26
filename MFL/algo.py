import numpy as np

def DAGC_R(ratio_matrix,average_d):
    ratio_matrix = np.sum(ratio_matrix,axis=1)
    ratio_matrix = np.array(ratio_matrix, dtype=float)
    ratio_matrix = ratio_matrix / np.sum(ratio_matrix)

    p = ratio_matrix

    p = np.array(p)
    n = len(p)
    p2 = []
    for i in range(n):
        p2.append(p[i] ** (2 / 3))
    p = np.array(p)
    p2 = np.array(p2)

    pp = np.sum(p2)

    phi_m = 999000
    d = np.zeros(n)

    for i in range(n):
        j = n - (i + 1)
        if (i == 0):
            Q = (pp - p2[j]) / (p2[n - 2])

            phi = (p[j] * (1 + Q)) + p[n - 2] * Q * (1 + Q)
            if (phi < phi_m):
                phi_m = phi
                for k in range(n):
                    if k == j:
                        d[k] = 1 / (Q + 1)
                    else:
                        d[k] = 1 / (Q + 1) * p2[k] / p2[n - 2]

        else:
            if (p[j] == p[j + 1]):
                phi = phi_m
            else:
                Q = (pp - p2[j]) / p2[n - 1]
                phi = (p[j] * (1 + Q)) + p[n - 1] * Q * (1 + Q)
            if (phi < phi_m):
                phi_m = phi
                for k in range(n):
                    if k == j:
                        d[k] = 1 / (Q + 1)
                    else:
                        d[k] = 1 / (Q + 1) * p2[k] / p2[n - 1]

    d = d * n * average_d
    return d


def DAGC_A(ratio_matrix,average_d):
    ratio_matrix = np.sum(ratio_matrix,axis=1)
    ratio_matrix = np.array(ratio_matrix, dtype=float)
    ratio_matrix = ratio_matrix / np.sum(ratio_matrix)

    p = ratio_matrix

    p = np.array(p)
    n = len(p)
    p2 = p ** (2/3)
    pp = np.sum(p2)
    d = pp / p2 / n * average_d
    return d

def get_skew_distribution(q,skew):
    sum_q = np.sum(q, axis=1)
    line_array = np.logspace(np.log10(skew), 0, num=len(sum_q)-1, endpoint=True, dtype=None, axis=0)
    line_array=np.append(line_array,line_array[-1])

    p = []
    for j in range(len(sum_q)):
        q[j] = np.array(q)[j] * line_array[j] / sum_q[j]
        p.append(np.sum(q[j]))
    q = np.array(q, dtype=float)
    return q