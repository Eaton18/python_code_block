"""
source code of: https://anomaly.io/understand-auto-cross-correlation-normalized-shift/index.html
"""

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.stattools import acf

CROSS_CORRELATION_FLG = False
NORMALIZED_CROSS_CORRELATION_FLG = False
AUTO_CORRELATION = False
CORRELATION_WITH_TIME_SHIFT = False
CLUSTER_CORRELATED_METRICS = True


def cross_correlation(set1, set2):
    return np.sum(set1 * set2)


def norm_cross_correlation(set1, set2):
    return np.sum(set1 * set2) / (np.linalg.norm(set1) * np.linalg.norm(set2))


if CROSS_CORRELATION_FLG:
    # Cross-Correlation
    a = [1, 2, -2, 4, 2, 3, 1, 0]
    b = [2, 3, -2, 3, 2, 4, 1, -1]
    c = [-2, 0, 4, 0, 1, 1, 0, -2]

    pyplot.figure(figsize=(12, 4))
    pyplot.title("Series")
    pyplot.plot(range(len(a)), a, color="#f44e2e", label="a")
    pyplot.plot(range(len(b)), b, color="#27ccc0", label="b")
    pyplot.plot(range(len(c)), c, color="#273ecc", label="c")
    pyplot.ylabel("value")
    pyplot.xlabel("index")
    pyplot.legend(loc="upper right")
    pyplot.show()
    # pyplot.fill_between(range(len(a)), [e + 1 for e in a], [e - 1 for e in a], color="blue", label="fill_a")

    # calculate Cross-Correlation
    a = np.array([1, 2, -2, 4, 2, 3, 1, 0])
    b = np.array([2, 3, -2, 3, 2, 4, 1, -1])
    c = np.array([-2, 0, 4, 0, 1, 1, 0, -2])

    print(f"Cross Correlation a,b: {cross_correlation(a, b)}")
    print(f"Cross Correlation a,c: {cross_correlation(a, c)}")
    print(f"Cross Correlation b,c: {cross_correlation(b, c)}")
    print(f"Cross Correlation a,b: {cross_correlation(a / 2, a)}")

    # do not assign a lambda expression, use a def

if NORMALIZED_CROSS_CORRELATION_FLG:
    a = np.array([1, 2, -2, 4, 2, 3, 1, 0])
    b = np.array([2, 3, -2, 3, 2, 4, 1, -1])
    c = np.array([-2, 0, 4, 0, 1, 1, 0, -2])

    print(f"Normalized Cross Correlation a,b: {norm_cross_correlation(a, b)}")
    print(f"Normalized Cross Correlation a,c: {norm_cross_correlation(a, c)}")
    print(f"Normalized Cross Correlation b,c: {norm_cross_correlation(b, c)}")

if AUTO_CORRELATION:
    np.random.seed(10)
    a = np.tile(np.array(range(8)), 8) + np.random.normal(loc=0.0, scale=0.5, size=64)
    # pyplot.figure(figsize=(12, 4))
    # pyplot.title("Series")
    # pyplot.plot(range(len(a)), a, color="#f44e2e", label="a")
    # pyplot.ylabel("value")
    # pyplot.xlabel("index")
    # pyplot.legend(loc="upper right")
    # pyplot.show()

    ar4 = a[:len(a) - 4]
    ar4_shift = a[4:]
    print(f"Auto Correlation with shift 8: {norm_cross_correlation(ar4, ar4_shift)}")

    pyplot.figure(figsize=(12, 4))
    pyplot.title("Series")
    pyplot.plot(range(len(ar4)), ar4, color="blue", label="ar4")
    pyplot.plot(range(len(ar4_shift)), ar4_shift, color="green", label="ar4_shift")
    pyplot.ylabel("value")
    pyplot.xlabel("index")
    pyplot.legend(loc="upper right")
    pyplot.show()

if CORRELATION_WITH_TIME_SHIFT:
    # Normalized Cross-Correlation with Time Shift
    #
    # a = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    # b = np.array([1, 2, 3, 3, 0, 1, 2, 3, 4, 0, 1, 1, 4, 4, 0, 1, 2, 3, 4, 0])
    # res_ary = ccf(a, b, unbiased=False)
    #
    # print(res_ary)
    #
    # pyplot.bar(range(len(res_ary)), res_ary, fc="blue")
    # pyplot.show()
    #
    # # pyplot.figure(figsize=(12, 4))
    # # pyplot.title("Series")
    # # pyplot.plot(range(len(a)), a, color="blue", label="a")
    # # pyplot.plot(range(len(b)), b, color="green", label="b")
    # # pyplot.ylabel("value")
    # # pyplot.xlabel("index")
    # # pyplot.legend(loc="upper right")
    # # pyplot.show()

    # Normalized Auto-Correlation with Time Shift

    np.random.seed(5)
    a = np.tile(np.array(range(8)), 8) + np.random.normal(loc=0.0, scale=0.5, size=64)
    res_ary = acf(a, nlags=30, fft=False)
    print(res_ary)
    # pyplot.bar(range(len(res_ary)), res_ary, fc="blue")
    # pyplot.show()

    pyplot.figure(figsize=(12, 4))
    pyplot.title("Series")
    pyplot.plot(range(len(a)), a, color="blue", label="a")
    pyplot.ylabel("value")
    pyplot.xlabel("index")
    pyplot.legend(loc="upper right")
    pyplot.show()


def get_correlation_table(metric_df):
    metric_cnt = metric_df.shape[1]
    correlation_table = np.zeros((metric_cnt, metric_cnt))
    for i in range(metric_cnt):
        metric_1 = metric_df.iloc[:, i]
        for j in range(metric_cnt):
            if i == j:
                continue
            else:
                metric_2 = metric_df.iloc[:, j]
                cc_ary = ccf(metric_1, metric_2, unbiased=False)
                correlation_table[i, j] = cc_ary[0]
    return correlation_table


def find_related_metric(correlation_table, orig, high_corr):
    metric_corr_table = correlation_table[orig]
    corr_metric_lst = []
    for i in range(len(metric_corr_table)):
        if metric_corr_table[i] > high_corr:
            corr_metric_lst.append(i)

    return corr_metric_lst


if CLUSTER_CORRELATED_METRICS:
    package_file = os.path.dirname(__file__)
    data_path = os.path.join(package_file, "dataset", "graphs45.csv")
    metric_df = pd.read_csv(data_path)
    correlation_table_ary = get_correlation_table(metric_df)

    orig = 3
    high_corr = 0.9
    corr_metric_lst = find_related_metric(correlation_table_ary, orig, high_corr)
    corr_metric_lst.append(orig)
    print(corr_metric_lst)

    # pyplot.figure(figsize=(18, 6))
    pyplot.figure()
    pyplot.title("Series")
    for idx in corr_metric_lst:
        metric = metric_df.iloc[:, idx]
        pyplot.plot(range(len(metric)), metric, label=f"graph_{idx + 1}")
    pyplot.ylabel("value")
    pyplot.xlabel("index")
    pyplot.legend(loc="upper right")
    pyplot.show()
