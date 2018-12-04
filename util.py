import numpy as np

def weighted_calc(groups, func):
    totals = np.array([np.sum(group) for group in groups])
    weights = totals / np.sum(totals)
    values = [func(group) for group in groups]
    return np.sum(weights * values)

def gini_impurity(group):
    n = np.sum(group)
    if n == 0:
        return np.inf
    return 1 - np.sum((np.array(group) / n) ** 2)

def stats(label_count):
    n = np.sum(label_count)
    probability = label_count[1] / n
    gini = gini_impurity(label_count)
    return n, probability, gini

def to_discrete(continuous_val, bins):
    for i in range(0, len(bins)):
        if continuous_val < bins[i]:
            return i
    return len(bins)

