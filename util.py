import numpy as np

def weighted_calc(groups, func):
    """performs a func on several groups, and gets the weighted average of the results"""
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
    """calculates total count (n), the probability of the positive class, and the gini impurity"""
    n = np.sum(label_count)
    probability = label_count[1] / n
    gini = gini_impurity(label_count)
    return n, probability, gini

def to_discrete(continuous_val, bins):
    """
    given a continuous value and bins, returns the discretized (binned) value.
    this isn't the best implementation. the bins are sorted, so this can be done
    in a binary-search style implementation. but i was too lazy
    """
    for i in range(0, len(bins)):
        if continuous_val < bins[i]:
            return i
    return len(bins)

