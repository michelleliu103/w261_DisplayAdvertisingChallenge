import numpy as np

def gini_impurity(group):
    n = np.sum(group)
    if n == 0:
        return np.inf
    return 1 - np.sum((np.array(group) / n) ** 2)

def stats(label_count):
    """
    calculates total count (n), the probability of the positive class, and the gini impurity
    the input is a list of 2, the counts for each label
    """
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

