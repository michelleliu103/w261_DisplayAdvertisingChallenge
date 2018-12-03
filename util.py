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

class TreeNode:
    def __init__(self, id, n, gini_impurity, probability):
        self.id = id
        self.n = n
        self.gini_impurity = gini_impurity
        self.probability = probability
        self.split_feat = None
        self.split_val = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None
    
    def get_assigned_node_id(self, feat_vector):
        if self.is_leaf():
            return self.id
        if feat_vector[self.split_feat] > self.split_val:
            return self.right.get_assigned_node_id(feat_vector)
        return self.left.get_assigned_node_id(feat_vector)

    def split(self, feat, val, left, right):
        if not self.is_leaf():
            raise ValueError("tree node {} is already split".format(self.id))
        self.split_feat = feat
        self.split_val = val
        self.left = left
        self.right = right

    def get_probability(self, feat_vector):
        if self.is_leaf():
            return self.probability
        if feat_vector[self.split_feat] > self.split_val:
            return self.right.get_probability(feat_vector)
        return self.left.get_probability(feat_vector)

    def __str__(self):
        return self.to_str()

    def to_str(self, child_prefix="", desc_prefix=""):
        if self.is_leaf():
            this_node_str = child_prefix + "Leaf(id={id}, n={n}, gini_impurity={gini_impurity}, probability={probability})".format(**self.__dict__)
            return this_node_str
        else:
            this_node_str = child_prefix + "TreeNode(id={id}, n={n}, gini_impurity={gini_impurity}, probability={probability}, split_feat={split_feat}, split_val={split_val})".format(**self.__dict__)
            return "\n".join((this_node_str, self.left.to_str(desc_prefix + "├──", desc_prefix + "│  "), self.right.to_str(desc_prefix + "└──", desc_prefix + "   ")))
