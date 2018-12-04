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
        self.parent = None

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
        left.parent = self
        right.parent = self

    def get_probability(self, feat_vector):
        if self.is_leaf():
            return self.probability
        if feat_vector[self.split_feat] > self.split_val:
            return self.right.get_probability(feat_vector)
        return self.left.get_probability(feat_vector)
    
    def feat_range(self, feat_idx):
        if feat_idx in self.ranges:
            return self.ranges[feat_idx]
        if self.parent is None:
            raise ValueError("unable to find a range for feature: {}".format(feat_idx))
        return self.parent.feat_range(feat_idx)
    
    def find_node(self, id):
        if self.id == id:
            return self
        if not self.is_leaf():
            return self.left.find_node(id) or self.right.find_node(id)

    def __str__(self):
        return self.to_str()

    def to_str(self, child_prefix="", desc_prefix=""):
        if self.is_leaf():
            this_node_str = child_prefix + "Leaf(id={id}, n={n}, gini_impurity={gini_impurity}, probability={probability})".format(**self.__dict__)
            return this_node_str
        else:
            this_node_str = child_prefix + "TreeNode(id={id}, n={n}, gini_impurity={gini_impurity}, probability={probability}, split_feat={split_feat}, split_val={split_val})".format(**self.__dict__)
            return "\n".join((this_node_str, self.left.to_str(desc_prefix + "├──", desc_prefix + "│  "), self.right.to_str(desc_prefix + "└──", desc_prefix + "   ")))


class DecisionTreeBinaryClassifier:
    def __init__(self, num_features, categorical_features_info, max_bins=32, max_depth=10, min_per_node=1):
        self.num_features = num_features
        self.categorical_features_info = categorical_features_info
        self.max_bins = max_bins
        self.max_depth = max_depth
        self.min_per_node = min_per_node
    
    def train(self, dataRDD):
        node_id_counter = count()
        def seq_op(counts, row):
            label = row[0]
            counts[int(label)] += 1
            return counts
        comb_op = lambda counts1, counts2: [counts1[0] + counts2[0], counts1[1] + counts2[1]]
        label_counts = dataRDD.aggregate([0, 0], seq_op, comb_op)
        n, probability, gini = stats(label_counts)

        fraction = sample_fraction_for_accurate_splits(n, self.max_bins)
        sample = dataRDD.sample(withReplacement=False, fraction=fraction)
        continuous_bins = sample.flatMap(spread_row) \
            .groupByKey() \
            .mapValues(partial(to_bins, self.max_bins)) \
            .collectAsMap()
        treeRDD = dataRDD.map(lambda pair: (pair[0], discretize(continuous_bins, pair[1])))
        self.continuous_bins = continuous_bins
        
        tree_root = TreeNode(next(node_id_counter), n, gini, probability)
        tree_root.ranges = { i: (0, self.max_bins) for i in range(self.num_features) }
        self.tree_root = tree_root

        frontier = [tree_root]
        depth = 0
        while len(frontier) > 0:
            print(depth, frontier)
            candidate_splits = {}
            for node in frontier:
                candidate_splits[node.id] = gen_candidate_splits(node, self.num_features)

            statsRDD = treeRDD.flatMap(partial(split_statistics, tree_root, candidate_splits)) \
                .reduceByKey(result_adder) \
                .mapValues(to_gini)
            
            best_splits = statsRDD.map(shift_key) \
                .reduceByKey(get_purest_split) \
                .collectAsMap()
            
            print(best_splits)
            
            new_frontier = []
            for node in frontier:
                if node.id not in best_splits:
                    continue
                split_data = best_splits[node.id]
                (split_feat, split_val), _, (left_n, left_proba, left_gini), (right_n, right_proba, right_gini) = split_data
                if left_n < self.min_per_node or right_n < self.min_per_node:
                    continue
                parent_range = node.feat_range(split_feat)
                left = TreeNode(next(node_id_counter), left_n, left_gini, left_proba)
                left.ranges = { split_feat: (parent_range[0], split_val + 1) }
                right = TreeNode(next(node_id_counter), right_n, right_gini, right_proba)
                right.ranges = { split_feat: (split_val + 1, parent_range[1]) }
                node.split(split_feat, split_val, left, right)
                if 0 < left.probability and left.probability < 1:
                    new_frontier.append(left)
                if 0 < right.probability and right.probability < 1:
                    new_frontier.append(right)
            depth += 1
            if depth >= self.max_depth:
                frontier = []
            else:
                frontier = new_frontier
        print(tree_root)
    
    def predict(self, dataRDD):
        return dataRDD.map(partial(discretize, self.continuous_bins)) \
            .map(self.tree_root.get_probability)


def sample_fraction_for_accurate_splits(total_num_rows, max_bins):
    """
    determines how many rows to sample from the RDD in order to calculate the bins for discretizing continuous values
    uses same heuristic as Spark MLlib
    https://github.com/apache/spark/blob/ebd899b8a865395e6f1137163cb508086696879b/mllib/src/main/scala/org/apache/spark/ml/tree/impl/RandomForest.scala#L1168-L1177
    """
    required_samples = max(max_bins * max_bins, 10_000)
    if required_samples >= total_num_rows:
        return 1
    return required_samples / total_num_rows

def spread_row(row):
    return [ (i, val) for i, val in enumerate(row[1]) if val != 0 ]

def to_bins(max_bins, sample):
    return np.quantile(list(sample), np.linspace(0, 1, max_bins + 1)[1:-1])

def discretize(bins, feat_vector):
    binned_feat_vector = []
    for i, val in enumerate(feat_vector):
        binned_feat_vector.append(to_discrete(val, bins[i]))
    return binned_feat_vector

def to_discrete(continuous_val, bins):
    for i in range(0, len(bins)):
        if continuous_val < bins[i]:
            return i
    return len(bins)

def gen_candidate_splits(node, num_feats):
    candidates = {}
    for i in range(num_feats):
        feat_range = node.feat_range(i)
        if feat_range[1] - feat_range[0] > 1:
            candidates[i] = list(range(*feat_range))
    return candidates

def split_statistics(tree_root, candidate_splits, pair):
    label, feat_vector = pair
    node_assignment = tree_root.get_assigned_node_id(feat_vector)
    if node_assignment not in candidate_splits:
        return []
    candidates_to_evaluate = candidate_splits[node_assignment]

    for feat_idx, split_vals in candidates_to_evaluate.items():
        for split_val in split_vals:
            results = [[0, 0], [0, 0]]
            to_right = feat_vector[feat_idx] > split_val
            results[int(to_right)][int(label)] = 1
            yield (node_assignment, feat_idx, split_val), results

def result_adder(results1, results2):
    return [[results1[0][0] + results2[0][0], results1[0][1] + results2[0][1]],
            [results1[1][0] + results2[1][0], results1[1][1] + results2[1][1]]]

def to_gini(results):
    results1, results2 = results
    stats1 = stats(results1)
    stats2 = stats(results2)
    if stats1[2] is np.inf or stats2[2] is np.inf:
        return np.inf, stats1, stats2
    total = stats1[0] + stats2[0]
    return stats1[0] / total * stats1[2] + stats2[0] / total * stats2[2], stats1, stats2

def shift_key(pair):
    tuple_key, tuple_data = pair
    return (tuple_key[0], (tuple_key[1:],) + tuple_data)

def get_purest_split(data1, data2):
    if data1[1] < data2[1]:
        return data1
    return data2
