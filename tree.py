import numpy as np
from itertools import count
from functools import partial
from collections import defaultdict

from util import stats, to_discrete


class TreeNode:
    """
    The data structure that holds the actual tree structure for the decision tree.
    The tree is just a bunch of these nodes connected with some attributes.
    A parent is connect to 2 children via `.left` and `.right`. Children also
    have a back-reference at `.parent`. The final prediction is made based on the
    `.probability` on the node reached which has no more children.

    Attributes:
        id: a unique (per tree) identifying integer
        n: the number of examples still in consideration
        gini_impurity: the gini impurity of the examples at this node
        probability: a float [0, 1] which is the probability of the positive class
        split_feat: index of the feature this node splits on
        split_val: the value to split on. values less-then-or-equal-to the value go left, greater-than goes right
        left: the left child TreeNode. `None` for a leaf node
        right: the right child TreeNode
        parent: the parent TreeNode. `None` for the root
        ranges: Dict[int -> (int, int)] A dict that maps a feature index to its range of potential values.
            Every time a node is split, its children have a reduced potential range of values for that feature
            which is being split on. Keep track of this to narrow down candidate splits in the future.
    """

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
        self.ranges = None

    def is_leaf(self):
        """returns True if this node has no children"""
        return self.left is None

    def get_assigned_node_id(self, feat_vector):
        """takes a data point and returns the ID of the node it's currently assigned to"""
        if self.is_leaf():
            return self.id

        # if the split_val is int, this is a (binned) numerical feature, which is ordinal
        if isinstance(self.split_val, int):
            if feat_vector[self.split_feat] > self.split_val:
                return self.right.get_assigned_node_id(feat_vector)
            return self.left.get_assigned_node_id(feat_vector)
        # otherwise, its categorical feature, and its non-ordinal
        elif isinstance(self.split_val, set):
            if feat_vector[self.split_feat] not in self.split_val:
                return self.right.get_assigned_node_id(feat_vector)
            return self.left.get_assigned_node_id(feat_vector)
        raise ValueError("invalid split_val: {}".format(self.split_val))

    def split(self, feat, val, left, right):
        """only called on a leaf node. splits the node and adds 2 child nodes"""
        if not self.is_leaf():
            raise ValueError("tree node {} is already split".format(self.id))
        self.split_feat = feat
        self.split_val = val
        self.left = left
        self.right = right
        left.parent = self
        right.parent = self

    def get_probability(self, feat_vector):
        """takes a data point and returns its probability of being the positive class"""
        if self.is_leaf():
            return self.probability

        # if the split_val is int, this is a (binned) numerical feature, which is ordinal
        if isinstance(self.split_val, int):
            if feat_vector[self.split_feat] > self.split_val:
                return self.right.get_probability(feat_vector)
            return self.left.get_probability(feat_vector)
        # otherwise, its categorical feature, and its non-ordinal
        elif isinstance(self.split_val, set):
            if feat_vector[self.split_feat] not in self.split_val:
                return self.right.get_probability(feat_vector)
            return self.left.get_probability(feat_vector)
        raise ValueError("invalid split_val: {}".format(self.split_val))

    def feat_range(self, feat_idx):
        """
        takes a feature index and returns (int, int) which represents the range of potential
        values that this feature can take. in order to save memory, the full ranges of all
        features isn't stored at every node. instead, each child stores the range for the feature
        that its parent was split on. only the root stores ranges for every single feature.
        this makes sense because the ranges for a child differ from its direct parent only by
        a single feature. so, in order to do a lookup, you need to recursively check the parent
        until the root is reached.
        """
        if feat_idx in self.ranges:
            return self.ranges[feat_idx]
        if self.parent is None:
            raise ValueError("unable to find a range for feature: {}".format(feat_idx))
        return self.parent.feat_range(feat_idx)

    def find_node(self, id):
        """get a node by its ID. really only useful for debugging"""
        if self.id == id:
            return self
        if not self.is_leaf():
            return self.left.find_node(id) or self.right.find_node(id)

    def __str__(self):
        return self.to_str()

    def to_str(self, child_prefix="", desc_prefix=""):
        """prints a nice human-readable string"""
        if self.is_leaf():
            this_node_str = child_prefix + "Leaf(id={id}, n={n}, gini_impurity={gini_impurity}, probability={probability})".format(**self.__dict__)
            return this_node_str
        else:
            this_node_str = child_prefix + "TreeNode(id={id}, n={n}, gini_impurity={gini_impurity}, probability={probability}, split_feat={split_feat}, split_val={split_val})".format(**self.__dict__)
            return "\n".join((this_node_str, self.left.to_str(desc_prefix + "├──", desc_prefix + "│  "), self.right.to_str(desc_prefix + "└──", desc_prefix + "   ")))


class DecisionTreeBinaryClassifier:
    """
    Attributes:
        num_features: the number of features
        categorical_features_info: a dict that specifies which features are categorical (currectly unused).
            maps a feature index to the "arity" or the number of possible values. simply pass an empty dict
            if all features are numerical
        max_bins: the numerical features are quantized (binned). This is the maximum number of bins to use
        max_depth: the maximum number of levels to have on the tree
        min_per_node: splits will be pruned if it would create a node with `n` fewer than this
        feature_subset_strategy: "all" or "sqrt". The number of features to consider for splits at each tree node.
            this is useful if this tree will be part of a random forest
    """
    def __init__(self, num_features, categorical_features_info, max_bins=32, max_depth=10, min_per_node=1, feature_subset_strategy="all"):
        self.num_features = num_features
        self.categorical_features_info = categorical_features_info
        self.max_bins = max_bins
        self.max_depth = max_depth
        self.min_per_node = min_per_node
        self.feature_subset_strategy = feature_subset_strategy

    def train(self, dataRDD):
        """
        trains the model. takes an RDD of tuple (label, feature_vector). The label should be binary,
        and the feature_vector must be a sequence
        """
        # a generator for node IDs
        node_id_counter = count()

        # compute some preliminary statistics
        def seq_op(counts, row):
            label = row[0]
            counts[int(label)] += 1
            return counts
        comb_op = lambda counts1, counts2: [counts1[0] + counts2[0], counts1[1] + counts2[1]]
        label_counts = dataRDD.aggregate([0, 0], seq_op, comb_op)
        n, probability, gini = stats(label_counts)

        # sample the training data to find where the bins should be to quantize the numerical features
        fraction = sample_fraction_for_accurate_splits(n, self.max_bins)
        sample = dataRDD.sample(withReplacement=False, fraction=fraction)
        continuous_bins = sample.flatMap(partial(spread_row, self.categorical_features_info)) \
            .groupByKey() \
            .mapValues(partial(to_bins, self.max_bins)) \
            .collectAsMap()
        self.continuous_bins = continuous_bins
        treeRDD = dataRDD.map(lambda pair: (pair[0], discretize(continuous_bins, pair[1]))).persist()

        # initialize the decision tree
        tree_root = TreeNode(next(node_id_counter), n, gini, probability)
        # give every feature the full range
        tree_root.ranges = { i: set(range(self.categorical_features_info[i])) if i in self.categorical_features_info else (0, self.max_bins) for i in range(self.num_features) }
        self.tree_root = tree_root
        # grow the tree level-by-level. this holds the nodes to grow on the nexxt iteration
        frontier = [tree_root]
        depth = 0

        while len(frontier) > 0:
            category_stats = {}
            if len(self.categorical_features_info) > 0:
                category_stats = treeRDD.flatMap(partial(spread_categories, tree_root, self.categorical_features_info)) \
                    .reduceByKey(np.add) \
                    .mapValues(counts_to_prob) \
                    .map(shift_key(1)) \
                    .groupByKey() \
                    .mapValues(dict) \
                    .collect()
                category_stats = to_nested_dict(category_stats)

            # have the master determine which splits should be considered
            candidate_splits = {}
            for node in frontier:
                candidate_splits[node.id] = gen_candidate_splits(node, self.num_features, self.feature_subset_strategy, category_stats)
            # collect statistics about each of the proposed splits.
            # for each training example, take all the candidate splits and emit a key-value pair indicating
            # how that example would get classified. the key-value pair looks like this
            # (node-ID, feature-index, split-value): [[1, 0], [0, 0]]
            # then those classifications are reduced to add them up. so the values become [[negative-examples in left split, positive-examples in left split], [neg-ex in right split, pos-ex in right split]]
            # finally, the label counts are mapped to statistics in the form:
            # (node-ID, feature-index, split-value): (weighted_gini, (total_n_left, probability_positive_left, gini_left), (total_n_right, probability_positive_right, gini_right)
            statsRDD = treeRDD.flatMap(partial(split_statistics, tree_root, candidate_splits)) \
                .reduceByKey(result_adder) \
                .mapValues(to_gini)

            # find the best split for each node in the frontier (lowest gini impurity)
            # the key is converted from (node-ID, feature-index, split-value) to just node-ID
            # this allows us to reduce and get for each node ID only the split with the lowest gini impurity
            best_splits = statsRDD.map(shift_key_3_to_1) \
                .reduceByKey(get_purest_split) \
                .collectAsMap()

            # start collecting the new level of children for the next iteration
            new_frontier = []
            # for each node, do the best split we found
            for node in frontier:
                # bail on this iteration of no splits were found
                if node.id not in best_splits:
                    continue
                split_data = best_splits[node.id]
                (split_feat, split_val), _, (left_n, left_proba, left_gini), (right_n, right_proba, right_gini) = split_data
                # bail if there aren't enough data points to make the split
                if left_n < self.min_per_node or right_n < self.min_per_node:
                    continue

                # create the child nodes, and give them the correct ranges
                parent_range = node.feat_range(split_feat)
                left = TreeNode(next(node_id_counter), left_n, left_gini, left_proba)
                right = TreeNode(next(node_id_counter), right_n, right_gini, right_proba)
                if split_feat not in self.categorical_features_info:
                    left.ranges = { split_feat: (parent_range[0], split_val + 1) }
                    right.ranges = { split_feat: (split_val + 1, parent_range[1]) }
                    node.split(split_feat, split_val, left, right)
                else:
                    parent_range_list = sorted(parent_range, key=lambda el: category_stats[node.id][split_feat].get(el, 0))
                    split_val_set = set(parent_range_list[:split_val + 1])
                    left.ranges = { split_feat: split_val_set }
                    right.ranges = { split_feat: parent_range - split_val_set }
                    node.split(split_feat, split_val_set, left, right)
                # only add these to the frontier if they aren't homogenous
                if 0 < left.probability and left.probability < 1:
                    new_frontier.append(left)
                if 0 < right.probability and right.probability < 1:
                    new_frontier.append(right)

            depth += 1
            # bail if the depth exceeds the max
            if depth >= self.max_depth:
                frontier = []
            else:
                frontier = new_frontier

        # remove from cache before returning
        treeRDD.unpersist()

    def predict(self, dataRDD):
        """take a data point and get the probability of being in the positive class. only do this after training"""
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

def spread_row(categorical_features_info, row):
    """for use by flatMap to make a separate row for each feature, instead of one row per training example"""
    return [ (i, val) for i, val in enumerate(row[1]) if val != 0 and i not in categorical_features_info ]

def to_bins(max_bins, sample):
    """given a sample of values for a feature and calculates where the bins should be"""
    return np.quantile(list(sample), np.linspace(0, 1, max_bins + 1)[1:-1])

def discretize(bins, feat_vector):
    """maps a feature vector to the binned (discretized) version"""
    binned_feat_vector = []
    for i, val in enumerate(feat_vector):
        if i in bins:
            binned_feat_vector.append(to_discrete(val, bins[i]))
        else:
            binned_feat_vector.append(val)
    return binned_feat_vector

def spread_categories(tree_root, categorical_features_info, pair):
    label, feat_vector = pair
    node_assignment = tree_root.get_assigned_node_id(feat_vector)
    for i in categorical_features_info:
        yield (node_assignment, i, feat_vector[i]), [int(label == 0), int(label == 1)]

def counts_to_prob(counts):
    return (counts[1] / sum(counts),)

def shift_key(k):
    def wrapper(pair):
        tuple_key, tuple_data = pair
        return (tuple_key[:-k], tuple_key[-k:] + tuple_data)
    return wrapper

def to_nested_dict(tuple_key_pairs):
    nested_dict = defaultdict(lambda: defaultdict(dict))
    for pair in tuple_key_pairs:
        key, val = pair
        nested_dict[key[0]][key[1]] = val
    return dict(nested_dict)

def gen_candidate_splits(node, num_feats, feature_subset_strategy, category_stats):
    """generates the splits to consider. returns a dict that maps the feature index to feature values"""
    candidates = {}
    if feature_subset_strategy == "sqrt":
        loop_range = np.random.choice(num_feats, int(np.ceil(np.sqrt(num_feats))))
    elif feature_subset_strategy == "all":
        loop_range = range(num_feats)
    else:
        raise ValueError("invalid feature_subset_strategy: {}".format(feature_subset_strategy))
    for i in loop_range:
        # look up the range of potential values for this particular feature at this node. it might not be
        # the full range of values if we split on this feature already in a previous level.
        feat_range = node.feat_range(i)

        # if the range is a tuple (of length 2), its a numerical feature, and the tuple values are the start and end of the range
        if isinstance(feat_range, tuple) and feat_range[1] - feat_range[0] > 1:
            candidates[i] = feat_range
        # otherwise its a categorical feature
        elif isinstance(feat_range, set):
            candidates[i] = sorted(feat_range, key=lambda el: category_stats[node.id][i].get(el, 0))
    return candidates

def split_statistics(tree_root, candidate_splits, pair):
    """
    see which node a data point is assigned to, and see how the class assignments would be
    at each split we are considering. encodes the assignment as a nested list of length 2 each
    [[1, 0], [0, 0]] all zeros except for a 1. this means the example is negative and it was sent
    to the left split.
    """
    label, feat_vector = pair
    node_assignment = tree_root.get_assigned_node_id(feat_vector)
    # bail if we aren't trying to split the node this data point falls in.
    # happens if a node is homogenous or has too few examples
    if node_assignment not in candidate_splits:
        return []
    candidates_to_evaluate = candidate_splits[node_assignment]

    category_split = set()
    for feat_idx, split_vals in candidates_to_evaluate.items():
        # if the range is a tuple (of length 2), its a numerical feature, and the tuple values are the start and end of the range
        if isinstance(split_vals, tuple):
            for split_val in range(*split_vals):
                results = [[0, 0], [0, 0]]
                to_right = feat_vector[feat_idx] > split_val
                results[int(to_right)][int(label)] = 1
                yield (node_assignment, feat_idx, split_val), results
        # otherwise, its a list of category values that go into the left split
        elif isinstance(split_vals, list):
            for i, split_val in enumerate(split_vals):
                category_split.add(split_val)
                results = [[0, 0], [0, 0]]
                to_right = feat_vector[feat_idx] not in category_split
                results[int(to_right)][int(label)] = 1
                yield (node_assignment, feat_idx, i), results
        else:
            raise ValueError("invalid split_vals: {}".format(split_vals))

def result_adder(results1, results2):
    # just add up all the values per-split per-class
    return [[results1[0][0] + results2[0][0], results1[0][1] + results2[0][1]],
            [results1[1][0] + results2[1][0], results1[1][1] + results2[1][1]]]

def to_gini(results):
    """
    calculate the stats (including gini impurities for each split), along with the weighted average of gini impurities
    """
    results1, results2 = results
    stats1 = stats(results1)
    stats2 = stats(results2)
    if stats1[2] is np.inf or stats2[2] is np.inf:
        return np.inf, stats1, stats2
    total = stats1[0] + stats2[0]
    return stats1[0] / total * stats1[2] + stats2[0] / total * stats2[2], stats1, stats2

def shift_key_3_to_1(pair):
    """
    take a key-value pair where the key is a 3-tuple and convert to a single key.
    the other 2 former-keys get merged into the value
    """
    tuple_key, tuple_data = pair
    return (tuple_key[0], (tuple_key[1:],) + tuple_data)

def get_purest_split(data1, data2):
    """a reducer that just picks the smaller weighted gini impurity"""
    if data1[1] < data2[1]:
        return data1
    return data2

