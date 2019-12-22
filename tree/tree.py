import numpy as np


def entropy(split_feature):
    _, counts = np.unique(split_feature, return_counts=True)
    prob = counts / split_feature.shape[0]
    return -np.sum(prob * np.log2(prob))


def gini(split_feature):
    _, counts = np.unique(split_feature, return_counts=True)
    prob = counts / split_feature.shape[0]
    return 1 - np.sum(prob ** 2)


def var(split_feature):
    return np.var(split_feature)


split_criterions = {'entropy': entropy, 'gini': gini, 'var': var}


class DecisionTree:

    def __init__(self, criterion, depth=0, min_samples_split=2, max_depth=5):

        self.criterion = criterion
        self.criterion_fn = split_criterions[criterion]
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.score = np.inf

    def fit(self, X, y):

        n_samples, n_features = X.shape

        if self.depth == self.max_depth or n_samples < self.min_samples_split:
            # Most common label.
            self.prediction = np.mean(y)
            self.is_leaf = True
            return self

        self.find_best_split(X, y)

        mask = X[:, self.best_feature_split] < self.best_split_value
        n_left = np.sum(mask)
        n_right = n_samples - n_left

        if n_left < self.min_samples_split or n_right < self.min_samples_split:
            self.is_leaf = True
            self.prediction = np.mean(y)
            return self

        self.is_leaf = False
        self.create_trees(X, y, mask)

    def find_best_split(self, X, y):

        n_samples, n_features = X.shape
        best_feature_split, best_split_score = 0, np.inf

        for feature in range(n_features):
            values = X[:, feature]
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_labels = y[sorted_indices]
            _, uniq_indices = np.unique(sorted_values, return_index=True)

            for idx in uniq_indices:
                left_y, right_y = sorted_labels[:idx], sorted_labels[idx:]
                left_split_val = self.criterion_fn(left_y)
                right_split_val = self.criterion_fn(right_y)
                current_idx_split_val = idx / n_samples * left_split_val + \
                    (1 - idx / n_samples) * right_split_val

                if current_idx_split_val < best_split_score:
                    best_feature_split = feature
                    best_split_value = sorted_values[idx]
                    best_split_score = current_idx_split_val

        self.best_feature_split = best_feature_split
        self.best_split_value = best_split_value
        self.score = best_split_score

    def create_trees(self, X, y, mask):
        left_X, left_y = X[mask, :], y[mask]
        right_X, right_y = X[~mask, :], y[~mask]

        self.left = DecisionTree(self.criterion, depth=self.depth + 1,
                                 min_samples_split=self.min_samples_split,
                                 max_depth=self.max_depth)
        self.right = DecisionTree(self.criterion, depth=self.depth + 1,
                                  min_samples_split=self.min_samples_split,
                                  max_depth=self.max_depth)

        self.left.fit(left_X, left_y)
        self.right.fit(right_X, right_y)

    def predict_sample(self, sample):

        if self.is_leaf:
            return self.prediction
        elif sample[self.best_feature_split] < self.best_split_value:
            return self.left.predict_sample(sample)
        else:
            return self.right.predict_sample(sample)

    def predict(self, X):
        return np.array([self.predict_sample(sample) for sample in X])


class DecisionTreeClassifier:

    def __init__(self, criterion='entropy', min_samples_split=2, max_depth=5):

        self.d_tree = DecisionTree(criterion=criterion,
                                   min_samples_split=min_samples_split,
                                   max_depth=max_depth)

    def fit(self, X, y):
        self.d_tree.fit(X, y)

    def predict(self, X):
        preds = self.d_tree.predict(X)
        return np.round(preds)


class DecisionTreeRegressor:

    def __init__(self, criterion='var', min_samples_split=2, max_depth=5):
        self.d_tree = DecisionTree(criterion=criterion,
                                   min_samples_split=min_samples_split,
                                   max_depth=max_depth)

    def fit(self, X, y):
        self.d_tree.fit(X, y)

    def predict(self, X):
        return self.d_tree.predict(X)
