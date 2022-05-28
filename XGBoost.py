import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class RegressionTree:
    """
    A RegressionTree object that is recursively called within itself to construct a regression tree. Based on
    Tianqi Chen's XGBoost the internal gain used to find the optimal split value uses both the gradient and hessian.
    The only thing not implemented in this version is sparsity aware fitting or the ability to handle NA values with a
    default direction.
    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas DataFrame of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the hessian inside a node is a measure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score)
    .As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosting.
    """

    def __init__(self, x, gradient, hessian, idxs=None, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=5,
                 lambda_=1.5, gamma=1):

        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = np.arange(len(x)) if idxs is None else idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        # 节点中对应的样本数
        self.row_count = len(self.idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols * self.col_count)]
        self.score = float('-inf')
        self.lhs = None
        self.rhs = None
        self.var_idx = None
        self.split = None
        # 找到最优分裂，并得到左右子树
        self.find_var_split()
        # 计算叶子节点权重
        self.leaf_weight = self.optimal_weight_in_each_leaf(self.gradient[self.idxs], self.hessian[self.idxs])

    def optimal_weight_in_each_leaf(self, gradient, hessian):
        """
        Calculates the optimal weight in each leaf
        """
        return -np.sum(gradient) / (np.sum(hessian) + self.lambda_)

    def find_var_split(self):
        """
        Scans through every column and calculates the best split point.
        The node is then split at this point and two new nodes are created.
        Depth is only parameter to change as we have added a new layer to tre structure.
        If no split is better than the score initialized at the beginning then no splits further splits are made
        """
        for c in self.column_subsample:
            self.find_greedy_split(c)
        if self.is_leaf:
            return
        x = self.split_col
        # np.nonzero returns the indices of the elements that are non-zero
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]

        self.lhs = RegressionTree(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[lhs],
                                  subsample_cols=self.subsample_cols, min_leaf=self.min_leaf,
                                  min_child_weight=self.min_child_weight, depth=self.depth - 1, lambda_=self.lambda_,
                                  gamma=self.gamma)
        self.rhs = RegressionTree(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[rhs],
                                  subsample_cols=self.subsample_cols, min_leaf=self.min_leaf,
                                  min_child_weight=self.min_child_weight, depth=self.depth - 1, lambda_=self.lambda_,
                                  gamma=self.gamma)

    def find_greedy_split(self, var_idx):
        """
         For a given feature greedily calculates the gain at each split.
         Globally updates the best score and split point if a better split point is found
        """
        x = self.x[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]

            # 左子树对应的样本indices
            lhs_indices = np.nonzero(lhs)[0]
            # 右子树对应的样本indices
            rhs_indices = np.nonzero(rhs)[0]
            # 1.左子树样本数小于min_leaf
            # 2.右子树样本数小于min_leaf
            # 3.左子树样本对应的hessian小于min_child_weight
            # 4.右子树样本对应的hessian小于min_child_weight
            if (len(lhs_indices) < self.min_leaf or len(rhs_indices) < self.min_leaf
                    or self.hessian[lhs_indices].sum() < self.min_child_weight
                    or self.hessian[rhs_indices].sum() < self.min_child_weight):
                continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                # 最优分裂对应的变量
                self.var_idx = var_idx
                # 最优分裂对应的gain
                self.score = curr_score
                # 最优分裂对应的阈值
                self.split = x[r]

    def gain(self, lhs, rhs):
        """
        Calculates the gain at a particular split point
        """
        gradient = self.gradient[self.idxs]
        hessian = self.hessian[self.idxs]

        lhs_gradient = gradient[lhs].sum()
        lhs_hessian = hessian[lhs].sum()

        rhs_gradient = gradient[rhs].sum()
        rhs_hessian = hessian[rhs].sum()

        lhs_score = lhs_gradient ** 2 / (lhs_hessian + self.lambda_)
        rhs_score = rhs_gradient ** 2 / (rhs_hessian + self.lambda_)
        parent_score = (lhs_gradient + rhs_gradient) ** 2 / (lhs_hessian + rhs_hessian + self.lambda_)

        gain = 0.5 * (lhs_score + rhs_score - parent_score) - self.gamma
        return gain

    @property
    def split_col(self):
        """
        splits a column
        """
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        """
        checks if node is a leaf
        """
        # 如果没找到分裂点或者树的深度已经达到最深则说明是叶子节点，无法再分裂
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.leaf_weight

        regression_tree = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return regression_tree.predict_row(xi)


class XGBoostClassifier:
    """
    XGBoost分类
    """

    def __init__(self, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5, learning_rate=0.4,
                 boosting_rounds=5, lambda_=1.5, gamma=1):
        self.estimators = []
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma
        self.y = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # first order gradient logLoss
    def gradient(self, preds, labels):
        preds = self.sigmoid(preds)
        return preds - labels

    # second order gradient logLoss
    def hessian(self, preds):
        preds = self.sigmoid(preds)
        return preds * (1 - preds)

    @staticmethod
    def log_odds(y):
        binary_yes = np.count_nonzero(y == 1)
        binary_no = np.count_nonzero(y == 0)
        return np.log(binary_yes / binary_no)

    def fit(self, X, y):
        self.y = y
        preds = np.full((X.shape[0],), self.log_odds(self.y), dtype='float64')
        for booster in range(self.boosting_rounds):
            gradient = self.gradient(preds, y)
            hessian = self.hessian(preds)
            boosting_tree = RegressionTree(X, gradient, hessian, subsample_cols=self.subsample_cols,
                                           min_leaf=self.min_leaf, min_child_weight=self.min_child_weight,
                                           depth=self.depth, lambda_=self.lambda_, gamma=self.gamma)
            preds += self.learning_rate * boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict_proba(self, X):
        preds = np.full((X.shape[0],), self.log_odds(self.y), dtype='float64')

        for estimator in self.estimators:
            preds += self.learning_rate * estimator.predict(X)

        return self.sigmoid(np.full((X.shape[0],), 1, dtype='float64') + preds)

    def predict(self, X, threshold=0.5):
        predict_proba = self.predict_proba(X)
        preds = np.where(predict_proba > threshold, 1, 0)
        return preds


class XGBoostRegressor:
    """
    XGBoost回归
    """

    def __init__(self, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5, learning_rate=0.4,
                 boosting_rounds=5, lambda_=1.5, gamma=1):
        self.estimators = []

        self.y = None
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma

    # first order gradient mean squared error
    @staticmethod
    def gradient(preds, labels):
        return 2 * (preds - labels)

    # second order gradient mean squared error
    @staticmethod
    def hessian(preds):
        return np.full((preds.shape[0],), 2.0)

    def fit(self, X, y):
        self.y = y
        preds = np.full((X.shape[0],), self.y.mean())

        for booster in range(self.boosting_rounds):
            gradient = self.gradient(preds, self.y)
            hessian = self.hessian(preds)
            boosting_tree = RegressionTree(X, gradient, hessian, subsample_cols=self.subsample_cols,
                                           min_leaf=self.min_leaf, min_child_weight=self.min_child_weight,
                                           depth=self.depth, lambda_=self.lambda_, gamma=self.gamma)
            preds += self.learning_rate * boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict(self, X):
        preds = np.full((X.shape[0],), self.y.mean())

        for estimator in self.estimators:
            preds += self.learning_rate * estimator.predict(X)

        return preds


if __name__ == '__main__':
    np.random.seed(0)
    data_X, data_y = make_classification(
        n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.3, random_state=42
    )

    model = XGBoostClassifier()
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_test_pred)
    print('test auc', auc)
