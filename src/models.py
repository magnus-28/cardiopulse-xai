"""
Pure NumPy/SciPy ML Models built from scratch for CPU/Edge efficiency
(runs smoothly on Lenovo IdeaPad Slim 5i Ultra / Intel Core Ultra 5 + Intel Arc).

Models:
  1. LogisticRegressionL2   (L-BFGS-B optimization with L2 penalty)
  2. RandomForestScratch    (Bagged Gini CART trees with feature subsampling)
  3. GradientBoostingScratch(Log-loss gradient boosted regression trees)
  4. ClinicalMLP            (2-layer ReLU Neural Network with Adam + L2)
  5. CalibratedEnsemble     (Weighted soft-voting + Platt probability calibration)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class LogisticRegressionL2:
    """L2-regularized Logistic Regression optimized via L-BFGS-B."""
    def __init__(self, C: float = 1.0):
        self.C = C
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionL2":
        n, d = X.shape
        reg = 1.0 / (self.C * n)

        def loss_and_grad(params):
            w, b = params[:-1], params[-1]
            z = X @ w + b
            p = expit(z)
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 0.5 * reg * np.sum(w ** 2)
            diff = (p - y) / n
            gw = X.T @ diff + reg * w
            gb = np.sum(diff)
            return loss, np.append(gw, gb)

        res = minimize(loss_and_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B")
        self.w = res.x[:-1]
        self.b = float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return expit(X @ self.w + self.b)


class _DecisionTreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class _FastCART:
    """Lightweight CART tree supporting both classification (Gini) and regression (MSE)."""
    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 8,
                 max_features: int | None = None, mode: str = "clf", seed: int = 0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.root: _DecisionTreeNode | None = None

    def _leaf_val(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        if n < 2 * self.min_samples_leaf:
            return None, None
        feats = np.arange(d)
        if self.max_features is not None and self.max_features < d:
            feats = self.rng.choice(d, size=self.max_features, replace=False)

        best_score = np.inf
        best_feat, best_thr = None, None
        total_sum = np.sum(y)
        total_sq = np.sum(y * y)

        for f in feats:
            col = X[:, f]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            # Evaluate up to 16 quantile thresholds for fast CPU training
            if len(uniq) > 16:
                thrs = np.quantile(col, np.linspace(0.1, 0.9, 15))
                thrs = np.unique(thrs)
            else:
                thrs = 0.5 * (uniq[:-1] + uniq[1:])

            for thr in thrs:
                left_mask = col <= thr
                n_l = int(np.sum(left_mask))
                n_r = n - n_l
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                sum_l = np.sum(y[left_mask])
                sum_r = total_sum - sum_l
                if self.mode == "clf":
                    p_l = sum_l / n_l
                    p_r = sum_r / n_r
                    score = n_l * (1.0 - p_l * p_l - (1 - p_l) ** 2) + n_r * (1.0 - p_r * p_r - (1 - p_r) ** 2)
                else:
                    # Minimizing weighted SSE = total_sq - (sum_l^2/n_l + sum_r^2/n_r)
                    score = total_sq - (sum_l * sum_l / n_l + sum_r * sum_r / n_r)
                if score < best_score:
                    best_score = score
                    best_feat = int(f)
                    best_thr = float(thr)
        return best_feat, best_thr

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _DecisionTreeNode:
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.std(y) < 1e-7:
            return _DecisionTreeNode(value=self._leaf_val(y))
        feat, thr = self._best_split(X, y)
        if feat is None:
            return _DecisionTreeNode(value=self._leaf_val(y))
        left_mask = X[:, feat] <= thr
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _DecisionTreeNode(feature=feat, threshold=thr, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FastCART":
        self.root = self._build(X, y, 0)
        return self

    def _predict_one(self, row: np.ndarray) -> float:
        node = self.root
        while node.value is None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(r) for r in X], dtype=float)


class RandomForestScratch:
    """Bagged CART Random Forest built from scratch."""
    def __init__(self, n_estimators: int = 40, max_depth: int = 5,
                 min_samples_leaf: int = 6, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.trees: list[_FastCART] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestScratch":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        max_feat = max(2, int(np.sqrt(d)))
        self.trees = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = _FastCART(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feat,
                mode="clf",
                seed=self.seed + i + 1,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.mean([t.predict(X) for t in self.trees], axis=0)
        return np.clip(preds, 1e-6, 1.0 - 1e-6)


class GradientBoostingScratch:
    """Binary Log-Loss Gradient Boosted Trees built from scratch."""
    def __init__(self, n_estimators: int = 55, learning_rate: float = 0.08,
                 max_depth: int = 3, min_samples_leaf: int = 10, seed: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.init_logit: float = 0.0
        self.trees: list[_FastCART] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingScratch":
        p0 = np.clip(np.mean(y), 1e-5, 1.0 - 1e-5)
        self.init_logit = float(np.log(p0 / (1.0 - p0)))
        F = np.full(len(y), self.init_logit, dtype=float)
        self.trees = []
        for i in range(self.n_estimators):
            p = expit(F)
            residuals = y - p
            tree = _FastCART(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                mode="reg",
                seed=self.seed + i,
            )
            tree.fit(X, residuals)
            F += self.learning_rate * tree.predict(X) * 2.0
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.init_logit, dtype=float)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X) * 2.0
        return expit(F)


class ClinicalMLP:
    """2-Layer Neural Network (ReLU + Adam + L2 weight decay) built in NumPy."""
    def __init__(self, hidden1: int = 32, hidden2: int = 16, lr: float = 0.012,
                 epochs: int = 240, l2: float = 0.008, seed: int = 42):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.params: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClinicalMLP":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        y_col = y.reshape(-1, 1)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden1)),
            "b1": np.zeros((1, self.hidden1)),
            "W2": rng.normal(0, np.sqrt(2.0 / self.hidden1), (self.hidden1, self.hidden2)),
            "b2": np.zeros((1, self.hidden2)),
            "W3": rng.normal(0, np.sqrt(2.0 / self.hidden2), (self.hidden2, 1)),
            "b3": np.zeros((1, 1)),
        }
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(val) for k, val in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.epochs + 1):
            z1 = X @ self.params["W1"] + self.params["b1"]
            a1 = np.maximum(0.0, z1)
            z2 = a1 @ self.params["W2"] + self.params["b2"]
            a2 = np.maximum(0.0, z2)
            z3 = a2 @ self.params["W3"] + self.params["b3"]
            p = expit(z3)

            dz3 = (p - y_col) / n
            dW3 = a2.T @ dz3 + self.l2 * self.params["W3"]
            db3 = np.sum(dz3, axis=0, keepdims=True)
            da2 = dz3 @ self.params["W3"].T
            dz2 = da2 * (z2 > 0)
            dW2 = a1.T @ dz2 + self.l2 * self.params["W2"]
            db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = dz2 @ self.params["W2"].T
            dz1 = da1 * (z1 > 0)
            dW1 = X.T @ dz1 + self.l2 * self.params["W1"]
            db1 = np.sum(dz1, axis=0, keepdims=True)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
            for k in self.params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                m_hat = m[k] / (1 - beta1 ** t)
                v_hat = v[k] / (1 - beta2 ** t)
                self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        a1 = np.maximum(0.0, X @ self.params["W1"] + self.params["b1"])
        a2 = np.maximum(0.0, a1 @ self.params["W2"] + self.params["b2"])
        return expit(a2 @ self.params["W3"] + self.params["b3"]).ravel()


class CalibratedEnsemble:
    """Weighted soft-voting ensemble with Platt scaling calibration."""
    def __init__(self, models: dict, weights: dict[str, float]):
        self.models = models
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.zeros(len(X), dtype=float)
        for name, model in self.models.items():
            p += self.weights[name] * model.predict_proba(X)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "CalibratedEnsemble":
        raw_p = self._raw_proba(X_cal)
        logits = np.log(raw_p / (1.0 - raw_p))

        def nll(ab):
            a, b = ab
            p = expit(a * logits + b)
            eps = 1e-12
            return -np.mean(y_cal * np.log(p + eps) + (1 - y_cal) * np.log(1 - p + eps))

        res = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead")
        self.platt_a, self.platt_b = float(res.x[0]), float(res.x[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_p = self._raw_proba(X)
        logits = np.log(raw_p / (1.0 - raw_p))
        return expit(self.platt_a * logits + self.platt_b)
