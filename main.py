import math
from collections import Counter
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_metrical_value(val):
    if pd.isnull(val):
        return None
    if isinstance(val, (int, float)):
        return val
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        content = val_str[1:-1].strip()
        parts = [x.strip() for x in content.split(",") if x.strip()]
        try:
            numbers = [float(x) for x in parts]
        except ValueError:
            return None
        return sum(numbers) / len(numbers) if numbers else None
    try:
        return float(val_str)
    except:
        return None

def clean_nominal_value(val):
    if pd.isnull(val):
        return "unknown"
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        content = val_str[1:-1].strip()
        parts = [x.strip() for x in content.split(",") if x.strip()]
        return parts[0] if parts else "unknown"
    return val_str

class Node:
    def __init__(self):
        self.is_leaf = False
        self.prediction = None
        self.split_feature = None
        self.split_value = None
        self.left_child = None
        self.right_child = None
        self.samples_count = None
        self.impurity = None

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_in_leaf=1, criterion="gini", max_candidates=20):
        self.max_depth = max_depth
        self.min_samples_in_leaf = min_samples_in_leaf
        self.criterion = criterion
        self.max_candidates = max_candidates
        self.root = None
        self.feature_types = None
        self.n_total = None
        self.global_classes = None
        self.feature_importances_ = {}

    def fit(self, X, y):
        if not X:
            raise ValueError("Empty dataset provided.")
        self.n_total = len(X)
        self.global_classes = set(y)
        self.feature_types = {feat: ("numeric" if isinstance(val, (int, float)) else "categorical") for feat, val in X[0].items()}
        indices = np.array(range(len(X)))
        self.root = self._build_tree(X, y, indices, depth=0)

    def _build_tree(self, X, y, indices, depth):
        node = Node()
        node.samples_count = len(indices)
        current_labels = [y[i] for i in indices]
        node.impurity = self._impurity(current_labels)
        if len(set(current_labels)) == 1 or len(indices) < self.min_samples_in_leaf or (self.max_depth is not None and depth >= self.max_depth):
            node.is_leaf = True
            node.prediction = Counter(current_labels).most_common(1)[0][0]
            return node
        best = self._find_best_split(X, y, indices)
        if best is None:
            node.is_leaf = True
            node.prediction = Counter(current_labels).most_common(1)[0][0]
            return node
        feature, threshold, gain, left_idx, right_idx = best
        node.split_feature = feature
        node.split_value = threshold
        node.left_child = self._build_tree(X, y, left_idx, depth + 1)
        node.right_child = self._build_tree(X, y, right_idx, depth + 1)
        return node

    def _impurity(self, labels):
        total = len(labels)
        if total == 0:
            return 0
        counts = Counter(labels)
        if self.criterion == "gini":
            impurity = 1.0
            for count in counts.values():
                impurity -= (count / total) ** 2
            return impurity
        elif self.criterion == "entropy":
            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            return entropy
        elif self.criterion == "scaled_entropy":
            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            scale = math.log2(len(self.global_classes)) if len(self.global_classes) > 1 else 1
            return entropy / scale
        else:
            raise ValueError("Unknown criterion: " + self.criterion)

    def _find_best_split(self, X, y, indices):
        best_gain = -1
        best_feature = best_threshold = None
        best_left = best_right = None
        parent_labels = [y[i] for i in indices]
        parent_impurity = self._impurity(parent_labels)
        indices_arr = np.array(indices)
        for feature in X[0].keys():
            ftype = self.feature_types[feature]
            feat_vals = np.array([X[i][feature] for i in indices])
            if ftype == "numeric":
                unique_vals = np.unique(feat_vals)
                if len(unique_vals) > self.max_candidates:
                    quantiles = np.linspace(0, 100, self.max_candidates + 2)[1:-1]
                    thresholds = [np.percentile(feat_vals, q) for q in quantiles]
                else:
                    thresholds = [(unique_vals[i - 1] + unique_vals[i]) / 2 for i in range(1, len(unique_vals))]
            else:
                thresholds = list(set(feat_vals))
            for thresh in thresholds:
                if ftype == "numeric":
                    mask = feat_vals <= thresh
                else:
                    mask = feat_vals == thresh
                left_idx = indices_arr[mask].tolist()
                right_idx = indices_arr[~mask].tolist()
                if not left_idx or not right_idx:
                    continue
                left_impurity = self._impurity([y[i] for i in left_idx])
                right_impurity = self._impurity([y[i] for i in right_idx])
                n = len(indices)
                gain = parent_impurity - ((len(left_idx) / n) * left_impurity + (len(right_idx) / n) * right_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = thresh
                    best_left, best_right = left_idx, right_idx
        if best_gain == -1:
            return None
        return best_feature, best_threshold, best_gain, best_left, best_right

    def _goes_left(self, value, threshold, ftype):
        return (value <= threshold) if ftype == "numeric" else (value == threshold)

    def predict(self, x):
        node = self.root
        while not node.is_leaf:
            feat = node.split_feature
            if self._goes_left(x[feat], node.split_value, self.feature_types[feat]):
                node = node.left_child
            else:
                node = node.right_child
        return node.prediction

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf:
            print(indent + f"Leaf: prediction = {node.prediction} (samples: {node.samples_count})")
        else:
            print(indent + f"Node: if {node.split_feature} <= {node.split_value}? (samples: {node.samples_count})")
            self.print_tree(node.left_child, depth + 1)
            self.print_tree(node.right_child, depth + 1)

    def compute_feature_importances(self):
        self.feature_importances_ = {}
        def recurse(node):
            if node.is_leaf:
                return
            left, right = node.left_child, node.right_child
            n = node.samples_count
            weighted_imp = ((left.samples_count / n) * left.impurity + (right.samples_count / n) * right.impurity)
            decrease = node.impurity - weighted_imp
            contrib = (n / self.n_total) * decrease
            self.feature_importances_[node.split_feature] = self.feature_importances_.get(node.split_feature, 0) + contrib
            recurse(left)
            recurse(right)
        recurse(self.root)

if __name__ == "__main__":
    file_path = "primary_data.csv"
    data = pd.read_csv(file_path, sep=";")
    data = data.drop(columns=["family", "name"], errors="ignore")
    for col in data.columns:
        if col == "class":
            continue
        if col in ["cap-diameter", "stem-height", "stem-width"]:
            data[col] = data[col].apply(clean_metrical_value)
        else:
            data[col] = data[col].apply(clean_nominal_value)
    X = data.drop(columns=["class"]).to_dict(orient="records")
    y = data["class"].tolist()
    n_samples = len(data)
    print(f"Total examples: {n_samples}")
    if n_samples < 1000:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_params = {"max_depth": 5, "min_samples_in_leaf": 5}
        fold_accs = []
        for train_idx, test_idx in kf.split(X):
            train_X = [X[i] for i in train_idx]
            train_y = [y[i] for i in train_idx]
            test_X = [X[i] for i in test_idx]
            test_y = [y[i] for i in test_idx]
            dt = DecisionTree(max_depth=best_params["max_depth"], min_samples_in_leaf=best_params["min_samples_in_leaf"], criterion="scaled_entropy")
            dt.fit(train_X, train_y)
            preds = [dt.predict(x) for x in test_X]
            fold_accs.append(accuracy_score(test_y, preds))
        cv_acc = np.mean(fold_accs)
        print(f"5-Fold CV Accuracy: {cv_acc:.3f}")
        final_tree = DecisionTree(max_depth=best_params["max_depth"], min_samples_in_leaf=best_params["min_samples_in_leaf"], criterion="scaled_entropy")
        final_tree.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_params = {"max_depth": None, "min_samples_in_leaf": 1}
        if best_params["max_depth"] is None:
            best_params["max_depth"] = 15
            print("Overriding max_depth=None with max_depth=15 for faster training on large dataset.")
        final_tree = DecisionTree(max_depth=best_params["max_depth"], min_samples_in_leaf=best_params["min_samples_in_leaf"], criterion="scaled_entropy")
        final_tree.fit(X_train, y_train)
        preds = [final_tree.predict(x) for x in X_test]
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {acc:.3f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("Classification Report:")
        print(classification_report(y_test, preds))
    final_tree.compute_feature_importances()
    print("\nLearned Tree Structure:")
    final_tree.print_tree()
    print("\nFeature Importances:")
    for feat, imp in final_tree.feature_importances_.items():
        print(f"{feat}: {imp:.4f}")
