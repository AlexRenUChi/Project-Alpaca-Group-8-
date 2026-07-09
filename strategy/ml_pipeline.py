"""Machine-learning trading signal pipeline (tasks 3 & 4).

StandardScaler -> PCA (keep >= 80% variance) -> classifier. Target is the sign
of the next-day return; the model outputs P(up) and we go Long when that
probability clears a threshold (default 0.6), otherwise Flat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from strategy.indicators import FEATURE_COLUMNS

MODEL_CHOICES = [
    "Random Forest",
    "Logistic Regression",
    "Gradient Boosting",
    "SVM",
    "MLP",
]


def _make_model(model_type: str, random_state: int = 42):
    if model_type == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if model_type == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_type == "SVM":
        return SVC(probability=True, class_weight="balanced", random_state=random_state)
    if model_type == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=random_state)
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def build_dataset(
    feature_df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with a binary target: next-day return > 0.

    Rows with any missing feature, or an unknown (last-row) target, are dropped.
    """
    d = feature_df.copy().replace([np.inf, -np.inf], np.nan)
    next_close = d["close"].shift(-1)
    target = (next_close > d["close"]).astype(float)
    target[next_close.isna()] = np.nan
    d["target"] = target

    d = d.dropna(subset=feature_columns + ["target"])
    X = d[feature_columns]
    y = d["target"].astype(int)
    return X, y


def latest_feature_row(
    feature_df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Most recent fully-populated feature row (target unknown, live inference)."""
    feats = feature_df[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if feats.empty:
        raise ValueError("No complete feature row available for inference.")
    return feats.iloc[[-1]]


def time_series_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    split = int(n * (1 - test_size))
    split = max(1, min(split, n - 1))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


class MLTradingModel:
    """Bundle of scaler + PCA + classifier with a probability-threshold signal."""

    def __init__(
        self,
        model_type: str = "Random Forest",
        variance_threshold: float = 0.80,
        prob_threshold: float = 0.60,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.variance_threshold = variance_threshold
        self.prob_threshold = prob_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.model = _make_model(model_type, random_state)
        self.n_components_: int | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_variance_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLTradingModel":
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        evr = self.pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        # Smallest number of components reaching the variance threshold.
        k = int(np.searchsorted(cum, self.variance_threshold) + 1)
        self.n_components_ = int(min(max(k, 1), len(evr)))
        self.explained_variance_ratio_ = evr
        self.cumulative_variance_ = cum
        X_pca = self.pca.transform(X_scaled)[:, : self.n_components_]
        self.model.fit(X_pca, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))[:, : self.n_components_]

    def predict_proba_long(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(self.transform(X))
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        # Degenerate training window with a single class.
        return np.zeros(len(X)) if classes[0] == 0 else np.ones(len(X))

    def predict_signal(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba_long(X) > self.prob_threshold).astype(int)


def run_ml_strategy(
    feature_df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.30,
    prob_threshold: float = 0.60,
    variance_threshold: float = 0.80,
) -> dict:
    """Train on the in-sample window, generate an out-of-sample Long/Flat signal.

    Returns everything the app needs to backtest and to reuse the model live.
    """
    X, y = build_dataset(feature_df)
    if len(X) < 100:
        raise ValueError("Not enough labelled rows to train the model.")

    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size)
    model = MLTradingModel(model_type, variance_threshold, prob_threshold).fit(X_train, y_train)

    proba_test = model.predict_proba_long(X_test)
    signal = pd.Series((proba_test > prob_threshold).astype(int), index=X_test.index, name="ML Signal")
    proba = pd.Series(proba_test, index=X_test.index, name="P(up)")

    directional_pred = (proba_test > 0.5).astype(int)
    accuracy = float((directional_pred == y_test.to_numpy()).mean())
    base_rate = float(y_test.mean())  # naive "always up" accuracy

    return {
        "model": model,
        "signal": signal,
        "proba": proba,
        "y_test": y_test,
        "accuracy": accuracy,
        "base_rate": base_rate,
        "test_index": X_test.index,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_components": model.n_components_,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "cumulative_variance": model.cumulative_variance_,
        "feature_columns": FEATURE_COLUMNS,
    }


def train_full_model(
    feature_df: pd.DataFrame,
    model_type: str = "Random Forest",
    prob_threshold: float = 0.60,
    variance_threshold: float = 0.80,
) -> MLTradingModel:
    """Train on all labelled history — used for live paper-trading inference."""
    X, y = build_dataset(feature_df)
    if len(X) < 100:
        raise ValueError("Not enough labelled rows to train the model.")
    return MLTradingModel(model_type, variance_threshold, prob_threshold).fit(X, y)
