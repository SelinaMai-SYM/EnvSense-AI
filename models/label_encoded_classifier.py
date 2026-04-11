from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


class LabelEncodedClassifier:
    """
    Adapt estimators that expect integer class ids so the rest of the project
    can keep working with human-readable string labels.
    """

    def __init__(self, estimator: Any, *, class_names: Sequence[str] | None = None) -> None:
        self.estimator = estimator
        self._preferred_class_names = tuple(class_names) if class_names is not None else None
        self.classes_: np.ndarray = np.array([], dtype=object)
        self._label_to_index: dict[object, int] = {}

    def _resolve_classes(self, y: Iterable[object]) -> np.ndarray:
        y_arr = np.asarray(list(y), dtype=object)
        unique_labels = {label for label in y_arr.tolist()}
        if self._preferred_class_names is None:
            ordered = sorted(unique_labels, key=str)
        else:
            ordered = [label for label in self._preferred_class_names if label in unique_labels]
        return np.asarray(ordered, dtype=object)

    def fit(self, X: Any, y: Iterable[object]) -> "LabelEncodedClassifier":
        y_arr = np.asarray(list(y), dtype=object)
        self.classes_ = self._resolve_classes(y_arr)
        self._label_to_index = {label: idx for idx, label in enumerate(self.classes_.tolist())}
        y_encoded = np.asarray([self._label_to_index[label] for label in y_arr.tolist()], dtype=int)
        self.estimator.fit(X, y_encoded)
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded = np.asarray(self.estimator.predict(X), dtype=int)
        return self.classes_[encoded]

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X), dtype=float)

