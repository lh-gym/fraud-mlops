from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ReusableFraudTransformer:
    """Reusable transformation layer shared by multiple model architectures."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    categorical_vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    is_fitted: bool = False

    def fit(self, frame: pd.DataFrame) -> "ReusableFraudTransformer":
        self.numeric_stats = {}
        for col in self.numeric_columns:
            values = pd.to_numeric(frame.get(col, 0.0), errors="coerce").fillna(0.0).astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            self.numeric_stats[col] = (mean, std if std > 1e-6 else 1.0)

        self.categorical_vocab = {}
        for col in self.categorical_columns:
            tokens = frame.get(col, "UNKNOWN").astype(str).fillna("UNKNOWN")
            sorted_tokens = sorted(tokens.unique().tolist())
            self.categorical_vocab[col] = {token: idx for idx, token in enumerate(sorted_tokens, start=1)}

        self.is_fitted = True
        return self

    def _encode_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = []
        for col in self.numeric_columns:
            values = pd.to_numeric(frame.get(col, 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
            mean, std = self.numeric_stats[col]
            matrix.append(((values - mean) / std).astype(np.float32))
        return np.column_stack(matrix) if matrix else np.empty((len(frame), 0), dtype=np.float32)

    def _encode_categorical(self, frame: pd.DataFrame) -> np.ndarray:
        encoded_parts = []
        rows = len(frame)
        for col in self.categorical_columns:
            vocab = self.categorical_vocab[col]
            width = len(vocab) + 1
            tokens = frame.get(col, "UNKNOWN").astype(str).fillna("UNKNOWN").to_numpy()
            indices = np.array([vocab.get(token, 0) for token in tokens], dtype=np.int64)
            matrix = np.zeros((rows, width), dtype=np.float32)
            matrix[np.arange(rows), indices] = 1.0
            encoded_parts.append(matrix)
        return np.concatenate(encoded_parts, axis=1) if encoded_parts else np.empty((rows, 0), dtype=np.float32)

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before transform.")
        num = self._encode_numeric(frame)
        cat = self._encode_categorical(frame)
        if num.size == 0:
            return cat
        if cat.size == 0:
            return num
        return np.concatenate([num, cat], axis=1).astype(np.float32)

    def transform_with_target(self, frame: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray]:
        x = self.transform_features(frame)
        y = pd.to_numeric(frame[target_column], errors="coerce").fillna(0).astype(np.float32).to_numpy()
        return x, y

    def to_state(self) -> dict:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_stats": self.numeric_stats,
            "categorical_vocab": self.categorical_vocab,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ReusableFraudTransformer":
        transformer = cls(
            numeric_columns=list(state["numeric_columns"]),
            categorical_columns=list(state["categorical_columns"]),
        )
        transformer.numeric_stats = {
            key: (float(value[0]), float(value[1])) for key, value in state["numeric_stats"].items()
        }
        transformer.categorical_vocab = {
            key: {token: int(idx) for token, idx in vocab.items()}
            for key, vocab in state["categorical_vocab"].items()
        }
        transformer.is_fitted = bool(state["is_fitted"])
        return transformer

