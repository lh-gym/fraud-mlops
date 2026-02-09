from __future__ import annotations

import numpy as np
import pytest



def test_train_model_returns_metrics() -> None:
    pytest.importorskip("torch")
    from fraud_mlops.train import train_model

    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(128, 10)).astype(np.float32)
    y_train = rng.integers(0, 2, size=128).astype(np.float32)
    x_val = rng.normal(size=(64, 10)).astype(np.float32)
    y_val = rng.integers(0, 2, size=64).astype(np.float32)

    result = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        hidden_dims=[16, 8],
        lr=1e-3,
        epochs=2,
        batch_size=32,
        dropout=0.1,
        backend="threading",
        seed=11,
    )

    assert 0.0 <= result.metrics["f1"] <= 1.0
    assert result.metrics["latency_p95_ms"] >= 0.0
