from __future__ import annotations

import pandas as pd

from fraud_mlops.transforms import ReusableFraudTransformer


def test_transformer_fit_transform_round_trip() -> None:
    frame = pd.DataFrame(
        {
            "claim_amount": [100.0, 150.0, 320.0],
            "customer_age": [25, 44, 37],
            "policy_tenure_months": [12, 24, 60],
            "prior_claims": [0, 2, 1],
            "payment_delay_days": [1, 7, 3],
            "channel": ["online", "agent", "partner"],
            "region": ["north", "south", "east"],
            "policy_type": ["auto", "home", "health"],
            "fraud_label": [0, 1, 0],
        }
    )

    transformer = ReusableFraudTransformer(
        numeric_columns=[
            "claim_amount",
            "customer_age",
            "policy_tenure_months",
            "prior_claims",
            "payment_delay_days",
        ],
        categorical_columns=["channel", "region", "policy_type"],
    ).fit(frame)

    x, y = transformer.transform_with_target(frame, "fraud_label")
    assert x.shape[0] == len(frame)
    assert y.shape == (len(frame),)

    clone = ReusableFraudTransformer.from_state(transformer.to_state())
    x2 = clone.transform_features(frame)
    assert x2.shape == x.shape

