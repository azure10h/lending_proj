#!/usr/bin/env python3
"""Validate preprocessing parity, ONNX inference, aggregate integrity, and privacy."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from train_model import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, prepare_accepted


ROOT = Path(__file__).resolve().parents[1]


def contract_vector(row: dict, contract: dict) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    warnings: list[str] = []
    for feature in contract["numericFeatures"]:
        rule = contract["numeric"][feature]
        value = float(row.get(feature, rule["impute"]))
        if not np.isfinite(value):
            value = rule["impute"]
        if value < rule["p01"] or value > rule["p99"]:
            warnings.append(feature)
            value = min(rule["p99"], max(rule["p01"], value))
        values.append((value - rule["mean"]) / (rule["scale"] or 1))
    for feature in contract["categoricalFeatures"]:
        current = str(row.get(feature, "Missing/Unknown"))
        values.extend(1.0 if current == category else 0.0 for category in contract["categories"][feature])
    return np.asarray(values, dtype=np.float32), warnings


def main():
    contract = json.loads((ROOT / "public/model/model-contract.json").read_text())
    dashboard = json.loads((ROOT / "public/data/dashboard-data.json").read_text())
    transformer = joblib.load(ROOT / "artifacts/preprocessor.joblib")
    champion = joblib.load(ROOT / "artifacts/champion.joblib")
    full, mature = prepare_accepted(ROOT.parent / "lending_club_loan_two.csv")
    sample = mature.sample(12, random_state=42).copy()

    # Browser preprocessing caps at training p01/p99, so compare against the same capped rows.
    capped = sample[FEATURES].copy()
    for feature in NUMERIC_FEATURES:
        rule = contract["numeric"][feature]
        capped[feature] = pd.to_numeric(capped[feature], errors="coerce").fillna(rule["impute"]).clip(rule["p01"], rule["p99"])
    python_matrix = transformer.transform(capped).astype(np.float32)
    contract_matrix = np.stack([contract_vector(row, contract)[0] for row in sample[FEATURES].to_dict("records")])
    transform_delta = float(np.max(np.abs(python_matrix - contract_matrix)))

    native = champion.predict_proba(python_matrix)[:, 1]
    session = ort.InferenceSession(str(ROOT / "public/model/model.onnx"), providers=["CPUExecutionProvider"])
    probabilities = session.run(["probabilities"], {"input": python_matrix})[0][:, 1]
    inference_delta = float(np.max(np.abs(native - probabilities)))

    public_text = (ROOT / "public/data/dashboard-data.json").read_text() + (ROOT / "public/model/model-contract.json").read_text()
    forbidden = ["address", "zip code", "emp_title", "Michelle Gateway"]
    leaked = [term for term in forbidden if term.lower() in public_text.lower()]
    assert dashboard["portfolio"]["loans"] == 396030
    assert dashboard["portfolio"]["chargedOff"] == 77673
    assert dashboard["rejected"]["total"] == 27648741
    assert len(contract["featureNames"]) == python_matrix.shape[1] == 48
    assert transform_delta < 1e-6, transform_delta
    assert inference_delta < 1e-5, inference_delta
    assert not leaked, leaked

    fixtures = []
    for row, raw_pd in zip(sample[FEATURES].to_dict("records")[:4], native[:4]):
        fixtures.append({
            "applicant": {k: (None if pd.isna(v) else v) for k, v in row.items()},
            "rawProbabilityOfDefault": float(raw_pd),
        })
    (ROOT / "tests/model-fixtures.json").write_text(json.dumps(fixtures, indent=2, default=str))
    print(json.dumps({
        "preprocessing_max_delta": transform_delta,
        "onnx_max_probability_delta": inference_delta,
        "feature_count": python_matrix.shape[1],
        "privacy_terms_found": leaked,
        "source_counts_reconciled": True,
    }, indent=2))


if __name__ == "__main__":
    main()
