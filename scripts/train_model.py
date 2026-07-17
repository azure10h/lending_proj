#!/usr/bin/env python3
"""Train the Lending Club underwriting research model and build web artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
PD_CUTOFF = 0.15
NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "credit_history_years",
]
CATEGORICAL_FEATURES = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def pct(value: float) -> float:
    return round(float(value) * 100, 2)


def parse_emp_length(value) -> str:
    if pd.isna(value):
        return "Missing/Unknown"
    return str(value).strip()


def prepare_accepted(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_date"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (
        (df["issue_date"] - df["earliest_date"]).dt.days / 365.25
    ).clip(lower=0)
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)", expand=False).fillna("Missing/Unknown")
    df["emp_length"] = df["emp_length"].map(parse_emp_length)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Missing/Unknown").astype(str).str.strip()
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)
    df["state"] = df["address"].astype(str).str.extract(r",\s*([A-Z]{2})\s+\d{5}\s*$", expand=False)

    cutoff_36 = pd.Timestamp("2015-12-31")
    cutoff_60 = pd.Timestamp("2013-12-31")
    mature = df[
        ((df["term"] == "36") & (df["issue_date"] <= cutoff_36))
        | ((df["term"] == "60") & (df["issue_date"] <= cutoff_60))
    ].copy()
    return df, mature


def temporal_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, valid_idx, test_idx = [], [], []
    for term, group in df.groupby("term"):
        months = np.array(sorted(group["issue_date"].dropna().unique()))
        train_end = months[max(0, math.ceil(len(months) * 0.70) - 1)]
        valid_end = months[max(0, math.ceil(len(months) * 0.85) - 1)]
        train_idx.extend(group.index[group["issue_date"] <= train_end].tolist())
        valid_idx.extend(group.index[(group["issue_date"] > train_end) & (group["issue_date"] <= valid_end)].tolist())
        test_idx.extend(group.index[group["issue_date"] > valid_end].tolist())
    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


def expected_calibration_error(y, p, bins=10) -> float:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), actual=("y", "mean"), pred=("p", "mean"))
    return float(((grouped["n"] / len(frame)) * (grouped["actual"] - grouped["pred"]).abs()).sum())


def calibrate(y_valid, raw_valid, raw_test):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_valid, y_valid)
    return iso, iso.predict(raw_test)


def metric_bundle(y, p, cutoff=PD_CUTOFF) -> dict:
    pred = (p > cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, p)
    approved = p <= cutoff
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "ks": float(np.max(tpr - fpr)),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "approval_rate": float(approved.mean()),
        "approved_bad_rate": float(y[approved].mean()) if approved.any() else None,
        "rejection_capture": float(y[~approved].sum() / max(1, y.sum())),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_points(y, p, bins=10):
    frame = pd.DataFrame({"actual": y, "predicted": p})
    frame["bucket"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    result = frame.groupby("bucket", observed=True).agg(
        actual=("actual", "mean"), predicted=("predicted", "mean"), loans=("actual", "size")
    )
    return [
        {"bucket": i + 1, "actual": pct(row.actual), "predicted": pct(row.predicted), "loans": int(row.loans)}
        for i, row in enumerate(result.itertuples())
    ]


def threshold_points(y, p):
    rows = []
    for threshold in np.arange(0.05, 0.305, 0.01):
        approved = p <= threshold
        rows.append({
            "threshold": round(float(threshold), 2),
            "approvalRate": pct(approved.mean()),
            "approvedBadRate": pct(y[approved].mean()) if approved.any() else None,
            "badCaptured": pct(y[~approved].sum() / max(1, y.sum())),
        })
    return rows


def aggregate_rejected(path: Path) -> dict:
    by_year: dict[int, dict] = {}
    states: dict[str, int] = {}
    total = 0
    amount = 0.0
    risk_present = 0
    for chunk in pd.read_csv(path, chunksize=500_000):
        dates = pd.to_datetime(chunk["Application Date"], errors="coerce")
        chunk["year"] = dates.dt.year
        chunk_amount = pd.to_numeric(chunk["Amount Requested"], errors="coerce").fillna(0)
        total += len(chunk)
        amount += float(chunk_amount.sum())
        risk_present += int(chunk["Risk_Score"].notna().sum())
        grouped = chunk.assign(amount=chunk_amount).groupby("year", dropna=True).agg(
            applications=("year", "size"), amount=("amount", "sum"), riskPresent=("Risk_Score", "count")
        )
        for year, row in grouped.iterrows():
            key = int(year)
            current = by_year.setdefault(key, {"applications": 0, "amount": 0.0, "riskPresent": 0})
            current["applications"] += int(row.applications)
            current["amount"] += float(row.amount)
            current["riskPresent"] += int(row.riskPresent)
        for state, count in chunk["State"].value_counts().items():
            states[str(state)] = states.get(str(state), 0) + int(count)
    return {
        "total": total,
        "amount": amount,
        "riskScoreCoverage": risk_present / total,
        "byYear": [
            {"year": year, **values, "riskCoverage": pct(values["riskPresent"] / values["applications"])}
            for year, values in sorted(by_year.items())
        ],
        "topStates": [{"state": state, "applications": count} for state, count in sorted(states.items(), key=lambda x: x[1], reverse=True)[:12]],
    }


def group_portfolio(df: pd.DataFrame, field: str, limit=None):
    grouped = df.groupby(field, dropna=False).agg(
        loans=("target", "size"), chargedOff=("target", "sum"), amount=("loan_amnt", "sum")
    ).reset_index()
    grouped["badRate"] = grouped["chargedOff"] / grouped["loans"] * 100
    grouped = grouped.sort_values("loans", ascending=False)
    if limit:
        grouped = grouped.head(limit)
    return [
        {
            "label": "Missing/Unknown" if pd.isna(row[field]) else str(row[field]),
            "loans": int(row.loans),
            "chargedOff": int(row.chargedOff),
            "amount": float(row.amount),
            "badRate": round(float(row.badRate), 2),
        }
        for _, row in grouped.iterrows()
    ]


def build_dashboard_data(full: pd.DataFrame, mature: pd.DataFrame, rejected: dict, model: dict) -> dict:
    yearly = full.assign(year=full["issue_date"].dt.year).groupby("year", dropna=True).agg(
        loans=("target", "size"), chargedOff=("target", "sum"), amount=("loan_amnt", "sum")
    ).reset_index()
    yearly["badRate"] = yearly["chargedOff"] / yearly["loans"] * 100
    missing = full[FEATURES].isna().mean().sort_values(ascending=False).head(10)
    return {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "portfolio": {
            "loans": int(len(full)),
            "chargedOff": int(full["target"].sum()),
            "badRate": pct(full["target"].mean()),
            "fundedAmount": float(full["loan_amnt"].sum()),
            "matureModelPopulation": int(len(mature)),
            "dateStart": full["issue_date"].min().strftime("%b %Y"),
            "dateEnd": full["issue_date"].max().strftime("%b %Y"),
        },
        "yearly": [
            {"year": int(r.year), "loans": int(r.loans), "chargedOff": int(r.chargedOff), "amount": float(r.amount), "badRate": round(float(r.badRate), 2)}
            for r in yearly.itertuples()
        ],
        "segments": {
            "grade": group_portfolio(full, "grade"),
            "term": group_portfolio(full, "term"),
            "purpose": group_portfolio(full, "purpose", 10),
            "homeOwnership": group_portfolio(full, "home_ownership"),
        },
        "missingness": [{"feature": str(name), "percent": pct(value)} for name, value in missing.items()],
        "rejected": rejected,
        "model": model,
    }


def export_onnx(model, feature_count: int, path: Path, kind: str):
    if kind == "LightGBM":
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        converted = convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, feature_count]))], target_opset=15, zipmap=False)
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        converted = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, feature_count]))], target_opset=15, options={id(model): {"zipmap": False}})
    path.write_bytes(converted.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "public")
    args = parser.parse_args()

    accepted_path = args.data_dir / "lending_club_loan_two.csv"
    rejected_path = args.data_dir / "rejected_2007_to_2018Q4.csv"
    output = args.output_dir
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "model").mkdir(parents=True, exist_ok=True)
    (Path(__file__).resolve().parents[1] / "artifacts").mkdir(exist_ok=True)

    print("Reading and preparing completed loans...")
    full, mature = prepare_accepted(accepted_path)
    train_idx, valid_idx, test_idx = temporal_split(mature)
    train = mature.loc[train_idx]
    valid = mature.loc[valid_idx]
    test = mature.loc[test_idx]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing/Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ], sparse_threshold=0)
    x_train = transformer.fit_transform(train[FEATURES]).astype(np.float32)
    x_valid = transformer.transform(valid[FEATURES]).astype(np.float32)
    x_test = transformer.transform(test[FEATURES]).astype(np.float32)
    y_train = train["target"].to_numpy()
    y_valid = valid["target"].to_numpy()
    y_test = test["target"].to_numpy()

    print(f"Training Logistic Regression and LightGBM on {len(train):,} mature loans...")
    logistic = LogisticRegression(C=0.2, max_iter=1000, random_state=SEED)
    logistic.fit(x_train, y_train)
    lightgbm = lgb.LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.03, num_leaves=31,
        max_depth=6, min_child_samples=100, subsample=0.9, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    lightgbm.fit(x_train, y_train)

    candidates = {}
    for name, estimator in [("Logistic Regression", logistic), ("LightGBM", lightgbm)]:
        valid_raw = estimator.predict_proba(x_valid)[:, 1]
        test_raw = estimator.predict_proba(x_test)[:, 1]
        iso, test_cal = calibrate(y_valid, valid_raw, test_raw)
        valid_cal = iso.predict(valid_raw)
        candidates[name] = {
            "estimator": estimator,
            "calibrator": iso,
            "validation": metric_bundle(y_valid, valid_cal),
            "test": metric_bundle(y_test, test_cal),
            "test_prob": test_cal,
        }

    valid_aucs = {name: data["validation"]["roc_auc"] for name, data in candidates.items()}
    best = max(valid_aucs, key=valid_aucs.get)
    if abs(valid_aucs["LightGBM"] - valid_aucs["Logistic Regression"]) < 0.01:
        best = "Logistic Regression"
    champion = candidates[best]
    test_metrics = champion["test"]
    validated = (
        test_metrics["roc_auc"] >= 0.65
        and test_metrics["ks"] >= 0.25
        and test_metrics["ece"] <= 0.03
        and test_metrics["brier"] < float(y_test.mean() * (1 - y_test.mean()))
    )

    print("Aggregating rejected applications...")
    rejected = aggregate_rejected(rejected_path)
    model_summary = {
        "champion": best,
        "version": "LC-PD-2026.07-v1",
        "validated": validated,
        "cutoff": PD_CUTOFF,
        "trainingRows": int(len(train)),
        "validationRows": int(len(valid)),
        "testRows": int(len(test)),
        "trainBadRate": pct(y_train.mean()),
        "validationBadRate": pct(y_valid.mean()),
        "testBadRate": pct(y_test.mean()),
        "candidates": {
            name: {"validation": data["validation"], "test": data["test"]}
            for name, data in candidates.items()
        },
        "calibration": calibration_points(y_test, champion["test_prob"]),
        "thresholds": threshold_points(y_test, champion["test_prob"]),
    }
    dashboard = build_dashboard_data(full, mature, rejected, model_summary)

    encoder = transformer.named_transformers_["categorical"].named_steps["onehot"]
    feature_names = list(transformer.get_feature_names_out())
    numeric_imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    scaler = transformer.named_transformers_["numeric"].named_steps["scaler"]
    quantiles = {
        col: {
            "p01": float(train[col].quantile(0.01)),
            "p50": float(train[col].quantile(0.50)),
            "p99": float(train[col].quantile(0.99)),
        }
        for col in NUMERIC_FEATURES
    }
    calibration = champion["calibrator"]
    contract = {
        "modelVersion": model_summary["version"],
        "champion": best,
        "validated": validated,
        "cutoff": PD_CUTOFF,
        "riskScoreDirection": "Higher is safer",
        "numericFeatures": NUMERIC_FEATURES,
        "categoricalFeatures": CATEGORICAL_FEATURES,
        "featureNames": feature_names,
        "numeric": {
            col: {
                "impute": float(numeric_imputer.statistics_[i]),
                "mean": float(scaler.mean_[i]),
                "scale": float(scaler.scale_[i]),
                **quantiles[col],
            }
            for i, col in enumerate(NUMERIC_FEATURES)
        },
        "categories": {
            col: [str(v) for v in encoder.categories_[i].tolist()]
            for i, col in enumerate(CATEGORICAL_FEATURES)
        },
        "calibration": {
            "x": [float(v) for v in calibration.X_thresholds_],
            "y": [float(v) for v in calibration.y_thresholds_],
        },
        "references": {
            "numeric": {col: float(train[col].median()) for col in NUMERIC_FEATURES},
            "categorical": {col: str(train[col].mode(dropna=True).iloc[0]) for col in CATEGORICAL_FEATURES},
        },
        "reasonLabels": {
            "loan_amnt": "Requested loan amount is high relative to the training population",
            "annual_inc": "Annual income is below the model's reference profile",
            "dti": "Debt-to-income ratio is elevated",
            "open_acc": "Open credit account profile increased estimated risk",
            "pub_rec": "Public records increased estimated risk",
            "revol_bal": "Revolving balance increased estimated risk",
            "revol_util": "Revolving credit utilization is elevated",
            "total_acc": "Total credit account history increased estimated risk",
            "mort_acc": "Mortgage account history increased estimated risk",
            "pub_rec_bankruptcies": "Public-record bankruptcies increased estimated risk",
            "credit_history_years": "Credit history is shorter than the model's reference profile",
            "term": "Requested loan term increased estimated risk",
            "emp_length": "Employment history increased estimated risk",
            "home_ownership": "Home ownership profile increased estimated risk",
            "verification_status": "Income verification profile increased estimated risk",
            "purpose": "Loan purpose increased estimated risk",
            "application_type": "Application type increased estimated risk",
        },
    }

    print(f"Exporting {best} model...")
    export_onnx(champion["estimator"], x_train.shape[1], output / "model" / "model.onnx", best)
    joblib.dump(transformer, Path(__file__).resolve().parents[1] / "artifacts" / "preprocessor.joblib")
    joblib.dump(champion["estimator"], Path(__file__).resolve().parents[1] / "artifacts" / "champion.joblib")
    (output / "data" / "dashboard-data.json").write_text(json.dumps(dashboard, default=json_default, separators=(",", ":")))
    (output / "model" / "model-contract.json").write_text(json.dumps(contract, default=json_default, separators=(",", ":")))
    (Path(__file__).resolve().parents[1] / "artifacts" / "metrics.json").write_text(json.dumps(model_summary, default=json_default, indent=2))

    print(json.dumps({
        "champion": best,
        "validated": validated,
        "validationAUC": round(champion["validation"]["roc_auc"], 4),
        "testAUC": round(test_metrics["roc_auc"], 4),
        "testKS": round(test_metrics["ks"], 4),
        "testECE": round(test_metrics["ece"], 4),
        "testRows": len(test),
    }, indent=2))


if __name__ == "__main__":
    main()
