# Lending Club Probability of Default Model Card

## Intended use

Research and demonstration of a consumer-loan probability-of-default workflow. The model is not validated for production credit decisions.

## Target and population

The target is `Charged Off` versus `Fully Paid` among completed Lending Club loans. Rejected applications are not assigned repayment outcomes and are excluded from model training. Cohorts are maturity-filtered using a 2018 Q4 observation cutoff.

## Candidate models

Regularized Logistic Regression and LightGBM are trained on the same application-time feature set. Both are calibrated with isotonic regression. The champion is selected on out-of-time validation performance, with Logistic Regression preferred when AUC differs by less than 0.01.

## Decision policy

The research policy approves an application when calibrated probability of default is at or below 15%. The dashboard exposes the cutoff for scenario analysis; changing the cutoff does not retrain the model.

## Exclusions and limitations

- Grade, sub-grade, interest rate, installment, and listing status are excluded to avoid underwriting-process leakage.
- Raw address, ZIP code, state, job title, and free text are excluded from the model.
- Protected-class attributes are not present. Fair-lending validation cannot be completed with this dataset.
- Declined applicants have no observed repayment outcomes, so sample-selection bias remains.
- The historical data ends in 2018 Q4 and is not representative of current economic or lending conditions.
- Applicant-level reasons are research explanations and are not a legally sufficient adverse-action notice.

## Monitoring triggers

Retrain or suspend use when discrimination, calibration, missingness, out-of-range input rates, or segment performance materially deteriorate. Production use requires independent validation, current institutional data, compliance review, audit controls, and outcome monitoring.
