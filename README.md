# Telco Customer Churn — Who to Call, and Whether It Pays

[![CI](https://github.com/Ampofowaa/churn-retention-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Ampofowaa/churn-retention-mlops/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1330%20passing-brightgreen)](CONTRIBUTING.md#testing)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](CONTRIBUTING.md#testing)

Tells a telecom retention team **which customers are worth contacting, and what the campaign earns**. Trained, calibrated, gated by a human review, and live on AWS.

**▶ Live demo: [telco-churn.duckdns.org](https://telco-churn.duckdns.org)** — no login. In **Batch Prediction**, click **Score the sample**.
Also live: [API docs](https://telco-churn.duckdns.org/docs) · [MLflow runs](https://telco-churn.duckdns.org/mlflow) (`reviewer` / `telco-reviewer-2026`)

![Scoring 50 customers in the Batch Prediction tab, then explaining one prediction](docs/img/batch_scoring.gif)

## Problem

Every retention contact costs money, and only some contacted customers stay. Calling everyone wastes budget; calling no one loses revenue. So the question isn't *who will churn?* but *who is worth contacting?*

## Approach

- **LightGBM, selected on PR-AUC** — the right metric when only 26.5 % of customers churn. It beat logistic regression across 100 paired CV folds.
- **Calibrated probabilities** — the contact threshold `t* = c / (r × LTV)` comes from costs, so a predicted 40 % must really mean 40 %.
- **Test set opened once**, then an automated gate and a human review before serving. Both rules are enforced by tests.

## Results

| Sealed test set, 1,409 customers | Model | Baseline |
|---|---|---|
| PR-AUC | **0.670** | 0.265 (churn rate for everyone) |
| Churners caught | **261 of 374** (recall 0.698) | — |
| Calibration slope (ideal 1.0) | **0.992** | — |
| Expected value, base cost scenario | **+$15,061** | beats calling everyone and calling no one |

Confidence intervals, all three cost scenarios, error analysis → [ANALYSIS.md §7](ANALYSIS.md#7-final-test-set-results)

---

## What drives churn

- **Contract type and tenure dominate** — together over a third of the model's signal (SHAP). Short-tenure, month-to-month customers are the highest-risk group by a wide margin.
- **Early outreach buys the most** — a customer's first few months move the risk score more than any other window.
- **Fairness reviewed, not just measured** — disparities across protected groups track real differences in churn risk rather than proxy discrimination, and are carried into monitoring.

## Limitations

- **Blind spot: long-tenure, annual-contract customers.** The model rarely flags them, and the root cause is a signal missing from nearly every error type — nothing in this dataset measures loyalty or satisfaction directly. A manual outreach rule and an engineered interaction feature close part of the gap; closing it fully needs new data ([business takeaways](ANALYSIS.md#business-takeaways)).
- **Retention rates are assumptions, not observations.** The dataset doesn't record who was contacted or who was saved, so expected value rests on industry-benchmark retention rates — which is why EV is reported across three scenarios and a sensitivity sweep, and never gates promotion.
- **One LTV per scenario, not per customer.** Ranking by expected value is therefore identical to ranking by churn probability; a per-customer LTV (paired with uplift modelling) is the leading next step.
- **Static, timestamp-less dataset.** There is no live label stream, so the retraining design compares challenger vs. champion on a held-back reserve cohort as a stand-in for fresh production data ([ANALYSIS.md §9](ANALYSIS.md#9-known-limitations) #13).

Full list → [ANALYSIS.md §9](ANALYSIS.md#9-known-limitations).

---

## Pipeline

```
Raw CSV
  └─ Ingest → Postgres (`customers_raw`)
       └─ Validation (Pandera, 5 quality gates)
            └─ Feature engineering (SQL view for charge-per-service; pandas feature search)
                 └─ Model training (LightGBM + Optuna tuning) → calibration → register challenger
                      └─ Threshold → sealed-test evaluation → error analysis → human review
                           └─ Champion registration
                                └─ Serving (FastAPI + Streamlit)
```

- **Data:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features, 26.5 % churn. Stratified split, sealed before any modelling: 5,634 dev (10×10 repeated CV for every decision) / 1,409 test (opened once).
- **Reproducible:** the nine file-producing stages are a content-hashed DVC DAG (`dvc.yaml`); every run, model, and decision artifact is logged to MLflow.
- **Served:** FastAPI hot-reloads the champion when its MLflow alias moves — promotion needs no redeploy. Batch scoring ranks customers by expected value and caps contacts at the team's capacity and budget. Shadow and canary rollout of a challenger are built in, off by default.
- **Deployed:** one EC2 box behind Caddy (HTTPS), with Postgres on RDS and MLflow artifacts on S3. CI runs on every push; CD deploys over SSM, smoke-tests, and rolls back automatically on failure. CloudWatch alarms and an operational dashboard (CPU, disk, memory, status checks) are live, alongside a Grafana Cloud dashboard covering request-level serving metrics; Grafana-managed alert rules are the one piece still to wire up.
- **Tested:** unit, integration (ephemeral Postgres via testcontainers), subprocess CLI, FastAPI and Streamlit suites, plus architecture tests that enforce the modelling rules in code.

Detail: [docs/architecture.md](docs/architecture.md) (diagrams, modelling stages, DVC, serving) · [ANALYSIS.md](ANALYSIS.md) (every decision and number)

**Stack:** Python 3.13 · uv · LightGBM · Optuna · scikit-learn · SHAP · Pandera · Postgres · MLflow · DVC · FastAPI · Streamlit · Docker · GitHub Actions · AWS (EC2, RDS, S3, ECR, CloudWatch) · Grafana Cloud

## Status

Phases 0–9 (data through serving), retraining foundations, the AWS deployment, and CloudWatch + Grafana Cloud monitoring (alarms and both dashboards) are done. **Planned:** Grafana-managed alert rules, then Prefect orchestration of monthly retraining and daily drift checks. Roadmap → [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) · history → [CHANGELOG.md](CHANGELOG.md)

## Run it locally

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and Docker. The raw CSV is committed, so there's nothing to download.

```bash
git clone https://github.com/Ampofowaa/churn-retention-mlops.git
cd churn-retention-mlops
uv sync --all-extras
uv run pytest                  # unit tests, no Docker needed
```

To train a model end to end (DVC pipeline → registration → human review → promotion) and serve it with `docker compose`, follow [CONTRIBUTING.md § Running the full stack locally](CONTRIBUTING.md#running-the-full-stack-locally).

## Project structure

```
src/telco_churn/
  data/        ingest, validation, dev/test split
  features/    SQL feature views, feature discovery & selection
  models/      training, calibration, threshold, evaluation, gate, registry
  serving/     FastAPI app, hot-reloading model bundle, contact policy
  ui/          Streamlit app
configs/       Hydra YAML (training, costs, promotion gate, serving)
sql/           Postgres schema + feature views
notebooks/     00–05, rendered with outputs: EDA → features → modelling → evaluation
tests/         unit, integration, Streamlit
infra/         Terraform + production compose
```

Full tree → [CONTRIBUTING.md § Project Structure](CONTRIBUTING.md#project-structure)

## Author

**Richlove Frimpong** — [LinkedIn](https://www.linkedin.com/in/richlove-frimpong) · [GitHub](https://github.com/Ampofowaa)
