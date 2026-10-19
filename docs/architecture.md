# Telco Churn — Architecture

> **What each view is for:** the five diagrams give the big picture. System Architecture is the tool-level infrastructure flow, ingest to monitoring; ML Workflow is the modelling lifecycle and its feedback loops; Model Family & Feature Selection shows the two decisions a human makes once rather than every cycle; Data Flow is the artifacts moving through ingestion into training; MLflow Layout is the run/registry structure one training cycle builds up inside MLflow. The sections after them zoom into single serving and registry mechanisms (shadow/canary, live customer lookup, emergency rollback), then close with reference tables for the modelling stages, DVC versioning and serving components.
>
> Phases 1–9 are implemented and verified against the code that produces them, along with retraining foundations (Phase 10a) and the AWS deployment (Phase 12a: EC2 + RDS + S3, `ci.yml`, SSM-deploy `cd.yml`). Orchestration (10b), the rest of CI/CD (11), cloud hardening (12b) and ML monitoring (13) are still target only. The System Architecture diagram is the one view spanning both ranges — everything through Serving and the Demo UI is built and running live; of the Monitoring boxes, only the Prometheus `/metrics` endpoint exists so far (no Grafana dashboard or Evidently drift monitor yet), and Orchestration is not built. Every other diagram stays within the built range. In ML Workflow, the Error Analysis 1 → Feature Engineering loop is an iteration within a single training cycle, not an Orchestration concern; the Business Review loop (back to Threshold when cost assumptions change, or to Feature Engineering on drift or new data) is what Orchestration (Phase 10b) will eventually trigger — for now it's manual. Keep the diagrams in sync whenever the underlying modules change shape.

## System Architecture

The tool-level view: what infrastructure or library sits at each step, from raw CSV to monitored production traffic.

```mermaid
flowchart TD
    A["Raw CSV\nIBM Telco Dataset"] -->|ingest| B[("Data Ingestion (Phase 1)\nPostgres 16 + SQLAlchemy\ncustomers_raw")]
    subgraph DVC["DVC (Phase 8) — reproducible pipeline: ingest through error analysis"]
        B -->|validate| C["Data Validation (Phase 2)\nPandera — 5 Quality Gates"]
        C -->|"feature engineering"| D["Feature Engineering (Phase 4)\nSQL Views + ColumnTransformer"]
        D -->|train| E["Model Training (Phase 5)\nLightGBM + Optuna + MLflow"]
        E -->|calibrate| F1["Calibration (Phase 6)\nCalibratedClassifierCV — fit + log only"]
        F2["Threshold (Phase 6)\ncost matrix → t*"] -->|"evaluate + gate"| G["Sealed-Test Evaluation +\nError Analysis (Phase 7)\nCustom gate (gate.py) + SHAP"]
    end
    F1 -->|"register-challenger\n(register.py — outside the DAG,\na registry alias isn't a file)"| RC[("Register Challenger\nMLflow registry, tagged pending")]
    RC -->|threshold| F2
    G -->|"review + promote (pass) /\nreject (fail)"| H["Model Registry (Phase 7)\nreview.py + register.py\nMLflow — champion / challenger"]
    subgraph CICD["CI/CD (Phase 11) & AWS Deployment (Phase 12)\nbuild, test, deploy"]
        H -->|serve| I["Serving (Phase 9)\nFastAPI + uvicorn — /predict"]
        I -->|UI| J["Demo UI (Phase 9)\nStreamlit"]
        I -->|metrics| K["Monitoring (Phase 13)\nPrometheus + Grafana"]
        I -->|logs| L["Monitoring (Phase 13)\nEvidently — Drift Monitor"]
        L -->|trigger| M["Orchestration (Phase 10)\nPrefect 3 — Retrain Flow"]
    end
    M -->|re-runs| D
```

DVC's real boundary spans the whole pipeline drawn inside the subgraph above, **including Calibration (F1) and Threshold (F2)** — both are deterministic, DVC-tracked stages (`dvc.yaml`'s `calibrate`/`threshold`), not decision steps DVC opts out of. What DVC genuinely cannot express is the registry mutation itself: minting the challenger version (`Register Challenger`) and the final promote/reject (`Model Registry`) are both registry writes, and a registry alias is not a file DVC's dependency graph can hash — that is the actual boundary. It is also why `dvc repro` runs in two calls per cycle, not one: `dvc repro calibrate` reproduces everything up to and including Calibration, then `register-challenger` mints a version, then `dvc repro error_analysis` carries Threshold through Error Analysis — Threshold structurally cannot run before a version exists for it to resolve.

## ML Workflow — a loop, not a straight line

The diagram below shows the **modelling lifecycle** the system runs on top of, including the two feedback loops that a linear summary hides.

```mermaid
flowchart TD
    A[Data Ingestion] --> B[Data Validation]
    B --> C["Exploratory Data Analysis (EDA)"]
    C --> D[Feature Engineering]
    D --> E[Baseline Models]
    E --> F["Error Analysis 1 — generative\nblind-spot profiling on baseline FNs"]
    F -.->|"hypothesis-driven\nfeatures back to FE"| D
    F --> G[Hyperparameter Tuning — Optuna]
    G --> H1[Calibration]
    H1 --> RC["Register Challenger\n(register.py — outside the DAG)"]
    RC --> H2[Threshold]
    H2 --> I[Sealed Test Evaluation]
    I --> J["Error Analysis 2 — confirmatory\nSHAP + FN/FP profiling of final model"]
    J --> K[Business Review]
    K --> L["Champion Promotion\nreview.py + register.py: gate pass +\napproved → alias flip; fail → rejected, alias unchanged"]
    K -.->|"cost assumptions\nrevised"| H2
    K -.->|"drift or\nnew data"| D
```

**Solid arrows** — main linear flow. **Dashed arrows** — feedback loops. Full rationale: the Error Analysis 1 loop in `ANALYSIS.md` §4a's "Generative diagnostic loop" subsection; the Business Review loops in §6 (cost-assumption uncertainty) and §8–§10 (drift-monitoring baseline, periodic recalibration).

## Model Family & Feature Selection — Human-Reviewed, Frozen Into Code (Phase 5)

Two decisions this system makes **exactly once, not every training cycle**: which model family to use, and which features to use. Both are made by a human reviewing evidence in a notebook, never computed automatically inside the routine pipeline — the ML Workflow diagram above glosses over this on purpose, since "Baseline Models" → "Hyperparameter Tuning" is the *routine* path, and this is the *periodic* one that occasionally updates what "Baseline Models" even means.

```mermaid
flowchart TD
    subgraph Human["Human-reviewed, periodic — never routine, never per-retrain"]
        T["Real trigger only:\nnew candidate family / drift signal /\nscheduled periodic review"] --> N1["notebooks/03a-model-selection.ipynb\nruns candidates.py + comparison.py"]
        T --> N2["notebooks/03b-feature-selection.ipynb\nruns feature_selection.py's ablation"]
        N1 --> D1{"today's family ==\nCOMMITTED_MODEL_FAMILY?"}
        N2 --> D2{"today's features ==\nCOMMITTED_FEATURES?"}
        D1 -->|"reconfirmed (expected)"| P1["No change —\ntoday's run is additional\nconfirming evidence only"]
        D1 -->|"changed (rare)"| P2["Reviewed PR: update\nCOMMITTED_MODEL_FAMILY +\n_DECISION_RUN_ID, ANALYSIS.md §4a"]
        D2 -->|"reconfirmed (expected)"| P3["No change"]
        D2 -->|"changed (rare)"| P4["Reviewed PR: update\nCOMMITTED_FEATURES +\n_DECISION_RUN_ID, ANALYSIS.md §4b"]
    end
    P1 --> F["Frozen constants:\nmodels/train/common.py::COMMITTED_MODEL_FAMILY\nfeatures/schema.py::COMMITTED_FEATURES"]
    P2 --> F
    P3 --> F
    P4 --> F
    F --> Auto["Automated pipeline (train.py, Steps 1-5)\nevery dvc repro / future Phase 10 retrain:\ntrains COMMITTED_MODEL_FAMILY on\nCOMMITTED_FEATURES — never re-derives either"]
    Auto --> Audit["feature_audit.py (Step 3)\naudits the already-frozen COMMITTED_FEATURES\nfor consistency, every cycle — never re-decides membership"]
```

**Neither notebook is wired into `models/train/__main__.py`.** `candidates.py`/`comparison.py` (Dummy vs. `LogisticRegressionCV` vs. LightGBM, paired-bootstrap comparison) and `feature_selection.py` (permutation-importance ablation) are explicitly notebook-only, on-demand tools — re-run "only on a real trigger... not on every retrain," per their own module docstrings. The routine automated pipeline never touches this decision at all; it just trains whichever family the frozen constant names, on whichever features the frozen constant lists.

**The decision-check inside each notebook is deliberately self-contained.** It compares *today's fresh computation* against the hardcoded constant (`today_family == COMMITTED_MODEL_FAMILY`) — it never looks up or compares against the *original* decision run via MLflow. That makes it correct regardless of which MLflow backend/experiment the re-run happens to log into: running it against a different tracking store (a fresh environment, a different deployment target) doesn't confuse the check, since the original run is only ever cited as a string in the printed message, never dereferenced. The expected, normal outcome on a routine re-run is "Reconfirmed — no change needed"; only a genuine "DECISION CHANGED" result calls for updating the frozen constants, via a reviewed PR that also updates the `_DECISION_RUN_ID` citation and `ANALYSIS.md`'s rationale.

**`feature_audit.py` is a different thing entirely, and easy to confuse with `feature_selection.py`.** It *is* part of the automated pipeline (Step 3 of `train.py`) and runs on every single cycle — but it only **audits** that the already-frozen `COMMITTED_FEATURES` still holds up (columns still exist, still behave consistently against the live feature space), producing a diagnostic. It never re-runs the importance ablation and never changes `COMMITTED_FEATURES` itself. Reading `feature_audit.py`'s presence in the automated pipeline as "feature selection happens every retrain" would be the wrong takeaway — selection is the rare, human-reviewed event above; auditing is the routine, automated check that the last selection decision hasn't silently gone stale.

**The `_DECISION_RUN_ID` citations only ever resolve inside the tracking store they were created in — they are not globally resolvable links.** `COMMITTED_MODEL_FAMILY_DECISION_RUN_ID`/`COMMITTED_FEATURES_DECISION_RUN_ID` point at the specific run that was actually reviewed when the decision was made, which — for this project's decisions to date — lives in local development's MLflow instance. A re-run of these notebooks against a different backend (a fresh environment, a cloud-hosted MLflow) produces a *new*, independent run with its own, different run_id; it can never retroactively become the cited run, no matter how it's re-run. That's fine, because the citation was never meant to be a durable, infrastructure-independent link in the first place — the durable record is git: `ANALYSIS.md §4a`/`§4b`'s narrative and numbers, and the rendered `notebooks/03a-model-selection.ipynb`/`03b-feature-selection.ipynb` themselves (committed with output cells, per this project's Notebook Conventions), both of which survive regardless of whether any particular MLflow backend still exists. A reader who can't reach the cited tracking store isn't missing the evidence — it's already sitting in the repo.

## Data Flow — Ingestion to Training

This shows *artifacts* — what each stage actually reads and writes, and which stages touch the database at all.

```mermaid
flowchart TD
    A["Raw CSV\ndatasets/raw/ — read-only,\nsource of truth"] -->|ingest.py| B[("Postgres\ncustomers_raw")]
    B -->|validate.py| V["Pandera\n5 Quality Gates"]
    V -.->|"reports/validation_receipt.json\nDVC dep — invalidation edge only,\nneither stage reads its contents"| C
    V -.->|"reports/validation_receipt.json"| D
    B -->|"split.py\nre-validates inline, then\nreads only (customerid, churn)"| C["split_manifest.parquet\ncustomerid → dev/test label\n(no features, no churn column)"]
    B -->|"features/build.py\nSQL views, ALL 7,043 customers,\nno validation call"| D["telco_churn_features.parquet\nfull engineered feature set,\ndev + test not yet separated"]
    C --> E["models/train/\nfile-only for training data —\nmerges by customerid, keeps dev rows"]
    D --> E
```

`V` is no longer a dead end as of Phase 8: `split.py` and `features/build.py` both still read their actual data straight from `customers_raw` (`validate.py`'s job is quality gating, not producing a queryable artifact), but `dvc.yaml`'s DAG now makes both stages `dep` on `reports/validation_receipt.json` as an invalidation edge — a failing gate exits 1 before either stage runs, and DVC will not consider `split`/`features` up to date without a passing validation behind them. This is DVC-enforced, not just documented intent. `models/train/` isn't fully DB-free either — `tuning.py` opens its own Postgres connection for Optuna's crash-resilient trial storage (a separate schema, same server as MLflow's backend); only the *training data* (features + split labels) is file-only.

## MLflow Layout — Training Through Promotion (Phases 5–7)

This zooms into what `models/train/` writes to MLflow as it runs — one training cycle, not one MLflow run. It actually produces eight top-level runs (three single-candidate fits, a comparison run, a selection run, one run that keeps growing through tuning/calibration/threshold/registration, and two sealed-test siblings), plus up to fifty more nested inside that growing run, one per Optuna trial. The underlying rule: where something is logged is decided by who needs to find it again later, not by which phase produced it.

```mermaid
flowchart TD
    A["models/train/candidates.py\ndummy_prior, logreg_cv, lgbm_default\n(one run per candidate, CV on telco_churn_dev)"] -->|"comparison.py"| B["model_comparison run (sibling)\ncomparison/: comparison_table.csv,\nbootstrap_delta_dist.png, pr_curves.png\ndiagnostics/: segment_fairness(_delta).csv,\nsegment_robustness(_delta).csv, fixed_recall_profile.csv"]
    B -->|"feature_audit.py"| C["feature_audit run (sibling)\nselection/: committed_features.txt, high_shap_dropouts.txt,\nshap_importance_audit.csv, permutation_importance_table.csv,\nper_fold_stability.csv, bootstrap_delta_dist.png, pr_curves.png"]
    C -->|"log_model.py\nOptuna-tuned LightGBM (tuning.py logs\nnested trial_NNN runs, up to 50)"| D["tuning_study run (root)\nfeature_space.txt, feature_columns.txt,\npreprocessing.pkl, training_manifest.json, tuning/\nmodel (LoggedModel — unregistered)"]
    D -->|"calibrate.py fits + logs;\nregister.register_challenger mints"| E["tuning_study/calibration/\ncalibration_summary.json, golden_predictions.json,\ndev_oof_predictions.parquet, figures/\ncalibrated_model (LoggedModel) registered as\ntelco-churn-pipeline — tags: training_data_scope=dev,\nlogged_model_id, promotion_status=pending\nalias: challenger"]
    E -->|"threshold.py\n(+ folded-in dev-OOF screen)"| F["tuning_study/threshold/\nthreshold.json, threshold_validation.json,\nev_curve.parquet, dev_oof_diagnostics.json, figures/"]
    F -->|"evaluate.py"| G["evaluation run (sibling)\nmetrics.json, economics.json,\npromotion_decision.json (gate.py decide_promotion),\ntest_predictions.parquet, figures/\nreports/eval_receipt.json (register.py's\nfirst-pass bootstrap pointer)"]
    G -->|"error_analysis.py"| H["error_analysis run (sibling)\nerror_analysis.json, shap_values.parquet, figures/\nslices/: error_concentration.json\nreports/error_analysis_receipt.json"]
    H -->|"models/review.py\n(gate.record_review)"| RV["promotion_review.json on the eval run\nappend-only entries: verdict, notes,\napprover, reviewed_at"]
    RV -->|"register.py\nMlflowClient only — opens no run,\ncomputes no metrics"| I{"reads promotion_decision.json +\npromotion_review.json + error_analysis.json;\ntags version: eval_run_id, test_pr_auc, test_recall,\ntest_brier, test_calibration_slope, error_analysis_run_id\n(from tag, else the two receipts above, on this\nversion's first pass); re-checks gate + review +\nsmoke check + golden parity"}
    I -->|"pass"| J["tuning_study/registration/\ndrift_reference.json, model_card.json\nversion: promotion_status → promoted,\npromotion_log entry appended\nalias champion → this version"]
    I -->|"fail (human reject / gate /\nsmoke-check / post-flip parity)"| K["version: promotion_status → rejected\n(reason recorded in version description)\nchampion alias untouched, or rolled back"]
```

**The registry version's state travels through tags, not just an alias.** Each version starts `promotion_status: pending` — the safe default, since a crash mid-cycle should never look like a pass or a legitimate rollback target. `register.py` is the only step that writes a model-version tag: it resolves the evaluation/error-analysis identity from the tag if a prior pass already wrote it, else from `evaluate.py`'s/`error_analysis.py`'s receipts, tags it, re-verifies the gate and the human review verdict, and flips the status to `promoted` or `rejected` — which is why a rollback should always target the highest version tagged `promoted`, never "the previous version number": a rejected or crashed cycle leaves behind a version that otherwise looks the same.

## Shadow/Canary Serving — `serving/predict.py` (Phase 9)

`register.py`'s mint-to-flip gap (`register_challenger` mints `challenger`; a
separate, later call flips `champion` on a pass) leaves both aliases
simultaneously resolvable for as long as evaluation and review take. That gap
is the precondition shadow/canary needs, and `serving/predict.py` is where
the mechanism actually lives — continuously available to any traffic that
hits scoring whenever a `challenger` exists, independent of whether a retrain
cycle is in progress.

```mermaid
flowchart TD
    A["score_request(X, customer_ids, runtime, cfg)"] --> B["Always score champion\n(predict_proba, vectorized)"]
    A --> C{"challenger present AND\n(shadow.enabled OR canary.enabled)?"}
    C -->|no| D["served = champion\nno challenger call at all"]
    C -->|yes| E["Also score challenger\n(predict_proba, vectorized,\nonly the evaluated rows)"]
    E --> F{"canary.enabled?"}
    F -->|no| G["canary_mask = all False"]
    F -->|yes| H["canary_mask[i] =\nsha256(customerid_i + ':' + challenger_version) % 10_000\n< fraction * 10_000\n(missing customerid -> always False)"]
    G --> I["evaluated_mask = shadow.enabled\n(canary never routes)"]
    H --> J["evaluated_mask = canary_mask OR shadow.enabled"]
    I --> K["For each evaluated row:\nlog shadow_canary_dual_score\n(both scores, both versions, served, mode)\nobserve champion_challenger_score_distance{mode}\ninc champion_challenger_agreement_total{outcome, mode}"]
    J --> K
    K --> L{"canary_mask[i]?"}
    L -->|"yes (canary routed)"| M["served = challenger for row i"]
    L -->|"no"| N["served = champion for row i\n(shadow only ever logs)"]
    D --> O
    M --> O
    N --> O
    O["Every row, evaluated or not:\ninc predictions_total{scoring_path, model_version}\nobserve predicted_probability{model_version}\ninc predictions_above_threshold_total{model_version}\n(iff served_proba >= served_threshold)"]
```

**Five Prometheus metrics come out of this one function, not two.** Only
`predictions_total` carries a routing-outcome label — `predicted_probability`
and `predictions_above_threshold_total` are labelled by `model_version`
alone. `predictions_total`'s own `scoring_path` label is self-describing
rather than a bare reuse of the dual-score entry's own `mode`
(`"shadow"`/`"canary"`): `"champion_only"` for a row never evaluated against
the challenger at all, `"champion_with_shadow"` for a row the challenger was
also scored against but didn't serve, `"challenger_canary"` for a row the
challenger actually served. (`mode="shadow"` alone would read as "the shadow
model served it" — backwards, since shadow mode's whole point is that the
champion always serves while the challenger is only evaluated alongside.) So
a canary configured at `fraction=0.1` can be confirmed as actually landing
near 10 % of traffic by reading
`predictions_total{scoring_path="challenger_canary"}` against the total,
rather than just trusting the config. The other two
(`champion_challenger_score_distance`, a histogram of
`abs(challenger_score - champion_score)`, and
`champion_challenger_agreement_total`, a counter of whether champion and
challenger landed on the same side of their own respective thresholds) are
observed only for evaluated rows, labelled `mode` and (for agreement)
`outcome` (`"agree"`/`"disagree"` — not a bare `true`/`false`, which would
lose its meaning the moment someone reads the label without the metric's
own HELP text). Named `champion_challenger_*`, not `shadow_canary_*`: the
comparison is always champion vs. challenger, and `mode` only tags *why*
this row was dual-scored — a `shadow_canary_` prefix sitting directly in
front of "score_distance"/"agreement" reads, out of context, as "distance
between shadow and canary" rather than naming the actual subject. None of
the five carry a `model_version` label on the shadow/canary-specific pair —
the champion/challenger comparison already spans exactly two versions, one
per side. `serving/app.py`'s `GET /metrics`
exposes all five with no extra wiring: `prometheus_fastapi_instrumentator`
exposes the whole default registry `prometheus_client` registered these
five onto at import time, not just the metrics it instrumented itself.

**Two independent, config-gated toggles (`configs/serving/api.yaml`:
`shadow.enabled`, `canary.enabled` + `canary.fraction`), both off by
default.** Shadow scores every request against the challenger for
comparison but never serves its opinion — zero routing risk, pure
observability. Canary actually routes a consistent-hash slice of traffic to
the challenger — bounded routing risk, needed because some failure modes
only surface once a model's output has real consequences. They solve
different problems and are built as separate mechanisms, not one flag with
two readings; both can be on at once, in which case a row outside the canary
bucket is still dual-scored (because shadow says so) but served by champion.

**Canary bucketing is consistent hashing on `customerid`, salted with the
challenger's own `model_version`
(`sha256(f"{customerid}:{challenger_version}") % 10_000 < fraction *
10_000`), never per-request randomness** — the same customer lands in the
same bucket for the life of one challenger's canary window instead of
flapping between models call to call. The salt is what makes that window
one *fresh, independent* customer slice rather than the identical subset on
every canary this project ever runs: since the challenger's version is part
of the hash key, a customer's bucket assignment reshuffles the next time
`challenger` moves to a different version, even though it stays fixed for
as long as the current version is under test. A request with no
`customerid` (a prospect scored in manual/what-if mode) is never
canary-routed: no targeting key means the default variant (champion), the
same rule feature-flag platforms (LaunchDarkly, OpenFeature) use.

**`resolve_champion_model`/`resolve_challenger_model` are the single
indirection point** — no call site anywhere resolves `models:/…@champion` or
`@challenger` directly. Both poll on a TTL, diff the target version's
MLflow-logged dependencies against what's installed
(`environment_parity.py::diff_environment`) before hot-reloading, and — for
the champion — fail-open on a refresh problem (keep serving the last-known-
good bundle, log a warning) but fail-closed on the very first load (no
champion in memory yet, nothing safe to fall back to). A challenger problem
is never a reason to disrupt champion traffic either way. The SHAP explainer
and the threshold policy travel inside the same `ModelBundle` a reload swaps
in one piece, so neither can end up paired with a different model version's
score.

**Evidence below is *not* from `tests/unit/test_predict.py` itself — that
suite deliberately fits its champion/challenger pair on synthetic data for
speed and determinism (see its module docstring). Regenerating this
snippet's realism requires the thing the unit tests are built to avoid:
real customers.** So it is produced by a small one-off script, run the same
way the mechanism test is structured (two real
`CalibratedClassifierCV(LGBMClassifier)` pipelines registered as
`champion`/`challenger` against a tmp-scoped MLflow store) but fit on real
`tenure`/`monthlycharges` values and real churn labels from
`data.split.partition`'s **dev** side only, loaded straight from
`datasets/processed/split_manifest.parquet` — never the sealed test side.
Six dev-partition customers, one per tenure sextile, were then scored
through the real `score_request` with `shadow.enabled=true` (champion `t*
= 0.4`, challenger `t* = 0.6`, versions 1/2):

```json
{"customerid": "1087-GRUYI", "champion_version": "1", "champion_score": 0.052926906761486600, "challenger_version": "2", "challenger_score": 0.052268186900639425, "served": "champion", "mode": "shadow", "event": "shadow_canary_dual_score"}
{"customerid": "4189-NAKJS", "champion_version": "1", "champion_score": 0.085580031785222140, "challenger_version": "2", "challenger_score": 0.084804168943478090, "served": "champion", "mode": "shadow", "event": "shadow_canary_dual_score"}
{"customerid": "7321-ZNSLA", "champion_version": "1", "champion_score": 0.381320170055880400, "challenger_version": "2", "challenger_score": 0.381786367680829000, "served": "champion", "mode": "shadow", "event": "shadow_canary_dual_score"}
{"customerid": "6968-GMKPR", "champion_version": "1", "champion_score": 0.131177544000207730, "challenger_version": "2", "challenger_score": 0.130389959439198650, "served": "champion", "mode": "shadow", "event": "shadow_canary_dual_score"}
{"customerid": "0971-QIFJK", "champion_version": "1", "champion_score": 0.044880156423729396, "challenger_version": "2", "challenger_score": 0.044270441986991246, "served": "champion", "mode": "shadow", "event": "shadow_canary_dual_score"}
{"customerid": "8909-BOLNL", "champion_version": "1", "champion_score": 0.382721088645554760, "challenger_version": "2", "challenger_score": 0.383196816141353000, "served": "champion", "mode": "shadow", "event": "shadow_canary_dual_score"}
```

Every row carries `served: "champion"` (shadow never routes) and both
model versions/scores side by side — exactly what an offline job needs to
later compute agreement rate and score-distance between champion and
challenger. This particular batch happens to land all six on the same side
of both thresholds, which is exactly what the Prometheus counters below
confirm rather than assert on faith:

```text
champion_challenger_score_distance_count{mode="shadow"}                    6
champion_challenger_score_distance_sum{mode="shadow"}                       0.003773806821085267
champion_challenger_agreement_total{outcome="agree", mode="shadow"}         6
predictions_total{scoring_path="champion_with_shadow", model_version="1"}   6
predicted_probability_count{model_version="1"}                              6
predicted_probability_sum{model_version="1"}                                1.0786058976720811
```

Switching to `canary.enabled=true, fraction=1.0` against the same runtime
routes every row to the challenger instead (`served: "challenger"`,
`mode: "canary"`); `fraction=0.0` routes none — both re-verified directly
against `tests/unit/test_predict.py::test_score_request_canary_fraction_one_routes_everyone_to_challenger`
and `::test_score_request_canary_fraction_zero_routes_everyone_to_champion`,
which pin this behavior on their own synthetic fixture (23/23 tests in that
file pass on the current code). Phase 10's `require_traffic_validation`
seam is where a real canary observation window's *result* would eventually
gate a promotion — this project's static dataset has no live traffic for
that window to observe, so that decision layer is honestly stubbed to
no-op rather than faked with replayed rows standing in for a real verdict.
The routing/logging/metrics *mechanism* documented here is real and tested;
a live production shadow/canary result is not, and must never be implied
as one.

## Live Customer Lookup — `customers_crm` (Phase 9)

`GET /customer/{id}` and `/predict/batch`'s ID-resolution path need a "live customer" source. The obvious wrong answer is to query `customers_raw` — the exact table the model trained on — which would make a Streamlit "Fetch customer" click silently present frozen training data as if it were current. `customers_crm` is the fix: a second table, generated from `customers_raw` but never read by anything upstream of serving.

```mermaid
flowchart TD
    A["Raw CSV\ndatasets/raw/ — read-only,\nsource of truth"] -->|"ingest.py\n(once)"| B[("Postgres\ncustomers_raw\nfrozen training snapshot")]
    B -->|"crm_data.py::generate_crm_rows\nseeded (random_state: 42), deterministic:\ntenure +1-6mo, contract upgrade-only,\ntotalcharges accrues, else unchanged"| C[("Postgres\ncustomers_crm\nsimulated current state")]
    C -->|"customer_lookup.py::lookup_customers"| D["GET /customer/{id}\nPOST /predict/batch ID resolution"]
    D -->|"CustomerLookupResponse\n{features, crm_snapshot_at}"| E["Streamlit UI\nScore a Customer tab"]
    B -.->|"never read at serving time"| D
```

**Why a derived table rather than reusing `customers_raw` or a hand-authored fixture.** `customers_raw` is read-only and DVC-hashed (`CLAUDE.md` § Data Handling) — mutating it to look "current" would invalidate the whole `ingest → ... → error_analysis` chain for a UI transparency fix, and would still be dishonest (it's the same rows the model was evaluated against, not new data). A hand-authored fixture would be an unrelated, disconnected dataset — the customerids a Batch Prediction or Score a Customer demo resolves would stop being the same customers the model card and evaluation report discuss. `customers_crm` keeps the same 7,043 customers and nudges them forward in time by a plausible amount instead, so a served prediction is still about a customer the reader can trace back to the training data, just at a later, simulated point in their account lifecycle.

**What is, and isn't, recorded.** `crm_snapshot_at` (stamped fresh at generation time) travels with every `GET /customer/{id}` response, so a caller can tell how stale the "live" row is. A durable, queryable record of *which* `customers_crm` row a given `/predict` or `/predict/batch` call actually scored is `serving/prediction_log.py`'s `prediction_log` table (Phase 10a-i): every prediction gets a durable row, `feature_snapshot` included, without `customers_crm` itself changing shape.

**Regeneration.** `make crm-data`, run after `db-up` + ingest (the table starts empty on a fresh Postgres volume — its DDL auto-creates via `docker-entrypoint-initdb.d`, but nothing populates it until this runs). Safe to re-run: the fixed seed reproduces the same nudges every time, so it's a truncate-and-reload, not an upsert — the same idempotency `data/ingest.py` already has for `customers_raw`.

## Emergency Rollback — `register.py::rollback_champion`

Two independent triggers write to the same append-only `promotion_log` tag on the registered model, so "what was champion before this one" is always a direct index into one history, never a tag-scan-and-`max()` reconstruction.

```mermaid
flowchart TD
    A["Trigger"] --> B1["Automated: post-flip golden-parity\ncheck fails inside run_registration_step"]
    A --> B2["Manual: on-call runs\nrollback_champion() directly"]
    B1 --> C["rollback_champion(name, target_version=None)"]
    B2 --> D["rollback_champion(name, target_version=<v>)"]
    C --> E{"Search all versions tagged\npromotion_status=promoted,\nexcluding the current champion"}
    D --> F{"Is <v> tagged\npromotion_status=promoted?"}
    E -->|"none found"| G["Raise — nothing earlier\nto roll back to"]
    E -->|"found"| H["Target = highest version number\namong the remaining promoted set"]
    F -->|"no"| I["Raise — refuse to roll back to\na version never validated as champion"]
    F -->|"yes"| H
    H --> J["promote_to_alias(champion, target)"]
    J --> K["Append {action: rolled_back, version,\nrolled_back_from, at} to promotion_log"]
```

**Never version arithmetic (`N-1`).** A rejected or crashed evaluation cycle can leave a higher-numbered version in the registry with no `champion` alias and no `promoted` tag — indistinguishable from a legitimate former champion by version number alone. Selecting on the `promotion_status` tag is the one query that can't land on that version by accident. `champion_history()` reads the same `promotion_log` tag back as an ordered list, so "what was champion two promotions ago" is `history[-3]`, not a re-derivation.

---

## Modelling Stages

Mechanism only — the reasoning behind each choice (why LightGBM over `LogisticRegressionCV`, why sigmoid calibration, why these guardrails) lives in `ANALYSIS.md`, linked per row rather than repeated here.

**Family and feature selection are periodic, human-reviewed decisions frozen into code — not steps the routine pipeline re-runs.** Both are made in a notebook, on a real trigger (a new candidate family, a drift signal, a scheduled review), never on every retrain; the decision is then frozen as a literal constant the automated pipeline just trains against. Full mechanism, the human-review loop, and why this doesn't drift silently → [docs/architecture.md § Model Family & Feature Selection](#model-family--feature-selection--human-reviewed-frozen-into-code-phase-5).

| Stage | Mechanism |
|---|---|
| **Family selection** | Three candidates cross-validated on the dev partition — `DummyClassifier(strategy='prior')` (leakage safeguard + BSS floor), `LogisticRegressionCV` baseline, LightGBM challenger. PR-AUC is the sole selection criterion. → [ANALYSIS.md §4a](../ANALYSIS.md#4a-model-selection) |
| **Feature selection** | Permutation-importance ablation over 100 CV folds, stability-voted rather than decided from a single all-dev fit. → [ANALYSIS.md §4b](../ANALYSIS.md#4b-feature-selection-concluded-ablation--importance-diagnostic) |
| **Hyperparameter tuning** | Optuna, TPE sampler, 50 trials; final pick is the 1-SE-rule value, not the raw-best trial. → [ANALYSIS.md §4c](../ANALYSIS.md#4c-hyperparameter-tuning-optuna) |
| **Calibration** | `CalibratedClassifierCV` (sigmoid) — fit and logged only; `calibrate.py` never registers a model on its own. → [ANALYSIS.md §5](../ANALYSIS.md#5-probability-calibration) |
| **Threshold** | Closed-form `t* = c / (r × LTV)` from `configs/costs.yaml`, derived out-of-fold with no leakage into the sealed test set. → [ANALYSIS.md §6](../ANALYSIS.md#6-business-impact--threshold-selection) |
| **Promotion gate** | PR-AUC drives selection; three veto-only guardrails (recall at `t*`, Brier, calibration slope) can reject a candidate but never promote one. → [ANALYSIS.md §0](../ANALYSIS.md#success-criterion--the-promotion-gate) |
| **Registration** | `register.py` is the sole registry-write entry point — mints the challenger, tags it `pending`, and flips (or rejects) the `champion` alias only after both the automated gate and a human review pass. → [ANALYSIS.md §8](../ANALYSIS.md#8-model-registration--promotion) |

Full modelling narrative, every number, every deviation from the archived exploratory pass → [ANALYSIS.md](../ANALYSIS.md).

## Pipeline Versioning (DVC)

`dvc.yaml` wraps the file-producing half of the pipeline as a reproducible, content-hashed DAG — the registry-mutating half (minting/tagging/flipping a model version) structurally cannot live in it, since a registry alias isn't a file DVC can hash.

| Component | What it does |
|---|---|
| **9-stage DAG** (`dvc.yaml`) | `ingest → validate → split → features → train → calibrate → threshold → evaluate → error_analysis`, each stage's `deps`/`params`/`outs` explicitly declared. `dvc repro <stage>` only re-runs that stage and its upstream chain when something it actually depends on changed — never the whole DAG unconditionally. |
| **Registry writes stay outside the DAG** | `register.py` (mint challenger, tag, flip/reject champion) runs as its own step, never a `dvc.yaml` stage. A full training cycle is therefore two `dvc repro` calls bracketing one registration step, not one — [CONTRIBUTING.md § Running the full stack locally](../CONTRIBUTING.md#running-the-full-stack-locally) run through it. |
| **Receipts bridge the two worlds** | Six small JSON receipts (`reports/*_receipt.json`) stand in for side effects DVC can't hash directly — a Postgres load, an MLflow run — giving `validate`/`train`/`calibrate`/`eval`/`error_analysis` a real dependency edge on "did the step before me actually succeed," not just "did its script exit 0." |
| **Local cache only** | No DVC remote yet — `.dvc/cache` is local-only through Phase 11; Phase 12 adds an S3 remote for shared caching, not a migration off the current setup. |

```bash
dvc repro <stage>   # re-run one stage and its upstream chain
dvc dag             # print the pipeline DAG
dvc metrics show    # diff metrics.json / params across commits
```

Run it yourself → [CONTRIBUTING.md § Running the full stack locally](../CONTRIBUTING.md#running-the-full-stack-locally). Full architecture → [docs/architecture.md](architecture.md).

## Serving Components

The champion (`telco-churn-pipeline@champion` in the MLflow registry) is served behind a FastAPI app, with a Streamlit UI as a thin client — both run as their own Docker images. **Live on AWS** (Phase 12a) at **[telco-churn.duckdns.org](https://telco-churn.duckdns.org)**, no setup required; for local development, the same stack comes up with one `docker compose up` ([CONTRIBUTING.md](../CONTRIBUTING.md#running-the-full-stack-locally)).

| Component | What it does |
|---|---|
| **FastAPI** (`serving/app.py`, `serving/predict.py`) | Loads the champion at startup and hot-reloads it on a TTL poll — a promotion is just an MLflow alias flip, so a new champion goes live without a redeploy. Exposes `POST /predict`, `POST /predict/batch`, `GET /customer/{id}`, `GET /health`/`GET /ready` (liveness vs. readiness), and `GET /metrics` (Prometheus). Runs on `uvicorn` inside `docker/api/Dockerfile`, a multi-stage build from the project's own `uv.lock`. |
| **Contact policy** | `POST /predict/batch` doesn't just score every row — it ranks them by expected value and caps who gets contacted at a configurable `contact_capacity`/`campaign_budget`, so the response reflects an operationally realistic campaign, not an unlimited-budget fantasy. |
| **Shadow/canary rollout** | Config-gated, off by default: whenever a `challenger` model exists in the registry, it can be dual-scored against every request for comparison (**shadow**, zero routing risk) or actually serve a consistent-hash slice of live traffic (**canary**). Mechanism, Prometheus metrics, and a real evidence log → [docs/architecture.md § Shadow/Canary Serving](#shadowcanary-serving--servingpredictpy-phase-9). |
| **Streamlit UI** (`ui/streamlit_app.py`) | **Score a Customer** tab: look up a real customer from `customers_crm` (a seeded "current state" derivation of the training data, never the frozen snapshot itself) or manually enter one for a what-if scenario. Mechanism, rationale, and what's/isn't durably recorded → [docs/architecture.md § Live Customer Lookup](#live-customer-lookup--customers_crm-phase-9). **Batch Prediction** tab: score a CSV upload, or the bundled sample in one click — [`examples/sample_batch_predictions.csv`](../examples/sample_batch_predictions.csv) demonstrates all three `/predict/batch` item shapes in one upload (ID-only, full inline, ID-plus-override). **Model Info** tab: read the champion's model card. All three drive the same FastAPI endpoints a real client would call. Runs from `docker/ui/Dockerfile`, same build pattern as the API image. |
| **Live deployment** (Phase 12a) | A single EC2 box (`t3.small`) behind Caddy — automatic TLS via Let's Encrypt, HTTPS-only, no laptop in the loop. Postgres runs on RDS and MLflow artifacts on S3 instead of local containers/volumes, but it's the same `docker compose` shape underneath (`infra/deploy/compose.prod.yml`), just pointed at managed backends. A fourth service, a persistent **reviewer-facing MLflow UI**, is reachable at [telco-churn.duckdns.org/mlflow](https://telco-churn.duckdns.org/mlflow) (login: `reviewer` / `telco-reviewer-2026`) — the actual registry and runs behind every number in the README, browsable without cloning the repo. |
| **Docker Compose** (local dev) | `docker-compose.yml` wires four services — `postgres`, `mlflow`, `fastapi`, `streamlit` — so `docker compose up -d --build` brings up the entire stack in one command on your own machine, with `fastapi`/`streamlit` waiting on `postgres`/`mlflow` via `depends_on`. |

Try the live deployment above — the Batch Prediction tab's **Score the sample** button needs no file; to prepare your own, **[download the sample CSV](https://raw.githubusercontent.com/Ampofowaa/TelcoChurn_PortfolioProject/main/examples/sample_batch_predictions.csv)** as a template — or run it yourself locally → [CONTRIBUTING.md](../CONTRIBUTING.md#running-the-full-stack-locally). The raw API requires an `X-API-Key` on its three data routes (`/predict`, `/predict/batch`, `/customer/{id}`) — see the schema and try it interactively, no key needed, at [telco-churn.duckdns.org/docs](https://telco-churn.duckdns.org/docs).
