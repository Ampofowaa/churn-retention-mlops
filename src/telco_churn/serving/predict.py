"""Loads the champion model from MLflow at process startup, keeps it fresh via
a hot-reload poll, and scores requests — including the shadow/canary
mechanism and SHAP explanation.

**Hot-reload over redeploy.** A promotion is just an MLflow alias flip;
nothing about it requires restarting this process. `resolve_serving_model`
polls on a short TTL for whether `champion`/`challenger` have moved and, on a
change, diffs the new version's MLflow-logged dependencies against what's
actually installed (`environment_parity.py::diff_environment`) before
swapping it in. Compatible -> hot-reload in place. Diverged -> refuse, log
loudly, keep serving the old version. `load_serving_state_with_retry` is the
one exception: the *initial* load (no model in memory yet) retries with
backoff on any failure rather than degrading, since there is no
"last-known-good" to fall back to yet — fail-open on refresh, fail-closed on
initial load.

**Shadow vs. canary — two independent, config-gated toggles, not one flag
with two readings.** Shadow scores every request against the challenger for
comparison but never serves its opinion. Canary actually routes a
consistent-hash slice of traffic (by `customerid`) to the challenger. A
request with no `customerid` (a prospect scored in manual/what-if mode) is
never canary-routed — no customerid means no stable bucket, so it always
gets the default variant (champion), the same "no targeting key -> default
variant" rule feature-flag platforms (LaunchDarkly, OpenFeature) use.
`score_request` is the one place this routing happens, and it is fully
vectorized (never a per-row Python loop) so `predict_batch` can call it once
on an assembled DataFrame the same way `predict_single` calls it on a 1-row
frame.

**The SHAP explainer and the threshold policy both reload in lockstep with
the model, inside the same `build_model_bundle` call that swaps a champion
in — never cached for the process's lifetime.** A `TreeExplainer` genuinely
cached across a promotion would keep explaining the old booster after the
probability next to it already reflects the new model; a threshold cached
across a promotion would judge a new model's score against a `t*` derived
for a different version. `ModelBundle` bundles model + preprocessor +
booster + explainer + threshold policy as one atomic, swap-in-one-piece unit
for exactly this reason.

**Feature assembly is serving's own job, not features/build.py's.**
`build_feature_df` (features/build.py) assumes `charge_per_service` has
already been computed by the `sql/features/charge_per_service.sql` view —
true during training (Postgres computes it before Python ever sees the row),
false here (a prospect or what-if request was never in Postgres to begin
with). `_assemble_model_input` instead calls
`features/generate.py::compute_charge_per_service` — the existing,
already-tested pandas mirror of that SQL view, built for exactly this
"no database connection" case — directly on the request's raw fields.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from prometheus_client import Counter, Gauge, Histogram

from telco_churn.features.generate import compute_charge_per_service
from telco_churn.models.artifacts import (
    committed_features_from_manifest,
    load_training_manifest,
)
from telco_churn.models.environment_parity import diff_environment
from telco_churn.models.explain import local_explanations
from telco_churn.models.policy_config import (
    CostScenario,
    load_threshold_payload,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.models.registry_alias import (
    resolve_challenger_version,
    resolve_champion_version,
)
from telco_churn.models.shap_values import (
    build_tree_explainer,
    explain_with_explainer,
    unwrap_calibrated_pipeline,
)
from telco_churn.serving.schemas import (
    Explanation,
    FeatureContribution,
    PredictRequest,
    PredictResponse,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import resolve_tracking_uri

__all__ = [
    "ModelBundle",
    "ServingRuntime",
    "ScoredBatch",
    "build_model_bundle",
    "resolve_champion_model",
    "resolve_challenger_model",
    "resolve_serving_model",
    "load_serving_state_with_retry",
    "score_request",
    "predict_single",
    "predict_batch",
]

logger = get_logger(__name__)

# threshold/threshold.json's shipped operating point — the same "base"
# literal evaluate.py's own policy-context loader reads.
_BASE_SCENARIO = "base"

# Canary bucketing resolution — sha256(f"{customerid}:{salt}") %
# _CANARY_BUCKETS < fraction * _CANARY_BUCKETS. 10_000 buckets gives
# two-decimal-place fraction precision (e.g. fraction=0.01 is exactly 100
# buckets).
_CANARY_BUCKETS = 10_000

# Module-level so a single registration survives across requests/hot-reloads
# within the process — prometheus_client registers these to the default
# registry on import, ready for app.py's GET /metrics to expose once built.
# No model_version label on either: the champion/challenger pairing already
# spans exactly two versions, one per side of the comparison.
#
# Named champion_challenger_*, not shadow_canary_* — the thing being
# compared is always champion vs. challenger; mode only tags *why* this
# particular row was dual-scored (shadow watches every request, canary is
# the routed subset whose challenger score actually went live). A
# shadow_canary_ prefix directly in front of "score_distance"/"agreement"
# reads, out of context, as "distance between shadow and canary" rather than
# "between champion and challenger" — the actual subject belongs in the
# name, not only in the HELP text.
_CHAMPION_CHALLENGER_SCORE_DISTANCE = Histogram(
    "champion_challenger_score_distance",
    "abs(challenger_score - champion_score) per dual-scored shadow/canary request.",
    labelnames=("mode",),
)
# Named without the _total suffix: prometheus_client's Counter appends it
# automatically (OpenMetrics convention) — the exposed metric name still
# comes out as champion_challenger_agreement_total, just not
# double-suffixed. outcome="agree"/"disagree", not agree="true"/"false":
# the metric name already contains "agreement", so a label key of "agree"
# repeated the word, and a bare true/false loses its meaning the moment
# someone reads the label without the HELP text alongside it.
_CHAMPION_CHALLENGER_AGREEMENT_TOTAL = Counter(
    "champion_challenger_agreement",
    "Whether champion and challenger landed on the same side of their "
    "respective thresholds for a dual-scored shadow/canary request.",
    labelnames=("outcome", "mode"),
)

# The three general-purpose serving metrics (as opposed to the two
# shadow/canary-only ones above). scoring_path names *who actually served
# this row*, self-describing rather than reusing the dual-score entry's own
# "mode" ("shadow"/"canary") verbatim — "shadow" on this label would read as
# "the shadow model served it", when shadow mode's entire point is that the
# champion always serves while the challenger is only evaluated alongside.
# champion_with_shadow/challenger_canary make that explicit, and let a canary
# configured at fraction=0.1 be confirmed as actually landing near 10% of
# traffic rather than just trusting the config. app.py's GET /metrics exposes
# these with no extra code of its own — prometheus_fastapi_instrumentator's
# Instrumentator exposes the whole default registry, not just metrics it
# registered itself.
_PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served, labelled by which model actually served the "
    "row and how it got there, and the model version that served it.",
    labelnames=("scoring_path", "model_version"),
)
# Explicit buckets: the default buckets (0.005s-10s) are shaped for web
# request latency, not a [0, 1] probability — 0.25-0.5 (where t* usually
# sits) would collapse into one bucket, the exact region resolution matters
# most for drift/PSI-style queries against this histogram.
_PREDICTED_PROBABILITY = Histogram(
    "predicted_probability",
    "Distribution of served prediction probabilities.",
    labelnames=("model_version",),
    buckets=(
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ),
)
_PREDICTIONS_ABOVE_THRESHOLD_TOTAL = Counter(
    "predictions_above_threshold_total",
    "Total predictions whose served probability crossed the operating "
    "threshold (t*) that was in effect for them.",
    labelnames=("model_version",),
)

# Isolates the model's own predict_proba cost from the surrounding request
# (DB customer-lookup, serialization, FastAPI overhead) that
# prometheus_fastapi_instrumentator's total-request-latency histogram already
# covers — a p99 spike on the request histogram alone can't tell you whether
# the model got slow or a dependency did; this can. labelled "champion"/
# "challenger" (which bundle actually ran predict_proba), never by
# scoring_path — that's a routing outcome, not who computed the score. One
# predict_proba call is vectorized over the whole batch (never a per-row
# loop), so one observation covers however many rows were in that call, not
# one observation per row. Explicit buckets: the default (0.005s-10s) is
# shaped for whole-HTTP-request latency; a single-row predict_proba on this
# model is expected sub-10ms, so the defaults would pile almost every
# observation into the first bucket or two.
_MODEL_INFERENCE_SECONDS = Histogram(
    "model_inference_seconds",
    "Wall-clock time of one predict_proba call, per model bundle.",
    labelnames=("model",),
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
# SHAP explanation is a known order of magnitude slower than predict_proba
# itself (TreeExplainer walks the whole booster) and only runs for
# include_explanation=True requests — a separate histogram, not a label on
# the one above, since it wraps a different call entirely. Explicit buckets
# shifted up relative to model_inference_seconds' own, to match that
# order-of-magnitude gap rather than reusing the same scale.
_EXPLANATION_SECONDS = Histogram(
    "explanation_seconds",
    "Wall-clock time of one local SHAP explanation (explain_with_explainer).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Pure mirrors of values build_model_bundle already resolved — never a
# recomputation, so neither gauge can diverge from what the pipeline
# actually used for this model_version.
_ACTIVE_THRESHOLD = Gauge(
    "active_threshold",
    "t* currently in effect for this model_version's base cost scenario — "
    "the same value served_threshold/decision are computed from.",
    labelnames=("model_version",),
)
# Mirrors configs/costs.yaml's base-scenario constants, not a live EV
# computation — economics.py already owns EV (with bootstrap CIs and
# costs_config_hash attribution); duplicating that formula here would be a
# second implementation of the same rule. Grafana multiplies this against
# the already-live predictions_above_threshold_total/contact counters at
# query time instead.
_COST_SCENARIO_CONSTANTS = Gauge(
    "cost_scenario_constants",
    "Base-scenario cost constant (ltv/cost/retention_rate), mirrored "
    "unchanged from the resolved CostScenario at model load — set once per "
    "model_version, never recomputed.",
    labelnames=("constant", "model_version"),
)

# A UNIX timestamp, not a duration — prometheus_client Gauges only hold
# whatever was last .set(), so a "seconds since" value would freeze between
# polls instead of climbing; the idiomatic pattern is to expose the last
# success's own timestamp and let the alert rule compute
# time() - champion_last_successful_poll_timestamp_seconds. Reset on every
# resolve_serving_model tick that returns at all — both
# resolve_champion_model/resolve_challenger_model already catch and log
# their own refresh failures internally (fail-open), so this only stops
# advancing if the poll loop itself has died, the one failure mode fail-open
# is built to hide from everything else.
_CHAMPION_LAST_SUCCESSFUL_POLL_TIMESTAMP_SECONDS = Gauge(
    "champion_last_successful_poll_timestamp_seconds",
    "Unix timestamp of the last poll tick that completed without raising — "
    "alert on time() - this exceeding a few poll intervals.",
)
# Both already read every poll for the version-change comparison
# resolve_champion_model makes anyway — exposing them costs one gauge
# update each, no extra MLflow calls. Catches a reload that's refused
# (dependency mismatch) or fails outright, which the poll-health gauge
# above can't see on its own (the poll tick still completes normally).
_CHAMPION_VERSION_REGISTRY = Gauge(
    "champion_version_registry",
    "The champion alias's current target version number in the MLflow "
    "registry, as last observed by the poll loop.",
)
_CHAMPION_VERSION_SERVED = Gauge(
    "champion_version_served",
    "The model_version this process is actually serving champion traffic "
    "with right now.",
)
# role distinguishes champion/challenger since both resolve_champion_model
# and resolve_challenger_model hit refused/failed sites of their own;
# outcome mirrors the existing logger.warning event names exactly, just
# promoted to a metric. Cold start (current=None, the very first load) is
# deliberately not counted here — this tracks reloads, i.e. something that
# already had a prior version to compare against, not the initial load.
_MODEL_RELOAD_TOTAL = Counter(
    "model_reload_total",
    "Outcome of a hot-reload attempt triggered by a poll tick that saw the "
    "registry alias move.",
    labelnames=("role", "outcome"),
)


@dataclass(frozen=True)
class ModelBundle:
    """One registered model version, fully loaded and serving-ready.

    Bundles the model with everything derived from its own MLflow run
    (preprocessor/booster, cached SHAP explainer, committed feature list,
    threshold policy) as one atomic unit — resolve_champion_model/
    resolve_challenger_model always swap this whole object in one piece,
    never any of its fields independently, so a refreshed model can never
    end up paired with a threshold or explainer derived for a different
    version.
    """

    model: Any
    preprocessor: Any
    booster: Any
    explainer: Any
    model_version: str
    run_id: str
    logged_model_id: str
    committed_features: list[str]
    scenarios: dict[str, CostScenario]
    thresholds: dict[str, float]
    base_scenario_name: str
    loaded_at: datetime


@dataclass(frozen=True)
class ServingRuntime:
    """The full serving state at a point in time: the champion (always
    present once loaded) and the challenger (present only while a
    `challenger` alias is registered — the common case is None)."""

    champion: ModelBundle
    challenger: ModelBundle | None


@dataclass(frozen=True)
class ScoredBatch:
    """score_request's output — one row per input row.

    served_* describes what was actually returned to the caller for that
    row (champion, unless canary routed it to the challenger).
    champion_proba/challenger_proba are the raw per-model scores whenever
    computed (challenger_proba is None unless shadow or canary is active),
    kept separate from served_proba so a caller can always tell "what the
    champion alone would have said" apart from "what was served." In
    canary-only mode (shadow disabled), the challenger is only ever asked to
    score rows that actually fell in the canary bucket — challenger_proba
    carries NaN at every other row's position, since no score was computed
    there to report.
    dual_score_rows is the exact set of structured log entries already
    written for whichever rows the challenger was evaluated against.
    """

    customer_ids: pd.Series
    served_proba: NDArray[np.float64]
    served_source: list[str]
    served_model_version: list[str]
    served_threshold: NDArray[np.float64]
    champion_proba: NDArray[np.float64]
    challenger_proba: NDArray[np.float64] | None
    dual_score_rows: list[dict[str, Any]]


def build_model_bundle(version: str, cfg: DictConfig) -> ModelBundle:
    """Load one explicit registry version and everything derived from its own MLflow run.

    Never resolves by alias — always an explicit version number, the same
    alias-avoidance discipline evaluate.py's own model resolution follows,
    since an alias is a moving pointer under a concurrent promotion.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    model_version_info = client.get_model_version(registered_model_name, version)
    if model_version_info.run_id is None:
        raise RuntimeError(
            f"Registered model version {version!r} of {registered_model_name!r} "
            "has no run_id — cannot resolve its threshold policy or training "
            "manifest."
        )
    run_id: str = model_version_info.run_id
    logged_model_id = model_version_info.tags.get("logged_model_id", "")

    model_uri = f"models:/{registered_model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    preprocessor, booster = unwrap_calibrated_pipeline(model)
    explainer = build_tree_explainer(booster)

    manifest = load_training_manifest(run_id, cfg)
    committed_features = committed_features_from_manifest(manifest)

    policy = load_threshold_payload(
        run_id, cfg, model_version_info.tags.get("threshold_run_id")
    )
    scenarios = resolve_policy_scenarios(policy)
    thresholds = resolve_policy_thresholds_by_scenario(policy)

    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        booster=booster,
        explainer=explainer,
        model_version=version,
        run_id=run_id,
        logged_model_id=logged_model_id,
        committed_features=committed_features,
        scenarios=scenarios,
        thresholds=thresholds,
        base_scenario_name=_BASE_SCENARIO,
        loaded_at=datetime.now(UTC),
    )
    base_scenario = scenarios[_BASE_SCENARIO]
    _ACTIVE_THRESHOLD.labels(model_version=version).set(thresholds[_BASE_SCENARIO])
    _COST_SCENARIO_CONSTANTS.labels(constant="ltv", model_version=version).set(
        base_scenario.ltv
    )
    _COST_SCENARIO_CONSTANTS.labels(constant="cost", model_version=version).set(
        base_scenario.cost
    )
    _COST_SCENARIO_CONSTANTS.labels(
        constant="retention_rate", model_version=version
    ).set(base_scenario.retention_rate)

    logger.info(
        "model_bundle_loaded",
        model_version=version,
        run_id=run_id,
        logged_model_id=logged_model_id,
    )
    return bundle


def _check_hot_reload_compatible(
    registered_model_name: str, target_version: str, cfg: DictConfig
) -> bool:
    """True iff target_version's logged dependencies match what's installed.

    Does not itself catch an MLflow call failing (a genuinely unreachable
    registry) — that propagates deliberately, so the caller's own refresh-vs-
    initial-load exception handling (resolve_champion_model/
    resolve_challenger_model) is the single place that decides fail-open vs.
    fail-closed, rather than this function pre-empting that choice.
    """
    packages = list(cfg.register.environment_packages)
    model_uri = f"models:/{registered_model_name}/{target_version}"
    diff = diff_environment(model_uri, packages)
    return not diff.mismatches


def resolve_champion_model(cfg: DictConfig, current: ModelBundle | None) -> ModelBundle:
    """Poll the `champion` alias and return the bundle that should now be serving.

    current=None is the cold-start path: any failure (MLflow unreachable, no
    champion registered yet) propagates, so the caller
    (load_serving_state_with_retry) can retry with backoff — fail-closed,
    there is no last-known-good to fall back to yet.

    current given is the refresh path: never raises. Same version -> no-op.
    Changed version but a dependency mismatch -> log and keep serving
    `current` (refuse the reload). Any other failure anywhere in this path
    -> log and keep serving `current` — fail-open, a transient control-plane
    blip must degrade to "stale but correct," never to "down."
    """
    if current is None:
        version = resolve_champion_version(cfg)
        if version is None:
            raise RuntimeError(
                "No champion model is registered yet (models:/"
                f"{cfg.mlflow.registered_model_name}@champion has no target) "
                "— cannot perform the initial load."
            )
        _CHAMPION_VERSION_REGISTRY.set(float(version))
        bundle = build_model_bundle(version, cfg)
        _CHAMPION_VERSION_SERVED.set(float(bundle.model_version))
        return bundle

    try:
        version = resolve_champion_version(cfg)
        if version is not None:
            _CHAMPION_VERSION_REGISTRY.set(float(version))
        if version is None or version == current.model_version:
            _CHAMPION_VERSION_SERVED.set(float(current.model_version))
            return current
        registered_model_name = str(cfg.mlflow.registered_model_name)
        if not _check_hot_reload_compatible(registered_model_name, version, cfg):
            logger.warning(
                "champion_hot_reload_refused_dependency_mismatch",
                current_version=current.model_version,
                target_version=version,
            )
            _MODEL_RELOAD_TOTAL.labels(role="champion", outcome="refused").inc()
            _CHAMPION_VERSION_SERVED.set(float(current.model_version))
            return current
        bundle = build_model_bundle(version, cfg)
    except Exception:
        logger.warning(
            "champion_refresh_failed",
            current_version=current.model_version,
            exc_info=True,
        )
        _MODEL_RELOAD_TOTAL.labels(role="champion", outcome="failed").inc()
        _CHAMPION_VERSION_SERVED.set(float(current.model_version))
        return current
    else:
        _MODEL_RELOAD_TOTAL.labels(role="champion", outcome="success").inc()
        _CHAMPION_VERSION_SERVED.set(float(bundle.model_version))
        return bundle


def resolve_challenger_model(
    cfg: DictConfig, current: ModelBundle | None
) -> ModelBundle | None:
    """Poll the `challenger` alias — the common case (no challenger staged)
    is None, never an error, at either cold start or refresh. Any resolution
    failure is treated the same way a refresh failure is: log and keep
    whatever challenger (or lack of one) was already loaded, since a
    challenger problem is never a reason to disrupt champion traffic.
    """
    try:
        version = resolve_challenger_version(cfg)
    except Exception:
        logger.warning("challenger_alias_resolution_failed", exc_info=True)
        _MODEL_RELOAD_TOTAL.labels(role="challenger", outcome="failed").inc()
        return current

    if version is None:
        return None
    if current is not None and version == current.model_version:
        return current

    try:
        registered_model_name = str(cfg.mlflow.registered_model_name)
        if current is not None and not _check_hot_reload_compatible(
            registered_model_name, version, cfg
        ):
            logger.warning(
                "challenger_hot_reload_refused_dependency_mismatch",
                current_version=current.model_version,
                target_version=version,
            )
            _MODEL_RELOAD_TOTAL.labels(role="challenger", outcome="refused").inc()
            return current
        bundle = build_model_bundle(version, cfg)
    except Exception:
        logger.warning("challenger_refresh_failed", exc_info=True)
        _MODEL_RELOAD_TOTAL.labels(role="challenger", outcome="failed").inc()
        return current
    else:
        _MODEL_RELOAD_TOTAL.labels(role="challenger", outcome="success").inc()
        return bundle


def resolve_serving_model(
    cfg: DictConfig, current: ServingRuntime | None
) -> ServingRuntime:
    """One TTL-poll tick: refresh champion and challenger together, return
    the combined runtime. What a background poll loop calls on each tick."""
    current_champion = current.champion if current is not None else None
    current_challenger = current.challenger if current is not None else None
    champion = resolve_champion_model(cfg, current_champion)
    challenger = resolve_challenger_model(cfg, current_challenger)
    _CHAMPION_LAST_SUCCESSFUL_POLL_TIMESTAMP_SECONDS.set_to_current_time()
    return ServingRuntime(champion=champion, challenger=challenger)


async def load_serving_state_with_retry(cfg: DictConfig) -> ServingRuntime:
    """Cold-start loop: retries resolve_serving_model(cfg, None) with capped
    exponential backoff until it succeeds. Handles both "MLflow isn't up yet"
    (a docker compose race) and "no champion registered yet" the same way —
    one mechanism, not two code paths. The caller (app.py's lifespan) awaits
    this once at startup; /ready stays 503 until it returns.
    """
    backoff = float(cfg.serving.initial_load.backoff_base_seconds)
    backoff_max = float(cfg.serving.initial_load.backoff_max_seconds)
    while True:
        try:
            # Threaded for the same reason as app.py's poll loop: a load is
            # blocking MLflow I/O that would otherwise freeze /health too.
            return await asyncio.to_thread(resolve_serving_model, cfg, None)
        except Exception:
            logger.warning(
                "initial_model_load_failed_retrying",
                backoff_seconds=backoff,
                exc_info=True,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, backoff_max)


def _assemble_model_input(
    X: pd.DataFrame, committed_features: list[str]
) -> pd.DataFrame:
    """Raw 19-field rows -> exactly the columns the model's ColumnTransformer expects.

    Computes charge_per_service locally (features/generate.py's pandas
    mirror of sql/features/charge_per_service.sql) only when the model's own
    committed_features actually needs it — feature selection may have
    dropped it for a given training cycle.
    """
    engineered = X.copy()
    if (
        "charge_per_service" in committed_features
        and "charge_per_service" not in engineered.columns
    ):
        engineered["charge_per_service"] = compute_charge_per_service(engineered)
    return engineered[committed_features]


def _canary_bucket_mask(
    customer_ids: pd.Series, fraction: float, salt: str
) -> NDArray[np.bool_]:
    """Consistent-hash canary routing mask:
    sha256(f"{customerid}:{salt}") % 10_000 < fraction * 10_000.

    Hashing (not per-request randomness) so the same customer lands in the
    same bucket for the life of the canary rather than flapping between
    models call to call. A missing customerid is always False — no targeting
    key means the default variant (champion), never a routing decision.

    `salt` is the challenger's own model_version (see score_request's call
    site), not a fixed value — folding it into the hash key means bucket
    assignment stays stable for the life of one challenger's canary window
    (its version doesn't change mid-window) but reshuffles independently the
    next time `challenger` moves to a different version, rather than the
    same customer subset being the canary population on every canary this
    project ever runs, forever, regardless of which model is under test.
    """
    n = len(customer_ids)
    if fraction <= 0:
        return np.zeros(n, dtype=bool)

    is_missing = customer_ids.isna().to_numpy()
    buckets = (
        customer_ids.fillna("")
        .astype(str)
        .map(
            lambda cid: (
                int(hashlib.sha256(f"{cid}:{salt}".encode()).hexdigest(), 16)
                % _CANARY_BUCKETS
            )
        )
    )
    mask: NDArray[np.bool_] = buckets.to_numpy() < fraction * _CANARY_BUCKETS
    mask[is_missing] = False
    return mask


def score_request(
    X: pd.DataFrame,
    customer_ids: pd.Series,
    runtime: ServingRuntime,
    cfg: DictConfig,
) -> ScoredBatch:
    """Score a batch of rows (one row is a valid batch of one), routing
    through shadow/canary when configured. Vectorized end to end — the
    champion (and, when active, the challenger) is scored with exactly one
    predict_proba call over the whole DataFrame, never a per-row loop, so
    predict_single and predict_batch can share this one implementation.
    """
    X = X.reset_index(drop=True)
    customer_ids = customer_ids.reset_index(drop=True)
    n = len(X)

    champion = runtime.champion
    X_champion = _assemble_model_input(X, champion.committed_features)
    _inference_start = time.perf_counter()
    champion_proba: NDArray[np.float64] = champion.model.predict_proba(X_champion)[:, 1]
    _MODEL_INFERENCE_SECONDS.labels(model="champion").observe(
        time.perf_counter() - _inference_start
    )

    served_source = ["champion"] * n
    served_proba = champion_proba.copy()
    served_model_version = [champion.model_version] * n
    served_threshold = np.full(
        n, champion.thresholds[champion.base_scenario_name], dtype=float
    )
    served_scoring_path: list[str] = ["champion_only"] * n
    challenger_proba: NDArray[np.float64] | None = None
    dual_score_rows: list[dict[str, Any]] = []

    challenger = runtime.challenger
    shadow_enabled = bool(cfg.serving.shadow.enabled)
    canary_enabled = bool(cfg.serving.canary.enabled)
    canary_fraction = float(cfg.serving.canary.fraction)

    if challenger is not None and (shadow_enabled or canary_enabled):
        canary_mask = (
            _canary_bucket_mask(
                customer_ids, canary_fraction, salt=challenger.model_version
            )
            if canary_enabled
            else np.zeros(n, dtype=bool)
        )
        evaluated_mask = canary_mask | shadow_enabled
        evaluated_idx = np.flatnonzero(evaluated_mask)

        # Score only the rows actually evaluated — in canary-only mode
        # (shadow disabled), that's the ~fraction bucketed slice, never the
        # full batch. A real canary router never double-scores like this;
        # only shadow/mirroring is supposed to score everything.
        challenger_proba = np.full(n, np.nan, dtype=float)
        if evaluated_idx.size > 0:
            X_challenger = _assemble_model_input(
                X.iloc[evaluated_idx], challenger.committed_features
            )
            _inference_start = time.perf_counter()
            challenger_proba[evaluated_idx] = challenger.model.predict_proba(
                X_challenger
            )[:, 1]
            _MODEL_INFERENCE_SECONDS.labels(model="challenger").observe(
                time.perf_counter() - _inference_start
            )

        for i in evaluated_idx:
            routed = bool(canary_mask[i])
            entry = {
                "customerid": customer_ids.iloc[i],
                "champion_version": champion.model_version,
                "champion_score": float(champion_proba[i]),
                "challenger_version": challenger.model_version,
                "challenger_score": float(challenger_proba[i]),
                "served": "challenger" if routed else "champion",
                "mode": "canary" if routed else "shadow",
            }
            dual_score_rows.append(entry)
            logger.info("shadow_canary_dual_score", **entry)

            mode = entry["mode"]
            # Self-describing for predictions_total: who actually served this
            # row, not the dual-score entry's bare "mode" — see the metric's
            # own comment for why "shadow" alone would read backwards here.
            served_scoring_path[i] = (
                "challenger_canary" if routed else "champion_with_shadow"
            )
            _CHAMPION_CHALLENGER_SCORE_DISTANCE.labels(mode=mode).observe(
                abs(entry["challenger_score"] - entry["champion_score"])
            )
            champion_decision = (
                entry["champion_score"]
                >= champion.thresholds[champion.base_scenario_name]
            )
            challenger_decision = (
                entry["challenger_score"]
                >= challenger.thresholds[challenger.base_scenario_name]
            )
            outcome = (
                "agree" if champion_decision == challenger_decision else "disagree"
            )
            _CHAMPION_CHALLENGER_AGREEMENT_TOTAL.labels(
                outcome=outcome, mode=mode
            ).inc()

        if canary_enabled and canary_mask.any():
            served_source = [
                "challenger" if routed else "champion" for routed in canary_mask
            ]
            served_proba = np.where(canary_mask, challenger_proba, champion_proba)
            served_model_version = [
                challenger.model_version if routed else champion.model_version
                for routed in canary_mask
            ]
            served_threshold = np.where(
                canary_mask,
                challenger.thresholds[challenger.base_scenario_name],
                served_threshold,
            )

    for row in range(n):
        _PREDICTIONS_TOTAL.labels(
            scoring_path=served_scoring_path[row],
            model_version=served_model_version[row],
        ).inc()
        _PREDICTED_PROBABILITY.labels(model_version=served_model_version[row]).observe(
            float(served_proba[row])
        )
        if served_proba[row] >= served_threshold[row]:
            _PREDICTIONS_ABOVE_THRESHOLD_TOTAL.labels(
                model_version=served_model_version[row]
            ).inc()

    return ScoredBatch(
        customer_ids=customer_ids,
        served_proba=served_proba,
        served_source=served_source,
        served_model_version=served_model_version,
        served_threshold=served_threshold,
        champion_proba=champion_proba,
        challenger_proba=challenger_proba,
        dual_score_rows=dual_score_rows,
    )


def _bundle_for_source(runtime: ServingRuntime, source: str) -> ModelBundle:
    if source == "challenger":
        assert runtime.challenger is not None
        return runtime.challenger
    return runtime.champion


def predict_single(
    request: PredictRequest, runtime: ServingRuntime, cfg: DictConfig
) -> tuple[PredictResponse, ScoredBatch]:
    """Score one customer, optionally with a local SHAP explanation.

    Never carries `contact`/`capacity_limit` — a single customer in
    isolation has no population to be capacity-ranked against; that policy
    is `serving/contact_policy.py::select_contacts`, wired into
    `predict_batch`'s caller instead.

    Also returns the underlying `ScoredBatch` — `serving/app.py`'s
    `POST /predict` handler needs it to build a `prediction_log` row
    (`serving/prediction_log.py::build_log_rows`), and re-deriving it with a
    second `score_request` call would score the same row twice.
    """
    row = request.model_dump(
        exclude={"customerid", "include_explanation", "resolution_kind"}
    )
    X = pd.DataFrame([row])
    customer_ids = pd.Series([request.customerid])

    scored = score_request(X, customer_ids, runtime, cfg)
    proba = float(scored.served_proba[0])
    threshold = float(scored.served_threshold[0])
    model_version = scored.served_model_version[0]

    explanation: Explanation | None = None
    if request.include_explanation:
        bundle = _bundle_for_source(runtime, scored.served_source[0])
        X_model = _assemble_model_input(X, bundle.committed_features)
        _explain_start = time.perf_counter()
        shap_values, feature_names, base_value, Xt = explain_with_explainer(
            bundle.explainer, bundle.preprocessor, X_model
        )
        _EXPLANATION_SECONDS.observe(time.perf_counter() - _explain_start)
        explained = local_explanations(
            shap_values,
            base_value,
            feature_names,
            indices=[0],
            top_k=int(cfg.serving.explanation.top_k),
            feature_values=Xt,
        )[0]
        explanation = Explanation(
            base_value=explained["base_value"],
            top_features=[
                FeatureContribution(
                    feature=feature["feature"],
                    shap_value=feature["shap_value"],
                    customer_value=feature.get("feature_value"),
                )
                for feature in explained["top_features"]
            ],
        )

    response = PredictResponse(
        customerid=request.customerid,
        probability=proba,
        threshold=threshold,
        decision=proba >= threshold,
        model_version=model_version,
        explanation=explanation,
    )
    return response, scored


def predict_batch(
    rows: pd.DataFrame,
    customer_ids: pd.Series,
    runtime: ServingRuntime,
    cfg: DictConfig,
) -> ScoredBatch:
    """Score an already-assembled, already-valid batch of raw feature rows.

    Thin wrapper around score_request — capacity-constrained contact
    selection (serving/contact_policy.py::select_contacts) and
    BatchPredictionResult assembly are the caller's job (serving/app.py),
    not this function's: ScoredBatch carries served_proba/champion_proba,
    not decision/contact, so app.py can apply select_contacts to whichever
    probability vector it decides is the right ranking input.
    """
    return score_request(rows, customer_ids, runtime, cfg)
