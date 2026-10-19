"""Unit tests for telco_churn.serving.predict.

Every test registers a real, small CalibratedClassifierCV(ensemble=False)
over [preprocessor -> LGBMClassifier] against a tmp-scoped MLflow registry —
the same real structure build_model_bundle's unwrap_calibrated_pipeline/
TreeExplainer path needs, mirroring test_shap_values.py's fixture. Champion
and challenger are two independently-fit models (different random seeds) so
canary/shadow routing produces genuinely different scores to assert against,
not two copies of the same number.
"""

from __future__ import annotations

import asyncio
import logging
import time

import mlflow
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf
from prometheus_client import REGISTRY
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models import environment_parity
from telco_churn.serving import predict
from telco_churn.serving.schemas import PredictRequest

_BINARY = ["informative_bin"]
_MULTI_CAT: list[str] = []
_NUMERIC = ["informative_num", "noise_num"]
_COMMITTED_FEATURES = [*_NUMERIC, *_BINARY]

_FAST_ESTIMATOR_PARAMS = {
    "n_estimators": 30,
    "num_leaves": 7,
    "min_child_samples": 5,
    "class_weight": "balanced",
    "n_jobs": 1,
    "verbose": -1,
}


def _make_synthetic_data(
    n: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    informative_num = rng.normal(0, 1, size=n)
    noise_num = rng.normal(0, 1, size=n)
    informative_bin = rng.choice(["Yes", "No"], size=n)

    logit = 1.5 * informative_num + 1.2 * (informative_bin == "Yes").astype(float) - 0.5
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob), name="churn")
    X = pd.DataFrame(
        {
            "informative_num": informative_num,
            "noise_num": noise_num,
            "informative_bin": informative_bin,
        }
    )
    return X, y


def _fit_calibrated_pipeline(seed: int) -> CalibratedClassifierCV:
    X, y = _make_synthetic_data(seed=seed)
    preprocessor = build_preprocessor(_BINARY, _MULTI_CAT, _NUMERIC)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(random_state=seed, **_FAST_ESTIMATOR_PARAMS)),
        ]
    )
    calibrated = CalibratedClassifierCV(
        pipeline,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        ensemble=False,
    )
    calibrated.fit(X, y)
    return calibrated


@pytest.fixture(scope="module")
def predict_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Module-scoped tmp SQLite MLflow store — see test_register.py's
    register_mlflow_uri for why module scope is safe (every test gets its
    own uniquely-named registered model via predict_cfg below)."""
    tmp_path = tmp_path_factory.mktemp("predict_mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    experiment_id = mlflow.create_experiment(
        "test_predict", artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


@pytest.fixture
def predict_cfg(predict_mlflow_uri: str, request: pytest.FixtureRequest) -> DictConfig:
    import re

    registered_model_name = "test-telco-churn-predict-" + re.sub(
        r"[^A-Za-z0-9_-]", "_", request.node.name
    )
    return OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": predict_mlflow_uri,
                "experiment_name": "test_predict",
                "registered_model_name": registered_model_name,
                "artifact_location": "",
            },
            "register": {"environment_packages": []},
            "serving": {
                "poll_interval_seconds": 30,
                "initial_load": {
                    "backoff_base_seconds": 0.01,
                    "backoff_max_seconds": 0.05,
                },
                "shadow": {"enabled": False},
                "canary": {"enabled": False, "fraction": 0.0},
                "explanation": {"top_k": 5},
            },
        }
    )


def _register_test_model(
    registered_model_name: str,
    seed: int,
    threshold: float = 0.4,
) -> tuple[str, str]:
    """Fit, log, and register one calibrated pipeline. Returns (version, run_id)."""
    calibrated = _fit_calibrated_pipeline(seed)
    X, _y = _make_synthetic_data(seed=seed)
    signature = infer_signature(X, calibrated.predict_proba(X))

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_dict(
            {"feature_selection": {"model_features": _COMMITTED_FEATURES}},
            "training_manifest.json",
        )
        mlflow.log_dict(
            {
                "threshold": threshold,
                "scenario": "base",
                "scenarios": {
                    "base": {
                        "threshold": threshold,
                        "costs": {"c": 60.0, "r": 0.3, "ltv": 500.0, "arpu": 70.0},
                    },
                },
            },
            "threshold/threshold.json",
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=calibrated,
            name="calibrated_model",
            signature=signature,
            input_example=X.head(2),
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=registered_model_name,
        )
    version = str(model_info.registered_model_version)
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        registered_model_name, version, "logged_model_id", model_info.model_id
    )
    return version, run_id


def _set_alias(registered_model_name: str, alias: str, version: str) -> None:
    mlflow.tracking.MlflowClient().set_registered_model_alias(
        registered_model_name, alias, version
    )


# ---------------------------------------------------------------------------
# build_model_bundle / resolve_champion_model / resolve_challenger_model
# ---------------------------------------------------------------------------


def test_resolve_champion_model_cold_start_raises_when_unregistered(
    predict_cfg: DictConfig,
) -> None:
    with pytest.raises(RuntimeError, match="No champion model"):
        predict.resolve_champion_model(predict_cfg, None)


def test_resolve_champion_model_cold_start_loads_the_aliased_version(
    predict_cfg: DictConfig,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, run_id = _register_test_model(name, seed=1)
    _set_alias(name, "champion", version)

    bundle = predict.resolve_champion_model(predict_cfg, None)

    assert bundle.model_version == version
    assert bundle.run_id == run_id
    assert bundle.committed_features == _COMMITTED_FEATURES
    assert bundle.thresholds["base"] == pytest.approx(0.4)
    assert bundle.explainer is not None


def test_build_model_bundle_resolves_threshold_through_threshold_run_id_tag(
    predict_cfg: DictConfig,
) -> None:
    """A costs.yaml-only re-run's threshold_run_id tag (register.py's
    tag_threshold_rerun) supersedes the training run's own threshold.json,
    without a new model version."""
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1, threshold=0.4)
    with mlflow.start_run() as rerun_run:
        mlflow.log_dict(
            {
                "threshold": 0.55,
                "scenario": "base",
                "scenarios": {
                    "base": {
                        "threshold": 0.55,
                        "costs": {"c": 60.0, "r": 0.3, "ltv": 500.0, "arpu": 70.0},
                    },
                },
            },
            "threshold/threshold.json",
        )
        rerun_run_id = rerun_run.info.run_id
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(name, version, "threshold_run_id", rerun_run_id)

    bundle = predict.build_model_bundle(version, predict_cfg)

    assert bundle.thresholds["base"] == pytest.approx(0.55)


def test_resolve_champion_model_refresh_is_a_noop_on_unchanged_alias(
    predict_cfg: DictConfig,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1)
    _set_alias(name, "champion", version)

    current = predict.resolve_champion_model(predict_cfg, None)
    refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed is current


def test_resolve_champion_model_refresh_loads_a_moved_alias(
    predict_cfg: DictConfig,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    v1, _r1 = _register_test_model(name, seed=1)
    _set_alias(name, "champion", v1)
    current = predict.resolve_champion_model(predict_cfg, None)

    v2, _r2 = _register_test_model(name, seed=2)
    _set_alias(name, "champion", v2)
    refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed.model_version == v2
    assert refreshed is not current


def test_resolve_champion_model_refresh_refuses_on_dependency_mismatch(
    predict_cfg: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A diverged environment must keep serving the old bundle, not raise
    and not hot-reload — the refuse-and-alert half of the hot-reload guard."""
    name = str(predict_cfg.mlflow.registered_model_name)
    v1, _r1 = _register_test_model(name, seed=1)
    _set_alias(name, "champion", v1)
    current = predict.resolve_champion_model(predict_cfg, None)

    v2, _r2 = _register_test_model(name, seed=2)
    _set_alias(name, "champion", v2)

    monkeypatch.setattr(
        predict,
        "diff_environment",
        lambda model_uri, packages: environment_parity.EnvironmentDiff(
            mismatches={"numpy": ("1.0.0", "2.0.0")}
        ),
    )
    with caplog.at_level(logging.WARNING):
        refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed is current
    assert refreshed.model_version == v1


def test_resolve_champion_model_refresh_fails_open_on_any_error(
    predict_cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh-time exception (a transient MLflow blip) must never raise —
    the last-known-good bundle is returned unchanged."""
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1)
    _set_alias(name, "champion", version)
    current = predict.resolve_champion_model(predict_cfg, None)

    def _boom(cfg: DictConfig) -> str | None:
        raise RuntimeError("mlflow unreachable")

    monkeypatch.setattr(predict, "resolve_champion_version", _boom)

    refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed is current


def test_resolve_challenger_model_returns_none_when_unset(
    predict_cfg: DictConfig,
) -> None:
    assert predict.resolve_challenger_model(predict_cfg, None) is None


def test_resolve_challenger_model_loads_the_aliased_version(
    predict_cfg: DictConfig,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=2)
    _set_alias(name, "challenger", version)

    bundle = predict.resolve_challenger_model(predict_cfg, None)

    assert bundle is not None
    assert bundle.model_version == version


# ---------------------------------------------------------------------------
# Group 12 serving metrics: active_threshold / cost_scenario_constants /
# champion_version_registry+served / model_reload_total /
# champion_last_successful_poll_timestamp_seconds
# ---------------------------------------------------------------------------


def test_build_model_bundle_sets_active_threshold_and_cost_scenario_gauges(
    predict_cfg: DictConfig,
) -> None:
    """Both gauges must be pure mirrors of what build_model_bundle already
    resolved — _register_test_model's fixed costs (c=60.0, r=0.3, ltv=500.0)
    and threshold=0.4, read back unchanged."""
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1, threshold=0.4)

    predict.build_model_bundle(version, predict_cfg)

    assert REGISTRY.get_sample_value(
        "active_threshold", {"model_version": version}
    ) == pytest.approx(0.4)
    assert REGISTRY.get_sample_value(
        "cost_scenario_constants", {"constant": "ltv", "model_version": version}
    ) == pytest.approx(500.0)
    assert REGISTRY.get_sample_value(
        "cost_scenario_constants", {"constant": "cost", "model_version": version}
    ) == pytest.approx(60.0)
    assert REGISTRY.get_sample_value(
        "cost_scenario_constants",
        {"constant": "retention_rate", "model_version": version},
    ) == pytest.approx(0.3)


def test_resolve_champion_model_updates_version_registry_and_served_gauges(
    predict_cfg: DictConfig,
) -> None:
    """Both gauges are plain overwrites (never accumulated), so each poll's
    own read reflects only that poll's outcome — cold start, then a moved
    alias, each checked right after the call that caused it."""
    name = str(predict_cfg.mlflow.registered_model_name)
    v1, _r1 = _register_test_model(name, seed=1)
    _set_alias(name, "champion", v1)

    current = predict.resolve_champion_model(predict_cfg, None)

    assert REGISTRY.get_sample_value("champion_version_registry") == pytest.approx(
        float(v1)
    )
    assert REGISTRY.get_sample_value("champion_version_served") == pytest.approx(
        float(v1)
    )

    v2, _r2 = _register_test_model(name, seed=2)
    _set_alias(name, "champion", v2)
    refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed.model_version == v2
    assert REGISTRY.get_sample_value("champion_version_registry") == pytest.approx(
        float(v2)
    )
    assert REGISTRY.get_sample_value("champion_version_served") == pytest.approx(
        float(v2)
    )


def _reload_total(role: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "model_reload_total", {"role": role, "outcome": outcome}
        )
        or 0.0
    )


def test_resolve_champion_model_reload_counter_increments_on_success(
    predict_cfg: DictConfig,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    v1, _r1 = _register_test_model(name, seed=1)
    _set_alias(name, "champion", v1)
    current = predict.resolve_champion_model(predict_cfg, None)

    v2, _r2 = _register_test_model(name, seed=2)
    _set_alias(name, "champion", v2)
    before = _reload_total("champion", "success")

    refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed.model_version == v2
    assert _reload_total("champion", "success") == before + 1


def test_resolve_champion_model_reload_counter_increments_on_refused(
    predict_cfg: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    v1, _r1 = _register_test_model(name, seed=1)
    _set_alias(name, "champion", v1)
    current = predict.resolve_champion_model(predict_cfg, None)

    v2, _r2 = _register_test_model(name, seed=2)
    _set_alias(name, "champion", v2)
    monkeypatch.setattr(
        predict,
        "diff_environment",
        lambda model_uri, packages: environment_parity.EnvironmentDiff(
            mismatches={"numpy": ("1.0.0", "2.0.0")}
        ),
    )
    before = _reload_total("champion", "refused")

    with caplog.at_level(logging.WARNING):
        refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed is current
    assert _reload_total("champion", "refused") == before + 1


def test_resolve_champion_model_reload_counter_increments_on_failure(
    predict_cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1)
    _set_alias(name, "champion", version)
    current = predict.resolve_champion_model(predict_cfg, None)

    def _boom(cfg: DictConfig) -> str | None:
        raise RuntimeError("mlflow unreachable")

    monkeypatch.setattr(predict, "resolve_champion_version", _boom)
    before = _reload_total("champion", "failed")

    refreshed = predict.resolve_champion_model(predict_cfg, current)

    assert refreshed is current
    assert _reload_total("champion", "failed") == before + 1


def test_resolve_challenger_model_reload_counter_increments_on_success(
    predict_cfg: DictConfig,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=2)
    _set_alias(name, "challenger", version)
    before = _reload_total("challenger", "success")

    bundle = predict.resolve_challenger_model(predict_cfg, None)

    assert bundle is not None
    assert _reload_total("challenger", "success") == before + 1


def test_resolve_challenger_model_reload_counter_increments_on_alias_resolution_failure(
    predict_cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(cfg: DictConfig) -> str | None:
        raise RuntimeError("mlflow unreachable")

    monkeypatch.setattr(predict, "resolve_challenger_version", _boom)
    before = _reload_total("challenger", "failed")

    result = predict.resolve_challenger_model(predict_cfg, None)

    assert result is None
    assert _reload_total("challenger", "failed") == before + 1


def test_resolve_serving_model_sets_last_successful_poll_timestamp(
    predict_cfg: DictConfig,
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1)
    _set_alias(name, "champion", version)

    before = time.time()
    predict.resolve_serving_model(predict_cfg, None)
    after = time.time()

    observed = REGISTRY.get_sample_value(
        "champion_last_successful_poll_timestamp_seconds"
    )

    assert observed is not None
    assert before <= observed <= after


# ---------------------------------------------------------------------------
# load_serving_state_with_retry
# ---------------------------------------------------------------------------


def test_load_serving_state_with_retry_succeeds_once_champion_appears(
    predict_cfg: DictConfig,
) -> None:
    """Fail-closed-with-retry: no champion yet -> keeps retrying; once one
    appears, the very next poll succeeds. Exercised without a real sleep by
    using the cfg's near-zero backoff knobs."""
    name = str(predict_cfg.mlflow.registered_model_name)

    async def _register_shortly() -> None:
        await asyncio.sleep(0.02)
        version, _run_id = _register_test_model(name, seed=1)
        _set_alias(name, "champion", version)

    async def _run() -> predict.ServingRuntime:
        register_task = asyncio.create_task(_register_shortly())
        runtime = await predict.load_serving_state_with_retry(predict_cfg)
        await register_task
        return runtime

    runtime = asyncio.run(_run())
    assert runtime.champion is not None


# ---------------------------------------------------------------------------
# score_request — shadow/canary routing
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime_with_challenger(predict_cfg: DictConfig) -> predict.ServingRuntime:
    name = str(predict_cfg.mlflow.registered_model_name)
    champion_version, _ = _register_test_model(name, seed=1, threshold=0.4)
    challenger_version, _ = _register_test_model(name, seed=99, threshold=0.6)
    _set_alias(name, "champion", champion_version)
    _set_alias(name, "challenger", challenger_version)
    champion = predict.resolve_champion_model(predict_cfg, None)
    challenger = predict.resolve_challenger_model(predict_cfg, None)
    return predict.ServingRuntime(champion=champion, challenger=challenger)


@pytest.fixture
def replayed_rows() -> tuple[pd.DataFrame, pd.Series]:
    """A handful of synthetic rows in the model's own committed feature
    space, plus customerids, covering score_request's routing edge cases
    deterministically (fraction=0/1, a dependency mismatch, a missing
    customerid) — this project's own test-authoring convention throughout
    (test_shap_values.py, test_register.py). PROJECT_PLAN.md's "replayed
    dev-set rows, never fabricated production evidence" instruction targets
    a different risk: a shadow/canary demonstration that could be mistaken
    for real production traffic if presented as evidence. That's satisfied
    by docs/architecture.md's own snippet, generated from actual dev-
    partition customers via data.split.partition (never the sealed test
    side) run through this exact mechanism — see that doc's Shadow/Canary
    section."""
    X, _y = _make_synthetic_data(n=20, seed=7)
    customer_ids = pd.Series([f"cust-{i:04d}" for i in range(len(X))])
    return X, customer_ids


def test_score_request_champion_only_when_no_challenger(
    predict_cfg: DictConfig, replayed_rows: tuple[pd.DataFrame, pd.Series]
) -> None:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1)
    _set_alias(name, "champion", version)
    champion = predict.resolve_champion_model(predict_cfg, None)
    runtime = predict.ServingRuntime(champion=champion, challenger=None)
    X, customer_ids = replayed_rows

    scored = predict.score_request(X, customer_ids, runtime, predict_cfg)

    assert scored.challenger_proba is None
    assert scored.dual_score_rows == []
    assert all(source == "champion" for source in scored.served_source)
    np.testing.assert_allclose(scored.served_proba, scored.champion_proba)


def test_score_request_shadow_never_routes_but_always_logs(
    predict_cfg: DictConfig,
    runtime_with_challenger: predict.ServingRuntime,
    replayed_rows: tuple[pd.DataFrame, pd.Series],
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = OmegaConf.merge(predict_cfg, {"serving": {"shadow": {"enabled": True}}})
    X, customer_ids = replayed_rows

    with caplog.at_level(logging.INFO):
        scored = predict.score_request(X, customer_ids, runtime_with_challenger, cfg)

    assert all(source == "champion" for source in scored.served_source)
    np.testing.assert_allclose(scored.served_proba, scored.champion_proba)
    assert scored.challenger_proba is not None
    assert len(scored.dual_score_rows) == len(X)
    assert all(row["mode"] == "shadow" for row in scored.dual_score_rows)
    assert all(row["served"] == "champion" for row in scored.dual_score_rows)


def _histogram_count(name: str, labels: dict[str, str] | None = None) -> float:
    """Current observation count of a default-registry histogram (0 if unseen)."""
    value = REGISTRY.get_sample_value(f"{name}_count", labels or {})
    return value or 0.0


def test_score_request_observes_inference_time_per_model_bundle(
    predict_cfg: DictConfig,
    runtime_with_challenger: predict.ServingRuntime,
    replayed_rows: tuple[pd.DataFrame, pd.Series],
) -> None:
    """One predict_proba call per bundle per request, each timed under its own
    model label — so a slow challenger can't hide inside champion latency."""
    cfg = OmegaConf.merge(predict_cfg, {"serving": {"shadow": {"enabled": True}}})
    X, customer_ids = replayed_rows
    champion_before = _histogram_count("model_inference_seconds", {"model": "champion"})
    challenger_before = _histogram_count(
        "model_inference_seconds", {"model": "challenger"}
    )

    predict.score_request(X, customer_ids, runtime_with_challenger, cfg)

    assert (
        _histogram_count("model_inference_seconds", {"model": "champion"})
        == champion_before + 1
    )
    assert (
        _histogram_count("model_inference_seconds", {"model": "challenger"})
        == challenger_before + 1
    )


def test_canary_bucket_mask_is_stable_for_a_repeated_salt() -> None:
    """The same challenger version (salt) must route the same customers to
    the same bucket call to call — the stability guarantee a canary window
    depends on."""
    customer_ids = pd.Series([f"CUST-{i}" for i in range(2000)])
    mask_a = predict._canary_bucket_mask(customer_ids, fraction=0.3, salt="1")
    mask_b = predict._canary_bucket_mask(customer_ids, fraction=0.3, salt="1")
    np.testing.assert_array_equal(mask_a, mask_b)


def test_canary_bucket_mask_reshuffles_when_the_salt_changes() -> None:
    """A new challenger version (a new salt) must draw a different customer
    subset — the fix for ANALYSIS.md §9 item 15: without a salt, the same
    subset would be the canary population on every canary, forever."""
    customer_ids = pd.Series([f"CUST-{i}" for i in range(2000)])
    mask_v1 = predict._canary_bucket_mask(customer_ids, fraction=0.3, salt="1")
    mask_v2 = predict._canary_bucket_mask(customer_ids, fraction=0.3, salt="2")
    assert not np.array_equal(mask_v1, mask_v2)


def test_score_request_canary_fraction_one_routes_everyone_to_challenger(
    predict_cfg: DictConfig,
    runtime_with_challenger: predict.ServingRuntime,
    replayed_rows: tuple[pd.DataFrame, pd.Series],
) -> None:
    cfg = OmegaConf.merge(
        predict_cfg, {"serving": {"canary": {"enabled": True, "fraction": 1.0}}}
    )
    X, customer_ids = replayed_rows

    scored = predict.score_request(X, customer_ids, runtime_with_challenger, cfg)

    assert all(source == "challenger" for source in scored.served_source)
    assert scored.challenger_proba is not None
    np.testing.assert_allclose(scored.served_proba, scored.challenger_proba)
    assert all(row["mode"] == "canary" for row in scored.dual_score_rows)


def test_score_request_canary_fraction_zero_routes_everyone_to_champion(
    predict_cfg: DictConfig,
    runtime_with_challenger: predict.ServingRuntime,
    replayed_rows: tuple[pd.DataFrame, pd.Series],
) -> None:
    cfg = OmegaConf.merge(
        predict_cfg, {"serving": {"canary": {"enabled": True, "fraction": 0.0}}}
    )
    X, customer_ids = replayed_rows

    scored = predict.score_request(X, customer_ids, runtime_with_challenger, cfg)

    assert all(source == "champion" for source in scored.served_source)
    np.testing.assert_allclose(scored.served_proba, scored.champion_proba)


def test_score_request_canary_only_partial_fraction_skips_unbucketed_rows(
    predict_cfg: DictConfig,
    runtime_with_challenger: predict.ServingRuntime,
    replayed_rows: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canary-only (shadow disabled): the challenger must only ever be asked
    to score the bucketed subset, never the full batch — a real canary
    router never double-scores every request the way shadow/mirroring does.
    """
    cfg = OmegaConf.merge(
        predict_cfg, {"serving": {"canary": {"enabled": True, "fraction": 0.3}}}
    )
    X, customer_ids = replayed_rows
    n = len(X)
    half = n // 2
    fixed_mask = np.array([True] * half + [False] * (n - half))
    monkeypatch.setattr(
        predict, "_canary_bucket_mask", lambda cids, frac, salt: fixed_mask
    )

    challenger = runtime_with_challenger.challenger
    assert challenger is not None
    real_predict_proba = challenger.model.predict_proba
    seen_sizes: list[int] = []

    def _spy_predict_proba(X_in: pd.DataFrame) -> np.ndarray:
        seen_sizes.append(len(X_in))
        return real_predict_proba(X_in)

    monkeypatch.setattr(challenger.model, "predict_proba", _spy_predict_proba)

    scored = predict.score_request(X, customer_ids, runtime_with_challenger, cfg)

    assert seen_sizes == [half]
    assert np.isnan(scored.challenger_proba[half:]).all()
    assert not np.isnan(scored.challenger_proba[:half]).any()
    assert len(scored.dual_score_rows) == half
    assert all(row["mode"] == "canary" for row in scored.dual_score_rows)


def _metric_sample_value(metric: object, sample_name: str, **labels: str) -> float:
    """Read one labelled sample's current value off a prometheus_client metric's
    public collect() API — never the private _count/_value/_sum internals,
    which differ across metric types and prometheus_client versions.

    A label combination a test hasn't touched yet (in this process, possibly
    because it's the first test to run) has no child registered at all, so
    collect() reports no sample for it — that's 0, not a missing metric, and
    treating it as such keeps this assertion order-independent across the
    module-level metric objects every test in this file shares.
    """
    for family in metric.collect():  # type: ignore[attr-defined]
        for sample in family.samples:
            if sample.name == sample_name and sample.labels == labels:
                return float(sample.value)
    return 0.0


def test_score_request_records_shadow_canary_prometheus_metrics(
    predict_cfg: DictConfig,
    runtime_with_challenger: predict.ServingRuntime,
    replayed_rows: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Every dual-scored row must observe champion_challenger_score_distance
    and increment champion_challenger_agreement_total exactly once, with the
    outcome label matching whether champion/challenger landed on the same
    side of their own respective thresholds.
    """
    cfg = OmegaConf.merge(predict_cfg, {"serving": {"shadow": {"enabled": True}}})
    X, customer_ids = replayed_rows

    distance_before = _metric_sample_value(
        predict._CHAMPION_CHALLENGER_SCORE_DISTANCE,
        "champion_challenger_score_distance_count",
        mode="shadow",
    )
    agree_true_before = _metric_sample_value(
        predict._CHAMPION_CHALLENGER_AGREEMENT_TOTAL,
        "champion_challenger_agreement_total",
        outcome="agree",
        mode="shadow",
    )
    agree_false_before = _metric_sample_value(
        predict._CHAMPION_CHALLENGER_AGREEMENT_TOTAL,
        "champion_challenger_agreement_total",
        outcome="disagree",
        mode="shadow",
    )

    scored = predict.score_request(X, customer_ids, runtime_with_challenger, cfg)

    champion = runtime_with_challenger.champion
    challenger = runtime_with_challenger.challenger
    assert challenger is not None
    expected_agree_true = 0
    expected_agree_false = 0
    for row in scored.dual_score_rows:
        champion_decision = (
            row["champion_score"] >= champion.thresholds[champion.base_scenario_name]
        )
        challenger_decision = (
            row["challenger_score"]
            >= challenger.thresholds[challenger.base_scenario_name]
        )
        if champion_decision == challenger_decision:
            expected_agree_true += 1
        else:
            expected_agree_false += 1

    distance_after = _metric_sample_value(
        predict._CHAMPION_CHALLENGER_SCORE_DISTANCE,
        "champion_challenger_score_distance_count",
        mode="shadow",
    )
    agree_true_after = _metric_sample_value(
        predict._CHAMPION_CHALLENGER_AGREEMENT_TOTAL,
        "champion_challenger_agreement_total",
        outcome="agree",
        mode="shadow",
    )
    agree_false_after = _metric_sample_value(
        predict._CHAMPION_CHALLENGER_AGREEMENT_TOTAL,
        "champion_challenger_agreement_total",
        outcome="disagree",
        mode="shadow",
    )

    assert distance_after - distance_before == len(scored.dual_score_rows)
    assert agree_true_after - agree_true_before == expected_agree_true
    assert agree_false_after - agree_false_before == expected_agree_false


def test_score_request_missing_customerid_always_serves_champion(
    predict_cfg: DictConfig, runtime_with_challenger: predict.ServingRuntime
) -> None:
    """No customerid -> no stable bucket -> always the default variant
    (champion), regardless of canary.fraction — PROJECT_PLAN.md's resolved
    "no targeting key -> default variant" rule."""
    cfg = OmegaConf.merge(
        predict_cfg, {"serving": {"canary": {"enabled": True, "fraction": 1.0}}}
    )
    X, _y = _make_synthetic_data(n=5, seed=3)
    customer_ids = pd.Series([None] * len(X))

    scored = predict.score_request(X, customer_ids, runtime_with_challenger, cfg)

    assert all(source == "champion" for source in scored.served_source)


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------


def _predict_request(customerid: str | None = "7590-VHVEG") -> PredictRequest:
    return PredictRequest(
        customerid=customerid,
        gender="Female",
        seniorcitizen=0,
        has_partner="Yes",
        dependents="No",
        tenure=12,
        phoneservice="Yes",
        multiplelines="No",
        internetservice="DSL",
        onlinesecurity="Yes",
        onlinebackup="No",
        deviceprotection="No",
        techsupport="No",
        streamingtv="No",
        streamingmovies="No",
        contract_type="Month-to-month",
        paperlessbilling="Yes",
        paymentmethod="Electronic check",
        monthlycharges=55.5,
        totalcharges=650.0,
        include_explanation=False,
    )


@pytest.fixture
def champion_only_runtime(predict_cfg: DictConfig) -> predict.ServingRuntime:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_model(name, seed=1, threshold=0.4)
    _set_alias(name, "champion", version)
    champion = predict.resolve_champion_model(predict_cfg, None)
    return predict.ServingRuntime(champion=champion, challenger=None)


# predict_single goes through PredictRequest's real 19 raw-field schema, so
# (unlike the synthetic informative_num/noise_num/informative_bin scheme
# above, used where score_request is driven directly) its fixture model must
# be fit on real CustomerFeatures field names for _assemble_model_input's
# column subsetting to find anything.
_RAW_COMMITTED_FEATURES = ["tenure", "monthlycharges"]


def _fit_raw_feature_pipeline(seed: int) -> CalibratedClassifierCV:
    rng = np.random.default_rng(seed)
    n = 200
    tenure = rng.integers(0, 72, size=n)
    monthlycharges = rng.uniform(20, 120, size=n)
    logit = -0.03 * tenure + 0.02 * monthlycharges - 1.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob), name="churn")
    X = pd.DataFrame({"tenure": tenure, "monthlycharges": monthlycharges})
    preprocessor = build_preprocessor([], [], _RAW_COMMITTED_FEATURES)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(random_state=seed, **_FAST_ESTIMATOR_PARAMS)),
        ]
    )
    calibrated = CalibratedClassifierCV(
        pipeline,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        ensemble=False,
    )
    calibrated.fit(X, y)
    return calibrated


def _register_raw_feature_model(
    registered_model_name: str, seed: int, threshold: float = 0.4
) -> tuple[str, str]:
    calibrated = _fit_raw_feature_pipeline(seed)
    X = pd.DataFrame({"tenure": [12, 60], "monthlycharges": [55.5, 90.0]})
    signature = infer_signature(X, calibrated.predict_proba(X))

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_dict(
            {"feature_selection": {"model_features": _RAW_COMMITTED_FEATURES}},
            "training_manifest.json",
        )
        mlflow.log_dict(
            {
                "threshold": threshold,
                "scenario": "base",
                "scenarios": {
                    "base": {
                        "threshold": threshold,
                        "costs": {"c": 60.0, "r": 0.3, "ltv": 500.0, "arpu": 70.0},
                    },
                },
            },
            "threshold/threshold.json",
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=calibrated,
            name="calibrated_model",
            signature=signature,
            input_example=X,
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=registered_model_name,
        )
    version = str(model_info.registered_model_version)
    return version, run_id


@pytest.fixture
def champion_only_runtime_raw_fields(
    predict_cfg: DictConfig,
) -> predict.ServingRuntime:
    name = str(predict_cfg.mlflow.registered_model_name)
    version, _run_id = _register_raw_feature_model(name, seed=1, threshold=0.4)
    _set_alias(name, "champion", version)
    champion = predict.resolve_champion_model(predict_cfg, None)
    return predict.ServingRuntime(champion=champion, challenger=None)


def test_predict_single_probability_matches_direct_model_call(
    predict_cfg: DictConfig, champion_only_runtime_raw_fields: predict.ServingRuntime
) -> None:
    """Serving-parity, at this module's own boundary: predict_single's
    probability for a request must exactly match the champion's own
    predict_proba on the equivalent engineered row — the invariant the full
    golden_predictions.json check (Task 4, through the HTTP layer) extends.
    """
    request = _predict_request()
    response, _scored = predict.predict_single(
        request, champion_only_runtime_raw_fields, predict_cfg
    )

    bundle = champion_only_runtime_raw_fields.champion
    row = request.model_dump(
        exclude={"customerid", "include_explanation", "resolution_kind"}
    )
    X = predict._assemble_model_input(pd.DataFrame([row]), bundle.committed_features)
    expected = float(bundle.model.predict_proba(X)[:, 1][0])

    assert response.probability == pytest.approx(expected)
    assert response.customerid == request.customerid
    assert response.model_version == bundle.model_version
    assert response.decision == (response.probability >= response.threshold)


def test_predict_single_with_explanation(
    predict_cfg: DictConfig, champion_only_runtime_raw_fields: predict.ServingRuntime
) -> None:
    request = _predict_request()
    request.include_explanation = True

    response, _scored = predict.predict_single(
        request, champion_only_runtime_raw_fields, predict_cfg
    )

    assert response.explanation is not None
    assert len(response.explanation.top_features) <= 5


def test_predict_single_with_explanation_observes_explanation_time(
    predict_cfg: DictConfig, champion_only_runtime_raw_fields: predict.ServingRuntime
) -> None:
    request = _predict_request()
    request.include_explanation = True
    before = _histogram_count("explanation_seconds")

    predict.predict_single(request, champion_only_runtime_raw_fields, predict_cfg)

    assert _histogram_count("explanation_seconds") == before + 1


def test_predict_single_without_explanation_omits_it(
    predict_cfg: DictConfig, champion_only_runtime_raw_fields: predict.ServingRuntime
) -> None:
    before = _histogram_count("explanation_seconds")
    response, _scored = predict.predict_single(
        _predict_request(), champion_only_runtime_raw_fields, predict_cfg
    )
    assert response.explanation is None
    assert _histogram_count("explanation_seconds") == before


def test_predict_single_matches_golden_predictions_json(
    predict_cfg: DictConfig,
) -> None:
    """Serving-parity against an actual golden_predictions.json artifact,
    logged in calibrate.py's own format (customerid/rows/p_hat, captured
    in-memory before pickling) — the artifact register.py's own smoke check
    reads, and the one the checklist item names directly. Exercises the full
    request path: PredictRequest -> _assemble_model_input ->
    ColumnTransformer -> predict_proba, for several distinct customers.
    """
    name = str(predict_cfg.mlflow.registered_model_name)
    calibrated = _fit_raw_feature_pipeline(seed=1)

    golden_requests = [
        _predict_request(customerid="cust-0001").model_copy(
            update={"tenure": 5, "monthlycharges": 30.0}
        ),
        _predict_request(customerid="cust-0002").model_copy(
            update={"tenure": 40, "monthlycharges": 75.0}
        ),
        _predict_request(customerid="cust-0003").model_copy(
            update={"tenure": 70, "monthlycharges": 110.0}
        ),
    ]
    golden_X = pd.DataFrame(
        [
            {"tenure": r.tenure, "monthlycharges": r.monthlycharges}
            for r in golden_requests
        ]
    )
    golden_p_hat = calibrated.predict_proba(golden_X)[:, 1]

    signature = infer_signature(golden_X, calibrated.predict_proba(golden_X))
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_dict(
            {"feature_selection": {"model_features": _RAW_COMMITTED_FEATURES}},
            "training_manifest.json",
        )
        mlflow.log_dict(
            {
                "threshold": 0.4,
                "scenario": "base",
                "scenarios": {
                    "base": {
                        "threshold": 0.4,
                        "costs": {"c": 60.0, "r": 0.3, "ltv": 500.0, "arpu": 70.0},
                    },
                },
            },
            "threshold/threshold.json",
        )
        mlflow.log_dict(
            {
                "purpose": (
                    "serving-parity fixture - reproducibility only; scores "
                    "are in-sample and are not performance evidence"
                ),
                "customerid": [r.customerid for r in golden_requests],
                "rows": golden_X.to_dict(orient="records"),
                "p_hat": golden_p_hat.tolist(),
            },
            "calibration/golden_predictions.json",
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=calibrated,
            name="calibrated_model",
            signature=signature,
            input_example=golden_X,
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=name,
        )
    version = str(model_info.registered_model_version)
    _set_alias(name, "champion", version)

    golden = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/calibration/golden_predictions.json"
    )
    runtime = predict.ServingRuntime(
        champion=predict.resolve_champion_model(predict_cfg, None), challenger=None
    )

    for request, expected in zip(golden_requests, golden["p_hat"], strict=True):
        response, _scored = predict.predict_single(request, runtime, predict_cfg)
        assert response.probability == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------


def test_predict_batch_scores_every_row_once_vectorized(
    predict_cfg: DictConfig,
    champion_only_runtime: predict.ServingRuntime,
    replayed_rows: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, customer_ids = replayed_rows
    scored = predict.predict_batch(X, customer_ids, champion_only_runtime, predict_cfg)

    assert len(scored.served_proba) == len(X)
    assert len(scored.served_model_version) == len(X)
    assert scored.challenger_proba is None
