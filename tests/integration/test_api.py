"""Integration tests: the FastAPI app (serving/app.py) via TestClient.

Exercises the real ASGI request path — lifespan startup/shutdown, the
`/predict`, `/predict/batch`, `/customer/{customerid}`, `/health`, `/ready`,
and `/metrics` routes — against a real (tmp-scoped) MLflow-registered
champion and a real (testcontainers) Postgres customers_raw table. Unlike
tests/unit/test_predict.py/test_contact_policy.py, which call predict.py's
functions directly, this file is the one place the HTTP layer itself
(FastAPI routing, request validation, dependency injection, the lifespan
hook) is under test.

Fixtures (serving_champion/serving_postgres_url/serving_env/
serving_customer_payload) live in conftest.py, shared with
test_serving_parity.py — see its module docstring for why the registered
model name can't be redirected per-test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sqlalchemy import create_engine, text

from telco_churn.serving.app import app

pytestmark = pytest.mark.integration


@pytest.fixture
def api_client(
    ready_serving_client: Callable[[], Any], serving_env: dict[str, Any]
) -> Iterator[TestClient]:
    """A TestClient whose lifespan startup has already loaded the seeded
    champion — the context-manager form is required for FastAPI to actually
    run the `lifespan` startup/shutdown hooks (a bare TestClient(app) without
    `with` never triggers them).
    """
    with ready_serving_client() as client:
        yield client


def test_health_returns_ok_with_no_dependency_checks(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_the_loaded_champion(
    api_client: TestClient, serving_champion: dict[str, Any]
) -> None:
    response = api_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_version"] == serving_champion["model_version"]
    assert body["run_id"] == serving_champion["run_id"]


def test_predict_returns_probability_threshold_decision_and_explanation(
    api_client: TestClient,
    serving_champion: dict[str, Any],
    serving_customer_payload: Any,
) -> None:
    payload = serving_customer_payload("prospect-0001", tenure=5, monthlycharges=30.0)

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["customerid"] == "prospect-0001"
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == pytest.approx(serving_champion["threshold"])
    assert body["decision"] == (body["probability"] >= body["threshold"])
    assert body["model_version"] == serving_champion["model_version"]
    assert body["explanation"] is not None
    assert 1 <= len(body["explanation"]["top_features"]) <= 5
    # decision/threshold: no contact/capacity_limit is ever computed for a
    # single customer in isolation — see serving/schemas.py::PredictResponse.
    assert "contact" not in body
    assert "capacity_limit" not in body


def test_predict_without_explanation_omits_it(
    api_client: TestClient, serving_customer_payload: Any
) -> None:
    payload = serving_customer_payload("prospect-0002", tenure=40, monthlycharges=75.0)
    payload["include_explanation"] = False

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["explanation"] is None


def test_customer_lookup_returns_crm_snapshot(api_client: TestClient) -> None:
    """Response is CustomerLookupResponse{features, crm_snapshot_at}, sourced
    from customers_crm — tenure/monthlycharges carry the seeded CRM nudge, so
    this asserts plausible bounds rather than the exact training-time value
    (see conftest.py::serving_postgres_url).
    """
    response = api_client.get("/customer/serve-0001")

    assert response.status_code == 200
    body = response.json()
    features = body["features"]
    assert features["customerid"] == "serve-0001"
    assert 5 + 1 <= features["tenure"] <= 5 + 6
    assert features["monthlycharges"] == pytest.approx(30.0)
    assert "churn" not in features
    assert body["crm_snapshot_at"] is not None


def test_customer_lookup_404_for_unknown_id(api_client: TestClient) -> None:
    response = api_client.get("/customer/does-not-exist")
    assert response.status_code == 404


def test_predict_batch_id_only_resolution_partial_success(
    api_client: TestClient, serving_champion: dict[str, Any]
) -> None:
    """Two known customerids resolve and score; one unknown customerid is
    reported as a resolution_error, not a whole-batch failure.
    """
    payload = [
        {"customerid": "serve-0001"},
        {"customerid": "serve-0002"},
        {"customerid": "does-not-exist"},
    ]

    response = api_client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert len(body["errors"]) == 1
    assert body["errors"][0]["customerid"] == "does-not-exist"
    assert body["errors"][0]["reason"] == "resolution_error"
    for result in body["results"]:
        assert result["model_version"] == serving_champion["model_version"]
        assert result["decision"] == (result["probability"] >= result["threshold"])
        assert isinstance(result["contact"], bool)
    assert body["capacity_limit"] > 0


def test_predict_batch_sets_contact_capacity_utilization_gauge(
    api_client: TestClient,
) -> None:
    """contact_capacity_utilization must mirror the exact ratio this same
    response's own contact/capacity_limit fields already imply — a pure
    aggregate of what select_contacts just returned, not a separate
    computation that could silently diverge from it."""
    payload = [{"customerid": "serve-0001"}, {"customerid": "serve-0002"}]

    response = api_client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    expected = sum(r["contact"] for r in body["results"]) / body["capacity_limit"]

    observed = REGISTRY.get_sample_value("contact_capacity_utilization")

    assert observed == pytest.approx(expected)


def test_predict_batch_over_max_size_returns_413(api_client: TestClient) -> None:
    payload = [{"customerid": f"overflow-{i}"} for i in range(501)]

    response = api_client.post("/predict/batch", json=payload)

    assert response.status_code == 413


def test_metrics_exposes_prediction_counters_after_traffic(
    api_client: TestClient, serving_customer_payload: Any
) -> None:
    payload = serving_customer_payload("prospect-0003", tenure=70, monthlycharges=110.0)
    predict_response = api_client.post("/predict", json=payload)
    assert predict_response.status_code == 200

    response = api_client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "predictions_total" in body
    assert "predicted_probability" in body
    assert "predictions_above_threshold_total" in body


# ---------------------------------------------------------------------------
# prediction_log — serving/prediction_log.py's BackgroundTasks write,
# exercised through the real HTTP path rather than a standalone subprocess
# test (prediction_log.py has no __main__). TestClient runs a request's
# background tasks synchronously as
# part of the request/response cycle, so the row is already committed by the
# time these assertions run.
# ---------------------------------------------------------------------------


def test_predict_writes_a_row_to_prediction_log(
    api_client: TestClient,
    serving_env: dict[str, Any],
    serving_customer_payload: Any,
) -> None:
    payload = serving_customer_payload("pl-single-0001", tenure=5, monthlycharges=30.0)
    payload["resolution_kind"] = "full_inline"

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    engine = create_engine(serving_env["postgres_url"])
    try:
        with engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT route, feature_snapshot, model_version, probability, "
                        "threshold, decision, contact, dual_score_mode, resolution_kind "
                        "FROM prediction_log WHERE customerid = :customerid"
                    ),
                    {"customerid": "pl-single-0001"},
                )
                .mappings()
                .one()
            )
    finally:
        engine.dispose()

    assert row["route"] == "single"
    assert row["model_version"] == body["model_version"]
    assert row["probability"] == pytest.approx(body["probability"])
    assert row["threshold"] == pytest.approx(body["threshold"])
    assert row["decision"] == body["decision"]
    assert row["contact"] is None
    assert row["dual_score_mode"] is None
    assert row["resolution_kind"] == "full_inline"
    assert "customerid" not in row["feature_snapshot"]
    assert "resolution_kind" not in row["feature_snapshot"]
    assert "include_explanation" not in row["feature_snapshot"]
    assert row["feature_snapshot"]["tenure"] == 5


def test_predict_batch_writes_rows_to_prediction_log(
    api_client: TestClient,
    serving_env: dict[str, Any],
    serving_customer_payload: Any,
) -> None:
    """Full-inline batch items (no lookup) rather than ID-only, so this test
    doesn't depend on — or interfere with — the shared, module-scoped
    customers_crm seed data other tests in this file already query against.
    """
    payload = [
        serving_customer_payload("pl-batch-0001", tenure=5, monthlycharges=30.0),
        serving_customer_payload("pl-batch-0002", tenure=40, monthlycharges=75.0),
    ]

    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200

    engine = create_engine(serving_env["postgres_url"])
    try:
        with engine.begin() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT customerid, route, contact, resolution_kind "
                        "FROM prediction_log WHERE customerid = ANY(:ids)"
                    ),
                    {"ids": ["pl-batch-0001", "pl-batch-0002"]},
                )
                .mappings()
                .all()
            )
    finally:
        engine.dispose()

    assert len(rows) == 2
    for row in rows:
        assert row["route"] == "batch"
        assert row["resolution_kind"] == "full_inline"
        assert isinstance(row["contact"], bool)


def test_prediction_log_disabled_skips_the_write(
    ready_serving_client: Callable[[], Any],
    api_client: TestClient,
    serving_env: dict[str, Any],
    compose_config_with_prediction_log_disabled: None,
    serving_customer_payload: Any,
) -> None:
    payload = serving_customer_payload(
        "pl-disabled-0001", tenure=5, monthlycharges=30.0
    )

    with ready_serving_client() as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200

    engine = create_engine(serving_env["postgres_url"])
    try:
        with engine.begin() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM prediction_log WHERE customerid = :customerid"
                ),
                {"customerid": "pl-disabled-0001"},
            ).scalar()
    finally:
        engine.dispose()

    assert count == 0


# ---------------------------------------------------------------------------
# serving.auth.enabled=true path — off by default (see require_api_key's
# docstring), but this is the exact code path Phase 12's public deployment
# turns on, so both the 401/pass-through behavior and the fail-fast startup
# guard need real coverage, not just the off-by-default happy path above.
# ---------------------------------------------------------------------------


def test_predict_401s_with_no_api_key_when_auth_enabled(
    ready_serving_client: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
    serving_env: dict[str, Any],
    compose_config_with_auth_enabled: None,
    serving_customer_payload: Any,
) -> None:
    monkeypatch.setenv("API_KEY", "test-secret-key")
    payload = serving_customer_payload(
        "prospect-auth-0001", tenure=5, monthlycharges=30.0
    )

    with ready_serving_client() as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 401


def test_predict_401s_with_wrong_api_key_when_auth_enabled(
    ready_serving_client: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
    serving_env: dict[str, Any],
    compose_config_with_auth_enabled: None,
    serving_customer_payload: Any,
) -> None:
    monkeypatch.setenv("API_KEY", "test-secret-key")
    payload = serving_customer_payload(
        "prospect-auth-0002", tenure=5, monthlycharges=30.0
    )

    with ready_serving_client() as client:
        response = client.post(
            "/predict", json=payload, headers={"X-API-Key": "wrong-key"}
        )

    assert response.status_code == 401


def test_predict_succeeds_with_correct_api_key_when_auth_enabled(
    ready_serving_client: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
    serving_env: dict[str, Any],
    compose_config_with_auth_enabled: None,
    serving_customer_payload: Any,
) -> None:
    monkeypatch.setenv("API_KEY", "test-secret-key")
    payload = serving_customer_payload(
        "prospect-auth-0003", tenure=5, monthlycharges=30.0
    )

    with ready_serving_client() as client:
        response = client.post(
            "/predict", json=payload, headers={"X-API-Key": "test-secret-key"}
        )

    assert response.status_code == 200


def test_health_and_ready_stay_unauthenticated_when_auth_enabled(
    ready_serving_client: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
    serving_env: dict[str, Any],
    compose_config_with_auth_enabled: None,
) -> None:
    monkeypatch.setenv("API_KEY", "test-secret-key")

    with ready_serving_client() as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/metrics").status_code == 200


def test_startup_raises_when_auth_enabled_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    serving_env: dict[str, Any],
    compose_config_with_auth_enabled: None,
) -> None:
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(app):
            pass
