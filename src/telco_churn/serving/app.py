"""FastAPI serving app — the process uvicorn runs (`telco_churn.serving.app:app`).

**Versioning decision, recorded rather than left implicit: routes ship
unversioned this phase (`/predict`, not `/predict/v1`).** There is no
external consumer yet — this phase's own tests, Phase 10's `/ready` poll, and
Phase 11's CI all live in this repo and can move in lockstep with a route
rename. Adopt an `/v1` prefix no later than Phase 11, before either of those
consumers exists to break.

**Config is resolved once, in `lifespan`, and stashed on `app.state` — not at
import time.** `compose_config()`/`activate_config()` need Hydra's config
directory and (for the champion load) a reachable MLflow; running them at
module import would make importing this module for a test fail before a test
even gets to choose what to mock. Every request handler reads `request.app.state.cfg`
instead of a module-level global.

**`serving/predict.py` (Task 3) already owns hot-reload, shadow/canary
routing, and most of the module's Prometheus metrics; `serving/customer_lookup.py`
owns Postgres ID-resolution — this module is a thin HTTP skin over both, not
a second implementation of either.** `GET /metrics` needs no extra wiring for
any of it: `prometheus_fastapi_instrumentator`'s `Instrumentator().expose(app)`
serves the entire default `prometheus_client` registry, which already
contains every metric `predict.py` registered at import time, plus this
module's own `prediction_log_write_failures_total`/`customer_lookup_timeouts_total`/
`contact_capacity_utilization` — owned here rather than elsewhere because
this module is what swallows the write failure / times out the lookup /
calls `select_contacts` (see `_log_prediction_background`,
`_lookup_customers_or_503`, and `/predict/batch` below).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import signal
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from omegaconf import DictConfig
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import ValidationError

from telco_churn.models.policy_config import load_costs_config
from telco_churn.serving import customer_lookup, predict, prediction_log
from telco_churn.serving.contact_policy import select_contacts
from telco_churn.serving.schemas import (
    BatchPredictionError,
    BatchPredictionResult,
    BatchPredictItem,
    BatchPredictResponse,
    CustomerFeatures,
    CustomerLookupResponse,
    PredictRequest,
    PredictResponse,
)
from telco_churn.utils.db import dispose_async_engine, get_async_engine
from telco_churn.utils.logging import configure_logging, get_logger
from telco_churn.utils.paths import activate_config, compose_config

__all__ = ["app", "lifespan", "require_api_key", "get_runtime"]

logger = get_logger(__name__)

# Owned here, not serving/prediction_log.py: this module is what swallows a
# write failure (see _log_prediction_background below), and a metric belongs
# to whichever module actually increments it — the same rule predict.py's
# own module-level metrics follow. No model_version label: a write failure
# is a Postgres-availability signal, not a per-model one.
_PREDICTION_LOG_WRITE_FAILURES_TOTAL = Counter(
    "prediction_log_write_failures",
    "Predictions whose prediction_log write failed and was swallowed — the "
    "response itself was still served; this counts only the logging side effect.",
    labelnames=("route",),
)
_CUSTOMER_LOOKUP_TIMEOUTS_TOTAL = Counter(
    "customer_lookup_timeouts",
    "customers_crm lookups abandoned after serving.database.lookup_timeout_seconds "
    "and answered 503 — a stalled-Postgres signal, independent of request volume.",
    labelnames=("route",),
)
# Owned here, not contact_policy.py: select_contacts is a pure ranking
# function with no notion of "a cycle," only the one batch it was called
# with — this module is the only caller, so it's the only place that can
# say how much of the cycle's capacity that one call actually used. No
# model_version label: capacity is a cycle-level operational constraint
# from configs/costs.yaml, not a per-model one.
_CONTACT_CAPACITY_UTILIZATION = Gauge(
    "contact_capacity_utilization",
    "Fraction of this cycle's capacity_limit actually filled by contact=true "
    "rows, from the most recent /predict/batch call.",
)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: load config, start the initial-load-then-poll background
    task, and return (`yield`) immediately. Shutdown: flip readiness first,
    then drain.

    **The initial champion load must never be awaited directly in this
    function's own body.** uvicorn's own `Server.startup()` awaits the ASGI
    lifespan's startup phase (this function, up to its `yield`) *before* it
    ever calls `loop.create_server(...)` — the listening socket isn't even
    opened yet, let alone accepting connections, until this coroutine
    reaches `yield`. `load_serving_state_with_retry` retries forever on a
    genuine cold start with no `champion` alias to resolve, so awaiting it
    here would mean the socket never opens at all: not just `/ready`, but
    `/health` too — the liveness probe this project's own design explicitly
    says "must never restart-loop a process that was never actually broken"
    (see `health()`'s docstring below) — would be unreachable for as long as
    MLflow has nothing to serve. Folding the initial load into the same
    background task the periodic hot-reload poll already runs on fixes
    this: `yield` happens right after the task is spawned, uvicorn opens the
    socket immediately, and `/health`/`/ready` (503, via `get_runtime`'s and
    `ready()`'s own `app.state.runtime is None`/`app.state.ready` checks)
    are reachable the whole time the task is still retrying underneath.

    The SIGTERM handler is registered separately from (and ahead of) the
    normal shutdown path below, because App Runner's scale-to-zero sends
    SIGTERM routinely, not just on deploys — by the time Starlette's own
    shutdown event fires, uvicorn has typically already stopped accepting
    new connections, which is too late for '/ready' to have warned the
    platform to stop routing here in the first place. Falls back to a no-op
    where add_signal_handler isn't available — Windows dev boxes (the
    container runtime this matters for is Linux, which supports it), and
    any test harness that runs this lifespan off the main thread (Starlette's
    TestClient does, via a background anyio portal); real uvicorn always
    runs lifespan on the main thread, so production is unaffected.
    """
    load_dotenv()
    configure_logging()
    cfg = compose_config()
    activate_config(cfg)
    if bool(cfg.serving.auth.enabled) and not os.environ.get("API_KEY"):
        raise RuntimeError(
            "serving.auth.enabled=true but the API_KEY environment variable "
            "is not set — every request to an authenticated route would be "
            "rejected unconditionally."
        )

    app.state.cfg = cfg
    app.state.ready = False
    app.state.runtime = None

    def _flip_not_ready() -> None:
        app.state.ready = False
        logger.warning("sigterm_received_readiness_flipped_to_not_ready")

    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _flip_not_ready)
    except (NotImplementedError, AttributeError, RuntimeError):
        # NotImplementedError/AttributeError: platform can't do this at all
        # (Windows dev boxes). RuntimeError: the platform can, but this
        # process/thread can't right now — add_signal_handler calls
        # signal.set_wakeup_fd() under the hood, which only works on the
        # main thread of the main interpreter. Starlette's TestClient runs
        # the app's lifespan inside a background anyio portal thread, not
        # the main thread, so every TestClient-driven test hits this path
        # even on Linux; real uvicorn always runs lifespan on the main
        # thread, so production is unaffected either way.
        logger.warning("sigterm_handler_unavailable_in_this_context")

    async def _load_and_poll_loop() -> None:
        interval = float(cfg.serving.poll_interval_seconds)
        app.state.runtime = await predict.load_serving_state_with_retry(cfg)
        app.state.ready = True
        logger.info(
            "serving_ready",
            model_version=app.state.runtime.champion.model_version,
            run_id=app.state.runtime.champion.run_id,
        )
        while True:
            await asyncio.sleep(interval)
            # Off the event loop: a refresh is blocking MLflow/registry I/O,
            # and run inline it would stall every in-flight request for as
            # long as MLflow takes to answer, every poll interval.
            app.state.runtime = await asyncio.to_thread(
                predict.resolve_serving_model, cfg, app.state.runtime
            )

    poll_task = asyncio.create_task(_load_and_poll_loop())

    yield

    app.state.ready = False
    poll_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await poll_task
    await dispose_async_engine()


app = FastAPI(title="Telco Churn Serving API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, include_in_schema=False)


def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    """Config-gated API-key dependency for the three customer-data routes.

    Off by default (configs/serving/api.yaml: auth.enabled) — never applied
    to /health, /ready, or /metrics, which must stay reachable by an infra
    probe or a Prometheus scraper with no credential to present.
    """
    cfg: DictConfig = request.app.state.cfg
    if not bool(cfg.serving.auth.enabled):
        return
    expected = os.environ.get("API_KEY")
    if not expected or not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


async def _lookup_customers_or_503(
    customerids: list[str],
    cfg: DictConfig,
    route: Literal["customer", "batch"],
) -> dict[str, dict[str, Any]]:
    """customer_lookup.lookup_customers, bounded by serving.database.lookup_timeout_seconds.

    A stalled Postgres connection otherwise holds the request open until the
    kernel gives up on the socket, long after any client has stopped waiting.
    Timing out here turns that into an explicit, retryable 503.
    """
    timeout_seconds = float(cfg.serving.database.lookup_timeout_seconds)
    try:
        return await asyncio.wait_for(
            customer_lookup.lookup_customers(customerids, get_async_engine()),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        _CUSTOMER_LOOKUP_TIMEOUTS_TOTAL.labels(route=route).inc()
        logger.error(
            "customer_lookup_timed_out",
            route=route,
            n_ids=len(customerids),
            timeout_seconds=timeout_seconds,
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail="customer lookup timed out; the database did not respond — retry",
        ) from None


def get_runtime(request: Request) -> predict.ServingRuntime:
    """The loaded ServingRuntime, or 503 before the first successful load."""
    runtime: predict.ServingRuntime | None = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return runtime


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


async def _log_prediction_background(
    feature_rows: list[dict[str, Any]],
    scored: predict.ScoredBatch,
    runtime: predict.ServingRuntime,
    contact: list[bool] | None,
    route: Literal["single", "batch", "nightly_batch"],
    request_id: str,
    resolution_kinds: list[
        Literal["id_only", "full_inline", "partial_override"] | None
    ],
) -> None:
    """BackgroundTasks target for both /predict and /predict/batch — fires
    after the response is already sent, so a Postgres hiccup here degrades
    to "this call wasn't captured," never to a failed prediction.
    Log-and-swallow, plus the prediction_log_write_failures_total counter
    so a sustained outage is
    discoverable rather than only ever silent.
    """
    try:
        rows = prediction_log.build_log_rows(
            feature_rows, scored, runtime, contact, route, request_id, resolution_kinds
        )
        await prediction_log.write_log_rows(rows, get_async_engine())
    except Exception:
        _PREDICTION_LOG_WRITE_FAILURES_TOTAL.labels(route=route).inc()
        logger.warning(
            "prediction_log_write_failed",
            request_id=request_id,
            route=route,
            exc_info=True,
        )


@app.post(
    "/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)]
)
async def predict_endpoint(
    payload: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    runtime: predict.ServingRuntime = Depends(get_runtime),
) -> PredictResponse:
    """Single-customer prediction — probability/threshold/decision, plus an
    opt-in local SHAP explanation. Never carries `contact`/`capacity_limit`:
    one customer in isolation has no population to be capacity-ranked
    against (serving/contact_policy.py::select_contacts is wired into
    /predict/batch alone, below).
    """
    cfg: DictConfig = request.app.state.cfg
    request_id = _request_id(request)
    response, scored = predict.predict_single(payload, runtime, cfg)
    logger.info(
        "prediction_served",
        request_id=request_id,
        customerid=response.customerid,
        features=payload.model_dump(exclude={"include_explanation", "resolution_kind"}),
        probability=response.probability,
        threshold=response.threshold,
        decision=response.decision,
        model_version=response.model_version,
    )
    if bool(cfg.serving.prediction_log.enabled):
        row = payload.model_dump(
            exclude={"customerid", "include_explanation", "resolution_kind"}
        )
        background_tasks.add_task(
            _log_prediction_background,
            [row],
            scored,
            runtime,
            None,
            "single",
            request_id,
            [payload.resolution_kind],
        )
    return response


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    dependencies=[Depends(require_api_key)],
)
async def predict_batch_endpoint(
    payload: list[dict[str, Any]],
    request: Request,
    background_tasks: BackgroundTasks,
    runtime: predict.ServingRuntime = Depends(get_runtime),
) -> BatchPredictResponse:
    """Batch scoring with partial success and capacity-constrained contact ranking.

    Accepts `list[dict[str, Any]]` rather than `list[CustomerFeatures]` so a
    malformed row can be collected as an error instead of 422-ing the whole
    batch — FastAPI validates a declared Pydantic-model list body in full
    before the handler ever runs, which partial success needs to bypass.

    `decision`/`contact` are computed against whichever model actually
    served each row — `decision_threshold` equals `threshold` for every row
    (both are that row's served model's own calibrated bar), rather than
    being pinned to the champion's `base` scenario `t*` regardless of who
    scored the row. Pinning to champion's threshold would make a
    canary-routed row's eligibility depend partly on an arbitrary
    hash-bucket assignment rather than the risk the serving model actually
    computed, and would disagree with `predict.py`'s
    `predictions_above_threshold_total` Prometheus counter, which already
    uses each row's own served threshold. The `contact_capacity`/
    `campaign_budget` cut downstream is still one pool shared by both
    cohorts — see `ANALYSIS.md` §9, item 14 for the capacity-interference
    limitation that remains.
    """
    cfg: DictConfig = request.app.state.cfg
    max_size = int(cfg.serving.batch.max_size)
    if len(payload) > max_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"batch of {len(payload)} rows exceeds batch.max_size="
                f"{max_size}; use pipelines/batch_predict.py (Phase 10) for "
                "a full population sweep"
            ),
        )

    request_id = _request_id(request)
    validated: dict[int, BatchPredictItem] = {}
    errors: list[BatchPredictionError] = []
    for idx, raw in enumerate(payload):
        try:
            validated[idx] = BatchPredictItem.model_validate(raw)
        except ValidationError as exc:
            errors.append(
                BatchPredictionError(
                    index=idx,
                    customerid=(
                        raw.get("customerid") if isinstance(raw, dict) else None
                    ),
                    reason="validation_error",
                    detail=customer_lookup.format_validation_error(exc),
                )
            )

    lookup_ids = sorted(
        {
            item.customerid
            for item in validated.values()
            if item.customerid is not None and not customer_lookup.is_complete(item)
        }
    )
    looked_up = (
        await _lookup_customers_or_503(lookup_ids, cfg, "batch") if lookup_ids else {}
    )
    rows, row_indices, row_customer_ids, resolution_errors, resolution_kinds = (
        customer_lookup.resolve_batch_rows(validated, looked_up)
    )
    errors.extend(resolution_errors)

    # Champion's own persisted cost scenario, used to rank every row
    # regardless of which model served it — including challenger-scored
    # canary rows, whose own persisted scenario is never consulted or
    # compared. See ANALYSIS.md §9, item 16.
    bundle = runtime.champion
    scenario = bundle.scenarios[bundle.base_scenario_name]
    costs_cfg = load_costs_config()

    scored: predict.ScoredBatch | None = None
    served_proba = np.array([], dtype=float)
    served_threshold = np.array([], dtype=float)
    if rows:
        scored = predict.predict_batch(
            pd.DataFrame(rows), pd.Series(row_customer_ids), runtime, cfg
        )
        served_proba = scored.served_proba
        served_threshold = scored.served_threshold

    contact_decision = select_contacts(
        served_proba,
        scenario,
        served_threshold,
        int(costs_cfg.contact_capacity),
        float(costs_cfg.campaign_budget),
    )
    if contact_decision.capacity_limit > 0:
        _CONTACT_CAPACITY_UTILIZATION.set(
            contact_decision.contact.sum() / contact_decision.capacity_limit
        )
    else:
        _CONTACT_CAPACITY_UTILIZATION.set(0.0)

    results: list[BatchPredictionResult] = []
    if scored is not None:
        for i in range(len(rows)):
            results.append(
                BatchPredictionResult(
                    index=row_indices[i],
                    customerid=row_customer_ids[i],
                    probability=float(scored.served_proba[i]),
                    threshold=float(scored.served_threshold[i]),
                    decision_threshold=float(scored.served_threshold[i]),
                    decision=bool(contact_decision.decision[i]),
                    contact=bool(contact_decision.contact[i]),
                    model_version=scored.served_model_version[i],
                    served_source=scored.served_source[i],
                )
            )
            logger.info(
                "prediction_served",
                request_id=request_id,
                customerid=row_customer_ids[i],
                features=rows[i],
                probability=float(scored.served_proba[i]),
                threshold=float(scored.served_threshold[i]),
                decision=bool(contact_decision.decision[i]),
                model_version=scored.served_model_version[i],
                served_source=scored.served_source[i],
            )

        if bool(cfg.serving.prediction_log.enabled):
            background_tasks.add_task(
                _log_prediction_background,
                rows,
                scored,
                runtime,
                contact_decision.contact.tolist(),
                "batch",
                request_id,
                resolution_kinds,
            )

    results.sort(key=lambda r: r.index)
    errors.sort(key=lambda e: e.index)
    return BatchPredictResponse(
        capacity_limit=contact_decision.capacity_limit,
        results=results,
        errors=errors,
    )


@app.get(
    "/customer/{customerid}",
    response_model=CustomerLookupResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_customer(customerid: str, request: Request) -> CustomerLookupResponse:
    """Read-only lookup of an existing customer's simulated-current state.

    Reads customers_crm, not customers_raw — a live lookup must hit a
    source distinct from the frozen training snapshot (see
    serving/customer_lookup.py). churn is deliberately excluded even though
    the raw column exists — it's the historical training-time outcome, not
    a live-monitored result, and surfacing it next to a freshly computed
    probability invites reading it as live validation.
    """
    found = await _lookup_customers_or_503(
        [customerid], request.app.state.cfg, "customer"
    )
    row = found.get(customerid)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"customerid '{customerid}' not found"
        )
    crm_snapshot_at = row["crm_snapshot_at"]
    features = CustomerFeatures.model_validate(row)
    return CustomerLookupResponse(features=features, crm_snapshot_at=crm_snapshot_at)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness only — no dependency checks. A Postgres/MLflow outage must
    never restart-loop a process that was never actually broken."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    """Readiness — 503 until the champion has loaded at least once.

    Scoped to the model-load check alone, deliberately not Postgres: POST
    /predict never touches Postgres at all, so gating readiness on it would
    pull the whole API out of rotation over a dependency /predict doesn't
    even need.
    """
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    bundle = request.app.state.runtime.champion
    return {
        "status": "ready",
        "model_version": bundle.model_version,
        "run_id": bundle.run_id,
    }
