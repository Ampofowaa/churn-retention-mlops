# FastAPI serving image. Built from the project's own uv.lock — never from
# pyproject.toml ranges resolved fresh at build time. register.py's
# golden-parity smoke check compares reloaded scores against
# golden_predictions.json at atol=1e-9, and that tolerance is only
# well-posed if numpy/scikit-learn/lightgbm never drift from the pinned
# versions calibrate.py logged the model under (a patch-release bump shifts
# floating-point output in the same 1e-7-1e-9 range as the tolerance itself,
# with no actual bug present). This image's dependencies are ordinary,
# project-versioned software — built once per code push, tested through
# ci.yml/cd.yml the normal way — which is what lets serving/predict.py's
# hot-reload design work at all: a container whose libraries are pinned
# per-model-build would have no stable "installed environment" for the
# reload guard (environment_parity.py::diff_environment) to diff a new
# champion's logged deps against.
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra serving

COPY src/ ./src/
COPY configs/ ./configs/
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/
RUN uv sync --frozen --no-dev --extra serving

FROM python:3.14-slim AS runtime

# libgomp1: LightGBM's OpenMP runtime dependency — not present in the slim
# base image; without it, `import lightgbm` fails at ctypes.cdll.LoadLibrary
# with "libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/configs /app/configs
# pyproject.toml: utils/paths.py::get_project_root() walks up from cwd
# looking for it as the project-root marker — importing telco_churn.models
# (predict.py's own import chain) transitively imports models/train/tuning.py,
# which resolves a sql/schema/*.sql Path at module level and needs this to
# succeed even though nothing at runtime reads that file.
COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml
# cd.yml's deploy sequence runs `docker compose run --rm api alembic upgrade
# head` against this same image - without these two, that step has no
# migration files to apply.
COPY --from=builder --chown=app:app /app/alembic.ini /app/alembic.ini
COPY --from=builder --chown=app:app /app/alembic /app/alembic

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# serving/app.py (Task 4) is the FastAPI entry point this runs.
CMD ["uvicorn", "telco_churn.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
