# Streamlit UI image. Built from the project's own uv.lock, same discipline
# as docker/api/Dockerfile — this is a thin HTTP client over the FastAPI
# service (see ui/streamlit_app.py's module docstring). The "About this
# model" tab imports telco_churn.models.artifacts (an MLflow alias/metadata
# lookup, not a model load), and models/__init__.py's lazy re-export keeps
# that from pulling in telco_churn.models.train — so, unlike docker/api/,
# lightgbm is never actually imported here and no libgomp1 runtime
# dependency is needed (pyproject.toml's extras-split comment confirms ui
# needs neither shap nor lightgbm's training tooling).
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra ui

COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --no-dev --extra ui

FROM python:3.14-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/configs /app/configs
# pyproject.toml: utils/paths.py::get_project_root() walks up from cwd
# looking for it as the project-root marker — streamlit_app.py's own
# compose_config()/activate_config() call (for the "About this model" tab)
# needs this to resolve, same reason docker/api/Dockerfile carries it.
COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml
# The Batch Prediction tab offers this as a one-click sample download.
COPY --chown=app:app examples/sample_batch_predictions.csv /app/examples/sample_batch_predictions.csv
# Theme only, never the whole .streamlit/ folder: a local secrets.toml there
# is gitignored but still inside the build context.
COPY --chown=app:app .streamlit/config.toml /app/.streamlit/config.toml

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app
EXPOSE 8501

CMD ["streamlit", "run", "src/telco_churn/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
