#!/usr/bin/env bash
set -euo pipefail

# Renders /opt/telco-churn/.env from every parameter under /telco-churn/*.
#
# Runs as systemd's ExecStartPre before `docker compose up` (Group 8) and
# again as part of cd.yml's deploy sequence via ssm:SendCommand (Group 11) —
# a fresh image tag needs the refreshed API_IMAGE/UI_IMAGE values in place
# before the next `docker compose pull`.
#
# Param-name -> env-var-name is an explicit map, not a blind
# uppercase-and-dash-to-underscore transform: rds-url-app/rds-url-mlflow
# fan out to POSTGRES_URL/MLFLOW_TRACKING_URI, names that don't match their
# SSM parameter's own name.
#
# Runtime deps this script assumes are already on the box (Group 8
# user-data installs both): aws-cli v2, jq.

ENV_FILE="/opt/telco-churn/.env"
PREFIX="/telco-churn"
REGION="us-east-1"

params_json=$(aws ssm get-parameters-by-path \
  --path "$PREFIX" \
  --recursive \
  --with-decryption \
  --region "$REGION" \
  --output json)

get_param() {
  echo "$params_json" | jq -r --arg name "$PREFIX/$1" '.Parameters[] | select(.Name == $name) | .Value'
}

umask 077
cat > "$ENV_FILE" <<EOF
API_KEY=$(get_param "api-key")
GRAFANA_ENDPOINT=$(get_param "grafana-endpoint")
GRAFANA_API_KEY=$(get_param "grafana-api-key")
GRAFANA_USERNAME=$(get_param "grafana-username")
API_IMAGE=$(get_param "api-image")
UI_IMAGE=$(get_param "ui-image")
POSTGRES_URL=$(get_param "rds-url-app")
MLFLOW_TRACKING_URI=$(get_param "rds-url-mlflow")
S3_BUCKET=$(get_param "s3-bucket")
DOMAIN=$(get_param "domain")
MLFLOW_ADMIN_PASSWORD=$(get_param "mlflow-admin-password")
MLFLOW_FLASK_SERVER_SECRET_KEY=$(get_param "mlflow-flask-secret-key")
MLFLOW_AUTH_DATABASE_URI=$(get_param "rds-url-mlflow-auth")
EOF

chmod 600 "$ENV_FILE"
