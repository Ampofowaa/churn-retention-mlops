# Group 7 — SSM Parameter Store entries.
#
# Four ownership models across twelve parameters, not two — each mismatch
# here is a real Terraform-fights-reality bug, not a style choice:
#   1. Human-generated  (api-key, grafana-*, mlflow-admin-password,
#                        mlflow-flask-secret-key) -> ignore_changes on value
#   2. CD-mutated       (api-image, ui-image)     -> ignore_changes on value
#   3. Terraform-derived (rds-url-*, s3-bucket)   -> no ignore_changes, self-corrects
#   4. Static, human-picked literal (domain)      -> plain resource, either works
#
# All twelve share the /telco-churn/* path, already covered by iam.tf's
# SsmParams/SsmParamsKms statements on the EC2 instance role — no new IAM
# grant needed here. SecureString values use the default alias/aws/ssm KMS
# key (what kms_ssm_key_arn already grants kms:Decrypt on) — no custom
# key_id. Tier defaults to Standard (free) everywhere; never set Advanced.

# --- Human-generated, ignore_changes required ---------------------------
#
# Terraform creates each with a placeholder; a human overwrites the real
# value afterwards via `aws ssm put-parameter --overwrite`. Without
# ignore_changes, the next unrelated `terraform apply` reverts that
# hand-set value back to the placeholder.

resource "aws_ssm_parameter" "api_key" {
  name        = "/telco-churn/api-key"
  description = "API key for the three data routes (/predict, /predict/batch, /customer/{id}); set by hand with openssl rand -hex 32"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-api-key"
  }
}

resource "aws_ssm_parameter" "grafana_endpoint" {
  name        = "/telco-churn/grafana-endpoint"
  description = "Grafana Cloud remote_write endpoint; set by hand from the Grafana Cloud console"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-grafana-endpoint"
  }
}

resource "aws_ssm_parameter" "grafana_api_key" {
  name        = "/telco-churn/grafana-api-key"
  description = "Grafana Cloud Access Policy token for Alloy's remote_write basic_auth password; set by hand from the Grafana Cloud console"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-grafana-api-key"
  }
}

# Confirmed 2026-10-19 against a real Grafana Cloud stack: remote_write auth
# is basic_auth (username = numeric Stack ID), not a bare bearer token as
# config.alloy originally assumed. Not a secret on its own (a public-facing
# stack identifier), but kept in SSM alongside the other two for one
# refresh-env.sh pull and because Terraform has no simpler per-human-value
# story than this one.
resource "aws_ssm_parameter" "grafana_username" {
  name        = "/telco-churn/grafana-username"
  description = "Grafana Cloud numeric Stack ID; the username half of Alloy's remote_write basic_auth. Set by hand from the Grafana Cloud console"
  type        = "String"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-grafana-username"
  }
}

# MLflow's basic-auth app (the reviewer-facing UI's login). The admin
# password only seeds the admin account on first start against an empty
# mlflow_auth database; the Flask secret key must stay fixed or CSRF tokens
# stop validating across restarts. Both: openssl rand -hex 32.
resource "aws_ssm_parameter" "mlflow_admin_password" {
  name        = "/telco-churn/mlflow-admin-password"
  description = "Admin password for the MLflow UI's basic-auth app; set by hand with openssl rand -hex 32"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-mlflow-admin-password"
  }
}

resource "aws_ssm_parameter" "mlflow_flask_secret_key" {
  name        = "/telco-churn/mlflow-flask-secret-key"
  description = "Static Flask secret key (CSRF) for the MLflow UI's basic-auth app; set by hand with openssl rand -hex 32"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-mlflow-flask-secret-key"
  }
}

# --- CD-mutated, ignore_changes required ---------------------------------
#
# cd.yml (Group 11) overwrites these on every merge to main. Without
# ignore_changes, an unrelated terraform apply (a Group 12b tweak, a stray
# apply after a drift-checking plan) reverts a live deployment back to
# whatever placeholder is hardcoded here.

resource "aws_ssm_parameter" "api_image" {
  name        = "/telco-churn/api-image"
  description = "Full ECR image URI:tag currently deployed for api; mutated by cd.yml"
  type        = "String"
  value       = "${aws_ecr_repository.this["api"].repository_url}:unset"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-api-image"
  }
}

resource "aws_ssm_parameter" "ui_image" {
  name        = "/telco-churn/ui-image"
  description = "Full ECR image URI:tag currently deployed for ui; mutated by cd.yml"
  type        = "String"
  value       = "${aws_ecr_repository.this["ui"].repository_url}:unset"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-ui-image"
  }
}

# --- Fully Terraform-derived, no ignore_changes (self-corrects) ---------
#
# Separate rds-url-* params, not one shared rds-url: a Postgres connection
# string names exactly one database, and this instance holds three
# (telco_churn, mlflow, mlflow_auth) on the same host/credentials.

resource "aws_ssm_parameter" "rds_url_app" {
  name        = "/telco-churn/rds-url-app"
  description = "Postgres connection string for the app database (customers_raw, customers_crm, prediction_log, ...); renders as POSTGRES_URL"
  type        = "SecureString"
  value       = "postgresql://${aws_db_instance.main.username}:${random_password.rds_master.result}@${aws_db_instance.main.address}:${aws_db_instance.main.port}/${aws_db_instance.main.db_name}"

  tags = {
    Name = "${var.project_name}-rds-url-app"
  }
}

resource "aws_ssm_parameter" "rds_url_mlflow" {
  name        = "/telco-churn/rds-url-mlflow"
  description = "Postgres connection string for the mlflow tracking database; renders as MLFLOW_TRACKING_URI (direct DB URI, no MLflow server process)"
  type        = "SecureString"
  value       = "postgresql://${aws_db_instance.main.username}:${random_password.rds_master.result}@${aws_db_instance.main.address}:${aws_db_instance.main.port}/mlflow"

  tags = {
    Name = "${var.project_name}-rds-url-mlflow"
  }
}

# Users and permissions for the MLflow UI's basic-auth app. Its own
# database, not tables inside mlflow: the auth app runs its own migrations,
# and keeping them apart means neither schema's upgrades can touch the other.
resource "aws_ssm_parameter" "rds_url_mlflow_auth" {
  name        = "/telco-churn/rds-url-mlflow-auth"
  description = "Postgres connection string for the MLflow UI's basic-auth users/permissions database"
  type        = "SecureString"
  value       = "postgresql://${aws_db_instance.main.username}:${random_password.rds_master.result}@${aws_db_instance.main.address}:${aws_db_instance.main.port}/mlflow_auth"

  tags = {
    Name = "${var.project_name}-rds-url-mlflow-auth"
  }
}

resource "aws_ssm_parameter" "s3_bucket" {
  name        = "/telco-churn/s3-bucket"
  description = "MLflow artifact-store bucket name"
  type        = "String"
  value       = aws_s3_bucket.mlflow.id

  tags = {
    Name = "${var.project_name}-s3-bucket"
  }
}

# --- Static, human-picked literal ----------------------------------------

resource "aws_ssm_parameter" "domain" {
  name        = "/telco-churn/domain"
  description = "Public DuckDNS domain serving both api and ui behind Caddy's path-based routing"
  type        = "String"
  value       = "telco-churn.duckdns.org"

  tags = {
    Name = "${var.project_name}-domain"
  }
}
