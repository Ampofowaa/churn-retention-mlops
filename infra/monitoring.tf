# Group 12 — CloudWatch log groups/alarms, SNS topic, billing alarms + AWS Budgets.
#
# Billing guardrails land ahead of the rest of Group 12 because they protect the
# credit balance while Groups 1–11 create billable resources. The SNS topic is
# shared: Group 12's operational alarms attach to aws_sns_topic.alerts too.
#
# AWS/Billing metrics exist only in us-east-1, which is this project's region.

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "billing" {
  for_each = var.billing_alarm_thresholds_usd

  alarm_name          = "${var.project_name}-estimated-charges-${each.key}usd"
  alarm_description   = "Month-to-date EstimatedCharges exceeded ${each.key} USD"
  namespace           = "AWS/Billing"
  metric_name         = "EstimatedCharges"
  dimensions          = { Currency = "USD" }
  statistic           = "Maximum"
  period              = 21600 # billing metric updates a few times a day
  evaluation_periods  = 1
  threshold           = each.value
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
}

# include_credit = false so the budget tracks gross usage; with credits netted
# off it would read ~$0 for the whole credit lifetime and never alert.
resource "aws_budgets_budget" "monthly" {
  name         = "${var.project_name}-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_types {
    include_credit = false
    include_refund = false
  }

  dynamic "notification" {
    for_each = [50, 80, 100]

    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}

# --- Log groups (Terraform-managed, 14d retention) ----------------------
#
# Docker's awslogs-create-group stays "true" in compose.prod.yml as a
# harmless no-op fallback once these already exist — pre-creating them here
# is what actually sets retention, since the driver's own create-group path
# never sets one (AWS default: logs never expire). Names must match both
# compose.prod.yml's awslogs-group values and iam.tf's local.log_group_arn
# wildcard exactly.
resource "aws_cloudwatch_log_group" "services" {
  for_each = toset(["api", "ui", "caddy", "mlflow", "alloy"])

  name              = "/telco-churn/${each.key}"
  retention_in_days = 14
}

# --- Operational alarms (CPU/status/mem/disk), shared SNS topic ---------
#
# CPUUtilization/StatusCheckFailed/CPUCreditBalance need no CloudWatch
# agent — they're hypervisor-level, visible to AWS from outside the
# instance. mem_used_percent/disk_used_percent need the agent already
# configured in user-data.sh.tpl, since memory/disk usage are OS-level
# facts the hypervisor can't see.
#
# treat_missing_data = "breaching" on all four below (not the billing
# alarm's "notBreaching") — if the agent or the instance itself goes
# silent, that must not read as "fine," exactly the moment something is
# most likely wrong. Billing data is legitimately sparse and keeps
# "notBreaching".

resource "aws_cloudwatch_metric_alarm" "cpu_utilization" {
  alarm_name          = "${var.project_name}-cpu-utilization-high"
  alarm_description   = "CPUUtilization > 90% for 15m. Check for a runaway process or a genuine traffic surge — also check CPUCreditBalance, since this box runs in Unlimited credit mode."
  namespace           = "AWS/EC2"
  metric_name         = "CPUUtilization"
  dimensions          = { InstanceId = aws_instance.main.id }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 90
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "breaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "status_check_failed" {
  alarm_name          = "${var.project_name}-status-check-failed"
  alarm_description   = "StatusCheckFailed >= 1. StatusCheckFailed_System: stop/start (migrates to new hardware). StatusCheckFailed_Instance: reboot (guest OS issue — corrupted filesystem, OOM, kernel panic)."
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed"
  dimensions          = { InstanceId = aws_instance.main.id }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "breaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "mem_used_percent" {
  alarm_name          = "${var.project_name}-mem-used-percent-high"
  alarm_description   = "mem_used_percent > 90%. This box has OOM'd twice before (Group 9 incidents) — SSM onto it and check `ps aux --sort=-%mem` for a leak before assuming it's transient."
  namespace           = "telco-churn"
  metric_name         = "mem_used_percent"
  dimensions          = { InstanceId = aws_instance.main.id }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 1
  threshold           = 90
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "breaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "disk_used_percent" {
  alarm_name        = "${var.project_name}-disk-used-percent-high"
  alarm_description = "disk_used_percent > 90%. Known cause: old SHA-tagged release images piling up — confirm `docker image prune -af` is actually running in prune_on_box.sh before investigating further."
  namespace         = "telco-churn"
  metric_name       = "disk_used_percent"
  # Unlike mem_used_percent, the agent's disk plugin auto-adds path/device/
  # fstype dimensions (to distinguish which mounted filesystem) on top of
  # append_dimensions - found 2026-10-19 when this alarm stayed stuck on
  # "no datapoints" even after the InstanceId dimension fix that recovered
  # mem_used_percent. CloudWatch metric identity is the FULL dimension set,
  # so an alarm naming only InstanceId was querying a metric that doesn't
  # exist. device/fstype are hardcoded to this box's actual root volume
  # (confirmed via `aws cloudwatch list-metrics`) - stable for a single
  # gp3 EBS root volume on this AMI, but would need updating if the root
  # device ever changes (e.g. a different instance type's NVMe naming).
  dimensions = {
    InstanceId = aws_instance.main.id
    path       = "/"
    device     = "nvme0n1p1"
    fstype     = "ext4"
  }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 1
  threshold           = 90
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "breaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
}

# CPUCreditBalance — a cost-surprise early warning, not a throttling guard.
# aws_instance.main has no credit_specification block, i.e. runs AWS's
# actual T3 default (Unlimited mode) — deliberately kept as-is, not changed,
# 2026-10-19 (see docs/monitoring-metrics.md). In Unlimited mode, running
# out of credits means a per-vCPU-hour surplus charge, not throttling, so
# this alarm's whole purpose is advance notice before any surplus billing
# starts. treat_missing_data stays "notBreaching" here, unlike the four
# above — stale data on this specific alarm isn't worth paging on.
resource "aws_cloudwatch_metric_alarm" "cpu_credit_balance_low" {
  alarm_name          = "${var.project_name}-cpu-credit-balance-low"
  alarm_description   = "CPUCreditBalance below ${var.cpu_credit_balance_alarm_threshold}. This box runs in Unlimited credit mode, so a low balance risks a per-vCPU-hour surplus charge, not throttling — check recent CPUUtilization for what's burning credits."
  namespace           = "AWS/EC2"
  metric_name         = "CPUCreditBalance"
  dimensions          = { InstanceId = aws_instance.main.id }
  statistic           = "Minimum"
  period              = 300
  evaluation_periods  = 1
  threshold           = var.cpu_credit_balance_alarm_threshold
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
}

# Native CloudWatch dashboard mirroring the 5 operational alarms above, so
# they're watchable in one console view next to where they fire — not a
# replacement for the Grafana Cloud dashboard, which only covers the
# Prometheus/Alloy app-side metrics and has no visibility into these
# host-level CloudWatch metrics at all. Free under CloudWatch's
# 3-dashboards tier.
#
# First cut (2026-10-19) tried embedding each alarm's ARN via
# `annotations.alarms` on its matching metric widget, to draw the threshold
# as a red line on the graph - PutDashboard rejected all 5 widgets with a
# schema "oneOf" error. Per AWS's own Dashboard Body Structure docs: a
# metric widget's `annotations` field is for alarm annotations OR
# horizontal/vertical annotations, never both, and "if you specify an alarm
# annotation, you cannot also specify a `metrics` array in the same
# widget" - the two are mutually exclusive, not layerable. Fixed by using
# the dedicated `alarm` widget type instead (one combined status widget,
# all 5 alarms, sorted so anything not OK surfaces first) alongside the 5
# plain metric widgets with no `annotations` block.
resource "aws_cloudwatch_dashboard" "operational" {
  dashboard_name = "${var.project_name}-operational"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "alarm"
        x      = 0
        y      = 0
        width  = 24
        height = 3
        properties = {
          title  = "Alarm status"
          sortBy = "stateUpdatedTimestamp"
          alarms = [
            aws_cloudwatch_metric_alarm.cpu_utilization.arn,
            aws_cloudwatch_metric_alarm.status_check_failed.arn,
            aws_cloudwatch_metric_alarm.mem_used_percent.arn,
            aws_cloudwatch_metric_alarm.disk_used_percent.arn,
            aws_cloudwatch_metric_alarm.cpu_credit_balance_low.arn,
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 3
        width  = 12
        height = 6
        properties = {
          title   = "CPU utilization"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/EC2", "CPUUtilization", "InstanceId", aws_instance.main.id, { stat = "Average", period = 300 }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 3
        width  = 12
        height = 6
        properties = {
          title   = "Status check failed"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/EC2", "StatusCheckFailed", "InstanceId", aws_instance.main.id, { stat = "Maximum", period = 60 }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 9
        width  = 12
        height = 6
        properties = {
          title   = "Memory used %"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["telco-churn", "mem_used_percent", "InstanceId", aws_instance.main.id, { stat = "Average", period = 300 }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 9
        width  = 12
        height = 6
        properties = {
          title   = "Disk used %"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          # Full dimension set required - see the matching comment on
          # aws_cloudwatch_metric_alarm.disk_used_percent above.
          metrics = [
            ["telco-churn", "disk_used_percent", "InstanceId", aws_instance.main.id, "path", "/", "device", "nvme0n1p1", "fstype", "ext4", { stat = "Average", period = 300 }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 15
        width  = 12
        height = 6
        properties = {
          title   = "CPU credit balance"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/EC2", "CPUCreditBalance", "InstanceId", aws_instance.main.id, { stat = "Minimum", period = 300 }]
          ]
        }
      }
    ]
  })
}
