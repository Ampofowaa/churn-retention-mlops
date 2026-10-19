variable "aws_region" {
  description = "AWS region for every resource this project creates"
  type        = string
  default     = "us-east-1"
}

variable "alert_email" {
  description = "Address for SNS alarm notifications and AWS Budgets alerts; set via TF_VAR_alert_email or a gitignored tfvars file"
  type        = string
}

variable "billing_alarm_thresholds_usd" {
  description = "EstimatedCharges alarm thresholds, keyed by the dollar figure used in the alarm name"
  type        = map(number)
  default = {
    "15"  = 15
    "100" = 100
    "140" = 140
  }
}

variable "monthly_budget_usd" {
  description = "AWS Budgets monthly limit; alerts fire at 50/80/100% of it"
  type        = number
  default     = 40
}

variable "project_name" {
  description = "Short name used as a prefix/tag across all resources"
  type        = string
  default     = "telco-churn"
}

variable "cpu_credit_balance_alarm_threshold" {
  description = "Low-balance CPUCreditBalance alarm threshold — a judgment call, best tuned once there's real traffic history rather than guessed upfront"
  type        = number
  default     = 50
}
