# Group 8 — EC2 instance, AMI lookup, EIP association, user-data.
#
# Ubuntu 24.04 LTS, not Amazon Linux 2023: SSM Agent ships pre-installed on
# Canonical's official AMI either way, so the SSM-only shell-access design
# (Session Manager, Run Command, the Group 6.5 RDS tunnel) is unaffected -
# Ubuntu wins on `apt` tooling familiarity and not being AWS-specific.

data "aws_ssm_parameter" "ubuntu_ami" {
  name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

# Terraform owns these three files as content-hashed S3 uploads, not
# user-data heredocs and not a git clone on the box - an edit to any one of
# them re-uploads on the next `terraform apply`, and Group 11's CD command
# re-syncs this same prefix on every deploy so a compose.prod.yml/Caddyfile
# change ships the same way an image-tag update does.
locals {
  deploy_config_files = {
    "compose.prod.yml" = "${path.module}/deploy/compose.prod.yml"
    "refresh-env.sh"   = "${path.module}/deploy/refresh-env.sh"
    "Caddyfile"        = "${path.module}/deploy/Caddyfile"
    "config.alloy"     = "${path.module}/deploy/config.alloy"
    # compose.prod.yml's mlflow service is the one prod service with a real
    # build: block (no CD pipeline mints it an image) - the box needs the
    # Dockerfile itself, not just a reference to it. Same file local dev's
    # docker-compose.yml builds from, uploaded under a key that preserves
    # its subpath so `aws s3 cp --recursive` (user-data.sh.tpl) lands it at
    # /opt/telco-churn/docker/mlflow/Dockerfile - exactly where
    # compose.prod.yml's `context: ./docker/mlflow` expects it.
    "docker/mlflow/Dockerfile" = "${path.module}/../docker/mlflow/Dockerfile"
  }
}

resource "aws_s3_object" "deploy_config" {
  for_each = local.deploy_config_files

  bucket = aws_s3_bucket.mlflow.id
  key    = "deploy-config/${each.key}"
  source = each.value
  etag   = filemd5(each.value)

  # Terraform only *seeds* these objects (the first boot's user-data needs
  # them present before any CD run exists). From then on cd.yml owns them —
  # it `aws s3 sync`s infra/deploy/ + docker/mlflow/Dockerfile here on every
  # deploy. Without ignore_changes, the next `terraform plan` would see
  # CD's newer content as drift and try to revert it. The bucket is
  # versioned, so an overwrite is always recoverable. Tags too: `aws s3 sync`
  # re-uploads each object without tags, so every plan after a deploy would
  # otherwise want to re-add the provider's default_tags, only for the next
  # deploy to strip them again.
  lifecycle {
    ignore_changes = [etag, source, tags, tags_all]
  }
}

resource "aws_instance" "main" {
  ami = data.aws_ssm_parameter.ubuntu_ami.value
  # Bumped from t3.micro (1GB) 2026-09-18, Group 9: the reviewer-facing
  # MLflow UI service (~150-250MB) doesn't fit alongside api (~300-500MB)
  # + ui (~150MB) + caddy (~15MB) + CloudWatch agent (~50MB) + Grafana
  # Alloy (~30-100MB) + OS (~150MB), which already sits at ~730-950MB
  # before adding it. t3.small is 2GB. Real recurring cost impact: ~+$7.59/mo
  # (t3.micro ~$7.59 -> t3.small ~$15.18/mo), bringing this phase's total
  # from ~$28-30/mo to ~$36-38/mo.
  instance_type = "t3.small"

  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_instance.name
  key_name               = null # SSM Session Manager is the only shell-access path

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  # Two known ForceNew footguns, both fixed 2026-09-16 after back-to-back
  # instance replacements in practice:
  #  - ami: ubuntu_ami tracks Canonical's "current" build, not a pinned ID -
  #    any apply after Canonical publishes a newer AMI would otherwise
  #    force-replace this instance.
  #  - associate_public_ip_address: explicitly setting this to false (to
  #    dodge a second billed public IPv4) is a known AWS-provider footgun on
  #    instances launched into a subnet with map_public_ip_on_launch = true -
  #    AWS doesn't always report it back consistently, so Terraform proposed
  #    replacing the instance on every single apply. Left unset (Computed)
  #    instead, accepting the ~$3.65/mo redundant public IPv4 as the price of
  #    not destroying the box on every apply.
  #  - user_data: also ForceNew, same footgun class as ami above - found
  #    2026-10-19 when a user-data.sh.tpl edit (CloudWatch agent
  #    append_dimensions fix) first showed up in `terraform plan` as
  #    "must be replaced" on the live production box. A template edit should
  #    never destroy the running instance; the template still documents the
  #    correct first-boot config for whenever a real replacement happens
  #    (AMI rotation, instance-type change), same split the ECR
  #    credential-helper fix already relies on - a template change only
  #    benefits a *future* instance, the *current* one needs the equivalent
  #    fix applied by hand via SSM.
  lifecycle {
    ignore_changes = [ami, associate_public_ip_address, user_data]
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  user_data = templatefile("${path.module}/deploy/user-data.sh.tpl", {
    s3_bucket  = aws_s3_bucket.mlflow.id
    aws_region = var.aws_region
    account_id = data.aws_caller_identity.current.account_id
  })

  # Guarantees the files exist in S3 before first boot's `aws s3 cp` runs -
  # nothing in the instance's own arguments references the object resources
  # directly, so this dependency wouldn't otherwise be implicit.
  depends_on = [aws_s3_object.deploy_config]

  tags = {
    Name = "${var.project_name}-ec2"
  }
}

resource "aws_eip_association" "main" {
  instance_id   = aws_instance.main.id
  allocation_id = aws_eip.instance.id
}
