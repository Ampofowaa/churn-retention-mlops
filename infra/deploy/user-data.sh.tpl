#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y ca-certificates curl gnupg unzip jq

# Docker's own apt repo, not Ubuntu archive's older docker.io package -
# needed for the `docker compose` plugin (docker-compose-plugin), not the
# standalone docker-compose binary.
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable --now docker

# Amazon ECR credential helper - IAM permission to pull (iam.tf's EcrAuth/
# EcrPull statements on this instance's own role) is not the same as being
# logged in: Docker itself has no registry credentials until something
# exchanges that IAM permission for one, and a manual `docker login` token
# expires after 12h with nothing to refresh it. This makes every
# `docker pull`/`docker compose pull` on this box transparently
# re-authenticate via the instance role, indefinitely, with no login step
# ever needed - the standard AWS-recommended fix for exactly this "long-
# running box pulling from ECR" case. Found missing during Group 9's first
# real manual bring-up (2026-09-19): `docker compose run --rm api alembic
# upgrade head` failed with "no basic auth credentials" the first time it
# tried to pull a real image, since nothing before this had ever configured
# Docker's registry auth on the box itself.
ECR_CRED_HELPER_VERSION="0.12.0"
curl -fsSL "https://amazon-ecr-credential-helper-releases.s3.us-east-2.amazonaws.com/$ECR_CRED_HELPER_VERSION/linux-amd64/docker-credential-ecr-login" \
  -o /usr/local/bin/docker-credential-ecr-login
curl -fsSL "https://amazon-ecr-credential-helper-releases.s3.us-east-2.amazonaws.com/$ECR_CRED_HELPER_VERSION/linux-amd64/docker-credential-ecr-login.sha256" \
  -o /tmp/docker-credential-ecr-login.sha256
(cd /usr/local/bin && sha256sum -c /tmp/docker-credential-ecr-login.sha256)
rm -f /tmp/docker-credential-ecr-login.sha256
chmod +x /usr/local/bin/docker-credential-ecr-login

mkdir -p /root/.docker
cat > /root/.docker/config.json <<DOCKERCONFIG
{
  "credHelpers": {
    "${account_id}.dkr.ecr.${aws_region}.amazonaws.com": "ecr-login"
  }
}
DOCKERCONFIG

# AWS CLI v2 - refresh-env.sh calls `aws ssm get-parameters-by-path`, and
# user-data below calls `aws s3 cp`.
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

# CloudWatch agent - host-level metrics only (mem/disk). Container logs ship
# via compose.prod.yml's awslogs logging driver instead, so this agent never
# tails a container log file or unwraps a json-file envelope. Full log-group/
# alarm setup is Group 12 - this just gets the agent running with a minimal
# config so Group 12 only has to narrow/extend it, not stand it up from zero.
curl -fsSL "https://amazoncloudwatch-agent-${aws_region}.s3.${aws_region}.amazonaws.com/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb" -o /tmp/amazon-cloudwatch-agent.deb
dpkg -i -E /tmp/amazon-cloudwatch-agent.deb
rm -f /tmp/amazon-cloudwatch-agent.deb

mkdir -p /opt/aws/amazon-cloudwatch-agent/etc
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'CWCONFIG'
{
  "metrics": {
    "namespace": "telco-churn",
    "append_dimensions": { "InstanceId": "$${aws:InstanceId}" },
    "metrics_collected": {
      "mem": { "measurement": ["mem_used_percent"] },
      "disk": { "measurement": ["disk_used_percent"], "resources": ["/"] }
    }
  }
}
CWCONFIG

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# compose.prod.yml/refresh-env.sh/Caddyfile are Terraform-owned aws_s3_object
# uploads (infra/compute.tf), not baked into this script and not git-cloned
# onto the box - this just fetches whatever Terraform last put there.
mkdir -p /opt/telco-churn
aws s3 cp "s3://${s3_bucket}/deploy-config/" /opt/telco-churn/ --recursive --region "${aws_region}"
chmod +x /opt/telco-churn/refresh-env.sh

cat > /etc/systemd/system/telco-churn.service <<'UNIT'
[Unit]
Description=Telco Churn serving stack
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/telco-churn
ExecStartPre=/opt/telco-churn/refresh-env.sh
ExecStart=/usr/bin/docker compose -f compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f compose.prod.yml down

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable telco-churn.service
# Not `enable --now`: on this very first boot, api-image/ui-image are still
# Group 7's ":unset" placeholders, so `docker compose up -d` would fail with
# no image to pull. Enabled (starts on every future boot/reboot) but started
# by hand only once Group 9 has pushed real images and set real tags.
