# Group 9 manual bring-up runbook

One-off manual deploy of the serving stack (`api`, `ui`, `caddy`, `mlflow`)
onto the real EC2 box, done once by hand **before** Group 11's CD pipeline
exists to automate it. Run this from your own terminal with real AWS
credentials — none of it can run from a Claude Code sandboxed session (no
AWS creds, no SSM access). Matches this project's established two-terminal
pattern (`feedback_aws_tunnel_workflow`): one terminal for local
AWS-credentialed work, a second for the interactive SSM session on the box.

Prerequisites: Groups 0–8 complete (the box exists, Group 6.5's data
bootstrap has already run, `resolve_champion_model` already confirmed
loadable against RDS+S3).

## Phase A–D: local (`infra/scripts/manual-bring-up-local.sh`)

```bash
bash infra/scripts/manual-bring-up-local.sh
```

Does, in order:
- **A** — `terraform apply`: picks up the `t3.small` RAM bump and re-uploads
  `compose.prod.yml`/`Caddyfile`/`docker/mlflow/Dockerfile` to S3. The
  instance-type change is an in-place resize (AWS stops/modifies/starts the
  *same* instance and EBS volumes, not a replace) — expect ~1–2 min of SSM
  connectivity loss during that, not data loss.
- **B** — sets the real `/telco-churn/api-key` SSM param (Group 7 left this
  as a placeholder).
- **C** — builds and pushes real `api`/`ui` images to ECR, tagged with the
  current git short SHA. **The ECR repos are `image_tag_mutability =
  IMMUTABLE`** — re-running this against the same commit will fail on push;
  commit something first, or hand-edit the tag, if you need a retry.
- **D** — points `/telco-churn/api-image`/`/telco-churn/ui-image` at the
  real tags just pushed.

Ends by printing the instance ID and the `aws ssm start-session` command for
Phase E.

## Phase E: on the box (`infra/scripts/manual-bring-up-on-box.sh`)

Open the SSM session the previous phase printed, then paste that script's
commands in (it's a reference to copy from, not something you `scp`/execute
remotely — the point is watching it happen live). It:

1. Re-syncs `/opt/telco-churn/` from S3 (the box only pulled this once, at
   first boot — nothing has refreshed it since, so this step is required
   even though `terraform apply` already updated S3 itself).
2. Re-renders `.env` via `refresh-env.sh` (real API key, real image tags).
3. Runs the one-off `alembic upgrade head` against RDS.
4. `systemctl start telco-churn.service` — the actual first-ever start;
   auto-builds `mlflow` (Dockerfile now present locally), auto-pulls
   `api`/`ui`, brings `caddy` up last.
5. `docker compose ps` + `systemctl status` as an on-box sanity check
   before you back out to verify from outside.

## Phase F: verify from your own machine

**The Caddyfile currently points at Let's Encrypt's *staging* CA** (its own
comment: "iterate against staging first ... so the production CA's
low-volume rate limit isn't burned on retries") — so the cert is untrusted
on this first pass. Use `-k`/`--insecure` for now; see the follow-up below
once everything checks out.

```bash
export MSYS_NO_PATHCONV=1  # Git Bash on Windows only - otherwise it rewrites
                            # the leading-slash param name into a Windows path
                            # and aws.exe rejects it as "not fully qualified"
DOMAIN=telco-churn.duckdns.org
API_KEY=$(aws ssm get-parameter --name /telco-churn/api-key --with-decryption --query Parameter.Value --output text --region us-east-1)

curl -sk -o /dev/null -w "health: %{http_code}\n"  "https://$DOMAIN/health"
curl -sk -o /dev/null -w "ready:  %{http_code}\n"  "https://$DOMAIN/ready"
# /predict is POST-only - a bare GET 405s regardless of auth, which would
# look like a pass for the wrong reason. -X POST is required for both of
# these to actually exercise require_api_key.
curl -sk -X POST -o /dev/null -w "predict no key (expect 401): %{http_code}\n" "https://$DOMAIN/predict"
curl -sk -X POST -o /dev/null -w "predict with key (expect 422 - no body, not 401): %{http_code}\n" \
  -H "X-API-Key: $API_KEY" "https://$DOMAIN/predict"
curl -sk -o /dev/null -w "ui root: %{http_code}\n" "https://$DOMAIN/"
curl -sk -u "reviewer:telco-reviewer-2026" -o /dev/null -w "mlflow (expect 200): %{http_code}\n" "https://$DOMAIN/mlflow/"
curl -sk -o /dev/null -w "mlflow no auth (expect 401): %{http_code}\n" "https://$DOMAIN/mlflow/"
```

The `/mlflow` login comes from MLflow's own basic-auth app
(`compose.prod.yml`, `--app-name basic-auth`), not from Caddy. The `reviewer`
account is view-only: `default_permission = READ`, and the Caddyfile refuses
experiment/model creation and every user-management write for all callers.
`scripts/seed_mlflow_reviewer.sh` (run by `cd.yml` after each deploy) creates
the account, or resets its password if someone changed it.

**One thing this hasn't been live-tested against real traffic**: whether
Caddy's `reverse_proxy` really does pass the original `Host: $DOMAIN` header
through to the `mlflow` container unmodified (the design `compose.prod.yml`'s
`MLFLOW_SERVER_ALLOWED_HOSTS` comment relies on). If the `mlflow` curl above
403s instead of the expected 200, that's the first thing to check —
`sudo docker compose -f compose.prod.yml logs mlflow --tail 50` on the box
will show the rejected Host header if so, and the fix is adding whatever
Host Caddy is actually sending to `MLFLOW_SERVER_ALLOWED_HOSTS` in
`compose.prod.yml`.

## Follow-up, once everything above is green: switch to the production CA

```
infra/deploy/Caddyfile
```

Comment out or delete the `{ acme_ca https://acme-staging-v02... }` global
block, then re-run Phase A (`terraform apply` re-uploads it) and re-sync +
restart just `caddy` on the box:

```bash
sudo aws s3 cp s3://<bucket>/deploy-config/Caddyfile /opt/telco-churn/Caddyfile --region us-east-1
sudo docker compose -f /opt/telco-churn/compose.prod.yml up -d caddy
```

Re-run the Phase F curls without `-k` to confirm the real cert issued.

# CD operations (Group 11 — `.github/workflows/cd.yml`)

## What a deploy does

A push to `main` touching `src/`, `configs/`, `alembic*`, either Dockerfile,
`pyproject.toml` or `uv.lock` (or a manual `workflow_dispatch`) builds `api`/`ui`
tagged with the commit SHA, points `/telco-churn/{api,ui}-image` at them, and
runs `scripts/deploy_on_box.sh` then `scripts/smoke_test_deployed.sh` on the box
over SSM. `docker image prune -af` runs before the pull and again after a passing
smoke test (see disk space, below). A model
promotion is **not** a deploy — `champion` hot-reloads via the registry alias.

## One-time setup

- Repo variable `AWS_ACCOUNT_ID` (Settings → Secrets and variables → Actions → Variables).
- `terraform apply` from your own terminal for `iam.tf`'s `ImageParams` /
  `ListCommandInvocations` / `DeployConfigUpload` grants on `telco-churn-ci-deploy`,
  and for `compute.tf`'s `ignore_changes` (state-only; re-uploads nothing).

## Deploy files: Terraform seeds, CD owns

`compose.prod.yml`, `Caddyfile`, `refresh-env.sh` and `docker/mlflow/Dockerfile`
reach the box via S3 (`deploy-config/`). Terraform's `aws_s3_object` only
*seeds* them for a fresh environment's first boot (`ignore_changes` on
`etag`/`source`, so `terraform plan` never fights CD). Every CD run then syncs
`infra/deploy/` + `docker/mlflow/Dockerfile` to that prefix before the SSM
deploy — no manual `terraform apply` for a config change. The bucket is
versioned, so a bad overwrite is recoverable. Those paths are in `cd.yml`'s
trigger filter; `RUNBOOK.md` and `user-data.sh.tpl` are excluded from the sync.

## Automatic rollback

If anything after the SSM params are flipped fails (deploy or smoke test),
`cd.yml` writes the previous SHAs back to SSM, restores the previous commit's
deploy files to S3 (`github.event.before`), re-runs the same deploy, and
re-smoke-tests. The job **stays red either way**. The old image is pulled from
ECR (immutable tags, last five kept), so it does not matter that the box may
have dropped its layers.

Limits:
- **`workflow_dispatch` runs** have no `before` commit, so only images roll back;
  a bad deploy file needs a manual restore from the versioned bucket.
- **First-ever deploy** has no previous SHA (`:unset` placeholder) — no rollback, warning only.
- **Migrations are forward-only.** Rollback never runs `alembic downgrade`; the
  old image must tolerate the new schema, so write migrations backwards-compatibly.
- **Rollback failing** logs `ROLLBACK ALSO FAILED`. Intervene by hand, below.

## Disk space

The root volume is 19 GB and each api+ui release is about 2.5 GB, so old
releases must not accumulate. Both `deploy_on_box.sh` (before the pull) and
`prune_on_box.sh` (after a green smoke test) run `docker image prune -af`,
which removes every image no container is using — the release currently serving,
which is also the rollback target, is in use and stays. An older release that a
rollback needs is re-pulled from ECR, which keeps the last five tags.

Symptom of a full disk: "Deploy on the box" fails with the SSM agent's
`ipc messaging received timeout signal`, not a Docker error — the agent's own
log (`/var/log/amazon/ssm/errors.log`) says `no space left on device`. Check
`df -h /` and `docker system df` on the box, then `docker image prune -af`.

## Manual rollback (image)

**Git Bash on Windows:** prefix each `aws ssm` command with `MSYS_NO_PATHCONV=1`
(or use a leading `//`), otherwise MSYS rewrites `/telco-churn/...` into a
Windows path and AWS answers `ParameterNotFound`. Not needed on the box or in CI.

```bash
aws ssm put-parameter --name /telco-churn/api-image --type String --overwrite --value <ecr-uri>:<good-sha>
aws ssm put-parameter --name /telco-churn/ui-image  --type String --overwrite --value <ecr-uri>:<good-sha>
scripts/ssm_run.sh scripts/deploy_on_box.sh
scripts/ssm_run.sh scripts/smoke_test_deployed.sh
```

## Manual rollback (model) — a different mechanism

A bad *model* is not fixed by an image rollback. Re-point `champion` at the
highest version tagged `promotion_status: promoted` — never "the previous
version number" (see CLAUDE.md, MLflow Model Registry). The API hot-reloads it
on its next TTL poll.

# Teardown

Not run now — for the 2027-01-16 shared-credit-balance cutoff (Group 0's
second calendar reminder), or an earlier decision to stop paying for this
deployment. Three steps, run from your own terminal in the same sitting —
leaving the final snapshot behind after `destroy` is the one way this still
quietly costs money afterward.

1. **Drop RDS deletion protection first** — `terraform destroy` cannot
   delete `aws_db_instance.main` while `deletion_protection = true` (set
   deliberately, Group 6). Either flip it to `false` in `infra/data.tf` and
   `terraform apply` that one change, or pass `-modify-db-instance` by hand:
   ```bash
   aws rds modify-db-instance --db-instance-identifier telco-churn-db \
     --no-deletion-protection --apply-immediately --region us-east-1
   ```
2. **`terraform destroy`** — tears down everything Terraform manages (EC2,
   RDS, S3, ECR, the log groups and alarms above, IAM, networking). RDS
   takes its own `final_snapshot_identifier` snapshot on the way down
   (Group 6) — that snapshot is **not** itself a Terraform resource, so
   `destroy` cannot remove it.
3. **Delete the final snapshot in this same sitting**, not later —
   forgetting this step is exactly how a $0 teardown keeps a real ~$1.90/mo
   charge running silently in the background:
   ```bash
   aws rds describe-db-snapshots --db-instance-identifier telco-churn-db \
     --query "DBSnapshots[].DBSnapshotIdentifier" --region us-east-1
   aws rds delete-db-snapshot --db-snapshot-identifier <identifier-from-above> \
     --region us-east-1
   ```

Not covered here, and not needed for a cost teardown: the MLflow experiment
history / model registry itself lives in S3 + the RDS `mlflow` database,
both already gone after step 2 — there is nothing further to clean up in
MLflow once the infrastructure underneath it is destroyed.
