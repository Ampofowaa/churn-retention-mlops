# Monitoring Metrics Reference

Every metric this project's monitoring stack produces or will produce — what
each one measures, where it comes from, what the labels mean, and what to
chart or alert on once Group 12's Grafana dashboard and CloudWatch alarms
exist. Two distinct backends, covered in their own sections below: the
**serving-layer** Prometheus metrics exposed by `serving/app.py`'s
`GET /metrics`, and the **infrastructure-layer** CloudWatch metrics the EC2
instance and its CloudWatch agent produce. `docs/architecture.md` documents
the shadow/canary *mechanism* the serving metrics instrument; this doc is
the operational companion — built to answer "what panel or alarm do I build
for this," not "why does the code work this way."

**Scope.** Statistical data/prediction drift (PSI, feature-level comparisons
against `drift_reference.json`) is a separate, batch-computed concern
belonging to `drift_check.py`, surfaced through CloudWatch rather than
Prometheus — see `docs/PHASE_12A_TODO.md` Group 12 for that split and why.
Business metrics (expected value, campaign economics) live in MLflow via
`evaluate.py`/`economics.py`, never duplicated here as a live computation —
only mirrored where explicitly noted below.

Status column: ✅ live today · 🧭 planned, not yet implemented (tracked in
`docs/PHASE_12A_TODO.md` Group 12).

---

# Serving-Layer Metrics (Prometheus)

## `predictions_total` ✅

**Counter.** Labels: `scoring_path` (`champion_only` / `champion_with_shadow`
/ `challenger_canary`), `model_version`.

Counts **rows scored**, not HTTP requests — a single `/predict/batch` call
with 500 rows increments this 500 times, once per row, inside
`score_request`'s per-row loop (`predict.py`). A lone `/predict` call and a
500-row batch are the same loop running once vs. 500 times.

`scoring_path` names *who actually served the row* and how it got there —
not a bare reuse of the dual-score entry's own `mode`, which would read
backwards (`mode="shadow"` sounds like the shadow/challenger model served
it, when shadow's entire point is that the champion always serves while the
challenger is only evaluated alongside).

**Chart:**
- Throughput: `sum(rate(predictions_total[5m]))`.
- Which version is serving, over time: `sum by (model_version) (rate(...))`
  — a promotion should show the old version's line drop to zero and the new
  one rise; visual proof the hot-reload actually worked.
- Canary fraction sanity check: `rate(..., scoring_path="challenger_canary")`
  over the total, should track the configured `canary.fraction`.
- **Do not alert on zero traffic** — a portfolio demo's normal state is
  near-zero requests. Use `up==0` (below) for "the service is down," not
  "nobody is calling it."

---

## `predicted_probability` ✅

**Histogram.** Label: `model_version`. Buckets: 0.05 steps across `[0, 1]`.

The shape of the churn-risk scores actually handed back to callers — the
fastest available prediction-drift proxy, since it moves immediately,
without waiting on labels. Observes `served_proba`: the probability the
caller *got*, not everything the service computed. In shadow mode, the
challenger's silently-computed score is never observed here — only the
champion's, since it's the one that served. Buckets are explicit because
the default web-latency-shaped bucket boundaries would collapse `0.25-0.5`
(where `t*` usually sits) into one bucket.

**Chart:**
- Median/p90 over time (`histogram_quantile`) — is typical predicted risk
  drifting?
- Split by `model_version` — during an active canary, this gives champion's
  and challenger's score distributions side by side for free.
- Not normally a hard alert target — no "correct" value for a distribution.
  Investigate-trigger, not a page.

**Known trap:** a shift in this histogram under a *single, unchanged*
`model_version` over an extended window is real prediction drift (the
model's inputs changed, since its code didn't). A difference between two
*different* version labels' baselines is not drift — that's just two
independently-fit models having two different distributions.

---

## `predictions_above_threshold_total` ✅

**Counter.** Label: `model_version`.

Mirrors the `decision` field exactly — `served_proba[row] >= served_threshold[row]`,
the identical boundary condition that produces the real `decision` returned
to callers. Tracks `decision`, **not** `contact`: this says nothing about
whether capacity-constrained contact selection actually reached these
customers (see the planned contact-capacity metric below).

**Chart:**
- Flag rate: `rate(predictions_above_threshold_total[1h])`.
- Flag rate as a fraction of total volume — should roughly track the
  model's calibrated prevalence at `t*`.
- Split by `model_version`.

**Diagnostic trap:** this rate can move because the *scores* moved, or
because `t*` itself moved (a `threshold.json` update, same model version,
zero change to `predicted_probability`). A real drift shift tends to be
gradual; a threshold/policy change is a step, at the exact moment the new
threshold loaded. The planned `active_threshold` gauge (below) resolves
this directly instead of requiring the inference.

---

## `model_inference_seconds` ✅

**Histogram.** Label: `model` (`champion` / `challenger` — a *role*, not a
version number; distinct from `model_version` used elsewhere). Buckets:
1ms-5s.

Wall-clock time of one `predict_proba` call, isolated from the DB lookup,
serialization, and FastAPI overhead that `prometheus_fastapi_instrumentator`'s
total-request-latency histogram already covers. **Vectorized**: one
`.observe()` covers the whole batch's `predict_proba` call, not one per row
— a 300-row batch produces exactly one observation, not 300. Buckets are
explicit because a single-row inference on this model is expected well
under 10ms; the default buckets would pile nearly everything into the first
one or two.

**Chart:**
- p99 per `model` — a spike here, with total request latency flat, means
  the model itself is the problem, not a dependency.
- A spike **without** a corresponding rise in request volume or batch size
  is worth checking against `CPUCreditBalance` (planned CloudWatch alarm) —
  on a `t3.small`, CPU-credit exhaustion throttles everything CPU-bound,
  including this, for reasons that have nothing to do with the model.

---

## `explanation_seconds` ✅

**Histogram.** No labels. Buckets: 10ms-10s (shifted up an order of
magnitude from `model_inference_seconds`, matching SHAP's known cost gap
over `predict_proba`).

Wall-clock time of one local SHAP explanation (`explain_with_explainer`),
only observed when `include_explanation=True`. The Streamlit UI sets this on
every call it makes — this histogram is, concretely, most of what a UI user
experiences as "prediction took a few seconds."

**Chart:**
- p99 over time. A sudden, whole-distribution shift (not just a longer
  tail) is the signature of a caching regression — the SHAP explainer is
  supposed to travel inside the `ModelBundle`, built once per reload, never
  rebuilt per request. If that caching ever broke, every explained call
  would get slower by roughly the same fixed amount, and this metric would
  show it immediately.

---

## `champion_challenger_score_distance` ✅

**Histogram.** Label: `mode` (`shadow` / `canary`) — genuinely means "which
testing regime produced this comparison," not a routing-outcome claim, so
`mode` itself needed no rename. The metric *name* did, though: it was
originally `shadow_canary_score_distance`, and read cold — without the HELP
text — that prefix sitting directly in front of "score_distance" parses as
"distance between shadow and canary," not "between champion and
challenger," which is the actual, constant subject of every observation.
Renamed so the comparison's subject is explicit in the name itself, not
only in the description.

`abs(challenger_score - champion_score)`, observed once per **evaluated**
row (inside the per-row comparison loop, not vectorized) — scales with row
count, unlike `model_inference_seconds`. No `model_version` label: a
dual-score comparison only ever spans the one currently-active
champion/challenger pair.

**Chart:**
- p50/p90 of distance, split by `mode`.
- **Read together with `champion_challenger_agreement_total`, never alone**
  — see below for why neither one tells the full story by itself.

---

## `champion_challenger_agreement_total` ✅

**Counter.** Labels: `outcome` (`agree`/`disagree`), `mode`
(`shadow`/`canary`). Exposed name carries `_total`; the Python source
constant (`"champion_challenger_agreement"`) does not —
`prometheus_client` appends it automatically for every `Counter`.

Label originally `agree` with bare `true`/`false` values — renamed to
`outcome`/`agree`-`disagree` because the old form repeated "agree" twice
(once in the metric name's "agreement," once in the label key) and a bare
boolean loses its meaning the instant someone reads the label without the
HELP text alongside it.

Whether champion and challenger landed on the same side of their *own
respective* thresholds for a dual-scored row — not whether their raw scores
were close.

**Chart:**
- Agreement rate: `rate(..., outcome="agree") / rate(...)` summed over
  `outcome` — the single most important canary-health number: what fraction
  of customers would actually be treated differently if the challenger took
  over. Low agreement isn't inherently bad; it's a measure of how much rides
  on the promotion decision.
- **Distance and agreement move independently, by design — chart both.** A
  large average distance with high agreement means most customers sit far
  from the threshold on both sides (the gap doesn't flip anything). A small
  distance with low agreement means customers cluster right at the
  threshold, where even a tiny score difference flips the decision.
- With both shadow and canary on simultaneously: the `canary` and `shadow`
  populations partition the same dual-scored set by the same hash (above vs.
  below the bucket cutoff). Their distance/agreement distributions should
  look statistically similar if the canary sample is unbiased — a
  meaningful difference between them is a sampling red flag, not a model
  one.
- Not normally a hard alert — disagreement is often the expected, desired
  outcome of running an active evaluation.

---

## `prediction_log_write_failures_total` ✅

**Counter** (`app.py`). Label: `route` (`single` / `batch` / `nightly_batch`
— the calling endpoint, a different meaning of "route" than `predictions_total`
ever had; no rename needed here since this one was never backwards, just a
different axis).

Counts predictions whose `prediction_log` write failed and was
log-and-swallowed — the response itself was still served (fail-open at this
boundary: a Postgres hiccup degrades to "this call wasn't captured in the
audit log," never to a failed prediction). No `model_version` label — a
write failure is a Postgres-availability signal, not a per-model one.

**Chart:**
- **Unlike most metrics in this doc, there's no healthy non-zero baseline to
  compare against** — in a working system this sits at zero. A sustained
  `rate(prediction_log_write_failures_total[15m]) > 0` for a few minutes
  *is* the alert, no threshold tuning needed — a silent audit-trail gap that
  nothing else would surface, since the caller-facing response never
  reflects it.

---

## `customer_lookup_timeouts_total` ✅

**Counter** (`app.py`). Label: `route` (`customer` / `batch` — `GET
/customer/{id}` vs. `/predict/batch`'s ID-resolution path; a plain `/predict`
call with full inline features never looks up anything, so it's absent from
this label's value set entirely).

Counts `customers_crm` lookups abandoned after
`serving.database.lookup_timeout_seconds` and answered `503` — a
timeout-as-circuit-breaker pattern: without it, a stalled Postgres
connection holds the request open until the kernel gives up on the socket,
with no signal to the caller and no way for the service to recover on its
own. Forcing an explicit, immediate, retryable `503` instead turns an
indefinite hang into a bounded, visible failure.

**Chart:**
- Same shape as `prediction_log_write_failures_total` above: no healthy
  baseline, any sustained rate above zero is the signal. More specific
  though — not "something in the logging pipeline broke," but **"Postgres
  itself is stalled or unreachable,"** a direct, unambiguous flag, cheaper
  to read than inferring DB trouble from elevated total latency.
- **If both this and `prediction_log_write_failures_total` start climbing
  at the same time, suspect one shared cause: Postgres degrading.** Expect
  `customer_lookup_timeouts_total` to move *first* under a gradual
  degradation — it has an explicit, usually-short timeout
  (`lookup_timeout_seconds`) guarding a read that blocks the response
  itself, while the prediction-log write happens in a `BackgroundTasks` call
  *after* the response is already sent, typically with a longer tolerance
  before it's abandoned. A lookup timing out doesn't mean "not found" —
  those are different outcomes entirely: a genuinely-missing `customerid`
  resolves normally (fast, a clean "no row") and never touches this counter
  at all; only a database that fails to answer *in time* does.

---

## Framework-provided (no project code involved)

`prometheus_fastapi_instrumentator`'s `Instrumentator().instrument(app)`
exposes these automatically — `app.py`'s `GET /metrics` serves the whole
default registry, not just the metrics this project registered itself:

- `http_requests_total{handler, method, status}` — every HTTP call across
  every route, success or failure. Diverges from `predictions_total`
  whenever a batch carries more than one row, or a call fails before
  scoring, or hits a non-scoring route (`/health`, `/customer/{id}`, …).
- `http_request_duration_seconds{handler, method}` — total request latency,
  including everything `model_inference_seconds`/`explanation_seconds`
  deliberately isolate out (DB lookups, serialization, framework overhead).
- `http_requests_inprogress` (gauge) — requests currently being handled;
  can legitimately go down, unlike any counter here.

---

## Planned, not yet implemented 🧭

Full rationale for each lives in `docs/PHASE_12A_TODO.md` Group 12 — summarized here for quick reference:

- **`active_threshold{model_version}`** (gauge) — mirrors `bundle.thresholds[bundle.base_scenario_name]`, set in `build_model_bundle`. Makes the `predictions_above_threshold_total` diagnostic trap above a direct read instead of an inference.
- **`champion_seconds_since_last_successful_poll`** (gauge) — resets on every completed poll tick; alert if it exceeds a few poll intervals. Detects the fail-open refresh mechanism silently failing — the specific case fail-open is built to hide.
- **`champion_version_mismatch`** (or `champion_version_registry`/`champion_version_served`, gauges) — the registry's current alias version vs. the currently-served version, both already read every poll. Catches a refused/failed reload the poll-health gauge above can't see.
- **`model_reload_total{outcome="success"|"refused"|"failed"}`** (counter) — currently these outcomes only hit `logger.warning`, never a metric.
- **Contact-capacity utilization** — fraction of `decision=true` rows capacity actually reached each cycle; the `decision`/`contact` split currently only exists in the response payload, never aggregated.
- **`cost_scenario_constants`** (gauge) — mirrors `configs/costs.yaml`'s base-scenario constants, set once at startup. Lets Grafana multiply against the already-live counters at query time for an "estimated campaign spend" panel, with zero duplicated EV logic.

Scrape/pipeline-health alerting decided alongside these (not a Prometheus metric, but directly relevant to reading every one of the serving-layer metrics above): `up{job="api"}==0` and `absent(up{job="api"})` — see `docs/PHASE_12A_TODO.md` Group 12. CloudWatch alarms (`CPUCreditBalance` included) are covered in their own section below, not planned/future — the underlying metrics already exist, only the alarms need building.

---

# Infrastructure-Layer Metrics (CloudWatch)

A different backend entirely from everything above — no `/metrics` scrape,
no Alloy, no Prometheus. These come from AWS's own EC2 monitoring and the
CloudWatch agent installed on the host (`infra/deploy/user-data.sh.tpl`),
surfaced through CloudWatch alarms into the same SNS topic as the serving
alerts. Backstops the serving layer from underneath: several of these can
still fire when the box is in a state bad enough that Alloy, the API, and
every Prometheus metric above have all gone silent together.

A key distinction that applies across all five: **CPU-related metrics need
no agent at all** — AWS's hypervisor can see how much CPU a VM consumes
without looking inside it, the same way you can tell an engine is working
hard without opening the hood. **Memory and disk-space usage are different**
— those are facts only the guest OS knows, so `mem_used_percent`/
`disk_used_percent` only exist once the CloudWatch agent is running inside
the instance to read and report them.

## `CPUUtilization` ✅ (alarm planned)

**AWS/EC2 namespace, no agent.** Percentage of allocated compute capacity in
use, reported every 5 minutes on basic monitoring (this project's tier — no
`monitoring = true` block in `infra/compute.tf`; detailed monitoring's
1-minute granularity isn't enabled, correctly, since nothing here needs it).

**Alert:** `> 90%` sustained for **15 minutes** — both numbers deliberate.
No sustained window would mean alerting on every normal burst (a
`/predict/batch` call, a SHAP explanation) as if it were an incident —
**alert fatigue**, the cost of a too-sensitive threshold. 90% leaves a 10%
margin to react before the system actually starts queueing or dropping
work, rather than only learning about trouble once it's happening.

**A structural blind spot, specific to this box's instance family:** on a
burstable `t3.small`, once CPU credits are exhausted *in Standard mode*,
AWS throttles the instance down to its baseline performance level (20%) —
a hard ceiling enforced by the hypervisor, completely independent of the
90% alarm threshold. The box can be genuinely starved for CPU while this
metric sits capped well under 90% the whole time, because AWS itself won't
let it report any higher. See `CPUCreditBalance` below — this is exactly
why that metric needed its own alarm rather than a lower `CPUUtilization`
threshold (which would never fire either, for the same reason). **This box
is actually configured in Unlimited mode, not Standard** (see
`CPUCreditBalance` below) — so this specific throttling scenario doesn't
apply to it today, but the reasoning is worth keeping for any future
Standard-mode instance.

## `StatusCheckFailed` ✅ (alarm planned)

**AWS/EC2 namespace, no agent.** Runs every minute regardless of monitoring
tier — unlike `CPUUtilization`, the 5-min/1-min granularity split only
applies to performance metrics, not status checks.

Actually the OR of two separate sub-checks, each with a different fix:
- **`StatusCheckFailed_System`** — the underlying AWS host/hardware is
  broken. Remedy: **stop/start** (migrates to different hardware; a reboot
  keeps you on the same broken host).
- **`StatusCheckFailed_Instance`** — the guest OS itself is unresponsive
  (corrupted filesystem, OOM, kernel panic). Remedy: a plain **reboot**.

This project alarms on the combined `StatusCheckFailed >= 1` rather than
the two sub-checks separately — the right call at single-instance scale; a
deliberate simplification, not an oversight, if ever revisited.

**Alert:** `>= 1`, no sustained-duration window needed — unlike
`CPUUtilization`, there's no equivalent of "a normal brief burst" for a
status check. If it fails even once, something is actually wrong.

**Why this metric earns a place in the monitoring design at all:** it's
evaluated entirely from outside the instance, so it's one of the only
signals that can still fire when the box is completely gone — a kernel
panic or dead host takes the CloudWatch agent, Alloy, and the API down with
it, but AWS's own external check keeps working. The outermost layer of the
same "silence must still alert" principle behind the `up==0`/`absent()`
alerts above: Alloy watches the API; `StatusCheckFailed` watches the box
Alloy itself runs on.

## `CPUCreditBalance` ✅ (alarm planned, risk framing corrected 2026-10-19)

**AWS/EC2 namespace, no agent.** Number of CPU credits currently banked on a
burstable instance; a `t3.small` earns them continuously and spends them
whenever running above its 20% baseline.

**What happens at zero depends entirely on credit mode, and this matters
concretely for this project:** checking `infra/compute.tf` shows no
`credit_specification` block at all — which means this instance is running
on **AWS's actual default for T3, Unlimited mode**, not Standard.

| Mode | At zero credits |
|---|---|
| Standard | Throttled to the 20% baseline (see `CPUUtilization` above) — no extra cost, but invisible degradation |
| **Unlimited (this box's real config)** | **No throttling — keeps running at full performance, but AWS bills a per-vCPU-hour surplus charge** for the overage |

So the risk this alarm actually guards against here is **a cost surprise**,
not silent performance throttling — arguably a better fit for a project
that already tracks spend this closely (shared Free Plan credit balance,
$40 budget, existing billing alarms) than the throttling story would have
been anyway.

**Alert:** a **low-balance** threshold, not zero — the point is advance
warning before any surplus billing starts, not after. Threshold best set
empirically once there's real traffic history, not guessed upfront.

**Open decision, not yet made:** keep Unlimited mode (accept occasional
small surplus charges for burst headroom) or set
`credit_specification { cpu_credits = "standard" }` explicitly (accept
throttling instead of any surprise billing)? Either is defensible for a
low-traffic demo box — the gap is that this is currently an unexamined AWS
default, not a decision anyone actually made.

## `mem_used_percent` ✅ (alarm planned)

**Custom CloudWatch namespace, via the CloudWatch agent** installed on the
host (not in a container — host-level only, per `user-data.sh.tpl`;
container logs go through Docker's `awslogs` driver instead). Needs the
agent because memory usage is an OS-level fact the hypervisor can't see.

**Alert:** `> 90%` — the metric most directly tied to this project's own
incident history: **two real OOM incidents already happened on this box.**
That history is exactly why `treat_missing_data = "breaching"` matters here
specifically (not `"notBreaching"`, which the billing alarm correctly uses
instead) — if the agent process itself dies, which is plausible on a box
that's already OOM'd before, the metric just stops arriving, and silence
must not read as "fine" at precisely the moment memory trouble is most
likely.

## `disk_used_percent` ✅ (alarm planned)

**Same CloudWatch agent, same namespace** as `mem_used_percent`.

**Alert:** `> 90%`, same `treat_missing_data = "breaching"` reasoning. Not
hypothetical: the CD rollback drill (`docker-build.yml`/`cd.yml`) hit a real
`no space left on device` failure from old SHA-tagged release images piling
up before `prune_on_box.sh` was fixed to prune aggressively (`-af` instead
of dangling-only). That failure surfaced as a confusing SSM timeout two
layers removed from the actual cause — this alarm, once built, is what
should surface it directly instead.
