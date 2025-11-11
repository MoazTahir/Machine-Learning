# Monitoring & Observability

Operational playbooks for keeping models healthy once they ship.

---

## Objectives

- Detect **data drift** (input distribution shifts) and **concept drift** (label/prediction relationship changes) before accuracy degrades.
- Track **service reliability** — latency, throughput, error rates — across `fastapi_app` deployments.
- Provide **actionable alerts** with clear next steps, reducing mean time to detect (MTTD) and mean time to recovery (MTTR).
- Preserve **audit trails** by versioning artefacts, metrics, and monitoring thresholds alongside code.

## Monitoring Stack Blueprint

1. **Data capture**
	- Log every prediction request/response pair to an append-only store (S3, BigQuery, or filesystem rotation under `Monitoring/logs/`). Include the model slug, version (from `LinearRegressionService._artifact_version`), and timestamps.
	- Sample raw features for privacy or throughput constraints, but always log aggregate statistics (mean, std, quantiles).

2. **Metric computation**
	- Reuse metric implementations from `Essentials Toolkit/Errors/metrics.py` to compute rolling accuracy, error distributions, or regression residuals.
	- Schedule batch jobs that join logged predictions with delayed ground truth labels to refresh downstream KPIs.

3. **Storage & visualization**
	- Push metrics into Prometheus (time-series) or a warehouse table for BI dashboards.
	- Publish curated Grafana/Looker/Dash dashboards summarising:
	  - request volume and latency percentiles;
	  - R2 / accuracy trend lines versus acceptance bands;
	  - drift detectors (KL divergence, PSI) per critical feature.

4. **Alerting**
	- Define threshold policies (static or adaptive) for each KPI, e.g. `rmse > 7000` for salary regression or `latency_p95 > 500ms`.
	- Route alerts to on-call channels (Slack, PagerDuty) with context links to the relevant dashboard and runbook section.

## Implementation Patterns

- **Middleware hooks** — extend the FastAPI router with request/response logging middleware that serializes payloads and predictions.
- **Background jobs** — use APScheduler or a serverless cron to trigger `evaluation/train_and_persist` style re-scoring jobs nightly, comparing new metrics against the last deployment.
- **Drift detection utilities** — add helpers under `Monitoring/drift.py` (future work) for PSI, KS tests, or embedding similarity calculations.
- **Canary analysis** — when rolling out new models, shadow a portion of traffic and compare live metrics to the reigning version before full promotion.

## Runbook Template

1. **Alert received** — capture alert ID, timestamp, impacted model slug.
2. **Triage** — inspect dashboards to confirm whether the breach is real or due to upstream data issues.
3. **Mitigation options**
	- Roll back to the last green model artefact (`artifacts/*.joblib`).
	- Trigger expedited re-training with the most recent clean data.
	- Apply feature flags or prediction caps if the issue is localized.
4. **Root cause analysis** — document findings in `Monitoring/incidents/YYYY-MM-DD-{slug}.md`.
5. **Post-mortem actions** — update thresholds, add tests, or extend logging to prevent recurrence.

## Getting Started Checklist

- [ ] Enable model version tagging inside every inference response (already available in `LinearRegressionService`).
- [ ] Wire structured logging for FastAPI requests (consider `structlog` or Python `logging` JSON handlers).
- [ ] Stand up a minimal Prometheus + Grafana stack or leverage managed observability services.
- [ ] Define the initial alert policy across accuracy, latency, and drift metrics.
- [ ] Schedule periodic review meetings to evaluate monitoring effectiveness and refine thresholds.

Keep this document evolving with concrete scripts, configuration snippets, and dashboard exports as the monitoring footprint matures.
