from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import logging
import base64
import json
from datetime import datetime, timezone

import httpx

from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/observability", tags=["Observability"])

# Langfuse public API (v4 events_only) — metrics endpoint contract
LANGFUSE_METRICS_PATH = "/api/public/v2/metrics"
ALLOWED_MEASURES = {"count", "totalCost", "inputTokens", "outputTokens", "totalTokens", "latency"}
ALLOWED_AGGREGATIONS = {"sum", "avg", "count", "min", "max", "p50", "p75", "p90", "p95", "p99", "uniq"}
ALLOWED_GRANULARITIES = {"hour", "day", "week", "month"}
# Metrics API v2 only allows grouping by non-high-cardinality dimensions.
# id/traceId/userId/sessionId/parentObservationId are filter-only → a 400.
ALLOWED_DIMENSIONS = {"providedModelName", "traceName", "level", "startTimeMonth"}


def _auth_header() -> str:
    pk = settings.LANGFUSE_PUBLIC_KEY
    sk = settings.LANGFUSE_SECRET_KEY
    if not pk or not sk:
        raise HTTPException(status_code=503, detail="Langfuse credentials not configured.")
    token = base64.b64encode(f"{pk}:{sk}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _parse_ts(value: str, name: str) -> str:
    """Parse an ISO timestamp and re-emit as UTC 'YYYY-MM-DDTHH:MM:SS.000Z'."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {name} timestamp: {value}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _build_metrics(metrics: List[dict]) -> List[dict]:
    """Whitelist metric definitions of form {measure, aggregation}."""
    validated = []
    for item in metrics:
        measure = item.get("measure")
        agg = item.get("aggregation")
        if measure not in ALLOWED_MEASURES:
            raise HTTPException(status_code=400, detail=f"Unsupported metric measure: {measure}")
        if agg not in ALLOWED_AGGREGATIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported metric aggregation: {agg}")
        validated.append({"measure": measure, "aggregation": agg})
    return validated


async def _query_metrics(query: dict) -> List[dict]:
    """Execute a Langfuse v2 metrics query and return the rows."""
    payload = json.dumps(query)
    headers = {
        "Authorization": _auth_header(),
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.LANGFUSE_BASE_URL}{LANGFUSE_METRICS_PATH}",
                params={"query": payload},
                headers=headers,
            )
    except httpx.HTTPError as e:
        logger.error(f"Langfuse metrics request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Langfuse unavailable: {e}")
    if resp.status_code != 200:
        logger.error(f"Langfuse metrics error {resp.status_code}: {resp.text[:500]}")
        raise HTTPException(status_code=502, detail=f"Langfuse error {resp.status_code}: {resp.text[:300]}")
    data = resp.json().get("data", [])
    return data if isinstance(data, list) else []


def _row_value(row: dict, measure: str, aggregation: str):
    key = f"{aggregation}_{measure}"
    val = row.get(key)
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if not isinstance(val, bool) else val
    return val


@router.get("/metrics/overview")
async def metrics_overview(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
):
    """Top-level KPIs (traces, cost, tokens, latency, error rate) for the given range."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")

    base = {"view": "observations", "fromTimestamp": from_ts, "toTimestamp": to_ts}
    metrics = [
        {"measure": "count", "aggregation": "count"},
        {"measure": "totalCost", "aggregation": "sum"},
        {"measure": "inputTokens", "aggregation": "sum"},
        {"measure": "outputTokens", "aggregation": "sum"},
        {"measure": "totalTokens", "aggregation": "sum"},
        {"measure": "latency", "aggregation": "avg"},
        {"measure": "latency", "aggregation": "p95"},
    ]

    all_rows = await _query_metrics({**base, "dimensions": [], "metrics": metrics, "filters": []})
    root_rows = await _query_metrics({
        **base,
        "dimensions": [],
        "metrics": [{"measure": "count", "aggregation": "count"}],
        "filters": [{"type": "boolean", "column": "isRootObservation", "operator": "=", "value": True}],
    })
    error_rows = await _query_metrics({
        **base,
        "dimensions": [],
        "metrics": [{"measure": "count", "aggregation": "count"}],
        "filters": [{"type": "string", "column": "level", "operator": "=", "value": "ERROR"}],
    })

    total = all_rows[0] if all_rows else {}
    root = root_rows[0] if root_rows else {}
    err = error_rows[0] if error_rows else {}

    observation_count = _row_value(total, "count", "count") or 0
    trace_count = _row_value(root, "count", "count") or 0
    error_count = _row_value(err, "count", "count") or 0

    return {
        "range": {"from": from_ts, "to": to_ts},
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "observationCount": int(observation_count),
        "traceCount": int(trace_count),
        "totalCost": _row_value(total, "totalCost", "sum") or 0,
        "inputTokens": _row_value(total, "inputTokens", "sum") or 0,
        "outputTokens": _row_value(total, "outputTokens", "sum") or 0,
        "totalTokens": _row_value(total, "totalTokens", "sum") or 0,
        "latencyAvgMs": _row_value(total, "latency", "avg"),
        "latencyP95Ms": _row_value(total, "latency", "p95"),
        "errorCount": int(error_count),
        "errorRate": round(error_count / observation_count, 4) if observation_count else 0,
    }


@router.get("/metrics/timeseries")
async def metrics_timeseries(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    granularity: str = Query("day", description="Bucket size: hour|day|week|month."),
):
    """Time series of observations/traces/errors plus cost, tokens and latency per bucket."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if granularity not in ALLOWED_GRANULARITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported granularity: {granularity}")

    base = {
        "view": "observations",
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "timeDimension": {"granularity": granularity},
    }
    metrics = [
        {"measure": "count", "aggregation": "count"},
        {"measure": "totalCost", "aggregation": "sum"},
        {"measure": "inputTokens", "aggregation": "sum"},
        {"measure": "outputTokens", "aggregation": "sum"},
        {"measure": "totalTokens", "aggregation": "sum"},
        {"measure": "latency", "aggregation": "avg"},
        {"measure": "latency", "aggregation": "p95"},
    ]
    count_metric = [{"measure": "count", "aggregation": "count"}]

    obs_rows = await _query_metrics({**base, "dimensions": [], "metrics": metrics, "filters": []})
    trace_rows = await _query_metrics({
        **base,
        "dimensions": [],
        "metrics": count_metric,
        "filters": [{"type": "boolean", "column": "isRootObservation", "operator": "=", "value": True}],
    })
    error_rows = await _query_metrics({
        **base,
        "dimensions": [],
        "metrics": count_metric,
        "filters": [{"type": "string", "column": "level", "operator": "=", "value": "ERROR"}],
    })

    # On some Langfuse v4 deployments the metrics endpoint returns null for
    # latency aggregations when a timeDimension is applied. When that happens
    # (and only then) compute per-day avg/p95 latency from the observations API
    # (latency is in seconds there; convert to milliseconds). The per-bucket day
    # grouping is only valid at `day` granularity; for hour/week/month we leave
    # the (null) metric value rather than repeat a day's latency per bucket.
    day_avg: dict = {}
    day_p95: dict = {}
    obs_raw = [
        {
            "time": r.get("time_dimension"),
            "latencyAvgMs": _row_value(r, "latency", "avg"),
            "latencyP95Ms": _row_value(r, "latency", "p95"),
        }
        for r in obs_rows
    ]
    latency_missing = obs_raw and all(
        item["latencyAvgMs"] is None and item["latencyP95Ms"] is None for item in obs_raw
    )
    if granularity == "day" and latency_missing:
        latency_rows, _ = await _fetch_all_observations(from_ts, to_ts)
        day_latency: dict = {}
        for o in latency_rows:
            st = o.get("startTime")
            lat = o.get("latency")
            if not st or lat is None:
                continue
            day_latency.setdefault(st[:10], []).append(float(lat) * 1000.0)
        day_avg = {d: sum(v) / len(v) for d, v in day_latency.items()}
        day_p95 = {d: _percentile(sorted(v), 0.95) for d, v in day_latency.items()}

    def series(rows, fields):
        result = []
        for r in rows:
            day = (r.get("time_dimension") or "")[:10]
            item = {
                "time": r.get("time_dimension"),
                **{k: _row_value(r, v[0], v[1]) for k, v in fields.items()},
            }
            if "latencyAvgMs" in item and day in day_avg:
                item["latencyAvgMs"] = round(day_avg[day], 2)
            if "latencyP95Ms" in item and day in day_p95:
                item["latencyP95Ms"] = round(day_p95[day], 2)
            result.append(item)
        return result

    return {
        "granularity": granularity,
        "range": {"from": from_ts, "to": to_ts},
        "observations": series(obs_rows, {
            "count": ("count", "count"),
            "cost": ("totalCost", "sum"),
            "inputTokens": ("inputTokens", "sum"),
            "outputTokens": ("outputTokens", "sum"),
            "latencyAvgMs": ("latency", "avg"),
            "latencyP95Ms": ("latency", "p95"),
        }),
        "traces": series(trace_rows, {"count": ("count", "count")}),
        "errors": series(error_rows, {"count": ("count", "count")}),
    }


@router.get("/metrics/by-model")
async def metrics_by_model(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    dimension: str = Query("providedModelName", description="Breakdown dimension."),
    limit: int = Query(20, ge=1, le=100),
):
    """Cost/token/latency breakdown grouped by a dimension (default: model name)."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if dimension not in ALLOWED_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported dimension: {dimension}")

    metrics = [
        {"measure": "count", "aggregation": "count"},
        {"measure": "totalCost", "aggregation": "sum"},
        {"measure": "inputTokens", "aggregation": "sum"},
        {"measure": "outputTokens", "aggregation": "sum"},
        {"measure": "latency", "aggregation": "avg"},
    ]

    rows = await _query_metrics({
        "view": "observations",
        "dimensions": [{"field": dimension}],
        "metrics": metrics,
        "filters": [],
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "config": {"row_limit": 1000},
    })

    result = []
    for r in rows:
        name = r.get(dimension)
        if not name or str(name).strip().lower() in ("unknown", "null", "none", "n/a"):
            continue
        result.append({
            "name": name,
            "count": int(_row_value(r, "count", "count") or 0),
            "cost": _row_value(r, "totalCost", "sum") or 0,
            "inputTokens": _row_value(r, "inputTokens", "sum") or 0,
            "outputTokens": _row_value(r, "outputTokens", "sum") or 0,
            "latencyAvgMs": _row_value(r, "latency", "avg"),
        })

    result.sort(key=lambda m: (m["cost"] or 0, m["count"] or 0), reverse=True)
    return {"dimension": dimension, "range": {"from": from_ts, "to": to_ts}, "models": result[:limit]}


MODEL_LATENCY_PERCENTILES = [("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]


@router.get("/metrics/model-latency")
async def metrics_model_latency(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    granularity: str = Query("day", description="Bucket size: hour|day|week|month."),
    models: Optional[str] = Query(None, description="Comma-separated model names to include (any of)."),
    limit: int = Query(20, ge=1, le=100),
):
    """Per-model latency percentiles (p50/p75/p90/p95/p99) per time bucket.

    Backed by the Metrics API v2 observations view: `providedModelName`
    dimension + `latency` percentile aggregations + a `day` timeDimension —
    the same query the Langfuse "Model latencies" dashboard widget runs.
    Falls back to computing percentiles from the observations API when the
    deployment returns null latency under a timeDimension (latency is in
    seconds there; converted to milliseconds to match the metrics API).
    """
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if granularity not in ALLOWED_GRANULARITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported granularity: {granularity}")

    selected = [m.strip() for m in models.split(",") if m.strip()] if models else []

    filters = [
        {"type": "string", "column": "type", "operator": "=", "value": "GENERATION"},
    ]
    if selected:
        filters.append({
            "type": "stringOptions",
            "column": "providedModelName",
            "operator": "any of",
            "value": selected,
        })

    rows = await _query_metrics({
        "view": "observations",
        "dimensions": [{"field": "providedModelName"}],
        "metrics": [
            {"measure": "latency", "aggregation": "p50"},
            {"measure": "latency", "aggregation": "p75"},
            {"measure": "latency", "aggregation": "p90"},
            {"measure": "latency", "aggregation": "p95"},
            {"measure": "latency", "aggregation": "p99"},
        ],
        "filters": filters,
        "timeDimension": {"granularity": granularity},
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "config": {"row_limit": 1000},
    })

    # Metrics API v2 may return null latency under a timeDimension on some
    # deployments; when that happens (or no rows came back at all) compute
    # per-day per-model percentiles from the observations API instead. This
    # is only valid at `day` granularity, mirroring the timeseries endpoint.
    fb: dict = {}
    if granularity == "day" and (not rows or all(
        all(_row_value(r, "latency", agg) is None for agg, _ in MODEL_LATENCY_PERCENTILES)
        for r in rows
    )):
        obs, _ = await _fetch_all_observations(from_ts, to_ts, fields="core,basic,model,metrics")
        for o in obs:
            st = o.get("startTime")
            lat = o.get("latency")
            model = o.get("providedModelName") or "Unknown"
            if not st or lat is None:
                continue
            if selected and model not in selected:
                continue
            fb.setdefault(model, {}).setdefault(st[:10], []).append(float(lat) * 1000.0)

    if not rows:
        rows = [
            {"providedModelName": model, "time_dimension": f"{day}T00:00:00.000Z"}
            for model, days in fb.items()
            for day in days
        ]

    series: dict = {}
    for r in rows:
        model = r.get("providedModelName") or "Unknown"
        day = (r.get("time_dimension") or "")[:10]
        item = {
            "time": r.get("time_dimension"),
            **{agg: _row_value(r, "latency", agg) for agg, _ in MODEL_LATENCY_PERCENTILES},
        }
        vals = fb.get(model, {}).get(day)
        if vals:
            ordered = sorted(vals)
            for agg, p in MODEL_LATENCY_PERCENTILES:
                item[agg] = round(_percentile(ordered, p), 2)
        series.setdefault(model, []).append(item)

    def _model_p95(items: list) -> float:
        vals = [it["p95"] for it in items if it["p95"] is not None]
        return max(vals) if vals else 0.0

    ordered_models = sorted(series.items(), key=lambda kv: _model_p95(kv[1]), reverse=True)[:limit]

    return {
        "range": {"from": from_ts, "to": to_ts},
        "granularity": granularity,
        "models": [m for m, _ in ordered_models],
        "series": [
            {"model": m, "points": items}
            for m, items in ordered_models
        ],
    }


# ── Score analytics (Langfuse v4 Metrics API v2 score views) ────────────────
# v4 groups numeric AND boolean scores under the `scores-numeric` view (boolean
# values are 0/1, so avg(value) is the true-rate); categorical scores use the
# `scores-categorical` view (count only, stringValue grouping).
SCORE_VIEWS = {
    "NUMERIC": "scores-numeric",
    "BOOLEAN": "scores-numeric",
    "CATEGORICAL": "scores-categorical",
}
SCORE_ANALYTICS_DATA_TYPES = {"NUMERIC", "BOOLEAN", "CATEGORICAL"}
SCORE_ANALYTICS_SOURCES = {"API", "ANNOTATION", "EVAL"}
# Score analytics aggregation list (value avg/min/max are only meaningful on
# the numeric view; categorical view just counts).
SCORE_SUMMARY_METRICS = [
    {"measure": "count", "aggregation": "count"},
    {"measure": "value", "aggregation": "avg"},
    {"measure": "value", "aggregation": "min"},
    {"measure": "value", "aggregation": "max"},
]
SCORE_COUNT_METRIC = [{"measure": "count", "aggregation": "count"}]


@router.get("/metrics/scores")
async def metrics_scores(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    granularity: Optional[str] = Query(None, description="Bucket size for a trend: hour|day|week|month. Omit for a plain summary."),
    name: Optional[str] = Query(None, description="Comma-separated score names to include (any of)."),
    dataType: Optional[str] = Query(None, description="Filter by score data type: NUMERIC|BOOLEAN|CATEGORICAL."),
    source: Optional[str] = Query(None, description="Filter by score source: API|ANNOTATION|EVAL."),
    limit: int = Query(20, ge=1, le=100),
):
    """Score analytics: per-score-name aggregates and (optionally) a time trend.

    Backed by Langfuse Metrics API v2 score views. Numeric and boolean scores
    come from the `scores-numeric` view (avg(value) == true-rate for booleans),
    categorical scores from the `scores-categorical` view (count only).
    """
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if granularity and granularity not in ALLOWED_GRANULARITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported granularity: {granularity}")
    if dataType and dataType.upper() not in SCORE_ANALYTICS_DATA_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported score dataType: {dataType}")
    if source and source.upper() not in SCORE_ANALYTICS_SOURCES:
        raise HTTPException(status_code=400, detail=f"Unsupported score source: {source}")

    dtype = (dataType or "NUMERIC").upper()
    # CATEGORICAL (and TEXT/CORRECTION) scores carry a string value, so the
    # value-based aggregations are meaningless — restrict to count only.
    view = SCORE_VIEWS[dtype]
    metrics = SCORE_COUNT_METRIC if view == "scores-categorical" else SCORE_SUMMARY_METRICS

    filters = []
    if name:
        names = [n.strip() for n in name.split(",") if n.strip()]
        if names:
            filters.append({
                "type": "stringOptions",
                "column": "name",
                "operator": "any of",
                "value": names,
            })
    if dataType:
        filters.append({"type": "string", "column": "dataType", "operator": "=", "value": dataType.upper()})
    if source:
        filters.append({"type": "string", "column": "source", "operator": "=", "value": source.upper()})

    base = {
        "view": view,
        # Group by score name AND data type so boolean scores can be
        # distinguished for the frontend's true/false (0/1) breakdown.
        "dimensions": [{"field": "name"}, {"field": "dataType"}],
        "metrics": metrics,
        "filters": filters,
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "config": {"row_limit": 1000},
    }

    # Summary: per-score-name aggregates over the whole range.
    summary_rows = await _query_metrics({**base})
    summary = []
    for r in summary_rows:
        row = {
            "name": r.get("name") or "Unknown",
            "dataType": r.get("dataType") or "",
            "count": int(_row_value(r, "count", "count") or 0),
        }
        if view != "scores-categorical":
            row["avgValue"] = _row_value(r, "value", "avg")
            row["minValue"] = _row_value(r, "value", "min")
            row["maxValue"] = _row_value(r, "value", "max")
        summary.append(row)
    summary.sort(key=lambda s: (s["count"], s.get("avgValue") or 0), reverse=True)
    summary = summary[:limit]

    # Trend: same query bucketed by time (only when a granularity is given).
    trend = []
    if granularity:
        trend_base = {
            **base,
            # Keep the time series grouped by name only so a score name stays
            # a single line even when it carries multiple data types.
            "dimensions": [{"field": "name"}],
            "timeDimension": {"granularity": granularity},
        }
        trend_rows = await _query_metrics({**trend_base})
        for r in trend_rows:
            row = {
                "time": r.get("time_dimension"),
                "name": r.get("name") or "Unknown",
                "count": int(_row_value(r, "count", "count") or 0),
            }
            if view != "scores-categorical":
                row["avgValue"] = _row_value(r, "value", "avg")
            trend.append(row)

    return {
        "range": {"from": from_ts, "to": to_ts},
        "view": view,
        "granularity": granularity,
        "summary": summary,
        "trend": trend,
    }


# ── Toxicity analytics (per-node `toxicity` scores) ─────────────────────────
# Nodes log a numeric `toxicity` score (0..1) per executed node via
# Utilities.toxicity.schedule_toxicity_score -> record_toxicity_score. The
# overall average is served from the Metrics API v2 scores-numeric view; the
# metrics API can't group by the high-cardinality userId/sessionId dimensions,
# so per-user / per-session breakdowns are computed from the v3 scores API plus
# trace enrichment (userId/sessionId resolve from the attached trace).
TOXICITY_SCORE_NAME = "toxicity"
TOXICITY_GROUP_BYS = {"user", "session"}
TOXICITY_SCORES_MAX_PAGES = 10


def _toxicity_stats(vals: list, threshold: float) -> dict:
    n = len(vals)
    if n == 0:
        return {
            "count": 0,
            "avgValue": None,
            "avgToxicityPercent": None,
            "toxicRate": None,
            "minValue": None,
            "maxValue": None,
        }
    avg = sum(vals) / n
    toxic = [v for v in vals if v >= threshold]
    return {
        "count": n,
        "avgValue": round(avg, 4),
        "avgToxicityPercent": round(avg * 100, 2),
        "toxicRate": round(len(toxic) / n, 4),
        "minValue": round(min(vals), 4),
        "maxValue": round(max(vals), 4),
    }


async def _fetch_toxicity_scores(from_ts: str, to_ts: str) -> List[dict]:
    """Page through all `toxicity` scores in range (v3 scores API)."""
    rows = []
    cursor = None
    for _ in range(TOXICITY_SCORES_MAX_PAGES):
        params = {
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "name": TOXICITY_SCORE_NAME,
            "limit": str(SCORES_FACET_PAGE_LIMIT),
        }
        if cursor:
            params["cursor"] = cursor
        body = await _query_scores(params, fields="details,subject")
        data = []
        if isinstance(body, dict):
            data = body.get("data") or []
            meta = body.get("meta") or {}
            cursor = meta.get("cursor")
        rows.extend(data)
        if not data or not cursor:
            break
    return rows


async def _enrich_toxicity_trace_context(scores: List[dict], from_ts: str, to_ts: str) -> None:
    """Fill userId/sessionId onto each toxicity score from its trace."""
    trace_ids = {s.get("traceId") for s in scores if s.get("traceId")}
    if not trace_ids:
        return
    try:
        params = {
            "fields": "core,basic,trace_context",
            "limit": str(OBS_PAGE_LIMIT),
            "filter": json.dumps([{
                "type": "stringOptions",
                "column": "traceId",
                "operator": "any of",
                "value": sorted(trace_ids),
            }]),
        }
        if from_ts and to_ts:
            params["fromStartTime"] = from_ts
            params["toStartTime"] = to_ts
        obs_resp = await _query_observations(params)
        rows = obs_resp.get("data") if isinstance(obs_resp, dict) else (obs_resp or [])
        meta: dict = {}
        for r in rows:
            tid = r.get("traceId") or r.get("id")
            if not tid:
                continue
            m = meta.setdefault(tid, {"userId": "", "sessionId": ""})
            if not m["userId"] and r.get("userId"):
                m["userId"] = r["userId"]
            if not m["sessionId"] and r.get("sessionId"):
                m["sessionId"] = r["sessionId"]
        for s in scores:
            m = meta.get(s.get("traceId")) or {}
            s["userId"] = m.get("userId") or ""
            s["sessionId"] = m.get("sessionId") or ""
    except Exception as e:
        logger.warning(f"Could not enrich toxicity trace context: {e}")


@router.get("/metrics/toxicity")
async def metrics_toxicity(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    userId: Optional[str] = Query(None, description="Filter by user ID (comma-separated = any of)."),
    sessionId: Optional[str] = Query(None, description="Filter by session ID."),
    traceName: Optional[str] = Query(None, description="Filter by workflow/trace name substring."),
    groupBy: Optional[str] = Query(None, description="Break down averages by user|session."),
    threshold: Optional[float] = Query(None, description="Toxicity threshold (0-1) for the toxic rate; defaults to TOXICITY_THRESHOLD."),
    limit: int = Query(20, ge=1, le=100),
):
    """Average toxicity of node outputs, optionally filtered/grouped by user or session.

    Every executed node logs a `toxicity` score (0..1) from Detoxify. The
    overall average is served straight from the Metrics API v2 scores-numeric
    view; `userId`/`sessionId`/`traceName` filters and `groupBy=user|session`
    breakdowns are computed from the v3 scores API + trace enrichment.
    """
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if groupBy and groupBy.lower() not in TOXICITY_GROUP_BYS:
        raise HTTPException(status_code=400, detail=f"Unsupported groupBy: {groupBy}")
    thr = settings.TOXICITY_THRESHOLD if threshold is None else threshold

    # Fast path: no filters / no breakdown -> average straight from the metrics API.
    if not userId and not sessionId and not traceName and not groupBy:
        rows = await _query_metrics({
            "view": "scores-numeric",
            "dimensions": [],
            "metrics": [
                {"measure": "value", "aggregation": "avg"},
                {"measure": "value", "aggregation": "min"},
                {"measure": "value", "aggregation": "max"},
                {"measure": "count", "aggregation": "count"},
            ],
            "filters": [{"type": "string", "column": "name", "operator": "=", "value": TOXICITY_SCORE_NAME}],
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "config": {"row_limit": 1000},
        })
        row = rows[0] if rows else {}
        avg = _row_value(row, "value", "avg")
        overall = {
            "count": int(_row_value(row, "count", "count") or 0),
            "avgValue": round(avg, 4) if avg is not None else None,
            "avgToxicityPercent": round(avg * 100, 2) if avg is not None else None,
            "toxicRate": None,
            "minValue": _row_value(row, "value", "min"),
            "maxValue": _row_value(row, "value", "max"),
        }
        return {
            "range": {"from": from_ts, "to": to_ts},
            "score": TOXICITY_SCORE_NAME,
            "threshold": thr,
            "overall": overall,
            "byUser": [],
            "bySession": [],
        }

    # Filtered / grouped path: page the `toxicity` scores, enrich trace context.
    scores = await _fetch_toxicity_scores(from_ts, to_ts)
    mapped = []
    for r in scores:
        trace_id, _, _, _ = _subject_ids(r)
        val = r.get("value")
        if val is None:
            continue
        mapped.append({
            "traceId": trace_id,
            "value": float(val),
            "timestamp": r.get("timestamp") or r.get("createdAt") or "",
            "userId": "",
            "sessionId": "",
        })
    await _enrich_toxicity_trace_context(mapped, from_ts, to_ts)

    if userId:
        users = {u.strip() for u in userId.split(",") if u.strip()}
        mapped = [s for s in mapped if s["userId"] in users]
    if sessionId:
        mapped = [s for s in mapped if s["sessionId"] == sessionId]
    if traceName:
        try:
            resolved = await _query_observations({
                "fromStartTime": from_ts,
                "toStartTime": to_ts,
                "isRootObservation": "true",
                "fields": "core,basic,trace_context",
                "limit": str(OBS_PAGE_LIMIT),
                "filter": json.dumps([{
                    "type": "string",
                    "column": "traceName",
                    "operator": "contains",
                    "value": traceName,
                }]),
            })
            resolved_rows = resolved.get("data") if isinstance(resolved, dict) else (resolved or [])
            resolved_ids = {r.get("traceId") for r in resolved_rows if r.get("traceId")}
        except Exception as e:
            logger.warning(f"Toxicity traceName resolution failed: {e}")
            resolved_ids = set()
        if not resolved_ids:
            mapped = []
        else:
            mapped = [s for s in mapped if s["traceId"] in resolved_ids]

    overall = _toxicity_stats([s["value"] for s in mapped], thr)

    def _latest(entries: list) -> str:
        """Most recent score timestamp within a group (for recency ordering)."""
        return max((e["timestamp"] for e in entries if e.get("timestamp")), default="")

    by_user = []
    by_session = []
    if groupBy and groupBy.lower() == "user":
        groups: dict = {}
        for s in mapped:
            key = s["userId"] or "(no user)"
            groups.setdefault(key, []).append(s)
        by_user = [
            {"userId": key, "lastActivity": _latest(entries), **_toxicity_stats([e["value"] for e in entries], thr)}
            for key, entries in groups.items()
        ]
        by_user.sort(key=lambda g: g.get("lastActivity") or "", reverse=True)
        by_user = by_user[:limit]
    elif groupBy and groupBy.lower() == "session":
        groups = {}
        for s in mapped:
            key = s["sessionId"] or "(no session)"
            groups.setdefault(key, []).append(s)
        by_session = [
            {"sessionId": key, "lastActivity": _latest(entries), **_toxicity_stats([e["value"] for e in entries], thr)}
            for key, entries in groups.items()
        ]
        by_session.sort(key=lambda g: g.get("lastActivity") or "", reverse=True)
        by_session = by_session[:limit]

    return {
        "range": {"from": from_ts, "to": to_ts},
        "score": TOXICITY_SCORE_NAME,
        "threshold": thr,
        "overall": overall,
        "byUser": by_user,
        "bySession": by_session,
    }


# ── Trace & Observation listing (Langfuse v4 observations API) ──────────────
LANGFUSE_OBSERVATIONS_PATH = "/api/public/v2/observations"
ALLOWED_LEVELS = {"DEFAULT", "DEBUG", "INFO", "WARNING", "ERROR"}
ALLOWED_TYPES = {"SPAN", "GENERATION", "CHAIN", "EVENT"}
ALLOWED_KINDS = {"workflow", "evaluation", "decision", "system"}
SYSTEM_CHAIN_NAME = "LangGraphUpdateState"
EVALUATOR_SPAN_PREFIX = "Execute evaluator"
# Severity ranking for picking a trace's "worst" level across its observations.
LEVEL_SEVERITY = {"DEBUG": 0, "DEFAULT": 1, "INFO": 1, "WARNING": 2, "ERROR": 3}
# Field groups for trace rows & trace detail: adds input/output (io),
# metadata and trace-level tags/release/traceName (trace_context).
OBS_DETAIL_FIELDS = "core,basic,usage,metrics,io,metadata,trace_context"


def _classify_kind(name: str, type_: str) -> str:
    """Best-effort classification of a root observation.

    LangGraph names the root CHAIN after the workflow's run_name, so a root
    CHAIN is a user workflow run. Non-workflow roots (events, the HITL state
    chain, evaluator spans) are classified separately so they can be filtered.
    """
    name = name or ""
    if type_ == "EVENT":
        return "decision"
    if name == SYSTEM_CHAIN_NAME:
        return "system"
    if type_ == "SPAN" and name.startswith(EVALUATOR_SPAN_PREFIX):
        return "evaluation"
    return "workflow"


def _map_observation(r: dict) -> dict:
    return {
        "id": r.get("id"),
        "traceId": r.get("traceId"),
        "parentObservationId": r.get("parentObservationId"),
        "name": r.get("name"),
        "workflowName": r.get("name") or "",
        "kind": _classify_kind(r.get("name") or "", r.get("type") or ""),
        "type": r.get("type"),
        "level": r.get("level"),
        "statusMessage": r.get("statusMessage") or "",
        "environment": r.get("environment"),
        "version": r.get("version") or "",
        "input": r.get("input") or "",
        "output": r.get("output") or "",
        "metadata": r.get("metadata") or {},
        "tags": r.get("tags") or [],
        "startTime": r.get("startTime"),
        "endTime": r.get("endTime"),
        "latencySec": r.get("latency"),
        "inputTokens": r.get("inputUsage"),
        "outputTokens": r.get("outputUsage"),
        "totalTokens": r.get("totalUsage"),
        "totalCost": r.get("totalCost"),
        "inputCost": r.get("inputCost"),
        "outputCost": r.get("outputCost"),
        "userId": r.get("userId") or "",
        "sessionId": r.get("sessionId") or "",
        "modelId": r.get("modelId"),
    }


async def _query_observations(params: dict) -> dict:
    """Execute a Langfuse v2 observations GET and return the parsed JSON body."""
    headers = {"Authorization": _auth_header(), "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.LANGFUSE_BASE_URL}{LANGFUSE_OBSERVATIONS_PATH}",
                params=params,
                headers=headers,
            )
    except httpx.HTTPError as e:
        logger.error(f"Langfuse observations request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Langfuse unavailable: {e}")
    if resp.status_code == 400 and "filter" in params:
        # Some deployments reject the advanced filter param — retry without it.
        logger.info("Langfuse observations rejected filter param (400); retrying without it.")
        return await _query_observations({k: v for k, v in params.items() if k != "filter"})
    if resp.status_code != 200:
        logger.error(f"Langfuse observations error {resp.status_code}: {resp.text[:500]}")
        raise HTTPException(status_code=502, detail=f"Langfuse error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


@router.get("/traces")
async def list_traces(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(None, description="Pagination cursor from a previous page."),
    level: Optional[str] = Query(None, description="Filter by level (DEFAULT|DEBUG|INFO|WARNING|ERROR)."),
    search: Optional[str] = Query(None, description="Filter by trace/observation name substring."),
    type: Optional[str] = Query(None, description="Filter by observation type (SPAN|GENERATION|CHAIN|EVENT)."),
    kind: Optional[str] = Query(None, description="Filter by trace kind (workflow|evaluation|decision|system)."),
    environment: Optional[str] = Query(None, description="Filter by environment tag."),
    userId: Optional[str] = Query(None, description="Filter by user ID."),
    sessionId: Optional[str] = Query(None, description="Filter by session ID."),
    tags: Optional[str] = Query(None, description="Comma-separated trace tags; matches traces carrying any of them."),
    traceId: Optional[str] = Query(None, description="Filter by exact trace ID (for direct navigation from other tabs)."),
):
    """Recent traces (root observations) with pagination and filters."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if level and level.upper() not in ALLOWED_LEVELS:
        raise HTTPException(status_code=400, detail=f"Unsupported level: {level}")
    if type and type.upper() not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {type}")
    if kind and kind not in ALLOWED_KINDS:
        raise HTTPException(status_code=400, detail=f"Unsupported kind: {kind}")

    # When navigating by exact traceId (e.g. from Scores tab), skip time-range
    # restriction so the trace is always found regardless of the dashboard period.
    if traceId:
        params = {
            "isRootObservation": "true",
            "fields": OBS_DETAIL_FIELDS,
            "limit": str(limit),
            "traceId": traceId,
        }
    else:
        params = {
            "fromStartTime": from_ts,
            "toStartTime": to_ts,
            "isRootObservation": "true",
            "fields": OBS_DETAIL_FIELDS,
            "limit": str(limit),
        }
        if cursor:
            params["cursor"] = cursor
        if level:
            params["level"] = level.upper()
        if type:
            params["type"] = type.upper()
        elif kind in ("workflow", "evaluation", "decision"):
            params["type"] = {"workflow": "CHAIN", "evaluation": "SPAN", "decision": "EVENT"}[kind]
        if environment:
            params["environment"] = environment
        if userId:
            params["userId"] = userId
        if sessionId:
            params["sessionId"] = sessionId

        # v2 observations has no `search` param — name matching must use the
        # JSON `filter` (traceName column), merged with traceTags any-of.
        filter_conditions = []
        if search:
            filter_conditions.append({
                "type": "string",
                "column": "traceName",
                "operator": "contains",
                "value": search,
            })
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            if tag_list:
                filter_conditions.append({
                    "type": "arrayOptions",
                    "column": "traceTags",
                    "operator": "any of",
                    "value": tag_list,
                })
        if filter_conditions:
            params["filter"] = json.dumps(filter_conditions)

    rows = await _query_observations(params)
    meta_cursor = None
    meta = None
    if isinstance(rows, dict):
        meta = rows.get("meta") or {}
        meta_cursor = meta.get("cursor")
        rows = rows.get("data") or []
    if kind:
        # Name-level nuances (system chain, evaluator spans) can't be expressed
        # via the upstream type/search params, so post-filter the page.
        if kind == "system":
            rows = [r for r in rows if (r.get("name") or "") == SYSTEM_CHAIN_NAME]
        elif kind == "evaluation":
            rows = [r for r in rows if (r.get("name") or "").startswith(EVALUATOR_SPAN_PREFIX)]
        elif kind == "workflow":
            rows = [r for r in rows if _classify_kind(r.get("name") or "", r.get("type") or "") == "workflow"]
        elif kind == "decision":
            rows = [r for r in rows if r.get("type") == "EVENT"]

    data = [_map_observation(r) for r in rows]
    trace_ids = {r["traceId"] for r in data if r.get("traceId")}
    if trace_ids:
        aggs = await _trace_aggregates(from_ts, to_ts, trace_ids)
        for row in data:
            a = aggs.get(row.get("traceId"))
            if a:
                row["inputTokens"] = a["inputTokens"]
                row["outputTokens"] = a["outputTokens"]
                row["totalTokens"] = a["totalTokens"]
                row["totalCost"] = a["totalCost"]
                row["inputCost"] = a["inputCost"]
                row["outputCost"] = a["outputCost"]
                row["durationSec"] = a["durationSec"]
                row["levelCounts"] = a["levelCounts"]
                # Trace row shows the highest-severity level across its observations.
                worst = _worst_level(a["levelCounts"])
                if worst:
                    row["level"] = worst
    return {
        "data": data,
        "cursor": meta_cursor,
        "count": len(data),
    }


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str):
    """All observations belonging to a single trace (spans & generations), oldest first."""
    if not trace_id or not trace_id.strip():
        raise HTTPException(status_code=400, detail="traceId is required.")
    rows = await _query_observations({
        "traceId": trace_id,
        "fields": OBS_DETAIL_FIELDS,
        "limit": "500",
    })
    if isinstance(rows, dict):
        rows = rows.get("data") or []
    observations = [_map_observation(r) for r in rows]
    # v2 observations always returns startTime-descending (no order param); the
    # UI expects oldest-first, so re-sort locally.
    observations.sort(key=lambda o: o["startTime"] or "")
    name = ""
    start = None
    end = None
    inputTokens = 0.0
    outputTokens = 0.0
    totalTokens = 0.0
    totalCost = 0.0
    levelCounts = {}
    for o in observations:
        if not name and o["name"]:
            name = o["name"]
        if o["startTime"] and (start is None or o["startTime"] < start):
            start = o["startTime"]
        if o["endTime"] and (end is None or o["endTime"] > end):
            end = o["endTime"]
        inputTokens += o["inputTokens"] or 0
        outputTokens += o["outputTokens"] or 0
        totalTokens += o["totalTokens"] or 0
        totalCost += o["totalCost"] or 0
        if o["level"]:
            levelCounts[o["level"]] = levelCounts.get(o["level"], 0) + 1
    durationSec = None
    if start and end:
        durationSec = round((_parse_iso(end) - _parse_iso(start)).total_seconds(), 4)

    # Pull trace-level metadata from the root observation (first/oldest one)
    root_obs = observations[0] if observations else {}
    user_id = root_obs.get("userId") or ""
    session_id = root_obs.get("sessionId") or ""
    environment = root_obs.get("environment") or ""
    tags = root_obs.get("tags") or []
    worst_level = _worst_level(levelCounts)

    return {
        "traceId": trace_id,
        "name": name,
        "startTime": start,
        "endTime": end,
        "durationSec": durationSec,
        "inputTokens": round(inputTokens, 2),
        "outputTokens": round(outputTokens, 2),
        "totalTokens": round(totalTokens, 2),
        "totalCost": round(totalCost, 6),
        "observationCount": len(observations),
        "levelCounts": levelCounts,
        "level": worst_level,
        "userId": user_id,
        "sessionId": session_id,
        "environment": environment,
        "tags": tags,
        "observations": observations,
    }


# ── Scores (Langfuse v3 scores API) ─────────────────────────────────────────
LANGFUSE_SCORES_PATH = "/api/public/v3/scores"
ALLOWED_DATA_TYPES = {"NUMERIC", "BOOLEAN", "CORRECTION", "CATEGORICAL", "TEXT"}
ALLOWED_SOURCES = {"ANNOTATION", "API", "EVAL"}
SCORES_MAX_LIMIT = 100
# Newer v3 filters: dropped on 400 for deployments that don't support them yet.
SCORES_OPTIONAL_FILTER_PARAMS = {"environment", "authorUserId", "configId", "queueId", "value", "valueMin", "valueMax"}


def _map_score(r: dict) -> dict:
    trace_id, observation_id, session_id, experiment_id = _subject_ids(r)
    sub = r.get("subject") or {}
    trace_obj = r.get("trace") or {}
    trace_name = (
        r.get("traceName")
        or trace_obj.get("name")
        or sub.get("traceName")
        or (sub.get("name") if sub.get("kind") == "trace" else "")
        or ""
    )
    user_id = r.get("authorUserId") or r.get("userId") or trace_obj.get("userId") or ""
    return {
        "id": r.get("id") or "",
        "name": r.get("name") or "",
        "value": r.get("value"),
        "stringValue": r.get("stringValue"),
        "dataType": r.get("dataType") or "",
        "source": r.get("source") or "",
        "environment": r.get("environment") or "",
        "timestamp": r.get("timestamp"),
        "createdAt": r.get("createdAt"),
        "traceId": trace_id,
        "traceName": trace_name,
        "observationId": observation_id,
        "sessionId": session_id,
        "experimentId": experiment_id,
        "authorUserId": user_id,
        "userId": user_id,
        # v3 scores have no level — enriched from the attached trace's
        # observations (highest-severity level) by _enrich_score_trace_names.
        "level": r.get("level") or "",
        "comment": r.get("comment"),
        "metadata": r.get("metadata"),
        "configId": r.get("configId") or "",
        "queueId": r.get("queueId") or "",
        "jobRunId": r.get("jobRunId") or "",
    }


def _subject_ids(r: dict) -> tuple:
    """Resolve trace/observation/session/experiment IDs from the v3 `subject` object.

    Langfuse v3 replaces the flat traceId/observationId/sessionId/datasetRunId
    response fields with a discriminated `subject` union: {kind, id} plus a
    nested traceId for observation-kind subjects. Flat fields are kept as a
    fallback for deployments that return them unchanged.
    """
    sub = r.get("subject") or {}
    kind = sub.get("kind")
    if kind == "trace":
        return sub.get("id") or r.get("traceId") or "", "", "", ""
    if kind == "observation":
        return sub.get("traceId") or r.get("traceId") or "", sub.get("id") or r.get("observationId") or "", "", ""
    if kind == "session":
        return "", "", sub.get("id") or r.get("sessionId") or "", ""
    if kind == "experiment":
        # Dataset-run (experiment) scores are attached to a run, not a trace —
        # expose the run ID and leave trace/observation/session empty.
        return "", "", "", sub.get("id") or r.get("datasetRunId") or r.get("experimentId") or ""
    return (
        r.get("traceId") or "",
        r.get("observationId") or "",
        r.get("sessionId") or "",
        r.get("datasetRunId") or r.get("experimentId") or "",
    )


async def _query_scores(params: dict, fields: Optional[str] = None) -> dict:
    """Execute a Langfuse v3 scores GET and return the parsed JSON body."""
    headers = {"Authorization": _auth_header(), "Accept": "application/json"}
    if fields:
        params = {**params, "fields": fields}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{settings.LANGFUSE_BASE_URL}{LANGFUSE_SCORES_PATH}",
                params=params,
                headers=headers,
            )
    except httpx.HTTPError as e:
        logger.error(f"Langfuse scores request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Langfuse unavailable: {e}")
    if resp.status_code == 400 and fields:
        # Some deployments don't support the v3 `fields` param yet — retry lean.
        logger.info("Langfuse scores rejected fields param (400); retrying without it.")
        return await _query_scores({k: v for k, v in params.items() if k != "fields"}, fields=None)
    if resp.status_code == 400 and SCORES_OPTIONAL_FILTER_PARAMS.intersection(params):
        # Some deployments reject the newer filter params — retry without them.
        logger.info("Langfuse scores rejected optional filter params (400); retrying without them.")
        return await _query_scores(
            {k: v for k, v in params.items() if k not in SCORES_OPTIONAL_FILTER_PARAMS},
            fields=fields,
        )
    if resp.status_code != 200:
        logger.error(f"Langfuse scores error {resp.status_code}: {resp.text[:500]}")
        raise HTTPException(status_code=502, detail=f"Langfuse error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


async def _enrich_score_trace_names(mapped_scores: List[dict], from_ts: Optional[str] = None, to_ts: Optional[str] = None) -> List[dict]:
    """Fill traceName, userId and the trace's worst level for each score.

    v3 scores carry no trace fields and no level, so these are resolved from
    the v2 observations API for the page's trace IDs. A deployment that rejects
    the traceId any-of filter falls back to an unfiltered page (still correct
    for any of the page's trace IDs that appear in it).
    """
    trace_ids = {s["traceId"] for s in mapped_scores if s.get("traceId")}
    if not trace_ids:
        return mapped_scores

    try:
        params = {
            "fields": "core,basic,trace_context",
            "limit": str(OBS_PAGE_LIMIT),
            "filter": json.dumps([{
                "type": "stringOptions",
                "column": "traceId",
                "operator": "any of",
                "value": sorted(trace_ids),
            }]),
        }
        if from_ts and to_ts:
            params["fromStartTime"] = from_ts
            params["toStartTime"] = to_ts
        obs_resp = await _query_observations(params)
        rows = obs_resp.get("data") if isinstance(obs_resp, dict) else (obs_resp or [])
        trace_name_map: dict = {}
        trace_user_map: dict = {}
        trace_level_map: dict = {}
        for r in rows:
            t_id = r.get("traceId") or r.get("id")
            if not t_id:
                continue
            t_name = r.get("traceName") or r.get("name") or ""
            if t_name:
                trace_name_map.setdefault(t_id, t_name)
            u_id = r.get("userId") or r.get("authorUserId") or ""
            if u_id:
                trace_user_map.setdefault(t_id, u_id)
            lvl = r.get("level")
            if lvl:
                cur = trace_level_map.get(t_id)
                if cur is None or LEVEL_SEVERITY.get(lvl, 0) > LEVEL_SEVERITY.get(cur, 0):
                    trace_level_map[t_id] = lvl

        for s in mapped_scores:
            t_id = s.get("traceId")
            if not t_id:
                continue
            if not s.get("traceName"):
                s["traceName"] = trace_name_map.get(t_id, "")
            if not s.get("authorUserId") and not s.get("userId"):
                uid = trace_user_map.get(t_id)
                if uid:
                    s["authorUserId"] = uid
                    s["userId"] = uid
            if not s.get("level"):
                s["level"] = trace_level_map.get(t_id, "")
    except Exception as e:
        logger.warning(f"Could not enrich trace info for scores: {e}")

    return mapped_scores


@router.get("/scores")
async def list_scores(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    name: Optional[str] = Query(None, description="Filter by score name."),
    dataType: Optional[str] = Query(None, description="Filter by data type (NUMERIC|BOOLEAN|CORRECTION|CATEGORICAL)."),
    source: Optional[str] = Query(None, description="Filter by source (ANNOTATION|API|EVAL)."),
    limit: int = Query(SCORES_MAX_LIMIT, ge=1, le=SCORES_MAX_LIMIT),
    cursor: Optional[str] = Query(None, description="Pagination cursor from a previous page."),
    environment: Optional[str] = Query(None, description="Filter by environment tag."),
    authorUserId: Optional[str] = Query(None, description="Filter by the user who created the score."),
    configId: Optional[str] = Query(None, description="Filter by score config ID."),
    queueId: Optional[str] = Query(None, description="Filter by eval queue ID."),
    value: Optional[float] = Query(None, description="Exact numeric score value."),
    valueMin: Optional[float] = Query(None, description="Minimum numeric score value (inclusive)."),
    valueMax: Optional[float] = Query(None, description="Maximum numeric score value (inclusive)."),
    traceId: Optional[str] = Query(None, description="Filter by trace ID (comma-separated = any of). Mutually exclusive with sessionId."),
    sessionId: Optional[str] = Query(None, description="Filter by session ID. Mutually exclusive with traceId."),
    observationId: Optional[str] = Query(None, description="Filter by observation ID. Requires traceId."),
    experimentId: Optional[str] = Query(None, description="Filter by dataset run (experiment) ID. Mutually exclusive with traceId/sessionId/observationId."),
    traceName: Optional[str] = Query(None, description="Filter by attached trace name substring (resolved to trace IDs)."),
):
    """Evaluation scores (annotations, API results, evals) within the given range."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if dataType and dataType.upper() not in ALLOWED_DATA_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported dataType: {dataType}")
    if source and source.upper() not in ALLOWED_SOURCES:
        raise HTTPException(status_code=400, detail=f"Unsupported source: {source}")
    if valueMin is not None and valueMax is not None and valueMin > valueMax:
        raise HTTPException(status_code=400, detail="valueMin must be <= valueMax.")
    # v3 rules: numeric bounds/exact value require dataType=NUMERIC (or an
    # explicit BOOLEAN/CATEGORICAL for the exact value); observationId needs
    # traceId, and traceId/sessionId are mutually exclusive.
    if (valueMin is not None or valueMax is not None) and dataType and dataType.upper() != "NUMERIC":
        raise HTTPException(status_code=400, detail="valueMin/valueMax require dataType=NUMERIC.")
    if value is not None and dataType and dataType.upper() not in ("NUMERIC", "BOOLEAN", "CATEGORICAL"):
        raise HTTPException(status_code=400, detail="value requires dataType of NUMERIC, BOOLEAN or CATEGORICAL.")
    if observationId and not traceId:
        raise HTTPException(status_code=400, detail="observationId requires traceId.")
    if traceId and sessionId:
        raise HTTPException(status_code=400, detail="traceId and sessionId are mutually exclusive.")
    # v3 rule: experimentId is mutually exclusive with traceId/sessionId/observationId.
    if experimentId and (traceId or sessionId or observationId):
        raise HTTPException(status_code=400, detail="experimentId is mutually exclusive with traceId, sessionId and observationId.")

    params = {
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
        "limit": str(limit),
    }
    if name:
        params["name"] = name
    if dataType:
        params["dataType"] = dataType.upper()
    if source:
        params["source"] = source.upper()
    if environment:
        params["environment"] = environment
    if authorUserId:
        params["authorUserId"] = authorUserId
    if configId:
        params["configId"] = configId
    if queueId:
        params["queueId"] = queueId
    if value is not None:
        params["value"] = str(value)
        params.setdefault("dataType", "NUMERIC")
    if valueMin is not None:
        params["valueMin"] = str(valueMin)
        params.setdefault("dataType", "NUMERIC")
    if valueMax is not None:
        params["valueMax"] = str(valueMax)
        params.setdefault("dataType", "NUMERIC")
    if traceId:
        params["traceId"] = traceId
    if sessionId:
        params["sessionId"] = sessionId
    if observationId:
        params["observationId"] = observationId
    if experimentId:
        params["experimentId"] = experimentId
    if cursor:
        params["cursor"] = cursor

    if traceName:
        # v3 scores have no traceName filter — resolve matching traces via the
        # observations API, then pass the trace IDs to the score query.
        try:
            resolved = await _query_observations({
                "fromStartTime": from_ts,
                "toStartTime": to_ts,
                "isRootObservation": "true",
                "fields": "core,basic",
                "limit": str(OBS_PAGE_LIMIT),
                "filter": json.dumps([{
                    "type": "string",
                    "column": "traceName",
                    "operator": "contains",
                    "value": traceName,
                }]),
            })
            resolved_rows = resolved.get("data") if isinstance(resolved, dict) else (resolved or [])
            resolved_ids = sorted({r.get("traceId") for r in resolved_rows if r.get("traceId")})
        except Exception as e:
            logger.warning(f"traceName resolution for scores failed: {e}")
            resolved_ids = []
        if not resolved_ids:
            return {"data": [], "count": 0, "cursor": None, "truncated": False, "limit": SCORES_MAX_LIMIT}
        params["traceId"] = ",".join(resolved_ids[:500])

    body = await _query_scores(params, fields="details,subject,annotation")
    meta_cursor = None
    if isinstance(body, dict):
        meta = body.get("meta") or {}
        meta_cursor = meta.get("cursor")
        rows = body.get("data") or []
    else:
        rows = []
    truncated = len(rows) >= SCORES_MAX_LIMIT
    mapped_scores = [_map_score(r) for r in rows]
    mapped_scores = await _enrich_score_trace_names(mapped_scores, from_ts, to_ts)
    return {
        "data": mapped_scores,
        "count": len(mapped_scores),
        "cursor": meta_cursor,
        "truncated": truncated,
        "limit": SCORES_MAX_LIMIT,
    }


# ── Filter facets (distinct values for dropdown filters) ────────────────────
SCORES_FACET_MAX_PAGES = 20
SCORES_FACET_PAGE_LIMIT = 100
ALLOWED_FACET_ENTITIES = {"traces", "scores"}


async def _fetch_scores_facets(from_ts: str, to_ts: str) -> dict:
    """Distinct authorUserId / configId across scores in range (bounded pages)."""
    author_user_ids = set()
    config_ids = set()
    cursor = None
    for _ in range(SCORES_FACET_MAX_PAGES):
        params = {
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "limit": str(SCORES_FACET_PAGE_LIMIT),
        }
        if cursor:
            params["cursor"] = cursor
        # authorUserId (annotation) and configId (details) are not in the v3
        # core group — request them explicitly or they come back empty.
        body = await _query_scores(params, fields="details,annotation")
        rows = []
        if isinstance(body, dict):
            rows = body.get("data") or []
            meta = body.get("meta") or {}
            cursor = meta.get("cursor")
        for r in rows:
            author_user_ids.add(r.get("authorUserId") or "")
            config_ids.add(r.get("configId") or "")
        if not rows or not cursor:
            break
    return {
        "authorUserId": sorted(v for v in author_user_ids if v),
        "configId": sorted(v for v in config_ids if v),
    }


@router.get("/facets")
async def list_facets(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    entity: str = Query("traces", description="Facet source: traces|scores."),
):
    """Distinct values for the dropdown filters, per entity."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    if entity not in ALLOWED_FACET_ENTITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported facet entity: {entity}")

    if entity == "scores":
        return await _fetch_scores_facets(from_ts, to_ts)

    roots, _ = await _fetch_all_observations(from_ts, to_ts, root_only=True, fields="core,basic,trace_context")
    user_ids = set()
    session_ids = set()
    tags = set()
    for r in roots:
        if r.get("userId"):
            user_ids.add(r["userId"])
        if r.get("sessionId"):
            session_ids.add(r["sessionId"])
        for t in (r.get("tags") or []):
            if t:
                tags.add(t)
    return {
        "userId": sorted(user_ids),
        "sessionId": sorted(session_ids),
        "tags": sorted(tags),
    }


# ── Sessions & workflows (grouped observations) ─────────────────────────────
OBS_PAGE_LIMIT = 1000
OBS_MAX_PAGES = 5


def _percentile(sorted_vals: list, p: float) -> float:
    """Linear-interpolated percentile of a sorted list; None when empty."""
    n = len(sorted_vals)
    if n == 0:
        return None
    k = (n - 1) * p
    f = int(k)
    c = min(f + 1, n - 1)
    w = k - f
    return sorted_vals[f] * (1 - w) + sorted_vals[c] * w


def _worst_level(level_counts: dict) -> Optional[str]:
    """Highest-severity level among the counted observations, or None."""
    if not level_counts:
        return None
    present = {lvl for lvl, cnt in level_counts.items() if cnt}
    if not present:
        return None
    return max(present, key=lambda lvl: LEVEL_SEVERITY.get(lvl, 0))


async def _fetch_all_observations(from_ts: str, to_ts: str, root_only: bool = False, fields: str = "core,basic,usage,metrics", trace_ids: Optional[set] = None):
    """Page through observations in range; returns (rows, truncated).

    Rows carry the requested field groups (default core,basic,usage,metrics).
    When root_only is True only root observations are returned. When trace_ids
    is given, only observations of those traces are fetched (via the traceId
    any-of filter); a Langfuse that rejects the filter falls back to unfiltered.
    """
    rows = []
    cursor = None
    truncated = False
    for _ in range(OBS_MAX_PAGES):
        params = {
            "fromStartTime": from_ts,
            "toStartTime": to_ts,
            "fields": fields,
            "limit": str(OBS_PAGE_LIMIT),
        }
        if root_only:
            params["isRootObservation"] = "true"
        if trace_ids:
            params["filter"] = json.dumps([{
                "type": "stringOptions",
                "column": "traceId",
                "operator": "any of",
                "value": sorted(trace_ids),
            }])
        if cursor:
            params["cursor"] = cursor
        body = await _query_observations(params)
        data, meta = [], {}
        if isinstance(body, dict):
            data = body.get("data") or []
            meta = body.get("meta") or {}
        cursor = meta.get("cursor")
        rows.extend(data)
        if not data or not cursor:
            break
    truncated = bool(cursor)
    return rows, truncated


async def _fetch_all_roots(from_ts: str, to_ts: str, fields: str = "core,basic,usage,metrics"):
    """Page through all root observations in range; returns (rows, truncated)."""
    return await _fetch_all_observations(from_ts, to_ts, root_only=True, fields=fields)


def _parse_iso(iso: str):
    """Parse an ISO-8601 timestamp (as returned by Langfuse) into a datetime."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


async def _trace_aggregates(from_ts: str, to_ts: str, trace_ids: Optional[set] = None) -> dict:
    """Per-trace usage/cost/duration rolled up from every observation.

    Roots carry no usage (usage lives on child GENERATION observations), so
    per-trace totals must be summed across all observations of each trace.
    Returns {traceId: {inputTokens, outputTokens, totalTokens, totalCost, durationSec}}.
    """
    observations, _ = await _fetch_all_observations(from_ts, to_ts, trace_ids=trace_ids)
    agg = {}
    for o in observations:
        tid = o.get("traceId")
        if not tid:
            continue
        a = agg.setdefault(tid, {
            "inputTokens": 0.0,
            "outputTokens": 0.0,
            "totalTokens": 0.0,
            "totalCost": 0.0,
            "inputCost": 0.0,
            "outputCost": 0.0,
            "startTime": None,
            "endTime": None,
            "levelCounts": {},
        })
        a["inputTokens"] += o.get("inputUsage") or 0
        a["outputTokens"] += o.get("outputUsage") or 0
        a["totalTokens"] += o.get("totalUsage") or 0
        a["totalCost"] += o.get("totalCost") or 0
        a["inputCost"] += o.get("inputCost") or 0
        a["outputCost"] += o.get("outputCost") or 0
        lvl = o.get("level")
        if lvl:
            a["levelCounts"][lvl] = a["levelCounts"].get(lvl, 0) + 1
        if o.get("startTime") and o.get("endTime"):
            st, et = _parse_iso(o["startTime"]), _parse_iso(o["endTime"])
            if a["startTime"] is None or st < a["startTime"]:
                a["startTime"] = st
            if a["endTime"] is None or et > a["endTime"]:
                a["endTime"] = et

    result = {}
    for tid, a in agg.items():
        dur = (a["endTime"] - a["startTime"]).total_seconds() if a["startTime"] and a["endTime"] else None
        result[tid] = {
            "inputTokens": round(a["inputTokens"], 2),
            "outputTokens": round(a["outputTokens"], 2),
            "totalTokens": round(a["totalTokens"], 2),
            "totalCost": round(a["totalCost"], 6),
            "inputCost": round(a["inputCost"], 6),
            "outputCost": round(a["outputCost"], 6),
            "durationSec": round(dur, 4) if dur is not None else None,
            "startTime": a["startTime"],
            "endTime": a["endTime"],
            "levelCounts": a["levelCounts"],
        }
    return result


def _span_seconds(start, end) -> Optional[float]:
    """Seconds between two datetimes, or None when either side is missing."""
    if not start or not end:
        return None
    return round((end - start).total_seconds(), 4)


@router.get("/sessions")
async def list_sessions(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    search: Optional[str] = Query(None, description="Filter sessions by session id or user substring."),
    page: int = Query(1, ge=1, description="Page number (1-based)."),
    limit: int = Query(20, ge=1, le=200, description="Number of sessions per page."),
    environment: Optional[str] = Query(None, description="Filter sessions by environment tag (comma-separated = any of)."),
    userId: Optional[str] = Query(None, description="Filter sessions by user ID (comma-separated = any of)."),
    sessionId: Optional[str] = Query(None, description="Filter sessions by session id substring."),
    tags: Optional[str] = Query(None, description="Filter sessions carrying any of these trace tags (comma-separated)."),
    minDurationSec: Optional[float] = Query(None, ge=0),
    maxDurationSec: Optional[float] = Query(None, ge=0),
    minInputTokens: Optional[float] = Query(None, ge=0),
    maxInputTokens: Optional[float] = Query(None, ge=0),
    minOutputTokens: Optional[float] = Query(None, ge=0),
    maxOutputTokens: Optional[float] = Query(None, ge=0),
    minTotalTokens: Optional[float] = Query(None, ge=0),
    maxTotalTokens: Optional[float] = Query(None, ge=0),
    minInputCost: Optional[float] = Query(None, ge=0),
    maxInputCost: Optional[float] = Query(None, ge=0),
    minOutputCost: Optional[float] = Query(None, ge=0),
    maxOutputCost: Optional[float] = Query(None, ge=0),
    minTotalCost: Optional[float] = Query(None, ge=0),
    maxTotalCost: Optional[float] = Query(None, ge=0),
):
    """Traces grouped by session (and user), most recently active first."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")
    for lo, hi, name in [
        (minDurationSec, maxDurationSec, "durationSec"),
        (minInputTokens, maxInputTokens, "inputTokens"),
        (minOutputTokens, maxOutputTokens, "outputTokens"),
        (minTotalTokens, maxTotalTokens, "totalTokens"),
        (minInputCost, maxInputCost, "inputCost"),
        (minOutputCost, maxOutputCost, "outputCost"),
        (minTotalCost, maxTotalCost, "totalCost"),
    ]:
        if lo is not None and hi is not None and lo > hi:
            raise HTTPException(status_code=400, detail=f"min{name.title()} must be <= max{name.title()}.")

    roots, truncated = await _fetch_all_roots(from_ts, to_ts, fields="core,basic,usage,metrics,trace_context")
    if environment:
        envs = {e.strip() for e in environment.split(",") if e.strip()}
        roots = [r for r in roots if (r.get("environment") or "") in envs]
    if userId:
        users = {u.strip() for u in userId.split(",") if u.strip()}
        roots = [r for r in roots if (r.get("userId") or "") in users]
    aggs = await _trace_aggregates(from_ts, to_ts)

    # A trace may expose several roots (workflow chain + HITL event) whose
    # sessionId fields disagree. Attribute each trace to a single session via
    # its primary root (CHAIN > SPAN > EVENT) so usage is neither split nor
    # double-counted across sessions.
    TYPE_RANK = {"CHAIN": 0, "SPAN": 1, "EVENT": 2}
    trace_session = {}
    for r in roots:
        tid = r.get("traceId")
        if not tid:
            continue
        sid = r.get("sessionId") or "(no session)"
        rank = TYPE_RANK.get(r.get("type"), 3)
        cur = trace_session.get(tid)
        if cur is None or rank < cur[0] or (
            rank == cur[0] and sid != "(no session)" and cur[1] == "(no session)"
        ):
            trace_session[tid] = (rank, sid)

    sessions = {}
    for r in roots:
        tid = r.get("traceId")
        sid = trace_session.get(tid, (3, r.get("sessionId") or "(no session)"))[1]
        s = sessions.get(sid)
        if s is None:
            s = sessions[sid] = {
                "sessionId": "" if sid == "(no session)" else sid,
                "userId": r.get("userId") or "",
                "traceCount": 0,
                "inputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 0,
                "totalCost": 0.0,
                "inputCost": 0.0,
                "outputCost": 0.0,
                "latencySumSec": 0.0,
                "tags": set(),
                "startTime": None,
                "endTime": None,
                "seenTraces": set(),
                "lastActivity": None,
                "traces": [],
                "workflows": [],
                "workflowCount": 0,
            }
        s["traceCount"] += 1
        s["latencySumSec"] += r.get("latency") or 0
        st = r.get("startTime")
        if st and (s["lastActivity"] is None or st > s["lastActivity"]):
            s["lastActivity"] = st
        if not s["userId"] and r.get("userId"):
            s["userId"] = r.get("userId")
        for t in (r.get("tags") or []):
            if t:
                s["tags"].add(t)
        if _classify_kind(r.get("name") or "", r.get("type") or "") == "workflow":
            wname = r.get("name") or ""
            if wname and wname not in s["workflows"]:
                s["workflows"].append(wname)

        # Usage/cost/duration live on child observations, so attribute them at
        # trace level (a trace may expose several roots — count usage once).
        tid = r.get("traceId")
        agg = aggs.get(tid) or {}
        if tid and tid not in s["seenTraces"]:
            s["seenTraces"].add(tid)
            s["inputTokens"] += agg.get("inputTokens") or 0
            s["outputTokens"] += agg.get("outputTokens") or 0
            s["totalTokens"] += agg.get("totalTokens") or 0
            s["totalCost"] += agg.get("totalCost") or 0
            s["inputCost"] += agg.get("inputCost") or 0
            s["outputCost"] += agg.get("outputCost") or 0
            if agg.get("startTime") and (s["startTime"] is None or agg["startTime"] < s["startTime"]):
                s["startTime"] = agg["startTime"]
            if agg.get("endTime") and (s["endTime"] is None or agg["endTime"] > s["endTime"]):
                s["endTime"] = agg["endTime"]

        s["traces"].append({
            "id": r.get("id"),
            "traceId": tid,
            "name": r.get("name"),
            "startTime": st,
            "level": r.get("level"),
            "latencySec": r.get("latency"),
            "durationSec": agg.get("durationSec"),
            "inputTokens": agg.get("inputTokens") or 0,
            "outputTokens": agg.get("outputTokens") or 0,
            "totalTokens": agg.get("totalTokens") or 0,
            "totalCost": agg.get("totalCost") or 0,
        })

    result = [
        {
            "sessionId": s["sessionId"],
            "userId": s["userId"],
            "tags": sorted(s["tags"]),
            "traceCount": s["traceCount"],
            "inputTokens": round(s["inputTokens"], 2),
            "outputTokens": round(s["outputTokens"], 2),
            "totalTokens": round(s["totalTokens"], 2),
            "totalCost": round(s["totalCost"], 6),
            "inputCost": round(s["inputCost"], 6),
            "outputCost": round(s["outputCost"], 6),
            "avgLatencySec": round(s["latencySumSec"] / s["traceCount"], 4) if s["traceCount"] else None,
            "durationSec": _span_seconds(s["startTime"], s["endTime"]),
            "lastActivity": s["lastActivity"],
            "traces": sorted(s["traces"], key=lambda t: t["startTime"] or "", reverse=True),
            "workflows": s["workflows"],
            "workflowCount": len(s["workflows"]),
        }
        for s in sessions.values()
    ]
    result.sort(key=lambda s: s["lastActivity"] or "", reverse=True)

    def _in_range(value, lo, hi):
        if lo is None and hi is None:
            return True
        if value is None:
            return False
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    num_bounds = [
        (minDurationSec, maxDurationSec, "durationSec"),
        (minInputTokens, maxInputTokens, "inputTokens"),
        (minOutputTokens, maxOutputTokens, "outputTokens"),
        (minTotalTokens, maxTotalTokens, "totalTokens"),
        (minInputCost, maxInputCost, "inputCost"),
        (minOutputCost, maxOutputCost, "outputCost"),
        (minTotalCost, maxTotalCost, "totalCost"),
    ]
    if tags:
        wanted = {t.strip() for t in tags.split(",") if t.strip()}
        result = [s for s in result if wanted & set(s["tags"])]
    if sessionId:
        needle = sessionId.lower()
        result = [s for s in result if needle in s["sessionId"].lower()]
    for lo, hi, key in num_bounds:
        if lo is None and hi is None:
            continue
        result = [s for s in result if _in_range(s.get(key), lo, hi)]
    if search:
        needle = search.lower()
        result = [s for s in result if needle in s["sessionId"].lower() or needle in (s["userId"] or "").lower()]

    total = len(result)
    start = (page - 1) * limit
    page_rows = result[start : start + limit]

    return {
        "data": page_rows,
        "count": len(page_rows),
        "total": total,
        "page": page,
        "limit": limit,
        "totalTraces": len(roots),
        "truncated": truncated,
        "range": {"from": from_ts, "to": to_ts},
    }


@router.get("/workflows")
async def list_workflows(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    limit: int = Query(20, ge=1, le=50),
):
    """Aggregate telemetry by workflow name (root chain name) for the range."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")

    # Roots carry the workflow name (LangGraph names the root chain after the
    # workflow). A trace may expose several roots (e.g. the workflow chain plus
    # a Human Decision event), so only chains classified as workflow count.
    roots, truncated = await _fetch_all_roots(from_ts, to_ts)
    trace_workflow = {}
    for r in roots:
        name = r.get("name") or ""
        if _classify_kind(name, r.get("type") or "") != "workflow":
            continue
        trace_workflow[r.get("traceId")] = name

    workflows = {}
    for r in roots:
        name = r.get("name") or ""
        if _classify_kind(name, r.get("type") or "") != "workflow":
            continue
        w = workflows.get(name)
        if w is None:
            w = workflows[name] = {
                "name": name,
                "executions": 0,
                "distinctSessions": set(),
                "totalTokens": 0,
                "totalCost": 0.0,
                "latencySumMs": 0.0,
                "latencyCount": 0,
                "lastActivity": None,
            }
        w["executions"] += 1
        sid = r.get("sessionId")
        if sid:
            w["distinctSessions"].add(sid)
        st = r.get("startTime")
        if st and (w["lastActivity"] is None or st > w["lastActivity"]):
            w["lastActivity"] = st

    # Token/cost/latency: query the Metrics API v2 (observation sums grouped by
    # traceName) instead of downloading every observation. Falls back to a
    # workflow-scoped observation fetch if Langfuse returns nothing that maps
    # back to the recognized workflow names.
    workflow_names = set(workflows)
    usage = {}  # workflow name -> {totalTokens, totalCost, avgLatencyMs}
    if workflow_names:
        metric_rows = await _query_metrics({
            "view": "observations",
            "dimensions": [{"field": "traceName"}],
            "metrics": [
                {"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "latency", "aggregation": "avg"},
            ],
            "filters": [],
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "config": {"row_limit": 1000},
        })
        for row in metric_rows:
            name = row.get("traceName")
            if name and name in workflow_names:
                usage[name] = {
                    "totalTokens": _row_value(row, "totalTokens", "sum") or 0,
                    "totalCost": _row_value(row, "totalCost", "sum") or 0,
                    "avgLatencyMs": _row_value(row, "latency", "avg"),
                }

    # On this Langfuse the observations/metrics API only populates traceName on
    # root rows, and roots carry no usage — the metrics-by-traceName path then
    # returns zeros for token/cost even though child generations have tokens.
    # When nothing mapped OR no real usage came back, fall back to summing the
    # workflow traces' observations directly (scoped to those traces only).
    has_real_usage = any(
        (u.get("totalTokens") or 0) or (u.get("totalCost") or 0) for u in usage.values()
    )
    if not has_real_usage and trace_workflow:
        observations, _ = await _fetch_all_observations(
            from_ts, to_ts, trace_ids={tid for tid in trace_workflow if tid}
        )
        for o in observations:
            wname = trace_workflow.get(o.get("traceId"))
            w = workflows.get(wname)
            if w is None:
                continue
            w["totalTokens"] += o.get("totalUsage") or 0
            w["totalCost"] += o.get("totalCost") or 0
            lat = o.get("latency")
            if lat is not None:
                w["latencySumMs"] += float(lat) * 1000.0
                w["latencyCount"] += 1
        for w in workflows.values():
            if w["latencyCount"]:
                usage[w["name"]] = {
                    "totalTokens": w["totalTokens"],
                    "totalCost": w["totalCost"],
                    "avgLatencyMs": w["latencySumMs"] / w["latencyCount"],
                }

    all_rows = []
    for w in workflows.values():
        u = usage.get(w["name"])
        all_rows.append({
            "name": w["name"],
            "executions": w["executions"],
            "distinctSessions": len(w["distinctSessions"]),
            "totalTokens": u["totalTokens"] if u else w["totalTokens"],
            "totalCost": round(u["totalCost"], 6) if u else round(w["totalCost"], 6),
            "avgLatencyMs": round(u["avgLatencyMs"], 2) if u and u["avgLatencyMs"] is not None else None,
            "lastActivity": w["lastActivity"],
        })
    all_rows.sort(key=lambda w: (w["executions"], w["totalTokens"]), reverse=True)

    summary = {
        "executions": sum(r["executions"] for r in all_rows),
        "workflowCount": len(all_rows),
        "totalTokens": sum(r["totalTokens"] for r in all_rows),
        "totalCost": round(sum(r["totalCost"] for r in all_rows), 6),
    }

    return {
        "range": {"from": from_ts, "to": to_ts},
        "summary": summary,
        "data": all_rows[:limit],
        "count": len(all_rows[:limit]),
        "totalExecutions": summary["executions"],
        "truncated": truncated,
    }


# ── Latency percentiles (Trace / Generation / Observation) ──────────────────
LATENCY_PERCENTILES = [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]


def _latency_row(vals: list) -> dict:
    """Compute p50/p90/p95/p99 of a list of latency values (seconds)."""
    ordered = sorted(vals)
    return {
        "p50": round(_percentile(ordered, 0.50), 4),
        "p90": round(_percentile(ordered, 0.90), 4),
        "p95": round(_percentile(ordered, 0.95), 4),
        "p99": round(_percentile(ordered, 0.99), 4),
    }


@router.get("/latency")
async def latency_percentiles(
    fromTimestamp: str = Query(..., description="Start of range (ISO 8601)."),
    toTimestamp: str = Query(..., description="End of range (ISO 8601)."),
    limit: int = Query(10, ge=1, le=25),
):
    """Latency percentiles grouped by trace, generation and observation name."""
    from_ts = _parse_ts(fromTimestamp, "fromTimestamp")
    to_ts = _parse_ts(toTimestamp, "toTimestamp")
    if from_ts >= to_ts:
        raise HTTPException(status_code=400, detail="fromTimestamp must be before toTimestamp.")

    # Single scan: one observations fetch (with trace_context for trace names)
    # replaces the previous roots + observations double fetch. Latency is in
    # seconds on the observations API.
    observations, _ = await _fetch_all_observations(
        from_ts, to_ts, fields="core,basic,metrics,trace_context"
    )

    from datetime import datetime as _dt

    def _parse(iso: str):
        return _dt.fromisoformat(iso.replace("Z", "+00:00"))

    # Trace latency: Langfuse names a trace after its newest root observation
    # (exposed per-observation as traceName in trace_context).
    tr_span = {}
    trace_name = {}
    for o in observations:
        tid = o.get("traceId")
        if not tid:
            continue
        trace_name.setdefault(tid, o.get("traceName") or o.get("name") or "(unnamed)")
        if not o.get("startTime") or not o.get("endTime"):
            continue
        st, et = _parse(o["startTime"]), _parse(o["endTime"])
        cur = tr_span.get(tid)
        if cur is None:
            tr_span[tid] = [st, et]
        else:
            tr_span[tid] = [min(cur[0], st), max(cur[1], et)]

    trace_groups = {}
    for tid, (s, e) in tr_span.items():
        name = trace_name.get(tid, "(unnamed)")
        trace_groups.setdefault(name, []).append((e - s).total_seconds())

    gen_groups = {}
    obs_groups = {}
    for o in observations:
        lat = o.get("latency")
        if lat is None:
            continue
        if o.get("type") == "GENERATION":
            gen_groups.setdefault(o.get("name") or "(unnamed)", []).append(float(lat))
        obs_groups.setdefault((o.get("type") or "", o.get("name") or "(unnamed)"), []).append(float(lat))

    def build(groups, key_fn):
        rows = []
        for key, vals in groups.items():
            row = {"count": len(vals), **key_fn(key), **_latency_row(vals)}
            rows.append(row)
        rows.sort(key=lambda r: r.get("p95") or 0, reverse=True)
        return rows[:limit]

    return {
        "range": {"from": from_ts, "to": to_ts},
        "trace": build(trace_groups, lambda n: {"name": n}),
        "generation": build(gen_groups, lambda n: {"name": n}),
        "observation": build(obs_groups, lambda k: {"type": k[0], "name": k[1]}),
    }


# ── Host system metrics (server-side, not Langfuse) ─────────────────────────

def _bytes2human(n: float) -> str:
    """Human-readable byte size (private psutil helper inlined to avoid the
    `psutil._common` import, which is not part of the public API)."""
    symbols = ("B", "K", "M", "G", "T", "P", "E")
    for sym in symbols:
        if n < 1024.0 or sym == symbols[-1]:
            return f"{n:.1f}{sym}" if sym != "B" else f"{int(n)}B"
        n /= 1024.0
    return "0B"


# Network counters are deltas, so keep the previous sample to compute rates.
_net_baseline = {"when": None, "sent": None, "recv": None}


@router.get("/system-metrics")
async def system_metrics():
    """Live host metrics of the AgentMesh backend: CPU/memory/disk, this
    process's own footprint, network rates and uptime.

    Uses psutil (cross-platform, already a backend dependency). Network rates
    are computed from counter deltas between calls; the first call has no
    baseline and reports 0 rates. Values are clamped to a sane range.
    """
    import psutil
    import platform as _platform

    pid = None
    try:
        proc = psutil.Process()
        pid = proc.pid
        p_cpu = proc.cpu_percent(interval=None)
        p_mem = proc.memory_info()
        p_threads = proc.num_threads()
    except (psutil.Error, OSError):
        proc = None

    global _net_baseline
    net = psutil.net_io_counters()
    now = datetime.now(timezone.utc)
    sent_rate = recv_rate = 0.0
    if _net_baseline["when"] is not None and _net_baseline["sent"] is not None:
        elapsed = max((now - _net_baseline["when"]).total_seconds(), 0.001)
        sent_rate = max((net.bytes_sent - _net_baseline["sent"]) / elapsed, 0.0)
        recv_rate = max((net.bytes_recv - _net_baseline["recv"]) / elapsed, 0.0)
    _net_baseline = {"when": now, "sent": net.bytes_sent, "recv": net.bytes_recv}

    try:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(psutil.disk_partitions(all=False)[0].mountpoint)
    except (psutil.Error, OSError, IndexError):
        raise HTTPException(status_code=503, detail="Host metrics unavailable.")

    try:
        boot = psutil.boot_time()
        uptime_sec = max(int(now.timestamp() - boot), 0)
    except (psutil.Error, OSError):
        uptime_sec = None

    cpu = psutil.cpu_percent(interval=None)
    cores = psutil.cpu_count()

    return {
        "generatedAt": now.isoformat(),
        "host": {
            "hostname": _platform.node() or None,
            "platform": _platform.platform() or None,
        },
        "uptimeSec": uptime_sec,
        "cpu": {
            "percent": max(0.0, min(100.0, cpu or 0.0)),
            "cores": cores,
            "usedCores": round((cores or 0) * (cpu or 0.0) / 100.0, 1),
        },
        "memory": {
            "percent": max(0.0, min(100.0, mem.percent)),
            "used": _bytes2human(mem.used),
            "total": _bytes2human(mem.total),
        },
        "disk": {
            "percent": max(0.0, min(100.0, disk.percent)),
            "used": _bytes2human(disk.used),
            "total": _bytes2human(disk.total),
        },
        "process": {
            "pid": pid,
            "cpuPercent": round(max(0.0, min(100.0, p_cpu or 0.0)), 1) if proc is not None else None,
            "memory": _bytes2human(getattr(p_mem, "rss", 0)) if proc is not None else None,
            "threads": p_threads if proc is not None else None,
        },
        "network": {
            "sentBytesPerSec": round(sent_rate, 2),
            "recvBytesPerSec": round(recv_rate, 2),
        },
    }
