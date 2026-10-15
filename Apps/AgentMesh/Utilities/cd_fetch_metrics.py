"""
CD Observability Bridge for AgentMesh
======================================
Queries the SAME deployed Prometheus and MongoDB that Credit Decisioning (CD) uses.
No new infrastructure needed — just reads the existing deployed data sources.

Data Sources:
  - Prometheus:  PROMETHEUS_URL env var (same deployed instance CD pushes metrics to)
  - MongoDB:     Cd_Host / Cd_Port / Cd_Username / Cd_Password (same deployed MongoDB)
                 Collections: metrics_persistence, metrics_snapshots

Usage:
    from Apps.AgentMesh.Utilities.cd_fetch_metrics import fetch_cd_metric, get_cd_mongo_db
"""

import os
import logging
import requests
from functools import lru_cache
from typing import Optional, Any

import psutil
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Config from .env (same vars CD uses) ────────────────────────────────────
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

CD_MONGO_HOST     = os.getenv("Cd_Host", "192.168.3.70")
CD_MONGO_PORT     = int(os.getenv("Cd_Port", "27017"))
CD_MONGO_USER     = os.getenv("Cd_Username", "admin")
CD_MONGO_PASS     = os.getenv("Cd_Password")
CD_MONGO_AUTH_DB  = os.getenv("Cd_AuthSource", "admin")
CD_MONGO_DATABASE = os.getenv("Cd_Auth_Database", "CreditDecisioning")


# ─── MongoDB connection (lazy singleton) ─────────────────────────────────────

_cd_client: Optional[MongoClient] = None

def get_cd_mongo_db():
    """
    Returns the shared CreditDecisioning MongoDB database object.
    Connects lazily on first call, reuses the connection after that.
    Uses the exact same connection details as CD's backend.
    """
    global _cd_client
    if _cd_client is None:
        try:
            if not CD_MONGO_PASS:
                raise ValueError("Cd_Password is not configured")
            _cd_client = MongoClient(
                host=CD_MONGO_HOST,
                port=CD_MONGO_PORT,
                username=CD_MONGO_USER,
                password=CD_MONGO_PASS,
                authSource=CD_MONGO_AUTH_DB,
                serverSelectionTimeoutMS=3000,
            )
            # Ping to confirm connection is live
            _cd_client.admin.command("ping")
            logger.info(f"[CD Observability] Connected to CD MongoDB at {CD_MONGO_HOST}:{CD_MONGO_PORT}")
        except Exception as e:
            logger.error(f"[CD Observability] Failed to connect to CD MongoDB: {e}")
            _cd_client = None
    return _cd_client[CD_MONGO_DATABASE] if _cd_client else None


# ─── Prometheus helpers ───────────────────────────────────────────────────────

def fetch_cd_metric(query: str, default: float = 0) -> float:
    """
    Run a PromQL query against the deployed Prometheus instance.
    Same logic as CD's fetch_metrics.py — just reused here.
    Returns a float, or `default` if Prometheus is unreachable / returns no data.
    """
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=2,
        )
        if response.status_code == 200:
            result = response.json().get("data", {}).get("result", [])
            if result:
                return float(result[0]["value"][1])
    except Exception as e:
        logger.debug(f"[CD Observability] Prometheus query failed: {query!r} — {e}")
    return default


# ─── MongoDB offset helpers ───────────────────────────────────────────────────

def _make_key(metric_name: str, labels: dict) -> str:
    """Reproduces CD's key format: 'metric_name|label1=v1,label2=v2' (sorted)."""
    sorted_labels = sorted(labels.items())
    label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
    return f"{metric_name}|{label_str}"


def get_cd_mongo_offset(metric_name: str, labels: dict = {}) -> float:
    """
    Read CD's MongoDB `metrics_persistence` collection for a specific metric+labels combo.
    This is the durable "offset" CD adds on top of the live Prometheus value to handle
    server restarts (Prometheus resets on restart; MongoDB doesn't).

    Returns the offset value, or 0 if not found.
    """
    db = get_cd_mongo_db()
    if db is None:
        return 0
    try:
        key = _make_key(metric_name, labels)
        doc = db.metrics_persistence.find_one({"key": key, "metric_type": {"$ne": "histogram"}})
        return float(doc["value"]) if doc and "value" in doc else 0
    except Exception as e:
        logger.debug(f"[CD Observability] MongoDB offset lookup failed: {e}")
        return 0


def get_cd_mongo_offset_aggregated(metric_name: str, partial_labels: dict = {}) -> float:
    """
    Sum ALL MongoDB offset entries for a metric where labels contain the partial_labels.
    e.g. sum input tokens across ALL agents:
        get_cd_mongo_offset_aggregated('creditdecisioning_llm_tokens_total', {'token_type': 'input'})
    """
    db = get_cd_mongo_db()
    if db is None:
        return 0
    try:
        docs = list(db.metrics_persistence.find({"metric_name": metric_name}))
        total = 0.0
        for doc in docs:
            doc_labels = doc.get("labels", {})
            if all(doc_labels.get(k) == v for k, v in partial_labels.items()):
                total += float(doc.get("value", 0))
        return total
    except Exception as e:
        logger.debug(f"[CD Observability] MongoDB aggregated offset lookup failed: {e}")
        return 0


def get_cd_metric_snapshot(metric_key: str, default: float = 0) -> float:
    """
    Read CD's MongoDB `metrics_snapshots` collection — used for gauges/rates
    that aren't cumulative counters (e.g. 'observability.toxicity_score').
    """
    db = get_cd_mongo_db()
    if db is None:
        return default
    try:
        doc = db.metrics_snapshots.find_one({"metric_key": metric_key})
        return float(doc["value"]) if doc and "value" in doc else default
    except Exception as e:
        logger.debug(f"[CD Observability] Snapshot lookup failed for {metric_key!r}: {e}")
        return default


def fetch_cd_metric_with_offset(query: str, metric_name: str, labels: dict = {}, default: float = 0) -> float:
    """
    Full CD-style fetch: Prometheus live value + MongoDB durable offset.
    Formula: total = prometheus_value + mongodb_offset
    This mirrors CD's fetch_metric_with_offset() exactly.
    """
    prometheus_val = fetch_cd_metric(query, default=0)
    mongo_offset = get_cd_mongo_offset_aggregated(metric_name, labels)
    total = prometheus_val + mongo_offset
    logger.debug(
        f"[CD Observability] {metric_name} → prometheus={prometheus_val}, offset={mongo_offset}, total={total}"
    )
    return total


# ─── System metrics ───────────────────────────────────────────────────────────

def get_system_cpu() -> tuple[str, str]:
    """Returns (cpu_value_str, subtitle_str)"""
    try:
        val = psutil.cpu_percent(interval=0.5)
        return f"{round(val, 1)}%", "Current system CPU usage"
    except Exception:
        return "N/A", "CPU metrics unavailable"


def get_system_memory() -> tuple[str, str]:
    """Returns (memory_value_str, subtitle_str)"""
    try:
        mem = psutil.virtual_memory()
        used_gb = round(mem.used / (1024 ** 3), 1)
        avail_gb = round(mem.available / (1024 ** 3), 1)
        return f"{used_gb}GB ({mem.percent}%)", f"{avail_gb}GB available"
    except Exception:
        return "N/A", "Memory metrics unavailable"
