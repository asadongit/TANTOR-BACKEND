import os
from dotenv import load_dotenv
# Load environment variables from the absolute path of the backend directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from contextlib import asynccontextmanager
import logging
import sys

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Core.Lifespan import agent_mesh_lifespan, register_agent_mesh_mounts
from Apps.Auth.agentmesh_auth_middleware import AgentMeshAuthenticationMiddleware
from Database.Database import engine
from Database.Models import Base

# Configure logging streams for Windows encoding compatibility
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8', errors='backslashreplace')
    ]
)

logger = logging.getLogger(__name__)

from Apps.AgentMesh.Utilities.Security import (
    install_sensitive_data_log_filter,
    redact_sensitive_data,
    sanitize_for_logging,
)

install_sensitive_data_log_filter()

async def infrastructure_startup():
    """Handles global infrastructure setup like Database creation and Schema sync."""
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine
    
    # 1. Database Creation (Postgres specific)
    if "postgresql" in settings.DATABASE_URL:
        admin_url = make_url(settings.async_database_url).set(database="postgres")
        base_url = admin_url.render_as_string(hide_password=False)

        default_engine = create_async_engine(base_url, isolation_level="AUTOCOMMIT")
        try:
            async with default_engine.connect() as conn:
                result = await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = 'agentmesh'"))
                if not result.scalar():
                    logger.info("Database 'agentmesh' not found. Creating...")
                    await conn.execute(text("CREATE DATABASE agentmesh"))
        except Exception as e:
            logger.error(f"Postgres initialization check failed: {e}")
        finally:
            await default_engine.dispose()

    # 2. Schema Synchronization
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema verified.")
    except Exception as e:
        logger.error(f"Schema sync error: {e}")

def _preload_toxicity_model():
    """Preload the Detoxify model in a background thread so the first workflow
    run isn't killed by the scoring timeout during model cold-start (~5-10s)."""
    import threading
    from Apps.AgentMesh.Utilities.toxicity import warm_up

    thread = threading.Thread(target=warm_up, daemon=True, name="toxicity-warmup")
    thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Global Application Lifespan - Orchestrates all sub-app services."""
    logger.info("Starting Global Infrastructure...")
    await infrastructure_startup()

    _preload_toxicity_model()
    
    # Delegate to AgentMesh domain lifespan
    async with agent_mesh_lifespan(app):
        logger.info("FastAPI application initialized successfully.")
        yield
    
    logger.info("FastAPI application shutting down...")

# Initialize FastAPI with Lifespan
app = FastAPI(lifespan=lifespan)


@app.exception_handler(HTTPException)
async def safe_http_exception_handler(request: Request, exc: HTTPException):
    """Ensure API error details cannot echo credentials supplied by a caller."""
    safe_detail = sanitize_for_logging(redact_sensitive_data(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": safe_detail},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def safe_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return validation metadata without echoing submitted credential values."""
    safe_errors = [
        {
            "type": item.get("type"),
            "loc": item.get("loc"),
            "msg": sanitize_for_logging(item.get("msg", "Invalid request")),
        }
        for item in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": safe_errors})

# Register Domain Static Mounts
register_agent_mesh_mounts(app)

# Import and Include routers
from Apps.AgentMesh.Api.api import router as AgentMeshRouter
from Apps.UseCases.KYC.Api.api import router as KYCUseCaseRouter

app.add_middleware(AgentMeshAuthenticationMiddleware)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Mute logs for health and high-frequency active-runs
        if request.url.path in ["/api/v1/agentmesh/active-runs", "/health"]:
            return await call_next(request)

        # Query strings may contain OAuth codes or webhook signatures.
        logger.debug(f"=== Incoming: {request.method} {request.url.path} ===")
        response = await call_next(request)
        logger.debug(f"=== Response: {response.status_code} ===")
        return response

app.add_middleware(LoggingMiddleware)

# Keep CORS outermost so browser clients also receive CORS headers on 401 responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["System"])
async def health_check():
    """Service health check for production monitoring."""
    return {"status": "healthy", "app": "AgentMesh", "version": "1.1.0"}

app.include_router(AgentMeshRouter, prefix="/api/v1/agentmesh", tags=["AgentMesh Core"])
app.include_router(KYCUseCaseRouter, prefix="/api/v1/kyc", tags=["KYC UseCase"])

logger.info("Global Router Map defined.")
