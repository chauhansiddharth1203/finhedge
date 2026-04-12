"""
main.py – FinHedge FastAPI application entry point.

Registers all routers, sets up Prometheus /metrics endpoint,
configures structured logging and global exception handlers.

Run locally:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from backend.api.routes import health, prediction, hedging, pipeline
from backend.config import LOG_LEVEL, API_HOST, API_PORT

# ── Logging setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/backend.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── App lifecycle ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinHedge backend starting up …")
    yield
    logger.info("FinHedge backend shutting down.")


# ── App factory ────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinHedge API",
    description=(
        "Stock price prediction and deep-hedging recommendation API. "
        "Powered by LSTM, XGBoost, and CVaR-optimal deep hedging."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS (allow Streamlit frontend) ───────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0       = time.time()
    response = await call_next(request)
    duration = time.time() - t0
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    return response


# ── Global exception handler ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(hedging.router)
app.include_router(pipeline.router)

# ── Prometheus metrics endpoint ────────────────────────────────────────────

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Root redirect ──────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {"message": "FinHedge API v1.0.0 — see /docs for API reference."}


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )
