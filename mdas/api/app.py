"""
mdas/api/app.py — FastAPI application entrypoint.

Run with:
    uvicorn mdas.api.app:app --reload

Docs then live at /docs (Swagger) and /redoc.
"""
from __future__ import annotations

from fastapi import FastAPI

from mdas.api.routers import inference, ingestion, simulation

app = FastAPI(
    title="MDAS",
    description="Multi-source Data Assimilation Service: ingestion -> heuristic "
    "inference -> MAPS simulation-init bridge.",
    version="0.1.0",
)

app.include_router(ingestion.router)
app.include_router(inference.router)
app.include_router(simulation.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
