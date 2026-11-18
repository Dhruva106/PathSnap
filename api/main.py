"""
api/main.py - Main FastAPI Application for PathSnap Map-Matching System
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routes.health import router as health_router
from api.routes.model_info import router as model_info_router
from api.routes.map_match import router as map_match_router
from api.routes.comparison import router as comparison_router
from api.services.map_matching_service import get_map_matching_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: Preloads champion model, HMM parameters,
    and road network utilities into memory prior to receiving traffic.
    """
    print("[PathSnap API] Starting server and preloading champion ML model...")
    service = get_map_matching_service()
    print(f"[PathSnap API] Model successfully preloaded: {service.metadata.get('model_name')}")
    yield
    print("[PathSnap API] Shutting down server...")


app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description=(
        "Production-grade RESTful API powering vehicular GPS map-matching and continuous "
        "topological road route reconstruction using Extreme Gradient Boosting (XGBoost) "
        "and Hidden Markov Model (HMM) Viterbi sequence decoding."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for frontend and evaluation clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds X-Process-Time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


# Register sub-routers
app.include_router(health_router)
app.include_router(model_info_router)
app.include_router(map_match_router)
app.include_router(comparison_router)

# Mount frontend static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
if os.path.exists(frontend_dir):
    @app.get("/ui", include_in_schema=False)
    def redirect_ui():
        return RedirectResponse(url="/ui/")

    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
