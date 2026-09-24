"""
Brain MRI Tumor Classification - Inference Backend API.
Built with FastAPI & Uvicorn.

Educational and Research Notice:
This application is intended for educational and research purposes only.
It is not a medical diagnostic system and should not be used for clinical decision-making.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ensure torch backend for Keras
os.environ.setdefault("KERAS_BACKEND", "torch")

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ExplanationResponse,
    ErrorResponse,
)
from backend.services.model_service import model_service
from backend.services.prediction_service import prediction_service
from backend.services.explanation_service import explanation_service

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("brain_tumor_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the trained model once at startup and performs resource teardown on shutdown.
    """
    logger.info("Initializing Brain MRI Classification Backend...")
    try:
        model_service.load_model()
        logger.info("Model loaded successfully. Ready to receive requests.")
    except Exception as exc:
        logger.critical(f"FATAL: Model loading failed during startup: {exc}")
        # Note: application can still report model_loaded=False via /health
    yield
    logger.info("Shutting down Brain MRI Classification Backend.")


# Initialize FastAPI Application
app = FastAPI(
    title="Brain MRI Classification Inference API",
    description=(
        "Production-grade inference backend for multi-class Brain MRI tumor classification "
        "and Grad-CAM explainability powered by EfficientNetB0.\n\n"
        "**Educational and Research Notice:**\n"
        "This application is intended for educational and research purposes only. "
        "It is not a medical diagnostic system and should not be used for clinical decision-making."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
# Supports local frontend dev as well as cloud deployment environments
ENV_CORS = os.getenv("CORS_ORIGINS")
CORS_ORIGINS = [orig.strip() for orig in ENV_CORS.split(",")] if ENV_CORS else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file endpoints to serve generated charts, reports, and demonstration samples
from fastapi.staticfiles import StaticFiles
from src.config import RESULTS_DIR, BASE_DIR

RESULTS_PATH = RESULTS_DIR
SAMPLES_PATH = BASE_DIR / "samples"


class SafeStaticFiles(StaticFiles):
    """
    Static file server that restricts served files strictly to approved visualization images
    (.png, .jpg, .jpeg, .webp, .svg). Prohibits any static serving of dataset CSVs, raw
    tables, or raw datasets to guarantee complete dataset isolation.
    """
    async def get_response(self, path: str, scope):
        allowed_exts = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
        p = Path(path)
        if (
            p.suffix.lower() not in allowed_exts
            or "archive" in path.lower()
            or "test_predictions" in path.lower()
            or "misclassified_images" in path.lower()
            or "dataset" in path.lower()
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or access restricted.",
            )
        return await super().get_response(path, scope)


if RESULTS_PATH.exists():
    app.mount("/results", SafeStaticFiles(directory=str(RESULTS_PATH)), name="results")

if SAMPLES_PATH.exists():
    app.mount("/samples", SafeStaticFiles(directory=str(SAMPLES_PATH)), name="samples")


# Standardized Custom Exception Handlers (no stack traces leaked)
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Returns standardized JSON error messages for known HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled exceptions, logs internal error, and returns safe response to client."""
    logger.error(f"Unhandled exception during {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An internal server error occurred while processing the request."},
    )


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
    description="Returns the operating status of the API and indicates whether the model is loaded in memory.",
)
async def health_check():
    """Checks service operational status and model readiness."""
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded(),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Model Architecture and Metadata",
    description="Returns model architecture, input dimensions, class mapping, benchmark test accuracy, and project scope.",
)
async def get_model_info():
    """Returns architectural parameters and benchmark statistics."""
    info = model_service.get_model_info()
    return ModelInfoResponse(**info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format or corrupted file"},
        415: {"model": ErrorResponse, "description": "Unsupported media format"},
        503: {"model": ErrorResponse, "description": "Model not loaded or unavailable"},
    },
    tags=["Inference"],
    summary="Predict Brain MRI Tumor Classification",
    description=(
        "Accepts a single brain MRI image file (JPG or PNG, max 10MB), applies deterministic "
        "aspect-ratio preserving letterbox preprocessing to 224x224, and returns the predicted class "
        "with model confidence scores across all 4 target classes.\n\n"
        "**Note:** Confidence represents model confidence score, not clinical probability."
    ),
)
async def predict_mri(file: UploadFile = File(...)):
    """Predicts tumor class from uploaded brain MRI scan."""
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or still loading.",
        )

    # Basic content-type sanity check
    content_type = (file.content_type or "").lower()
    if content_type and not (content_type.startswith("image/") or "octet-stream" in content_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload a valid JPG or PNG image.",
        )

    try:
        image_bytes = await file.read()
        result = prediction_service.predict(image_bytes, filename=file.filename or "upload.jpg")
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Inference processing failed for {file.filename}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during image inference processing.",
        )


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format or corrupted file"},
        415: {"model": ErrorResponse, "description": "Unsupported media format"},
        503: {"model": ErrorResponse, "description": "Model not loaded or unavailable"},
    },
    tags=["Explainability"],
    summary="Grad-CAM Visual Explanation",
    description=(
        "Accepts a brain MRI image, computes gradient class activation maps (Grad-CAM) "
        "on the final convolutional layer of EfficientNetB0, and returns the classification "
        "along with base64 Data URIs of the overlay, heatmap, and 3-panel figure.\n\n"
        "**Interpretation:** Visualizations show spatial feature contributions to the model's "
        "decision. They do not represent biological or clinical lesion boundaries."
    ),
)
async def explain_mri(file: UploadFile = File(...)):
    """Generates Grad-CAM activation visualization for uploaded brain MRI scan."""
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or still loading.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and not (content_type.startswith("image/") or "octet-stream" in content_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload a valid JPG or PNG image.",
        )

    try:
        image_bytes = await file.read()
        result = explanation_service.explain(image_bytes, filename=file.filename or "upload.jpg")
        return ExplanationResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Explanation generation failed for {file.filename}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating Grad-CAM explainability.",
        )


# Mount built frontend single-page application (SPA) if present
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    from fastapi.responses import FileResponse

    if (FRONTEND_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend_assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(request: Request, full_path: str):
        # Do not intercept API endpoints, docs, or explicit static mounts
        if full_path.startswith(("health", "model-info", "predict", "explain", "docs", "redoc", "openapi.json", "results", "samples")):
            raise HTTPException(status_code=404, detail="Endpoint not found")
        
        target_file = FRONTEND_DIST / full_path
        if target_file.is_file():
            return FileResponse(target_file)
        
        index_file = FRONTEND_DIST / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend build index.html not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
