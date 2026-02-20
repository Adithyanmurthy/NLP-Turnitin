"""
FastAPI Application for Content Integrity Platform
Person 4: Main FastAPI app with middleware and configuration

Run with: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import time
from pathlib import Path

from api.routes import router, set_pipeline
from api.models import ErrorResponse
from src.pipeline import ContentIntegrityPipeline
from src.config import load_config, CONFIG
from src.utils import setup_logging, get_logger

# Setup logging
setup_logging(CONFIG.log_level)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Content Integrity & Authorship Intelligence Platform",
    description="""
    A comprehensive platform for analyzing text content:
    - **AI Detection**: Detect AI-generated text with high accuracy
    - **Plagiarism Detection**: Find copied or paraphrased content
    - **Humanization**: Transform AI text to human-like writing (target ≤5% AI score)
    - **Deplagiarization**: Rewrite plagiarized sections to eliminate plagiarism
    - **File Upload**: Supports TXT, PDF, DOCX, HTML, MD formats
    
    Built with state-of-the-art NLP models and ensemble techniques.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Rate limiting middleware (simple implementation)
request_counts = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = CONFIG.rate_limit_per_minute

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting by IP address"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    request_counts[client_ip] = [
        t for t in request_counts.get(client_ip, [])
        if current_time - t < RATE_LIMIT_WINDOW
    ]
    
    # Check rate limit
    if len(request_counts.get(client_ip, [])) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "detail": f"Maximum {RATE_LIMIT_MAX} requests per minute"
            }
        )
    
    # Add current request
    request_counts.setdefault(client_ip, []).append(current_time)
    
    response = await call_next(request)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if CONFIG.log_level == "DEBUG" else "An error occurred"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    logger.info("Starting Content Integrity Platform API...")
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = ContentIntegrityPipeline(CONFIG)
        set_pipeline(pipeline)
        
        # Log module status
        health = pipeline.health_check()
        logger.info(f"Module status: {health}")
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.exception("Failed to initialize pipeline")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Content Integrity Platform API...")

# Include routes
app.include_router(router)

# Mount static files for frontend (if exists)
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    logger.info(f"Serving frontend from {frontend_dir}")

# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to API documentation"""
    return {
        "message": "Content Integrity Platform API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.app:app",
        host=CONFIG.api_host,
        port=CONFIG.api_port,
        reload=True,
        log_level=CONFIG.log_level.lower()
    )
