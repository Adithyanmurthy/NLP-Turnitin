"""
API routes for Content Integrity Platform
Person 4: FastAPI route handlers
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
import time
import tempfile
import os

from api.models import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
    ErrorResponse,
    AIDetectionResult,
    PlagiarismResult,
    HumanizationResult,
    DeplagiarizationResult,
    FileUploadResponse,
)
from src.pipeline import ContentIntegrityPipeline
from src.file_parser import parse_file, SUPPORTED_EXTENSIONS
from src.utils import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Global pipeline instance (initialized in app.py)
pipeline: ContentIntegrityPipeline = None


def set_pipeline(p: ContentIntegrityPipeline):
    """Set the global pipeline instance"""
    global pipeline
    pipeline = p


@router.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "name": "Content Integrity & Authorship Intelligence Platform",
        "version": "1.0.0",
        "description": "AI Detection, Plagiarism Detection, Humanization & Deplagiarization API",
        "supported_formats": [".txt", ".pdf", ".docx", ".html", ".md"],
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "detect_ai": "/detect-ai",
            "check_plagiarism": "/check-plagiarism",
            "humanize": "/humanize",
            "deplagiarize": "/deplagiarize",
            "upload": "/upload",
            "upload_and_analyze": "/upload-and-analyze"
        }
    }


@router.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns the status of all modules and system health
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    health_status = pipeline.health_check()
    
    # Determine overall status
    all_healthy = all([
        health_status['ai_detector'],
        health_status['plagiarism_detector'],
        health_status['humanizer']
    ])
    
    if all_healthy:
        overall_status = "healthy"
    elif any([health_status['ai_detector'], health_status['plagiarism_detector']]):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        ai_detector=health_status['ai_detector'],
        plagiarism_detector=health_status['plagiarism_detector'],
        humanizer=health_status['humanizer'],
        cache_enabled=health_status['cache_enabled'],
        version="1.0.0",
        deplagiarizer=health_status.get('deplagiarizer', False),
        supported_formats=health_status.get('supported_formats', [])
    )


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text(request: AnalysisRequest):
    """
    Complete text analysis endpoint
    
    Runs AI detection, plagiarism check, and/or humanization based on request parameters
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        logger.info(f"Analysis request received: text_length={len(request.text)}")
        
        # Run analysis
        report = pipeline.analyze(
            text=request.text,
            check_ai=request.check_ai,
            check_plagiarism=request.check_plagiarism,
            humanize=request.humanize,
            deplagiarize=request.deplagiarize,
            use_cache=request.use_cache
        )
        
        # Convert to response model
        response = AnalysisResponse(
            input_length=report['input_length'],
            timestamp=report['timestamp'],
            processing_time=report['processing_time']
        )
        
        # Add AI detection results
        if 'ai_detection' in report:
            ai_data = report['ai_detection']
            response.ai_detection = AIDetectionResult(**ai_data)
        
        # Add plagiarism results
        if 'plagiarism' in report:
            plag_data = report['plagiarism']
            response.plagiarism = PlagiarismResult(**plag_data)
        
        # Add humanization results
        if 'humanization' in report:
            human_data = report['humanization']
            response.humanization = HumanizationResult(**human_data)
        
        # Add deplagiarization results
        if 'deplagiarization' in report:
            deplag_data = report['deplagiarization']
            response.deplagiarization = DeplagiarizationResult(**deplag_data)
        
        logger.info(f"Analysis completed in {report['processing_time']:.2f}s")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/detect-ai", response_model=AIDetectionResult, tags=["AI Detection"])
async def detect_ai(text: str, use_cache: bool = True):
    """
    AI detection only endpoint
    
    Detects whether text is AI-generated or human-written
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        logger.info(f"AI detection request: text_length={len(text)}")
        result = pipeline.detect_ai(text, use_cache)
        return AIDetectionResult(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("AI detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI detection failed: {str(e)}"
        )


@router.post("/check-plagiarism", response_model=PlagiarismResult, tags=["Plagiarism"])
async def check_plagiarism(text: str, use_cache: bool = True):
    """
    Plagiarism check only endpoint
    
    Checks text for plagiarism against reference corpus
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        logger.info(f"Plagiarism check request: text_length={len(text)}")
        result = pipeline.check_plagiarism(text, use_cache)
        return PlagiarismResult(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Plagiarism check failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plagiarism check failed: {str(e)}"
        )


@router.post("/humanize", response_model=HumanizationResult, tags=["Humanization"])
async def humanize_text(text: str, use_cache: bool = False):
    """
    Humanization only endpoint
    
    Transforms AI-generated text to human-like text (target: ≤5% AI score)
    Uses multi-model fallback: Flan-T5 → PEGASUS → Mistral
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        logger.info(f"Humanization request: text_length={len(text)}")
        result = pipeline.humanize(text, use_cache)
        return HumanizationResult(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Humanization failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Humanization failed: {str(e)}"
        )


@router.post("/deplagiarize", response_model=DeplagiarizationResult, tags=["Deplagiarization"])
async def deplagiarize_text(text: str):
    """
    Deplagiarization endpoint
    
    Detects plagiarized sections and rewrites them until plagiarism ≤5%.
    Uses Person 2's detector to find matches, Person 3's humanizer to rewrite.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    try:
        logger.info(f"Deplagiarization request: text_length={len(text)}")
        result = pipeline.deplagiarize(text)
        return DeplagiarizationResult(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Deplagiarization failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deplagiarization failed: {str(e)}"
        )


@router.post("/upload", response_model=FileUploadResponse, tags=["File Upload"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and extract text from it.
    
    Supports: TXT, PDF, DOCX, HTML, MD
    Returns extracted text that can be used with other endpoints.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    # Check file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Parse the file
        text, fmt = parse_file(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        logger.info(f"File uploaded: {filename} ({fmt}, {len(text)} chars)")
        return FileUploadResponse(text=text, format=fmt, length=len(text))
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("File upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )


@router.post("/upload-and-analyze", tags=["File Upload"])
async def upload_and_analyze(
    file: UploadFile = File(...),
    check_ai: bool = Form(True),
    check_plagiarism: bool = Form(True),
    humanize: bool = Form(False),
    deplagiarize: bool = Form(False),
):
    """
    Upload a file and run full analysis in one step.
    
    Supports: TXT, PDF, DOCX, HTML, MD
    Combines file upload + text analysis into a single request.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    
    try:
        # Save and parse file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        text, fmt = parse_file(tmp_path)
        os.unlink(tmp_path)
        
        logger.info(f"Upload+Analyze: {filename} ({fmt}, {len(text)} chars)")
        
        # Run analysis on extracted text
        report = pipeline.analyze(
            text=text,
            check_ai=check_ai,
            check_plagiarism=check_plagiarism,
            humanize=humanize,
            deplagiarize=deplagiarize,
            use_cache=True,
        )
        
        # Build response
        response = AnalysisResponse(
            input_length=report['input_length'],
            timestamp=report['timestamp'],
            processing_time=report['processing_time'],
        )
        if 'ai_detection' in report:
            response.ai_detection = AIDetectionResult(**report['ai_detection'])
        if 'plagiarism' in report:
            response.plagiarism = PlagiarismResult(**report['plagiarism'])
        if 'humanization' in report:
            response.humanization = HumanizationResult(**report['humanization'])
        if 'deplagiarization' in report:
            response.deplagiarization = DeplagiarizationResult(**report['deplagiarization'])
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Upload and analyze failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )
