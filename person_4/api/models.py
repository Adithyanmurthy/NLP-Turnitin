"""
Pydantic models for API request/response validation
Person 4: API data models
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class AnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., min_length=10, max_length=50000, description="Text to analyze")
    check_ai: bool = Field(True, description="Run AI detection")
    check_plagiarism: bool = Field(True, description="Check for plagiarism")
    humanize: bool = Field(False, description="Humanize the text")
    deplagiarize: bool = Field(False, description="Deplagiarize the text (rewrite plagiarized sections)")
    use_cache: bool = Field(True, description="Use cached results if available")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v


class AIDetectionResult(BaseModel):
    """AI detection result model"""
    score: float = Field(..., ge=0.0, le=1.0, description="AI probability score (0=human, 1=AI)")
    label: str = Field(..., description="Classification label (human/ai)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model_scores: Dict[str, float] = Field(default_factory=dict, description="Individual model scores")
    error: Optional[str] = Field(None, description="Error message if detection failed")


class PlagiarismMatch(BaseModel):
    """Single plagiarism match"""
    source: str = Field(..., description="Source document identifier")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    sentences: List[Dict[str, Any]] = Field(default_factory=list, description="Matched sentence pairs")


class PlagiarismResult(BaseModel):
    """Plagiarism detection result model"""
    score: float = Field(..., ge=0.0, le=1.0, description="Overall plagiarism score")
    matches: List[PlagiarismMatch] = Field(default_factory=list, description="List of plagiarism matches")
    total_matches: int = Field(..., ge=0, description="Total number of matches found")
    highest_similarity: float = Field(..., ge=0.0, le=1.0, description="Highest similarity score")
    error: Optional[str] = Field(None, description="Error message if check failed")


class HumanizationResult(BaseModel):
    """Humanization result model"""
    text: str = Field(..., description="Humanized output text")
    ai_score_before: float = Field(..., ge=0.0, le=1.0, description="AI score before humanization")
    ai_score_after: float = Field(..., ge=0.0, le=1.0, description="AI score after humanization")
    iterations: int = Field(..., ge=0, description="Number of refinement iterations")
    success: bool = Field(..., description="Whether target AI score was achieved")
    model_used: Optional[str] = Field(None, description="Which humanization model was used")
    error: Optional[str] = Field(None, description="Error message if humanization failed")


class DeplagiarizationResult(BaseModel):
    """Deplagiarization result model"""
    text: str = Field(..., description="Deplagiarized output text")
    plagiarism_score_before: float = Field(..., ge=0.0, le=1.0, description="Plagiarism score before")
    plagiarism_score_after: float = Field(..., ge=0.0, le=1.0, description="Plagiarism score after")
    sentences_rewritten: int = Field(..., ge=0, description="Number of sentences rewritten")
    iterations: int = Field(..., ge=0, description="Number of rewrite cycles")
    success: bool = Field(..., description="Whether target plagiarism score was achieved")
    error: Optional[str] = Field(None, description="Error message if deplagiarization failed")


class FileUploadResponse(BaseModel):
    """Response for file upload endpoint"""
    text: str = Field(..., description="Extracted text from uploaded file")
    format: str = Field(..., description="Detected file format")
    length: int = Field(..., description="Length of extracted text in characters")


class AnalysisResponse(BaseModel):
    """Complete analysis response model"""
    input_length: int = Field(..., description="Length of input text in characters")
    timestamp: float = Field(..., description="Unix timestamp of analysis")
    processing_time: float = Field(..., description="Processing time in seconds")
    ai_detection: Optional[AIDetectionResult] = Field(None, description="AI detection results")
    plagiarism: Optional[PlagiarismResult] = Field(None, description="Plagiarism check results")
    humanization: Optional[HumanizationResult] = Field(None, description="Humanization results")
    deplagiarization: Optional[DeplagiarizationResult] = Field(None, description="Deplagiarization results")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall status (healthy/degraded/unhealthy)")
    ai_detector: bool = Field(..., description="AI detector module status")
    plagiarism_detector: bool = Field(..., description="Plagiarism detector module status")
    humanizer: bool = Field(..., description="Humanizer module status")
    deplagiarizer: bool = Field(False, description="Deplagiarizer module status")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    supported_formats: List[str] = Field(default_factory=list, description="Supported file formats")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
