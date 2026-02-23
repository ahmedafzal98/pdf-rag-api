"""Pydantic models for request/response validation"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from uuid import UUID
import os


class ProcessingTask(BaseModel):
    """Task payload sent to queue - Type-safe"""
    task_id: str
    file_path: str
    filename: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class TaskStatusResponse(BaseModel):
    """API response for task status"""
    task_id: str
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    filename: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class PDFMetadata(BaseModel):
    """Extracted PDF metadata"""
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    subject: Optional[str] = None
    title: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None


class PDFExtractionResult(BaseModel):
    """Complete PDF extraction result"""
    task_id: str
    filename: str
    page_count: int
    text: str
    metadata: PDFMetadata
    extraction_time_seconds: float
    summary: Optional[str] = None


class UploadResponse(BaseModel):
    """Response after file upload"""
    task_ids: List[str]
    total_files: int
    message: str


class TaskListResponse(BaseModel):
    """Paginated task list response"""
    tasks: List[TaskStatusResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    redis_connected: bool
    celery_workers: int
    queue_depth: int
    timestamp: str


class FileUploadValidator:
    """Validator for file uploads"""
    
    ALLOWED_EXTENSIONS = {".pdf"}
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Check if filename has valid PDF extension"""
        _, ext = os.path.splitext(filename.lower())
        return ext in FileUploadValidator.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int) -> bool:
        """Check if file size is within limit"""
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    
    @staticmethod
    def validate_file_count(count: int, max_count: int) -> bool:
        """Check if file count is within limit"""
        return 0 < count <= max_count
