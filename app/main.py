"""FastAPI application - Producer (REST API)"""
import os
import uuid
import json
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas_api import (
    UploadResponse, TaskStatusResponse, PDFExtractionResult,
    TaskListResponse, HealthResponse, FileUploadValidator
)
from app.dependencies import (
    redis_client, timeit, rate_limit, log_request,
    get_all_tasks_generator, stream_task_results
)
from app.aws_services import aws_services
from app.database import init_db, get_async_db
from app.db_models import User, Document
from app.schemas import DocumentResponse, DocumentCreate, UserResponse, UserCreate, ChatRequest, ChatResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import Depends
from app.chat_service import chat_service

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing System",
    description="Scalable PDF processing with FastAPI, Celery, and Redis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= STARTUP EVENT =============

@app.on_event("startup")
async def startup_event():
    """
    Initialize database on application startup
    Creates all tables if they don't exist
    """
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        # Don't prevent app startup if DB fails - Redis can still work
        pass


# ============= ENDPOINTS =============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Document Processing System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@timeit
async def health_check():
    """
    Health check endpoint
    Returns system status, Redis connection, and SQS queue depth
    
    Note: SQS queue depth is fetched with a timeout to prevent slow health checks
    """
    try:
        # Check Redis connection (fast)
        redis_client.ping()
        redis_connected = True
        
        # Get SQS queue depth with error handling (can be slow)
        queue_depth = 0
        try:
            # Use a cached value if available (10 second cache)
            cache_key = "health:sqs_queue_depth"
            cached_depth = redis_client.get(cache_key)
            
            if cached_depth:
                queue_depth = int(cached_depth)
            else:
                # Fetch from AWS (can be slow on first call)
                queue_attrs = aws_services.get_queue_attributes()
                queue_depth = int(queue_attrs.get('ApproximateNumberOfMessages', 0)) if queue_attrs else 0
                # Cache for 10 seconds
                redis_client.setex(cache_key, 10, queue_depth)
        except Exception as sqs_error:
            logger.warning(f"Failed to get SQS queue depth: {sqs_error}")
            queue_depth = -1  # Indicates SQS check failed
        
        # Note: SQS workers are independent processes, not tracked here
        celery_workers = 0  # Deprecated - using SQS now
        
        return HealthResponse(
            status="healthy",
            redis_connected=redis_connected,
            celery_workers=celery_workers,
            queue_depth=queue_depth,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            redis_connected=False,
            celery_workers=0,
            queue_depth=0,
            timestamp=datetime.now().isoformat()
        )


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Upload"])
@timeit
@log_request
@rate_limit(max_requests=10, window_seconds=60)
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    user_id: int = 1,  # Accept user_id as query parameter (default to 1)
    prompt: Optional[str] = None,  # Optional prompt for AI summarization
    db: AsyncSession = Depends(get_async_db)
):
    """
    Upload multiple PDF files for processing
    
    Decorators:
    - @timeit: Monitors endpoint performance
    - @log_request: Audit trail
    - @rate_limit: Max 10 uploads per minute per IP
    
    Returns:
    - task_ids: List of task IDs for tracking
    - 202 Accepted: Files queued for processing
    """
    # Validate file count
    if not FileUploadValidator.validate_file_count(len(files), settings.max_files_per_request):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum {settings.max_files_per_request} files per request."
        )
    
    # Check SQS queue depth for backpressure
    queue_attrs = aws_services.get_queue_attributes()
    queue_depth = int(queue_attrs.get('ApproximateNumberOfMessages', 0)) if queue_attrs else 0
    if queue_depth > 1000:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is at capacity. Please try again in a few minutes."
        )
    
    task_ids = []
    
    for file in files:
        # Validate file type
        if not FileUploadValidator.validate_filename(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.filename}. Only PDF files are allowed."
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file size
        if not FileUploadValidator.validate_file_size(file_size, settings.max_file_size_mb):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large: {file.filename}. Maximum size is {settings.max_file_size_mb}MB."
            )
        
        # Generate unique task ID and S3 key
        task_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        s3_key = f"uploads/{task_id}{file_extension}"
        
        # Upload file to S3
        upload_success = aws_services.upload_file_to_s3(
            file_content=content,
            s3_key=s3_key,
            content_type="application/pdf"
        )
        
        if not upload_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload {file.filename} to S3"
            )
        
        # ⭐ NEW: Create Document record in PostgreSQL
        # Get or create user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            # Create user if doesn't exist
            user = User(
                email=f"user{user_id}@docprocessor.local",
                api_key=f"user{user_id}_key_" + datetime.now().strftime("%Y%m%d%H%M%S")
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        
        # Create document record in PostgreSQL
        document = Document(
            user_id=user.id,
            filename=file.filename,
            s3_key=s3_key,
            status="PENDING"
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # ⭐ NEW: Use PostgreSQL document ID as task_id
        task_id = str(document.id)
        
        # ⚡ KEEP: Create task metadata in Redis (for real-time progress)
        task_data = {
            "task_id": task_id,
            "document_id": document.id,  # ← Link to PostgreSQL
            "status": "PENDING",
            "progress": 0,
            "filename": file.filename,
            "s3_key": s3_key,
            "s3_bucket": settings.s3_bucket_name,
            "created_at": datetime.now().isoformat(),
            "started_at": "",
            "completed_at": "",
            "error": "",
            "prompt": prompt or ""
        }
        
        redis_client.hset(f"task:{task_id}", mapping=task_data)
        redis_client.rpush("all_tasks", task_id)
        
        # Send message to SQS
        sqs_message = {
            "task_id": task_id,
            "s3_bucket": settings.s3_bucket_name,
            "s3_key": s3_key,
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "prompt": prompt or ""
        }
        
        message_id = aws_services.send_message_to_sqs(
            message_body=sqs_message,
            message_attributes={"task_id": task_id}
        )
        
        if not message_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue {file.filename} for processing"
            )
        
        task_ids.append(task_id)
    
    return UploadResponse(
        task_ids=task_ids,
        total_files=len(files),
        message=f"Successfully queued {len(files)} file(s) for processing"
    )


@app.get("/status/{task_id}", response_model=TaskStatusResponse, tags=["Status"])
@timeit
async def get_task_status(task_id: str):
    """
    Get processing status for a specific task
    
    Returns:
    - Task status: PENDING, PROCESSING, COMPLETED, FAILED
    - Progress: 0-100%
    - Timestamps and error messages (if any)
    """
    # Fetch task data from Redis
    task_data = redis_client.hgetall(f"task:{task_id}")
    
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}"
        )
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_data.get("status", "UNKNOWN"),
        progress=float(task_data.get("progress", 0)),
        filename=task_data.get("filename", ""),
        created_at=task_data.get("created_at", ""),
        started_at=task_data.get("started_at") or None,
        completed_at=task_data.get("completed_at") or None,
        error=task_data.get("error") or None
    )


@app.get("/result/{task_id}", response_model=PDFExtractionResult, tags=["Results"])
@timeit
async def get_task_result(task_id: str, db: AsyncSession = Depends(get_async_db)):
    """
    Get extraction results for a completed task
    
    Returns:
    - Extracted text, tables, images, and metadata
    - 404 if task not found or not completed
    
    ⭐ NEW: Falls back to PostgreSQL if Redis result expired
    """
    # Check task status first
    task_data = redis_client.hgetall(f"task:{task_id}")
    
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}"
        )
    
    if task_data.get("status") != "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task is not completed yet. Current status: {task_data.get('status')}"
        )
    
    # Try Redis first (fast)
    result_json = redis_client.get(f"result:{task_id}")
    
    if result_json:
        # Parse and return result from Redis
        result_dict = json.loads(result_json)
        return PDFExtractionResult(**result_dict)
    
    # ⭐ NEW: Fallback to PostgreSQL if Redis expired
    db_result = await db.execute(select(Document).where(Document.id == int(task_id)))
    document = db_result.scalar_one_or_none()
    
    if not document or not document.result_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result not found for task: {task_id}. It may have expired."
        )
    
    # Reconstruct result from PostgreSQL
    return PDFExtractionResult(
        task_id=task_id,
        filename=document.filename,
        page_count=document.page_count or 0,
        text=document.result_text,
        metadata={},  # Metadata not stored separately
        extraction_time_seconds=document.extraction_time_seconds or 0.0,
        summary=document.summary
    )


@app.get("/results/stream/{task_id}", tags=["Results"])
@timeit
async def stream_task_result(task_id: str):
    """
    Stream task results in JSON lines format
    
    Generator usage: Streams large results without loading all into memory
    
    Returns:
    - StreamingResponse with application/x-ndjson content type
    """
    # Check if task exists
    task_data = redis_client.hgetall(f"task:{task_id}")
    
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}"
        )
    
    # Stream results using generator
    return StreamingResponse(
        stream_task_results(task_id),
        media_type="application/x-ndjson"
    )


@app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
@timeit
async def list_tasks(page: int = 1, page_size: int = 50):
    """
    List all tasks with pagination
    
    Generator usage: Fetches tasks in batches for memory efficiency
    
    Query params:
    - page: Page number (default: 1)
    - page_size: Items per page (default: 50, max: 100)
    """
    # Validate pagination params
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be >= 1"
        )
    
    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be between 1 and 100"
        )
    
    # Get total count
    total = redis_client.llen("all_tasks")
    
    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size - 1
    
    # Get task IDs for this page
    task_ids = redis_client.lrange("all_tasks", start_idx, end_idx)
    
    # Fetch task data
    tasks = []
    for task_id in task_ids:
        task_data = redis_client.hgetall(f"task:{task_id}")
        if task_data:
            tasks.append(TaskStatusResponse(
                task_id=task_id,
                status=task_data.get("status", "UNKNOWN"),
                progress=float(task_data.get("progress", 0)),
                filename=task_data.get("filename", ""),
                created_at=task_data.get("created_at", ""),
                started_at=task_data.get("started_at") or None,
                completed_at=task_data.get("completed_at") or None,
                error=task_data.get("error") or None
            ))
    
    return TaskListResponse(
        tasks=tasks,
        total=total,
        page=page,
        page_size=page_size
    )


@app.delete("/task/{task_id}", tags=["Tasks"])
@timeit
async def delete_task(task_id: str, db: AsyncSession = Depends(get_async_db)):
    """
    Delete a task and its results from Redis, PostgreSQL, and S3
    
    Works for:
    - Recent tasks (in Redis + PostgreSQL)
    - Old tasks (Redis expired, only in PostgreSQL)
    
    Returns:
    - Success message
    """
    # Check if task exists in Redis (for recent tasks)
    task_data = redis_client.hgetall(f"task:{task_id}")
    task_exists_in_redis = bool(task_data)
    
    # Check if task exists in PostgreSQL (for old or recent tasks)
    document = None
    try:
        doc_result = await db.execute(select(Document).where(Document.id == int(task_id)))
        document = doc_result.scalar_one_or_none()
    except ValueError:
        # task_id is not an integer, might be old UUID format
        pass
    
    # If task doesn't exist anywhere, return 404
    if not task_exists_in_redis and not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}"
        )
    
    logger.info(f"🗑️ Deleting task {task_id} (Redis: {task_exists_in_redis}, PostgreSQL: {document is not None})")
    
    # Delete file from S3 (if we have the S3 key)
    s3_key = task_data.get("s3_key") if task_data else (document.s3_key if document else None)
    if s3_key:
        try:
            aws_services.delete_file_from_s3(s3_key)
            logger.info(f"🗑️ Deleted from S3: {s3_key}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete from S3: {e}")
    else:
        logger.info(f"ℹ️ No S3 key found, skipping S3 deletion")
    
    # Delete from Redis (if exists)
    if task_exists_in_redis:
        redis_client.delete(f"task:{task_id}")
        redis_client.delete(f"result:{task_id}")
        redis_client.lrem("all_tasks", 0, task_id)
        logger.info(f"🗑️ Deleted from Redis: task:{task_id}")
    else:
        logger.info(f"ℹ️ Task not in Redis (likely expired), skipping Redis deletion")
    
    # Delete from PostgreSQL (if exists)
    if document:
        try:
            await db.delete(document)
            await db.commit()
            logger.info(f"🗑️ Deleted from PostgreSQL: document_id={task_id}")
        except Exception as e:
            logger.error(f"❌ Failed to delete from PostgreSQL: {e}")
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete from database: {str(e)}"
            )
    
    return {"message": f"Task {task_id} deleted successfully"}


# ============= POSTGRESQL CRUD ENDPOINTS =============

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_async_db)):
    """
    Create a new user
    
    ⭐ NEW: PostgreSQL endpoint
    """
    # Check if user already exists
    existing_result = await db.execute(select(User).where(User.email == user.email))
    existing_user = existing_result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Create new user
    db_user = User(email=user.email, api_key=user.api_key)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    return db_user


@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user(user_id: int, db: AsyncSession = Depends(get_async_db)):
    """
    Get a user by ID
    
    ⭐ NEW: PostgreSQL endpoint
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/documents", response_model=List[DocumentResponse], tags=["Documents"])
async def list_documents(
    user_id: int = 1,
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """
    List all documents with optional filtering
    
    ⭐ NEW: PostgreSQL endpoint (with multi-tenancy)
    Query params:
    - user_id: Filter documents by user ID (required for multi-tenancy)
    - skip: Number of records to skip (default: 0)
    - limit: Maximum records to return (default: 100, max: 1000)
    - status_filter: Filter by status (PENDING, PROCESSING, COMPLETED, FAILED)
    """
    if limit > 1000:
        limit = 1000
    
    query = select(Document).where(Document.user_id == user_id)
    
    if status_filter:
        query = query.where(Document.status == status_filter)
    
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    documents = result.scalars().all()
    return documents


@app.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(document_id: int, db: AsyncSession = Depends(get_async_db)):
    """
    Get a document by ID
    
    ⭐ NEW: PostgreSQL endpoint
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


# ============= RAG CHAT ENDPOINTS =============

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_documents(
    chat_request: ChatRequest,
    user_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Chat with your documents using RAG (Retrieval-Augmented Generation)
    
    ⭐ NEW: Phase 2 - RAG Chat Functionality
    
    This endpoint allows users to ask questions about their uploaded documents.
    It uses vector similarity search to find relevant chunks and GPT-4 to generate answers.
    
    **How it works:**
    1. Your question is converted to a vector embedding (1536 dimensions)
    2. We search for the most similar chunks in your documents using pgvector
    3. The top 5 most relevant chunks are retrieved
    4. These chunks are sent to GPT-4 along with your question
    5. GPT-4 generates an answer based only on the provided context
    
    **Parameters:**
    - `question`: Your question (required)
    - `document_id`: Optional - limit search to a specific document
    - `top_k`: Number of chunks to retrieve (default: 5, max: 20)
    - `model`: OpenAI model to use (default: gpt-4o)
    - `user_id`: Your user ID (query parameter, required for multi-tenancy)
    
    **Returns:**
    - `answer`: AI-generated answer based on your documents
    - `sources`: List of source chunks with similarity scores
    - `chunks_found`: Number of relevant chunks found
    - `usage`: Token usage statistics for cost tracking
    
    **Multi-Tenancy:**
    - You can only search documents you've uploaded
    - Results are filtered by your user_id automatically
    
    **Example Request:**
    ```json
    {
        "question": "What is the revenue mentioned in the financial report?",
        "document_id": 123,
        "top_k": 5
    }
    ```
    
    **Example Response:**
    ```json
    {
        "answer": "Based on the financial report, the total revenue for Q4 2023 is $10.5 million.",
        "sources": [
            {
                "document_id": 123,
                "filename": "Q4_Report.pdf",
                "chunk_index": 5,
                "similarity": 0.92,
                "preview": "Q4 2023 Financial Results: Total Revenue: $10.5M..."
            }
        ],
        "chunks_found": 5,
        "usage": {
            "prompt_tokens": 450,
            "completion_tokens": 25,
            "total_tokens": 475
        }
    }
    ```
    """
    try:
        # Validate user_id is provided
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id query parameter is required"
            )
        
        # Validate document_id exists and belongs to user if provided
        if chat_request.document_id:
            doc_result = await db.execute(
                select(Document).where(
                    Document.id == chat_request.document_id,
                    Document.user_id == user_id
                )
            )
            document = doc_result.scalar_one_or_none()
            
            if not document:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {chat_request.document_id} not found or you don't have access to it"
                )
        
        # Call chat service
        result = await chat_service.chat(
            db=db,
            user_id=user_id,
            question=chat_request.question,
            document_id=chat_request.document_id,
            top_k=chat_request.top_k,
            model=chat_request.model
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat service error: {str(e)}"
        )


# ============= ERROR HANDLERS =============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
