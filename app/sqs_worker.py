"""SQS Worker - Replaces Celery for AWS-native processing"""
import os
import sys
import time
import signal
import asyncio
import logging
from datetime import datetime
from typing import Optional
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.aws_services import aws_services
from app.dependencies import redis_client
from app.schemas_api import PDFExtractionResult, PDFMetadata

# Database imports (for PostgreSQL integration)
from app.database import SessionLocal, AsyncSessionLocal
from app.db_models import Document

# Import PDF processing functions from tasks.py
from app.tasks import (
    extract_text_from_pdf,
    extract_metadata_from_pdf
)

# Import RAG service for chunking and embedding
from app.rag_service import rag_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"⚠️  Received signal {signum}. Shutting down gracefully...")
    shutdown_requested = True


def update_task_progress(task_id: str, progress: int, status: str = "PROCESSING"):
    """Update task progress in Redis and PostgreSQL"""
    try:
        # Update Redis (real-time)
        redis_client.hset(f"task:{task_id}", "progress", progress)
        redis_client.hset(f"task:{task_id}", "status", status)
        logger.info(f"📊 Task {task_id}: {progress}% complete")
        
        # ⭐ NEW: Update PostgreSQL status if PROCESSING
        if status == "PROCESSING" and progress == 0:
            db = SessionLocal()
            try:
                document = db.query(Document).filter(Document.id == int(task_id)).first()
                if document and document.status != "PROCESSING":
                    document.status = "PROCESSING"
                    document.started_at = datetime.now()
                    db.commit()
                    logger.info(f"💾 Updated PostgreSQL: document {task_id} started processing")
            except Exception as db_error:
                logger.error(f"❌ Failed to update PostgreSQL status: {db_error}")
                db.rollback()
            finally:
                db.close()
    except Exception as e:
        logger.error(f"Error updating progress for {task_id}: {e}")


def generate_summary(text: str, prompt: str) -> Optional[str]:
    """Call OpenAI to summarize extracted text using the provided prompt."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful document summarizer. Be concise and accurate."},
                {"role": "user", "content": f"{prompt}\n\n---\n\n{text[:12000]}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"❌ Failed to generate summary: {e}")
        return None


def process_pdf_from_s3(task_id: str, s3_bucket: str, s3_key: str, filename: str, prompt: str = "") -> bool:
    """
    Process PDF file from S3
    
    Args:
        task_id: Unique task identifier
        s3_bucket: S3 bucket name
        s3_key: S3 object key
        filename: Original filename
        prompt: Optional prompt for AI summarization
    
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = datetime.now()
    temp_file_path = None
    
    try:
        # Update status to PROCESSING
        redis_client.hset(f"task:{task_id}", "status", "PROCESSING")
        redis_client.hset(f"task:{task_id}", "started_at", start_time.isoformat())
        
        logger.info(f"🚀 Starting PDF processing: {filename} (Task: {task_id})")
        
        # Step 1: Download PDF from S3
        update_task_progress(task_id, 0, "PROCESSING")
        logger.info(f"📥 Downloading from S3: {s3_key}")
        
        pdf_content = aws_services.download_file_from_s3(s3_key)
        if not pdf_content:
            raise Exception(f"Failed to download file from S3: {s3_key}")
        
        update_task_progress(task_id, 10, "PROCESSING")
        
        # Step 2: Save to temporary file (PDF libraries need file path)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name
        
        logger.info(f"💾 Saved to temp file: {temp_file_path}")
        update_task_progress(task_id, 20, "PROCESSING")
        
        # Step 3: Extract text using LlamaParse
        logger.info(f"📄 Extracting text from {filename}...")
        text, page_count = extract_text_from_pdf(temp_file_path)
        update_task_progress(task_id, 40, "PROCESSING")
        
        # Step 3.1: Generate summary if a prompt was provided
        summary = None
        if prompt and text.strip():
            logger.info(f"🤖 Generating summary for {filename} using prompt...")
            summary = generate_summary(text, prompt)
            if summary:
                logger.info(f"✅ Summary generated ({len(summary)} chars)")
            else:
                logger.warning(f"⚠️  Summary generation returned empty result")
        
        # Step 3.5: RAG Ingestion (Chunk + Embed + Store) ⭐ NEW
        logger.info(f"🤖 Starting RAG ingestion for {filename}...")
        logger.info(f"   Text length: {len(text)} characters")
        
        # Resolve user_id with a sync session first, then run the async ingest
        _user_id = None
        db_sync = SessionLocal()
        try:
            doc_row = db_sync.query(Document).filter(Document.id == int(task_id)).first()
            if doc_row:
                _user_id = doc_row.user_id
        except Exception as _e:
            logger.error(f"❌ Could not resolve user_id for RAG ingestion: {_e}")
        finally:
            db_sync.close()
        
        if _user_id is not None and text.strip():
            async def _run_rag_ingestion():
                async with AsyncSessionLocal() as async_db:
                    return await rag_service.ingest_document(
                        db=async_db,
                        document_id=int(task_id),
                        user_id=_user_id,
                        text=text,
                        metadata={
                            "filename": filename,
                            "page_count": page_count,
                            "s3_key": s3_key,
                        },
                    )
            
            try:
                rag_result = asyncio.run(_run_rag_ingestion())
                if rag_result["success"]:
                    logger.info(f"✅ RAG ingestion completed: {rag_result['chunks_created']} chunks created")
                    logger.info(f"   Duration: {rag_result['duration_seconds']}s")
                else:
                    logger.error(f"❌ RAG ingestion failed: {rag_result.get('error', 'Unknown error')}")
            except Exception as rag_error:
                logger.error(f"❌ RAG ingestion error: {rag_error}")
                # Don't fail the entire task - continue with legacy processing
        else:
            logger.warning(f"⚠️  Skipping RAG ingestion (document not found or empty text)")
        
        update_task_progress(task_id, 50, "PROCESSING")
        
        # Step 4: Extract metadata
        logger.info(f"ℹ️  Extracting metadata from {filename}...")
        metadata = extract_metadata_from_pdf(temp_file_path)
        update_task_progress(task_id, 70, "PROCESSING")
        
        # Step 5: Compile results
        logger.info(f"💾 Compiling results for {filename}...")
        
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        result = PDFExtractionResult(
            task_id=task_id,
            filename=filename,
            page_count=page_count,
            text=text,
            metadata=metadata,
            extraction_time_seconds=round(extraction_time, 2),
            summary=summary
        )
        
        # Store result in Redis with TTL (for fast access)
        result_json = result.model_dump_json()
        redis_client.setex(
            f"result:{task_id}",
            settings.task_result_ttl,
            result_json
        )
        
        # Update task status in Redis (real-time)
        redis_client.hset(f"task:{task_id}", "status", "COMPLETED")
        redis_client.hset(f"task:{task_id}", "completed_at", end_time.isoformat())
        redis_client.hset(f"task:{task_id}", "progress", 100)
        
        update_task_progress(task_id, 100, "COMPLETED")
        
        # ⭐ NEW: Save result to PostgreSQL (permanent storage)
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == int(task_id)).first()
            if document:
                document.status = "COMPLETED"
                document.result_text = text  # Save extracted text
                document.page_count = page_count
                document.extraction_time_seconds = extraction_time
                document.completed_at = end_time
                if prompt:
                    document.prompt = prompt
                if summary:
                    document.summary = summary
                db.commit()
                logger.info(f"💾 Saved result to PostgreSQL (document_id={task_id})")
            else:
                logger.warning(f"⚠️  Document {task_id} not found in PostgreSQL")
        except Exception as db_error:
            logger.error(f"❌ Failed to save to PostgreSQL: {db_error}")
            db.rollback()
        finally:
            db.close()
        
        logger.info(f"✅ Completed processing {filename} in {extraction_time:.2f}s")
        logger.info(f"   📊 Stats: {page_count} pages, {len(text)} chars")
        
        return True
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error processing {filename}: {error_msg}")
        
        # Update task status in Redis to FAILED
        redis_client.hset(f"task:{task_id}", "status", "FAILED")
        redis_client.hset(f"task:{task_id}", "error", error_msg)
        redis_client.hset(f"task:{task_id}", "completed_at", datetime.now().isoformat())
        
        # ⭐ NEW: Update PostgreSQL with error
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == int(task_id)).first()
            if document:
                document.status = "FAILED"
                document.error_message = error_msg
                document.completed_at = datetime.now()
                db.commit()
                logger.info(f"💾 Saved error to PostgreSQL (document_id={task_id})")
        except Exception as db_error:
            logger.error(f"❌ Failed to save error to PostgreSQL: {db_error}")
            db.rollback()
        finally:
            db.close()
        
        return False
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"🗑️  Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")


def worker_loop():
    """
    Main worker loop - polls SQS and processes messages
    """
    logger.info("=" * 70)
    logger.info("🚀 SQS Worker Started")
    logger.info("=" * 70)
    logger.info(f"Region: {settings.aws_region}")
    logger.info(f"Queue: {settings.sqs_queue_url}")
    logger.info(f"S3 Bucket: {settings.s3_bucket_name}")
    logger.info("=" * 70)
    logger.info("Waiting for messages... (Press Ctrl+C to stop)")
    logger.info("")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while not shutdown_requested:
        try:
            # Poll SQS for messages (long polling - waits up to 20 seconds)
            messages = aws_services.receive_messages_from_sqs(
                max_messages=1,
                wait_time_seconds=20,
                visibility_timeout=900  # 15 minutes
            )
            
            # Reset error counter on successful poll
            consecutive_errors = 0
            
            if not messages:
                # No messages available, continue polling
                continue
            
            # Process each message
            for message in messages:
                if shutdown_requested:
                    logger.info("⚠️  Shutdown requested, stopping message processing...")
                    break
                
                try:
                    # Extract message data
                    body = message['body']
                    receipt_handle = message['receipt_handle']
                    
                    task_id = body.get('task_id')
                    s3_bucket = body.get('s3_bucket')
                    s3_key = body.get('s3_key')
                    filename = body.get('filename')
                    prompt = body.get('prompt', '')
                    
                    logger.info(f"📨 Received message for task: {task_id}")
                    if prompt:
                        logger.info(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
                    
                    # Validate message data
                    if not all([task_id, s3_bucket, s3_key, filename]):
                        logger.error(f"❌ Invalid message format: {body}")
                        # Delete invalid message
                        aws_services.delete_message_from_sqs(receipt_handle)
                        continue
                    
                    # Process the PDF
                    success = process_pdf_from_s3(task_id, s3_bucket, s3_key, filename, prompt)
                    
                    if success:
                        # Delete message from SQS (acknowledge processing)
                        aws_services.delete_message_from_sqs(receipt_handle)
                        logger.info(f"✅ Message processed and deleted from SQS")
                    else:
                        # Don't delete - let it retry (will go to DLQ after 3 attempts)
                        logger.warning(f"⚠️  Processing failed, message will be retried")
                
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    # Don't delete message - let SQS retry
        
        except KeyboardInterrupt:
            logger.info("⚠️  Keyboard interrupt received...")
            break
        
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"❌ Worker error: {e}")
            logger.error(f"Consecutive errors: {consecutive_errors}/{max_consecutive_errors}")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.error("❌ Too many consecutive errors. Exiting...")
                break
            
            # Back off before retrying
            time.sleep(5)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🛑 SQS Worker Stopped")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        worker_loop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
