"""Chat Service - Handles RAG query and retrieval for chat functionality"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from openai import AsyncOpenAI

from app.config import settings
from app.db_models import DocumentChunk, Document

# Configure logging
logger = logging.getLogger(__name__)


class ChatService:
    """
    Service for RAG-based chat functionality
    
    Handles:
    - Question embedding
    - Vector similarity search in PostgreSQL
    - Context building from retrieved chunks
    - LLM answer generation
    """
    
    def __init__(self):
        """Initialize chat service with OpenAI client"""
        if not settings.openai_api_key:
            logger.warning("⚠️  OpenAI API key not configured - chat will not work")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("✅ Chat Service initialized with AsyncOpenAI client")
    
    async def embed_question(self, question: str) -> List[float]:
        """
        Generate embedding for user's question using OpenAI
        
        Args:
            question: User's question text
        
        Returns:
            List of 1536 floats representing the question embedding
        
        Raises:
            ValueError: If OpenAI client is not configured
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured. Add OPENAI_API_KEY to .env file")
        
        logger.info(f"🔢 Embedding question: '{question[:50]}...'")
        
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            )
            
            embedding = response.data[0].embedding
            logger.info(f"✅ Generated question embedding ({len(embedding)} dimensions)")
            return embedding
        
        except Exception as e:
            logger.error(f"❌ Error generating question embedding: {e}")
            raise
    
    async def search_similar_chunks(
        self,
        db: AsyncSession,
        question_embedding: List[float],
        user_id: int,
        document_id: Optional[int] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks most similar to the question using pgvector (async).

        Args:
            db: SQLAlchemy async database session
            question_embedding: Embedded question vector
            user_id: User ID for multi-tenancy filtering
            document_id: Optional - limit search to specific document
            top_k: Number of chunks to retrieve (default: 5)

        Returns:
            List of dictionaries with chunk data and similarity scores
        """
        import time
        start_time = time.time()
        
        logger.info(f"🔍 Searching for top {top_k} similar chunks...")
        logger.info(f"   User ID: {user_id}")
        if document_id:
            logger.info(f"   Limited to document ID: {document_id}")
        
        try:
            # Convert embedding to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, question_embedding)) + "]"
            
            # Build SQL query with pgvector cosine distance operator (<=>)
            # Note: Use string formatting for the vector literal to avoid binding issues
            if document_id:
                query = text(f"""
                    SELECT 
                        dc.id,
                        dc.document_id,
                        dc.chunk_index,
                        dc.text_content,
                        d.filename,
                        1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.user_id = :user_id 
                      AND dc.document_id = :document_id
                    ORDER BY dc.embedding <=> '{embedding_str}'::vector
                    LIMIT :top_k
                """)
                result = await db.execute(
                    query,
                    {"user_id": user_id, "document_id": document_id, "top_k": top_k}
                )
            else:
                query = text(f"""
                    SELECT 
                        dc.id,
                        dc.document_id,
                        dc.chunk_index,
                        dc.text_content,
                        d.filename,
                        1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.user_id = :user_id
                    ORDER BY dc.embedding <=> '{embedding_str}'::vector
                    LIMIT :top_k
                """)
                result = await db.execute(
                    query,
                    {"user_id": user_id, "top_k": top_k}
                )
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "chunk_index": row[2],
                    "text": row[3],
                    "filename": row[4],
                    "similarity": float(row[5])
                })
            
            search_time_ms = (time.time() - start_time) * 1000
            logger.info(f"✅ Found {len(chunks)} similar chunks in {search_time_ms:.0f}ms ⚡")
            for i, chunk in enumerate(chunks):
                logger.info(f"   {i+1}. {chunk['filename']} (chunk {chunk['chunk_index']}) - similarity: {chunk['similarity']:.4f}")
            
            return chunks
        
        except Exception as e:
            logger.error(f"❌ Error searching chunks: {e}")
            raise
    
    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: List of chunk dictionaries from search_similar_chunks()
        
        Returns:
            Formatted context string for LLM prompt
        """
        if not chunks:
            return "No relevant information found."
        
        logger.info(f"📝 Building context from {len(chunks)} chunks...")
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Format each chunk with source information
            chunk_text = f"[Source: {chunk['filename']}, Chunk {chunk['chunk_index']}]\n{chunk['text']}"
            context_parts.append(chunk_text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"✅ Context built: {len(context)} characters")
        return context
    
    async def generate_answer(
        self,
        question: str,
        context: str,
        model: str = "gpt-4o"
        ) -> Dict[str, Any]:
        """
        Generate answer using GPT-4 based on retrieved context
        
        Args:
            question: User's question
            context: Retrieved context from documents
            model: OpenAI model to use (default: gpt-4o for accuracy)
        
        Returns:
            Dictionary with answer and usage statistics
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        logger.info(f"🤖 Generating answer using {model}...")
        
        try:
            # Build prompt
            system_prompt = (
                "You are a helpful assistant that answers questions based on provided context. "
                "Answer the user's question based ONLY on the information in the context. "
                "If the context doesn't contain enough information to answer the question, "
                "say 'I don't have enough information to answer that question based on the provided documents.'"
            )
            
            user_prompt = f"""Context from documents:

{context}

---

Question: {question}

Please provide a clear and concise answer based on the context above."""
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Extract usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"✅ Generated answer ({len(answer)} chars)")
            logger.info(f"   Tokens used: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
            
            return {
                "answer": answer,
                "usage": usage,
                "model": model
            }
        
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            raise
    
    async def chat(
        self,
        db: AsyncSession,
        user_id: int,
        question: str,
        document_id: Optional[int] = None,
        top_k: int = 5,
        model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        Complete RAG chat pipeline: embed → search → retrieve → generate
        
        This is the main function to call from the API endpoint.
        
        Args:
            db: SQLAlchemy database session
            user_id: User ID for multi-tenancy
            question: User's question
            document_id: Optional - limit search to specific document
            top_k: Number of chunks to retrieve (default: 5)
            model: OpenAI model to use (default: gpt-4o)
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"💬 Starting chat session")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Question: '{question}'")
        
        try:
            # Step 1: Embed the question
            question_embedding = await self.embed_question(question)
            
            # Step 2: Search for similar chunks
            chunks = await self.search_similar_chunks(
                db=db,
                question_embedding=question_embedding,
                user_id=user_id,
                document_id=document_id,
                top_k=top_k
            )
            
            if not chunks:
                logger.warning("⚠️  No chunks found - returning empty answer")
                return {
                    "answer": "I couldn't find any relevant information in your documents to answer this question.",
                    "sources": [],
                    "chunks_found": 0,
                    "usage": None,
                    "error": "No relevant chunks found"
                }
            
            # Step 3: Build context
            context = self.build_context(chunks)
            
            # Step 4: Generate answer
            result = await self.generate_answer(
                question=question,
                context=context,
                model=model
            )
            
            # Step 5: Prepare response with sources
            sources = []
            for chunk in chunks:
                sources.append({
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "similarity": chunk["similarity"],
                    "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                })
            
            logger.info(f"✅ Chat completed successfully")
            
            return {
                "answer": result["answer"],
                "sources": sources,
                "chunks_found": len(chunks),
                "model": result["model"],
                "usage": result["usage"]
            }
        
        except Exception as e:
            logger.error(f"❌ Chat failed: {e}")
            return {
                "answer": "An error occurred while processing your question.",
                "sources": [],
                "chunks_found": 0,
                "usage": None,
                "error": str(e)
            }


# Global chat service instance
chat_service = ChatService()
