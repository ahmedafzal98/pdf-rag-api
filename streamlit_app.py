"""
Streamlit Frontend for Document Processing System with RAG
A beautiful, modern UI for PDF extraction and AI-powered chat
"""

import streamlit as st
import requests
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000"  # Your FastAPI backend
DEFAULT_USER_ID = 1  # Default user ID for multi-tenancy

# Page config
st.set_page_config(
    page_title="PDF Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-pending {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .status-processing {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .status-completed {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .status-failed {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f9fafb;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Helper Functions
def check_backend_health() -> bool:
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_pdf(file, user_id: int = DEFAULT_USER_ID, prompt: str = "") -> Optional[Dict[str, Any]]:
    """Upload PDF to backend"""
    try:
        # Backend expects "files" parameter (plural) for List[UploadFile]
        files = {"files": (file.name, file, "application/pdf")}
        # Pass user_id and optional prompt as query parameters
        params = {"user_id": user_id}
        if prompt and prompt.strip():
            params["prompt"] = prompt.strip()
        response = requests.post(f"{API_BASE_URL}/upload", files=files, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {str(e)}")
        return None


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/status/{task_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None


def get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task result from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/result/{task_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None


def get_all_tasks() -> Optional[Dict[str, Any]]:
    """Get all tasks from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None


def get_documents(user_id: int = DEFAULT_USER_ID, status_filter: str = None) -> Optional[List[Dict[str, Any]]]:
    """Get documents from PostgreSQL backend"""
    try:
        params = {"user_id": user_id}
        if status_filter:
            params["status_filter"] = status_filter
        response = requests.get(f"{API_BASE_URL}/documents", params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch documents: {str(e)}")
        return None


def chat_with_document(user_id: int, question: str, document_id: int = None, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Send chat request to RAG backend"""
    try:
        payload = {
            "question": question,
            "top_k": top_k
        }
        if document_id:
            payload["document_id"] = document_id
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            params={"user_id": user_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Chat request failed: {str(e)}")
        return None


def delete_task(task_id: str) -> bool:
    """Delete a task"""
    try:
        response = requests.delete(f"{API_BASE_URL}/task/{task_id}", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Delete failed: {str(e)}")
        return False


def format_status_badge(status: str) -> str:
    """Format status with colored badge"""
    status_upper = status.upper()
    css_class = f"status-{status.lower()}"
    return f'<span class="{css_class}">{status_upper}</span>'


def format_datetime(dt_str: str) -> str:
    """Format datetime string"""
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


# Initialize session state
if 'uploaded_task_id' not in st.session_state:
    st.session_state.uploaded_task_id = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'view_task_redirect' not in st.session_state:
    st.session_state.view_task_redirect = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_document_id' not in st.session_state:
    st.session_state.selected_document_id = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = DEFAULT_USER_ID


# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🤖 AI Document Processor & Chat</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1rem;">Upload PDFs • Extract with AI • Chat with Your Data</p>', unsafe_allow_html=True)
    
    # Check backend health
    if not check_backend_health():
        st.error("⚠️ Backend server is not running! Please start the FastAPI backend on http://localhost:8000")
        st.code("python -m uvicorn app.main:app --reload", language="bash")
        return
    
    st.success("✅ Connected to backend server")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Settings")
        
        # User ID selector
        st.session_state.user_id = st.number_input(
            "User ID",
            min_value=1,
            value=st.session_state.user_id,
            help="Your user ID for multi-tenancy"
        )
        
        st.divider()
        
        st.subheader("⚙️ Options")
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh documents",
            value=st.session_state.auto_refresh,
            help="Automatically refresh document list"
        )
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        # Backend info
        st.caption("🔗 Backend")
        st.code(API_BASE_URL, language="text")
    
        st.caption("👤 Current User")
        st.code(f"User ID: {st.session_state.user_id}", language="text")
        
        st.divider()
        
        # Quick stats
        docs = get_documents(st.session_state.user_id)
        if docs:
            completed = sum(1 for d in docs if d.get('status') == 'COMPLETED')
            st.metric("📄 Documents", len(docs))
            st.metric("✅ Ready to Chat", completed)
    
    st.divider()
    
    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload & Process", "💬 Chat with Data"])
    
    with tab1:
        show_upload_and_documents_tab()
    
    with tab2:
        show_chat_tab()


def show_upload_and_documents_tab():
    """Tab 1: Upload & Process Documents (Phase 1)"""
    st.header("📤 Upload & Process Documents")
    
    # Upload section
    st.subheader("Upload New Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents"
    )
    
    # Optional prompt for AI summarization
    prompt = st.text_area(
        "✨ Summarization Prompt (optional)",
        placeholder="e.g. Summarize the key findings of this document in 3 bullet points.",
        help="If provided, the AI will generate a summary of each PDF based on your prompt. Leave blank to skip summarization.",
        height=90,
        key="upload_prompt"
    )
    
    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.info(f"📎 **{len(uploaded_files)} file(s) selected** (Total: {total_size:.2f} MB)")
            if prompt and prompt.strip():
                st.info(f"🤖 Summarization prompt set — summary will be generated after extraction.")
        
        with col2:
            if st.button("🚀 Process All", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {i+1}/{len(uploaded_files)}: {file.name}...")
                    result = upload_pdf(file, user_id=st.session_state.user_id, prompt=prompt)
                    
                    if result:
                        st.success(f"✅ {file.name} uploaded! Task ID: {result.get('task_id')}")
                    else:
                        st.error(f"❌ Failed to upload {file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                st.balloons()
                time.sleep(1)
                st.rerun()
    
    st.divider()
    
    # Documents list section
    st.subheader("📄 Your Documents")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()
    
    # Fetch documents
    documents = get_documents(st.session_state.user_id)
    
    if not documents:
        st.info("📭 No documents yet. Upload a PDF to get started!")
        return
    
    # Filter options
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "PENDING", "PROCESSING", "COMPLETED", "FAILED"],
        key="doc_status_filter"
    )
    
    # Display documents as cards
    for doc in documents:
        doc_status = doc.get("status", "UNKNOWN")
        
        if status_filter != "All" and doc_status != status_filter:
            continue
        
        doc_id = doc.get("id")
        filename = doc.get("filename", "Unknown")
        created_at = format_datetime(doc.get("created_at", ""))
        
        # Determine if ready for chat
        ready_for_chat = doc_status == "COMPLETED"
        
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.markdown(f"**{filename}**")
                st.caption(f"ID: {doc_id} • Uploaded: {created_at}")
            
            with col2:
                if ready_for_chat:
                    st.markdown('✅ <span style="color: #059669;">Ready to Chat</span>', unsafe_allow_html=True)
                elif doc_status == "PROCESSING":
                    st.markdown('⏳ <span style="color: #2563eb;">Processing...</span>', unsafe_allow_html=True)
                elif doc_status == "PENDING":
                    st.markdown('⏸️ <span style="color: #d97706;">Pending</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'❌ <span style="color: #dc2626;">{doc_status}</span>', unsafe_allow_html=True)
            
            with col3:
                if doc.get("page_count"):
                    st.caption(f"📄 {doc.get('page_count')} pages")
                if doc.get("result_text"):
                    st.caption(f"📝 {len(doc.get('result_text', ''))} chars")
                if doc.get("summary"):
                    st.caption("🤖 Summary available")
            
            with col4:
                if ready_for_chat:
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("📄 Result", key=f"result_{doc_id}", use_container_width=True):
                            st.session_state[f"show_result_{doc_id}"] = not st.session_state.get(f"show_result_{doc_id}", False)
                    with btn_col2:
                        if st.button("💬 Chat", key=f"chat_{doc_id}", use_container_width=True):
                            st.session_state.selected_document_id = doc_id
                            st.session_state.chat_history = []
                            st.info("Switch to the 'Chat with Data' tab to start chatting!")
                else:
                    st.button("⏳ Wait", key=f"wait_{doc_id}", disabled=True, use_container_width=True)
            
            # Inline summary preview (always shown when available)
            if doc.get("summary"):
                with st.expander("🤖 AI Summary", expanded=False):
                    st.success(doc["summary"])
                    st.download_button(
                        "💾 Download Summary",
                        doc["summary"],
                        file_name=f"{filename}_summary.txt",
                        mime="text/plain",
                        key=f"dl_summary_{doc_id}"
                    )

            # Inline result panel (toggled by the Result button)
            if st.session_state.get(f"show_result_{doc_id}", False):
                show_task_result(str(doc_id))
            
            st.divider()
    
    # Auto-refresh if processing
    if st.session_state.auto_refresh:
        has_processing = any(d.get("status") in ["PENDING", "PROCESSING"] for d in documents)
        if has_processing:
            time.sleep(2)
            st.rerun()


def show_chat_tab():
    """Tab 2: Chat with Documents (Phase 2)"""
    st.header("💬 Chat with Your Documents")
    
    # Document selector
    documents = get_documents(st.session_state.user_id, status_filter="COMPLETED")
    
    if not documents:
        st.warning("📭 No documents ready for chat yet!")
        st.info("👉 Upload and process documents in the 'Upload & Process' tab first.")
        return
    
    # Dropdown for document selection
    doc_options = {
        "🌐 Search All Documents (Cross-Document Query)": None,
        **{f"{doc['filename']} (ID: {doc['id']})": doc['id'] for doc in documents}
    }
    
    selected_doc_name = st.selectbox(
        "Select a document to chat with",
        options=list(doc_options.keys()),
        index=0,
        key="doc_selector",
        help="Choose a specific document or search across all your documents"
    )
    
    st.session_state.selected_document_id = doc_options[selected_doc_name]
    
    if st.session_state.selected_document_id:
        st.success(f"📄 Chatting with: **{selected_doc_name}**")
    else:
        st.info(f"🌐 Searching across **ALL {len(documents)} documents** in your database")
    
    # Clear chat button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"**{i}. {source['filename']}** (chunk {source['chunk_index']}, similarity: {source['similarity']:.4f})")
                        st.text(source['preview'])
                        st.divider()
            
            # Show token usage if available
            if message["role"] == "assistant" and "usage" in message and message["usage"]:
                with st.expander("💰 Token Usage", expanded=False):
                    usage = message["usage"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prompt", usage.get('prompt_tokens', 0))
                    col2.metric("Completion", usage.get('completion_tokens', 0))
                    col3.metric("Total", usage.get('total_tokens', 0))
                    total_tokens = usage.get('total_tokens', 0)
                    if total_tokens > 0:
                        st.caption(f"Estimated cost: ${total_tokens * 0.00000015:.6f}")
    
    # Chat input
    user_question = st.chat_input("Ask a question about your document...")
    
    if user_question:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Show loading spinner
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Call chat API
                response = chat_with_document(
                    user_id=st.session_state.user_id,
                    question=user_question,
                    document_id=st.session_state.selected_document_id,
                    top_k=5
                )
            
            if response:
                answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                sources = response.get("sources", [])
                usage = response.get("usage", {})
                
                # Display answer
                st.markdown(answer)
                
                # Add assistant message to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "usage": usage
                })
                
                # Show sources
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.caption(f"**{i}. {source['filename']}** (chunk {source['chunk_index']}, similarity: {source['similarity']:.4f})")
                            st.text(source['preview'])
                            st.divider()
                
                # Show token usage
                if usage:
                    with st.expander("💰 Token Usage", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Prompt", usage.get('prompt_tokens', 0))
                        col2.metric("Completion", usage.get('completion_tokens', 0))
                        col3.metric("Total", usage.get('total_tokens', 0))
                        total_tokens = usage.get('total_tokens', 0)
                        if total_tokens > 0:
                            st.caption(f"Estimated cost: ${total_tokens * 0.00000015:.6f}")
                
                # Rerun to update chat display
                st.rerun()
            else:
                error_msg = "❌ Failed to get response from the chat service. Please try again."
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })


def show_upload_page():
    """Upload PDF page"""
    st.header("📤 Upload PDF Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF file(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or multiple PDF documents to extract text, tables, and images"
        )
        
        if uploaded_files:
            # Show files info
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.info(f"📎 **{len(uploaded_files)} file(s) selected** (Total: {total_size:.2f} MB)")
            
            # Show individual file details
            with st.expander("📋 File Details", expanded=True):
                for i, file in enumerate(uploaded_files, 1):
                    file_size_mb = file.size / (1024 * 1024)
                    st.text(f"{i}. {file.name} ({file_size_mb:.2f} MB)")
            
            if st.button("🚀 Start Processing All", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                uploaded_task_ids = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {i+1}/{len(uploaded_files)}: {file.name}...")
                    
                    result = upload_pdf(file, user_id=st.session_state.user_id)
                    
                    if result:
                        uploaded_task_ids.append(result.get("task_id"))
                        st.success(f"✅ {file.name} uploaded! Task ID: {result.get('task_id')}")
                    else:
                        st.error(f"❌ Failed to upload {file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                if uploaded_task_ids:
                    st.session_state.uploaded_task_id = uploaded_task_ids[-1]  # Show last uploaded
                    st.balloons()
                    st.success(f"🎉 Successfully uploaded {len(uploaded_task_ids)} file(s)!")
                    st.info("💡 Go to '📊 All Tasks' page to view all uploads.")
    
    with col2:
        st.markdown("### 📋 Processing Steps")
        st.markdown("""
        1. **Upload** - File(s) sent to S3
        2. **Queue** - Tasks added to SQS
        3. **Extract** - LlamaParse AI processing
        4. **Store** - Save to PostgreSQL
        5. **Complete** - Results available
        
        ✨ **You can upload multiple PDFs at once!**
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Maximum **20 files** per batch
        - Each file up to **50 MB**
        - View all tasks in **📊 All Tasks** page
        """)
    
    st.divider()
    
    # Show status of last uploaded task
    if st.session_state.uploaded_task_id:
        st.header("📊 Current Task Status")
        show_task_status(st.session_state.uploaded_task_id)
        
        # Auto-refresh with faster polling for active tasks
        if st.session_state.auto_refresh:
            # Get current status
            status_data = get_task_status(st.session_state.uploaded_task_id)
            current_status = status_data.get("status", "UNKNOWN") if status_data else "UNKNOWN"
            
            # Fast polling (1 second) for PENDING/PROCESSING, slower (3 seconds) for completed
            if current_status in ["PENDING", "PROCESSING"]:
                time.sleep(1)  # Poll every 1 second for active tasks
            else:
                time.sleep(3)  # Poll every 3 seconds for completed/failed tasks
            st.rerun()


def show_task_status(task_id: str):
    """Display detailed task status"""
    status_data = get_task_status(task_id)
    
    if not status_data:
        st.error(f"❌ Task {task_id} not found")
        return
    
    status = status_data.get("status", "UNKNOWN")
    progress = status_data.get("progress", 0)
    
    # Status card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Task ID", task_id)
    with col2:
        st.markdown(f"**Status:** {format_status_badge(status)}", unsafe_allow_html=True)
    with col3:
        st.metric("Progress", f"{progress}%")
    with col4:
        st.metric("File", status_data.get("filename", "N/A"))
    
    # Progress bar with detailed steps
    if status in ["PENDING", "PROCESSING"]:
        st.progress(progress / 100)
        
        # Show detailed step based on progress
        if status == "PENDING":
            st.info("⏳ **Task is pending...** Waiting for worker to pick it up from SQS queue.")
        elif progress == 0:
            st.info("🚀 **Starting processing...** Initializing worker and downloading from S3.")
        elif progress <= 10:
            st.info("📥 **Downloading from S3...** Fetching PDF file from cloud storage.")
        elif progress <= 20:
            st.info("💾 **Preparing file...** Saving to temporary location for processing.")
        elif progress <= 40:
            st.info("📄 **Extracting text...** Using LlamaParse AI to extract content (this may take a moment).")
        elif progress <= 50:
            st.info("🤖 **Generating summary...** Calling AI with your prompt to summarize the document.")
        elif progress <= 70:
            st.info("📊 **Running RAG ingestion...** Chunking and embedding document for chat.")
        elif progress <= 80:
            st.info("ℹ️ **Extracting metadata...** Reading document properties and information.")
        elif progress <= 90:
            st.info("ℹ️ **Extracting metadata...** Reading document properties and information.")
        else:
            st.info("💾 **Finalizing...** Saving results to PostgreSQL and Redis cache.")
    
    # Timestamps
    with st.expander("⏰ Timeline", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Created At")
            st.text(format_datetime(status_data.get("created_at", "")))
        with col2:
            st.caption("Started At")
            st.text(format_datetime(status_data.get("started_at", "")))
        with col3:
            st.caption("Completed At")
            st.text(format_datetime(status_data.get("completed_at", "")))
    
    # Show result if completed
    if status == "COMPLETED":
        st.success("✅ Processing completed!")
        
        if st.button("📥 View Results", type="primary"):
            show_task_result(task_id)
    
    # Show error if failed
    if status == "FAILED":
        error_msg = status_data.get("error", "Unknown error")
        st.error(f"❌ Processing failed: {error_msg}")
    
    # Actions
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Delete Task", type="secondary", use_container_width=True):
            if delete_task(task_id):
                st.success("Task deleted successfully!")
                st.session_state.uploaded_task_id = None
                time.sleep(1)
                st.rerun()


def show_task_result(task_id: str):
    """Display task result with extracted content"""
    result = get_task_result(task_id)
    
    if not result:
        st.error("❌ Result not available")
        return
    
    st.header("📥 Extraction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pages", result.get("page_count", 0))
    with col2:
        st.metric("Characters", f"{len(result.get('text', '')):,}")
    with col3:
        st.metric("Tables", "In text")
    with col4:
        extraction_time = result.get("extraction_time_seconds", 0)
        st.metric("Time", f"{extraction_time:.1f}s")
    
    # AI Summary section — shown only when a summary was generated
    summary = result.get("summary")
    if summary:
        st.divider()
        st.subheader("🤖 AI Summary")
        st.success(summary)
        st.download_button(
            "💾 Download Summary",
            summary,
            file_name=f"{result.get('filename', 'document')}_summary.txt",
            mime="text/plain",
        )
    
    st.divider()
    
    # Tabs for different content types
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text", "📊 Tables", "🖼️ Images", "📋 Metadata"])
    
    with tab1:
        st.subheader("Extracted Text")
        text_content = result.get("text", "")
        
        if text_content:
            st.text_area(
                "Full Text Content",
                text_content,
                height=400,
                label_visibility="collapsed"
            )
            
            # Download button
            st.download_button(
                "💾 Download Text",
                text_content,
                file_name=f"{result.get('filename', 'document')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("No text content extracted")
    
    with tab2:
        st.subheader("Extracted Tables")
        st.info("ℹ️ Tables are now extracted as part of the text content using LlamaParse. View the 'Text' tab to see tables in markdown format.")
    
    with tab3:
        st.subheader("Extracted Images")
        images = result.get("images", [])
        
        if images:
            cols = st.columns(3)
            for i, image in enumerate(images):
                with cols[i % 3]:
                    st.markdown(f"**Image {i+1}** (Page {image.get('page', 'N/A')})")
                    st.caption(f"Format: {image.get('format', 'N/A')}")
                    st.caption(f"Size: {image.get('width', 0)}x{image.get('height', 0)}")
                    st.divider()
        else:
            st.info("No images found in the document")
    
    with tab4:
        st.subheader("Document Metadata")
        metadata = result.get("metadata", {})
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Document Info**")
                st.text(f"Title: {metadata.get('title', 'N/A')}")
                st.text(f"Author: {metadata.get('author', 'N/A')}")
                st.text(f"Subject: {metadata.get('subject', 'N/A')}")
                st.text(f"Creator: {metadata.get('creator', 'N/A')}")
            
            with col2:
                st.markdown("**Technical Info**")
                st.text(f"Producer: {metadata.get('producer', 'N/A')}")
                st.text(f"Created: {metadata.get('creation_date', 'N/A')}")
                st.text(f"Modified: {metadata.get('modification_date', 'N/A')}")
        else:
            st.info("No metadata available")


def show_all_tasks_page():
    """Show all tasks page"""
    st.header("📊 All Tasks")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Fetch all tasks
    tasks_data = get_all_tasks()
    
    if not tasks_data:
        st.warning("No tasks found or backend error")
        return
    
    tasks = tasks_data.get("tasks", [])
    
    if not tasks:
        st.info("📭 No tasks yet. Upload a PDF to get started!")
        return
    
    # Summary metrics
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.get("status") == "COMPLETED")
    processing_tasks = sum(1 for t in tasks if t.get("status") == "PROCESSING")
    failed_tasks = sum(1 for t in tasks if t.get("status") == "FAILED")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total_tasks)
    col2.metric("Completed", completed_tasks)
    col3.metric("Processing", processing_tasks)
    col4.metric("Failed", failed_tasks)
    
    st.divider()
    
    # Filter
    filter_status = st.selectbox(
        "Filter by status",
        ["All", "PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    )
    
    # Display tasks as cards
    for task in tasks:
        task_status = task.get("status", "UNKNOWN")
        
        # Apply filter
        if filter_status != "All" and task_status != filter_status:
            continue
        
        task_id = task.get("task_id", "N/A")
        
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.markdown(f"**{task.get('filename', 'Unknown')}**")
                st.caption(f"Task ID: {task_id}")
            
            with col2:
                st.markdown(format_status_badge(task_status), unsafe_allow_html=True)
            
            with col3:
                progress = task.get("progress", 0)
                st.caption(f"Progress: {progress}%")
            
            with col4:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("👁️ View", key=f"view_{task_id}", use_container_width=True):
                        # Set the task ID and force navigation to upload page
                        st.session_state.uploaded_task_id = task_id
                        st.session_state.view_task_redirect = True
                        st.rerun()
                
                with col_b:
                    if st.button("🗑️", key=f"delete_{task_id}", use_container_width=True):
                        if delete_task(task_id):
                            st.success("Deleted!")
                            time.sleep(0.5)
                            st.rerun()
            
            st.divider()
    
    # Auto-refresh with adaptive polling
    if st.session_state.auto_refresh:
        # Check if any tasks are actively processing
        has_active_tasks = any(
            t.get("status") in ["PENDING", "PROCESSING"] 
            for t in tasks 
            if filter_status == "All" or t.get("status") == filter_status
        )
        
        # Fast polling (1 second) if active tasks, slower (3 seconds) otherwise
        if has_active_tasks:
            time.sleep(1)
        else:
            time.sleep(3)
        st.rerun()


def show_about_page():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 🚀 PDF Document Processing System
    
    This is a **production-grade hybrid architecture** for intelligent PDF processing.
    
    #### 🏗️ Architecture Components:
    
    1. **Frontend (Streamlit)**
       - Beautiful, modern UI
       - Real-time progress tracking
       - File upload and result viewing
    
    2. **Backend (FastAPI)**
       - RESTful API endpoints
       - Async task processing
       - Error handling & validation
    
    3. **Database Layer**
       - **PostgreSQL**: Persistent storage for documents, users, and metadata
       - **Redis**: Real-time progress tracking and caching
    
    4. **Cloud Services**
       - **AWS S3**: PDF file storage
       - **AWS SQS**: Asynchronous job queue
       - **LlamaCloud**: AI-powered PDF extraction
    
    5. **Worker (Background)**
       - Polls SQS for jobs
       - Extracts text using LlamaParse (primary) or PyPDF2/pdfplumber (fallback)
       - Saves results to PostgreSQL and Redis
    
    #### 🔄 Processing Flow:
    
    ```
    1. User uploads PDF → FastAPI receives file
    2. File saved to S3 → Document record created in PostgreSQL (status=PENDING)
    3. Task sent to SQS → Worker picks up job
    4. Worker updates status to PROCESSING → Real-time progress in Redis
    5. LlamaParse extracts content → AI-powered text, tables, images
    6. Results saved to PostgreSQL (permanent) → Redis cache for fast access
    7. Status updated to COMPLETED → User can view results
    ```
    
    #### 💡 Key Features:
    
    - ✅ **AI-Powered Extraction**: Uses LlamaParse for superior accuracy
    - ✅ **Hybrid Storage**: PostgreSQL for persistence, Redis for speed
    - ✅ **Fault Tolerant**: Automatic fallback to PyPDF2 if LlamaParse fails
    - ✅ **Real-Time Progress**: Live updates during processing
    - ✅ **Scalable**: SQS queue handles high load
    - ✅ **Production Ready**: Error handling, logging, monitoring
    
    #### 🛠️ Tech Stack:
    
    - **Backend**: FastAPI, SQLAlchemy, Pydantic
    - **Frontend**: Streamlit
    - **Databases**: PostgreSQL, Redis
    - **Cloud**: AWS (S3, SQS)
    - **AI**: LlamaCloud API
    - **Containerization**: Docker Compose
    
    ---
    
    Made with ❤️ for intelligent document processing
    """)
    
    st.divider()
    
    st.subheader("🔧 Quick Links")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Backend API**")
        st.markdown(f"[API Docs]({API_BASE_URL}/docs)")
        st.markdown(f"[Health Check]({API_BASE_URL}/health)")
    
    with col2:
        st.markdown("**Database**")
        st.code("PostgreSQL: localhost:5433")
        st.code("Redis: localhost:6379")
    
    with col3:
        st.markdown("**Documentation**")
        st.markdown("📄 MIGRATION_GUIDE.md")
        st.markdown("🚀 QUICK_START.md")


if __name__ == "__main__":
    main()
