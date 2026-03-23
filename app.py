"""
FastAPI Application
Main entry point for the Multi-Agent Document Analyst API
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
from pydantic import BaseModel

from tools.parser import parse_pdf
from tools.retriever import build_retriever
from graph import build_graph
from agents.supervisor import detect_domain


# Validate API key at startup
def _validate_api_key():
    """Validate that OPENAI_API_KEY is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")
    if api_key == "your-openai-api-key-here":
        raise ValueError("Please set a valid OPENAI_API_KEY in your .env file.")


# Define the app
app = FastAPI(
    title="Multi-Agent Document Analyst",
    description="""
A LangGraph-based multi-agent system where specialized agents collaborate 
to analyze both financial and medical documents.
    """,
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Validate API key on startup."""
    _validate_api_key()


# Define request model
class AnalyzeRequest(BaseModel):
    question: str
    domain: Optional[str] = "auto"


# Define response model
class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    agent_used: str
    domain_detected: Optional[str] = None


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Agent Document Analyst",
        "version": "1.0.0",
        "description": "LangGraph-based multi-agent system for financial and medical document analysis",
        "endpoints": {
            "/": "API information",
            "/analyze": "Analyze a document with a question (POST)",
            "/health": "Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...),
    domain: Optional[str] = Form(None)
):
    """
    Analyze a document and answer a question.
    
    Args:
        file: PDF file to analyze
        question: Question to answer about the document
        domain: Optional domain hint ("financial" or "medical")
        
    Returns:
        JSON response with the answer and agent used
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=".pdf"
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        # Parse the PDF
        text = parse_pdf(tmp_path)
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF"
            )
        
        # Detect domain from question before building retriever
        detected_domain = domain if domain and domain != "auto" else detect_domain(question)
        
        # Build retriever with detected domain
        retriever = build_retriever(text, domain=detected_domain)
        
        # Build the graph
        graph = build_graph()
        
        # Run through LangGraph
        result = graph.invoke({
            "question": question,
            "retriever": retriever,
            "domain": detected_domain,
            "answer": "",
            "agent_used": ""
        })
        
        return AnalyzeResponse(
            question=question,
            answer=result["answer"],
            agent_used=result["agent_used"],
            domain_detected=detected_domain
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/analyze-text")
async def analyze_text(
    text: str = Form(...),
    question: str = Form(...),
    domain: Optional[str] = Form(None)
):
    """
    Analyze text and answer a question (without PDF upload).
    
    Args:
        text: Text to analyze
        question: Question to answer about the text
        domain: Optional domain hint ("financial" or "medical")
        
    Returns:
        JSON response with the answer and agent used
    """
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    # Detect domain from question before building retriever
    detected_domain = domain if domain and domain != "auto" else detect_domain(question)
    
    # Build retriever with detected domain
    retriever = build_retriever(text, domain=detected_domain)
    
    # Build the graph
    graph = build_graph()
    
    # Run through LangGraph
    result = graph.invoke({
        "question": question,
        "retriever": retriever,
        "domain": detected_domain,
        "answer": "",
        "agent_used": ""
    })
    
    return AnalyzeResponse(
        question=question,
        answer=result["answer"],
        agent_used=result["agent_used"],
        domain_detected=detected_domain
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)