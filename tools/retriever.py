"""
FAISS Retrieval Tool
Builds vector store and retriever for document chunks
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def build_retriever(text: str, domain: str = "general", chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Build a FAISS retriever from text.
    
    Args:
        text: Input text to chunk and index
        domain: Domain for metadata (e.g., 'financial' or 'medical')
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        A retriever object
    """
    # Create text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Create document chunks with metadata
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"domain": domain}]
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Build FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Return as retriever
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def build_retriever_from_file(filepath: str, domain: str = "general"):
    """
    Build a retriever directly from a PDF file.
    
    Args:
        filepath: Path to the PDF file
        domain: Domain for metadata
        
    Returns:
        A retriever object
    """
    from tools.parser import parse_pdf
    
    text = parse_pdf(filepath)
    return build_retriever(text, domain)


if __name__ == "__main__":
    # Test the retriever
    sample_text = """
    Financial Report Q3 2024
    Revenue: $10.5 million
    Net Profit: $2.3 million
    Operating Expenses: $5.2 million
    Cash Flow: $3.1 million
    
    Risk Factors:
    1. Market volatility
    2. Currency fluctuations
    3. Regulatory changes
    """
    
    retriever = build_retriever(sample_text, domain="financial")
    docs = retriever.invoke("What was the net profit?")
    print(f"Retrieved {len(docs)} documents")
    for doc in docs:
        print(f"- {doc.page_content[:100]}...")