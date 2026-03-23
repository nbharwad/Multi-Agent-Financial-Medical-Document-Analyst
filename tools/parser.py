"""
PDF Parser Tool
Uses PyMuPDF (fitz) to extract text from PDF documents
"""

import fitz  # PyMuPDF


def parse_pdf(filepath: str) -> str:
    """
    Parse a PDF file and extract text from all pages.
    
    Args:
        filepath: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    doc = fitz.open(filepath)
    pages_text = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages_text.append(text)
    
    doc.close()
    return "\n".join(pages_text)


def parse_pdf_page(filepath: str, page_num: int) -> str:
    """
    Parse a specific page from a PDF file.
    
    Args:
        filepath: Path to the PDF file
        page_num: Page number (0-indexed)
        
    Returns:
        Extracted text from the specified page
    """
    doc = fitz.open(filepath)
    
    if page_num >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} does not exist in the document")
    
    text = doc[page_num].get_text()
    doc.close()
    return text


if __name__ == "__main__":
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        text = parse_pdf(sys.argv[1])
        print(f"Extracted {len(text)} characters")
        print(text[:500])
    else:
        print("Usage: python parser.py <pdf_file>")