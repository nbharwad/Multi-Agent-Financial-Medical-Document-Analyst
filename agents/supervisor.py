"""
Supervisor Agent
Routes questions to the appropriate specialist agent based on domain detection
"""

from typing import Literal


# Financial keywords for domain detection
FINANCIAL_KEYWORDS = [
    "revenue", "profit", "loss", "balance sheet", "earnings",
    "debt", "equity", "cash flow", "risk", "stock", "market",
    "investment", "dividend", "quarterly", "annual", "financial",
    "income", "expense", "asset", "liability", "portfolio",
    "margin", "ratio", "valuation", "share", "trading"
]

# Medical keywords for domain detection
MEDICAL_KEYWORDS = [
    "diagnosis", "medication", "prescription", "doctor", "patient",
    "symptom", "treatment", "lab", "test", "result", "vital",
    "blood pressure", "heart rate", "diabetes", "hypertension",
    "clinical", "hospital", "health", "medical", "disease",
    "therapy", "surgery", "appointment", "clinic", "report"
]


def supervisor(state: dict) -> Literal["financial", "medical"]:
    """
    Supervisor agent that routes questions to the appropriate specialist.
    
    Args:
        state: A dictionary containing:
            - question: The user's question
            
    Returns:
        A string indicating which agent to route to: "financial" or "medical"
    """
    question = state["question"].lower()
    
    # Check for financial keywords
    financial_score = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in question)
    
    # Check for medical keywords
    medical_score = sum(1 for keyword in MEDICAL_KEYWORDS if keyword in question)
    
    # Determine routing based on scores
    if financial_score > medical_score:
        return "financial"
    elif medical_score > financial_score:
        return "medical"
    else:
        # Default to financial if no clear match, or could use LLM for smarter routing
        return "financial"


def detect_domain(question: str) -> str:
    """
    Detect the domain of a question.
    
    Args:
        question: The user's question
        
    Returns:
        A string indicating the detected domain: "financial" or "medical"
    """
    question_lower = question.lower()
    
    financial_score = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in question_lower)
    medical_score = sum(1 for keyword in MEDICAL_KEYWORDS if keyword in question_lower)
    
    if financial_score > medical_score:
        return "financial"
    elif medical_score > financial_score:
        return "medical"
    else:
        # Default to financial if no clear match
        return "financial"


if __name__ == "__main__":
    # Test the supervisor
    test_questions = [
        "What was the net revenue in Q3 and how does it compare to Q2?",
        "What are the top 3 risk factors mentioned?",
        "What diagnosis was given and what medications were prescribed?",
        "What were the abnormal lab values and what do they indicate?",
        "Should I buy this stock?",
        "What's my prognosis?"
    ]
    
    for q in test_questions:
        domain = detect_domain(q)
        print(f"Question: {q}")
        print(f"Detected domain: {domain}")
        print()