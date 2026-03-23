"""
Financial Analyst Agent
Specializes in analyzing financial documents - balance sheets, earnings, risk reports
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# Module-level LLM instance for reuse
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000
)


def financial_agent(state: dict) -> dict:
    """
    Financial Analyst Agent that retrieves context and reasons over financial documents.
    
    Args:
        state: A dictionary containing:
            - question: The user's question
            - retriever: The FAISS retriever for the document
            - domain: The domain type
            
    Returns:
        A dictionary containing:
            - answer: The agent's response
            - agent_used: The name of the agent used
    """
    retriever = state["retriever"]
    question = state["question"]
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Handle empty document context
    if not docs:
        return {
            "answer": "No relevant documents found to answer the question. Please ensure the document contains financial information.",
            "agent_used": "financial"
        }
    
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a senior financial analyst with expertise in:
    - Balance sheet analysis
    - Earnings reports and revenue analysis
    - Cash flow statements
    - Financial risk assessment
    - Key financial ratios and metrics
    
    Use the context below to answer the question precisely.
    Focus on numbers, ratios, risks, and financial indicators.
    If the context doesn't contain enough information to answer, state that clearly.

    Context:
    {context}

    Question: {question}

    Provide a detailed, professional response.
    """)
    
    # Initialize the LLM (using module-level instance for efficiency)
    llm = _llm
    
    # Create and invoke the chain
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return {
        "answer": response.content,
        "agent_used": "financial"
    }


def get_financial_system_prompt() -> str:
    """
    Returns the system prompt for the financial agent.
    """
    return """You are a senior financial analyst with expertise in:
    - Balance sheet analysis
    - Earnings reports and revenue analysis
    - Cash flow statements
    - Financial risk assessment
    - Key financial ratios and metrics
    
    Use the context provided to answer financial questions precisely.
    Focus on numbers, ratios, risks, and financial indicators."""


if __name__ == "__main__":
    # Test the financial agent
    from tools.retriever import build_retriever
    
    sample_text = """
    Financial Report Q3 2024
    
    Revenue: $10.5 million
    Net Profit: $2.3 million
    Operating Expenses: $5.2 million
    Cash Flow: $3.1 million
    Total Assets: $45 million
    Total Liabilities: $28 million
    Shareholder Equity: $17 million
    
    Key Financial Ratios:
    - Profit Margin: 21.9%
    - Current Ratio: 1.5
    - Debt to Equity Ratio: 1.65
    
    Risk Factors:
    1. Market volatility - High
    2. Currency fluctuations - Medium
    3. Regulatory changes - Medium
    4. Interest rate risk - High
    """
    
    retriever = build_retriever(sample_text, domain="financial")
    
    state = {
        "question": "What was the net profit and what are the key risk factors?",
        "retriever": retriever,
        "domain": "financial"
    }
    
    result = financial_agent(state)
    print(f"Agent: {result['agent_used']}")
    print(f"Answer: {result['answer']}")