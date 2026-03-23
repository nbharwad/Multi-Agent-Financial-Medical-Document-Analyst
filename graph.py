"""
LangGraph Graph Definition
Defines the multi-agent workflow with supervisor routing
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.financial_agent import financial_agent
from agents.medical_agent import medical_agent
from agents.supervisor import supervisor


class AgentState(TypedDict):
    """State schema for the LangGraph workflow."""
    question: str
    domain: str
    retriever: object
    answer: str
    agent_used: str


def build_graph():
    """
    Build and compile the LangGraph workflow.
    
    The graph consists of:
    - A supervisor node that routes to the appropriate specialist
    - Financial agent node
    - Medical agent node
    
    Returns:
        A compiled LangGraph state machine
    """
    # Create the state graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("financial_agent", financial_agent)
    graph.add_node("medical_agent", medical_agent)
    
    # Set conditional entry point using supervisor
    graph.set_conditional_entry_point(
        supervisor,
        {
            "financial": "financial_agent",
            "medical": "medical_agent"
        }
    )
    
    # Add edges to end
    graph.add_edge("financial_agent", END)
    graph.add_edge("medical_agent", END)
    
    # Compile the graph
    return graph.compile()


def run_analysis(question: str, retriever, domain: str = "auto") -> dict:
    """
    Run the analysis through the LangGraph workflow.
    
    Args:
        question: The user's question
        retriever: The FAISS retriever for the document
        domain: The domain type (auto-detected if not specified)
        
    Returns:
        A dictionary containing the answer and agent used
    """
    # Build the graph
    compiled_graph = build_graph()
    
    # Initial state
    initial_state = {
        "question": question,
        "domain": domain,
        "retriever": retriever,
        "answer": "",
        "agent_used": ""
    }
    
    # Run the graph
    result = compiled_graph.invoke(initial_state)
    
    return result


if __name__ == "__main__":
    # Test the graph
    from tools.retriever import build_retriever
    
    # Test with financial document
    financial_text = """
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
    
    retriever = build_retriever(financial_text, domain="financial")
    
    question = "What was the net profit and what are the key risk factors?"
    result = run_analysis(question, retriever)
    
    print(f"Question: {question}")
    print(f"Agent used: {result['agent_used']}")
    print(f"Answer: {result['answer']}")