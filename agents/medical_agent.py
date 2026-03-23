"""
Medical Analyst Agent
Specializes in analyzing medical documents - clinical notes, lab reports, diagnoses
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# Module-level LLM instance for reuse
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000
)


def medical_agent(state: dict) -> dict:
    """
    Medical Analyst Agent that retrieves context and reasons over medical documents.
    
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
            "answer": "No relevant documents found to answer the question. Please ensure the document contains medical information.",
            "agent_used": "medical"
        }
    
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a senior medical analyst with expertise in:
    - Clinical notes and patient histories
    - Laboratory reports and diagnostic results
    - Medical diagnoses and ICD-10 coding
    - Medication prescriptions and dosing
    - Vital signs and clinical findings
    
    Use the context below to answer the question precisely.
    Focus on diagnoses, medications, lab values, and clinical findings.
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
        "agent_used": "medical"
    }


def get_medical_system_prompt() -> str:
    """
    Returns the system prompt for the medical agent.
    """
    return """You are a senior medical analyst with expertise in:
    - Clinical notes and patient histories
    - Laboratory reports and diagnostic results
    - Medical diagnoses and ICD-10 coding
    - Medication prescriptions and dosing
    - Vital signs and clinical findings
    
    Use the context provided to answer medical questions precisely.
    Focus on diagnoses, medications, lab values, and clinical findings."""


if __name__ == "__main__":
    # Test the medical agent
    from tools.retriever import build_retriever
    
    sample_text = """
    Patient Medical Record
    
    Patient: John Doe
    Date of Visit: 2024-09-15
    
    Chief Complaint: Persistent headaches and fatigue
    
    Vital Signs:
    - Blood Pressure: 145/92 mmHg
    - Heart Rate: 88 bpm
    - Temperature: 98.6°F
    - Respiratory Rate: 16/min
    
    Laboratory Results:
    - Hemoglobin: 12.5 g/dL (Low)
    - WBC: 7,500 /mcL (Normal)
    - Platelets: 250,000 /mcL (Normal)
    - Glucose: 110 mg/dL (High)
    - Cholesterol: 240 mg/dL (High)
    
    Diagnosis:
    1. Hypertension (ICD-10: I10)
    2. Type 2 Diabetes Mellitus (ICD-10: E11.9)
    3. Anemia (ICD-10: D50.9)
    
    Medications:
    1. Lisinopril 10mg - Once daily (for hypertension)
    2. Metformin 500mg - Twice daily (for diabetes)
    3. Ferrous sulfate 325mg - Once daily (for anemia)
    
    Plan:
    - Continue current medications
    - Follow up in 4 weeks
    - Recommend dietary changes
    - Increase physical activity
    """
    
    retriever = build_retriever(sample_text, domain="medical")
    
    state = {
        "question": "What diagnosis was given and what medications were prescribed?",
        "retriever": retriever,
        "domain": "medical"
    }
    
    result = medical_agent(state)
    print(f"Agent: {result['agent_used']}")
    print(f"Answer: {result['answer']}")