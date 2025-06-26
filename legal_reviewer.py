import os
import json
from typing import Dict, List, Any, TypedDict
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import PyPDF2
import docx
from io import BytesIO

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4"
    TEMPERATURE = 0.1

# State definition for LangGraph
class ReviewState(TypedDict):
    document_text: str
    document_type: str
    summary: str
    risk_flags: List[Dict[str, Any]]
    lawyer_questions: List[str]
    plain_english_clauses: List[Dict[str, str]]
    current_step: str
    error: str

@dataclass
class RiskFlag:
    severity: str  # "low", "medium", "high", "critical"
    category: str
    description: str
    clause_text: str
    recommendation: str

class DocumentProcessor:
    """Handles document upload and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        try:
            doc = docx.Document(BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(file_bytes: bytes) -> str:
        try:
            return file_bytes.decode('utf-8')
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")

class LegalReviewAgent:
    """Main agent that orchestrates the legal document review process"""
    
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(ReviewState)
        
        # Add nodes
        workflow.add_node("summarize", self._summarize_document)
        workflow.add_node("flag_risks", self._flag_risks)
        workflow.add_node("generate_questions", self._generate_lawyer_questions)
        workflow.add_node("rewrite_clauses", self._rewrite_in_plain_english)
        
        # Define the flow
        workflow.set_entry_point("summarize")
        workflow.add_edge("summarize", "flag_risks")
        workflow.add_edge("flag_risks", "generate_questions")
        workflow.add_edge("generate_questions", "rewrite_clauses")
        workflow.add_edge("rewrite_clauses", END)
        
        return workflow.compile()
    
    def _summarize_document(self, state: ReviewState) -> ReviewState:
        """Step 1: Summarize the document terms"""
        
        prompt = ChatPromptTemplate.from_template("""
        You are a legal expert tasked with summarizing a legal document. 
        
        Document Text:
        {document_text}
        
        Please provide a comprehensive summary that includes:
        1. Document type and purpose
        2. Key parties involved
        3. Main terms and conditions
        4. Important dates and deadlines
        5. Financial obligations
        6. Key rights and responsibilities
        
        Keep the summary clear, concise, and focused on the most important legal aspects.
        """)
        
        try:
            response = self.llm.invoke(prompt.format(document_text=state["document_text"]))
            state["summary"] = response.content
            state["current_step"] = "summarize_complete"
        except Exception as e:
            state["error"] = f"Error in summarization: {str(e)}"
        
        return state
    
    def _flag_risks(self, state: ReviewState) -> ReviewState:
        """Step 2: Flag potential risks"""
        
        prompt = ChatPromptTemplate.from_template("""
        You are a legal risk assessment expert. Analyze the following document for potential risks and red flags.
        
        Document Text:
        {document_text}
        
        For each risk you identify, provide:
        1. Severity level (critical, high, medium, low)
        2. Risk category (e.g., financial, liability, compliance, termination, etc.)
        3. Description of the risk
        4. Specific clause or text that contains the risk
        5. Recommendation for addressing the risk
        
        Format your response as a JSON array of risk objects:
        [
            {{
                "severity": "high",
                "category": "financial",
                "description": "Unlimited liability exposure",
                "clause_text": "specific clause text here",
                "recommendation": "recommend adding liability cap"
            }}
        ]
        
        Only return the JSON array, no other text.
        """)
        
        try:
            response = self.llm.invoke(prompt.format(document_text=state["document_text"]))
            risk_flags = json.loads(response.content)
            state["risk_flags"] = risk_flags
            state["current_step"] = "risk_analysis_complete"
        except Exception as e:
            state["error"] = f"Error in risk analysis: {str(e)}"
            state["risk_flags"] = []
        
        return state
    
    def _generate_lawyer_questions(self, state: ReviewState) -> ReviewState:
        """Step 3: Generate questions for a lawyer"""
        
        prompt = ChatPromptTemplate.from_template("""
        Based on the legal document and identified risks, generate important questions that should be asked to a lawyer.
        
        Document Summary:
        {summary}
        
        Identified Risks:
        {risks}
        
        Generate 8-12 specific, actionable questions that a client should ask their lawyer about this document. 
        Focus on:
        1. Clarifying ambiguous terms
        2. Understanding implications of risky clauses
        3. Negotiation opportunities
        4. Compliance requirements
        5. Exit strategies and termination procedures
        
        Format as a JSON array of strings:
        ["Question 1?", "Question 2?", ...]
        
        Only return the JSON array, no other text.
        """)
        
        try:
            risks_text = json.dumps(state["risk_flags"], indent=2)
            response = self.llm.invoke(prompt.format(
                summary=state["summary"],
                risks=risks_text
            ))
            questions = json.loads(response.content)
            state["lawyer_questions"] = questions
            state["current_step"] = "questions_generated"
        except Exception as e:
            state["error"] = f"Error generating questions: {str(e)}"
            state["lawyer_questions"] = []
        
        return state
    
    def _rewrite_in_plain_english(self, state: ReviewState) -> ReviewState:
        """Step 4: Rewrite complex clauses in plain English"""
        
        prompt = ChatPromptTemplate.from_template("""
        You are tasked with rewriting complex legal clauses in plain English to make them understandable to non-lawyers.
        
        Document Text:
        {document_text}
        
        Identify the 5-8 most complex or important clauses in this document and rewrite them in plain English.
        For each clause:
        1. Extract the original complex text
        2. Provide a plain English explanation
        3. Highlight what this means in practical terms
        
        Format as a JSON array:
        [
            {{
                "original_clause": "original legal text here",
                "plain_english": "simple explanation here",
                "practical_meaning": "what this means in real terms"
            }}
        ]
        
        Only return the JSON array, no other text.
        """)
        
        try:
            response = self.llm.invoke(prompt.format(document_text=state["document_text"]))
            clauses = json.loads(response.content)
            state["plain_english_clauses"] = clauses
            state["current_step"] = "rewriting_complete"
        except Exception as e:
            state["error"] = f"Error rewriting clauses: {str(e)}"
            state["plain_english_clauses"] = []
        
        return state
    
    def review_document(self, document_text: str, document_type: str = "contract") -> ReviewState:
        """Main method to review a document"""
        
        initial_state = ReviewState(
            document_text=document_text,
            document_type=document_type,
            summary="",
            risk_flags=[],
            lawyer_questions=[],
            plain_english_clauses=[],
            current_step="starting",
            error=""
        )
        
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            initial_state["error"] = f"Workflow error: {str(e)}"
            return initial_state

def main():
    """Streamlit UI for the Legal Document Reviewer"""
    
    st.set_page_config(
        page_title="Legal Document Reviewer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Document Reviewer")
    st.markdown("Upload a legal document for comprehensive AI-powered analysis")
    
    # Sidebar for API key input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.markdown("---")
        st.markdown("### Supported Formats")
        st.markdown("- PDF (.pdf)")
        st.markdown("- Word (.docx)")
        st.markdown("- Text (.txt)")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a legal document",
        type=['pdf', 'docx', 'txt'],
        help="Upload a contract, agreement, or other legal document for analysis"
    )
    
    if uploaded_file is not None:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar to proceed.")
            return
        
        try:
            # Extract text from uploaded file
            file_bytes = uploaded_file.read()
            
            if uploaded_file.type == "application/pdf":
                document_text = DocumentProcessor.extract_text_from_pdf(file_bytes)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                document_text = DocumentProcessor.extract_text_from_docx(file_bytes)
            elif uploaded_file.type == "text/plain":
                document_text = DocumentProcessor.extract_text_from_txt(file_bytes)
            else:
                st.error("Unsupported file type")
                return
            
            if len(document_text.strip()) == 0:
                st.error("No text could be extracted from the document")
                return
            
            # Show document preview
            with st.expander("📄 Document Preview"):
                st.text_area("Extracted Text", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)
            
            # Process document
            if st.button("🔍 Start Analysis", type="primary"):
                
                with st.spinner("Analyzing document... This may take a few minutes."):
                    try:
                        agent = LegalReviewAgent()
                        result = agent.review_document(document_text)
                        
                        if result["error"]:
                            st.error(f"Analysis failed: {result['error']}")
                            return
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Summary Section
                        st.header("📋 Document Summary")
                        st.markdown(result["summary"])
                        
                        # Risk Flags Section
                        st.header("⚠️ Risk Analysis")
                        if result["risk_flags"]:
                            for i, risk in enumerate(result["risk_flags"]):
                                severity_color = {
                                    "critical": "🔴",
                                    "high": "🟠", 
                                    "medium": "🟡",
                                    "low": "🟢"
                                }.get(risk["severity"], "⚪")
                                
                                with st.expander(f"{severity_color} {risk['category'].title()} Risk - {risk['severity'].title()} Severity"):
                                    st.markdown(f"**Description:** {risk['description']}")
                                    st.markdown(f"**Relevant Clause:** {risk['clause_text']}")
                                    st.markdown(f"**Recommendation:** {risk['recommendation']}")
                        else:
                            st.info("No significant risks identified")
                        
                        # Lawyer Questions Section  
                        st.header("❓ Questions for Your Lawyer")
                        if result["lawyer_questions"]:
                            for i, question in enumerate(result["lawyer_questions"], 1):
                                st.markdown(f"{i}. {question}")
                        else:
                            st.info("No specific questions generated")
                        
                        # Plain English Section
                        st.header("📝 Complex Clauses in Plain English")
                        if result["plain_english_clauses"]:
                            for i, clause in enumerate(result["plain_english_clauses"]):
                                with st.expander(f"Clause {i+1}"):
                                    st.markdown("**Original Legal Text:**")
                                    st.code(clause["original_clause"])
                                    st.markdown("**Plain English:**")
                                    st.markdown(clause["plain_english"])
                                    st.markdown("**Practical Meaning:**")
                                    st.info(clause["practical_meaning"])
                        else:
                            st.info("No complex clauses identified for simplification")
                        
                        # Export functionality
                        st.header("💾 Export Results")
                        
                        export_data = {
                            "document_summary": result["summary"],
                            "risk_flags": result["risk_flags"],
                            "lawyer_questions": result["lawyer_questions"],
                            "plain_english_clauses": result["plain_english_clauses"]
                        }
                        
                        st.download_button(
                            label="📥 Download Analysis Report (JSON)",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"legal_analysis_{uploaded_file.name}.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()