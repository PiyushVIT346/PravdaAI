"""Main legal AI assistant service."""
import logging
from typing import Dict, Any
import google.generativeai
from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph, END
from config.settings import Config
from models.enums import QueryType, QueryState
from services.document_service import DocumentService
from services.classifier import QueryClassifier
from services.query_handlers import QueryHandlers

logger = logging.getLogger(__name__)


class LegalAIAssistant:
    """Main class for legal AI assistant functionality."""
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the legal AI assistant."""
        api_key = gemini_api_key or Config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Configure Google AI
        google.generativeai.configure(api_key=api_key)
        
        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model=Config.LLM_MODEL,
            api_key=api_key,
            temperature=Config.LLM_TEMPERATURE
        )
        
        # Initialize services
        self.doc_service = DocumentService()
        self.classifier = QueryClassifier(self.llm)
        self.handlers = QueryHandlers(self.llm, self.doc_service)
        
        # Load documents
        self.doc_service.load_clause_pdf()
        self.doc_service.load_law_pdfs()
        
        # Create workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for query processing."""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classifier.classify_intent)
        workflow.add_node("classify_topic", self.classifier.classify_topic)
        workflow.add_node("handle_clause", self.handlers.handle_clause_meaning)
        workflow.add_node("handle_summary", self.handlers.handle_document_summary)
        workflow.add_node("handle_general", self.handlers.handle_general_law_question)
        workflow.add_node("handle_user_doc", self.handlers.handle_user_doc_question)
        
        workflow.set_entry_point("classify_intent")
        
        def route_by_intent(state: QueryState) -> str:
            """Route based on query intent."""
            if state.query_type == QueryType.CLAUSE_MEANING:
                return "handle_clause"
            elif state.query_type == QueryType.DOCUMENT_SUMMARY:
                return "handle_summary"
            elif state.query_type == QueryType.GENERAL_LAW_QUESTION:
                return "classify_topic"
            elif state.query_type == QueryType.USER_DOC_QUESTION:
                return "handle_user_doc"
            else:
                return "handle_general"
        
        def route_after_topic(state: QueryState) -> str:
            """Route after topic classification."""
            return "handle_general"
        
        workflow.add_conditional_edges(
            "classify_intent",
            route_by_intent,
            {
                "handle_clause": "handle_clause",
                "handle_summary": "handle_summary",
                "classify_topic": "classify_topic",
                "handle_user_doc": "handle_user_doc",
                "handle_general": "handle_general"
            }
        )
        
        workflow.add_conditional_edges(
            "classify_topic",
            route_after_topic,
            {"handle_general": "handle_general"}
        )
        
        workflow.add_edge("handle_clause", END)
        workflow.add_edge("handle_summary", END)
        workflow.add_edge("handle_general", END)
        workflow.add_edge("handle_user_doc", END)
        
        return workflow.compile()
    
    def process_document(self, file_path: str) -> bool:
        """Process and index an uploaded document."""
        return self.doc_service.process_user_document(file_path)
    
    def query(self, user_query: str, uploaded_file: str = None) -> Dict[str, Any]:
        """Main query processing method."""
        try:
            state = QueryState(query=user_query, uploaded_file=uploaded_file)
            result = self.workflow.invoke(state)
            
            query_type = result.get("query_type")
            law_topic = result.get("law_topic")
            answer = result.get("answer")
            
            return {
                "query": user_query,
                "query_type": getattr(query_type, "value", query_type),
                "law_topic": getattr(law_topic, "value", law_topic),
                "answer": answer,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": user_query,
                "answer": "An error occurred while processing your query. Please try again.",
                "success": False,
                "error": str(e)
            }