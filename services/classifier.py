"""Query classification service."""
import logging
from langchain_google_genai import GoogleGenerativeAI
from models.enums import QueryType, LawTopic, QueryState

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Handles query classification logic."""
    
    def __init__(self, llm: GoogleGenerativeAI):
        """Initialize classifier with LLM."""
        self.llm = llm
    
    def classify_intent(self, state: QueryState) -> QueryState:
        """Classify the user query into one of the predefined intents."""
        classification_prompt = """
        Classify the following query into one of these categories:
        1. understanding_clause_meaning - User wants to understand what is meaning of any difficult legal term or word
        2. summary_of_document - User wants a summary of a document
        3. general_question_from_law - User has a question related to clause or any section of law 
        4. question_from_doc_uploaded - User has a question about a document they uploaded
        
        Query: {query}
        
        Respond with only the category name.
        """
        
        try:
            result = self.llm.invoke(classification_prompt.format(query=state.query))
            query_type = QueryType(result.strip().lower())
            state.query_type = query_type
            logger.info(f"Query classified as: {query_type.value}")
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            state.query_type = QueryType.GENERAL_LAW_QUESTION
        
        return state
    
    def classify_topic(self, state: QueryState) -> QueryState:
        """Classify the legal topic for general law questions."""
        if state.query_type != QueryType.GENERAL_LAW_QUESTION:
            return state
        
        topic_prompt = """
        Classify the following legal query into one of these topics:

        administrative_and_goverance_rule - Laws governing government agencies and operations
        citizenship_and_immigration - Legal matters related to citizenship status
        criminal_and_penal_law - Offenses against the state or society
        emergence_and_special_provisions - Exceptional or crisis situations
        enforcement_and_public_security - Public order and safety
        social_economic_and_cultural_and_political_acts - Social, economic, cultural, and political rights
        
        Choose among these names only.

        Query: {query}
        
        Respond with only the topic name.
        """
        
        try:
            result = self.llm.invoke(topic_prompt.format(query=state.query))
            law_topic = LawTopic(result.strip().lower())
            state.law_topic = law_topic
            logger.info(f"Law topic classified as: {law_topic.value}")
        except Exception as e:
            logger.error(f"Error in law topic classification: {e}")
            state.law_topic = None
        
        return state