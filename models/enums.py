"""Data models and enums for the legal AI assistant."""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Any
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class QueryType(Enum):
    """Query classification types."""
    CLAUSE_MEANING = "understanding_clause_meaning"
    DOCUMENT_SUMMARY = "summary_of_document"
    GENERAL_LAW_QUESTION = "general_question_from_law"
    USER_DOC_QUESTION = "question_from_doc_uploaded"


class LawTopic(Enum):
    """Legal topic classifications."""
    ADMINISTRATIVE_AND_GOVERNANCE = "administrative_and_goverance_rule"
    CITIZENSHIP_AND_IMMIGRATION = "citizenship_and_immigration"
    CRIMINAL_AND_PENAL = "criminal_and_penal_law"
    EMERGENCY_AND_SPECIAL = "emergence_and_special_provisions"
    ENFORCEMENT_AND_SECURITY = "enforcement_and_public_security"
    SOCIAL_ECONOMIC_CULTURAL = "social_economic_and_cultural_and_political_acts"


@dataclass
class QueryState:
    """State management for the LangGraph workflow."""
    query: str
    query_type: Optional[QueryType] = None
    law_topic: Optional[LawTopic] = None
    documents: Optional[List[Any]] = None
    answer: Optional[str] = None
    context: Optional[str] = None
    uploaded_file: Optional[str] = None
    messages: Annotated[List, add_messages] = None