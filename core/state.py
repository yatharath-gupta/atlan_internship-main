# core/state.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class RetrievedChunk:
    """Represents a single retrieved document chunk from the knowledge base."""
    content: str
    url: str
    title: str
    chunk_id: str
    chunk_index: int
    similarity_score: float
    token_count: int
    metadata: Dict

@dataclass
class RAGResponse:
    """Represents the final, complete output of the RAG system for a given query."""
    query: str
    answer: str
    sources: List[RetrievedChunk]
    confidence_score: float
    response_metadata: Dict

@dataclass
class TicketClassification:
    """Holds the classification results from the Triage Agent."""
    topic: str
    sentiment: str
    priority: str
    reasoning: str = "No reasoning provided."
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    

@dataclass
class TicketState:
    """

    Manages the state of a ticket as it moves through the AI agent pipeline.
    This object is created by the orchestrator and passed between agents.
    """
    original_query: str
    classification: Optional[TicketClassification] = None
    sub_questions: List[str] = field(default_factory=list)
    retrieved_context: Dict[str, List[RetrievedChunk]] = field(default_factory=dict)
    draft_response: Optional[str] = None
    final_response: Optional[RAGResponse] = None
    error_stage: Optional[str] = None
    review_feedback: Optional[str] = None