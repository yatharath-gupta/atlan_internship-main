# core/rag_system.py

import logging
import json
import time
from typing import List, Dict, Callable, Optional, Generator

# Third-party imports
import google.generativeai as genai
# try:
#     __import__('pysqlite3')
#     import sys
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#     import chromadb
#     CHROMADB_AVAILABLE = True
# except (ImportError, RuntimeError):
#     chromadb = None
#     CHROMADB_AVAILABLE = True
# Local application imports
from .state import TicketState, RAGResponse, TicketClassification
from .agents import (
    run_triage_agent,
    run_decomposition_agent,
    run_retrieval_agent,
    run_synthesizer_agent,
    run_reviewer_agent,
)
from prompts import get_classification_prompt

# Set up logging
logger = logging.getLogger(__name__)
import chromadb
CHROMADB_AVAILABLE = True
class AtlanRAGSystem:
    def __init__(self, 
                 gemini_api_keys: List[str],
                 chromadb_config: Dict[str, str] = None,
                 generation_model: str = "gemini-2.5-flash",
                 embedding_model: str = "gemini-embedding-001",
                 collection_name: str = "new_atlan_docs"):
        
        if not gemini_api_keys or not isinstance(gemini_api_keys, list) or gemini_api_keys == ['']:
            raise ValueError("Gemini API keys are required and must be provided as a list.")

        self.gemini_api_keys = gemini_api_keys
        self.current_key_index = 0
        self.generation_model_name = generation_model
        self.embedding_model = embedding_model
        self.chromadb_available = CHROMADB_AVAILABLE
        self.state: Optional[TicketState] = None
        
        self._configure_gemini()
        self.generation_model = genai.GenerativeModel(self.generation_model_name)

        if self.chromadb_available and chromadb_config and chromadb:
            try:
                self.chroma_client = chromadb.CloudClient(**chromadb_config)
                self.collection = self.chroma_client.get_or_create_collection(collection_name)
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.chromadb_available = False
    
    def _configure_gemini(self):
        genai.configure(api_key=self.gemini_api_keys[self.current_key_index])
    
    def _rotate_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
        self._configure_gemini()
        logger.info(f"Rotated to Gemini API key index {self.current_key_index}")

    def _make_api_request_with_rotation(self, request_func, *args, **kwargs):
        """Make an API request and automatically rotate to the next key after each call."""
        try:
            result = request_func(*args, **kwargs)
            # Rotate to next API key after successful request
            self._rotate_api_key()
            return result
        except Exception as e:
            # If there's an error, still rotate but also handle the error
            if "quota" in str(e).lower():
                logger.warning(f"Quota exceeded on API key {self.current_key_index}, rotating...")
            self._rotate_api_key()
            raise e

    def _is_simple_query(self, query: str) -> bool:
        return len(query.split()) < 15 and '?' not in query

# In core/rag_system.py, inside the AtlanRAGSystem class

    def classify_ticket(self, ticket_text: str, max_retries: int = 3) -> TicketClassification:
        if not ticket_text or not ticket_text.strip():
            # Fallback for empty input - this will now work correctly.
            return TicketClassification(
                topic="General", sentiment="Neutral", priority="P2",
                reasoning="Fallback due to empty input."
            )

        classification_prompt = get_classification_prompt(ticket_text=ticket_text)

        for attempt in range(max_retries):
            try:
                response = self._make_api_request_with_rotation(
                    self.generation_model.generate_content, 
                    classification_prompt
                )
                response_text = response.text.strip().lstrip("```json").rstrip("```")
                result = json.loads(response_text)
                return TicketClassification(
                    topic=result.get('topic', 'General'),
                    sentiment=result.get('sentiment', 'Neutral'),
                    priority=result.get('priority', 'P2'),
                    reasoning=result.get('reasoning', 'No reasoning provided by model.'),
                    confidence_scores=result.get('confidence_scores', {})
                )
            except (json.JSONDecodeError, AttributeError, Exception) as e:
                logger.warning(f"Classification attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(1)
        
        # The fallback logic now perfectly matches the dataclass defaults.
        logger.warning("All classification attempts failed. Using fallback classification.")
        return TicketClassification(
            topic="General", sentiment="Neutral", priority="P2",
            reasoning="Fallback classification due to repeated API or parsing errors."
        )

    def generate_query_embedding(self, query: str, max_retries: int = 3) -> List[float]:
        if not query:
            raise ValueError("Cannot generate embedding for an empty query.")
        for attempt in range(max_retries):
            try:
                response = self._make_api_request_with_rotation(
                    genai.embed_content,
                    model=self.embedding_model,
                    content=query,
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768
                )
                return response['embedding']
            except Exception as e:
                if attempt == max_retries - 1: 
                    raise e
                time.sleep(1)
        raise Exception("Failed to generate query embedding.")

    def orchestrate_response(self, query: str, status_callback: Optional[Callable] = None) -> Generator[str, None, None]:
        self.state = TicketState(original_query=query)
        reasoning_path = []
        escalation_status = "Not Required"

        def update_status(agent_name, status):
            if status_callback:
                status_callback(agent_name, status)

        try:
            update_status("Triage", "⏳ Running...")
            self.state.classification = run_triage_agent(self.state, self)
            reasoning_path.append("Triage")
            update_status("Triage", "✅ Complete")

            RAG_TOPICS = ["How-to", "Product", "Best Practices", "API/SDK", "SSO"]
            if self.state.classification.topic not in RAG_TOPICS:
                update_status("Routing", "✅ Complete")
                routing_message = f"This ticket has been classified as a '{self.state.classification.topic}' issue and has been routed to the appropriate team for expert assistance."
                self.state.final_response = RAGResponse(query=query, answer=routing_message, sources=[], confidence_score=0.99,
                                                 response_metadata={'agent_workflow': 'routed', 'classification': self.state.classification})
                yield routing_message
                return

            use_decomposition = not self._is_simple_query(query)
            use_review = use_decomposition

            if use_decomposition:
                update_status("Decomposition", "⏳ Running...")
                self.state.sub_questions = run_decomposition_agent(self.state, self.generation_model)
                reasoning_path.append("Decomposition")
                update_status("Decomposition", "✅ Complete")
            else:
                self.state.sub_questions = [query]
                update_status("Decomposition", "➡️ Skipped")

            update_status("Retrieval", "⏳ Running...")
            self.state.retrieved_context = run_retrieval_agent(self.state, self)
            reasoning_path.append("Retrieval")
            update_status("Retrieval", "✅ Complete")

            final_answer_stream = []
            review_passed = False
            for attempt in range(2):
                update_status("Synthesis", f"⏳ Running (Attempt {attempt+1})...")
                answer_stream = run_synthesizer_agent(self.state, self.generation_model)
                draft_chunks = [chunk for chunk in answer_stream]
                self.state.draft_response = "".join(draft_chunks)
                reasoning_path.append(f"Synthesis_{attempt+1}")
                update_status("Synthesis", f"✅ Complete (Attempt {attempt+1})")

                if use_review:
                    update_status("Review", f"⏳ Running (Attempt {attempt+1})...")
                    review_passed, review_feedback = run_reviewer_agent(self.state, self.generation_model)
                    reasoning_path.append(f"Review_{attempt+1}")
                    if review_passed:
                        update_status("Review", "✅ Passed")
                        final_answer_stream = draft_chunks
                        break
                    else:
                        update_status("Review", f"❌ Failed. Retrying...")
                        self.state.review_feedback = review_feedback
                else:
                    review_passed = True
                    final_answer_stream = draft_chunks
                    update_status("Review", "➡️ Skipped")
                    break
            
            final_draft = self.state.draft_response
            if use_review and not review_passed:
                final_draft += "\n\n---\n*Internal Note: QA review failed. Human oversight recommended.*"

            retrieved_chunks = [c for chunks in self.state.retrieved_context.values() for c in chunks]
            retrieval_confidence = (sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks)) if retrieved_chunks else 0.0
            final_confidence = (retrieval_confidence * 0.7) + (0.3 if review_passed else 0.0)
            
            CONFIDENCE_THRESHOLD = 0.60
            escalation_suffix = None
            if final_confidence < CONFIDENCE_THRESHOLD:
                escalation_status = "Flagged for Human Review"
                escalation_suffix = f"\n\n---\n*To ensure you receive the most complete information, I have also flagged this conversation for review by a human support specialist. They will follow up if there are additional details to add.*"
                logger.warning(f"Final confidence ({final_confidence:.2f}) is below threshold. Appending escalation notice.")

            for chunk in final_answer_stream:
                yield chunk
            
            if escalation_suffix:
                yield escalation_suffix
            
            final_draft_with_suffix = final_draft + (escalation_suffix or "")
            
            all_sources = [c for chunks in self.state.retrieved_context.values() for c in chunks]
            unique_sources = list({s.url: s for s in all_sources if s.url}.values())
            self.state.final_response = RAGResponse(
                query=query, answer=final_draft_with_suffix, sources=unique_sources, confidence_score=final_confidence,
                response_metadata={'agent_workflow': 'success', 'classification': self.state.classification,
                                   'sub_questions': self.state.sub_questions, 'reasoning_path': reasoning_path,
                                   'escalation_status': escalation_status})

        except Exception as e:
            logger.critical(f"A critical error occurred at stage '{self.state.error_stage}'. Error: {e}", exc_info=True)
            error_message = f"I apologize, an error occurred at the '{self.state.error_stage}' step. Please try rephrasing your question."
            self.state.final_response = RAGResponse(query=query, answer=error_message, sources=[], confidence_score=0.1,
                                             response_metadata={'agent_workflow': 'failed', 'error_stage': self.state.error_stage})
            yield error_message