# core/agents.py

import google.generativeai as genai
import json
import logging
from typing import List, Dict, TYPE_CHECKING, Generator, Tuple

# Import prompts from the prompts module
from prompts import (
    get_decomposition_prompt,
    get_synthesizer_prompt,
    get_reviewer_prompt
)

# Import data structures from the state module
from .state import TicketState, RetrievedChunk, TicketClassification

# Use TYPE_CHECKING to avoid circular import issues with AtlanRAGSystem
if TYPE_CHECKING:
    from .rag_system import AtlanRAGSystem

# Set up logging
logger = logging.getLogger(__name__)


def run_triage_agent(state: TicketState, rag_system: 'AtlanRAGSystem') -> TicketClassification:
    """Agent 1: Classifies the ticket for topic, sentiment, and priority."""
    state.error_stage = "Triage"
    return rag_system.classify_ticket(state.original_query)


def run_decomposition_agent(state: TicketState, model: genai.GenerativeModel) -> List[str]:
    """Agent 2: Decomposes a complex user query into distinct, answerable sub-questions."""
    state.error_stage = "Decomposition"
    prompt = get_decomposition_prompt(state.original_query, state.classification)

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()

        result = json.loads(response_text)
        questions = result.get('questions', [state.original_query])
        
        if not isinstance(questions, list) or not questions:
            return [state.original_query]
        return questions
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"Decomposition agent failed: {e}. Falling back to original query.")
        return [state.original_query]


def run_retrieval_agent(state: TicketState, rag_system: 'AtlanRAGSystem') -> Dict[str, List[RetrievedChunk]]:
    """Agent 3: For each sub-question, retrieves relevant, high-quality document chunks."""
    state.error_stage = "Retrieval"
    all_context = {}
    RELEVANCE_THRESHOLD = 0.05  # Safety Guardrail: Minimum similarity score to be considered relevant.

    if not rag_system.chromadb_available or not rag_system.collection:
        logger.warning("ChromaDB not available. Retrieval agent returning empty context.")
        for question in state.sub_questions:
            all_context[question] = []
        return all_context

    metadata_filter = {"domain": state.classification.topic}

    for question in state.sub_questions:
        try:
            query_embedding = rag_system.generate_query_embedding(question)
            results = rag_system.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where=metadata_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            retrieved_chunks = []
            if results and results.get('documents') and results['documents']:
                # Fix: Handle nested lists properly - ChromaDB returns nested lists
                documents = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
                metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
                distances = results['distances'][0] if isinstance(results['distances'][0], list) else results['distances']
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # Ensure distance is a number - handle both single values and nested lists
                    if isinstance(distance, (list, tuple)):
                        distance = distance[0] if distance else 0.0
                    
                    similarity_score = 1 - float(distance)
                    
                    # Apply the relevance threshold guardrail
                    if similarity_score > RELEVANCE_THRESHOLD:
                        chunk = RetrievedChunk(
                            content=doc,
                            url=metadata.get('url', ''),
                            title=metadata.get('title', ''),
                            chunk_id=metadata.get('chunk_id', ''),
                            chunk_index=metadata.get('chunk_index', 0),
                            similarity_score=similarity_score,
                            token_count=metadata.get('token_count', 0),
                            metadata=metadata
                        )
                        retrieved_chunks.append(chunk)
            
            all_context[question] = sorted(retrieved_chunks, key=lambda x: x.similarity_score, reverse=True)
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for question '{question}': {e}")
            all_context[question] = []
    return all_context


def run_synthesizer_agent(state: TicketState, model: genai.GenerativeModel) -> Generator[str, None, None]:
    """Agent 4: Synthesizes retrieved information into a streaming draft response, following strict grounding rules."""
    state.error_stage = "Synthesis"
    prompt = get_synthesizer_prompt(state)
    
    try:
        response_stream = model.generate_content(
            prompt,
            stream=True,
            generation_config=genai.GenerationConfig(
                temperature=0.1, # Lower temperature for more factual responses
                top_p=0.9,
                max_output_tokens=2500,
            )
        )
        for chunk in response_stream:
            yield chunk.text
            
    except Exception as e:
        logger.error(f"Synthesizer agent failed to generate stream: {e}")
        yield "I apologize, but I encountered an error while formulating a response."


def run_reviewer_agent(state: TicketState, model: genai.GenerativeModel) -> Tuple[bool, str]:
    """
    Agent 5: Reviews the draft response for quality, grounding, and tone.
    Returns: A tuple containing (bool: review_passed, str: feedback).
    """
    state.error_stage = "Review"
    if not state.draft_response or len(state.draft_response) < 50:
        return False, "Draft response was too short or empty."

    prompt = get_reviewer_prompt(state.draft_response, state)
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.0) # Zero temperature for deterministic decision
        )
        review_text = response.text.strip().upper()
        
        if "PASS" in review_text:
            return True, "Response meets quality standards."
        else:
            feedback = review_text.replace('FAIL', '').strip()
            # Provide a default feedback message if the model's reason is empty
            return False, feedback if feedback else "Response did not meet quality and grounding standards."
            
    except Exception as e:
        logger.error(f"Reviewer agent failed: {e}. Defaulting to PASS to avoid blocking the flow.")
        # In a production system, this failure might be handled more strictly.
        return True, "Review agent encountered a technical error."