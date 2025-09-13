# conversational_mode.py

import streamlit as st
import time
from typing import List, Optional
from dataclasses import dataclass, field
import logging

# Local application imports
from core.state import TicketClassification, RetrievedChunk, RAGResponse
from utils import get_sentiment_emoji

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Represents a single message in the conversation, with rich metadata."""
    role: str
    content: str
    timestamp: str
    # Store the full response object for detailed analysis in the UI
    final_response_obj: Optional[RAGResponse] = None

class ConversationManager:
    """Manages the application's conversation state."""
    def __init__(self, rag_system):
        self.rag_system = rag_system
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

    def add_message(self, role: str, content: str, **kwargs):
        """Adds a new message to the conversation history."""
        message = ConversationMessage(
            role=role, content=content,
            timestamp=time.strftime("%H:%M:%S"), **kwargs
        )
        st.session_state.conversation_history.append(message)

    def clear_conversation(self):
        """Clears the chat history."""
        st.session_state.conversation_history = []
        logger.info("Conversation cleared.")

def render_message(message: ConversationMessage):
    """Renders a single chat message with its associated analysis expander."""
    avatar = "👤" if message.role == "user" else "🤖"
    with st.chat_message(message.role, avatar=avatar):
        st.markdown(message.content)
        
        # For assistant messages, provide an expander with the detailed analysis
        if message.role == "assistant" and message.final_response_obj:
            response_obj = message.final_response_obj
            with st.expander("View Analysis", expanded=False):
                metadata = response_obj.response_metadata
                classification = metadata.get('classification')
                
                if metadata.get('agent_workflow') == 'failed':
                    st.error(f"Workflow failed at stage: {metadata.get('error_stage')}")
                elif classification:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Topic", classification.topic)
                    col2.metric("Sentiment", f"{get_sentiment_emoji(classification.sentiment)} {classification.sentiment}")
                    col3.metric("Priority", classification.priority)
                    st.metric("Confidence", f"{response_obj.confidence_score:.1%}")
                    
                    st.markdown(f"**Escalation Status:** `{metadata.get('escalation_status', 'N/A')}`")

                    st.markdown("**Agent Reasoning Path:**")
                    path_str = " ➡️ ".join(metadata.get('reasoning_path', []))
                    st.code(path_str, language=None)

                if response_obj.sources:
                    st.markdown("**Sources Considered:**")
                    for source in response_obj.sources:
                        st.markdown(f"- [{source.title}]({source.url})")

        st.caption(f"⏰ {message.timestamp}")

def conversational_ai_mode():
    """Renders the full Conversational AI interface."""
    
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(st.session_state.rag_system)
    
    conv_mgr = st.session_state.conversation_manager
    rag_system = st.session_state.rag_system
    
    st.markdown('<h1 class="main-header">💬 Conversational AI Agent</h1>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat"):
        conv_mgr.clear_conversation()
        st.rerun()

    # Display past messages
    for message in st.session_state.conversation_history:
        render_message(message)

    # Add a welcome message if the conversation is new
    if not st.session_state.conversation_history:
        initial_message = "👋 **Welcome to the Atlan AI Assistant!** How can I help you today?"
        conv_mgr.add_message("assistant", initial_message)
        st.rerun()

    # Handle chat input from the user
    if prompt := st.chat_input("Ask me anything about Atlan..."):
        conv_mgr.add_message("user", prompt)
        st.rerun()

    # If the last message is from the user, generate a response
    if st.session_state.conversation_history and st.session_state.conversation_history[-1].role == "user":
        last_user_message = st.session_state.conversation_history[-1]
        
        with st.chat_message("assistant", avatar="🤖"):
            # Setup UI elements for live updates
            response_box = st.empty()
            analysis_expander = st.expander("View Live Analysis", expanded=True)
            with analysis_expander:
                status_placeholders = {
                    "Triage": st.empty(), "Decomposition": st.empty(), "Retrieval": st.empty(),
                    "Synthesis": st.empty(), "Review": st.empty(), "Routing": st.empty()
                }
                for agent in status_placeholders:
                    status_placeholders[agent].markdown(f"**{agent}:** ⚪ Pending")

            # Define the callback function for real-time status updates
            def status_callback(agent_name, status):
                if agent_name in status_placeholders:
                    status_placeholders[agent_name].markdown(f"**{agent_name}:** {status}")

            # Execute the full agentic workflow and stream the response
            try:
                response_generator = rag_system.orchestrate_response(last_user_message.content, status_callback)
                full_response_text = response_box.write_stream(response_generator)
                
                # After streaming is complete, get the final RAGResponse object
                final_response_obj = rag_system.state.final_response
                
                # Add the complete assistant message to history
                conv_mgr.add_message(
                    "assistant",
                    content=full_response_text,
                    final_response_obj=final_response_obj
                )
                
                # A single rerun at the end to finalize the display
                st.rerun()

            except Exception as e:
                logger.error(f"An unexpected error occurred in the conversational mode: {e}", exc_info=True)
                st.error("An unexpected error occurred. Please try again.")