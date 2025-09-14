# app.py

import streamlit as st
import logging

# Local application imports
from conversational_mode import conversational_ai_mode
from core.rag_system import AtlanRAGSystem, CHROMADB_AVAILABLE
from utils import get_topic_color, get_sentiment_emoji, load_sample_tickets

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Atlan AI Customer Support Copilot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main-header { color: #0052CC; font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem; }
    .ticket-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .topic-badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; color: white; font-size: 0.75rem; font-weight: bold; margin-right: 0.5rem; }
    .sentiment-label { font-size: 1.2rem; margin-right: 0.5rem; }
    .priority-badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; color: white; font-size: 0.75rem; font-weight: bold; }
    .p0-badge { background-color: #FF5630; }
    .p1-badge { background-color: #FF8B00; }
    .p2-badge { background-color: #FFC400; color: black; }
    .backend-view { background-color: #F5F5F5; border-left: 4px solid #0052CC; padding: 1rem; margin: 1rem 0; border-radius: 4px; }
    .frontend-view { background-color: #FFFFFF; border: 1px solid #e0e0e0; padding: 1rem; margin: 1rem 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .source-list { background-color: #F8F9FA; padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    try:
        CHABI = st.secrets["GEMINI_API_KEYS"]
        
        CHROMADB_CONFIG = None
        if "CHROMADB_CONFIG" in st.secrets:
            CHROMADB_CONFIG = dict(st.secrets["CHROMADB_CONFIG"])
        
        return AtlanRAGSystem(
            gemini_api_keys=CHABI,
            chromadb_config=CHROMADB_CONFIG
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()

def bulk_ticket_dashboard():
    st.markdown('<h1 class="main-header">Bulk Ticket Classification</h1>', unsafe_allow_html=True)
    rag_system = st.session_state.rag_system
    
    if "classified_tickets" not in st.session_state:
        with st.spinner("Processing tickets in bulk..."):
            tickets = load_sample_tickets()
            classified_tickets = []
            progress_bar = st.progress(0, text="Classifying tickets...")
            for i, ticket in enumerate(tickets):
                classification = rag_system.classify_ticket(ticket["text"])
                classified_ticket = {**ticket, "classification": classification}
                classified_tickets.append(classified_ticket)
                progress_bar.progress((i + 1) / len(tickets), text=f"Classified ticket {i+1}/{len(tickets)}")
            st.session_state.classified_tickets = classified_tickets
        st.success(f"Successfully processed {len(classified_tickets)} tickets.")

    for ticket in st.session_state.get("classified_tickets", []):
        with st.container(border=True):
            st.subheader(f"Ticket {ticket['id']}: {ticket['subject']}")
            classification = ticket["classification"]
            topic_color = get_topic_color(classification.topic)
            sentiment_emoji = get_sentiment_emoji(classification.sentiment)
            st.markdown(f'''
                <span class="topic-badge" style="background-color: {topic_color};">{classification.topic}</span>
                <span class="sentiment-label">{sentiment_emoji} {classification.sentiment}</span>
                <span class="priority-badge {classification.priority.lower()}-badge">{classification.priority}</span>
            ''', unsafe_allow_html=True)
            with st.expander("View Ticket Body"):
                st.write(ticket['text'])

def interactive_ai_agent():
    st.markdown('<h1 class="main-header">Interactive AI Agent</h1>', unsafe_allow_html=True)
    rag_system = st.session_state.rag_system
    
    ticket_text = st.text_area(
        "Enter your ticket or question:",
        height=150,
        placeholder="How can I set up SSO with Okta and what are the required attributes?"
    )
    
    if st.button("Submit Ticket", type="primary", use_container_width=True):
        if not ticket_text.strip():
            st.warning("Please enter a ticket or question.")
            return

        st.markdown("---")
        st.markdown("## Internal Analysis")
        with st.container(border=True):
            status_placeholders = {
                "Triage": st.empty(), "Decomposition": st.empty(), "Retrieval": st.empty(),
                "Synthesis": st.empty(), "Review": st.empty(), "Routing": st.empty()
            }
            for agent in status_placeholders:
                status_placeholders[agent].markdown(f"**{agent}:** ⚪ Pending")

            def status_callback(agent_name, status):
                if agent_name in status_placeholders:
                    status_placeholders[agent_name].markdown(f"**{agent_name}:** {status}")

            st.markdown("## Customer Response")
            with st.container(border=True):
                response_box = st.empty()
                response_generator = rag_system.orchestrate_response(ticket_text, status_callback)
                response_box.write_stream(response_generator)

            final_response = rag_system.state.final_response if hasattr(rag_system, 'state') else None
            
            if final_response and final_response.sources:
                with st.expander("View Sources"):
                    for source in final_response.sources:
                        st.write(f"- [{source.title}]({source.url})")

def main():
    st.session_state.rag_system = initialize_rag_system()
    
    with st.sidebar:
        st.markdown("## Atlan AI Support Copilot")
        st.markdown("---")
        page = st.radio(
            "Select Mode",
            ["Interactive AI Agent", "Conversational AI Agent", "Bulk Ticket Dashboard"],
            key="navigation"
        )
        st.markdown("---")
        st.markdown("### System Status")
        if "rag_system" in st.session_state and st.session_state.rag_system:
            if CHROMADB_AVAILABLE and st.session_state.rag_system.collection:
                st.success("RAG System Online")
                st.write(f"Knowledge Base: {st.session_state.rag_system.collection.count()} docs")
            else:
                st.warning("Classification-Only Mode")
        else:
            st.error("RAG System Offline")

    if page == "Interactive AI Agent":
        interactive_ai_agent()
    elif page == "Conversational AI Agent":
        conversational_ai_mode()
    elif page == "Bulk Ticket Dashboard":
        bulk_ticket_dashboard()

if __name__ == "__main__":
    main()
