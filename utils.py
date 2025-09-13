# utils.py

from typing import List, Dict
from sample import SAMPLE

def get_topic_color(topic: str) -> str:
    """Returns a consistent hex color code for a given topic string."""
    colors = {
        "How-to": "#36B37E",
        "Product": "#0052CC",
        "Connector": "#8777D9",
        "Lineage": "#FF8B00",
        "API/SDK": "#403294",
        "SSO": "#FF5630",
        "Glossary": "#00875A",
        "Best Practices": "#0747A6",
        "Sensitive Data": "#DE350B",
        "General": "#6B778C"
    }
    return colors.get(topic, "#6B778C")

def get_sentiment_emoji(sentiment: str) -> str:
    """Returns an emoji representation for a given sentiment string."""
    emojis = {
        "Frustrated": "😤",
        "Curious": "🤔",
        "Angry": "😠",
        "Neutral": "😐"
    }
    return emojis.get(sentiment, "😐")

def load_sample_tickets() -> List[Dict]:
    """

    Loads the sample tickets from the sample.py file and transforms them
    into the format expected by the application UI.
    """
    transformed_tickets = []
    for ticket in SAMPLE:
        transformed_tickets.append({
            "id": ticket.get("id"),
            "text": ticket.get("body"),
            "subject": ticket.get("subject")
        })
    return transformed_tickets