from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app.faq_retreiver import retrieve_faq_answer

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Few-shot examples (optional, improves classification)
examples = [
    ("What is your refund policy?", "faq"),
    ("How can I track my order?", "faq"),
    ("Can you check the status of my last subscription?", "personalized"),
    ("I want to cancel my account", "personalized"),
]

def classify_node(state: dict) -> dict:
    query = state["query"]

    # Prompt template
    prompt = PromptTemplate.from_template(
        """Classify the user query into one of two categories: 'faq' or 'personalized'.
        
User Query: {query}

Respond with only one word: faq or personalized.
"""
    )

    response = llm.invoke(prompt.format_messages(query=query)).content.strip().lower()

    # Fallback logic
    if response not in ["faq", "personalized"]:
        response = "faq"

    state["route"] = response
    return state

def faq_agent_node(state):
    docs = retrieve_faq_answer(state["query"])
    state["context"] = docs
    return state

def data_agent_node(state):
    contact = get_contact_info(state["email"])
    state["context"] = contact
    return state

def response_node(state):
    context = state["context"]
    query = state["query"]
    prompt = f"User Query: {query}\nContext: {context}\nReply helpfully."
    reply = llm.invoke(prompt).content
    state["response"] = reply
    return state

def feedback_node(state):
    # Simulate feedback; in real use-case, integrate HubSpot chatflow
    user_feedback = "yes"  # or "no"
    state["feedback"] = user_feedback
    return state

def ticket_node(state):
    ticket_id = create_ticket(state["email"], state["query"], state["response"])
    state["ticket_id"] = ticket_id
    return state
