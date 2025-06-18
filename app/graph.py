from langgraph.graph import StateGraph

from app.nodes import (
    classify_node, faq_agent_node, data_agent_node,
    response_node, feedback_node, ticket_node
)

def process_query(email, query):
    state = {
        "email": email,
        "query": query
    }

    builder = StateGraph()
    builder.add_node("Classifier", classify_node)
    builder.add_node("FAQAgent", faq_agent_node)
    builder.add_node("DataAgent", data_agent_node)
    builder.add_node("Response", response_node)
    builder.add_node("Feedback", feedback_node)
    builder.add_node("Ticket", ticket_node)

    builder.set_entry_point("Classifier")
    builder.add_conditional_edges("Classifier", lambda x: x["route"], {
        "faq": "FAQAgent",
        "personalized": "DataAgent"
    })
    builder.add_edge("FAQAgent", "Response")
    builder.add_edge("DataAgent", "Response")
    builder.add_edge("Response", "Feedback")
    builder.add_conditional_edges("Feedback", lambda x: x["feedback"], {
        "yes": "END",
        "no": "Ticket"
    })
    builder.add_edge("Ticket", "END")

    graph = builder.compile()
    result = graph.invoke(state)
    return result
