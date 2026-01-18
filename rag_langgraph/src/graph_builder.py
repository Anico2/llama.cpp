import logging
from pathlib import Path
from langgraph.graph import END, StateGraph
from state import AgentState
from components.nodes import (
    retrieve,
    grade_documents,
    generate,
    transform_query,
    generate_no_context,
    give_up,
    evaluate_generation_no_context,
)

logger = logging.getLogger(Path(__file__).name.removesuffix(".py"))


def decide_to_generate(state):
    """
    Logic:
    1. If search succeeded (relevant docs found) -> Generate
    2. If search failed (irrelevant docs):
       - If retries < 3 -> Rewrite Query
       - If retries = 3 -> Try Generating without Context (one trial)
       - If retries > 3 -> Give Up
    """

    logger.info(
        f"(decide_to_generate) {state['search_failed'] = } || {state['retry_count'] = }"
    )
    if state["search_failed"] == "NO":
        logger.info("(decide_to_generate) Documents are relevant. Generating answer.")
        return "generate"

    if state["retry_count"] < 3:
        logger.info("(decide_to_generate) Search failed. Rewriting query.")
        return "transform_query"

    elif state["retry_count"] == 3:
        logger.info(
            "(decide_to_generate) Max retries reached. Trying generation without context."
        )
        return "generate_no_context"

    else:
        logger.info("(decide_to_generate) Unable to answer. Giving up.")
        return "give_up"


def check_no_context_validity(state):
    """Checks if the no-context answer was useful, otherwise gives up."""
    return END if state["is_acceptable"] else "give_up"


def build_graph():
    """Builds and returns the RAG agent graph workflow.

    This function constructs the entire state graph for the RAG agent,
    defining nodes, edges, and conditional flows based on the agent's logic.
    It returns a compiled StateGraph instance ready for invocation.
    """

    # Starting point: we create the graph instance
    workflow = StateGraph(AgentState)

    # Then, we add all the nodes that will compose the graph.
    # Each node corresponds to a function defined in components/nodes.py
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate_no_context", generate_no_context)
    workflow.add_node("evaluate_generation_no_context", evaluate_generation_no_context)
    workflow.add_node("give_up", give_up)

    # We have to define the entry point of the graph: this is the node where execution
    # starts. In this case, as starting point we have to retrieve relevant documents.
    workflow.set_entry_point("retrieve")

    # Being a graph, we have to define links (without cycles) between
    # the nodes. Here we define the standard flow of execution.
    # In this case for example, after the retrieval, we always grade the documents.
    # These are all unconditional/deterministic edges: we go from A to B.
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate_no_context", "evaluate_generation_no_context")
    workflow.add_edge("generate", END)
    workflow.add_edge("give_up", END)

    # We now add some conditionals

    # If the function 'decide_to_generate':
    # returns 'transform_query': we go to that node and then loop back to 'retrieval'.
    # returns 'generate': we go to that node to produce the final answer.
    # returns 'generate_no_context': we go to that node to try a one-shot generation without context.
    # returns 'give_up': we go to that node to terminate the flow.
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
            "generate_no_context": "generate_no_context",
            "give_up": "give_up",
        },
    )

    # If the function 'check_no_context_validity':
    # returns END: we end the flow successfully.
    # returns 'give_up': we go to that node to terminate the flow.
    workflow.add_conditional_edges(
        "evaluate_generation_no_context",
        check_no_context_validity,
        {END: END, "give_up": "give_up"},
    )

    return workflow.compile()
