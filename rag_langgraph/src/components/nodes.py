"""Defines the nodes used in the RAG LangGraph application."""

import logging
from pathlib import Path
from .chains import get_grader_chain, get_rewriter_chain, get_generator_chain
from .utils import get_retriever

logger = logging.getLogger(Path(__file__).name.removesuffix(".py"))


def retrieve(state):
    """Retrieves documents based on the question in state.
    This is also the root node of the graph.

    The retriever is a LangChain vector retriever, that can we invoke
    on the documents. For details about api see: https://reference.langchain.com/python/langchain_core/retrievers/
    For rapid inspection on retriever object, use the following methods/attributes on it:
    model_fiels, pipe, vectorstore

    The function returns the retrieved documents under 'documents' key in state.
    """
    logger.info("(retrieve) Fetching documents ")
    question = state["question"]

    retriever = get_retriever()
    documents = retriever.invoke(question)
    logger.info(f"(retrieve) Retrieved {len(documents)} documents.")

    # NOTE: as stated in the langgraph docs "The graph state is the union
    # of the state channels defined at initialization". So even if we only
    # return documents key, the other keys (like question) remain in state.
    return {"documents": documents}


def grade_documents(state):
    """Node that grades retrieved documents for relevance.

    It uses the grader chain to filter out irrelevant documents.
    Search fails if no documents remain after filtering. The
    success/failure is indicated via 'search_failed' key in state.

    The grader chain is enforced to return a binary score 'YES'/'NO'.
    """

    logger.info("(grade_documents) Grading retrieved documents for relevance")
    question, documents = state["question"], state["documents"]

    grader, filtered_docs = get_grader_chain(), []

    # NOTE: we should check which implementation is faster: batch or async calls
    batch_inputs = [{"question": question, "document": d.page_content} for d in documents]
    scores = grader.batch(batch_inputs)
    for i, score in enumerate(scores):
        d = documents[i]
        grade = getattr(score, "binary_score", "") or score.get("binary_score", "")

        if grade.upper() == "YES":
            logger.info(
                f"(grade_documents) Doc relevant: {d.metadata.get('source', 'unknown')}"
            )
            filtered_docs.append(d)
        else:
            logger.info("(grade_documents) Doc irrelevant: filtering out.")
    
    # for d in documents:
    #     score = grader.invoke({"question": question, "document": d.page_content})

    #     # For safety reason, check if the score object has the attribute (pydantic)
    #     # or is a dict (json parser fallback)
    #     grade = getattr(score, "binary_score", "") or score.get("binary_score", "")

    #     if grade.upper() == "YES":
    #         logger.info(
    #             f"(grade_documents) Doc relevant: {d.metadata.get('source', 'unknown')}"
    #         )
    #         filtered_docs.append(d)
    #     else:
    #         logger.info("(grade_documents) Doc irrelevant: filtering out.")
    
    search_failed = "YES" if not filtered_docs else "NO"

    if search_failed == "YES":
        logger.warning(
            "(grade_documents) All documents filtered out. Flagging for rewrite."
        )
        search_failed = "YES"

    return {"documents": filtered_docs, "search_failed": search_failed}


def transform_query(state):
    """Node that rewrites the query for better retrieval.

    It uses the rewriter chain to produce a better version of the question,
    optimized for vectorstore retrieval.
    """
    logger.info("(transform_query) Rewriting the query for better retrieval")
    question, rewriter = state["question"], get_rewriter_chain()

    better_question = rewriter.invoke({"question": question})
    if hasattr(better_question, "content"):
        better_question = better_question.content

    logger.info(f"(transform_query) New Query: {better_question}")

    # Important to increment retry count
    current_retry = state["retry_count"]

    return {"question": better_question, "retry_count": current_retry + 1}


def generate(state):
    """Generates the final answer using the retrieved documents as context.

    It uses the generator chain to produce the final answer.

    """
    logger.info("(generate) Generating answer with context")
    question, documents = state["question"], state["documents"]

    generator = get_generator_chain()
    generation = generator.invoke({"context": documents, "question": question})

    if hasattr(generation, "content"):
        generation = generation.content

    return {"generation": generation}


def generate_no_context(state):
    """
    Generates the final answer without any context.
    This is a one-shot train when retrieval fails multiple times.
    """
    logger.info("(generate_no_context) Trying generation without context")
    question, generator = state["question"], get_generator_chain()

    # Pass empty context string and llm should answer based on its own knowledge
    generation = generator.invoke({"context": "", "question": question})

    if hasattr(generation, "content"):
        generation = generation.content

    return {"generation": generation}


def evaluate_generation_no_context(state):
    """Evaluates the no-context generation for acceptability.
    It uses the grader chain to determine if the generation is acceptable.

    The grader chain is enforced to return a binary score 'YES'/'NO'.
    If acceptable, sets 'is_acceptable' to True in state, else False.

    """
    logger.info("(evaluate_generation_no_context) Evaluating no-context generation")
    generation, question = state["generation"], state["question"]
    grader = get_grader_chain()

    score = grader.invoke({"question": question, "document": generation})

    grade = getattr(score, "binary_score", "") or score.get("binary_score", "")

    if grade.lower().upper() == "YES":
        logger.info("(evaluate_generation_no_context) Generation is acceptable.")
        return {"is_acceptable": True}
    
    logger.warning("(evaluate_generation_no_context) Generation is NOT acceptable.")
    return {"is_acceptable": False}


def give_up(state):
    """Final node when the agent gives up on answering the question."""

    logger.info("(give_up) Giving up on answering the question.")
    give_up_message = (
        """I apologize, but I could not find relevant information in the documents"""
        """ to answer your question and my own knowledge is insufficient."""
    )
    return {"generation": give_up_message}
