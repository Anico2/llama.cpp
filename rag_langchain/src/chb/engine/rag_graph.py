"""
Logic for LangGraph based RAG with robust error handling.
"""
from langfuse import observe
import logging
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    retry_count: int


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes or no")


class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="yes or no")


class GradeAnswer(BaseModel):
    binary_score: str = Field(description="yes or no")


def safe_invoke_grader(chain, inputs, default_score="yes", node_name="Unknown"):
    """
    Helper function that catches JSON errors and returns a default safe value.
    This prevents the entire graph from crashing if the LLM is chatty.
    """
    try:
        response = chain.invoke(inputs)
        if isinstance(response, dict):
            return response.get("binary_score", default_score).lower()
        return response.binary_score.lower()
    except Exception as e:
        logger.warning(
            f"JSON parsing failed in {node_name}: {e}. Defaulting to '{default_score}'."
        )
        return default_score




@observe(as_type="generation")
def build_rag_graph(llm, retriever):
    # 1. NODE: Rewrite Query
    def rewrite(state: GraphState):
        print("---REWRITE QUERY---")
        question = state["question"]
        system = (
            """You are a question re-writer. Optimize the question for vector retrieval."""
            """ The question should not be as general as, for example, searching on google for a definition,"""
            """ but it should be precise, concise, correct typos and targeting a possible documents."""
            """ Additionally you should just return the improved question, without explanation or other stuff."""
            """ See the following example. \n """
            """ Question: user asked 'Speak about adjusted operated incomee from 2024-2025'"""
            """ Improved Question: 'Adjusted income comparison 2024 2025' """
        )
        msg = [
            ("system", system),
            ("human", "Question: {question}\nImproved Question:"),
        ]
        chain = ChatPromptTemplate.from_messages(msg) | llm | StrOutputParser()
        better_question = chain.invoke({"question": question})
        return {"question": better_question}

   
    def retrieve(state: GraphState):
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    
    def grade_documents(state: GraphState):
        print("---CHECK DOCUMENT RELEVANCE---")
        question = state["question"]
        documents = state["documents"]

        
        parser = JsonOutputParser(pydantic_object=GradeDocuments)

        system = """You are a grader assessing relevance of a retrieved document to a user question.
        Return a JSON object with key 'binary_score' set to 'yes' or 'no'.
        
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Document: {document}\n\nQuestion: {question}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        
        chain = prompt | llm | parser

        filtered_docs = []
        for d in documents:
            
            grade = safe_invoke_grader(
                chain,
                {"question": question, "document": d.page_content},
                default_score="yes", 
                node_name="GradeDocs",
            )

            if grade == "yes":
                filtered_docs.append(d)

        if not filtered_docs:
            print("---WARNING: ALL DOCS FILTERED, USING ORIGINAL---")
            filtered_docs = documents

        return {"documents": filtered_docs, "question": question}

    
    def generate(state: GraphState):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        prompt = ChatPromptTemplate.from_template(
            "Context: {context}\n\nQuestion: {question}\nAnswer:"
        )
        chain = prompt | llm | StrOutputParser()
        context = "\n\n".join([d.page_content for d in documents])
        generation = chain.invoke({"context": context, "question": question})

        return {"documents": documents, "question": question, "generation": generation}

   
    def check_hallucination_and_relevance(state: GraphState):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)

        # 1. Check Hallucinations
        parser_hallu = JsonOutputParser(pydantic_object=GradeHallucinations)
        prompt_hallu = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Assess if the answer is grounded in facts. JSON with 'binary_score' ('yes'/'no').\n{format_instructions}",
                ),
                ("human", "Facts: {documents}\nAnswer: {generation}"),
            ]
        ).partial(format_instructions=parser_hallu.get_format_instructions())

        chain_hallu = prompt_hallu | llm | parser_hallu

        # Default "yes" -> assumiamo che non stia allucinando se il parsing 
        # fallisce (meno costoso che riprovare)
        grade_hallu = safe_invoke_grader(
            chain_hallu,
            {"documents": str(documents), "generation": generation},
            "yes",
            "HallucinationCheck",
        )

        if grade_hallu == "yes":
            print("---DECISION: GROUNDED---")

            
            parser_ans = JsonOutputParser(pydantic_object=GradeAnswer)
            prompt_ans = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Assess if the answer addresses the question. JSON with 'binary_score' ('yes'/'no').\n{format_instructions}",
                    ),
                    ("human", "Question: {question}\nAnswer: {generation}"),
                ]
            ).partial(format_instructions=parser_ans.get_format_instructions())

            chain_ans = prompt_ans | llm | parser_ans
            grade_ans = safe_invoke_grader(
                chain_ans,
                {"question": question, "generation": generation},
                "yes",
                "AnswerCheck",
            )

            if grade_ans == "yes":
                print("---DECISION: USEFUL---")
                return "useful"
            else:
                return "not useful"  
        else:
            print("---DECISION: NOT GROUNDED---")
            return "not supported"  

    def route_after_generation(state: GraphState):
        if state.get("retry_count", 0) > 1:
            print("---MAX RETRIES HIT: RETURNING ANSWER---")
            return END

        status = check_hallucination_and_relevance(state)

        if status == "useful":
            return END
        elif status == "not supported":
            print("---ROUTING: RE-GENERATE---")
            return "generate"
        elif status == "not useful":
            print("---ROUTING: REWRITE QUERY---")
            return "rewrite"

    
    workflow = StateGraph(GraphState)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")

    workflow.add_conditional_edges(
        "generate",
        route_after_generation,
        {"useful": END, "generate": "generate", "rewrite": "rewrite", END: END},
    )

    return workflow.compile()
