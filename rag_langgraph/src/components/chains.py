
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from .utils import llm

# We need a structured output for the grader #
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

### define chain getters ###
def get_grader_chain():
    """
    Returns a chain that outputs a JSON-like object with 'binary_score'.
    Uses .with_structured_output() which works best with tool-calling models
    """

    # We ask the model to act as a judge.
    grader_system_prompt = (
        """You are a grader assessing relevance of a retrieved document to a user question. \n """
        """If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n"""
        """Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    )

    grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    try:
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        return grader_prompt | structured_llm_grader

    except Exception:
        parser = JsonOutputParser(pydantic_object=GradeDocuments)
    
        # We inject the format instructions directly into the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader. Output JSON only. \n{format_instructions}"),
            ("human", "Doc: {document} \n Question: {question}")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        return prompt | llm | parser


def get_rewriter_chain():
    """
    Returns a chain that outputs a string (the new question).
    """

    # The rewriter prompt should be creative and open-ended
    rewriter_system_prompt = (
        """You are a question re-writer that converts an input question to a better version that is optimized"""
        """for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    )

    rewriter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rewriter_system_prompt),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    return rewriter_prompt | llm | StrOutputParser()


def get_generator_chain():
    """
    Returns a chain that outputs a string (the final answer).
    """
    generator_prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. Use the following pieces 
            of retrieved context to answer the question. 
            If the context is useful, include relevant information from it in your answer.
            If the context is an empty string("" or ''), answer based on your own knowledge.
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.

            Question: {question} 

            \n 
            Context: {context} 

            \n
            Answer:"""
        )
    return generator_prompt | llm | StrOutputParser()