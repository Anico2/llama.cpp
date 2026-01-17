import logging
import time
from abc import ABC, abstractmethod
from typing import Literal, Dict
from functools import wraps

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RAG-System")

def log_execution(func):
    """Decorator to log function name and execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        strategy_name = args[0].__class__.__name__
        logger.info(f"Starting {strategy_name} for query...")
        
        result = func(*args, **kwargs)
        
        duration = time.time() - start_time
        logger.info(f"{strategy_name} finished in {duration:.2f}s")
        return result
    return wrapper

class BaseRAGStrategy(ABC):
    @abstractmethod
    def answer(self, query: str) -> dict:
        pass

class SimpleRAGStrategy(BaseRAGStrategy):
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    @log_execution
    def answer(self, query: str) -> dict:
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents.")
        
        context_text = "\n\n".join([d.page_content for d in docs])
        chain = self.prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke({"context": context_text, "question": query})

        return {
            "answer": answer_text,
            "source_documents": docs,
            "context_used": len(docs) > 0
        }

class RRRStrategy(BaseRAGStrategy):
    """Rewrite-Retrieve-Read Strategy."""
    def __init__(self, retriever, llm, prompt, cache=True):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.cache_rewrites = cache
        self._rewrite_cache = {}
        self._answer_cache = {} 
        
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite the input query to improve clarity: {x}. Rewritten_Query: "
        )
        self.rewriter_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    @log_execution
    def answer(self, query: str) -> dict:
        if self.cache_rewrites and query in self._rewrite_cache:
            rewritten_query = self._rewrite_cache[query]
            logger.info("Using cached rewrite.")
        else:
            rewritten_query = self.rewriter_chain.invoke({"x": query}).replace("Rewritten_Query: ", "").strip()
            self._rewrite_cache[query] = rewritten_query
            logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")

        docs = self.retriever.invoke(rewritten_query)
        context_text = "\n\n".join([d.page_content for d in docs])
        chain = self.prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke({"context": context_text, "question": query})

        return {"answer": answer_text, "source_documents": docs, "context_used": len(docs) > 0}

class MultiQueryRAGStrategy(BaseRAGStrategy):
    def __init__(self, retriever, llm, prompt, num_queries=3):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.num_queries = num_queries

        self.perspectives_prompt = ChatPromptTemplate.from_template(
            f"""You are an AI language model assistant. 
            Your task is to generate {self.num_queries} different versions 
            of the given user question to retrieve relevant documents from a vector database. 
            Provide these {self.num_queries} alternative questions separated by newlines.
            Original question: {{question}}"""
        )

    def get_unique_union(self, document_lists):
        deduped_docs = {}
        for sublist in document_lists:
            for doc in sublist:
                if doc.page_content not in deduped_docs:
                    deduped_docs[doc.page_content] = doc
        return list(deduped_docs.values())

    @log_execution
    def answer(self, query: str) -> dict:
        logger.info(f"Generating {self.num_queries} sub-queries...")
        
        query_gen_chain = self.perspectives_prompt | self.llm | StrOutputParser()
        raw_output = query_gen_chain.invoke({"question": query})
        
        
        queries = [q.strip() for q in raw_output.split("\n") if q.strip()]
        logger.info(f"Sub-queries generated: {queries}")
        
        
        doc_lists = [self.retriever.invoke(q) for q in queries]
        
        # Important to deduplicate results
        combined_docs = self.get_unique_union(doc_lists)
        logger.info(f"Total unique docs retrieved: {len(combined_docs)}")
        
        context_text = "\n\n".join([d.page_content for d in combined_docs])
        chain = self.prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke({"context": context_text, "question": query})

        return {
            "answer": answer_text,
            "source_documents": combined_docs,
            "context_used": len(combined_docs) > 0
        }

class RouteChoice(BaseModel):
    choice: Literal["simple", "multi_query", "rrr"] = Field(
        description="Choose the best strategy based on question complexity."
    )

class RouterRAGStrategy(BaseRAGStrategy):
    def __init__(self, strategies: Dict[str, BaseRAGStrategy], llm):
        self.strategies = strategies
        self.llm = llm

    @log_execution
    def answer(self, query: str) -> dict:
        # NOTE .with_structured_output ensure response matches RouteChoice model
        router_llm = self.llm.with_structured_output(RouteChoice)
        route = router_llm.invoke(f"Route this question: {query}")
        
        logger.info(f"Routing to '{route.choice}' strategy.")
        
        return self.strategies[route.choice].answer(query)