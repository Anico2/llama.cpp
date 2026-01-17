"""
rag.py
"""
import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Literal, Dict, List
from functools import wraps

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
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
    def __init__(self, retriever, llm, prompt, cache=True):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.cache_rewrites = cache
        self._rewrite_cache = {}
        
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite the input query to improve clarity for search retrieval. Keep it concise.\nInput: {x}\nRewritten:"
        )
        self.rewriter_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    @log_execution
    def answer(self, query: str) -> dict:
        if self.cache_rewrites and query in self._rewrite_cache:
            rewritten_query = self._rewrite_cache[query]
            logger.info("Using cached rewrite.")
        else:
            rewritten_query = self.rewriter_chain.invoke({"x": query}).replace("Rewritten:", "").strip()
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
            f"Generate {self.num_queries} varied search queries based on: {{question}}. Return only the queries separated by newlines."
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
        logger.info(f"Generating sub-queries...")
        query_gen_chain = self.perspectives_prompt | self.llm | StrOutputParser()
        raw_output = query_gen_chain.invoke({"question": query})
        queries = [q.strip() for q in raw_output.split("\n") if q.strip()]
        
        # Limit distinct queries to avoid flooding
        queries = queries[:self.num_queries]
        logger.info(f"Sub-queries: {queries}")
        
        doc_lists = [self.retriever.invoke(q) for q in queries]
        combined_docs = self.get_unique_union(doc_lists)
        logger.info(f"Unique docs found: {len(combined_docs)}")
        
        context_text = "\n\n".join([d.page_content for d in combined_docs])
        chain = self.prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke({"context": context_text, "question": query})

        return {
            "answer": answer_text,
            "source_documents": combined_docs,
            "context_used": len(combined_docs) > 0
        }

class RelevanceScore(BaseModel):
    id: int
    relevance_score: int = Field(description="Score from 0-10, where 10 is highly relevant")
    reasoning: str = Field(description="Brief reason for the score")

class RankedDocs(BaseModel):
    rankings: List[RelevanceScore]

class RerankRAGStrategy(BaseRAGStrategy):
    """
    Retrieves a k set of documents (as put in the passed retriever), 
    then uses the LLM to score/rank them.
    """
    def __init__(self, retriever, llm, prompt, top_k=2):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.top_k = top_k
        
    @log_execution
    def answer(self, query: str) -> dict:
        # 1. Retrieve Candidate Pool 
    
        candidates = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(candidates)} candidates for reranking.")
        
        if not candidates:
            return {"answer": "No documents found.", "source_documents": [], "context_used": False}

        # 2. Rerank using LLM
        snippets = []
        for idx, doc in enumerate(candidates):
            # Truncate content for reranking speed
            content_preview = doc.page_content[:400].replace("\n", " ")
            snippets.append(f"ID: {idx}\nContent: {content_preview}")
            
        snippets_text = "\n\n".join(snippets)
        
        rerank_prompt = f"""
        You are a relevance ranker.
        Question: {query}
        
        Documents:
        {snippets_text}
        
        Task: return a JSON object with a list 'rankings'. 
        Each item must have 'id' (int) and 'relevance_score' (0-10).
        Only include documents with score > 3.
        """
        
        try:
            ranker = self.llm.with_structured_output(RankedDocs)
            result = ranker.invoke(rerank_prompt)
            
            # Sort by score desc
            sorted_ranks = sorted(result.rankings, key=lambda x: x.relevance_score, reverse=True)
            
            top_ids = [r.id for r in sorted_ranks[:self.top_k]]
            final_docs = [candidates[i] for i in top_ids if i < len(candidates)]
            
            logger.info(f"Reranked: Kept {len(final_docs)} top docs out of {len(candidates)}.")
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to top {self.top_k} raw results.")
            final_docs = candidates[:self.top_k]

        # 3. Generate Answer
        context_text = "\n\n".join([d.page_content for d in final_docs])
        chain = self.prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke({"context": context_text, "question": query})

        return {
            "answer": answer_text,
            "source_documents": final_docs,
            "context_used": len(final_docs) > 0
        }

class RouteChoice(BaseModel):
    choice: Literal["simple", "multi_query", "RRR", "rerank"] = Field(
        description="Choose the best strategy based on question complexity."
    )

class RouterRAGStrategy(BaseRAGStrategy):
    def __init__(self, strategies: Dict[str, BaseRAGStrategy], llm):
        self.strategies = strategies
        self.llm = llm

    @log_execution
    def answer(self, query: str) -> dict:
        router_llm = self.llm.with_structured_output(RouteChoice)
        try:
            route = router_llm.invoke(f"Route this question: {query}")
            choice = route.choice
        except Exception:
            # we fall back to simple if routing fails
            choice = "simple" 
            
        logger.info(f"Routing to '{choice}' strategy.")
        return self.strategies[choice].answer(query)