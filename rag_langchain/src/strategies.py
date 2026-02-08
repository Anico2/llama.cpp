import logging
import json
import os
from abc import ABC, abstractmethod
from typing import Literal, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Redis
from langfuse.langchain.CallbackHandler import LangchainCallbackHandler
import redis
from pydantic import BaseModel, Field

from utils import log_execution, EncoderClient, DecoderClient

logger = logging.getLogger(__name__)
try:
    from rag_graph import build_rag_graph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not found. Graph strategy will fail if selected.")


class RAGStrategy(ABC):
    @abstractmethod
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """Answer interface

        Args:
            query (str): user query.
            callbacks (LangchainCallbackHandler | None): Langfuse callback (e.g. Langufuse traces)

        Returns:
            dict: dictionary with decoder answer
        """
        pass


class CachedRAGStrategy(RAGStrategy):
    """
    Wraps another strategy to provide Semantic Caching via Redis.
    If a similar query exists (similarity > threshold), returns cached answer.
    """

    def __init__(
        self,
        base_strategy: RAGStrategy,
        embeddings: EncoderClient,
        redis_url: str | None = None,
        threshold: float = 0.9,
    ):
        """_summary_

        Args:
            base_strategy (RAGStrategy): strategy instance
            embeddings (EncoderClient): encoder client
            redis_url (str | None): redis endpoint.
            threshold (float): similarity threshold.
        """
        self.base_strategy = base_strategy
        self.embeddings = embeddings
        self.threshold = threshold
        self.redis_url = redis_url if redis_url else os.environ["REDIS_URL_CACHE"]
        self.index_name = "semantic_cache"

        try:
            # Redis vector store for semantic checking
            self.cache_store = Redis(
                redis_url=self.redis_url,
                index_name=self.index_name,
                embedding=self.embeddings,
                key_prefix="cache_doc:",
            )
        except Exception as e:
            logger.error(f"Failed to initialize Semantic Cache: {e}")
            self.cache_store = None

    @log_execution(logger=logger)
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """See base class"""
        if not self.cache_store:
            return self.base_strategy.answer(query, callbacks)

        # 1. Check Cache
        try:
            # Check for similar queries in Redis
            # standard Redis vector check returns documents and distance/scores
            results = self.cache_store.similarity_search_with_score(query, k=1)
            if results:
                doc, score = results[0]
                # Assuming Redis returns distance (0 = identical).
                # If threshold is similarity (0.9), max distance is 0.1
                max_distance = 1 - self.threshold

                if score <= max_distance:
                    logger.info(
                        f"Cache HIT (score: {score:.4f}). Returning cached answer."
                    )
                    cached_data = json.loads(doc.metadata.get("answer_json", "{}"))
                    return cached_data
                else:
                    logger.info(
                        f"Cache MISS (closest score: {score:.4f} > limit {max_distance})."
                    )
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

        # 2. Execute Strategy (MISS)
        result = self.base_strategy.answer(query, callbacks)

        # 3. Update Cache
        try:
            cache_payload = {
                "answer": result.get("answer", ""),
                "context_used": result.get("context_used", False),
                # minimal doc storage to save space
                "source_documents": [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in result.get("source_documents", [])
                ],
            }

            metadata = {"answer_json": json.dumps(cache_payload)}

            self.cache_store.add_documents(
                [Document(page_content=query, metadata=metadata)]
            )
            logger.info("Cache updated with new query/answer.")

        except Exception as e:
            logger.warning(f"Failed to write to cache: {e}")

        return result


class SimpleRAGStrategy(RAGStrategy):
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: DecoderClient,
        prompt: ChatPromptTemplate,
    ):
        """Simplest strategy which uses the user query to retrieve similar
            item and sends the first to llm. Not optimal.

        Args:
            retriever (VectorStoreRetriever): retriever client.
            llm (DecoderClient): decoder client.
            prompt (ChatPromptTemplate): prompt template.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    @log_execution(logger=logger)
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """See base class"""

        docs = self.retriever.invoke(query, config={"callbacks": callbacks})
        docs_ = docs[:1]
        logger.info(f"Retrieved {len(docs)} documents. Sending {len(docs_)} to LLM.")
        context_text = "\n\n".join([d.page_content for d in docs_])

        chain = self.prompt | self.llm | StrOutputParser()

        answer_text = chain.invoke(
            {"context": context_text, "question": query},
            config={"callbacks": callbacks},
        )

        return {
            "answer": answer_text,
            "source_documents": docs,
            "context_used": len(docs) > 0,
        }


class RRRStrategy(RAGStrategy):
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: DecoderClient,
        prompt: ChatPromptTemplate,
        cache: bool = True,
    ):
        """Rewrite-Retrieve-Read strategy. We first rewrite the user query before
            performing similarity search

        Args:
            retriever (VectorStoreRetriever): retriever client.
            llm (DecoderClient): decoder client.
            prompt (ChatPromptTemplate): prompt template.
            cache (bool): Whether to use already rewritten query.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.cache_rewrites = cache
        self._rewrite_cache = {}
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite the input query to improve clarity for search retrieval. "
            "Keep it concise. Keep the original user intent. "
            "Your only task is to refine the query (if needed) to make it more effective "
            "for vector similarity search. Just provide the rewritten query. "
            "\nInput: {x} \nRewritten:"
        )
        self.rewriter_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    @log_execution(logger=logger)
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """See base class"""

        if self.cache_rewrites and query in self._rewrite_cache:
            rewritten_query = self._rewrite_cache[query]
            logger.info("Using cached rewrite.")
        else:
            rewritten_query = (
                self.rewriter_chain.invoke(
                    {"x": query}, config={"callbacks": callbacks}
                )
                .replace("Rewritten:", "")
                .strip()
            )
            self._rewrite_cache[query] = rewritten_query
            logger.info(
                f"Original query: \n<{query}>, \n\n Rewritten query: \n <{rewritten_query}>"
            )

        docs = self.retriever.invoke(rewritten_query, config={"callbacks": callbacks})
        context_text = "\n\n".join([d.page_content for d in docs])

        chain = self.prompt | self.llm | StrOutputParser()

        answer_text = chain.invoke(
            {"context": context_text, "question": query},
            config={"callbacks": callbacks},
        )

        return {
            "answer": answer_text,
            "source_documents": docs,
            "context_used": len(docs) > 0,
        }


class MultiQueryRAGStrategy(RAGStrategy):
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: DecoderClient,
        prompt: ChatPromptTemplate,
        num_queries: int = 3,
    ):
        """Strategy that generates different queries and use each of them, in turn,
            to retrieve more documents than a single query. The documents are sent to llm.
            TODO: refine this, adding a reranking step

        Args:
            retriever (VectorStoreRetriever): retriever client.
            llm (DecoderClient): decoder client.
            prompt (ChatPromptTemplate): prompt template.
            num_queries (int): how many queries to generate.
        """
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

    @log_execution(logger=logger)
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """See base class"""

        logger.info(f"Generating {self.num_queries} sub-queries")
        query_gen_chain = self.perspectives_prompt | self.llm | StrOutputParser()
        raw_output = query_gen_chain.invoke(
            {"question": query}, config={"callbacks": callbacks}
        )
        queries = [q.strip() for q in raw_output.split("\n") if q.strip()]
        queries = queries[: self.num_queries]
        logger.info(f"Sub-queries: {queries}")
        doc_lists = [
            self.retriever.invoke(q, config={"callbacks": callbacks}) for q in queries
        ]
        combined_docs = self.get_unique_union(doc_lists)
        logger.info(f"Unique docs found: {len(combined_docs)}")
        context_text = "\n\n".join([d.page_content for d in combined_docs])
        chain = self.prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke(
            {"context": context_text, "question": query},
            config={"callbacks": callbacks},
        )

        return {
            "answer": answer_text,
            "source_documents": combined_docs,
            "context_used": len(combined_docs) > 0,
        }


class RelevanceScore(BaseModel):
    id: int
    relevance_score: int = Field(
        description="Score from 0-10, where 10 is highly relevant"
    )
    reasoning: str = Field(description="Brief reason for the score")


class RankedDocs(BaseModel):
    rankings: List[RelevanceScore]


class RerankRAGStrategy(RAGStrategy):
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: DecoderClient,
        prompt: ChatPromptTemplate,
        top_k: int = 2,
    ):
        """Strategy that after retrieving documents ask the decoder to rank them, keeping the first k.
            TODO: currently documents are ranked by the same decoder, we should
                implement the option to use another one

        Args:
            retriever (VectorStoreRetriever): retriever client
            llm (DecoderClient): decoder client
            prompt (ChatPromptTemplate): 
            top_k (int): Documents to keep after reranking.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.top_k = top_k

    @log_execution(logger=logger)
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """See base class"""

        candidates = self.retriever.invoke(query, config={"callbacks": callbacks})
        logger.info(f"Retrieved {len(candidates)} candidates for reranking.")

        if not candidates:
            return {
                "answer": "No documents found.",
                "source_documents": [],
                "context_used": False,
            }

        snippets = []
        for idx, doc in enumerate(candidates):
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
            result = ranker.invoke(rerank_prompt, config={"callbacks": callbacks})

            sorted_ranks = sorted(
                result.rankings, key=lambda x: x.relevance_score, reverse=True
            )

            top_ids = [r.id for r in sorted_ranks[: self.top_k]]
            final_docs = [candidates[i] for i in top_ids if i < len(candidates)]

            logger.info(
                f"Reranked: Kept {len(final_docs)} top docs out of {len(candidates)}."
            )

        except Exception as e:
            logger.error(
                f"Reranking failed: {e}. Falling back to top {self.top_k} raw results."
            )
            final_docs = candidates[: self.top_k]

        context_text = "\n\n".join([d.page_content for d in final_docs])
        chain = self.prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke(
            {"context": context_text, "question": query},
            config={"callbacks": callbacks},
        )

        return {
            "answer": answer_text,
            "source_documents": final_docs,
            "context_used": len(final_docs) > 0,
        }


class RouteChoice(BaseModel):
    choice: Literal["simple", "multi_query", "rrr", "rerank"] = Field(
        description="Choose the best strategy based on question complexity."
    )


class RouterRAGStrategy(RAGStrategy):
    def __init__(self, strategies: Dict[str, RAGStrategy], llm: DecoderClient):
        """This strategy asks to a decoder to choose the correct strategy
            based on the query complexity.
            TODO: refine this.

        Args:
            strategies (Dict[str, RAGStrategy]): available strategies to choose from.
            llm (DecoderClient): decoder client.
        """
        self.strategies = strategies
        self.llm = llm

    @log_execution(logger=logger)
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """See base class"""
        router_llm = self.llm.with_structured_output(RouteChoice)
        try:
            route = router_llm.invoke(
                f"Route this question: {query}", config={"callbacks": callbacks}
            )
            choice = route.choice
            logger.info(f"Routing to '{choice}' strategy.")
        except Exception:
            logger.info("Auto routing failing. Fallback to simple strategy.")

        return self.strategies[choice].answer(query)


class LangGraphStrategy(RAGStrategy):
    def __init__(self, retriever: VectorStoreRetriever, llm: DecoderClient):
        """Strategy that uses a graph to retrieve, ask and potentially refine the flow.
            TODO: increase capability of this strategy

        Args:
            retriever (VectorStoreRetriever): retriever client.
            llm (DecoderClient): decoder client

        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph not installed")
        self.app = build_rag_graph(llm, retriever)

    @log_execution(logger=logger)
    def answer(
        self, query: str, callbacks: LangchainCallbackHandler | None = None
    ) -> dict:
        """See base class"""

        inputs = {"question": query, "retry_count": 0}
        final_state = self.app.invoke(inputs, config={"callbacks": callbacks})

        return {
            "answer": final_state.get("generation", "No answer generated."),
            "source_documents": final_state.get("documents", []),
            "context_used": len(final_state.get("documents", [])) > 0,
        }
