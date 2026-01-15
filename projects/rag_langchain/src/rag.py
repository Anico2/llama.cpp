from abc import ABC, abstractmethod
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


class BaseRAGStrategy(ABC):
    @abstractmethod
    def answer(self, query: str):
        ...



class SimpleRAGStrategy(BaseRAGStrategy):
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def answer(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        formatted_prompt = self.prompt.invoke({"context": docs, "question": query})
        return self.llm.invoke(formatted_prompt)

class RRRStrategy(BaseRAGStrategy):
    def __init__(self, retriever, llm, prompt, cache=True):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.cache_rewrites = cache
        self._rewrite_cache = {}
        self._answer_cache = {}

        self.rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite the following question into a concise search query suitable "
            "for retrieving relevant documents. End the query with '###'.\n"
            "Question: {x}"
        )
        self.rewriter_chain = self.rewrite_prompt | self.llm | self.parse_rewriter_output

    def parse_rewriter_output(self, message):
        return message.content.strip('"').strip("###")

    def answer(self, query: str) -> str:
        if query in self._answer_cache:
            return self._answer_cache[query]

        if self.cache_rewrites and query in self._rewrite_cache:
            rewritten_query = self._rewrite_cache[query]
            print("Using cached rewritten query:", rewritten_query)
        else:
            rewritten_query = self.rewriter_chain.invoke(query)
            if self.cache_rewrites:
                self._rewrite_cache[query] = rewritten_query
            print("Rewritten query:", rewritten_query)

        docs = self.retriever.invoke(rewritten_query)
        formatted_prompt = self.prompt.invoke({"context": docs, "question": query})
        answer = self.llm.invoke(formatted_prompt)

        if self.cache_rewrites:
            self._answer_cache[query] = answer

        return answer

class MultiQueryRAGStrategy(BaseRAGStrategy):
    def __init__(self, retriever, llm, prompt, num_queries=5):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.num_queries = num_queries

        self.perspectives_prompt = ChatPromptTemplate.from_template(
            f"""You are an AI language model assistant. Your task is to generate {self.num_queries} different versions 
            of the given user question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome 
            some of the limitations of distance-based similarity search. 
            Provide these alternative questions separated by newlines. 
            Original question: {{question}}"""
        )

    def parse_queries_output(self, message):
        return [q.strip() for q in message.content.split("\n") if q.strip()]

    def get_unique_union(self, document_lists):
        deduped_docs = {doc.page_content: doc for sublist in document_lists for doc in sublist}
        return list(deduped_docs.values())

    def answer(self, query: str) -> str:
        query_gen = self.perspectives_prompt | self.llm | self.parse_queries_output
        queries = query_gen.invoke(query)
        doc_lists = [self.retriever.invoke(q) for q in queries]
        combined_docs = self.get_unique_union(doc_lists)
        formatted_prompt = self.prompt.invoke({"context": combined_docs, "question": query})
        return self.llm.invoke(formatted_prompt)
