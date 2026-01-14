import os
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv
from abc import ABC, abstractmethod

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.indexes import SQLRecordManager, index
from pgvector_utils import check_pgvector_running

pr = Path(__file__).parent.parent
load_dotenv(pr / ".env", override=True)

with open(pr / "config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

check_pgvector_running(start_if_missing=False)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--rag_mode",
    type=str,
    choices=["simple", "rrr", "multi_query"],
    default="simple",
    help="RAG mode to use"
)
args = parser.parse_args()
rag_mode = args.rag_mode
print(f"RAG mode set to: {rag_mode}")


raw_documents = []
for pdf_dir in CONFIG["PDF_DIRS"]:
    loader = PyPDFDirectoryLoader(pdf_dir)
    raw_documents.extend(loader.load())

ts_config = CONFIG["text_splitter"]
text_splitter = RecursiveCharacterTextSplitter(
    separators=ts_config["separators"],
    chunk_size=ts_config["chunk_size"],
    chunk_overlap=ts_config["chunk_overlap"],
    add_start_index=True
)
documents = text_splitter.split_documents(raw_documents)
print(f"Loaded {len(raw_documents)} documents and created {len(documents)} chunks")


embeddings_model = OpenAIEmbeddings(
    model=CONFIG["embeddings"]["model"],
    openai_api_base=os.environ.get("EMBEDDING_ENDPOINT"),
    openai_api_key=os.environ.get("EMBEDDING_API_KEY"),
    chunk_size=ts_config["chunk_size"],
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
)

vs_config = CONFIG["vectorstore"]
collection_name = vs_config["collection_name"]
connection = os.environ.get("DATABASE_URL")
namespace = "pdf_docs_namespace"

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,  # recommended for metadata storage
)

record_manager = SQLRecordManager(namespace, db_url=connection)
record_manager.create_schema()  # ensures table exists

# Add unique source metadata if missing (needed for incremental indexing)
for doc in documents:
    if "source" not in doc.metadata:
        # You can use original file path or chunk index
        doc.metadata["source"] = doc.metadata.get("source_file", f"chunk_{id(doc)}")

# Incremental indexing: only new or updated documents are added
index(
    documents,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
    key_encoder="sha256"
)

retriever = vectorstore.as_retriever(search_kwargs=vs_config["search_kwargs"])


llm_config = CONFIG["llm"]
llm = ChatOpenAI(
    model=llm_config["model"],
    temperature=llm_config.get("temperature", 0),
    openai_api_base=os.environ.get("MODEL_ENDPOINT"),
    openai_api_key=os.environ.get("MODEL_API_KEY"),
)

qa_prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the following context (if the context is not useful, say 'I don't know'):
{context}

Question: {question}"""
)


class BaseRAGStrategy(ABC):
    @abstractmethod
    def answer(self, query: str) -> str:
        pass


class RRRStrategy(BaseRAGStrategy):
    def __init__(self, retriever, llm, prompt, cache_rewrites=True):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.cache_rewrites = cache_rewrites
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


class SimpleRAGStrategy(BaseRAGStrategy):
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def answer(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        formatted_prompt = self.prompt.invoke({"context": docs, "question": query})
        return self.llm.invoke(formatted_prompt)


RAG_STRATEGIES = {
    "simple": SimpleRAGStrategy,
    "rrr": RRRStrategy,
    "multi_query": MultiQueryRAGStrategy,
}


strategy_class = RAG_STRATEGIES[rag_mode]
strategy = strategy_class(retriever=retriever, llm=llm, prompt=qa_prompt)

while True:
    query = input("\nAsk a question (or 'exit'/'quit'/'q'): ")
    if query.lower() in ["exit", "quit", "q"]:
        break
    result = strategy.answer(query)
    print("\nAnswer:\n", result.content if hasattr(result, "content") else result)
