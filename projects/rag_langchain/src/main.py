import os
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.indexes import SQLRecordManager, index

from pgvector_utils import check_pgvector_running
from rag import (
    SimpleRAGStrategy,
    RRRStrategy,
    MultiQueryRAGStrategy,
)


PROJECT_ROOT = Path(__file__).parent.parent


def load_env_config():
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    with open(PROJECT_ROOT / "config.yml") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="PDF RAG pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rag-mode",
        choices=["simple", "rrr", "multi_query"],
        default="simple",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable rewrite caching (RRR only)",
    )
    return parser.parse_args()


def load_split_documents(pdf_dirs, cfg):
    raw_docs = []
    for d in pdf_dirs:
        raw_docs.extend(PyPDFDirectoryLoader(d).load())
    
    splitter = RecursiveCharacterTextSplitter(**cfg)
    docs = splitter.split_documents(raw_docs)
    
    # ensure metadata
    for d in docs:
        d.metadata.setdefault(
            "source",
            d.metadata.get("source_file", f"chunk_{id(d)}"),
        )
    return raw_docs, docs

    
def main():
    cfg = load_env_config()
    
    args = parse_args()
    
    check_pgvector_running(start_if_missing=False)

    raw_docs, docs = load_split_documents(cfg["PDF_DIRS"], cfg["text_splitter"])

    print(f"Loaded {len(raw_docs)} docs → {len(docs)} chunks")

    embeddings = OpenAIEmbeddings(
        model=cfg["embeddings"]["model"],
        openai_api_base=os.environ["EMBEDDING_ENDPOINT"],
        openai_api_key=os.environ["EMBEDDING_API_KEY"],
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=cfg["vectorstore"]["collection_name"],
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
    )

    record_manager = SQLRecordManager(
        "pdf_docs_namespace",
        db_url=os.environ["DATABASE_URL"],
    )
    record_manager.create_schema()

    index(
        docs,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source",
        key_encoder="sha256",
    )

    retriever = vectorstore.as_retriever(
        search_kwargs=cfg["vectorstore"]["search_kwargs"]
    )

    llm = ChatOpenAI(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"].get("temperature", 0),
        openai_api_base=os.environ["MODEL_ENDPOINT"],
        openai_api_key=os.environ["MODEL_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_template(
        """ Answer the question enclosed in () using your knowlodge and the context in <<< >>>.
            If the provided context is relevant, use it to enrich your knowledge
            and and append to your answer the string "CTX_USED". 
            If not relevant, ignore the context and answer the question with your 
            knowledge only and append to your answer the string "CTX_NOT_USED".

            <<<
            {context}
            >>>

            Question: ({question})
            """
    )

    strategies = {
        "simple": SimpleRAGStrategy,
        "rrr": lambda **kw: RRRStrategy(cache=not args.no_cache, **kw),
        "multi_query": MultiQueryRAGStrategy,
    }

    rag = strategies[args.rag_mode](
        retriever=retriever,
        llm=llm,
        prompt=prompt,
    )

    while True:
        q = input("\nAsk a question (q to quit): ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        answer = rag.answer(q)
        print("\nAnswer:\n", answer.content)


if __name__ == "__main__":
    main()
