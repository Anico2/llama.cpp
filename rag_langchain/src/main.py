import os
import yaml
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFium2Loader
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
    RouterRAGStrategy 
)
from chunking import LLMSemanticChunker

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-Main")

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
        "--ingest",
        action="store_true",
        help="Run document ingestion/indexing before querying.",
    )
    parser.add_argument(
        "--rag-mode",
        choices=["simple", "rrr", "multi_query", "rerank", "auto"],
        default="simple",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable rewrite caching (RRR only)",
    )
    return parser.parse_args()

def load_split_documents(pdf_dirs, cfg, llm):
    raw_docs = []
    for d in pdf_dirs:
        path = Path(d)
        if not path.exists(): continue
        for pdf_file in path.rglob("*.pdf"):
            try:
                loader = PyPDFium2Loader(str(pdf_file))
                raw_docs.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")

    chunk_strategy = cfg.get("chunk_strategy", "recursive")
    
    if chunk_strategy == "semantic":
        logger.info("Using LLM-Native Semantic Chunking (slower)")
        splitter = LLMSemanticChunker(llm=llm)
    else:
        logger.info("Using Recursive Character Chunking...")
        splitter = RecursiveCharacterTextSplitter(**cfg["text_splitter"])

    docs = splitter.split_documents(raw_docs)

    for d in docs:
        d.metadata.setdefault("source", d.metadata.get("source", "unknown"))
        
    return raw_docs, docs

def main():
    cfg = load_env_config()
    args = parse_args()
    
    check_pgvector_running(start_if_missing=False)

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

    llm = ChatOpenAI(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"].get("temperature", 0),
        openai_api_base=os.environ["MODEL_ENDPOINT"],
        openai_api_key=os.environ["MODEL_API_KEY"],
    )

    # For efficiency, we optionally skip ingestion if not needed
    if args.ingest:
        logger.info("Starting ingestion...")
        raw_docs, docs = load_split_documents(cfg["PDF_DIRS"], cfg, llm)
        logger.info(f"Loaded {len(raw_docs)} docs → {len(docs)} chunks")

        record_manager = SQLRecordManager(
            f"pdf_namespace_{cfg['vectorstore']['collection_name']}",
            db_url=os.environ["DATABASE_URL"],
        )
        record_manager.create_schema()

        indexing_result = index(
            docs,
            record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="source",
            key_encoder="sha256",
        )
        logger.info(f"Indexing Summary: {indexing_result}")

    
    retriever = vectorstore.as_retriever(
        search_kwargs=cfg["vectorstore"]["search_kwargs"]
    )
    
    # General RAG prompt
    prompt = ChatPromptTemplate.from_template(
        """Answer the question enclosed in parentheses based on the context provided.
        If the context is irrelevant or empty, answer based on your own knowledge 
        but mention that the provided context was insufficient.

        Context:
        <<<
        {context}
        >>>

        Question: ({question})
        """
    )

    # Collect all available strategies here. If user passes auto mode,
    # the llm will pick the best strategy based on question complexity.
    strategy_instances = {
        "simple": SimpleRAGStrategy(retriever, llm, prompt),
        "rrr": RRRStrategy(retriever, llm, prompt, cache=not args.no_cache),
        "multi_query": MultiQueryRAGStrategy(retriever, llm, prompt),
    }

    logger.info(f"RAG Mode: {args.rag_mode.upper()}")
    if args.rag_mode == "auto":
        rag = RouterRAGStrategy(strategies=strategy_instances, llm=llm)
    else:
        rag = strategy_instances[args.rag_mode]

    print(f"\n--- RAG Pipeline Ready [{args.rag_mode}] ---")
    
    while True:
        try:
            q = input("\nAsk a question (q to terminate): ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if not q: 
                continue

            result = rag.answer(q)
            
            print("\n" + "="*20 + " ANSWER " + "="*20)
            print(result["answer"])
            print("="*48)
            
            if result.get("context_used"):
                print("Unique sources:")
                
                seen_sources = set()
                for doc in result["source_documents"]:
                    src = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    source_line = f" - {src} (Page {page})"
                    if source_line not in seen_sources:
                        print(source_line)
                        seen_sources.add(source_line)
            else:
                print("Note: No relevant documents found in the database.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during query: {e}")

if __name__ == "__main__":
    main()