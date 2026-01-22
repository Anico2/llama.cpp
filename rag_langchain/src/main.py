import os
import yaml
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFium2Loader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pgvector_utils import check_pgvector_running
from rag import (
    SimpleRAGStrategy,
    RRRStrategy,
    MultiQueryRAGStrategy,
    RouterRAGStrategy,
    RerankRAGStrategy,
)
from chunking import LLMSemanticChunker, QAChunking, TableOfContentsChunker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAG-Main")

PROJECT_ROOT = Path(__file__).parent.parent

def load_env_config():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    with open(PROJECT_ROOT / "config.yml") as f:
        return yaml.safe_load(f)

def get_model_classes(cfg, llm_model=None, embedding_model=None):
    llm_model = llm_model if llm_model else cfg["llm"]["model"]
    llm = ChatOpenAI(
        model=llm_model,
        temperature=cfg["llm"].get("temperature", 0),
        openai_api_base=os.environ["MODEL_ENDPOINT"],
        openai_api_key=os.environ["MODEL_API_KEY"],
    )
    embedding_model = embedding_model if embedding_model else cfg["embeddings"]["model"]
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=os.environ["EMBEDDING_ENDPOINT"],
        openai_api_key=os.environ["EMBEDDING_API_KEY"],
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )

    return llm, embeddings


def get_vectorstore(cfg, embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=cfg["vectorstore"]["collection_name"],
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
    )
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
        "--pdf-read-mode",
        choices=["single", "page"],
        default="page",
        help="PDF reading mode: 'single' reads entire PDF as one document; 'page' reads each page separately.",
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

def load_split_documents(
     cfg, llm, pdf_read_mode: str = "page", loader="pdfplumber"
):
    assert pdf_read_mode in ["single", "page"]

    if pdf_read_mode == "single":
        delim = "\n-------CUSTOM_END-------\n"
    else:
        delim = None

    assert loader in ["pdfplumber", "pypdfium2"]

    raw_docs = []
    for d in cfg["PDF_DIRS"]:
        path = Path(d)
        if not path.exists():
            continue
        for pdf_file in path.rglob("*.pdf"):
            try:
                if loader == "pdfplumber":
                    loader = PDFPlumberLoader(str(pdf_file))
                else:
                    loader = PyPDFium2Loader(
                        str(pdf_file), mode=pdf_read_mode, pages_delimiter=delim
                    )
                # more on https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfium2

                raw_docs.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")

    chunk_strategy = cfg.get("chunk_strategy", "recursive")

    assert chunk_strategy in ["qa", "recursive", "semantic", "toc"]
    if chunk_strategy == "semantic":
        logger.info("Using LLM-Native Semantic Chunking (slower)")
        splitter = LLMSemanticChunker(llm=llm)
    elif chunk_strategy == "toc":
        logger.info("Using Table of Contents Chunking")
        splitter = TableOfContentsChunker(llm=llm, delim=delim)
    elif chunk_strategy == "qa":
        splitter = QAChunking(llm=llm)
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

    llm, embeddings = get_model_classes(cfg)

    vectorstore = get_vectorstore(cfg, embeddings)

    # For efficiency, we optionally skip ingestion if not needed
    if args.ingest:
        logger.info("Starting ingestion...")
        raw_docs, docs = load_split_documents(
            cfg=cfg, 
            llm=llm, 
            pdf_read_mode=args.pdf_read_mode,
            loader="pypdfium2"
        )
        logger.info(f"Loaded {len(raw_docs)} docs → {len(docs)} chunks")

        vectorstore.add_documents(documents=docs)
        logger.info(f" added {len(docs) = }")

    search_kwargs = cfg["vectorstore"]["search_kwargs"]
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
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
    common = {"retriever": retriever, "llm": llm, "prompt": prompt}
    strategy_instances = {
        "simple": SimpleRAGStrategy(**common),
        "rrr": RRRStrategy(**common, cache=not args.no_cache),
        "multi_query": MultiQueryRAGStrategy(**common),
        "rerank": RerankRAGStrategy(**common),
    }

    logger.info(f"RAG Mode: {args.rag_mode.upper()}")

    if args.rag_mode == "auto":
        rag = RouterRAGStrategy(strategies=strategy_instances, llm=llm)
    else:
        # Check if the user requested a mode that isn't instantiated
        if args.rag_mode not in strategy_instances:
            logger.warning(f"Mode {args.rag_mode} not found, defaulting to simple")
            rag = strategy_instances["simple"]
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

            print("\n" + "=" * 20 + " ANSWER " + "=" * 20)
            print(result["answer"])
            print("=" * 50)

            if result.get("context_used"):
                print("Sources:")
                for doc in result["source_documents"]:
                    print(f"""{doc.metadata.get("source", "") = }""")
                    print(f"""{doc.metadata.get("page", "") = }""")
                    print(f"""{doc.metadata.get("toc_start_index", "") = }""")
                    print(f"""{doc.metadata.get("toc_end_index", "") = }""")
                    print(f"""{doc.metadata.get("section_length_chars", "") = }""")
                    print(f"""{doc.metadata.get("section_num_pages", "") = }""")
                    print("----")
            else:
                print("Note: No relevant documents found in the database.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during query: {e}")

if __name__ == "__main__":
    main()