
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFium2Loader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from strategies import (
    SimpleRAGStrategy,
    RRRStrategy,
    MultiQueryRAGStrategy,
    RouterRAGStrategy,
    RerankRAGStrategy,
)
from chunking import LLMSemanticChunker, QAChunking, TableOfContentsChunker
from utils import get_model_classes, get_vectorstore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent



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

def rag_ingest(cfg, llm, vectorstore):
    logger.info("Starting ingestion...")
    raw_docs, docs = load_split_documents(
        cfg=cfg, 
        llm=llm, 
        pdf_read_mode=cfg["pdf_read_mode"],
        loader="pypdfium2"
    )
    logger.info(f"Loaded {len(raw_docs)} docs → {len(docs)} chunks")

    vectorstore.add_documents(documents=docs)
    logger.info(f"Ingestion complete. Added {len(docs) = } to vectorstore.")

def prompt_template():
    return ChatPromptTemplate.from_template(
        """Answer the question enclosed in parentheses based on the context provided.
        If the context is irrelevant or empty, answer based on your own knowledge 
        but mention that the provided context was insufficient.

        Context: {context}
        \n\n\n
        Question: {question}
        """
    )

def get_strategy_instances(cfg, llm, vectorstore):
    # Collect all available strategies here. If user passes auto mode,
    # the llm will pick the best strategy based on question complexity.
    k = cfg["vectorstore"]["search_kwargs"].get("k", 4)
    logger.info(f"Using top-{k} documents for retrieval.")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    common = {"retriever": retriever, "llm": llm, "prompt": prompt_template()}
    strategy_instances = {
        "simple": SimpleRAGStrategy(**common),
        "rrr": RRRStrategy(**common, cache=not cfg["no_cache"]),
        "multi_query": MultiQueryRAGStrategy(**common),
        "rerank": RerankRAGStrategy(**common),
    }

    logger.info(f"RAG Mode: {cfg['rag_mode'].upper()}")

    if cfg["rag_mode"] == "auto":
        rag = RouterRAGStrategy(strategies=strategy_instances, llm=llm)
    else:
        # Check if the user requested a mode that isn't instantiated
        if cfg["rag_mode"] not in strategy_instances:
            logger.warning(f"Mode {cfg['rag_mode']} not found, defaulting to simple")
            rag = strategy_instances["simple"]
        else:
            rag = strategy_instances[cfg["rag_mode"]]

    return rag

def rag_system(cfg):

    llm, embeddings = get_model_classes(cfg)

    vectorstore = get_vectorstore(cfg, embeddings)

    if cfg["ingest"]:
        rag_ingest(cfg, llm, vectorstore)
    
    rag = get_strategy_instances(cfg, llm, vectorstore=vectorstore)

    return rag