import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFium2Loader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents.base import Document as LCDocument

from chb.engine.strategies import (
    RAGStrategy,
    SimpleRAGStrategy,
    RRRStrategy,
    MultiQueryRAGStrategy,
    RouterRAGStrategy,
    RerankRAGStrategy,
    LangGraphStrategy,
    CachedRAGStrategy,
)
from chb.ingestion.chunking import LLMSemanticChunker, QAChunking, TableOfContentsChunker
from chb.utils.clients import (
    get_model_classes,
    get_vectorstore,
    VectorStoreClient,
    EncoderClient,
    DecoderClient,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_split_documents(cfg: dict, llm: DecoderClient) -> tuple[list[LCDocument], list[LCDocument]]:
    """Load and split documents. Currently only pdf are supported

    Args:
        cfg (dict): dictionary with config
        llm (DecoderClient): llm client

    Returns:
        tuple[list[LCDocument], list[LCDocument]]: list with raw docs, list with splitted docs
    """
    loader, pdf_read_mode = cfg["pdf"]["loader"], cfg["pdf"]["read_mode"]
    assert pdf_read_mode in ["single", "page"], pdf_read_mode
    assert loader in ["pdfplumber", "pypdfium2"], loader

    delim = cfg["pdf"]["delim"]
    if delim == "default":
        delim = None

    raw_docs = []
    for d in cfg["pdf"]["paths"]:
        path = Path(d)
        if not path.exists():
            continue

        raw_docs += pdf_loader(
            path=path, loader=loader, pdf_read_mode=pdf_read_mode, delim=delim
        )

    if not raw_docs:
        raise ValueError("No documents found.")

    # Use chosen strategy
    chunk_strategy = cfg["pdf"]["chunk_strategy"]

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


def pdf_loader(
    path: str, loader: str, pdf_read_mode: str, delim: str | None = None
) -> list:
    """Pdf loader

    Args:
        path (str): path to search for pdf
        loader (str): specific loader to use
        pdf_read_mode (str): read_mode, either by page or as a single text file.
        delim (str | None): delim (useful for 'single' read mode)

    Returns:
        list: list with loaded docs. If single, each document is represent as on LCDocument only.
    """
    raw_docs_ = []
    for pdf_file in path.rglob("*.pdf"):
        try:
            if loader == "pdfplumber":
                loader = PDFPlumberLoader(str(pdf_file))
            else:
                loader = PyPDFium2Loader(
                    str(pdf_file), mode=pdf_read_mode, pages_delimiter=delim
                )
            # more on https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfium2

            raw_docs_.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")

    return raw_docs_


def rag_ingest(cfg: dict, llm: DecoderClient, vectorstore: VectorStoreClient) -> None:
    """Ingestion entry point

    Args:
        cfg (dict): dictionary with config
        llm (DecoderClient): decoder client
        vectorstore (VectorStoreClient): vector store client

    Returns: None
    """
    logger.info("Starting ingestion")
    raw_docs, docs = load_split_documents(cfg=cfg, llm=llm)
    logger.info(f"Loaded {len(raw_docs)} docs → {len(docs)} chunks")
    breakpoint()
    # If using Qdrant, we might want to force recreation if strictly ingesting fresh
    # But standard add_documents works if the store is already initialized in get_vectorstore
    try:
        vectorstore.add_documents(documents=docs)
        logger.info(f"Ingestion complete. Added {len(docs) = } to vectorstore.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")


def prompt_template() -> ChatPromptTemplate:
    """Function for applying template. This should also work as a system message.

    Returns: ChatPromptTemplate
    """
    return ChatPromptTemplate.from_template(
        """Answer the question based on the context provided.
        If the context is irrelevant or empty, answer based on your own knowledge 
        but mention that the provided context was insufficient.

        Context: {context}
        \n\n\n
        Question: "{question}"
        """
    )


def get_strategy_instances(
    cfg: dict,
    llm: DecoderClient,
    embeddings: EncoderClient,
    vectorstore: VectorStoreClient,
) -> RAGStrategy:
    """Routing for retrieving the strategy instance

    Args:
        cfg (dict): dictionary with config
        llm (DecoderClient): decoder client
        embeddings (EncoderClient): encoder client
        vectorstore (VectorStoreClient): vector store client

    Returns:
        RAGStrategy: instance of the chose strategy
    """

    k = cfg["vectorstore"]["search_kwargs"]["k"]
    logger.info(f"Using top-{k} documents for retrieval.")
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": k})

    logger.info(f"RAG Mode: {cfg['rag_mode'].upper()}")

    selected_strategy = None
    if cfg["rag_mode"] == "graph":
        selected_strategy = LangGraphStrategy(retriever=retriever, llm=llm)

    else:
        common = {"retriever": retriever, "llm": llm, "prompt": prompt_template()}
        strategy_instances = {
            "simple": SimpleRAGStrategy(**common),
            "rrr": RRRStrategy(**common, cache=not cfg["no_cache"]),
            "multi_query": MultiQueryRAGStrategy(**common),
            "rerank": RerankRAGStrategy(**common),
        }
        if cfg["rag_mode"] == "auto":
            selected_strategy = RouterRAGStrategy(
                strategies=strategy_instances, llm=llm
            )
        else:
            selected_strategy = strategy_instances[cfg["rag_mode"]]

    # Semantic Caching Layer
    cache_cfg = cfg.get("semantic_caching", {})
    if cache_cfg.get("enabled", False):
        logger.info("Enabling Semantic Caching (Redis).")
        threshold = cache_cfg.get("threshold", 0.9)
        return CachedRAGStrategy(
            base_strategy=selected_strategy, embeddings=embeddings, threshold=threshold
        )

    return selected_strategy


def rag_system(cfg: dict) -> RAGStrategy:
    """Entry point of the whole rag system

    Args:
        cfg (dict): dictionary with config

    Returns:
        RAGStrategy: instance of the chose strategy
    """
    # retrieve encoder and decoder
    clients = get_model_classes(cfg)

    llm: DecoderClient = clients[0]
    embeddings: EncoderClient = clients[1]

    # retrieve the client of the chosen db
    vectorstore: VectorStoreClient = get_vectorstore(cfg=cfg, embeddings=embeddings)

    if cfg["ingest"]:
        # It means that we have either create or update the documents collection
        rag_ingest(cfg=cfg, llm=llm, vectorstore=vectorstore)

    return get_strategy_instances(
        cfg=cfg, llm=llm, embeddings=embeddings, vectorstore=vectorstore
    )
