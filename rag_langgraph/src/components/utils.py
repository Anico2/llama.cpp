import os
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


logger = logging.getLogger(Path(__file__).name.removesuffix(".py"))

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(override=True)
with open(PROJECT_ROOT / "config.yml") as f:
        cfg = yaml.safe_load(f)

llm = ChatOpenAI(
    base_url=os.environ["MODEL_ENDPOINT"],
    api_key=os.environ["MODEL_API_KEY"],
    model=cfg["llm"]["model"],
    temperature=0,
)

embeddings_model = OpenAIEmbeddings(
        model=cfg["embeddings"]["model"],
        openai_api_base=os.environ["EMBEDDING_ENDPOINT"],
        openai_api_key=os.environ["EMBEDDING_API_KEY"],
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )

def get_vectorstore():
    """Returns the PGVector store instance."""
    
    return PGVector(
        embeddings=embeddings_model,
        collection_name="agentic_documents",
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
    )

def get_retriever(k: int = 3):
    """Returns the retriever interface for the graph."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})

def ingest_documents(pdf_directory: str):
    """Loads PDFs, splits them, and indexes them into PGVector."""
    logger.info(f"Scanning directory: {pdf_directory}")
    
    docs = []
    if not os.path.exists(pdf_directory):
        logger.error(f"Directory not found: {pdf_directory}")
        raise FileNotFoundError(f"Directory not found: {pdf_directory}")

    
    for root, _, files in os.walk(pdf_directory):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            file_path = os.path.join(root, file)
            try:
                loader = PyPDFium2Loader(file_path)
                docs.extend(loader.load())
                logger.info(f"Loaded: {file}")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")

    if not docs:
        logger.error("No documents found to ingest.")
        raise ValueError("No documents found to ingest.")
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["text_splitter"]["chunk_size"], 
        chunk_overlap=cfg["text_splitter"]["chunk_overlap"]
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} docs into {len(splits)} chunks.")

    # Index
    logger.info("Indexing chunks into PGVector...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(splits)
    logger.info("Ingestion complete.")