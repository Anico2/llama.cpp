import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
import argparse

from langchain_postgres import PGVector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_milvus import Milvus
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_litellm import ChatLiteLLM

VectorStoreClient = (
    PGVector | InMemoryVectorStore | Milvus | QdrantVectorStore | QdrantClient
)
EncoderClient = OpenAIEmbeddings
DecoderClient = ChatLiteLLM | ChatOpenAI

def load_env_config() -> dict:
    """Function for leading .env and config

    Returns:
        dict: contains the elements of config
    """
    pr = Path(os.environ["PROJECT_ROOT"])
    load_dotenv(pr / ".env", override=True)
    with open(pr / "config/config.yml") as f:
        cfg = yaml.safe_load(f)

    return cfg


def parse_args() -> dict:
    """Parser

    Returns:
        dict: parsed input parameters
    """
    parser = argparse.ArgumentParser(
        description="PDF RAG pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["rag", "eval[ragas]", "eval[mlflow]"],
        default="rag",
        help="Task to perform: 'rag' for retrieval-augmented generation, 'eval[ragas]' for RAGas evaluation, 'eval[mlflow]' for MLflow evaluation.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion/indexing before querying.",
    )
    parser.add_argument(
        "--rag-mode",
        choices=[
            "graph",
            "simple",
            "rrr",
            "multi_query",
            "rerank",
            "auto",
        ],
        default="simple",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable rewrite caching (RRR only)",
    )
    return parser.parse_args()


