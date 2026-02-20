import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
import argparse
import datetime

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


def parse_args_fn() -> dict:
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


def get_application_config(parse_cli_args: bool = True, overrides: dict = None) -> dict:
    """
    Centralizes configuration setup.
    
    Args:
        parse_cli_args (bool): If True, parses sys.argv (used in main.py). 
                               If False, uses defaults (used in MCP/Streamlit).
        overrides (dict): Manual dictionary to override config (e.g. for MCP).
    """
    cfg = load_env_config()
    

    if parse_cli_args:
        args = parse_args_fn()
        cfg.update(vars(args))
    else:
        # Default values
        defaults = {
            "task": "rag",
            "ingest": False,
            "rag_mode": "auto",
            "no_cache": False,
            "interactive": False
        }
        cfg.update(defaults)

    if overrides:
        cfg.update(overrides)

    # mlflow experiment
    datatime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = cfg["services"]["mlflow"]["experiment_name"]
    if experiment == "":
        cfg["mlflow_experiment"] = f"{cfg['task']}_{datatime_str}"
    else:
        cfg["mlflow_experiment"] = f"{experiment}"
    
    return cfg