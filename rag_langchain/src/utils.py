import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import argparse
from functools import wraps
import time

from langchain_postgres import PGVector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_milvus import Milvus
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_litellm import ChatLiteLLM
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI

VectorStoreClient = (
    PGVector | InMemoryVectorStore | Milvus | QdrantVectorStore | QdrantClient
)
EncoderClient = OpenAIEmbeddings
DecoderClient = ChatLiteLLM | ChatOpenAI

PROJECT_ROOT = Path(__file__).parent.parent


def get_model_classes(
    cfg: dict, llm_model: str | None = None, embedding_model: str | None = None
) -> tuple[DecoderClient, EncoderClient]:
    """Function for retrieving interfaces for decoder and
        encoder model, respectively.

    Args:
        cfg (dict): dictionary with config.
        llm_model (str | None): string with llm name. To override config.
        embedding_model (str | None): string with encoder name. To override config.

    Returns:
        tuple[DecoderClient, EncoderClient]: dec/enc clients
    """
    llm_model = llm_model if llm_model else cfg["llm"]["model"]
    use_litellm = cfg["services"]["litellm"]["start"]
    if use_litellm:
        litellm_failed = False
        try:
            llm = ChatLiteLLM(
                model=llm_model,
                temperature=cfg["llm"].get("temperature", 0),
                api_base=os.environ["MODEL_ENDPOINT_LITELLM"],
                api_key=os.environ["MODEL_API_KEY"],
            )
        except Exception as e:
            print(e)
            litellm_failed = True
    if not use_litellm or (use_litellm and litellm_failed):
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


def get_vectorstore(cfg: dict, embeddings: EncoderClient) -> VectorStoreClient:
    """Retrieve the client to connect to the chosen vector db

    Args:
        cfg (dict): dict with config.
        embeddings (EncoderClient): embedding client

    Returns:
        VectorStoreClient: db client, to be used for embedding and retrieval
    """
    # Read type from config, default to pgvector if missing
    v_type = cfg["vectorstore"].get("type", "pgvector").lower()
    cl = cfg["vectorstore"]["collection_name"]
    if v_type == "pgvector":
        return PGVector(
            embeddings=embeddings,
            collection_name=cl,
            connection=os.environ["PG_VECTOR_DB_URL_"],
            use_jsonb=True,
        )

    if v_type == "qdrant":
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")

        # NOTE: If prefer_grpc is True, port 6334 should be exposed
        client = QdrantClient(url=url, prefer_grpc=True)

        if not client.collection_exists(collection_name=cl):
            vec_size = len(embeddings.embed_query("find_dimensions"))
            client.create_collection(
                collection_name=cl,
                vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            )
        return QdrantVectorStore(
            client=client, embedding=embeddings, collection_name=cl
        )

    if v_type == "inmemory":
        return InMemoryVectorStore(embeddings=embeddings)

    if v_type == "milvus":
        return Milvus(
            embeddings=embeddings,
            collection_name=cfg["vectorstore"]["collection_name"],
            connection_args=cfg["vectorstore"]["connection_args"],
        )

    raise ValueError(f"Unknown vectorstore type: {v_type}")


def load_env_config() -> dict:
    """Function for leading .env and config

    Returns:
        dict: contains the elements of config
    """
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    with open(PROJECT_ROOT / "config.yml") as f:
        return yaml.safe_load(f)


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


def get_eval_model_classes(
    llm_model: str | None = None,
    embedding_model: str | None = None,
    async_mode: bool = True,
) -> tuple[DecoderClient, EncoderClient]:
    if not async_mode:
        llm_model = llm_model if llm_model else "gpt-4o-mini"
        client = AsyncOpenAI(
            api_key=os.environ["MODEL_API_KEY"],
            base_url=os.environ["MODEL_ENDPOINT"],
        )
        llm = llm_factory(llm_model, client=client)

        embedding_model = (
            embedding_model if embedding_model else "text-embedding-3-small"
        )
        embeddings_cl = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=os.environ["EMBEDDING_ENDPOINT"],
            openai_api_key=os.environ["EMBEDDING_API_KEY"],
            tiktoken_enabled=False, # important parameter since we are not using openai models
            check_embedding_ctx_length=False, # important parameter since we are not using openai models
        )
        embeddings = embedding_factory(
            "openai", model=embedding_model, client=embeddings_cl
        )

        return llm, embeddings

    # async mode
    client = AsyncOpenAI(
        api_key=os.environ["MODEL_API_KEY"],
        base_url=os.environ["MODEL_ENDPOINT"],
    )
    llm_model = llm_model if llm_model else "gpt-4o-mini"
    llm = llm_factory(model=llm_model, client=client)

    embedding_model = embedding_model if embedding_model else "text-embedding-3-small"

    lc_embeddings = AsyncOpenAI(
        api_key=os.environ["EMBEDDING_API_KEY"],
        base_url=os.environ["EMBEDDING_ENDPOINT"],
    )

    embeddings = embedding_factory(
        "openai", model=embedding_model, client=lc_embeddings
    )

    return llm, embeddings


def log_execution(logger):
    """Decorator factory to log function name and execution time."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            strategy_name = args[0].__class__.__name__
            logger.info(f"Starting {strategy_name} for query...")
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{strategy_name} finished in {duration:.2f}s")
            return result

        return wrapper

    return decorator
