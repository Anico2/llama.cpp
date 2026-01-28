import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import argparse


from langchain_postgres import PGVector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_litellm import ChatLiteLLM
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI


PROJECT_ROOT = Path(__file__).parent.parent


def get_model_classes(cfg, llm_model=None, embedding_model=None):
    llm_model = llm_model if llm_model else cfg["llm"]["model"]
    if cfg["litellm"]["enabled"]:
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
    if not cfg["litellm"]["enabled"] or (cfg["litellm"]["enabled"] and litellm_failed):
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


def get_vectorstore(cfg, embeddings, vectorstore_type="pgvector"):
    if vectorstore_type == "pgvector":
        return PGVector(
            embeddings=embeddings,
            collection_name=cfg["vectorstore"]["collection_name"],
            connection=os.environ["PG_VECTOR_DB_URL_"],
            use_jsonb=True,
        )
    if vectorstore_type == "inmemory":
        return InMemoryVectorStore(embeddings=embeddings)

    if vectorstore_type == "milvus":
        return Milvus(
            embeddings=embeddings,
            collection_name=cfg["vectorstore"]["collection_name"],
            connection_args=cfg["vectorstore"]["connection_args"],
        )


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


def get_eval_model_classes(llm_model=None, embedding_model=None, async_mode=True):
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
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
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
