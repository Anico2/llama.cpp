import sys
import logging
import asyncio
import datetime

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langfuse.langchain.CallbackHandler import LangchainCallbackHandler

from chb.evals.eval_ragas import eval_ragas_main
from chb.evals.eval_mlflow import eval_mlflow_main
from chb.engine.rag import rag_system
from chb.services.services import services_handler
from chb.utils.params import get_application_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main_rag(cfg: dict):
    
    try:
        langfuse = get_client()
        if not langfuse.auth_check():
            raise Exception

        langfuse_handler: LangchainCallbackHandler = CallbackHandler()
        logger.info("Langfuse OK")
    except Exception as e:
        logger.error(f"Failed to use Langfuse: {e}")
        langfuse_handler = None
    rag = rag_system(cfg)

    print(f"\n--- RAG Pipeline Ready [{cfg['rag_mode']}] ---")

    if not cfg["interactive"]:
        return rag.answer(cfg["query"], callbacks=[langfuse_handler])

    while True:
        try:
            q = input("\nAsk a question (q to terminate): ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if not q:
                continue

            result = rag.answer(query=q, callbacks=[langfuse_handler])

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

    cfg = get_application_config(parse_cli_args=True) 
    breakpoint()
    # NOTE: services can be started/stoppend singularly and stopped all together
    srv = cfg["services"]
    with services_handler(
        cfg,
        stop_all=srv["stop_all"],
        stop_langfuse=srv["langfuse"]["stop"],
        stop_mlflow=srv["mlflow"]["stop"],
        stop_llama_server=srv["llamacpp"]["stop"],
        stop_pgvector=srv["pgvector"]["stop"],
        stop_redis=srv["redis"]["stop"],
        stop_qdrant=srv["qdrant"]["stop"]
    ):
        if cfg["task"] == "rag":
            cfg["interactive"] = True
            # TODO: make asyncronous as eval
            main_rag(cfg)
            sys.exit(0)

        assert cfg["task"] in ["eval[ragas]", "eval[mlflow]"]
        if cfg["task"] == "eval[ragas]":
            res = asyncio.run(eval_ragas_main(cfg))
            res.save()
        elif cfg["task"] == "eval[mlflow]":
            res = eval_mlflow_main(cfg)
            print(res)
        
        sys.exit(0)