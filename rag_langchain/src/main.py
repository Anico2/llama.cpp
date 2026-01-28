import sys
import logging
import asyncio
import datetime

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from eval_ragas import eval_ragas_main
from eval_mlflow import eval_mlflow_main
from rag import rag_system
from services import services_handler
from utils import load_env_config, parse_args

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main_rag(cfg):
    try:
        _ = get_client()
        langfuse_handler = CallbackHandler()
    except Exception as e:
        print(e)
        logger.error("Failed to use Langfuse.")
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

            result = rag.answer(q, callbacks=[langfuse_handler])

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
    cfg, args = load_env_config(), parse_args()
    cfg = {**cfg, **vars(args)}  # put evertyhing in one dict only

    datatime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["experiment"] = f"{cfg['task']}_{datatime_str}"

    with services_handler(
        cfg,
        suppress_out=True,
        stop_all=True,
        stop_langfuse=False,
        stop_mlflow=False,
        stop_llama_server=False,
        stop_pgvector=False
    ):
        if cfg["task"] == "rag":
            cfg["interactive"] = True
            main_rag(cfg)
            sys.exit(0)

        if cfg["task"] == "eval[ragas]":
            res = asyncio.run(eval_ragas_main(cfg))
            res.save()
        else:
            res = eval_mlflow_main(cfg)
            print(res)
