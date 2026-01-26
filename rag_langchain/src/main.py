import sys
import logging
import asyncio
import datetime
from utils import load_env_config, parse_args
from rag import rag_system
from eval_ragas import eval_ragas_main
from eval_mlflow import eval_mlflow_main
import mlflow

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def mlflow_setup(cfg):
    try:
        mlflow.set_tracking_uri(cfg["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(cfg["experiment"])
        mlflow.openai.autolog()
        mlflow.langchain.autolog()
        logger.info(f"MLflow running! URI: {cfg['MLFLOW_TRACKING_URI']}")
    except Exception:
        logger.error("Failed to set MLflow tracking URI or enable autologging.")


def main_rag(cfg):
    rag = rag_system(cfg)

    print(f"\n--- RAG Pipeline Ready [{cfg['rag_mode']}] ---")

    if not cfg["interactive"]:
        return rag.answer(cfg["query"])

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
    cfg, args = load_env_config(), parse_args()
    cfg = {**cfg, **vars(args)}  # put evertyhing in one dict only

    datatime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["experiment"] = f"{cfg['task']}_{datatime_str}"

    mlflow_setup(cfg)

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