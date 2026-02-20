import logging
import os

import mlflow
from mlflow.genai.scorers import RetrievalRelevance, Safety
from chb.engine.rag import rag_system
from .judges import correctness_judge, quality_judge, JUDGE_MODEL
from .guidelines import english
from .datasets import get_dataset
logger = logging.getLogger(__name__)

def eval_mlflow_main(cfg):
    """
    Main evaluation pipeline using the mlflow.genai.evaluate API.
    """
    logger.info("Initializing RAG system for evaluation...")

    os.environ["OPENAI_API_BASE"] = os.environ["JUDGE_MODEL_ENDPOINT"]
    os.environ["OPENAI_API_KEY"] = os.environ["JUDGE_MODEL_API_KEY"]

    rag = rag_system(cfg)

    @mlflow.trace
    def predict_fn(question):
        result = rag.answer(question)
        return result.get("answer", "")

    mlflow.set_experiment(cfg["mlflow_experiment"])

    logger.info(f"Starting MLflow evaluation run: {cfg['mlflow_experiment']}")

    with mlflow.start_run(run_name=cfg["mlflow_experiment"]):
        mlflow.log_params(cfg)

        dataset = get_dataset()
        
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=predict_fn,
            scorers=[
                RetrievalRelevance(),
                quality_judge,
                correctness_judge,
                english,
                Safety(model=JUDGE_MODEL),
            ],
        )

    logger.info("Evaluation complete.")

    return results
