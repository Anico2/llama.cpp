import logging
import os
from typing import Literal

import mlflow

from rag import rag_system

logger = logging.getLogger(__name__)

JUDGE_MODEL = "openai:/ignored-by-llamacpp"

quality_judge = mlflow.genai.judges.make_judge(
    name="response_quality",
    instructions=(
        "You are an impartial judge. Evaluate if the response in {{ outputs }} "
        "correctly answers the question in {{ inputs }}. "
        "The response should be accurate, complete, and professional.\n\n"
        "IMPORTANT OUTPUT RULES:\n"
        "1. Return ONLY a valid JSON object.\n"
        "2. The JSON must have two keys: 'result' and 'rationale'.\n"
        "3. 'result' MUST be exactly one of these strings: 'yes' or 'no'. Do not use 'Correct'.\n"
        "4. 'rationale' must be a short explanation string enclosed in double quotes."
    ),
    model=JUDGE_MODEL,
    feedback_value_type=Literal["yes", "no"],
)

correctness_judge = mlflow.genai.judges.make_judge(
    name="correctness",
    instructions=(
        "You are an impartial judge. Compare the {{ outputs }} against the {{ expectations }}.\n"
        "Rate how well they match on a scale of 1 to 5.\n\n"
        "IMPORTANT OUTPUT RULES:\n"
        "1. Return ONLY a valid JSON object.\n"
        "2. The JSON must have two keys: 'result' and 'rationale'.\n"
        "3. 'result' must be an integer between 1 and 5.\n"
        "4. 'rationale' must be a valid JSON string explaining the score. ESCAPE ALL QUOTES inside the rationale."
    ),
    model=JUDGE_MODEL,
    feedback_value_type=int,
)


samples = [
    {
        "inputs": {
            "question": "What were the primary drivers for the decrease in Net revenues for the six months ended June 30, 2025 compared to the same period in 2024?"
        },
        "expectations": {
            "answer": "Net revenues decreased by 12.7% primarily due to lower shipment volumes in North America and Enlarged Europe, unfavorable foreign exchange effects, and mix impacts. [cite_start]Specific operational drivers included a €7.457 billion negative impact from Volume & Mix, a €1.912 billion negative impact from Vehicle Net Price, and a €1.387 billion negative impact from FX and Other."
        },
    },
    {
        "inputs": {
            "question": "Why did Stellantis decide to discontinue its hydrogen fuel cell technology development program in 2025?"
        },
        "expectations": {
            "answer": "Management concluded that due to limited availability of hydrogen refueling infrastructure, high capital requirements, and the need for stronger consumer purchasing incentives, the adoption of hydrogen-powered light commercial vehicles would not happen before the end of the decade. [cite_start]Consequently, they discontinued the program, resulting in significant impairments and charges totaling over €733 million, including the impairment of the Symbio joint venture."
        },
    },
]


def eval_mlflow_main(cfg):
    """
    Main evaluation pipeline using the mlflow.genai.evaluate API.
    """
    logger.info("Initializing RAG system for evaluation...")
    
    os.environ["OPENAI_API_BASE"] = os.environ["JUDGE_MODEL_ENDPOINT"]
    os.environ["OPENAI_API_KEY"] = os.environ["JUDGE_MODEL_API_KEY"]

    rag = rag_system(cfg)

    def predict_fn(question):
        result = rag.answer(question)
        return result.get("answer", "")

    logger.info(f"Starting MLflow evaluation run: {cfg['experiment']}")

    with mlflow.start_run(run_name="genai_eval_run"):
        mlflow.log_params(cfg)

        results = mlflow.genai.evaluate(
            data=samples,
            predict_fn=predict_fn,
            scorers=[quality_judge, correctness_judge],
        )

    logger.info("Evaluation complete.")

    return results
