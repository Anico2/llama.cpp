import logging
import time

from ragas import experiment, Dataset
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision
from pydantic import BaseModel

from chb.engine.rag import rag_system
from chb.utils.clients import get_eval_model_classes

# Initialize models
llm, embeddings = get_eval_model_classes(async_mode=True)
logger = logging.getLogger(__name__)



class ExperimentResult(BaseModel):
    experiment_name: str | None
    question: str
    good_answer: str
    response: str
    retrieved_contexts: list[str] | None
    latency_seconds: float | None
    metric_context_precision: float | None
    metric_relevancy: float | None


def get_dataset():
    dataset = Dataset(
        name="stellantis_h1_2025_eval", backend="local/csv", root_dir="./evals"
    )
    data_samples = [
        {
            "question": "What were the Net Revenues for Stellantis in H1 2025 compared to H1 2024?",
            "good_answer": "- H1 2025 Net revenues were €74,261 million \n- H1 2024 Net revenues were €85,017 million \n- This represents a decrease of 12.7% ",
        },
        {
            "question": "Why did Stellantis discontinue its hydrogen fuel cell technology program in 2025?",
            "good_answer": "- Limited availability of hydrogen refueling infrastructure \n- High capital requirements \n- The need for stronger consumer purchasing incentives ",
        },
        {
            "question": "What factors caused the decrease in North America's Adjusted Operating Income in H1 2025?",
            "good_answer": "- Significant unfavorable impacts from volume and mix \n- Increased sales incentives \n- Unfavorable variable cost absorption and warranty costs ",
        },
        {
            "question": "How did the 'One Big Beautiful Bill Act' (OBBB) impact Stellantis' CAFE penalty provisions?",
            "good_answer": "- The act eliminated CAFE fines/penalties (revised rate to $0.00) \n- Resulted in a net expense of €269 million \n- Comprised of: impairment of regulatory credit assets (€609m), onerous contracts (€504m), offset by elimination of CAFE provision (€844m) [cite: 1338, 1339]",
        },
        {
            "question": "What was the Industrial Free Cash Flow for the first half of 2025?",
            "good_answer": "- Net cash absorption (negative) of €3,005 million \n- This is a decrease of €2,613 million compared to H1 2024 ",
        },
    ]
    for sample in data_samples[4:]:
        row = {"question": sample["question"], "good_answer": sample["good_answer"]}
        dataset.append(row)
    dataset.save()

    return dataset


@experiment(ExperimentResult)
async def run_eval(row, cfg: dict):
    # Start timer
    start_time = time.time()

    response = rag_system(cfg).answer(row["question"])

    end_time = time.time()
    latency = end_time - start_time

    # Safe extraction of contexts for the UI
    try:
        contexts = [doc.page_content for doc in response.get("source_documents", [])]
    except Exception:
        contexts = []

    # Answer Relevancy
    relevancy_score = None
    try:
        relevancy_result = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
            user_input=row["question"], response=response["answer"]
        )
        relevancy_score = relevancy_result.value
    except Exception as e:
        logger.error(f"Answer Relevancy evaluation failed: {e}")

    # Context Precision
    context_precision_score = None
    try:
        if contexts:
            context_precision_result = await ContextPrecision(llm=llm).ascore(
                user_input=row["question"],
                reference=row["good_answer"],
                retrieved_contexts=contexts,
            )
            context_precision_score = context_precision_result.value
        else:
            context_precision_score = 0.0
    except Exception as e:
        logger.error(f"Context Precision evaluation failed: {e}")

    # Return the enriched object
    return {
        "experiment_name": cfg["experiment"],
        "question": row["question"],
        "good_answer": row["good_answer"],
        "response": response["answer"],
        "retrieved_contexts": contexts,
        "latency_seconds": round(latency, 2),
        "metric_context_precision": context_precision_score,
        "metric_relevancy": relevancy_score,
    }


async def eval_ragas_main(cfg):
    dataset = get_dataset()

    results = await run_eval.arun(dataset, cfg=cfg)

    return results
