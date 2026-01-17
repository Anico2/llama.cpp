import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset

# LangChain & Ragas Imports
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# Import your project modules (DRY - Don't Repeat Yourself)
from main import (
    load_env_config, 
    load_split_documents, 
    get_model_classes, 
    get_vectorstore
)
from rag import (
    SimpleRAGStrategy, 
    RRRStrategy, 
    MultiQueryRAGStrategy, 
    RerankRAGStrategy
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAG-Eval")

def generate_test_data(docs, llm, embeddings, size=10):
    """
    Generates synthetic Q&A pairs to use as a test set.
    """
    logger.info(f"Generating {size} synthetic test cases using local LLM...")

    # Ragas requires wrappers for local models
    #from openai import OpenAI; 
    #from ragas.llms import llm_factory; 
    #llm = llm_factory('gpt-4o-mini', client=OpenAI(api_key='...'))
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_emb,
    )

    testset = generator.generate_with_langchain_docs(
        docs, testset_size=size, raise_exceptions=False
    )

    df = testset.to_pandas()
    logger.info(f"Generated {len(df)} test pairs.")
    return df

def run_evaluation(test_df, strategies, llm, embeddings):
    """
    Runs every strategy against the test set and calculates metrics.
    """
    # Wrap models for the 'Judge'
    judge_llm = LangchainLLMWrapper(llm)
    judge_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Define Metrics
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    results = {}

    for name, strategy in strategies.items():
        logger.info(f"--- Evaluating Strategy: {name.upper()} ---")

        answers = []
        contexts = []

        for q in test_df["question"]:
            try:
                response = strategy.answer(q)
                answers.append(response["answer"])
                # Extract page content strings from Document objects
                c_text = [d.page_content for d in response["source_documents"]]
                contexts.append(c_text)
            except Exception as e:
                logger.error(f"Failed to answer '{q}': {e}")
                answers.append("Error")
                contexts.append([""])

        # Create dataset for this run
        data_dict = {
            "question": test_df["question"].tolist(),
            "answer": answers,
            "contexts": contexts,
            "ground_truth": test_df["ground_truth"].tolist(),
        }
        dataset = Dataset.from_dict(data_dict)

        # Evaluate
        score = evaluate(
            dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings
        )
        results[name] = score
        logger.info(f"Strategy {name} Results: {score}")

    return results

def plot_results(results):
    """Visualizes the comparison"""
    data = []
    for strategy, metrics in results.items():
        row = {"Strategy": strategy}
        row.update(metrics)
        data.append(row)

    df = pd.DataFrame(data)

    # Reshape for Seaborn
    df_melted = df.melt(id_vars="Strategy", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Strategy", y="Score", hue="Metric")
    plt.title("RAG Strategy Comparison (Local Evaluation)")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("rag_benchmark.png")
    logger.info("Saved benchmark chart to 'rag_benchmark.png'")

def main():
    
    cfg = load_env_config()
    
    # get models (we use the same model for LLM and Judge here)
    llm_model = cfg["evaluation"]["judge_llm"]["model"]
    llm, embeddings = get_model_classes(cfg, llm_model=llm_model)

    raw_docs, _ = load_split_documents(cfg["PDF_DIRS"], cfg, llm)
    if not raw_docs:
        raise ValueError("No documents loaded. Please check the PDF_DIRS and ingestion step.")

    # Use first n pages only to speed up testset generation
    subset_docs = raw_docs[:30]

    # Get vectorstore using shared logic
    vectorstore = get_vectorstore(cfg, embeddings)
    
    retriever = vectorstore.as_retriever(
        search_kwargs=cfg["vectorstore"]["search_kwargs"]
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on context. 
        Context: {context} 
        Question: ({question})"""
    )

    common = {"retriever": retriever, "llm": llm, "prompt": prompt}
    strategies = {
        "simple": SimpleRAGStrategy(**common),
        "rrr": RRRStrategy(**common),
        "multi_query": MultiQueryRAGStrategy(**common),
        "rerank": RerankRAGStrategy(**common),
    }

    #If test data do not exist, we generate it
    default_path = os.path.join(os.getcwd(), "test_dataset.csv")
    test_file = cfg["evaluation"].get("test_file_path", default_path)
    if os.path.exists(test_file):
        logger.info(f"Loading existing test data from {test_file}")
        test_df = pd.read_csv(test_file)
    else:
        logger.info("No existing test data found. Generating new test data.")
        test_df = generate_test_data(subset_docs, llm, embeddings, size=10)
        test_df.to_csv(test_file, index=False)

    
    results = run_evaluation(test_df, strategies, llm, embeddings)

    
    plot_results(results)

if __name__ == "__main__":
    main()