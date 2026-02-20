from typing import Literal

import mlflow

JUDGE_MODEL = "openai:/ignored-by-llamacpp"

# see API: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.make_judge
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
    inference_params=None
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
    inference_params=None
)

