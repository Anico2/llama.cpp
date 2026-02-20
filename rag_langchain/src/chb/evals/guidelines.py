from mlflow.genai.scorers import Guidelines

english = Guidelines(
    name="english",
    guidelines=[
        "The response must be in English",
        "The response must be grammatically correct",
    ],
)