import sys
import requests
import openai

prompt = None
model_arg = None

for arg in sys.argv[1:]:
    if arg.startswith("prompt="):
        prompt = arg.split("=", 1)[1]
    elif arg.startswith("model="):
        model_arg = arg.split("=", 1)[1]

if not prompt:
    print("Usage: python script.py prompt='Your question here' [model=model_name]")
    sys.exit(1)

models_resp = requests.get("http://localhost:8080/models")
if models_resp.status_code != 200:
    print("Error: Could not fetch models from server", models_resp.text)
    sys.exit(1)

available_models = [m["id"] for m in models_resp.json()["data"]]
if not available_models:
    print("No models available on the server.")
    sys.exit(1)

if model_arg:
    if model_arg not in available_models:
        print(f"Model '{model_arg}' not found on server. Available models:")
        for m in available_models:
            print(" ", m)
        sys.exit(1)
    model_to_use = model_arg
else:
    model_to_use = available_models[0]

print(f"Using model: {model_to_use}")

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="ignored"
)

completion = client.chat.completions.create(
    model=model_to_use,
    messages=[
        {"role": "system", "content": "You are an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=1000
)

print(completion.choices[0].message.content)
