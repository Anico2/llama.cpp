Branch containing different projects (some are work in progress).  
The engine is based on llama.cpp, used for running quantized llms, and an intel-GPU running with SYCL backend.


## 1. How to download, quantize and add a model to the models folder
1. Download model folder (files + model.safetensor). To do this check [HuggingFace examples](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct?clone=true). 
    Generally the fastest way to do it is to run:
```sh
uvx hf download HuggingFaceTB/SmolLM2-1.7B-Instruct
```

2. From run_llama.cpp root run:
```sh
uv run --with-requirements requirements/requirements-convert_hf_to_gguf.txt \
python convert_hf_to_gguf.py SmolLM2-1.7B-Instruct --outfile models_gguf/SmolLM2-1.7B-Instruct.gguf
```

3. Then quantize, using 8 threads, running the following (may need to first execute /opt/intel/oneapi/setvars.sh if using SYCL and check llama.cpp repo for alternative quantizations):

```sh
./build/bin/llama-quantize run_llamacpp/models_gguf/SmolLM2-1.7B-Instruct.gguf \
run_llamacpp/models_gguf/SmolLM2-1.7B-Q6_K-Instruct.gguf Q6_K 8
```


## 2. How to run llama-server

The following can be used to serve all the models locally available:
 ```sh
 ./run_llamacpp/main.sh mode=server model=all
 ```
We can acces the UI at port 8080 and choose the model from the drop-down menu.


We can also run an embedding model at 8081 port, that can be queried to embed
documents in a vector store.

```sh
/run_llamacpp/main.sh mode=server_plus_embed model=all embed_model=nomic_embed
```

Use, instead, the following to run a model with llama-completion mode (see main.sh for other examples):
```sh
./run_llamacpp/main.sh model=mistral7binstr_q4 prompt="Write a poem about space"
```

## 3. How to query a model using OpenAI compatible API

The following, from the folder *chat_openai*, can be used to start an interactive sessione,
    with possibility to save/restart conversations:

```sh
uv run chat.py  model=Qwen2.5-3B-Q6_K-Instruct
```

## 4. How to query a running LLM providing custom context from pdf

1. If not running, start llama-server with decoder model and model embedding endpoints, if not running. Run postgres server, changing configurations if necessary.

```sh
cd rag_langchain
docker compose up -d
```

2. Create venv with requirements and activate it with:
```sh
uv sync
source .venv/bin/activate
```

3. Start interactive session, where llm can answer questions, optionally using custom context,
embedded from *documents* folder. Use rag-mode to choose the rag strategy (Rewrite-Retrieve-Read in the following example):

```sh
python src/main.py --rag-mode=rrr
```

To change rag parameters, used models and other configs, use the *config.yml* file.

## 5. Memvid (work in progress)

Install the cli (for python sdk use the venv):
```
curl -fsSL https://raw.githubusercontent.com/memvid/preflight-installer/main/install.sh | bash
```
