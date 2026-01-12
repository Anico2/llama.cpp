
## How to add a model to the model repo
1. From llama.cpp root run (assuming SmolLM2-1.7B-Instruct folder is present 
    and with the model.safetensor downloaded)

    ```
    uv run \
    --with-requirements requirements/requirements-convert_hf_to_gguf.txt \
    python convert_hf_to_gguf.py \
    ../SmolLM2-1.7B-Instruct \
    --outfile ../models/SmolLM2-1.7B-Instruct.gguf
    ```


2. Then quantize with (may need to first execute /opt/intel/oneapi/setvars.sh)
    ```
    ./build/bin/llama-quantize \
    ../models/SmolLM2-1.7B-Instruct.gguf \
    ../models/SmolLM2-1.7B-Q6_K-Instruct.gguf \
    Q6_K $(nproc)
    ```


## How to run llama-server (from llama.cpp root), using all the models in the model folder.

 ```
 ./projects/run_models/llama.sh mode=server model=all
 ```

 In this way, we can switch model, either from the UI or we can making an API
 call using Openai API template, as in the following:

```
python3 ./run_with_openai_template.py \
prompt="Give me the first sentences of the book '1984' by orwell" \
model=Qwen2.5-3B-Q6_K-Instruct
```

