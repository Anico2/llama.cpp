# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=2.15.0",
#     "requests>=2.32.5",
# ]
# ///
"""
Local OpenAI-Compatible Chat Client, targeting llama.cpp running servers.

USAGE: 
    uv run chat.py 
    uv run chat.py --session mychat 
    uv run chat.py --load mychat 

OPTIONS: 
    --endpoint URL OpenAI-compatible API endpoint (default: http://localhost:8080) 
    --model NAME Model ID to use (defaults to first available). See main.sh for available models. 
    --session NAME Name of session to create/use (default: "default") 
    --load NAME Load an existing session from disk 
    -temperature FLOAT Sampling temperature (default: 0.7) 
    --top-p FLOAT Nucleus sampling probability (default: 0.95) 
    --max-tokens INT Max completion tokens per response (default: 800) 

INTERACTIVE COMMANDS: 
    /save Save the current session to disk 
    /reset Clear conversation history (keeps system prompt) 
    /exit Exit the chat (auto-saves on exit) 

NOTES: 
    - Conversations are stored as JSON files in ./sessions 
    - Supports streaming responses and simple tool/function calls 
    - Long conversations are automatically summarized to preserve context 
"""


import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import openai


DATA_DIR = Path("./sessions")
DATA_DIR.mkdir(exist_ok=True)


@dataclass(slots=True)
class ChatConfig:
    endpoint: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    max_turns: int = 20
    summary_trigger_turns: int = 30

    @property
    def max_messages(self) -> int:
        return 1 + self.max_turns * 2


@dataclass(slots=True)
class SessionState:
    name: str
    messages: list


@dataclass(slots=True)
class ChatStats:
    start: float
    tokens: int = 0

    def done(self) -> str:
        return f"[stats: ~{self.tokens} tokens | {time.time() - self.start:.2f}s]"


SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are an AI assistant. Be helpful, concise, and accurate.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://localhost:8080")
    p.add_argument("--model")
    p.add_argument("--session", default="default")
    p.add_argument("--load")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=800)
    return p.parse_args()


def session_path(name: str) -> Path:
    return DATA_DIR / f"{name}.json"


def save_session(state: SessionState):
    session_path(state.name).write_text(json.dumps(state.messages, indent=2))
    print(f"[saved session '{state.name}']")


def load_session(name: str) -> SessionState:
    path = session_path(name)
    if not path.exists():
        print(f"No session named '{name}'")
        sys.exit(1)
    return SessionState(name, json.loads(path.read_text()))


def fetch_models(endpoint: str) -> list[str]:
    r = requests.get(f"{endpoint}/models", timeout=60)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def get_time():
    return time.ctime()


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current system time",
        "parameters": {"type": "object", "properties": {}},
    },
}]


def run_tool(name: str) -> str:
    return get_time() if name == "get_time" else "Unknown tool"


def maybe_summarize(client, cfg: ChatConfig, messages: list):
    if len(messages) < cfg.summary_trigger_turns * 2:
        return messages

    print("\n[Summarizing long-term memory...]\n")

    summary = client.chat.completions.create(
        model=cfg.model,
        messages=[
            messages[0],
            {"role": "user", "content": "Summarize the conversation so far in 5 concise bullet points."},
        ],
        temperature=0.2,
        max_completion_tokens=200,
    ).choices[0].message.content

    return [
        messages[0],
        {"role": "system", "content": "Conversation summary:\n" + summary},
    ]


def chat_loop(client, cfg: ChatConfig, state: SessionState):
    print("\nType your message. Commands: /save /reset /exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input in ["/exit", "/q", "/quit"]:
            break
        if user_input == "/save":
            save_session(state)
            continue
        if user_input == "/reset":
            state.messages[:] = state.messages[:1]
            print("[conversation reset]")
            continue

        state.messages.append({"role": "user", "content": user_input})

        if len(state.messages) > cfg.max_messages:
            state.messages[:] = state.messages[:1] + state.messages[-cfg.max_messages + 1:]

        stats = ChatStats(start=time.time())
        assistant_parts = []

        tool_call = None
        assistant_tool_calls = []

        stream = client.chat.completions.create(
            model=cfg.model,
            messages=state.messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_completion_tokens=cfg.max_tokens,
            stream=True,
            stop=["<|im_end|>", "<|im_start|>"]
        )

        print("\nAssistant: ", end="", flush=True)

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        tool_call = tc
                        assistant_tool_calls.append(tc)
                continue

            if delta.content:
                print(delta.content, end="", flush=True)
                assistant_parts.append(delta.content)
                stats.tokens += len(delta.content.split())

        print(f"\n\n{stats.done()}\n")

        if assistant_tool_calls:
            state.messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments or "{}",
                        },
                    }
                ],
            })

            result = run_tool(tool_call.function.name)

            state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

            print(f"[tool result: {result}]")
            continue

        state.messages.append({
            "role": "assistant",
            "content": "".join(assistant_parts),
        })

        state.messages[:] = maybe_summarize(client, cfg, state.messages)


def main():
    args = parse_args()
    models = fetch_models(args.endpoint)

    model = args.model or models[0]
    if model not in models:
        print(f"Model not available. Availables {models}")
        sys.exit(1)

    cfg = ChatConfig(
        endpoint=args.endpoint,
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    client = openai.OpenAI(
        base_url=f"{cfg.endpoint}/v1",
        api_key="",
        timeout=60,
    )

    state = load_session(args.load) if args.load else SessionState(args.session, [SYSTEM_MESSAGE])

    print(f"Using model: {cfg.model}")
    print(f"Session: {state.name}")

    chat_loop(client, cfg, state)
    save_session(state)


if __name__ == "__main__":
    main()
