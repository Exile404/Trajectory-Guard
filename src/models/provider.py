"""Model provider abstraction.

One switch (PROVIDER in .env) selects the backend. Every backend returns a
LangChain chat model with the same .invoke(messages) interface, so the graph
nodes never change when we flip ollama to nim to bedrock.

Backends:
  ollama  : local models via snap ollama (phases 1 to 5).
  nim     : NVIDIA NIM free hosted API, OpenAI compatible (hosted final runs).
  bedrock : AWS Bedrock (phase 6, AWS learning).

Backend libs are imported lazily so only the selected backend needs its deps.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _provider() -> str:
    return os.getenv("PROVIDER", "ollama").strip().lower()


def get_llm(temperature: float = 0.0, max_tokens: int | None = None, model: str | None = None,
            num_ctx: int | None = None):
    """Return a LangChain chat model for the active PROVIDER.

    temperature 0.0 keeps code generation and diagnosis deterministic.
    max_tokens caps output, None means backend default.
    model overrides the env model id.
    num_ctx (ollama only) raises the context window; ollama's default silently
    truncates the oldest tokens on long prompts.
    """
    provider = _provider()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = dict(
            model=model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=temperature,
            num_predict=max_tokens or -1,
        )
        if num_ctx:
            kwargs["num_ctx"] = num_ctx
        return ChatOllama(**kwargs)

    if provider == "nim":
        from langchain_openai import ChatOpenAI

        # NIM free tier sheds load with 429/5xx and a 550B generation can run
        # for minutes: let the SDK retry fast transients and never kill a slow
        # response client-side.
        return ChatOpenAI(
            model=model or os.getenv("NIM_MODEL", "moonshotai/kimi-k2.6"),
            base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.getenv("NIM_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=600.0,
            max_retries=3,
        )

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        model_id = model or os.getenv("BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0")
        # Opus 4.8 / Sonnet 5 (anthropic.*) reject temperature/top_p/top_k with a
        # 400; omit temperature for them, keep it for Nova and everything else.
        temp = None if "anthropic" in model_id.lower() else temperature
        return ChatBedrockConverse(
            model=model_id,
            region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
            temperature=temp,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown PROVIDER '{provider}'. Use ollama, nim, or bedrock.")


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    llm = get_llm(temperature=0.0)
    print("provider:", _provider())
    resp = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    print("model says:", repr(resp.content))