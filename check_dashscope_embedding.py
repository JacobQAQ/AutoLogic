"""Check DashScope Bailian embedding availability in OpenAI-compatible mode.

Usage:
  set DASHSCOPE_API_KEY=your_key
  python check_dashscope_embedding.py
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI


BASE_URL = os.environ.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("LOGICRAG_EMBEDDING_MODEL", "text-embedding-v3")


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("LOGICRAG_EMBEDDING_API_KEY")
    if not api_key:
        print("DASHSCOPE_API_KEY is not set.")
        return 2

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    try:
        response = client.embeddings.create(
            model=MODEL,
            input=["LogicRAG embedding API connectivity test."],
        )
    except Exception as exc:
        print(f"Embedding API check failed: {type(exc).__name__}: {exc}")
        return 1

    vector = response.data[0].embedding
    print("Embedding API check succeeded.")
    print(f"model={MODEL}")
    print(f"base_url={BASE_URL}")
    print(f"dimension={len(vector)}")
    print(f"sample_first_3={[round(float(x), 6) for x in vector[:3]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
