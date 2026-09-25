"""
One-shot live smoke test against the real Jev API.

Usage:
    export JEV_AI_API_KEY=...      # or put it in .env (see .env.example)
    python scripts/jev_live_check.py

Never prints the key. Makes exactly one /systemone call and one /models call.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from core import jev_client  # noqa: E402


def main() -> None:
    if not os.environ.get("JEV_AI_API_KEY"):
        print("JEV_AI_API_KEY is not set (checked environment and .env). Aborting.")
        sys.exit(1)

    print("Connected models:")
    try:
        for m in jev_client.list_models():
            print(f"  - {m.name}: {m.description}")
    except jev_client.JevError as exc:
        print(f"  (could not list models: {exc})")

    print("\nCalling /systemone with model=jev-latest ...")
    result = jev_client.ask(
        state="My payment failed. Please help.",
        questions={
            "urgent": {
                "type": "noul",
                "instructions": "Does this message need urgent support?",
            }
        },
    )

    print(f"Model:   {result.model}")
    print(f"Run ID:  {result.run_id}")
    print(f"Answers: {result.answers}")
    print(f"Usage:   input_tokens={result.usage.input_tokens} "
          f"output_tokens={result.usage.output_tokens}")

    urgent_p = result.answers.get("urgent", {}).get("noul")
    assert urgent_p is not None, "expected answers.urgent.noul in the response"
    print(f"\nanswers.urgent.noul = {urgent_p} (probability the message is urgent)")


if __name__ == "__main__":
    main()
