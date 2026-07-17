"""
llm.py — single swap point for a real language model.

No API key was available while building this prototype, so `generate()` below
is a stub: it returns None, and every agent that calls it falls back to a
template composed from *retrieved KB data* (see rag.py) rather than inventing
free text. That keeps the demo 100% truthful — nothing you see in the UI is a
hallucination, because there's no LLM in the loop yet.

TO GO LIVE: install `anthropic` (`pip install anthropic`), set an API key
below or via the ANTHROPIC_API_KEY env var, and uncomment the real call.
Every call site elsewhere in the codebase is already written to just use
whatever `generate()` returns, so no other file needs to change.
"""

import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 400):
    """Returns a generated string, or None if no LLM is configured (stub mode).

    Callers MUST handle a None return by falling back to their own
    template/rules-based composition (see agents.py for examples).
    """
    if not ANTHROPIC_API_KEY:
        return None

    # --- Uncomment once you have a key -------------------------------------
    # import anthropic
    # client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # msg = client.messages.create(
    #     model="claude-sonnet-5",
    #     max_tokens=max_tokens,
    #     system=system_prompt,
    #     messages=[{"role": "user", "content": user_prompt}],
    # )
    # return msg.content[0].text
    # -------------------------------------------------------------------------
    return None


def is_live() -> bool:
    return bool(ANTHROPIC_API_KEY)
