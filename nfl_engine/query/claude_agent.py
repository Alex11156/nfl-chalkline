"""Optional Claude API mode: free-form questions answered via tool use.

Active only when ANTHROPIC_API_KEY is set (or another credential source the
SDK resolves). Claude interprets the question, calls the same engine tools
the local parser uses, and writes a grounded, narrated answer.
"""
import json
import os

from nfl_engine.query.tools import TOOL_SPECS, call_tool

MODEL = "claude-opus-5"

SYSTEM = """You are an NFL prediction assistant backed by a statistical engine
trained on play-by-play data from 1999-2025 (walk-forward validated). Use the
tools to ground every number you state — never invent statistics or
predictions. The engine's honest accuracy: ~65% straight-up winners (Vegas
closing lines hit ~66%), ~51% against the spread, fantasy PPR MAE ~4.6 points.
Convey genuine uncertainty; a 55% edge is meaningful but not a lock, and no
model reliably beats the closing line. For situational/historical questions
prefer situational_query or a sql_query over guessing. Current data runs
through the 2025 season (2026 season has not started). Keep answers concise
and quantitative, and mention floors/ceilings when projecting players."""


def available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    try:
        import anthropic  # noqa: F401
        from pathlib import Path
        return (Path.home() / ".config" / "anthropic").exists()
    except ImportError:
        return False


def _api_tools() -> list[dict]:
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["input_schema"]} for t in TOOL_SPECS]


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Run the tool-use loop. Returns {'text': ..., 'tool_calls': [...]}."""
    import anthropic

    client = anthropic.Anthropic()
    messages = list(history or []) + [{"role": "user", "content": question}]
    user_start = len(messages) - 1
    tool_calls = []

    for _ in range(12):  # hard cap on loop iterations
        response = client.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM,
            tools=_api_tools(), messages=messages,
        )
        if response.stop_reason == "refusal":
            return {"text": "The model declined to answer that question.",
                    "tool_calls": tool_calls}
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in response.content if b.type == "text")
            return {"text": text, "tool_calls": tool_calls}

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            args = tu.input if isinstance(tu.input, dict) else json.loads(tu.input)
            out = call_tool(tu.name, args)
            tool_calls.append({"tool": tu.name, "args": args})
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(out, default=str)[:20000]})
        messages.append({"role": "user", "content": results})

    return {"text": "Hit the tool-call limit before finishing — try a narrower question.",
            "tool_calls": tool_calls}
