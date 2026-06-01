#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    },
}

RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "city name"}},
        "required": ["city"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:7893/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=Path("/tmp/glm51_endpoint_matrix.json"))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def post(base_url: str, path: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except Exception as exc:
        return {
            "status": None,
            "elapsed_s": round(time.monotonic() - start, 3),
            "ok": False,
            "error": repr(exc),
        }

    try:
        data = json.loads(raw)
    except Exception:
        data = None
    return {
        "status": status,
        "elapsed_s": round(time.monotonic() - start, 3),
        "ok": 200 <= status < 300,
        "raw": raw[:4000],
        "json": data,
    }


def summarize_chat(data: dict[str, object]) -> dict[str, object]:
    try:
        choice = data["choices"][0]  # type: ignore[index]
        msg = choice["message"]
    except Exception:
        return {"parsed": False}
    return {
        "parsed": True,
        "finish_reason": choice.get("finish_reason"),
        "content_preview": (msg.get("content") or "")[:200],
        "reasoning_preview": (msg.get("reasoning_content") or "")[:200],
        "tool_call_count": len(msg.get("tool_calls") or []),
        "tool_calls": msg.get("tool_calls"),
    }


def summarize_completion(data: dict[str, object]) -> dict[str, object]:
    try:
        choice = data["choices"][0]  # type: ignore[index]
    except Exception:
        return {"parsed": False}
    return {
        "parsed": True,
        "finish_reason": choice.get("finish_reason"),
        "text_preview": (choice.get("text") or "")[:240],
    }


def summarize_responses(data: dict[str, object]) -> dict[str, object]:
    output = data.get("output") if isinstance(data, dict) else None
    calls = []
    text_parts = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"function_call", "tool_call"}:
                calls.append(item)
            for content_item in item.get("content") or []:
                if isinstance(content_item, dict) and "text" in content_item:
                    text_parts.append(content_item["text"])
    return {
        "parsed": isinstance(data, dict),
        "status": data.get("status") if isinstance(data, dict) else None,
        "output_types": [item.get("type") for item in output if isinstance(item, dict)]
        if isinstance(output, list)
        else None,
        "text_preview": "".join(text_parts)[:240],
        "tool_call_count": len(calls),
        "tool_calls": calls,
    }


def main() -> None:
    args = parse_args()
    prompt = "Do not show reasoning. Reply with exactly this word: hello"
    tool_prompt = "Use the get_weather tool to look up weather for Beijing. Do not answer directly."

    tests = [
        ("/v1/completions no tools", "/completions", {"model": args.model, "prompt": prompt, "temperature": 0, "max_tokens": 32}, summarize_completion),
        ("/v1/chat/completions no tools", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 512}, summarize_chat),
        ("/v1/responses no tools", "/responses", {"model": args.model, "input": prompt, "temperature": 0, "max_output_tokens": 512}, summarize_responses),
        ("/v1/chat/completions tools auto", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": tool_prompt}], "tools": [CHAT_TOOL], "tool_choice": "auto", "temperature": 0, "max_tokens": 128}, summarize_chat),
        ("/v1/responses tools auto", "/responses", {"model": args.model, "input": tool_prompt, "tools": [RESPONSES_TOOL], "tool_choice": "auto", "temperature": 0, "max_output_tokens": 128}, summarize_responses),
        ("/v1/chat/completions tools required", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": tool_prompt}], "tools": [CHAT_TOOL], "tool_choice": "required", "temperature": 0, "max_tokens": 128}, summarize_chat),
        ("/v1/responses tools required", "/responses", {"model": args.model, "input": tool_prompt, "tools": [RESPONSES_TOOL], "tool_choice": "required", "temperature": 0, "max_output_tokens": 128}, summarize_responses),
        ("/v1/chat/completions forced function", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": tool_prompt}], "tools": [CHAT_TOOL], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "temperature": 0, "max_tokens": 128}, summarize_chat),
        ("/v1/responses forced function", "/responses", {"model": args.model, "input": tool_prompt, "tools": [RESPONSES_TOOL], "tool_choice": {"type": "function", "name": "get_weather"}, "temperature": 0, "max_output_tokens": 128}, summarize_responses),
    ]

    results = []
    for name, path, payload, summarizer in tests:
        result = post(args.base_url, path, payload, args.timeout)
        summary: dict[str, object] = {}
        if result.get("json") is not None:
            summary = summarizer(result["json"])  # type: ignore[arg-type]
        row = {
            "name": name,
            "status": result.get("status"),
            "ok": result.get("ok"),
            "elapsed_s": result.get("elapsed_s"),
            "summary": summary,
            "error": result.get("error"),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        results.append({"name": name, "path": path, "payload": payload, "result": result, "summary": summary})

    args.out.write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "model": args.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"RESULT_JSON={args.out}")


if __name__ == "__main__":
    main()
