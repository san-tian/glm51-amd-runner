#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


TERMINAL_SUCCESS = {"succeeded", "completed"}
TERMINAL_FAILURE = {"failed"}
URL_FIELDS = ("oss_http_url", "download_url", "oss_url")
DEFAULT_GPU_LEASE_API_KEY = ""


def request_json(method, url, api_key, payload=None):
    data = None
    headers = {"x-api-key": api_key}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {body}") from exc


def get_nested(mapping, *path):
    current = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_job_id(response):
    for path in (("job_id",), ("id",), ("job", "id"), ("data", "job_id")):
        value = get_nested(response, *path)
        if value:
            return str(value)
    raise RuntimeError(f"Could not find job_id in create response: {json.dumps(response, ensure_ascii=False)}")


def extract_result_url(result):
    if not isinstance(result, dict):
        return None
    for field in URL_FIELDS:
        value = result.get(field)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def print_summary(status_response, final_url=None):
    result = status_response.get("result") if isinstance(status_response, dict) else {}
    if not isinstance(result, dict):
        result = {}

    rows = {
        "job_id": status_response.get("job_id") or status_response.get("id"),
        "status": status_response.get("status"),
        "stage": status_response.get("stage"),
        "oss_key": result.get("oss_key"),
        "archive_size_bytes": result.get("archive_size_bytes"),
        "expires_at": result.get("expires_at"),
        "download_url": final_url,
    }
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a tinker:// model URL to a signed HTTP(S) archive URL via GPU Lease Manager async jobs."
    )
    parser.add_argument("model_url", nargs="?", default=os.environ.get("TINKER_URL"))
    parser.add_argument("--base-url", default=os.environ.get("GPU_LEASE_BASE_URL", "https://eval-service.macaron.im"))
    parser.add_argument("--api-key", default=os.environ.get("GPU_LEASE_API_KEY", DEFAULT_GPU_LEASE_API_KEY))
    parser.add_argument("--poll-interval", type=float, default=7.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    if not args.model_url:
        print("ERROR: provide a tinker:// URL argument or set TINKER_URL", file=sys.stderr)
        return 2
    if not args.model_url.startswith("tinker://"):
        print("ERROR: model URL must start with tinker://", file=sys.stderr)
        return 2
    if not args.api_key:
        print("ERROR: set GPU_LEASE_API_KEY", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    create_url = f"{base_url}/api/transfer/jobs"
    create_response = request_json("POST", create_url, args.api_key, {"model_url": args.model_url})
    job_id = extract_job_id(create_response)
    print(f"job_id: {job_id}", flush=True)

    status_url = f"{base_url}/api/transfer/jobs/{job_id}"
    deadline = time.monotonic() + args.timeout_seconds
    last_response = None

    while time.monotonic() < deadline:
        last_response = request_json("GET", status_url, args.api_key)
        status = str(last_response.get("status", "")).lower()
        stage = last_response.get("stage")
        print(f"status: {status} stage: {stage}", flush=True)

        if status in TERMINAL_FAILURE:
            print_summary(last_response)
            print("ERROR: transfer job failed", file=sys.stderr)
            if "error" in last_response:
                print(json.dumps(last_response["error"], ensure_ascii=False, indent=2), file=sys.stderr)
            return 1

        if status in TERMINAL_SUCCESS:
            result = last_response.get("result")
            final_url = extract_result_url(result)
            if not final_url:
                print_summary(last_response)
                print("ERROR: succeeded but no HTTP(S) URL found in result. Do not use oss:// as a download URL.", file=sys.stderr)
                return 1
            print_summary(last_response, final_url)
            return 0

        time.sleep(args.poll_interval)

    if last_response is not None:
        print_summary(last_response)
    print(f"ERROR: timed out after {args.timeout_seconds} seconds waiting for {job_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
