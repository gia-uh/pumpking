"""
A minimal fake of the OpenAI chat completions API, for tests and examples that must run
without network access or an API key.

It answers POST .../chat/completions. When the request carries a JSON schema response format,
it returns a JSON document that satisfies the schema, filling every string with the first
sentence of the user message so that providers can match quotes back to the source text.
Otherwise it returns that sentence as plain text.

Run it standalone with `python tests/fake_openai.py --port 8765`, or start it in-process with
`serve()`.
"""

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple


def _first_sentence(text: str) -> str:
    for line in text.splitlines():
        line = re.sub(r"^\s*(-|\*|\d+\.|TEXT \d+:)\s*", "", line).strip()
        if len(line.split()) >= 3 and not line.startswith("#"):
            return line
    return text.strip() or "fake"


def _instance(schema: Dict[str, Any], defs: Dict[str, Any], text: str) -> Any:
    if "$ref" in schema:
        return _instance(defs[schema["$ref"].split("/")[-1]], defs, text)
    if "anyOf" in schema:
        return _instance(schema["anyOf"][0], defs, text)
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        return {name: _instance(sub, defs, text) for name, sub in props.items()}
    if kind == "array":
        return [_instance(schema.get("items", {}), defs, text)]
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return text


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = [m["content"] for m in body["messages"] if m["role"] == "user"]
        text = _first_sentence(user[-1] if user else "")

        response_format = body.get("response_format") or {}
        if response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            content = json.dumps(_instance(schema, schema.get("$defs", {}), text))
        else:
            content = text

        payload = json.dumps(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def serve(port: int = 0) -> Tuple[ThreadingHTTPServer, str]:
    """Starts the fake server in a daemon thread and returns it with its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"Fake OpenAI API on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
