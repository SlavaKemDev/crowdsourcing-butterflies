"""Yandex Tasks API worker for the HW5 local queue."""

import argparse
import fcntl
import getpass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID


API_ORIGIN = "https://tasks.yandex.ru"
BASE_URL = API_ORIGIN + "/api/v1"


def valid_token(token):
    return isinstance(token, str) and 0 < len(token) <= 4096 and token.isascii() and all(
        33 <= ord(character) <= 126 for character in token)


class AuthServer(HTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


class AuthHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; form-action 'self'; "
                         "frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def allowed(self):
        return (self.path == self.server.auth_path
                and self.headers.get("Host") == self.server.auth_host
                and self.client_address[0] == "127.0.0.1"
                and self.server.token is None)

    def do_GET(self):
        if not self.allowed():
            self.reply(404, "Not found")
            return
        self.reply(200, '<!doctype html><html lang="ru"><meta charset="utf-8">'
                   '<title>Локальный доступ к API</title>'
                   '<h1>Локальный доступ к API</h1>'
                   '<p>Токен останется только в памяти локального процесса.</p>'
                   '<form method="post" autocomplete="off">'
                   '<label>OAuth-токен <input type="password" name="token" '
                   'autocomplete="off" maxlength="4096" required></label>'
                   '<button type="submit">Использовать токен локально</button></form></html>')

    def do_POST(self):
        if not self.allowed() or self.headers.get("Origin") != self.server.auth_origin:
            self.reply(403, "Forbidden")
            return
        length = self.headers.get("Content-Length", "")
        if (not length.isdecimal() or not 0 < int(length) <= 8192
                or len(self.headers.get_all("Content-Length", [])) != 1
                or self.headers.get("Transfer-Encoding")
                or self.headers.get_content_type() != "application/x-www-form-urlencoded"):
            self.reply(400, "Invalid request")
            return
        try:
            fields = parse_qs(self.rfile.read(int(length)).decode("ascii"),
                              strict_parsing=True, max_num_fields=1)
            token = fields.get("token", [""])[0]
            if set(fields) != {"token"} or not valid_token(token):
                raise ValueError("Invalid token")
        except (ValueError, UnicodeDecodeError):
            self.reply(400, "Invalid token")
            return
        self.server.token = token
        self.reply(200, "Токен получен локально. Можно закрыть вкладку.")


def auth_web(timeout=600):
    with AuthServer(("127.0.0.1", 0), AuthHandler) as server:
        server.auth_host = f"127.0.0.1:{server.server_port}"
        server.auth_origin = "http://" + server.auth_host
        server.auth_path = "/" + secrets.token_urlsafe(32)
        server.token = None
        server.timeout = 1
        print("LOCAL_AUTH_URL=" + server.auth_origin + server.auth_path, flush=True)
        deadline = time.monotonic() + timeout
        while server.token is None and time.monotonic() < deadline:
            server.handle_request()
        if server.token is None:
            raise ValueError("Local token input timed out")
        return server.token


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class APIError(Exception):
    def __init__(self, status, data):
        self.status = status
        self.data = data
        super().__init__(f"API HTTP {status}")


def validate_action(action):
    if not isinstance(action, dict):
        raise ValueError("Request must be a JSON object")
    method = action.get("method", "GET")
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        raise ValueError("Unsupported HTTP method")
    path = action.get("path", "")
    if not isinstance(path, str):
        raise ValueError("API path must be a string")
    parsed = urlsplit(path)
    if (not path.startswith("/") or path.startswith("//")
            or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
            or ".." in unquote(path).split("/") or "\\" in unquote(path)):
        raise ValueError("Use a relative API path and separate params")
    if not isinstance(action.get("params", {}), dict):
        raise ValueError("params must be an object")
    version = action.get("api_version", "v1")
    if version not in {"v1", "new"}:
        raise ValueError("Unsupported API version")
    if version == "new":
        allowed = (method == "PUT" and re.fullmatch(r"/requester/poolModeration/[0-9]+", path)
                   or method == "GET" and re.fullmatch(r"/requester/pools/[0-9]+/validate", path))
        if action.get("params") or not allowed:
            raise ValueError("Only pool validation and moderation are supported by the new API")
    if action.get("paginate") and method != "GET":
        raise ValueError("Only GET requests can be paginated")
    if "body" in action and "body_file" in action:
        raise ValueError("Use body or body_file, not both")
    if method == "GET" and ("body" in action or "body_file" in action):
        raise ValueError("GET requests cannot have a body")


def request(action, token, opener=None):
    validate_action(action)
    params = action.get("params", {})
    base = API_ORIGIN + "/api/new" if action.get("api_version") == "new" else BASE_URL
    url = base + action["path"]
    if params:
        url += "?" + urlencode(params, doseq=True)
    body = action.get("body")
    if "body_file" in action:
        body = json.loads(Path(action["body_file"]).read_text(encoding="utf-8"))
    if (action.get("api_version") == "new" and action.get("method") == "PUT"
            and body != {"status": "READY"}):
        raise ValueError("Pool moderation only accepts status READY")
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = Request(url, data=payload, method=action.get("method", "GET"), headers={
        "Authorization": "OAuth " + token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    opener = opener or build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=30) as response:
            raw = response.read()
            return {"http_status": response.status, "data": decode(raw)}
    except HTTPError as error:
        raise APIError(error.code, decode(error.read())) from None


def decode(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return {"text": raw.decode("utf-8", errors="replace")}


def execute(action, token, opener=None):
    validate_action(action)
    if not action.get("paginate"):
        return request(action, token, opener)
    params = dict(action.get("params", {}))
    params["sort"] = "id"
    params.setdefault("limit", 100)
    items, seen = [], set()
    page_count = 0
    while True:
        result = request({**action, "params": params}, token, opener)
        page = result["data"]
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise ValueError("Paginated response has no items array")
        page_count += 1
        for item in page["items"]:
            item_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(item_id, str) or item_id in seen:
                raise ValueError("Pagination returned missing or repeated IDs")
            seen.add(item_id)
            items.append(item)
        if page.get("has_more") is False:
            return {"http_status": result["http_status"], "page_count": page_count,
                    "data": {"items": items, "has_more": False}}
        if page.get("has_more") is not True or not page["items"]:
            raise ValueError("Invalid pagination continuation")
        params["id_gt"] = page["items"][-1]["id"]


def write_json(path, data, token=""):
    content = json.dumps(data, ensure_ascii=False, indent=2)
    if token:
        content = content.replace(token, "[REDACTED]")
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_file(path, run_dir, token):
    request_id = str(UUID(path.stem))
    if request_id != path.stem:
        raise ValueError("Request filename must be a canonical UUID")
    response_path = run_dir / "responses" / path.name
    receipt_path = run_dir / "receipts" / path.name
    if response_path.exists():
        return
    response = {"request_id": request_id, "ok": False}
    if receipt_path.exists():
        response.update(error="interrupted_request", outcome="unknown",
                        message="Check remote state before submitting a new request")
    else:
        write_json(receipt_path, {"request_id": request_id, "started_at": time.time()})
        try:
            action = json.loads(path.read_text(encoding="utf-8"))
            response.update(execute(action, token), ok=True)
        except APIError as error:
            response.update(error="api_error", http_status=error.status, data=error.data)
        except (OSError, ValueError, URLError) as error:
            response.update(error=type(error).__name__, message=str(error), outcome="unknown")
    response["finished_at"] = time.time()
    write_json(response_path, response, token)
    print(request_id, "OK" if response["ok"] else "ERROR", flush=True)


def serve(run_dir, use_auth_web=False):
    os.umask(0o077)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("requests", "responses", "receipts"):
        (run_dir / name).mkdir(exist_ok=True)
    with (run_dir / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("API_WORKER_PID=" + str(os.getpid()), flush=True)
        if use_auth_web:
            token = auth_web()
        else:
            if not sys.stdin.isatty():
                raise ValueError("Use --auth-web or start in a terminal")
            token = getpass.getpass("OAuth token (hidden, memory only): ").strip()
        if not valid_token(token):
            raise ValueError("Invalid OAuth token")
        print("API worker ready:", run_dir, flush=True)
        while not (run_dir / "STOP").exists():
            for path in sorted((run_dir / "requests").glob("*.json")):
                process_file(path, run_dir, token)
            time.sleep(0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--auth-web", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).parent / "run" / "private" / "api")
    args = parser.parse_args()
    try:
        serve(args.run_dir.resolve(), args.auth_web)
    except (KeyboardInterrupt, BlockingIOError):
        print("API worker stopped.")
