"""Submit requests to the local Yandex Tasks worker."""

import json
from pathlib import Path
import time
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run" / "private" / "api"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def api(path, method="GET", *, body=None, body_file=None, params=None, paginate=False, api_version="v1"):
    request_id = str(uuid4())
    action = {"path": path, "method": method, "api_version": api_version}
    for key, value in (("body", body), ("body_file", str(body_file) if body_file else None),
                       ("params", params), ("paginate", True if paginate else None)):
        if value is not None:
            action[key] = value
    write_json(RUN / "requests" / f"{request_id}.json", action)
    response_path = RUN / "responses" / f"{request_id}.json"
    deadline = time.monotonic() + 90
    while not response_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Check request {request_id} before retrying")
        time.sleep(0.2)
    result = json.loads(response_path.read_text(encoding="utf-8"))
    if not result["ok"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result["data"]
