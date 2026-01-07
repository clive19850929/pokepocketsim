from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, Optional


def _traceback_head(traceback_payload: Any, max_lines: int = 10) -> list[str]:
    if isinstance(traceback_payload, list):
        return [str(line).strip() for line in traceback_payload[:max_lines]]
    if isinstance(traceback_payload, str):
        return [line.strip() for line in traceback_payload.splitlines()[:max_lines]]
    return []


def _selected_vec_head(payload: Dict[str, Any], head_len: int = 3) -> Optional[list[float]]:
    action_context = payload.get("action_context")
    if not isinstance(action_context, dict):
        return None
    selected_vec = action_context.get("selected_vec")
    if isinstance(selected_vec, tuple):
        selected_vec = list(selected_vec)
    if not isinstance(selected_vec, list):
        return None
    head = []
    for item in selected_vec[:head_len]:
        try:
            head.append(float(item))
        except Exception:
            head.append(0.0)
    return head


def _signature_components(payload: Dict[str, Any]) -> Dict[str, Any]:
    error_type = payload.get("error_type")
    exception_class = payload.get("exception_class") or error_type
    action_context = payload.get("action_context")
    selected_source = None
    if isinstance(action_context, dict):
        selected_source = action_context.get("selected_source")
    return {
        "error_type": error_type,
        "exception_class": exception_class,
        "traceback_head": _traceback_head(payload.get("traceback")),
        "selected_source": selected_source,
        "selected_vec_head": _selected_vec_head(payload),
    }


def _signature_hash(payload: Dict[str, Any]) -> str:
    sig = _signature_components(payload)
    serialized = json.dumps(sig, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_seen_signatures(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    if isinstance(data, list):
        return {str(item) for item in data}
    return set()


def _save_seen_signatures(path: str, signatures: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(signatures), f, ensure_ascii=False, indent=2)


def write_debug_dump(
    payload: Dict[str, Any],
    dump_dir: str = "debug_dumps",
    signature_dir: str = "logs",
) -> Optional[str]:
    signature_hash = _signature_hash(payload)
    seen_path = os.path.join(signature_dir, "seen_signatures.json")
    seen = _load_seen_signatures(seen_path)
    if signature_hash in seen:
        print(f"[DEBUG_DUMP][SKIP] signature={signature_hash} already seen", flush=True)
        return None
    seen.add(signature_hash)
    _save_seen_signatures(seen_path, seen)

    os.makedirs(dump_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    pid = os.getpid()
    run_id = uuid.uuid4().hex[:6]
    path = os.path.join(dump_dir, f"debug_dump_{ts}_pid{pid}_{run_id}.json")

    payload = dict(payload)
    payload.setdefault("signature_hash", signature_hash)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
