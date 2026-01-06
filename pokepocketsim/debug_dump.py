from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict


def write_debug_dump(payload: Dict[str, Any], dump_dir: str = "debug_dumps") -> str:
    os.makedirs(dump_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    pid = os.getpid()
    run_id = uuid.uuid4().hex[:6]
    path = os.path.join(dump_dir, f"debug_dump_{ts}_pid{pid}_{run_id}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
