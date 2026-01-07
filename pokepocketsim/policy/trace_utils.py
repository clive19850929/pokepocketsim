from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


def hash_vecs(vecs: Any) -> Optional[str]:
    try:
        payload = json.dumps(vecs, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        return None
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
