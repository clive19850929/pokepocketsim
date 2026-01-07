#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union


TRACE_TAGS = {
    "TRACE_A": ("[MCTS_ENV][LA5][TRACE_A]", "[ONLINE_MIX][LA5][TRACE_A]"),
    "TRACE_B": ("[ONLINE_MIX][TRACE_B]", "[AZ][MCTS][TRACE_B]"),
    "TRACE_C": ("[MCTS_ENV][LA5][TRACE_C]",),
}


def parse_kv(line: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        kv[key] = val
    return kv


def parse_trace_json(line: str) -> Optional[Dict[str, Union[str, int, float, bool, None, list, dict]]]:
    if "trace_json=" not in line:
        return None
    _, payload = line.split("trace_json=", 1)
    payload = payload.strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


class TraceEvent:
    def __init__(self) -> None:
        self.lines: Dict[str, List[int]] = defaultdict(list)
        self.hashes: Dict[str, Set[str]] = defaultdict(set)
        self.env_fps: Set[str] = set()
        self.online_fps: Set[str] = set()
        self.target_in_ids_false_lines: List[int] = []

    def add_line(self, kind: str, line_no: int, kv: Dict[str, Union[str, int, float, bool, None, list, dict]], source: str) -> None:
        self.lines[kind].append(line_no)
        la_hash = kv.get("legal_actions_vec_hash")
        if la_hash not in (None, ""):
            self.hashes[kind].add(str(la_hash))

        target_in_ids = kv.get("target_in_ids")
        if target_in_ids in {0, "0", False, "False", "false"}:
            self.target_in_ids_false_lines.append(line_no)

        online_fp = kv.get("state_fingerprint_online")
        env_fp = kv.get("state_fingerprint_env")
        if source == "ONLINE_MIX" and kv.get("state_fingerprint") not in (None, ""):
            online_fp = kv.get("state_fingerprint")
        if source in {"MCTS_ENV", "AZ"} and kv.get("state_fingerprint") not in (None, ""):
            env_fp = kv.get("state_fingerprint")

        if online_fp not in (None, "", "NA"):
            self.online_fps.add(str(online_fp))
        if env_fp not in (None, "", "NA"):
            self.env_fps.add(str(env_fp))


def classify_line(line: str) -> Optional[str]:
    for kind, tags in TRACE_TAGS.items():
        if any(tag in line for tag in tags):
            return kind
    return None


def detect_source(line: str) -> str:
    if "[ONLINE_MIX]" in line:
        return "ONLINE_MIX"
    if "[MCTS_ENV]" in line:
        return "MCTS_ENV"
    if "[AZ]" in line:
        return "AZ"
    return "UNKNOWN"


def check_trace(log_path: str) -> int:
    events: Dict[str, TraceEvent] = defaultdict(TraceEvent)
    no_event_lines: List[Tuple[int, str]] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            kind = classify_line(line)
            if kind is None:
                continue
            kv_json = parse_trace_json(line)
            kv = kv_json if kv_json is not None else parse_kv(line)
            event_id = kv.get("event_id")
            if not event_id:
                no_event_lines.append((line_no, line.strip()[:80]))
                continue
            source = detect_source(line)
            events[str(event_id)].add_line(kind, line_no, kv, source)

    print(f"log={log_path}")
    print(f"events={len(events)}")
    if no_event_lines:
        print(f"lines_without_event_id={len(no_event_lines)} sample={no_event_lines[:6]}")

    print("\n[TRACE COUNTS]")
    for event_id in sorted(events.keys()):
        ev = events[event_id]
        a = len(ev.lines.get("TRACE_A", []))
        b = len(ev.lines.get("TRACE_B", []))
        c = len(ev.lines.get("TRACE_C", []))
        print(f"event_id={event_id} TRACE_A={a} TRACE_B={b} TRACE_C={c}")

    print("\n[HASH CONSISTENCY]")
    for event_id in sorted(events.keys()):
        ev = events[event_id]
        a_hashes = ev.hashes.get("TRACE_A", set())
        b_hashes = ev.hashes.get("TRACE_B", set())
        c_hashes = ev.hashes.get("TRACE_C", set())
        mismatches = []
        if a_hashes and b_hashes:
            if len(a_hashes) == 1 and len(b_hashes) == 1:
                if a_hashes != b_hashes:
                    mismatches.append("A!=B")
            elif a_hashes != b_hashes:
                mismatches.append("A!=B(set)")
        if a_hashes and c_hashes:
            if len(a_hashes) == 1 and len(c_hashes) == 1:
                if a_hashes != c_hashes:
                    mismatches.append("A!=C")
            elif a_hashes != c_hashes:
                mismatches.append("A!=C(set)")
        if mismatches:
            print(
                f"event_id={event_id} mismatch={','.join(mismatches)} "
                f"A={sorted(a_hashes)} B={sorted(b_hashes)} C={sorted(c_hashes)}"
            )

    print("\n[TARGET_IN_IDS_FALSE]")
    for event_id in sorted(events.keys()):
        ev = events[event_id]
        if not ev.target_in_ids_false_lines:
            continue
        print(
            "event_id={eid} target_in_ids_false_lines={false_lines} "
            "TRACE_A_lines={a_lines} TRACE_B_lines={b_lines} TRACE_C_lines={c_lines}".format(
                eid=event_id,
                false_lines=ev.target_in_ids_false_lines,
                a_lines=ev.lines.get("TRACE_A", []),
                b_lines=ev.lines.get("TRACE_B", []),
                c_lines=ev.lines.get("TRACE_C", []),
            )
        )

    print("\n[FINGERPRINTS]")
    for event_id in sorted(events.keys()):
        ev = events[event_id]
        online = sorted(ev.online_fps)
        env = sorted(ev.env_fps)
        status = []
        if len(online) > 1:
            status.append("online=揃わない")
        if len(env) > 1:
            status.append("env=揃わない")
        if online and env and set(online) != set(env):
            status.append("online/env=揃わない")
        status_txt = ",".join(status) if status else "揃う"
        print(f"event_id={event_id} online={online} env={env} status={status_txt}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TRACE_A/B/C correlation in a battle log.")
    parser.add_argument("log_path", help="Path to battle log file")
    args = parser.parse_args()
    log_path = os.path.abspath(args.log_path)
    if not os.path.exists(log_path):
        print(f"error: log file not found: {log_path}")
        return 2
    return check_trace(log_path)


if __name__ == "__main__":
    raise SystemExit(main())
