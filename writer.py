# ============================================================
# writer.py（ai vs ai.py から分離）
#
# 役割:
# - worker から受け取った batch（raw/ids/private_ids の JSONL 行）を
#   1つの writer プロセスで安全に統合出力する
#
# 注意:
# - Windows multiprocessing(spawn) でも安全に呼び出せるよう、writer_loop は
#   必要な設定値（出力パス/バッファ/ローテ等）を引数で受け取る
# ============================================================

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event


def _next_rotated_path(base_path: str, idx: int) -> str:
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:05d}{ext}"

def writer_loop(queue: "Queue", stop_event: "Event", RAW_JSONL_PATH: str, IDS_JSONL_PATH: str, PRIVATE_IDS_JSON_PATH: str, IO_BUFFERING_BYTES: int, JSONL_ROTATE_LINES: int, WRITER_FSYNC: bool, WRITER_FLUSH_SEC: float, BATCH_FLUSH_INTERVAL: float):
    """
    集中ライター。全ワーカーから受け取ったJSONL行を
    RAW_JSONL_PATH / IDS_JSONL_PATH / PRIVATE_IDS_JSON_PATH の「同一ファイル」に安全に追記する。
    """
    import queue as _queue_mod

    raw_idx = 0
    ids_idx = 0
    priv_idx = 0
    raw_path = RAW_JSONL_PATH
    ids_path = IDS_JSONL_PATH
    priv_path = PRIVATE_IDS_JSON_PATH

    # どこに出力しているか見失わないように絶対パスも表示
    print(f"[OUT] raw={raw_path} (abs={os.path.abspath(raw_path)})")
    print(f"[OUT] ids={ids_path} (abs={os.path.abspath(ids_path)})")
    print(f"[OUT] private_ids={priv_path} (abs={os.path.abspath(priv_path)})")

    # ディレクトリが無いと open で落ちるので、念のため作る
    try:
        rd = os.path.dirname(os.path.abspath(raw_path))
        if rd:
            os.makedirs(rd, exist_ok=True)
        idd = os.path.dirname(os.path.abspath(ids_path))
        if idd:
            os.makedirs(idd, exist_ok=True)
        pd = os.path.dirname(os.path.abspath(priv_path))
        if pd:
            os.makedirs(pd, exist_ok=True)
    except Exception:
        pass

    # 小さな対戦数/途中確認でも「ファイルに見える」ように、時間ベースでも flush する
    flush_every_sec = float(os.getenv("WRITER_FLUSH_SEC", "2.0"))
    do_fsync = (str(os.getenv("WRITER_FSYNC", "0")).strip() == "1")
    last_flush_t = time.time()

    raw_f = open(raw_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
    ids_f = open(ids_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
    priv_ids_f = open(priv_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
    raw_lines_written = 0
    id_lines_written = 0
    priv_id_lines_written = 0
    batch_count = 0

    def _flush_all() -> None:
        nonlocal last_flush_t
        try:
            raw_f.flush(); ids_f.flush(); priv_ids_f.flush()
        except Exception:
            pass
        if do_fsync:
            try: os.fsync(raw_f.fileno())
            except Exception: pass
            try: os.fsync(ids_f.fileno())
            except Exception: pass
            try: os.fsync(priv_ids_f.fileno())
            except Exception: pass
        last_flush_t = time.time()

    try:
        while True:
            try:
                item = queue.get(timeout=0.5)
            except _queue_mod.Empty:
                # writer 側で stop_event を見て穏当終了 & 定期 flush
                if stop_event.is_set():
                    break
                if flush_every_sec > 0 and (time.time() - last_flush_t) >= flush_every_sec:
                    _flush_all()
                continue

            if item is None:
                # POISON PILL（終端）
                break

            try:
                # item 形式（想定）: ("batch", raw_lines: List[str], id_lines: List[str], priv_id_lines: List[str])
                if not isinstance(item, (list, tuple)) or len(item) == 0:
                    print(f"[writer] 予期しないメッセージ形式: {type(item)} → スキップします")
                    continue

                kind = item[0]
                if kind != "batch":
                    # 不明メッセージは無視（将来の拡張用）
                    continue

                # 安全に各配列を取り出す（長さ不足でも落ちないように）
                # item: ("batch", game_id, raw_lines, id_lines, priv_id_lines) を優先して解釈（旧形式も許容）
                game_id = item[1] if len(item) >= 2 and isinstance(item[1], str) else None
                _base_i = 2 if game_id is not None else 1

                raw_lines      = item[_base_i]     if len(item) > _base_i     and isinstance(item[_base_i], list) else []
                id_lines       = item[_base_i + 1] if len(item) > _base_i + 1 and isinstance(item[_base_i + 1], list) else []
                priv_id_lines  = item[_base_i + 2] if len(item) > _base_i + 2 and isinstance(item[_base_i + 2], list) else []

                # 受け取った game_id.log に parent 側の [DONE]/[WINRATE]/[writer] も追記する
                # NOTE: tee はプロセス開始時に 1 回だけ起動する。ここでの再起動は stdout チェーンの多重化を招く。
                if game_id is not None:
                    pass

                # まとめ書き（ループより高速）
                if raw_lines:
                    raw_f.write('\n'.join(raw_lines) + '\n')
                    raw_lines_written += len(raw_lines)

                if id_lines:
                    ids_f.write('\n'.join(id_lines) + '\n')
                    id_lines_written += len(id_lines)

                if priv_id_lines:
                    priv_ids_f.write('\n'.join(priv_id_lines) + '\n')
                    priv_id_lines_written += len(priv_id_lines)

                batch_count += 1
                if BATCH_FLUSH_INTERVAL > 0 and (batch_count % BATCH_FLUSH_INTERVAL) == 0:
                    _flush_all()
                elif flush_every_sec > 0 and (time.time() - last_flush_t) >= flush_every_sec:
                    _flush_all()

                if JSONL_ROTATE_LINES > 0 and raw_lines_written >= JSONL_ROTATE_LINES:
                    _flush_all()
                    raw_f.close()
                    raw_idx += 1
                    raw_path = _next_rotated_path(RAW_JSONL_PATH, raw_idx)
                    raw_f = open(raw_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
                    raw_lines_written = 0
                    print(f"[writer] rotate raw -> {raw_path}")

                if JSONL_ROTATE_LINES > 0 and id_lines_written >= JSONL_ROTATE_LINES:
                    _flush_all()
                    ids_f.close()
                    ids_idx += 1
                    ids_path = _next_rotated_path(IDS_JSONL_PATH, ids_idx)
                    ids_f = open(ids_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
                    id_lines_written = 0
                    print(f"[writer] rotate ids -> {ids_path}")

                if JSONL_ROTATE_LINES > 0 and priv_id_lines_written >= JSONL_ROTATE_LINES:
                    _flush_all()
                    priv_ids_f.close()
                    priv_idx += 1
                    priv_path = _next_rotated_path(PRIVATE_IDS_JSON_PATH, priv_idx)
                    priv_ids_f = open(priv_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
                    priv_id_lines_written = 0
                    print(f"[writer] rotate private_ids -> {priv_path}")

            except Exception as e:
                print(f"[writer] 追記中に例外: {e}  ※このバッチはスキップします")
                continue
    finally:
        try:
            _flush_all()
        except Exception:
            pass
        try:
            raw_f.close()
        except Exception:
            pass
        try:
            ids_f.close()
        except Exception:
            pass
        try:
            priv_ids_f.close()
        except Exception:
            pass
        try:
            raw_sz = os.path.getsize(raw_path) if os.path.exists(raw_path) else -1
            ids_sz = os.path.getsize(ids_path) if os.path.exists(ids_path) else -1
            prv_sz = os.path.getsize(priv_path) if os.path.exists(priv_path) else -1
            print(f"[writer] closed: raw_lines={raw_lines_written} ids_lines={id_lines_written} priv_lines={priv_id_lines_written} batches={batch_count}")
            print(f"[writer] files: raw={raw_path} size={raw_sz}  ids={ids_path} size={ids_sz}  priv={priv_path} size={prv_sz}")
        except Exception:
            pass
