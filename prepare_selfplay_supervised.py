#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase B で生成した自己対戦 ID ログから、Phase C 用の教師データ (s_t, π_t, z) を抽出するスクリプト。

入力:
  - ai_vs_ai_match_all_private_ids.jsonl など
    各行には少なくとも以下のフィールドが含まれていることを想定:
      * obs_vec : List[float]        ... 観測ベクトル
      * action_candidates_vec または legal_actions_19d : List[List[float]] or List[List[int]]
      * pi      : List[float]        ... MCTS で再計算した行動分布
      * pi_source: str               ... "recomputed_mcts" を想定

  さらに、可能であれば:
      * z         : float            ... 試合の最終勝敗ラベル（+1 / -1 / 0）
      * end_reason: str              ... 終了理由ラベル（BASICS_OUT / PRIZE_OUT / DECK_OUT / UNKNOWN 等）
    が既に含まれていることが望ましい。無い場合は z は暫定的に 0.0、end_reason は "UNKNOWN" を入れる。

出力:
  - selfplay_supervised_dataset.jsonl (デフォルト)
    各行は以下の形式の JSON:
      {
        "obs_vec": [...],
        "action_candidates_vec": [[...], [...], ...],
        "pi": [...],
        "z": float,
        "end_reason": "PRIZE_OUT" など
      }

使い方:
  python prepare_selfplay_supervised.py \
      --input ai_vs_ai_match_all_private_ids.jsonl \
      --output selfplay_supervised_dataset.jsonl
"""

import argparse
import json
import sys
import os
import random
from typing import Any, Dict, Iterable, Optional

# DECK_OUT 終局ゲームに対して「終盤ターンのサンプル」を間引く設定
#   USE_DECKOUT_TURN_FILTER: True のときにフィルタを有効化
#   DECKOUT_TURN_THRESHOLD : この turn_index 以降のステップを削除対象とする
USE_DECKOUT_TURN_FILTER = False
DECKOUT_TURN_THRESHOLD = 100

# 1ゲームあたりに学習で使う最大ステップ数
#   - 例: 64 にすると、各 game_id から最大 64 行だけをランダムに抽出
#   - -1 にすると無効（= これまで通り、全ステップを使用）
PER_GAME_MAX_STEPS = -1

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _is_numeric_list(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    return all(isinstance(v, (int, float)) for v in x)


def _is_nested_list(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    return all(isinstance(v, list) and v for v in x)

def build_record(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # --- DeckOut 終局ゲームの終盤ステップを間引くフィルタ ---
    # end_reason == "DECK_OUT" かつ turn_index（または state_before.turn）が
    # DECKOUT_TURN_THRESHOLD を超えているステップはスキップする。
    end_reason_raw = entry.get("end_reason")
    end_reason_upper = end_reason_raw.upper() if isinstance(end_reason_raw, str) else None

    turn_idx = entry.get("turn_index")
    if turn_idx is None:
        sb = entry.get("state_before") or {}
        if isinstance(sb, dict):
            turn_idx = sb.get("turn")

    if (
        USE_DECKOUT_TURN_FILTER
        and end_reason_upper == "DECK_OUT"
        and isinstance(turn_idx, int)
        and turn_idx >= DECKOUT_TURN_THRESHOLD
    ):
        return None

    obs_vec = entry.get("obs_vec")
    if not _is_numeric_list(obs_vec):
        return None

    cand = entry.get("action_candidates_vec")
    if not _is_nested_list(cand):
        cand = entry.get("legal_actions_19d")
        if not _is_nested_list(cand):
            return None

    pi = entry.get("pi")
    if not _is_numeric_list(pi):
        return None

    if len(pi) != len(cand):
        return None

    pi_source = entry.get("pi_source")
    if not isinstance(pi_source, str):
        return None
    if (pi_source != "recomputed_mcts") and (pi_source != "raw") and (not pi_source.startswith("raw:")):
        return None

    z = entry.get("z")
    if isinstance(z, (int, float)):
        z_val = float(z)
    else:
        z_val = 0.0

    end_reason = entry.get("end_reason")
    if isinstance(end_reason, str):
        end_reason_val = end_reason.upper()
    else:
        end_reason_val = "UNKNOWN"

    rec = {
        "obs_vec": obs_vec,
        "action_candidates_vec": cand,
        "pi": pi,
        "z": z_val,
        "end_reason": end_reason_val,
    }
    return rec

def prepare_selfplay_supervised(input_path: str, output_path: str) -> int:
    """
    コード内から直接呼び出す用のエントリポイント。
      input_path  から JSONL を読み、
      output_path に Phase C 用教師データ JSONL を書き出す。
    """
    total = 0
    kept = 0

    # ゲームごとの採用ステップ数カウンタ（メモリ軽量）
    per_game_counts: Dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as fw:
        for entry in _iter_jsonl(input_path):
            total += 1
            rec = build_record(entry)
            if rec is None:
                continue

            # game_id を取得（無ければ _game_id、さらに無ければ "UNKNOWN"）
            gid = entry.get("game_id")
            if gid is None:
                gid = entry.get("_game_id")
            if gid is None:
                gid = "UNKNOWN"
            gid_str = str(gid)

            # PER_GAME_MAX_STEPS > 0 のときだけ per-game 制限をかける
            if PER_GAME_MAX_STEPS is not None and PER_GAME_MAX_STEPS > 0:
                current = per_game_counts.get(gid_str, 0)
                if current >= PER_GAME_MAX_STEPS:
                    # このゲームからはこれ以上サンプルを取らない
                    continue
                per_game_counts[gid_str] = current + 1

            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    out_abspath = os.path.abspath(output_path)
    print(f"[PREP] input_lines={total} kept={kept} -> {out_abspath}")
    return kept


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"D:\date\ai_vs_ai_match_all_ids.jsonl",
        help="Phase B で生成した *_ids.jsonl のパス（省略時は D:\\date\\ai_vs_ai_match_all_ids.jsonl）",
    )
    parser.add_argument(
        "--output",
        default="selfplay_supervised_dataset.jsonl",
        help="教師データ JSONL の出力先パス（省略時は selfplay_supervised_dataset.jsonl）",
    )
    args = parser.parse_args(argv)

    input_dir = os.path.dirname(os.path.abspath(args.input))
    if os.path.isabs(args.output):
        output_path = args.output
    else:
        output_path = os.path.join(input_dir, args.output)

    kept = prepare_selfplay_supervised(args.input, output_path)
    return 0


if __name__ == "__main__":
    main()
