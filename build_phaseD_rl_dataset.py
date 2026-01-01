#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase D 用の RL データセットを構築するスクリプト。

入力:
  - Phase B / C の自己対戦 PRIVATE_IDS ログ（例: ai_vs_ai_match_all_private_ids.jsonl）
    各行には少なくとも以下のフィールドが含まれていることを想定:
      * obs_full_vec または obs_vec : List[float]
      * action_candidates_vec       : List[List[float]]  # 各候補の 32 次元ベクトルなど
      * pi                          : List[float]        # 候補上の分布（MCTS / one-hot など）
      * z                           : float              # me 視点の value ラベル（[-1, +1]）
      * end_reason                  : str                # "PRIZE_OUT" / "DECK_OUT" など

  ※ record_type="game_summary" の集約行などは自動でスキップします。

出力:
  - d3rlpy などで読みやすい単一の .npz:
      observations : (N, obs_dim) float32
      actions      : (N, act_dim) float32
      rewards      : (N,)         float32   # ここでは z を基準にした設計値を利用
      terminals    : (N,)         float32   # すべて 1.0 （1-step エピソードとして扱う）
      sample_weight: (N,)         float32   # end_reason によるサンプル重み
      end_reason_ids: (N,)        int64     # end_reason を整数 ID にしたもの

  - メタ情報 JSON:
      * 入力パス
      * サンプル数 / 次元数
      * end_reason ラベル <-> ID の対応
      * 使用した END_REASON_WEIGHT_TABLE / reward 設計 など

使い方（例）:
  python build_phaseD_rl_dataset.py \
      --input D:\date\ai_vs_ai_match_all_private_ids.jsonl \
      --output D:\date\phaseD_rl_dataset_all.npz

  ※オプションを省略すると、--input のディレクトリに
     phaseD_rl_dataset_all.npz / phaseD_rl_dataset_all_meta.json を出力します。
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ============================================================
# ==================== 設定ブロック ==========================
# ============================================================

# デフォルト入力 (PRIVATE_IDS ログ)
DEFAULT_INPUT_PATH = r"D:\date\ai_vs_ai_match_all_private_ids.jsonl"

# end_reason によるサンプル重み
#   学習時に「サイド勝ち / タネ切れ勝ち」をやや重視し、
#   デッキアウト勝ちはかなり弱めに扱う想定。
END_REASON_WEIGHT_TABLE: Dict[str, float] = {
    "PRIZE_OUT":    1.0,
    "BASICS_OUT":   1.0,
    "DECK_OUT":     0.03,
    "TIMEOUT":      0.5,
    "SUDDEN_DEATH": 0.7,
    "CONCEDE":      0.5,
    "UNKNOWN":      0.5,
}
DEFAULT_SAMPLE_WEIGHT: float = 1.0

# reward 設計:
#   Phase D では基本的なラベルとして z を使いつつ、
#   end_reason ごとの係数やスケール・クリップをここでまとめて管理する。
USE_Z_AS_REWARD: bool = True  # True の場合、z をベースに reward を作る
REWARD_BASE_SCALE: float = 1.0  # 全体に掛けるスケール係数

# end_reason ごとに reward を微調整する係数
#   - サイド勝ち / タネ切れ勝ちを 1.0 付近
#   - DECK_OUT は reward の絶対値を弱める（CQL に「勝ちだが旨味は小さい」と見せる）
REWARD_END_REASON_SCALE: Dict[str, float] = {
    "PRIZE_OUT":    1.0,
    "BASICS_OUT":   1.0,
    "DECK_OUT":     0.3,
    "TIMEOUT":      0.7,
    "SUDDEN_DEATH": 0.9,
    "CONCEDE":      0.7,
    "UNKNOWN":      0.7,
}

# reward の絶対値クリップ（None ならクリップしない）
REWARD_CLIP_ABS: Optional[float] = 3.0

# obs の取り方:
#   True  -> obs_full_vec があれば優先し、無ければ obs_vec を使う
#   False -> obs_vec のみ使う（obs_full_vec は無視）
USE_OBS_FULL_VEC: bool = False

# pi から行動候補の index を決めるときの最低合計値 (数値的な安全対策)
MIN_PI_SUM: float = 1e-6

# record_type="game_summary" のような集約レコードをスキップするかどうか
SKIP_SUMMARY_RECORDS: bool = True

# デッキアウトの占有率を抑えるための上限 (0〜1, 1.0 なら制限なし)
#   例: 0.7 なら、最終的に kept サンプル中の DECK_OUT が 70% を超えないように
MAX_DECKOUT_FRACTION: float = 0.7

# ============================================================
# ================== 設定ブロックここまで ====================
# ============================================================


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """JSONL を 1 行ずつ読み込むシンプルなイテレータ。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj


def _is_numeric_list(x: Any) -> bool:
    return isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x)


def _is_nested_numeric_list(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    for row in x:
        if not isinstance(row, list) or not row:
            return False
        if not all(isinstance(v, (int, float)) for v in row):
            return False
    return True

def _pick_obs_vec(entry: Dict[str, Any]) -> Optional[List[float]]:
    """obs_full_vec / obs_vec から観測ベクトルを 1 つ選んで返す。"""
    if USE_OBS_FULL_VEC:
        obs_full = entry.get("obs_full_vec")
        if _is_numeric_list(obs_full):
            return [float(v) for v in obs_full]

    obs = entry.get("obs_vec")
    if _is_numeric_list(obs):
        return [float(v) for v in obs]

    return None

def _pick_action_candidates_vec(entry: Dict[str, Any]) -> Optional[List[List[float]]]:
    """action_candidates_vec を数値 2 次元配列として取り出す。"""
    cand = entry.get("action_candidates_vec")
    if _is_nested_numeric_list(cand):
        return [[float(v) for v in row] for row in cand]
    return None


def _pick_pi(entry: Dict[str, Any], num_actions: int) -> Optional[List[float]]:
    """
    pi を取り出して正規化する。
    - len(pi) == num_actions の場合のみ採用。
    """
    pi = entry.get("pi")
    if not _is_numeric_list(pi):
        return None
    if len(pi) != num_actions:
        return None

    # 数値・正規化
    arr = np.asarray(pi, dtype=np.float64)
    s = float(arr.sum())
    if not np.isfinite(s) or s < MIN_PI_SUM:
        return None
    arr = arr / s
    return arr.astype(np.float32).tolist()


def _pick_action_index(entry: Dict[str, Any], pi: List[float], num_actions: int) -> Optional[int]:
    """
    行動候補リスト上の index を決める。
      1) a_idx が [0, num_actions) ならそれを優先
      2) そうでなければ argmax(pi)
    """
    a_idx = entry.get("a_idx")
    if isinstance(a_idx, int) and 0 <= a_idx < num_actions:
        return int(a_idx)

    # フォールバック: pi の argmax
    if pi and len(pi) == num_actions:
        arr = np.asarray(pi, dtype=np.float32)
        idx = int(np.argmax(arr))
        if 0 <= idx < num_actions:
            return idx

    return None


def _pick_end_reason(entry: Dict[str, Any]) -> str:
    """
    end_reason を抽出して大文字化する。
    entry.top / _end_reason / meta などをあさって、見つからなければ "UNKNOWN"。
    """
    r = entry.get("end_reason") or entry.get("_end_reason")

    if r is None:
        meta = entry.get("meta")
        if isinstance(meta, Dict):
            r = meta.get("end_reason")
            if r is None:
                gr = meta.get("game_result") or meta.get("result") or {}
                if isinstance(gr, Dict):
                    r = gr.get("end_reason") or gr.get("reason")

    if not r:
        r = "UNKNOWN"

    return str(r).upper()


def _pick_reward(entry: Dict[str, Any], end_reason: str) -> Optional[float]:
    """
    reward として使う値を決定する。
    Phase D では基本的に z をベースにしつつ、end_reason によってスケールする。
    """
    if not USE_Z_AS_REWARD:
        return None

    z = entry.get("z")
    if not isinstance(z, (int, float)):
        return None

    r = float(z) * REWARD_BASE_SCALE
    scale = float(REWARD_END_REASON_SCALE.get(end_reason, 1.0))
    r = r * scale

    if REWARD_CLIP_ABS is not None and REWARD_CLIP_ABS > 0.0:
        lim = float(REWARD_CLIP_ABS)
        if r > lim:
            r = lim
        elif r < -lim:
            r = -lim

    return r


def _calc_sample_weight(end_reason: str) -> float:
    """end_reason に対応するサンプル重みを返す。"""
    return float(END_REASON_WEIGHT_TABLE.get(end_reason, DEFAULT_SAMPLE_WEIGHT))


def build_phaseD_rl_dataset(
    input_path: str,
    output_npz_path: str,
    output_meta_path: str,
) -> Tuple[int, int]:
    """
    PRIVATE_IDS JSONL から Phase D 用の RL データセットを作成するメイン処理。

    戻り値:
      (total_lines, kept_samples)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input not found: {input_path}")

    obs_list: List[List[float]] = []
    act_list: List[List[float]] = []
    rew_list: List[float] = []
    done_list: List[float] = []
    weight_list: List[float] = []
    end_reason_str_list: List[str] = []

    total_lines = 0
    kept = 0
    deckout_kept = 0
    reason_counts: Dict[str, int] = {}

    for entry in _iter_jsonl(input_path):
        total_lines += 1

        if total_lines % 100000 == 0:
            print(
                "[PHASED_RL_DATASET] progress: input_lines={total} kept={kept}".format(
                    total=total_lines,
                    kept=kept,
                )
            )

        # record_type="game_summary" などはスキップ
        if SKIP_SUMMARY_RECORDS:
            rec_type = entry.get("record_type")
            if isinstance(rec_type, str) and rec_type.lower() == "game_summary":
                continue

        # 観測ベクトル
        obs = _pick_obs_vec(entry)
        if obs is None:
            continue

        # 候補ベクトル (32d など)
        cands = _pick_action_candidates_vec(entry)
        if cands is None or not cands:
            continue
        num_actions = len(cands)

        # pi
        pi = _pick_pi(entry, num_actions)
        if pi is None:
            continue

        # 行動 index
        a_idx = _pick_action_index(entry, pi, num_actions)
        if a_idx is None:
            continue

        # end_reason / sample_weight
        end_reason = _pick_end_reason(entry)

        # DECK_OUT の割合を制限する（MAX_DECKOUT_FRACTION < 1.0 のとき）
        if (
            MAX_DECKOUT_FRACTION < 1.0
            and end_reason == "DECK_OUT"
            and kept > 0
            and (deckout_kept / kept) >= MAX_DECKOUT_FRACTION
        ):
            continue

        # reward (z をベースに end_reason 係数などで設計)
        reward = _pick_reward(entry, end_reason)
        if reward is None:
            continue

        w = _calc_sample_weight(end_reason)

        # 実際の行動ベクトル
        try:
            act_vec = cands[a_idx]
        except Exception:
            continue

        # ここでは 1-step エピソードとして扱うため:
        #   next_obs は学習側で無視してもよいように
        #   terminals=1.0 のみを保存する。
        obs_list.append(obs)
        act_list.append(act_vec)
        rew_list.append(float(reward))
        done_list.append(1.0)
        weight_list.append(float(w))
        end_reason_str_list.append(end_reason)

        kept += 1
        reason_counts[end_reason] = reason_counts.get(end_reason, 0) + 1
        if end_reason == "DECK_OUT":
            deckout_kept += 1

    if not obs_list:
        raise RuntimeError(f"no valid samples found in {input_path}")

    # numpy 配列へ変換
    observations = np.asarray(obs_list, dtype=np.float32)
    actions = np.asarray(act_list, dtype=np.float32)
    rewards = np.asarray(rew_list, dtype=np.float32)
    terminals = np.asarray(done_list, dtype=np.float32)
    sample_weight = np.asarray(weight_list, dtype=np.float32)

    # end_reason を ID にエンコード
    unique_reasons = sorted(set(end_reason_str_list))
    reason2id = {r: i for i, r in enumerate(unique_reasons)}
    end_reason_ids = np.asarray(
        [reason2id[r] for r in end_reason_str_list],
        dtype=np.int64,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_npz_path)), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        sample_weight=sample_weight,
        end_reason_ids=end_reason_ids,
    )

    # メタ情報を書き出し
    meta = {
        "input_path": os.path.abspath(input_path),
        "output_npz_path": os.path.abspath(output_npz_path),
        "num_samples": int(kept),
        "total_lines": int(total_lines),
        "obs_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "use_obs_full_vec": USE_OBS_FULL_VEC,
        "use_z_as_reward": USE_Z_AS_REWARD,
        "reward_base_scale": REWARD_BASE_SCALE,
        "reward_end_reason_scale": REWARD_END_REASON_SCALE,
        "reward_clip_abs": REWARD_CLIP_ABS,
        "max_deckout_fraction": MAX_DECKOUT_FRACTION,
        "reason2id": reason2id,
        "end_reason_weight_table": END_REASON_WEIGHT_TABLE,
        "default_sample_weight": DEFAULT_SAMPLE_WEIGHT,
        "num_samples_by_end_reason": reason_counts,
    }

    with open(output_meta_path, "w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)

    print(
        "[PHASED_RL_DATASET] input_lines={total} kept={kept} "
        "obs_dim={obs_dim} action_dim={act_dim} -> {out}".format(
            total=total_lines,
            kept=kept,
            obs_dim=observations.shape[1],
            act_dim=actions.shape[1],
            out=os.path.abspath(output_npz_path),
        )
    )
    print("[PHASED_RL_DATASET] meta ->", os.path.abspath(output_meta_path))
    print(
        "[PHASED_RL_DATASET] final dims: obs_dim={obs_dim} action_dim={act_dim}".format(
            obs_dim=observations.shape[1],
            act_dim=actions.shape[1],
        )
    )

    return total_lines, kept


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase D 用 RL データセット (.npz) を構築するスクリプト",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"入力 PRIVATE_IDS JSONL のパス（既定: {DEFAULT_INPUT_PATH}）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "出力 .npz のパス。省略時は --input と同じディレクトリに "
            "phaseD_rl_dataset_all.npz を生成"
        ),
    )
    parser.add_argument(
        "--meta-output",
        default=None,
        help=(
            "メタ情報 JSON のパス。省略時は output の拡張子を .json に置き換えたもの "
            "(例: phaseD_rl_dataset_all_meta.json)"
        ),
    )
    args = parser.parse_args(argv)

    input_path = args.input

    if args.output:
        output_npz = args.output
    else:
        in_dir = os.path.dirname(os.path.abspath(input_path))
        output_npz = os.path.join(in_dir, "phaseD_rl_dataset_all.npz")

    if args.meta_output:
        output_meta = args.meta_output
    else:
        base, ext = os.path.splitext(output_npz)
        if base.endswith("_all"):
            output_meta = base + "_meta.json"
        else:
            output_meta = base + "_meta.json"

    build_phaseD_rl_dataset(input_path, output_npz, output_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
