# pokepocketsim/policy/action_encoding.py
from __future__ import annotations
import json, os
from typing import Dict, List, Tuple, Callable
import numpy as np

# 位置系キー判定（将来追加してもここだけ直せばOK）
_POS_KEYS = ("bench_idx", "stack_index", "target_index")
_BENCH_NORM = 8.0  # 仕様: 位置系は常に /8.0

def _is_pos_key(k: str) -> bool:
    s = str(k).lower()
    return any(t in s for t in _POS_KEYS)

def load_action_types(path: str) -> List[int]:
    """
    action_types.json から one-hot 母集合（typeの並び）を読み込む。
    - 配列[int] or 配列[str]（ActionType名）の両方を許容
    - 未登録タイプのフォールバック先として 0 を含むことを推奨
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("action_types.json は非空の配列である必要があります。")

    # int のみならそのまま
    if all(isinstance(x, int) for x in raw):
        return [int(x) for x in raw]

    # 文字列は ActionType 名とみなす（数値文字列も許可）
    from pokepocketsim.action import ActionType

    out: List[int] = []
    for x in raw:
        # すでに int ならそのまま
        if isinstance(x, int):
            out.append(int(x))
            continue

        # str 以外はエラー
        if not isinstance(x, str):
            raise ValueError("action_types.json には int か str だけを入れてください。")

        xs = x.strip()

        # "0" などの数値文字列はそのまま int として解釈
        if xs.isdigit():
            out.append(int(xs))
            continue

        # 1) そのままの名前（"END_TURN" など）で Enum を探す
        if hasattr(ActionType, xs):
            out.append(int(getattr(ActionType, xs).value))
            continue

        # 2) 小文字スネークなどは大文字にして再トライ（"end_turn" -> "END_TURN"）
        xs_upper = xs.upper()
        if hasattr(ActionType, xs_upper):
            out.append(int(getattr(ActionType, xs_upper).value))
            continue

        # ここまで来たら解決不能
        raise ValueError(
            f"action_types.json の要素 '{x}' を ActionType に解決できません。"
        )

    return out

def build_type2idx(action_types: List[int]) -> Dict[int, int]:
    """
    type → one-hot index の写像を作る。
    未登録 type のフォールバック先は「値=0 のエントリ」があればその index、
    無ければ index=0（先頭）を使用（= その他）。
    """
    t2i = {int(t): i for i, t in enumerate(action_types)}
    # フォールバック先
    fb = t2i.get(0, 0)
    t2i["_fallback_"] = fb  # sentinel
    return t2i

def compute_layout(card_id2idx: Dict[int, int], action_types: List[int], max_args: int = 3) -> Tuple[int, int, int]:
    """
    戻り値: (K, V, action_vec_dim=K+max_args+1)
      K = len(action_types)
      V = len(card_id2idx)+1 (0=PAD)
    """
    K = int(len(action_types))
    V = int(len(card_id2idx) + 1)  # 0 = PAD
    return K, V, K + max_args + 1

def encode_action_from_vec(
    five_ints: List[int],
    *,
    card_id2idx: Dict[int, int],
    type2idx: Dict[int, int],
    ACTION_SCHEMAS: Dict[int, List[str]],
    TYPE_SCHEMAS: Dict[int, List[str]],
    max_args: int = 3,
) -> np.ndarray:
    """
    仕様に基づく唯一の写像：
      全長: K + 3 + 1
      先頭K: type one-hot（未登録は 'その他' にフォールバック）
      中間3: スキーマ順の引数（位置系は/8.0、カードID系は idx/V）
      末尾1: main_id を idx/V
    five_ints = [type, main_id, a1, a2, a3]（不足は0）
    """
    v = list(five_ints) + [0, 0, 0, 0, 0]
    a_type, main_id, a1, a2, a3 = (int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]))

    K = len([k for k in type2idx.keys() if k != "_fallback_"])
    V = int(len(card_id2idx) + 1)  # 0=PAD
    out = np.zeros(K + max_args + 1, dtype=np.float32)

    # --- one-hot(type) ---
    oh = type2idx.get(a_type, type2idx["_fallback_"])
    if 0 <= oh < K:
        out[oh] = 1.0
    else:
        out[type2idx["_fallback_"]] = 1.0

    # --- arg keys 決定（ACTION_SCHEMAS → TYPE_SCHEMAS） ---
    arg_keys = ACTION_SCHEMAS.get(main_id) or TYPE_SCHEMAS.get(a_type, [])
    raw_vals = [a1, a2, a3]
    for slot, key in enumerate(arg_keys[:max_args]):
        raw = int(raw_vals[slot]) if slot < len(raw_vals) else 0
        if _is_pos_key(key):
            out[K + slot] = float(max(0, raw)) / _BENCH_NORM
        else:
            # カードID系 → 語彙 idx / V（未知は0）
            idx = int(card_id2idx.get(raw, 0))
            out[K + slot] = float(idx) / float(max(1, V))

    # --- 末尾 main_id スカラー ---
    main_idx = int(card_id2idx.get(main_id, 0))
    out[-1] = float(main_idx) / float(max(1, V))
    return out

def build_encoder_from_files(
    vocab_path: str,
    action_types_path: str,
    ACTION_SCHEMAS: Dict[int, List[str]],
    TYPE_SCHEMAS: Dict[int, List[str]],
    max_args: int = 3,
) -> Tuple[Callable[[List[int]], np.ndarray], Dict[int, int], List[int], Tuple[int,int,int]]:
    """
    ファイルから（学習と同一の）エンコーダを構築して返す。
    戻り値: (encoder_fn, card_id2idx, action_types, (K,V,dim))
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        _map = json.load(f)
    card_id2idx = {int(k): int(v) for k, v in _map.items()}

    action_types = load_action_types(action_types_path)
    type2idx = build_type2idx(action_types)

    def _enc(five_ints: List[int]) -> np.ndarray:
        return encode_action_from_vec(
            five_ints,
            card_id2idx=card_id2idx,
            type2idx=type2idx,
            ACTION_SCHEMAS=ACTION_SCHEMAS,
            TYPE_SCHEMAS=TYPE_SCHEMAS,
            max_args=max_args,
        )

    layout = compute_layout(card_id2idx, action_types, max_args)
    return _enc, card_id2idx, action_types, layout
