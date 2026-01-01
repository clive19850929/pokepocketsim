## 機械学習用 ID 変換

from enum import IntEnum
from typing import List

# ---------- 1. アクション種別 ----------
class ActionType(IntEnum):
    END_TURN       = 0
    PLAY_BENCH     = 1
    ATTACH_ENERGY  = 2
    USE_ITEM       = 3
    USE_SUPPORTER  = 4
    RETREAT        = 5
    ATTACK         = 6
    EVOLVE         = 7
    USE_ABILITY    = 8
    PLAY_STADIUM   = 9
    STADIUM_EFFECT = 10
    ATTACH_TOOL    = 11

AType = ActionType

# ---------- 2. 定数テーブル ----------
POKEMON_IDS  = list(range(10001, 49999)) 
ENERGY_TYPES = list(range(1, 9))
ITEM_IDS     = list(range(50001, 69999)) 
SUPPORTER_IDS= list(range(70001, 79999)) 
ATTACK_IDS   = list(range(100001, 199999))
STADIUM_IDS  = list(range(80001, 89999)) 
TOOL_IDS     = list(range(90001, 99999)) 


# ---------- 3. Multi‑Discrete 空間（各桁の基数） ----------
# [atype, p1, p2, p3, p4] という 5要素ベクトルで表現
RADIX = [
    len(AType),          # 0: アクション種別
    100012,              # 1: card_id / energy_type / ability_id…
    7,                   # 2: target index (0=battle,1‑5 bench,6=不用)
    100012,              # 3: サブパラメータ1（不要なら0）
    100012,              # 4: サブパラメータ2（不要なら0）
]

# ---------- 4. 変換関数 ----------
def vec2id(vec: List[int]) -> int:
    """固定長ベクトル → 一意の整数 ID"""
    idx = 0
    mult = 1
    for v, base in zip(vec, RADIX):
        idx += v * mult
        mult *= base
    return idx

def id2vec(idx: int) -> List[int]:
    """整数 ID → ベクトル"""
    vec = []
    for base in RADIX:
        vec.append(idx % base)
        idx //= base
    return vec

# ---------- 6. Action デコーダ ----------
def decode_action(vec: List[int]) -> dict:
    """
    Multi-Discreteベクトル → dict (ACTION_SCHEMAS対応)
    vec = [atype, p1, p2, p3, p4]
    """
    ACTION_SCHEMAS = {
        50001: ["trashed_id1", "trashed_id2", "selected_id"],               # ハイパーボール
        50002: ["selected_bench_idx", "new_active_id", "old_active_id"],    # ポケモンキャッチャー
        50003: ["selected_bench_idx", "new_active_id", "old_active_id"],    # ポケモンいれかえ
        50004: ["selected_id"],                                             # ポケギア3.0
        50005: ["target_index", "energy_id"],                               # クラッシュハンマー
        70002: ["selected_bench_idx", "new_active_id", "old_active_id"],    # ボスの指令
        70006: ["evo_id", "energy_id"],                                     # トウコ
        80001: ["selected_id"],                                             # ボウルタウン
        # ... 他カードもここに追加 ...
    }
    atype, p1, p2, p3, p4 = vec
    params = [p1, p2, p3, p4]
    result = {"action_id": atype}

    if atype in ACTION_SCHEMAS:
        keys = ACTION_SCHEMAS[atype]
        for i, key in enumerate(keys):
            result[key] = params[i] if i < len(params) else None
    else:
        result["params"] = params

    return result

# ---------- 5. 動的に ActionMask を作るヘルパ ----------
def build_action_mask(state) -> List[int]:
    """
    state: ゲーム状態オブジェクト（自作クラスや dict でOK）
    戻り値: 長さ = ∏RADIX のバイナリマスク（0=選べない,1=選べる）
    実運用では LargeMask → Sparse list にする方がメモリ効率的。
    """
    total_size = 1
    for base in RADIX:
        total_size *= base
    mask = [0] * total_size
    # アクション空間全体（例えば4次元なら [12][50000][7][10000]...）の全組み合わせに対応する「全ビットを0で初期化」。
    # 1次元化した「マスク配列」を作る（全部「0=無効」からスタート）


    # END_TURN は常に有効(ターン終了はいつでも押せるので、必ず1にする)
    mask[vec2id([AType.END_TURN, 0, 0, 0, 0])] = 1

    # ベンチに出せるポケモンのアクションだけ有効
    for pid in [c.id for c in state.hand if c.is_basic and len(state.bench) < 5]:
        v = [AType.PLAY_BENCH, pid, 0, 0, 0]
        mask[vec2id(v)] = 1

    # ③ エネルギー付与(付けられるエネルギータイプとターゲット（バトル場＋ベンチ分）だけを有効にする)
    for e_type in ENERGY_TYPES:
        for tgt in range(len(state.bench) + 1):  # 0=battle,1‑bench0 …
            v = [AType.ATTACH_ENERGY, e_type, tgt, 0, 0]
            mask[vec2id(v)] = 1


    # …以下、状態に応じて必要な分だけ 1 を立てる …

    return mask