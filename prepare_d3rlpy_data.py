#!/usr/bin/env python3
"""
d3rlpy 用データ前処理
・ポケモンカードAIの対戦ログ(JSON/JSONL) をd3rlpy強化学習用npzデータセットへ変換する前処理
・特徴量ベクトル抽出
・Enum ID を自動で連番インデックス化
・相手公開情報も状態に含める
・private: deck_bag_counts / prize_enum / hand_enum に対応（TopN順序は使用しない）
"""
from __future__ import annotations
import ujson as json, os, glob
# orjson が無い環境でも動くようにフォールバック
from pokepocketsim.action import ACTION_SCHEMAS

try:
    import orjson as _fastjson
    _FAST_JSON = "orjson"
except Exception:
    _fastjson = None
    _FAST_JSON = "ujson"  # 表示用
from typing import Dict, Any, List, Tuple, Iterator
import numpy as np
from tqdm import tqdm
import time
from pokepocketsim.policy.action_encoding import build_encoder_from_files, compute_layout

# JSON パース計測用（iter_jsonl_fast で加算）
_PARSE_MS = 0.0
_PARSED_LINES = 0

_ROOT_DIR = r"D:\date"
# 入力 JSONL 名（private/ids）からステージを決定して BASE_DIR を切替える
_stage_name = os.environ.get("STAGE_NAME")  # 明示指定があれば優先（"v1"|"v2"|"v3"|"v4"|"root"）
if _stage_name not in ("v1", "v2", "v3", "v4", "root"):
    _stage_name = None  # main() 内で最終決定（既定は v4 = ルート直下）

BASE_DIR = _ROOT_DIR  # 既定はルート直下（v4 想定）
VOCAB_PATH = os.path.join(BASE_DIR, "card_id2idx.json")
ACTION_TYPES_PATH = os.path.join(BASE_DIR, "action_types.json")

TYPE_SCHEMAS = {
    7: ["stack_index"],   # 進化(EVOLVE)。0=バトル場/1..=ベンチ など実装に応じて
}

MAX_ARGS  = 3    # 最大引数数
HP_MAX    = 500.0   # HP 正規化上限
DECK_MAX  = 60.0    # 山札枚数正規化上限

# スタジアムIDは実装に合わせて修正。ここは例
STADIUM_IDS_8BENCH = {999999, 999999}  # 例: ゼロの大空洞・スカイフィールドのカードID

# ---------------- ユーティリティ ---------------- #
def iter_jsonl_fast(path: str, batch_lines: int = 50000, measure: bool = False):
    """JSONL を高速にバッチで読む（orjson が無ければ ujson に自動フォールバック）。"""
    global _PARSE_MS, _PARSED_LINES
    use_orjson = _fastjson is not None
    with open(path, "rb", buffering=32*1024*1024) as f:  # 余裕あれば 16MB でもOK
        batch = []
        for line in f:
            if line.strip():
                batch.append(line)
                if len(batch) >= batch_lines:
                    if use_orjson:
                        for ln in batch:
                            try:
                                t0 = time.perf_counter() if measure else 0.0
                                obj = _fastjson.loads(ln)
                                if measure:
                                    _PARSE_MS += (time.perf_counter() - t0)
                                    _PARSED_LINES += 1
                                yield obj
                            except Exception:
                                continue
                    else:
                        for ln in batch:
                            try:
                                t0 = time.perf_counter() if measure else 0.0
                                obj = json.loads(ln.decode("utf-8"))
                                if measure:
                                    _PARSE_MS += (time.perf_counter() - t0)
                                    _PARSED_LINES += 1
                                yield obj
                            except Exception:
                                continue
                    batch.clear()
        if batch:
            if use_orjson:
                for ln in batch:
                    try:
                        t0 = time.perf_counter() if measure else 0.0
                        obj = _fastjson.loads(ln)
                        if measure:
                            _PARSE_MS += (time.perf_counter() - t0)
                            _PARSED_LINES += 1
                        yield obj
                    except Exception:
                        continue
            else:
                for ln in batch:
                    try:
                        t0 = time.perf_counter() if measure else 0.0
                        obj = json.loads(ln.decode("utf-8"))
                        if measure:
                            _PARSE_MS += (time.perf_counter() - t0)
                            _PARSED_LINES += 1
                        yield obj
                    except Exception:
                        continue

def _extract_card_ids(field) -> List[int]:
    """
    手札やベンチ配列からカード ID を平坦化して抽出する。
    0 は空スロットなので **無視**。
    """
    out: List[int] = []
    for x in field:
        if isinstance(x, int):
            if x != 0:
                out.append(x)
        elif isinstance(x, list) and x and isinstance(x[0], int):
            if x[0] != 0:
                out.append(x[0])
        elif isinstance(x, dict):
            cid = x.get("name")
            if isinstance(cid, int) and cid != 0:
                out.append(cid)
    return out

def _extract_ids_from_pairs(pairs) -> List[int]:
    """
    deck_bag_counts 等の [[cid, cnt], ...] から cid 群を取り出す（cnt は捨てる）。
    """
    out: List[int] = []
    if isinstance(pairs, list):
        for itm in pairs:
            if isinstance(itm, (list, tuple)) and len(itm) >= 1 and isinstance(itm[0], int):
                if itm[0] != 0:
                    out.append(itm[0])
            elif isinstance(itm, dict) and "cid" in itm:
                cid = itm.get("cid", 0)
                if isinstance(cid, int) and cid != 0:
                    out.append(cid)
    return out

def _extract_private_vocab_targets(st: Dict[str, Any]) -> List[int]:
    """
    state dict から private 領域のカードID（cid 群）を収集して語彙拡張に使う。
    （TopN順序は対象外）
    """
    out: List[int] = []
    for side_key in ("me_private", "opp_private"):
        pv = st.get(side_key) or {}
        # deck_bag_counts: [[cid, cnt], ...]
        out.extend(_extract_ids_from_pairs(pv.get("deck_bag_counts", [])))
        # prize_enum: [cid, ...]
        out.extend([cid for cid in pv.get("prize_enum", []) if isinstance(cid, int) and cid != 0])
        # hand_enum: 相手側のみだが、あれば拾う
        out.extend([cid for cid in pv.get("hand_enum", []) if isinstance(cid, int) and cid != 0])
    return out

def _pick_reward_from_state(state: Dict[str, Any]) -> float | None:
    if not state:
        return None
    for k in ("shaped_reward", "reward_shaped", "reward"):
        if k in state:
            try:
                return float(state[k])
            except Exception:
                pass
    return None

def pick_reward(sb: Dict[str, Any] | None,
                sa: Dict[str, Any] | None,
                ar: Dict[str, Any] | None) -> float:
    for src in (sa, sb, ar):
        r = _pick_reward_from_state(src)
        if r is not None:
            return r
    return 0.0

def load_vocab(path: str) -> Dict[int, int]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}

def save_vocab(vocab: Dict[int, int]) -> None:
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 語彙数 {len(vocab)} → {VOCAB_PATH}")

def load_action_types(path: str) -> List[int] | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lst = json.load(f)
            try:
                return [int(x) for x in lst]
            except Exception:
                return None
    return None

def save_action_types(types: List[int]) -> None:
    with open(ACTION_TYPES_PATH, "w", encoding="utf-8") as f:
        json.dump(types, f, ensure_ascii=False, indent=2)
    print(f"[INFO] アクション型 {len(types)}件 → {ACTION_TYPES_PATH}")


def update_vocab_with_entries(vocab: Dict[int, int],
                            entries: List[Dict[str, Any]]) -> Dict[int, int]:
    max_idx = max(vocab.values(), default=0)
    def _add(cid: int):
        nonlocal max_idx
        if cid not in vocab:
            max_idx += 1
            vocab[cid] = max_idx

    for rec in entries:
        for key in ("state_before", "state_after"):
            st = rec.get(key) or {}
            # 手札
            for cid in _extract_card_ids(st.get("me", {}).get("hand", [])):
                _add(cid)
            # me/opp アクティブ＋ベンチ
            for section in ("me", "opp"):
                active_id = st.get(section, {}).get("active_pokemon", {}).get("name")
                if isinstance(active_id, int) and active_id != 0:
                    _add(active_id)
                for bp in st.get(section, {}).get("bench_pokemon", []):
                    if isinstance(bp, int) and bp != 0:
                        _add(bp)
                    elif isinstance(bp, list) and bp and isinstance(bp[0], int) and bp[0] != 0:
                        _add(bp[0])
                    elif isinstance(bp, dict):
                        cid = bp.get("name")
                        if isinstance(cid, int) and cid != 0:
                            _add(cid)
            # ★ private 情報（bag / prize / hand）
            for cid in _extract_private_vocab_targets(st):
                _add(cid)
    return vocab

# ========================================================= #
#                    D3RLPyDataPreprocessor                 #
# ========================================================= #
class D3RLPyDataPreprocessor:
    """
    ⚠️ 1 プロセス内で 1 度だけ語彙を確定し、その後は固定
    private 情報:
      - deck_bag_counts   : 語彙サイズのカウントベクトル（/DECK_MAX 正規化）
      - prize_enum/hand_enum: 語彙サイズのカウントベクトル（/DECK_MAX 正規化）
    """
    _vocab_loaded = False
    _card_id2idx: Dict[int, int] = {}

    def __init__(self,
                 entries: List[Dict[str, Any]],
                 fixed_vocab: Dict[int, int] | None = None):
        # ----- 語彙の確定 -----
        if fixed_vocab is not None:
            D3RLPyDataPreprocessor._card_id2idx = fixed_vocab
            D3RLPyDataPreprocessor._vocab_loaded = True
        elif not D3RLPyDataPreprocessor._vocab_loaded:
            vocab = load_vocab(VOCAB_PATH)
            old_len = len(vocab)
            vocab = update_vocab_with_entries(vocab, entries)
            if len(vocab) > old_len:
                save_vocab(vocab)
            D3RLPyDataPreprocessor._card_id2idx = vocab
            D3RLPyDataPreprocessor._vocab_loaded = True

        # インスタンスが参照する語彙
        self.card_id2idx = D3RLPyDataPreprocessor._card_id2idx
        self.vocab_size = len(self.card_id2idx) + 1  # 0 は PAD

        # 行動ベクトル長（action_types.json を唯一の源泉として使用）
        atypes = load_action_types(ACTION_TYPES_PATH) or []
        if 0 not in atypes:
            atypes.append(0)
        self.action_types = sorted(set(int(x) for x in atypes))
        self.type2idx = {aid: i for i, aid in enumerate(self.action_types)}
        self.action_vec_dim = len(self.action_types) + MAX_ARGS + 1

        # --- 追加: private ベクトル用の簡易 LRU キャッシュ ---
        self._bag_pairs_cache = {}
        self._bag_list_cache  = {}
        self._bag_cache_max   = 10000

    # -------------------- State Encoding -------------------- #
    def _get_bench_max(self, state: Dict[str, Any]) -> int:
        for side in ("me", "opp"):
            stadium = state.get(side, {}).get("active_stadium")
            if isinstance(stadium, int) and stadium in STADIUM_IDS_8BENCH:
                return 8
            if isinstance(stadium, dict):
                if stadium.get("name") in STADIUM_IDS_8BENCH:
                    return 8
        return 5

    def _base_num_len(self, obs_bench_max: int) -> int:
        """数値特徴（one-hot以外）のベース長 = 83 + 15 * OBS_BENCH_MAX"""
        return 83 + 15 * obs_bench_max

    def _one_hot11(self, val_id: int) -> list[float]:
        """0-9, 99→長さ11 one-hot（99 は index10 に立てる）"""
        v = [0.0] * 11
        if 0 <= val_id <= 9:
            v[val_id] = 1.0
        elif val_id == 99:
            v[10] = 1.0
        return v

    def _set_hot(self, one_hot_vec: np.ndarray, cid: int | dict | None):
        if isinstance(cid, dict):
            cid = cid.get("name")
        if not isinstance(cid, int) or cid == 0:
            return
        idx = self.card_id2idx.get(cid)
        if idx:
            one_hot_vec[idx] = 1.0

    def _bag_vec_from_pairs(self, pairs: List[List[int]]) -> np.ndarray:
        """
        [[cid, cnt], ...] → 語彙長ベクトル（/DECK_MAX 正規化）
        """
        if not isinstance(pairs, list) or not pairs:
            return np.zeros(self.vocab_size, dtype=np.float32)
        key_items = []
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cid, cnt = int(item[0]), int(item[1])
            elif isinstance(item, dict):
                cid, cnt = int(item.get("cid", 0)), int(item.get("cnt", 0))
            else:
                continue
            if cid != 0 and cnt != 0:
                key_items.append((cid, cnt))
        if not key_items:
            return np.zeros(self.vocab_size, dtype=np.float32)
        key = tuple(key_items)

        hit = self._bag_pairs_cache.get(key)
        if hit is not None:
            return hit

        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for cid, cnt in key_items:
            idx = self.card_id2idx.get(cid)
            if idx:
                vec[idx] += float(cnt) / DECK_MAX

        if len(self._bag_pairs_cache) >= self._bag_cache_max:
            self._bag_pairs_cache.pop(next(iter(self._bag_pairs_cache)))
        self._bag_pairs_cache[key] = vec
        return vec


    def _bag_vec_from_list(self, ids: List[int]) -> np.ndarray:
        """
        [cid, ...] → 語彙長ベクトル（カウントを /DECK_MAX 正規化）
        """
        if not ids:
            return np.zeros(self.vocab_size, dtype=np.float32)
        key = tuple(sorted(int(cid) for cid in ids if isinstance(cid, int) and cid != 0))
        hit = self._bag_list_cache.get(key)
        if hit is not None:
            return hit

        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for cid in key:
            idx = self.card_id2idx.get(cid)
            if idx:
                vec[idx] += 1.0 / DECK_MAX

        if len(self._bag_list_cache) >= self._bag_cache_max:
            self._bag_list_cache.pop(next(iter(self._bag_list_cache)))
        self._bag_list_cache[key] = vec
        return vec


    def _scalar_from_cid(self, cid: int) -> float:
        """カード/行動IDを語彙インデックスに写像して 0..1 のスカラーに"""
        if not isinstance(cid, int) or cid == 0:
            return 0.0
        idx = self.card_id2idx.get(cid, 0)
        return float(idx) / float(max(1, self.vocab_size))

    def encode_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        観測ベクトル = [base one-hot | 数値特徴 | private(me) | private(opp)]
        private(me)  = [bag vec | prize vec]
        private(opp) = [bag vec | prize vec | hand vec]
        """
        OBS_BENCH_MAX = 8

        if not state:
            # 最低限の長さを計算するためにゼロベクトルを構築
            base_one_hot = np.zeros(self.vocab_size, dtype=np.float32)
            base_num_len = self._base_num_len(OBS_BENCH_MAX)
            base_num = np.zeros(base_num_len, dtype=np.float32)
            priv_me  = np.zeros(self.vocab_size + self.vocab_size, dtype=np.float32)                # bag + prize
            priv_op  = np.zeros(self.vocab_size + self.vocab_size + self.vocab_size, dtype=np.float32)  # bag + prize + hand
            return np.concatenate([base_one_hot, base_num, priv_me, priv_op], axis=0)

        me, opp = state.get("me", {}), state.get("opp", {})

        # -------- 1) base one-hot (既存) --------
        one_hot = np.zeros(self.vocab_size, dtype=np.float32)
        # stadium
        self._set_hot(one_hot, me.get("active_stadium"))
        self._set_hot(one_hot, opp.get("active_stadium"))
        # hand / bench / active / opponent
        for cid in _extract_card_ids(me.get("hand", [])):
            self._set_hot(one_hot, cid)
        for bp in me.get("bench_pokemon", []):
            cid = bp if isinstance(bp, int) else (
                bp[0] if isinstance(bp, list) and bp else
                bp.get("name") if isinstance(bp, dict) else None
            )
            self._set_hot(one_hot, cid)
        self._set_hot(one_hot, me.get("active_pokemon", {}).get("name"))
        self._set_hot(one_hot, opp.get("active_pokemon", {}).get("name"))
        for bp in opp.get("bench_pokemon", []):
            cid = bp if isinstance(bp, int) else (
                bp[0] if isinstance(bp, list) and bp else
                bp.get("name") if isinstance(bp, dict) else None
            )
            self._set_hot(one_hot, cid)

        # -------- 2) 数値特徴（既存計算を維持） --------
        BENCH_MAX = self._get_bench_max(state)
        num_feats: List[float] = []

        def _bench_nonempty(bp):
            return (
                isinstance(bp, dict) or
                (isinstance(bp, int) and bp != 0) or
                (isinstance(bp, list) and len(bp) > 0 and bp[0] != 0)
            )

        # --- 自分 ---
        num_feats.append(min(float(me.get("deck_count", 0)), DECK_MAX) / DECK_MAX)
        num_feats.append(min(float(me.get("prize_count", 0)), 6.0) / 6.0)
        num_feats.append(min(float(me.get("hand_count", 0)), 30.0) / 30.0)
        num_feats.append(min(float(me.get("discard_pile_count", 0)), 60.0) / 60.0)
        num_feats.append(float(BENCH_MAX) / 8.0)  # 自分のベンチ最大値

        bench_me = me.get("bench_pokemon", [])
        bench_me_nonempty = [bp for bp in bench_me if _bench_nonempty(bp)]
        num_feats.append(float(len(bench_me_nonempty)) / BENCH_MAX)

        act = me.get("active_pokemon", {})
        hp = float(act.get("hp", 0) or 0)
        hp = max(0.0, min(HP_MAX, hp))
        dmg = float(act.get("damage_counters", 0) or 0) * 10.0
        dmg = max(0.0, min(HP_MAX, dmg))
        num_feats += [
            min(hp, HP_MAX) / HP_MAX,
            min(dmg, HP_MAX) / HP_MAX,
            float(len(act.get("energies", []))) / 20.0,
            float(act.get("can_evolve", 0)),
            float(act.get("retreat_cost", 0) or 0) / 6.0,
            float(len(act.get("conditions", []))) / 5.0,
            float(len(act.get("tools", []))) / 4.0,
            float(act.get("ability_present", 0)),
            float(act.get("ability_used", act.get("has_used_ability", 0))),
        ]
        num_feats += self._one_hot11(act.get("weakness_type", 99))
        num_feats.append(float(act.get("has_weakness", 0)))
        num_feats += self._one_hot11(act.get("resistance_type", 99))
        num_feats.append(float(act.get("has_resistance", 0)))
        num_feats.append(float(act.get("is_basic", 0)))
        num_feats.append(float(act.get("is_ex", 0)))

        # 各ベンチスロット数値（0 埋めスロットはゼロべき）
        for i in range(OBS_BENCH_MAX):
            if i < len(bench_me):
                bp = bench_me[i]
                if isinstance(bp, dict):
                    hp_b = float(bp.get("hp", 0) or 0)
                    hp_b = max(0.0, min(HP_MAX, hp_b))
                    dmg_b = float(bp.get("damage_counters", 0) or 0) * 10.0
                    dmg_b = max(0.0, min(HP_MAX, dmg_b))
                    n_energy = float(len(bp.get("energies", []))) / 20.0
                    can_evolve = float(bp.get("can_evolve", 0))
                    retreat = float(bp.get("retreat_cost", 0) or 0) / 6.0
                    n_tools = float(len(bp.get("tools", []))) / 4.0
                    abi_pres = float(bp.get("ability_present", 0))
                    used_abi = float(bp.get("ability_used", bp.get("has_used_ability", 0)))
                else:
                    hp_b = dmg_b = n_energy = can_evolve = retreat = n_tools = 0.0
                    abi_pres = 0.0
                    used_abi = 0.0
                num_feats += [
                    min(hp_b, HP_MAX) / HP_MAX,
                    min(dmg_b, HP_MAX) / HP_MAX,
                    n_energy,
                    can_evolve,
                    retreat,
                    n_tools,
                    abi_pres,
                    used_abi,
                ]
            else:
                num_feats += [0.0] * 8

        num_feats += [
            float(me.get("supporter_used", 0)),
            float(me.get("energy_attached", 0)),
            float(me.get("retreat_this_turn", 0)),
            float(me.get("stadium_used", 0)),
            float(me.get("was_knocked_out_by_attack_last_turn", 0)),
        ]

        # --- 相手 ---
        if opp:
            num_feats.append(min(float(opp.get("deck_count", 0)), DECK_MAX) / DECK_MAX)
            num_feats.append(min(float(opp.get("prize_count", 0)), 6.0) / 6.0)
            num_feats.append(min(float(opp.get("hand_count", 0)), 30.0) / 30.0)
            num_feats.append(min(float(opp.get("discard_pile_count", 0)), 60.0) / 60.0)
            num_feats.append(float(BENCH_MAX) / 8.0)

            bench_opp = opp.get("bench_pokemon", [])
            bench_opp_nonempty = [bp for bp in bench_opp if _bench_nonempty(bp)]
            num_feats.append(float(len(bench_opp_nonempty)) / BENCH_MAX)

            oact = opp.get("active_pokemon", {})
            hp_o = float(oact.get("hp", 0) or 0)
            hp_o = max(0.0, min(HP_MAX, hp_o))
            dmg_o = float(oact.get("damage_counters", 0) or 0) * 10.0
            dmg_o = max(0.0, min(HP_MAX, dmg_o))
            num_feats += [
                min(hp_o, HP_MAX) / HP_MAX,
                min(dmg_o, HP_MAX) / HP_MAX,
                float(len(oact.get("energies", []))) / 20.0,
                float(oact.get("retreat_cost", 0) or 0) / 6.0,
                float(len(oact.get("conditions", []))) / 5.0,
                float(len(oact.get("tools", []))) / 4.0,
            ]
            # 弱点/抵抗など
            num_feats += self._one_hot11(oact.get("weakness_type", 99))
            num_feats.append(float(oact.get("has_weakness", 0)))
            num_feats += self._one_hot11(oact.get("resistance_type", 99))
            num_feats.append(float(oact.get("has_resistance", 0)))
            num_feats.append(float(oact.get("is_basic", 0)))
            num_feats.append(float(oact.get("is_ex", 0)))

            for i in range(OBS_BENCH_MAX):
                if i < len(bench_opp):
                    bp = bench_opp[i]
                    if isinstance(bp, dict):
                        hp_b = float(bp.get("hp", 0) or 0)
                        hp_b = max(0.0, min(HP_MAX, hp_b))
                        dmg_b = float(bp.get("damage_counters", 0) or 0) * 10.0
                        dmg_b = max(0.0, min(HP_MAX, dmg_b))
                        n_energy = float(len(bp.get("energies", []))) / 20.0
                        can_evolve = 1.0 if bp.get("can_evolve", 0) else 0.0
                        retreat = float(bp.get("retreat_cost", 0) or 0) / 6.0
                        n_tools = float(len(bp.get("tools", []))) / 4.0
                        used_abi = 1.0 if bp.get("has_used_ability", 0) else 0.0
                    else:
                        hp_b = dmg_b = n_energy = can_evolve = retreat = n_tools = used_abi = 0.0
                    num_feats += [
                        min(hp_b, HP_MAX) / HP_MAX,
                        min(dmg_b, HP_MAX) / HP_MAX,
                        n_energy,
                        can_evolve,
                        retreat,
                        n_tools,
                        used_abi,
                    ]
                else:
                    num_feats += [0.0] * 7
        else:
            num_feats += [0.0] * (38 + 7 * OBS_BENCH_MAX)

        base_vec = np.concatenate([one_hot, np.asarray(num_feats, dtype=np.float32)], axis=0)

        # -------- 3) private features --------
        me_pv  = state.get("me_private")  or {}
        opp_pv = state.get("opp_private") or {}

        # me side（TopN順序なし）
        me_bag   = self._bag_vec_from_pairs(me_pv.get("deck_bag_counts", []))
        me_prize = self._bag_vec_from_list(me_pv.get("prize_enum", []))
        priv_me  = np.concatenate([me_bag, me_prize], axis=0)

        # opp side（TopN順序なし）
        opp_bag   = self._bag_vec_from_pairs(opp_pv.get("deck_bag_counts", []))
        opp_prize = self._bag_vec_from_list(opp_pv.get("prize_enum", []))
        opp_hand  = self._bag_vec_from_list(opp_pv.get("hand_enum", []))
        priv_opp  = np.concatenate([opp_bag, opp_prize, opp_hand], axis=0)

        # 結合
        vec = np.concatenate([base_vec, priv_me, priv_opp], axis=0)

        # 期待次元（固定）
        base_num_len = self._base_num_len(OBS_BENCH_MAX)
        base_obs_dim = self.vocab_size + base_num_len
        expected_dim = base_obs_dim + (5 * self.vocab_size)   # me:2 + opp:3 = 5
        # 必要ならパディング/切り詰め
        if vec.shape[0] < expected_dim:
            vec = np.pad(vec, (0, expected_dim - vec.shape[0]))
        elif vec.shape[0] > expected_dim:
            vec = vec[:expected_dim]

        return vec

    # -------------------- Action Encoding -------------------- #
    def encode_action(self, act_res: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.action_vec_dim, dtype=np.float32)
        action = act_res.get("action") or []
        if not action:
            return vec

        # 第一要素=アクション「型」、第二要素=カード/技などのID
        action_type = action[0] if len(action) > 0 else 0
        action_id   = action[1] if len(action) > 1 else action_type

        # ① one-hot（アクション型ベース、未登録は "その他(0)"）
        vec[self.type2idx.get(action_type, self.type2idx[0])] = 1.0

        # ② 引数スキーマの決定：カードID優先 → なければアクション型（例: 進化=7）
        arg_keys = ACTION_SCHEMAS.get(action_id)
        if not arg_keys:
            arg_keys = TYPE_SCHEMAS.get(action_type, [])

        # ③ 引数のエンコード
        slot = 0
        for key in arg_keys:
            if slot >= MAX_ARGS:
                break
            val = act_res.get(key, 0)
            if isinstance(val, int) and ("bench_idx" in key or "stack_index" in key or "target_index" in key):
                vec[len(self.action_types) + slot] = float(val) / 8.0  # 必要なら分母を実装に合わせて調整
                slot += 1
            elif isinstance(val, int) and val != 0:
                idx = self.card_id2idx.get(val, 0)
                vec[len(self.action_types) + slot] = float(idx) / float(max(1, self.vocab_size))
                slot += 1


        # ④ 末尾に「生の action_id スカラー」を常に付与（未登録カードでも識別力を確保）
        vec[-1] = self._scalar_from_cid(action_id)
        return vec

    def encode_action_from_vec(self, action_vec: List[int]) -> np.ndarray:
        """
        ★ 変更：5 整数ベクトル [type, main, p3, p4, p5] をスキーマに沿って
        act_res の各キーに展開してから encode_action に渡す。
        """
        vec = list(action_vec) if isinstance(action_vec, (list, tuple)) else []
        vec = (vec + [0, 0, 0, 0, 0])[:5]
        action_type, action_id, a1, a2, a3 = vec

        act_res = {"action": vec}

        # スキーマ決定（カードID優先 → 型）
        arg_keys = ACTION_SCHEMAS.get(action_id)
        if not arg_keys:
            arg_keys = TYPE_SCHEMAS.get(action_type, [])

        # 順に詰め替え（最大 MAX_ARGS まで）
        vals = [a1, a2, a3]
        for i, key in enumerate(arg_keys[:MAX_ARGS]):
            act_res[key] = vals[i]

        return self.encode_action(act_res)


    # -------------------- Transition List -------------------- #
    def process(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trans = []
        for e in entries:
            sb = e.get("state_before") or {}
            sa = e.get("state_after")  or {}
            ar = e.get("action_result") or {}

            if "reward" in e:
                reward = float(e["reward"])
            else:
                reward = pick_reward(sb, sa, ar)

            terminal = int(e.get("done", sa.get("done", 0)))

            trans.append(dict(
                observation      = self.encode_state(sb),
                action           = self.encode_action(ar),
                reward           = reward,
                next_observation = self.encode_state(sa),
                terminal         = terminal,
            ))
        return trans

    def save_dataset(self, out_path: str, entries: List[Dict[str, Any]]) -> None:
        trans = self.process(entries)
        ds = {
            "observations":      np.array([t["observation"]      for t in trans]),
            "actions":           np.stack([t["action"]           for t in trans]),  # (N, action_vec_dim)
            "rewards":           np.array([t["reward"] for t in trans], dtype=np.float32),
            "next_observations": np.array([t["next_observation"] for t in trans]),
            "terminals":         np.array([t["terminal"]         for t in trans]),
        }
        state_dim = ds["observations"].shape[1]
        np.savez_compressed(out_path, **ds)
        print(f"[DONE] {out_path} — 遷移 {len(trans)} 件, 状態次元 {state_dim}, アクション次元 {self.action_vec_dim}")

# ========================================================= #
#                 ★ サブステップも読むイテレータ            #
# ========================================================= #
def iterate_examples_from_jsonl(
    jsonl_path: str,
    include_substeps: bool = True,
) -> Iterator[Tuple[Dict[str, Any], List[int], Dict[str, Any], float, int]]:
    """
    JSONL を1行ずつ読み、(state_before, action_vec(=5ints), state_after, reward, done)
    を yield する。include_substeps=True のとき、action_result.substeps も展開する。
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue

            sb = e.get("state_before")
            sa = e.get("state_after")
            if not isinstance(sb, dict) or not isinstance(sa, dict):
                continue

            ar = e.get("action_result") or {}
            # トップレベルの action_vec（優先順位: 'action' → 'macro' → 'action_vec'）
            top = None
            for k in ("action", "macro", "action_vec"):
                v = ar.get(k)
                if isinstance(v, list):
                    top = v
                    break

            # 報酬・終局
            reward_top = float(e.get("reward", pick_reward(sb, sa, ar)))
            done_top   = int(e.get("done", sa.get("done", 0)))
            meta       = e.get("meta") or {}
            is_terminal = (done_top == 1) or bool(meta.get("terminal"))
            # ★ 終局ダミー行の救済：アクションが無くても 0 ベクトルで流す
            if not isinstance(top, list) and is_terminal:
                top = [0, 0, 0, 0, 0]
            if isinstance(top, list):
                yield sb, top, sa, reward_top, done_top

            # substeps
            if include_substeps and isinstance(ar.get("substeps"), list):
                for step in ar["substeps"]:
                    if not isinstance(step, dict):
                        continue
                    sbs = step.get("state_before")
                    sas = step.get("state_after")
                    av  = step.get("action_vec")
                    if not (isinstance(sbs, dict) and isinstance(sas, dict) and isinstance(av, list)):
                        continue
                    rew_s = float(step.get("reward", 0.0) or 0.0)
                    done_s = int(step.get("done", 0) or 0)
                    yield sbs, av, sas, rew_s, done_s

# ========================================================= #
#        ★ 追加：B案（z後に重み）用 post_scale を作る関数     #
# ========================================================= #
def _build_post_scale(vocab_size: int, obs_bench_max: int, total_dim: int) -> np.ndarray:
    """
    Zスコア後の係数ベクトル。自分側だけを強調する。
      環境変数（例）:
        OBS_W_VOCAB, OBS_W_NUMERIC, OBS_W_PRIV_ME, OBS_W_PRIV_OPP
        OBS_W_MY_HP, OBS_W_MY_BENCH_FILL, OBS_W_MY_ACTIVE_ENERGY, OBS_W_MY_ENERGY_ATTACHED
    """
    try:
        w_vocab   = float(os.environ.get("OBS_W_VOCAB", "1.0"))
        w_numeric = float(os.environ.get("OBS_W_NUMERIC", "1.0"))
        w_pme     = float(os.environ.get("OBS_W_PRIV_ME", "1.0"))
        w_popp    = float(os.environ.get("OBS_W_PRIV_OPP", "1.0"))
        # ★ 自分側のみのピンポイント強調
        w_my_hp            = float(os.environ.get("OBS_W_MY_HP", "1.1"))                 # 自分アクティブHP
        w_my_bench_fill    = float(os.environ.get("OBS_W_MY_BENCH_FILL", "1.1"))         # 自分ベンチ充足率
        w_my_act_energy    = float(os.environ.get("OBS_W_MY_ACTIVE_ENERGY", "1.1"))      # 自分アクティブのエネ枚数
        w_my_energy_attached = float(os.environ.get("OBS_W_MY_ENERGY_ATTACHED", "1.1"))  # 今ターン付けたエネフラグ
    except Exception:
        w_vocab = w_numeric = w_pme = w_popp = 1.0
        w_my_hp = w_my_bench_fill = w_my_act_energy = w_my_energy_attached = 1.0

    base_num_len = 83 + 15 * obs_bench_max  # numeric全体（自分+相手）
    V = vocab_size

    post = np.ones((total_dim,), dtype=np.float32)

    # ---- ブロック係数（従来通り）----
    post[0:V] *= w_vocab
    post[V:V+base_num_len] *= w_numeric
    start = V + base_num_len
    post[start : start + 2*V] *= w_pme                  # private(me)
    start = V + base_num_len + 2*V
    post[start : start + 3*V] *= w_popp                 # private(opp)

    # ---- 自分側 numeric 内の個別強調（相手側は触らない）----
    # encode_state の並びに基づく「自分側 numeric の相対インデックス」
    # 自分側 numeric は base_num_len の先頭から始まり、相手側は後続。
    # 自分側 numeric の並び（抜粋）:
    #   p0: deck, p1: prize, p2: hand, p3: discard, p4: bench_max/8, p5: bench_fill(=非空/最大)
    #   p6: active.hp, p7: active.damage, p8: active.energies, ...（以下省略）
    #   ... ベンチ8スロット×8指標 ...
    #   p105: supporter_used, p106: energy_attached, p107: retreat_this_turn, p108: stadium_used, p109: was_KO_last_turn
    my_numeric_start = V  # numeric ブロックの先頭
    p_bench_fill     = 5
    p_hp             = 6
    p_active_energy  = 8
    p_energy_attached = 106

    # 自分側のみ倍率を掛ける
    post[my_numeric_start + p_bench_fill]     *= w_my_bench_fill
    post[my_numeric_start + p_hp]             *= w_my_hp
    post[my_numeric_start + p_active_energy]  *= w_my_act_energy
    post[my_numeric_start + p_energy_attached]*= w_my_energy_attached

    return post

# ========================================================= #
#                         CLI 部分                          #
# ========================================================= #
def stream_log(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def iterate_examples_from_jsonl_fast(jsonl_path: str, include_substeps: bool = True):
    for e in iter_jsonl_fast(jsonl_path, batch_lines=10000):
        sb = e.get("state_before"); sa = e.get("state_after")
        if not isinstance(sb, dict) or not isinstance(sa, dict):
            continue
        ar = e.get("action_result") or {}
        top = None
        for k in ("action", "macro", "action_vec"):
            v = ar.get(k)
            if isinstance(v, list):
                top = v; break
        reward_top = float(e.get("reward", pick_reward(sb, sa, ar)))
        done_top   = int(e.get("done", sa.get("done", 0)))
        meta       = e.get("meta") or {}
        is_terminal = (done_top == 1) or bool(meta.get("terminal"))
        # ★ 終局ダミー行の救済：アクションが無くても 0 ベクトルで流す
        if not isinstance(top, list) and is_terminal:
            top = [0, 0, 0, 0, 0]
        if isinstance(top, list):
            yield sb, top, sa, reward_top, done_top
        if include_substeps and isinstance(ar.get("substeps"), list):
            for step in ar["substeps"]:
                if not isinstance(step, dict): continue
                sbs = step.get("state_before"); sas = step.get("state_after"); av  = step.get("action_vec")
                if not (isinstance(sbs, dict) and isinstance(sas, dict) and isinstance(av, list)): continue
                rew_s = float(step.get("reward", 0.0) or 0.0)
                done_s = int(step.get("done", 0) or 0)
                yield sbs, av, sas, rew_s, done_s

def main():
    global BASE_DIR, VOCAB_PATH, ACTION_TYPES_PATH

    # 入力ファイル名を名前で振り分け（環境変数でも指定可）
    in_name = os.environ.get("INPUT_JSONL_NAME")
    if not in_name:
        # 既定: private → ids の優先順で自動検出（_ROOT_DIR 直下を探索）
        for _cand in ("ai_vs_ai_match_all_private_ids.jsonl", "ai_vs_ai_match_all_ids.jsonl"):
            if os.path.exists(os.path.join(_ROOT_DIR, _cand)):
                in_name = _cand
                break
        if not in_name:
            in_name = "ai_vs_ai_match_all_ids.jsonl"

    # ステージ決定（明示 STAGE_NAME 優先）
    stage = os.environ.get("STAGE_NAME")
    if stage not in ("v1", "v2", "v3", "v4", "root"):
        # 既定は v4（CQL 段階）＝ ルート直下に集約
        stage = "v4"

    # BASE_DIR 再設定：
    #  - v4 / root → ルート直下（D:\date）
    #  - v1/v2/v3 → 従来通りサブディレクトリ配下（D:\date\<stage>）
    if stage in ("v4", "root"):
        BASE_DIR = _ROOT_DIR
    else:
        BASE_DIR = os.path.join(_ROOT_DIR, stage)
    os.makedirs(BASE_DIR, exist_ok=True)

    VOCAB_PATH = os.path.join(BASE_DIR, "card_id2idx.json")
    ACTION_TYPES_PATH = os.path.join(BASE_DIR, "action_types.json")

    VOCAB_ONLY = in_name.endswith("_private_ids.jsonl")  # private は語彙更新のみ
    in_file = os.path.join(_ROOT_DIR, in_name)

    # 一時バッチ保存・出力
    out_dir = os.path.join(BASE_DIR, "state_return_chunks")
    os.makedirs(out_dir, exist_ok=True)


    batch_size = 30000
    out_file = "d3rlpy_dataset_all.npz"
    INCLUDE_SUBSTEPS = True

    # 追加：トグル（必要に応じて False にすると高速化）
    PBRS_CHECK_ENABLED = True
    SCAN_SUBSTEPS_IN_FIRST_PASS = True  # 語彙構築時に substeps も走査するか

    # 情報表示（JSONエンジンと入力サイズ）
    try:
        size = os.path.getsize(in_file)
        print(f"[INFO] JSON engine={_FAST_JSON}, file={in_file}, size={size/1e9:.2f} GB")
    except Exception:
        print(f"[INFO] JSON engine={_FAST_JSON}, file={in_file}")
    print(f"[INFO] temp chunk dir: {out_dir}")

    # ---- 1st pass: 語彙構築 + PBRSチェック（同時に） ----
    print("[INFO] 語彙リスト構築 + PBRSチェック（1パス）...")
    vocab: Dict[int, int] = {}
    action_types_set = set([0])  # ★ 先頭(type)の集合を収集して固定化する
    def _add(cid: int):
        if isinstance(cid, int) and cid != 0:
            vocab.setdefault(cid, len(vocab) + 1)

    # ストリーミング・チェッカー
    try:
        from log_checker import PbrsChecker
        checker = PbrsChecker(max_errors=200) if PBRS_CHECK_ENABLED else None
    except Exception:
        class _DummyChecker:
            def __init__(self, max_errors=200): self.errors=[]; self.total=0; self.max_errors=max_errors
            def feed(self, rec): self.total += 1; return True
            def done(self): return self.errors, self.total
        checker = _DummyChecker() if PBRS_CHECK_ENABLED else None

    for entry in iter_jsonl_fast(in_file, batch_lines=20000, measure=True):

        # ---- アクション型の収集（トップレベル）----
        ar = entry.get("action_result") or {}
        top = None
        for k in ("action", "macro", "action_vec"):
            v = ar.get(k)
            if isinstance(v, list) and len(v) >= 1:
                top = v
                try:
                    action_types_set.add(int(v[0]))
                except Exception:
                    pass
                break

        # ---- アクション型の収集（substeps）----
        if SCAN_SUBSTEPS_IN_FIRST_PASS and isinstance(ar.get("substeps"), list):
            for step in ar["substeps"]:
                if not isinstance(step, dict):
                    continue
                av = step.get("action_vec")
                if isinstance(av, list) and len(av) >= 1:
                    try:
                        action_types_set.add(int(av[0]))
                    except Exception:
                        pass

        # ---- PBRSチェック ----
        if checker is not None:
            res = checker.feed(entry)
            # 早期打ち切り（エラー閾値）
            if hasattr(checker, "errors") and len(checker.errors) >= getattr(checker, "max_errors", 999999):
                break

        # ---- 語彙更新（元の構築ロジックと同じ）----
        for key in ("state_before", "state_after"):
            st = entry.get(key)
            if not st:
                continue
            for cid in _extract_card_ids(st.get("me", {}).get("hand", [])):
                _add(cid)
            for section in ("me", "opp"):
                active_id = st.get(section, {}).get("active_pokemon", {}).get("name")
                _add(active_id)
                for bp in st.get(section, {}).get("bench_pokemon", []):
                    if isinstance(bp, int) and bp != 0:
                        _add(bp)
                    elif isinstance(bp, list) and bp and isinstance(bp[0], int) and bp[0] != 0:
                        _add(bp[0])
                    elif isinstance(bp, dict):
                        cid = bp.get("name"); _add(cid)
            for cid in _extract_private_vocab_targets(st):
                _add(cid)

        # substeps も語彙へ（トグル）
        if SCAN_SUBSTEPS_IN_FIRST_PASS:
            ar = entry.get("action_result") or {}
            if isinstance(ar.get("substeps"), list):
                for step in ar["substeps"]:
                    if not isinstance(step, dict): continue
                    for key in ("state_before", "state_after"):
                        st2 = step.get(key)
                        if not st2: continue
                        for cid in _extract_card_ids(st2.get("me", {}).get("hand", [])):
                            _add(cid)
                        for section in ("me", "opp"):
                            active_id = st2.get(section, {}).get("active_pokemon", {}).get("name")
                            _add(active_id)
                            for bp in st2.get(section, {}).get("bench_pokemon", []):
                                if isinstance(bp, int) and bp != 0:
                                    _add(bp)
                                elif isinstance(bp, list) and bp and isinstance(bp[0], int) and bp[0] != 0:
                                    _add(bp[0])
                                elif isinstance(bp, dict):
                                    cid = bp.get("name"); _add(cid)
                        for cid in _extract_private_vocab_targets(st2):
                            _add(cid)

    if checker is not None:
        errs, total = checker.done()
        print(f"[PBRS] checked={total}, errors={len(errs)}")
        if errs:
            # 必要なら最初の数件だけ可視化
            for e in errs[:5]:
                print(f"[PBRS-ERR] {e}")
    else:
        print("[PBRS] skipped")

    # JSON パース時間の内訳を表示
    try:
        if _PARSED_LINES > 0:
            print(f"[PERF] parse: {_PARSED_LINES} lines, {_PARSE_MS:.3f} s, throughput ≈ {int(_PARSED_LINES/max(1e-9,_PARSE_MS))} lines/s")
    except Exception:
        pass

    print(f"[INFO] 語彙リスト {len(vocab)}件")
    old = load_vocab(VOCAB_PATH)
    if len(vocab) > len(old):
        save_vocab(vocab)
    else:
        print("[INFO] 語彙に変更なし → 既存ファイルを保持")

    # ★ アクション型の保存（順序は昇順・0 を含める）
    try:
        new_types = sorted(action_types_set)
        old_types = load_action_types(ACTION_TYPES_PATH) or []
        if not old_types or old_types != new_types:
            save_action_types(new_types)
        else:
            print("[INFO] アクション型に変更なし → 既存ファイルを保持")
        # 強制チェック（K=1 なら五要素=5次元に落ちる）
        print(f"[CHECK] action_types(K)={len(new_types)}  sample={new_types[:10]}")
        if len(new_types) <= 1:
            raise RuntimeError(
                "action_types.json の内容が不正です（K<=1）。ログの 'action'/'macro'/'action_vec' 先頭要素が拾えていません。"
            )
    except Exception as e:
        print(f"[WARN] action_types.json の保存に失敗: {e}")
        raise

    # ★ private 語彙更新のみモード → ここで終了
    if VOCAB_ONLY:
        print("[INFO] private 語彙更新のみ: card_id2idx.json / action_types.json の更新のみ実行し、データセット生成はスキップします。")
        return

    # ---------------- 2nd pass: ストリーミングでバッチ保存 ----------------
    state_dim = None
    action_dim = None

    states, actions, rewards, next_states, terminals = [], [], [], [], []
    batch_idx = 0
    total = 0

    # 学習と推論で“唯一の”アクションエンコードを使う（K+4 次元）
    enc_fn, card_id2idx, action_types, (K, V, ACTION_VEC_DIM) = build_encoder_from_files(
        vocab_path=VOCAB_PATH,
        action_types_path=ACTION_TYPES_PATH,
        ACTION_SCHEMAS=ACTION_SCHEMAS,
        TYPE_SCHEMAS=TYPE_SCHEMAS,
        max_args=3,
    )

    # 状態エンコード用プリプロセッサ（語彙を enc と完全同期させる）
    pre = D3RLPyDataPreprocessor([{}], fixed_vocab=card_id2idx)

    # 監査ログ（要件8 相当）
    print(f"[PREP] vocab_size(V-1)={V-1}")
    expected_state_dim = 6*V + 204
    print(f"[PREP] expected_state_dim={expected_state_dim}")
    print(f"[PREP] action_vec_dim=K+4={K}+4={ACTION_VEC_DIM} (K=len(action_types.json)={K})")

    # ★ 追記：出現順で action_vec(5ints) にIDを振るマップ
    spec2id: Dict[Tuple[int, int, int, int, int], int] = {}

    # 五要素 → 19次元エンコード（唯一の関数）
    def _to19(five_ints):
        a5 = (list(five_ints) + [0, 0, 0, 0, 0])[:5]
        return enc_fn(a5)

    for sb, act_vec, sa, r, d in tqdm(
        iterate_examples_from_jsonl_fast(in_file, INCLUDE_SUBSTEPS),
        desc=f"2nd pass ({_FAST_JSON})"):
        states.append(pre.encode_state(sb))
        actions.append(np.asarray(_to19(act_vec), dtype=np.float32))  # ← 共通エンコーダ
        rewards.append(float(r))
        next_states.append(pre.encode_state(sa))
        terminals.append(int(d))

        # ★ 追記：action_id マップ用に 5 整数タプルへ正規化し、初出順にIDを採番
        spec = tuple(int(x) for x in (act_vec + [0, 0, 0, 0, 0])[:5])
        if spec not in spec2id:
            spec2id[spec] = len(spec2id)

        total += 1
        if len(states) == batch_size:
            if state_dim is None:
                state_dim = states[-1].shape[0]
                action_dim = actions[-1].shape[0]
            fname = os.path.join(out_dir, f"dataset_{batch_idx:03d}.npz")
            np.savez(fname,
                observations=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                next_observations=np.array(next_states),
                terminals=np.array(terminals)
            )
            print(f"[WRITE] {fname}: {len(states)}件")
            batch_idx += 1
            states, actions, rewards, next_states, terminals = [], [], [], [], []

    # 残り
    if states:
        fname = os.path.join(out_dir, f"dataset_{batch_idx:03d}.npz")
        np.savez(fname,
            observations=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_observations=np.array(next_states),
            terminals=np.array(terminals)
        )
        print(f"[WRITE] {fname}: {len(states)}件")

    # ---------------- 3rd: バッチ自動結合 ----------------
    print("[INFO] バッチ結合中...")
    npz_files = sorted(glob.glob(os.path.join(out_dir, "dataset_*.npz")))
    all_obs, all_actions, all_rewards, all_next_obs, all_terminals = [], [], [], [], []
    for fname in npz_files:
        with np.load(fname) as data:
            all_obs.append(data["observations"])
            all_actions.append(data["actions"])
            all_rewards.append(data["rewards"])
            all_next_obs.append(data["next_observations"])
            all_terminals.append(data["terminals"])
    all_obs = np.concatenate(all_obs, axis=0) if all_obs else np.empty((0, pre.encode_state({}).shape[0]), dtype=np.float32)
    all_actions = np.concatenate(all_actions, axis=0) if all_actions else np.empty((0, pre.action_vec_dim), dtype=np.float32)
    all_rewards = np.concatenate(all_rewards, axis=0) if all_rewards else np.empty((0,), dtype=np.float32)
    all_next_obs = np.concatenate(all_next_obs, axis=0) if all_next_obs else np.empty((0, pre.encode_state({}).shape[0]), dtype=np.float32)
    all_terminals = np.concatenate(all_terminals, axis=0) if all_terminals else np.empty((0,), dtype=np.int32)

    out_file = os.path.join(BASE_DIR, "d3rlpy_dataset_all.npz")
    np.savez_compressed(out_file,
        observations=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        next_observations=all_next_obs,
        terminals=all_terminals
    )
    print(f"[DONE] {out_file} 出力: {all_obs.shape[0]}件, 状態次元 {all_obs.shape[1]}, アクション次元 {all_actions.shape[1]}")
    if all_actions.shape[1] != ACTION_VEC_DIM:
        raise RuntimeError(
            f"actions の次元が不一致です（期待 {ACTION_VEC_DIM}, 実際 {all_actions.shape[1]}）。"
            "唯一のエンコード（build_encoder_from_files）の出力と食い違っています。"
        )

    # ★ 追記：scaler.npz を保存（学習時と同一統計を推論で使う想定）
    if all_obs.shape[0] > 0:
        mean = all_obs.mean(axis=0).astype(np.float32)
        std  = all_obs.std(axis=0).astype(np.float32)
        clip_min = np.float32(-5.0)
        clip_max = np.float32(5.0)

        # ==== ここから追加：post_scale を作って一緒に保存（B案対応） ====
        V = pre.vocab_size                             # (= len(card_id2idx) + 1)
        obs_bench_max = 8                              # encode_state 側の固定と合わせる
        total_dim = all_obs.shape[1]
        post_scale = _build_post_scale(V, obs_bench_max, total_dim)
        # ================================================================

        np.savez(
            os.path.join(BASE_DIR, "scaler.npz"),
            mean=mean,
            std=std,
            clip_min=clip_min,
            clip_max=clip_max,
            post_scale=post_scale,   # ← 追加：B案で使用
        )
        print(f"[INFO] scaler.npz 保存: mean/std 形状 {mean.shape}/{std.shape}, clip=[{float(clip_min)},{float(clip_max)}], post_scale.shape={post_scale.shape}")

        # ★ ここから追加：obs_mask.npy を作成（partial 用マスク）
        try:
            V = pre.vocab_size  # (= len(card_id2idx) + 1)
            base_num_len = pre._base_num_len(8)  # OBS_BENCH_MAX は encode_state 内で 8 固定
            base_obs_dim = V + base_num_len
            private_dim  = 5 * V  # me:2 + opp:3
            expected_dim = base_obs_dim + private_dim
            if all_obs.shape[1] != expected_dim:
                print(f"[WARN] obs_mask 期待次元と一致しません: expected={expected_dim}, actual={all_obs.shape[1]}. マスクを全1で出力します。")
                mask = np.ones((all_obs.shape[1],), dtype=np.float32)
            else:
                mask = np.concatenate([
                    np.ones((base_obs_dim,), dtype=np.float32),   # 公開（可視）
                    np.zeros((private_dim,), dtype=np.float32),   # private 全無効
                ], axis=0)
            np.save(os.path.join(BASE_DIR, "obs_mask.npy"), mask.astype(np.float32))
            on = int(mask.sum())
            print(f"[INFO] obs_mask.npy 保存: shape={mask.shape}, 有効次元={on}/{mask.shape[0]}")
        except Exception as e:
            print(f"[WARN] obs_mask.npy の作成に失敗: {e}")

    else:
        print("[WARN] 観測が空のため scaler.npz は作成しません")

    # ★ 追記：id_action_map.json を保存（action_id ↔ 5ints）
    id_map_path = os.path.join(BASE_DIR, "id_action_map.json")
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump({str(v): list(k) for k, v in spec2id.items()}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] id_action_map.json 保存: {len(spec2id)} 個の一意 action_vec")

    # --- バッチファイルの削除 ---
    import shutil
    for fname in npz_files:
        try:
            os.remove(fname)
            print(f"[DEL] {fname}")
        except Exception as e:
            print(f"[WARN] {fname} の削除失敗: {e}")
    try:
        shutil.rmtree(out_dir)
        print(f"[DEL] ディレクトリ {out_dir} 削除")
    except Exception as e:
        print(f"[WARN] ディレクトリ削除失敗: {e}")

if __name__ == "__main__":
    main()
