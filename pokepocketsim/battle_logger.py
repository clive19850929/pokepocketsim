# pokepocketsim/battle_logger.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import OrderedDict, Counter

if TYPE_CHECKING:
    from .player import Player
    from .action import Action

TYPE_ID = {
    "grass": 0, "fire": 1, "water": 2, "lightning": 3,
    "psychic": 4, "fighting": 5, "dark": 6, "metal": 7,
    "colorless": 8, "dragon": 9
}
NONE_TYPE = 99          # 弱点・抵抗力が「なし」のとき

def type_to_id(val):
    """タイプ名 → 0-9 の整数（未知・None は 99）"""
    if val is None:
        return NONE_TYPE

    if hasattr(val, "name"):
        key = val.name
    else:
        key = str(val)
        if "." in key:  # "EnergyType.FIRE" → "FIRE"
            key = key.split(".")[-1]

    key = key.strip().lower()
    if key == "none":
        return NONE_TYPE
    return TYPE_ID.get(key, NONE_TYPE)

def _hp0(x):
    return max(0, (x if x is not None else 0))

def _enum_from_card(obj) -> int:
    """
    カードオブジェクトから学習用の整数ID（enum）を取り出す。
    - Card.card_enum が int のときはそれを返す
    - Enum の value が (id, name) のタプルなら先頭を使う
    - 無ければ 0 を返す
    """
    try:
        enum_ = getattr(obj, "card_enum", None)
        if isinstance(enum_, int):
            return enum_
        if hasattr(enum_, "value"):
            v = enum_.value
            if isinstance(v, (tuple, list)) and v:
                return int(v[0])
            if isinstance(v, int):
                return v
        # fallback
        return int(getattr(obj, "id", 0)) or 0
    except Exception:
        return 0

def _enum_list_from_zone(zone) -> List[int]:
    """
    zone に含まれるカードを enum 配列に変換。
    - zone は通常 list[Card]
    - 進化スタックなど list[list[Card]] の場合は最上段（[-1]）だけ採用
    """
    out: List[int] = []
    if not zone:
        return out
    for c in zone:
        if isinstance(c, list) and c:
            out.append(_enum_from_card(c[-1]))
        else:
            out.append(_enum_from_card(c))
    return out

def _bag_counts_from_enums(enums: List[int], skip_front: int = 0) -> List[List[int]]:
    """
    山札の残りを「順序なしの多重集合」として [ [card_id, count], ... ] 形式で返す。
    JSONの辞書キー文字列化を避けるためリスト化している。
    ※ 先頭 skip_front の除外は行わない（常に全デッキで集計）
    """
    rest = list(enums)
    c = Counter(rest)
    # 0 は不明/プレースホルダの可能性があるので落としておく（任意）
    c.pop(0, None)
    # 安定性のため card_id 昇順で出す
    return [[cid, int(cnt)] for cid, cnt in sorted(c.items())]

# ここ（クラス定義の前 or 後でもOK）に置く
def _effective_retreat(top) -> Optional[int]:
    f = getattr(top, "get_effective_retreat_cost", None)
    return f() if callable(f) else getattr(top, "retreat_cost", None)


class BattleLogger:
    """
    対戦ログとML用JSON出力を司るクラス。
    Player から委譲されて使われる想定。
    """
    def __init__(self, player: "Player") -> None:
        self.player = player
        # サブステップ用ワーク
        self._pending_substeps: List[dict] = []
        self._current_substep: Optional[dict] = None
        # 直前の合法手（5整数ベクトル）
        self._last_legal_actions_before: List[List[int]] = []
        self._terminal_logged: bool = False
        self._last_state_after = None

    # --- 互換目的の無効ガード（外部からの呼び出しがあっても常に False を返す）---
    def _primary_guard(self) -> bool:
        return False

    # ---- 外部（Player）から参照/設定されるフィールドの委譲 ----
    @property
    def last_legal_actions_before(self) -> List[List[int]]:
        return self._last_legal_actions_before

    @last_legal_actions_before.setter
    def last_legal_actions_before(self, v: List[List[int]]):
        self._last_legal_actions_before = v

    @property
    def pending_substeps(self) -> List[dict]:
        return self._pending_substeps

    # ---------------- 基本入出力（ファイルのみ） ----------------
# 修正後（_terminal_guard：フラッシュ中だけガードを無効化）
    def _terminal_guard(self) -> bool:
        """終局後の一切の追記を防ぐ（True を返したら出力中止）"""
        p = self.player
        try:
            # バッファ書き戻し中はガードを一時的に無効化する
            if getattr(self, "_is_flushing_buffer", False):
                return False
            if hasattr(p, "match") and p.match and getattr(p.match, "_terminal_emitted", False):
                return True
        except Exception:
            pass
        return False

    def log_print(self, *args, **kwargs) -> None:
        """通常ログファイルへ出力（標準出力はしない）"""
        p = self.player

        # === 追記ガード ===
        if self._terminal_guard():
            return

        if hasattr(p, 'match') and p.match and getattr(p.match, 'log_file', None):
            fp = getattr(p.match, "_log_fp", None)
            if fp is None:
                fp = open(p.match.log_file, 'a', encoding='utf-8', buffering=1)
                setattr(p.match, "_log_fp", fp)
            print(*args, **kwargs, file=fp)

    def ml_log_print(self, *args, **kwargs) -> None:
        """ML用ログファイルへ出力（標準出力はしない）"""
        p = self.player

        # === 追記ガード ===
        if self._terminal_guard():
            return

        if hasattr(p, 'match') and p.match and getattr(p.match, 'ml_log_file', None):
            with open(p.match.ml_log_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)


    # ---------------- 状態スナップショット ----------------
    def build_state_snapshot(self) -> Dict[str, Any]:
        """
        state_{before,after} とサブステップ state_{before,after} の
        項目セットを完全一致させるための単一ビルダー。
        """
        p = self.player

        # --- 追加：公開情報としてのスタジアム名を一本化 ---
        stadium_name_shared = 0
        m = getattr(p, "match", None)
        try:
            # Match 側にグローバルな active_stadium があるなら最優先
            if m and getattr(m, "active_stadium", None):
                stadium_name_shared = m.active_stadium.name
            elif p.active_stadium:
                stadium_name_shared = p.active_stadium.name
            elif p.opponent and p.opponent.active_stadium:
                stadium_name_shared = p.opponent.active_stadium.name
            else:
                stadium_name_shared = 0
        except Exception:
            stadium_name_shared = 0


        # --- 自分（完全情報）---
        if p.active_card:
            top = p.active_card[-1]
            attached_energies = getattr(top, "attached_energies", [])
            energies = [e.name for e in attached_energies] if attached_energies else []
            conditions = [c.__class__.__name__ for c in getattr(top, "conditions", [])]
            can_evolve = int(getattr(top, "can_evolve", False))
            damage_counters = getattr(top, "damage_counters", 0)
            active_pokemon_obj = {
                "name": top.name,
                "hp": _hp0(top.hp),
                "damage_counters": damage_counters,
                "energies": energies,
                "conditions": conditions,
                "can_evolve": can_evolve,
                "ability_present": 1 if getattr(top, "ability", None) else 0,
                "ability_used": 1 if (getattr(top, "ability", None) and getattr(top, "has_used_ability", False)) else 0,
                "tools": [tool.name for tool in getattr(top, "tools", [])] if getattr(top, "tools", []) else [],
                "type": str(getattr(top, "type", None)),
                "has_weakness": 0
                    if (getattr(top, "weakness", None) in (None, "None"))
                    else 1,
                "weakness_type": type_to_id(getattr(top, "weakness", None)),
                "has_resistance": 0
                    if (getattr(top, "resistance", None) in (None, "None"))
                    else 1,
                "resistance_type": type_to_id(getattr(top, "resistance", None)),
                "is_basic": 1 if getattr(top, "is_basic", False) else 0,
                "is_ex":    1 if getattr(top, "is_ex",    False) else 0,
                "retreat_cost": _effective_retreat(top),
            }
        else:
            active_pokemon_obj = {"name": None, "hp": None, "damage_counters": 0, "energies": [], "conditions": [], "can_evolve": 0, "tools": [], "type": None, "weakness": None, "retreat_cost": None}

        bench_pokemon_list = []
        for idx in range(5):
            if idx < len(p.bench) and p.bench[idx]:
                top = p.bench[idx][-1]
                attached_energies = getattr(top, "attached_energies", [])
                energies = [e.name for e in attached_energies] if attached_energies else []
                can_evolve = int(getattr(top, "can_evolve", False))
                damage_counters = getattr(top, "damage_counters", 0)
                bench_pokemon_list.append(
                    {
                    "name": top.name,
                    "hp": _hp0(top.hp),
                    "damage_counters": damage_counters,
                    "energies": energies,
                    "can_evolve": int(getattr(top, "can_evolve", False)),
                    "ability_present": 1 if getattr(top, "ability", None) else 0,
                    "ability_used": 1 if (getattr(top, "ability", None) and getattr(top, "has_used_ability", False)) else 0,
                    "tools": [tool.name for tool in getattr(top, "tools", [])] if getattr(top, "tools", []) else [],
                    "type": str(getattr(top, "type", None)),
                    "has_weakness": 0
                        if (getattr(top, "weakness", None) in (None, "None"))
                        else 1,
                    "weakness_type": type_to_id(getattr(top, "weakness", None)),
                    "has_resistance": 0
                        if (getattr(top, "resistance", None) in (None, "None"))
                        else 1,
                    "resistance_type": type_to_id(getattr(top, "resistance", None)),
                    "is_basic": 1 if getattr(top, "is_basic", False) else 0,
                    "is_ex":    1 if getattr(top, "is_ex",    False) else 0,
                    "retreat_cost": _effective_retreat(top),
                })
            else:
                bench_pokemon_list.append(0)


        # 手札 / トラッシュの同名集計
        def _count_names(arr):
            d = {}
            for c in arr:
                n = c.name
                d[n] = d.get(n, 0) + 1
            out = []
            for n, cnt in d.items():
                out.append(n if cnt == 1 else [n, cnt])
            return out
        hand_array = _count_names(p.hand)
        discard_array = _count_names(p.discard_pile)

        me_state = {
            "player": p.name,
            "hand": hand_array,
            "hand_count": len(p.hand),
            "bench_count": len(p.bench),
            "prize_count": len(p.prize_cards),
            "deck_count": len(p.deck.cards),
            "active_pokemon": active_pokemon_obj,
            "bench_pokemon": bench_pokemon_list,
            "active_stadium": stadium_name_shared,
            "discard_pile": discard_array,
            "discard_pile_count": len(p.discard_pile),
            "supporter_used": int(getattr(p, "has_used_supporter", False)),
            "energy_attached": int(getattr(p, "has_added_energy", False)),
            "retreat_this_turn": int(getattr(p, "has_retreat_this_turn", False)),
            "stadium_used": int(getattr(p, "has_used_stadium", False)),
            "was_knocked_out_by_attack_last_turn": int(getattr(p, "was_knocked_out_by_attack_last_turn", False)),
        }

        # --- 相手（公開情報のみ）---
        if p.opponent:
            if p.opponent.active_card:
                top = p.opponent.active_card[-1]
                attached_energies = getattr(top, "attached_energies", [])
                energies = [e.name for e in attached_energies] if attached_energies else []
                conditions = [c.__class__.__name__ for c in getattr(top, "conditions", [])]
                damage_counters = getattr(top, "damage_counters", 0)
                opp_active_obj = {
                    "name": top.name,
                    "hp": _hp0(top.hp),
                    "damage_counters": damage_counters,
                    "energies": energies,
                    "conditions": conditions,
                    "type": str(getattr(top, "type", None)),
                    "has_weakness": 0
                        if (getattr(top, "weakness", None) in (None, "None"))
                        else 1,
                    "weakness_type": type_to_id(getattr(top, "weakness", None)),
                    "has_resistance": 0
                        if (getattr(top, "resistance", None) in (None, "None"))
                        else 1,
                    "resistance_type": type_to_id(getattr(top, "resistance", None)),
                    "is_basic": 1 if getattr(top, "is_basic", False) else 0,
                    "is_ex":    1 if getattr(top, "is_ex",    False) else 0,
                    "retreat_cost": _effective_retreat(top),
                    "tools": [tool.name for tool in getattr(top, "tools", [])] if getattr(top, "tools", []) else [],
                }
            else:
                opp_active_obj = {"name": None, "hp": None, "damage_counters": 0, "energies": [], "conditions": [], "tools": []}

            opp_bench_list = []
            for idx in range(5):
                if idx < len(p.opponent.bench) and p.opponent.bench[idx]:
                    top = p.opponent.bench[idx][-1]
                    attached_energies = getattr(top, "attached_energies", [])
                    energies = [e.name for e in attached_energies] if attached_energies else []
                    damage_counters = getattr(top, "damage_counters", 0)
                    opp_bench_list.append(
                        {"name": top.name,
                        "hp": _hp0(top.hp),
                        "damage_counters": damage_counters,
                        "energies": energies,
                        "type": str(getattr(top, "type", None)),
                        "has_weakness": 0
                            if (getattr(top, "weakness", None) in (None, "None"))
                            else 1,
                        "weakness_type": type_to_id(getattr(top, "weakness", None)),
                        "has_resistance": 0
                            if (getattr(top, "resistance", None) in (None, "None"))
                            else 1,
                        "resistance_type": type_to_id(getattr(top, "resistance", None)),
                        "is_basic": 1 if getattr(top, "is_basic", False) else 0,
                        "is_ex":    1 if getattr(top, "is_ex",    False) else 0,
                        "retreat_cost": _effective_retreat(top),
                        "tools": [tool.name for tool in getattr(top, "tools", [])] if getattr(top, "tools", []) else []}
                    )
                else:
                    opp_bench_list.append(0)

            opp_discard_count = {}
            for card in p.opponent.discard_pile:
                n = card.name
                opp_discard_count[n] = opp_discard_count.get(n, 0) + 1
            opp_discard_array = [n if c == 1 else [n, c] for n, c in opp_discard_count.items()]

            opp_state = {
                "player": p.opponent.name,
                "hand_count": len(p.opponent.hand),
                "bench_count": len(p.opponent.bench),
                "prize_count": len(p.opponent.prize_cards),
                "deck_count": len(p.opponent.deck.cards),
                "active_pokemon": opp_active_obj,
                "bench_pokemon": opp_bench_list,
                "discard_pile": opp_discard_array,
                "discard_pile_count": len(p.opponent.discard_pile),
                "active_stadium": stadium_name_shared,
            }
        else:
            opp_state = None

        # ---- ここから追加：神視点（完全情報）とトップN順序 ----
        # ログに完全情報を含めるか？（未定義なら True）
        log_full = True
        try:
            log_full = bool(getattr(p.match, "log_full_info", True))
        except Exception:
            pass

        me_private = None
        opp_private = None

        if log_full:
            # --- 自分 ---
            my_deck_enums = _enum_list_from_zone(getattr(p.deck, "cards", []))
            my_prize_enums = _enum_list_from_zone(getattr(p, "prize_cards", []))

            my_bag = _bag_counts_from_enums(my_deck_enums)

            me_private = {
                # 残りの山札（順序なしのカウント）
                "deck_bag_counts": my_bag,
                # サイド6枚の内訳（enum配列）
                "prize_enum": my_prize_enums,
            }

            # --- 相手 ---
            if p.opponent:
                opp_deck_enums  = _enum_list_from_zone(getattr(p.opponent.deck, "cards", []))
                opp_prize_enums = _enum_list_from_zone(getattr(p.opponent, "prize_cards", []))
                opp_hand_enums  = _enum_list_from_zone(getattr(p.opponent, "hand", []))

                opp_bag = _bag_counts_from_enums(opp_deck_enums)

                opp_private = {
                    "deck_bag_counts": opp_bag,
                    "prize_enum": opp_prize_enums,
                    "hand_enum": opp_hand_enums,
                }
        # ---- 追加ここまで ----

        return {
            "game_id": getattr(p.match, "game_id", None),
            "turn": getattr(p.match, "turn", None),
            "current_player": p.name,
            "me": me_state,
            "opp": opp_state,
            "me_private": me_private,   # 自分の完全情報（1,2 + 手札）
            "opp_private": opp_private, # 相手の完全情報（3,4,5）
        }

    # ---------------- サブステップ ----------------
    def begin_substep(self, phase: str, legal_actions: List[List[int]]):
        self._current_substep = {
            "phase": phase,
            "state_before": self.build_state_snapshot(),
            "legal_actions": legal_actions,
        }

    def end_substep(self, action_vec: List[int], action_index: int):
        if not self._current_substep:
            return
        self._current_substep["action_vec"] = action_vec
        self._current_substep["action_index"] = action_index
        self._current_substep["state_after"] = self.build_state_snapshot()
        self._pending_substeps.append(self._current_substep)
        self._current_substep = None

    # ---------------- ACTION_RESULT ----------------
    def build_action_result(self, action: "Action") -> Dict[str, Any]:
        p = self.player

        def _as_int(x) -> int:
            try:
                if isinstance(x, int):
                    return x
                if hasattr(x, "value"):
                    v = x.value
                    if isinstance(v, int):
                        return v
                    if isinstance(v, (tuple, list)) and v:
                        return int(v[0])
                return int(x)
            except Exception:
                return 0

        def _energy_to_enum_id(x) -> int:
            try:
                if isinstance(x, int):
                    return x
                target = str(x) if x is not None else ""
                if not target:
                    return 0
                zones = [
                    getattr(p, "discard_pile", []),
                    getattr(p, "hand", []),
                ]
                if getattr(p, "deck", None) and getattr(p.deck, "cards", None):
                    zones.append(p.deck.cards)
                if getattr(p, "active_card", None):
                    zones.append(p.active_card)
                for stk in getattr(p, "bench", []):
                    zones.append(stk)
                for zone in zones:
                    for c in zone:
                        obj = c[-1] if isinstance(c, list) and c else c
                        if getattr(obj, "id", None) and str(obj.id) == target:
                            enum_id = getattr(obj, "card_enum", None)
                            return _as_int(enum_id)
                return 0
            except Exception:
                return 0

        action_arr = action.serialize(p) if hasattr(action, "serialize") else [_as_int(getattr(action, "action_type", 0)), 0, 0, 0, 0]

        # USE_ABILITY の整形
        try:
            from .action import ActionType
            if getattr(action, "action_type", None) == ActionType.USE_ABILITY:
                ability_id = _as_int(getattr(action, "ability_id", None)) or _as_int(action_arr[1])
                bench_idx = 0
                energy_enum = 0
                extra = getattr(action, "extra", {}) if isinstance(getattr(action, "extra", None), dict) else {}
                if "energy_id" in extra:
                    energy_enum = _energy_to_enum_id(extra.get("energy_id"))
                bench_idx = _as_int(extra.get("selected_bench_idx", extra.get("bench_idx", 0)))
                if bench_idx == 0:
                    maybe_bench = _as_int(action_arr[2])
                    if maybe_bench in (1, 2, 3, 4, 5):
                        bench_idx = maybe_bench
                action_arr = [_as_int(ActionType.USE_ABILITY), ability_id, energy_enum, bench_idx, 0]
        except Exception:
            pass

        result_obj: Dict[str, Any] = {"action": action_arr}
        try:
            macro = [_as_int(action_arr[0]), _as_int(action_arr[1]), 0, 0, 0]
            result_obj["macro"] = macro
        except Exception:
            pass

        if hasattr(action, "extra") and action.extra:
            result_obj.update(action.extra)
        result_obj.pop("target_id", None)

        # サブステップ取り込み
        if self._pending_substeps:
            result_obj["substeps"] = self._pending_substeps
            self._pending_substeps = []

        return result_obj

    def log_step(
        self,
        pre_state: Dict[str, Any],
        legal_actions: List[List[int]],
        action_result: Optional[Dict[str, Any]],
        post_state: Dict[str, Any],
        force_reward: Optional[float] = None,
        force_done: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        1 ステップ分のログを出力する。force_* が指定されたらそれを優先する。
        meta は追加メタ情報（例: {"terminal": True, "reason": "deck_out"}）を格納する。
        """
        p = self.player

        # === 追記ガード（どこからでも一律）===
        if self._terminal_guard():
            # 既に他方が終局フラグを立てていても、自分側のバッファがあれば一度だけフラッシュする
            try:
                if getattr(self, "_deckout_filter_enabled", False) and getattr(self, "_buf_calls", None):
                    end_r = None
                    if isinstance(meta, dict) and "reason" in meta:
                        end_r = meta.get("reason")
                    self.end_buffering(end_r)
            finally:
                return

        # --- 追加: SKIP_DECKOUT_LOGGING を見て自動でフィルタ開始 ---
        try:
            m = getattr(p, "match", None)
            if m and bool(getattr(m, "skip_deckout_logging", False)) and not getattr(self, "_deckout_filter_enabled", False):
                self.set_deckout_filter(True)
                self.begin_buffering()
        except Exception:
            pass

        terminal_flag = bool(meta and meta.get("terminal"))

        # ★ 終局メタの理由を確実に捕捉（既に入っていればOK）
        if terminal_flag:
            try:
                self._capture_state_and_reason_from_meta(meta)  # meta["reason"] を内部に保存
            except Exception:
                pass

        # 参考: 直後で latest state を覚える現在の実装もOK
        self._last_state_after = post_state

        # None が来たら自動復元（最後の保険）
        if legal_actions is None:
            legal_actions = []
        # 終局レコードでは action_result=None を維持する（復元しない）
        if action_result is None and not terminal_flag:
            try:
                action_result = self.build_action_result(getattr(p, "last_action", None)) if hasattr(p, "last_action") else {}
            except Exception:
                action_result = {}
        if post_state is None:
            try:
                post_state = self.build_state_snapshot()
            except Exception:
                post_state = {}

        # --- 1) 環境報酬を計算 ---
        try:
            reward, done = p.compute_reward()
        except Exception:
            reward, done = 0.0, 0

        # --- 1.2) post_state からの終局（非デッキアウト系）推定 ---
        # compute_reward が done==1 を返さず、meta も無い場合の保険
        if not terminal_flag and done != 1:
            try:
                st = post_state if isinstance(post_state, dict) else {}
                me = st.get("me", {}) or {}
                opp = st.get("opp", {}) or {}

                def _has_active(side: dict) -> bool:
                    active = side.get("active_pokemon") or {}
                    if not isinstance(active, dict):
                        return False
                    hp = active.get("hp")
                    # 数値HPがあれば >0、無ければ（不明扱いで）True
                    if isinstance(hp, (int, float)):
                        return hp > 0
                    return True

                inferred_reason: Optional[str] = None
                # 賞札切れ（どちらかが 0）
                if (isinstance(me.get("prize_count"), int) and me["prize_count"] == 0) or \
                   (isinstance(opp.get("prize_count"), int) and opp["prize_count"] == 0):
                    inferred_reason = "PRIZE_OUT"
                else:
                    # ベンチ切れ（相手側/自分側どちらでも）
                    me_bench = me.get("bench_count")
                    opp_bench = opp.get("bench_count")
                    if (isinstance(me_bench, int) and me_bench == 0 and not _has_active(me)) or \
                       (isinstance(opp_bench, int) and opp_bench == 0 and not _has_active(opp)):
                        inferred_reason = "BASICS_OUT"

                if inferred_reason:
                    meta = dict(meta or {})
                    meta.setdefault("terminal", True)
                    meta.setdefault("reason", inferred_reason)
                    terminal_flag = True
                    done = 1

                    try:
                        setattr(p.match, "_terminal_counted", True)
                    except Exception:
                        pass

                    try:
                        if getattr(p.match, "winner", None):
                            reward = 1.0 if p.match.winner == p.name else -1.0
                    except Exception:
                        pass
            except Exception:
                pass

        # --- 1.5) 非ダミーの終局アクションでは ±1 を必ず入れる ---
        if done == 1 and not terminal_flag:
            # まず winner を優先
            try:
                if getattr(p.match, "winner", None):
                    reward = 1.0 if p.match.winner == p.name else -1.0
                else:
                    # winner 未設定なら compute_reward の値を再確認
                    r_env, d_env = p.compute_reward()
                    if d_env == 1 and r_env != 0:
                        reward = float(r_env)
                    else:
                        # deck_out など理由から推定（必要に応じて）
                        if isinstance(meta, dict) and str(meta.get("reason", "")).strip().upper() in ("DECK_OUT", "DECKOUT"):
                            # デッキ切れを起こした側が -1
                            my_deck_left = len(getattr(getattr(p, "deck", None), "cards", []) or [])
                            reward = -1.0 if my_deck_left == 0 else +1.0

            except Exception:
                pass

        # --- 2) PBRS を Action 単位で注入（終局の遷移も含む／ダミー行は除外）---
        if getattr(p.match, "use_reward_shaping", False) \
                and getattr(p.match, "reward_shaping", None) \
                and not terminal_flag:
            # この遷移が終端かどうかを PBRS に伝える（φ(s_terminal)=0 の扱いへ）
            setattr(p.match, "_transition_done", bool(done))
            try:
                reward = p.match.reward_shaping.calculate_shaped_reward(p, p.match, reward)
            finally:
                # 一時フラグのリーク防止
                try:
                    delattr(p.match, "_transition_done")
                except Exception:
                    pass

        # --- 2.5) この遷移で終局が確定したことをマーク（ダミー行の二重計上防止） ---
        if done == 1 and not terminal_flag:
            try:
                setattr(p.match, "_terminal_counted", True)
            except Exception:
                pass

        # --- 3) 引数で強制指定があればさらに上書き ---
        if force_reward is not None:
            reward = force_reward
        if force_done is not None:
            done = force_done

        # --- 3.5) meta の理由が「終局」を示しているなら、この遷移を done=1 として確実に記録 ---
        try:
            reason_upper = str((meta or {}).get("reason", "")).strip().upper() if isinstance(meta, dict) else ""
            if reason_upper in ("PRIZE_OUT", "BASICS_OUT", "TURN_LIMIT"):
                done = 1
        except Exception:
            pass

        # ---------------- 統合レコード(JSONL) ----------------
        rec = {
            "state_before": pre_state,
            "legal_actions": legal_actions,
            # 終局レコードは None を維持
            "action_result": None if terminal_flag else action_result,
            "state_after": post_state,
            "reward": reward,
            "done": done,
        }
        if meta is not None:
            rec["meta"] = meta

# 修正後（log_step 内／JSON出力〜終局処理の順序）
        import json
        line = json.dumps(rec, ensure_ascii=False)
        if p.match and getattr(p.match, "ml_log_file", None):
            self.ml_log_print(line)
        if p.match and getattr(p.match, "log_file", None):
            self.log_print(line)

        # ---------------- 解析器向けの行出力（互換用・任意） ----------------
        emit_compat = bool(getattr(p.match, "emit_compat_lines", False))
        if emit_compat:
            # [ACTION_RESULT]
            ar_line = "[ACTION_RESULT] " + json.dumps(rec["action_result"], ensure_ascii=False, default=str)
            if p.match and getattr(p.match, "log_file", None):
                self.log_print(ar_line)
            if p.match and getattr(p.match, "ml_log_file", None):
                self.ml_log_print(ar_line)

            # [STATE_OBJ_AFTER]
            snap = post_state if isinstance(post_state, dict) else self.build_state_snapshot()
            after_obj = OrderedDict({
                "game_id": snap.get("game_id"),
                "turn": snap.get("turn"),
                "current_player": p.name,
                "me": snap.get("me", {}),
                "opp": snap.get("opp", {}) or {},
            })
            after_obj["reward"] = reward
            after_obj["done"] = done

            after_line = "[STATE_OBJ_AFTER] " + json.dumps(after_obj, ensure_ascii=False)
            if p.match and getattr(p.match, "log_file", None):
                self.log_print(after_line)
            if p.match and getattr(p.match, "ml_log_file", None):
                self.ml_log_print(after_line)

        # --- 終局時は、まずバッファを確実に吐き出してから抑止フラグを立てる ---
        try:
            if done == 1:
                end_r = None
                if isinstance(meta, dict) and "reason" in meta:
                    end_r = meta.get("reason")
                self.end_buffering(end_r)   # 先にフラッシュ
                try:
                    setattr(p.match, "_terminal_emitted", True)  # フラッシュ後に立てる
                except Exception:
                    pass
        except Exception:
            pass


    def _energy_suffix_from_names(self, energies: List[str]) -> str:
        """
        スナップショット内の 'energies': ['雷エネルギー', ...] を
        ' [雷エネルギー×2, 闘エネルギー]' のようなサフィックスに整形する。
        """
        if not energies:
            return ""
        counts = Counter(energies)
        parts = [f"{k}×{v}" if v > 1 else k for k, v in counts.items()]
        return " [" + ", ".join(parts) + "]"

    def _fmt_public_pokemon_from_snapshot(self, poke_obj: Optional[Dict[str, Any]]) -> str:
        """
        build_state_snapshot() が返す 'active_pokemon' / 'bench_pokemon[i]' の dict を
        '名前 (HP:xx) [エネルギー…]' 形式に整形する。
        """
        if not poke_obj or poke_obj.get("name") in (None, 0):
            return "(空)"
        name = poke_obj.get("name", "???")
        hp   = _hp0(poke_obj.get("hp"))
        suffix = self._energy_suffix_from_names(poke_obj.get("energies") or [])
        return f"{name} (HP:{hp}){suffix}"

    def _fmt_public_bench_from_snapshot(self, bench_list: Optional[List[Any]]) -> str:
        """
        スナップショットの 'bench_pokemon' をカンマ区切りで整形。
        0/None は空スロットとしてスキップ。
        """
        if not bench_list:
            return "(空)"
        items: List[str] = []
        for x in bench_list:
            if not x or x == 0:
                continue
            items.append(self._fmt_public_pokemon_from_snapshot(x))
        return ", ".join(items) if items else "(空)"


    # ---------------- 既存の詳細出力APIを委譲で残す ----------------
    def print_player_state(self, actions=None, action_tuple2id=None):
        """
        既存 player.print_player_state の移植（ログ構造は player.py の元実装に合わせてある）
        ※ スナップショット(build_state_snapshot)一本化で未使用変数を解消
        """
        p = self.player
        if not p.match:
            return
        # === 追記ガードのみ ===
        if self._terminal_guard():
            return

        import json

        # 行動集合（重複排除 → 5整数ベクトルへ）
        if actions is None:
            actions = []
        unique_actions, seen = [], set()
        for a in actions:
            key = tuple(a.serialize(p))
            if key not in seen:
                unique_actions.append(a)
                seen.add(key)
        actions = unique_actions
        legal_actions = [a.serialize(p) for a in actions]

        # 直前合法手を保持（act直前で使う）
        self.last_legal_actions_before = legal_actions

        # スナップショット一本化
        snap = self.build_state_snapshot()
        reward_val, done_val = p.compute_reward()

        from collections import OrderedDict
        state_before = OrderedDict({
            "game_id": snap.get("game_id"),
            "turn": snap.get("turn"),
            "current_player": p.name,
            "me": snap.get("me"),
            "opp": snap.get("opp"),
            "me_private": snap.get("me_private"),
            "opp_private": snap.get("opp_private"),
            "reward": reward_val,
            "done": done_val,
            "legal_actions": legal_actions,
        })

        # 出力
        json_line = "[STATE_BEFORE]    " + json.dumps(state_before, ensure_ascii=False)
        if getattr(p.match, "log_file", None):
            self.log_print(json_line)
        if getattr(p.match, "ml_log_file", None):
            self.ml_log_print(json.dumps(state_before, ensure_ascii=False))

        # LEGAL_ACTIONS を別行で
        legal_line = "[LEGAL_ACTIONS] " + json.dumps(legal_actions, ensure_ascii=False)
        if getattr(p.match, "log_file", None):
            self.log_print(legal_line)
        if getattr(p.match, "ml_log_file", None):
            self.ml_log_print(legal_line)

        # ---- 追加: 人向けの整形表示（カテゴリ順：ポケモン→グッズ→サポーター→どうぐ→スタジアム→エネルギー） ----
        # Player._format_card_list_for_display を使って手札・トラッシュの表示順を統一
        try:
            self.log_print(f"自分の手札: {p._format_card_list_for_display(getattr(p, 'hand', []))}")
            self.log_print(f"自分のトラッシュ: {p._format_card_list_for_display(getattr(p, 'discard_pile', []))}")
            if getattr(p, 'opponent', None):
                self.log_print(f"相手のトラッシュ: {p._format_card_list_for_display(getattr(p.opponent, 'discard_pile', []))}")

            # ---- 追加: 公開盤面（エネルギー表記付き）の人向け整形出力 ----
            me  = snap.get("me", {}) or {}
            opp = snap.get("opp", {}) or {}

            self.log_print("[あなた] 現在の盤面（公開情報）")
            self.log_print("----------------------------------------------------------------------")
            # 相手側
            self.log_print(f"相手のバトル場: {self._fmt_public_pokemon_from_snapshot(opp.get('active_pokemon'))}")
            self.log_print(f"相手のベンチ: {self._fmt_public_bench_from_snapshot(opp.get('bench_pokemon'))}")
            self.log_print(f"相手の手札: {opp.get('hand_count', 0)} 枚（非公開）")
            self.log_print(f"相手の山札残り: {opp.get('deck_count', 0)} 枚")
            self.log_print("----------------------------------------------------------------------")
            # 自分側
            self.log_print(f"自分のバトル場: {self._fmt_public_pokemon_from_snapshot(me.get('active_pokemon'))}")
            self.log_print(f"自分のベンチ: {self._fmt_public_bench_from_snapshot(me.get('bench_pokemon'))}")
            # 手札は上の行で一覧を出しているため、ここでは枚数のみ
            self.log_print(f"自分の手札: {me.get('hand_count', 0)} 枚")
            self.log_print(f"自分の山札残り: {me.get('deck_count', 0)} 枚")

            # スタジアム（どちらかが場にあれば名称、無ければ(空)）
            act_st = me.get("active_stadium", 0) or opp.get("active_stadium", 0) or 0
            self.log_print(f"スタジアム: {act_st if act_st else '(空)'}")

            # ターン・手番
            self.log_print(f"Turn: {snap.get('turn')} | 手番: {snap.get('current_player')}")
            # ---- 追加ここまで ----
        except Exception:
            # 念のため、整形に失敗してもログ出力全体は止めない
            pass

    def print_state_after(self, turn=None):
        """
        既存 player.print_state_after の移植。build_state_snapshot を用いて
        [STATE_OBJ_AFTER] を出力。
        """
        # === 追記ガードのみ ===
        if self._terminal_guard():
            return

        import json
        from collections import OrderedDict
        p = self.player

        # ★追加: 終局かつ互換出力フラグが True のときのみ出力
        if not (p.match and getattr(p.match, "game_over", False) and bool(getattr(p.match, "emit_compat_lines", False))):
            return

        snap = self.build_state_snapshot()
        log_obj = OrderedDict({
            "game_id": snap.get("game_id"),
            "turn": turn if turn is not None else snap.get("turn"),
            "current_player": p.name,
            "me": snap.get("me"),
            "opp": snap.get("opp") if snap.get("opp") is not None else {},
            "me_private": snap.get("me_private"),
            "opp_private": snap.get("opp_private"),
        })

        reward_val, done_val = p.compute_reward()
        # 必要ならここで 1 回だけ shape（終局行なら通常 done_val==1 なので通らない）
        if getattr(p.match, "use_reward_shaping", False) and getattr(p.match, "reward_shaping", None) and done_val == 0:
            reward_val = p.match.reward_shaping.calculate_shaped_reward(p, p.match, reward_val)

        log_obj["reward"] = reward_val
        log_obj["done"] = done_val

        state_after_output = "[STATE_OBJ_AFTER] " + json.dumps(log_obj, ensure_ascii=False)
        if getattr(p.match, "log_file", None):
            self.log_print(state_after_output)
        if getattr(p.match, "ml_log_file", None):
            self.ml_log_print(state_after_output)

    def print_action_result(self, action: "Action"):
        """
        既存 player.print_action_result の移植。
        """
        # === 追記ガードのみ ===
        if self._terminal_guard():
            return

        import json
        p = self.player
        if not (p.match and bool(getattr(p.match, "emit_compat_lines", False))):
            return
        result_obj = self.build_action_result(action)
        action_result_output = "[ACTION_RESULT] " + json.dumps(result_obj, ensure_ascii=False, default=str)
        if getattr(p.match, "log_file", None):
            self.log_print(action_result_output)
        if getattr(p.match, "ml_log_file", None):
            self.ml_log_print(action_result_output)

    def log_terminal_step(self, reason: Optional[str] = None):
        """
        ゲーム終了時に必ず 1 回だけ、完整形の 1 レコードを出力する。
        - legal_actions: []
        - action_result: None（ダミー行動ベクトルは使わない）
        - done: 1 を強制
        - reward: winner が分かるなら +1/-1 を強制、なければ compute_reward に委譲
        - meta: {"terminal": True, "reason": <文字列>}
        """
        r = str(reason or "").upper()

        p = self.player

        # --- 追加: マッチ横断の多重防止（どこから呼ばれても一度きり） ---
        try:
            if getattr(p.match, "_terminal_emitted", False):
                return
        except Exception:
            pass

        # 多重防止（インスタンス内）
        if getattr(self, "_terminal_logged", False):
            return

        # ★ 通常終局（PRIZE_OUT / BASICS_OUT）の扱いを調整
        # 既に非ダミー遷移で終局が記録済みなら、ここでは何も出さずバッファだけ閉じる
        terminal_already_counted = bool(getattr(p.match, "_terminal_counted", False))
        if r in ("PRIZE_OUT", "BASICS_OUT"):
            if terminal_already_counted:
                try:
                    self.end_buffering(r)
                except Exception:
                    pass
                # 終局出力済み扱い
                self._terminal_logged = True
                try:
                    setattr(p.match, "_terminal_emitted", True)
                except Exception:
                    pass
                return
            # ★ 未計上なら、このまま「ダミー1行」を出力する（落とさない）

        # スナップショット（before/after は同形で OK）
        try:
            pre_state = self.build_state_snapshot()
        except Exception:
            pre_state = {}
        try:
            post_state = self.build_state_snapshot()
        except Exception:
            post_state = {}

        # ターミナルでは次アクション集合は空
        legal_actions: List[List[int]] = []
        action_result = None

        # --- 報酬の強制指定ロジック ---
        if terminal_already_counted:
            # （例外的に）DECK_OUT/TURN_LIMIT のときのみ、監査目的のダミーを 0点で出す
            force_reward: Optional[float] = 0.0
        else:
            # ここで初めて終局報酬を支払う
            force_reward = None
            try:
                if getattr(p, "match", None) and getattr(p.match, "winner", None):
                    force_reward = 1.0 if p.match.winner == p.name else -1.0
                else:
                    if (reason or "").lower() == "deck_out":
                        my_deck_left = len(getattr(getattr(p, "deck", None), "cards", []) or [])
                        force_reward = -1.0 if my_deck_left == 0 else +1.0
                if force_reward is None:
                    force_reward = 0.0  # safety
                # ターミナルPBRSの締め（実装があれば）
                rs = getattr(p.match, "reward_shaping", None)
                if getattr(p.match, "use_reward_shaping", False) and rs is not None:
                    bonus = float(rs.terminal_bonus(p, p.match, terminal_phi=0.0))
                    force_reward = float(force_reward) + bonus
            except Exception:
                force_reward = 0.0

        self.log_step(
            pre_state=pre_state,
            legal_actions=legal_actions,
            action_result=action_result,   # ダミー行なので None を維持
            post_state=post_state,
            force_reward=force_reward,
            force_done=1,
            meta={"terminal": True, "reason": (reason or "game_over")},
        )

        # --- バッファを必ず閉じる ---
        try:
            self.end_buffering(reason)
        except Exception:
            pass

        # --- この試合はもう終局を出力済み ---
        self._terminal_logged = True
        try:
            setattr(p.match, "_terminal_emitted", True)
        except Exception:
            pass

    def close(self):
        """開いているログ用ファイルを安全に閉じる（冪等）"""
        # 1) まずバッファをフラッシュ
        try:
            if getattr(self, "_deckout_filter_enabled", False) and getattr(self, "_buf_calls", None):
                self.end_buffering(None)
        except Exception:
            pass

        # 2) まだ終局を出していなければフォールバックで1行
        try:
            p = self.player
            already_emitted = bool(getattr(getattr(p, "match", None), "_terminal_emitted", False))
            if not already_emitted:
                inferred = self._infer_end_reason_from_last_state()
                self.log_terminal_step(inferred or "game_over")
        except Exception:
            pass

        # 3) ファイルを閉じる
        p = self.player
        m = getattr(p, "match", None)
        if m is not None:
            fp = getattr(m, "_log_fp", None)
            if fp is not None:
                try:
                    fp.close()
                except Exception:
                    pass
                finally:
                    setattr(m, "_log_fp", None)


# === ここから追記: 山札切れ用のバッファリング機能 ===

    def set_deckout_filter(self, enabled: bool) -> None:
        self._deckout_filter_enabled = bool(enabled)

    def begin_buffering(self) -> None:
        if not getattr(self, "_deckout_filter_enabled", False):
            return
        self._buf_calls = []
        self._last_state = None
        self._explicit_end_reason = None

        # 元メソッドを退避（人向け＋ML の両方）
        self._orig_log_print = getattr(self, "log_print", None)

        # バッファに積むだけに差し替え
        def _wrap_log_print(*args, **kwargs) -> None:
            self._buf_calls.append(("log", args, kwargs))

        if self._orig_log_print:
            self.log_print = _wrap_log_print  # type: ignore[assignment]

    def end_buffering(self, end_reason: str | None) -> None:
        # 書き戻し中は _terminal_guard を一時無効化
        self._is_flushing_buffer = True
        try:
            explicit_raw = end_reason or getattr(self, "_explicit_end_reason", None)
            explicit_reason = (str(explicit_raw).strip().upper()) if explicit_raw else None

            calls = getattr(self, "_buf_calls", [])
            if explicit_reason == "DECK_OUT":
                # 明示 DECK_OUT は人向けログのみ破棄（ML はそもそもバッファされていない）
                pass
            else:
                # それ以外は溜め順で書き戻す（人向けログのみ）
                for kind, args, kwargs in calls:
                    if kind == "log" and getattr(self, "_orig_log_print", None):
                        self._orig_log_print(*args, **kwargs)
        finally:
            self._is_flushing_buffer = False
            if getattr(self, "_orig_log_print", None):
                self.log_print = self._orig_log_print
            self._buf_calls = []
            self._orig_log_print = None

    def _capture_state_and_reason_from_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        if isinstance(meta, dict) and "reason" in meta:
            r = str(meta.get("reason") or "").strip().upper()
            if r == "DECK_OUT" or r == "DECKOUT":
                self._explicit_end_reason = "DECK_OUT"
            elif r:
                self._explicit_end_reason = r

    def _infer_end_reason_from_last_state(self) -> str | None:
        st = getattr(self, "_last_state_after", None) or getattr(self, "_last_state", None) or {}
        me = st.get("me", {})
        opp = st.get("opp", {})

        me_prize = me.get("prize_count")
        opp_prize = opp.get("prize_count")
        me_deck  = me.get("deck_count")
        opp_deck = opp.get("deck_count")
        me_bench = me.get("bench_count")
        opp_bench = opp.get("bench_count")

        if (isinstance(me_prize, int) and me_prize == 0) or (isinstance(opp_prize, int) and opp_prize == 0):
            return "PRIZE_OUT"

        def has_active_basic(side: dict) -> bool:
            active = side.get("active_pokemon")
            if not isinstance(active, dict):
                return False
            hp = active.get("hp")
            if isinstance(hp, (int, float)):
                return hp > 0
            return True

        me_has_active = has_active_basic(me)
        opp_has_active = has_active_basic(opp)
        if (isinstance(me_bench, int) and me_bench == 0 and not me_has_active) or \
           (isinstance(opp_bench, int) and opp_bench == 0 and not opp_has_active):
            return "BASICS_OUT"

        if (isinstance(me_deck, int) and me_deck == 0) or (isinstance(opp_deck, int) and opp_deck == 0):
            return "DECK_OUT"

        return None
# === 追記ここまで ===
