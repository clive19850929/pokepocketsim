#  pokepocketsim/action.py
from __future__ import annotations

# --- ACTION_SCHEMAS のローダ（Pyright対応） ---
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, cast, Callable, Any, Optional
import importlib
import sys
import os

if TYPE_CHECKING:
    # 型チェック用の宣言だけ与える（実体は下の動的ロードで入る）
    ACTION_SCHEMAS: Dict[int, List[str]]
else:
    try:
        # 1) 同一パッケージ内の action_schemas を動的 import
        pkg = __package__ or ""
        mod = importlib.import_module(f"{pkg}.action_schemas" if pkg else "action_schemas")
        ACTION_SCHEMAS = cast(Dict[int, List[str]], getattr(mod, "ACTION_SCHEMAS", {}))
    except Exception:
        try:
            # 2) ルート直下の prepare_d3rlpy_data をフォールバックで import
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            mod2 = importlib.import_module("prepare_d3rlpy_data")
            ACTION_SCHEMAS = cast(Dict[int, List[str]], getattr(mod2, "ACTION_SCHEMAS", {}))
        except Exception:
            # 3) どちらも無ければ空で運用（汎用デフォルト分岐を使う）
            ACTION_SCHEMAS = {}

# ---------- アクション種別 ----------
class ActionType(IntEnum):
    END_TURN = 0
    PLAY_BENCH = 1
    ATTACH_ENERGY = 2
    USE_ITEM = 3
    USE_SUPPORTER = 4
    RETREAT = 5
    ATTACK = 6
    EVOLVE = 7
    USE_ABILITY = 8
    PLAY_STADIUM = 9
    STADIUM_EFFECT = 10
    ATTACH_TOOL = 11
    DRAW_CARD = 12
    GET_BASIC_POKEMON = 13
    GET_EVOLVE_POKEMON = 14
    SET_BENCH_POKEMON = 15
    REMOVE_BENCH_POKEMON = 16
    SHOW_RANDOM_BENCH_POKEMON = 17
    SET_OPPONENT_BENCH = 18
    SET_ACTIVE_CARD = 19
    TAKE_PRIZE = 20

# ---------- 実行可能アクション ----------
class Action:
    """
    Player・Match 間で受け渡しされる最小単位のアクション。
    * name : 表示／ログ用文字列
    * func : 実際に呼び出す関数
    * action_type : ActionType
    * can_continue_turn : True なら実行後も同じターンで追加行動可
    * card_class : グッズ／サポーター等の種類を ML ログに残したい場合に任意指定
    """
    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        action_type: ActionType,
        *,
        can_continue_turn: bool = True,
        card_class: Optional[type] = None,
        extra: Optional[dict] = None,     # ★payload/extraを追加
    ) -> None:
        self.name = name
        self.func = func
        self.action_type = action_type
        self.can_continue_turn = can_continue_turn
        self.card_class = card_class
        self.extra = (extra or {})

    # -- 実行 ---------------------------------------------------------------
    def act(self, *args, **kwargs) -> bool:
        """実際に関数を呼び出し，戻り値（bool or None）を turn 継続フラグとして返す"""
        result = self.func(*args, **kwargs)
        if isinstance(result, bool):
            return result
        return self.can_continue_turn

    __call__ = act  # Action()(...) でも呼べるように

    # -- ユーティリティ ------------------------------------------------------
    @staticmethod
    def find_action(actions: List["Action"], target_name: str) -> "Action":
        """name が一致する Action を返す（見つからなければ先頭を返す）"""
        for a in actions:
            if a.name == target_name:
                return a
        if not actions:
            raise ValueError("Action list is empty")
        return actions[0]

    @staticmethod
    def normalize_5int_vec(v: Any, player=None) -> list[int]:
        """
        Action / list / tuple 等を「常に 5-int」へ正規化する。

        forced_actions に積まれる値が Action だったり list だったりしても、
        ログ/学習を 5-int 前提で破綻させないための入口。
        """
        # ---- helpers ---------------------------------------------------------
        def _to_int(x) -> int:
            try:
                return int(x)
            except Exception:
                return 0

        try:
            if isinstance(v, Action):
                v = v.serialize(player)
        except Exception:
            pass

        if isinstance(v, (list, tuple)):
            out = []
            for x in list(v)[:5]:
                out.append(_to_int(x))
            while len(out) < 5:
                out.append(0)
            return out

        return [0, 0, 0, 0, 0]

    def serialize(self, player=None) -> list[int]:
        t = int(self.action_type.value) if hasattr(self.action_type, "value") else int(self.action_type)
        out = [t, 0, 0, 0, 0]

        # ---- helpers ---------------------------------------------------------
        def _to_int(x) -> int:
            # ここでは “整数ID” のみを入れる設計（UUIDや文字列は事前にID化しておく）
            try:
                return int(x)
            except Exception:
                return 0

        extra = getattr(self, "extra", None) or {}

        def _is_mcts_simulation_match(m) -> bool:
            try:
                if m is None:
                    return False
            except Exception:
                return False

            try:
                if bool(getattr(m, "_is_mcts_simulation", False)):
                    return True
            except Exception:
                pass

            try:
                if (getattr(m, "log_mode", True) is False and
                    getattr(m, "log_file", None) is None and
                    getattr(m, "ml_log_file", None) is None):
                    return True
            except Exception:
                pass

            return False

        def _emit_mcts_dbg() -> None:
            try:
                m = getattr(player, "match", None) if player is not None else None
                if not _is_mcts_simulation_match(m):
                    return
                c = int(getattr(m, "_mcts_action_serialize_dbg_count", 0) or 0)
                if c < 20:
                    setattr(m, "_mcts_action_serialize_dbg_count", c + 1)
                    print(
                        f"[ACTION_SERIALIZE][MCTS] type={type(self).__name__} action_type={getattr(self, 'action_type', None)} name={getattr(self, 'name', None)} out={out}",
                        flush=True,
                    )
            except Exception:
                pass

        # ---- main_id 決定 ----------------------------------------------------
        # 既定：card_id / ability_id / attack_id のいずれかを action_type ごとに採用
        card_enum  = _to_int(getattr(self, "card_id", 0))
        ability_id = _to_int(getattr(self, "ability_id", 0))
        attack_id  = _to_int(extra.get("attack_id", getattr(self, "attack_id", 0)))

        if self.action_type in (
            ActionType.PLAY_BENCH,
            ActionType.USE_ITEM,
            ActionType.USE_SUPPORTER,
            ActionType.PLAY_STADIUM,                # ← STADIUM は参照しない
            ActionType.ATTACH_TOOL,                 # ← TOOL は参照しない
            ActionType.ATTACH_ENERGY,
            ActionType.EVOLVE,
        ):
            out[1] = card_enum
            schema_key = card_enum
        elif self.action_type == ActionType.USE_ABILITY:
            out[1] = ability_id
            schema_key = ability_id
        elif self.action_type == ActionType.ATTACK:
            out[1] = attack_id
            schema_key = attack_id  # 攻撃固有の拡張が必要になれば schema を足せる
        else:
            # END_TURN, RETREAT, SET_ACTIVE_CARD 等
            schema_key = 0  # スキーマ未定義

        # ---- schema に従って p3..p5 を詰める ---------------------------------
        # 例：ACTION_SCHEMAS = { 50001: ["trashed_id1","trashed_id2","selected_id"], ... }
        schema_fields = ACTION_SCHEMAS.get(schema_key)
        if schema_fields:
            vals = []
            for fname in schema_fields[:3]:
                vals.append(_to_int(extra.get(fname, 0)))
            # 足りない分は0埋め
            while len(vals) < 3:
                vals.append(0)
            out[2], out[3], out[4] = vals[0], vals[1], vals[2]
            _emit_mcts_dbg()
            return Action.normalize_5int_vec(out, player=player)

        # ---- スキーマが無い場合の汎用デフォルト -------------------------------
        # よく使う共通パラメータ（bench_idx, stack_index 等）を安全側で詰める
        bench_idx   = _to_int(extra.get("selected_bench_idx", extra.get("bench_idx", 0)))
        stack_index = _to_int(extra.get("stack_index", 0))

        # TAKE_PRIZE は「どのサイドを取ったか」が学習/ログ上の最小情報になりやすい
        prize_idx   = _to_int(extra.get("selected_prize_idx", extra.get("prize_idx", extra.get("prize_index", 0))))
        prize_enum  = _to_int(extra.get("prize_card_enum", extra.get("prize_enum", 0)))

        if self.action_type == ActionType.ATTACH_ENERGY:
            out[2] = bench_idx
        elif self.action_type == ActionType.EVOLVE:
            out[2] = stack_index
        elif self.action_type == ActionType.TAKE_PRIZE:
            out[1] = prize_enum
            out[2] = prize_idx
        elif self.action_type in (ActionType.PLAY_BENCH, ActionType.ATTACH_TOOL, ActionType.USE_ITEM,
                                ActionType.USE_SUPPORTER, ActionType.PLAY_STADIUM,
                                ActionType.RETREAT, ActionType.SET_ACTIVE_CARD):
            out[2] = bench_idx  # ベンチ対象のあるカードはここに入れる慣習

        # END_TURN などは 0 のまま
        _emit_mcts_dbg()
        return Action.normalize_5int_vec(out, player=player)

    def to_id_vec(self, player=None) -> list[int]:
        return Action.normalize_5int_vec(self.serialize(player), player=player)

    def to_int_vec(self, player=None) -> list[int]:
        return self.to_id_vec(player)

    def to_5int_vec(self, player=None) -> list[int]:
        return self.to_id_vec(player)

    # -- 表示系 --------------------------------------------------------------
    def __str__(self) -> str:
        return self.name

    __repr__ = __str__
