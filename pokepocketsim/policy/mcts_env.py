from __future__ import annotations

from typing import Any, List, Optional

from ..match import Match
from ..player import Player
from .az_mcts_policy import MCTSSimEnvProtocol


class MatchPlayerSimEnv(MCTSSimEnvProtocol):
    """
    AlphaZero 型 MCTS から見た「シミュレーション用環境」のラッパークラス（暫定実装）。

    本ファイルは “器だけ” ではなく、現状のコードでは以下を実際に提供する:
    - clone():
        Match / Player を deepcopy して、探索用に独立なシミュレーション環境を作る。
        deepcopy が壊れやすい IO / logger などは memo 固定で共有し、診断ログも出せる。
        さらに、clone 後はログ出力や学習系の副作用を抑えるためのフラグ/属性を無効化する。
    - legal_actions():
        現在の手番プレイヤーの gather_actions(match) を呼び、match.converter.convert_legal_actions(...) により
        “5-int ID” を生成して返す（MCTS の唯一の正）。
    - step(action):
        legal_actions() と同じ変換規則（match.converter.convert_legal_actions）で 5-int を再生成し、
        渡された 5-int と一致する index の Action を選んで act_and_regather_actions で 1 手進める。
    - is_terminal():
        match.game_over を参照して終局判定を行う。
    - result():
        match.winner とルートプレイヤー名から +1 / -1 / 0 を返す（非終局では例外）。

    重要:
    - 変換不能/不整合（obs_vec 生成不能、action decode不能、手番不整合、モデル入力不整合など）は
      一切フォールバックせず RuntimeError で停止する。
    """

    def __init__(self, match: Match, player: Player) -> None:
        """
        MCTS のルートとなる Match / Player を受け取り、シミュレーション環境の
        ラッパーを構築する。

        注意:
        - match / player は poke-pocket-sim の実環境オブジェクトであり、
          clone() が実装されるまでは破壊的操作に用いるべきではない。
        """
        self._match: Match = match
        self._player: Player = player
        self._root_player_name = getattr(player, "name", None)

        self._root_player_slot = None
        try:
            if getattr(match, "starting_player", None) is player:
                self._root_player_slot = "starting_player"
            elif getattr(match, "second_player", None) is player:
                self._root_player_slot = "second_player"
        except Exception:
            self._root_player_slot = None

        # legal_actions()/step() 間の 5-int ⇄ Action 対応を安定化するキャッシュ
        self._la_cache_turn = None
        self._la_cache_player_name = None
        self._la_cache_actions = None
        self._la_cache_ids = None
        self._la_cache_lookup = None
        self._la_cache_forced_len = None
        self._la_cache_forced_player_name = None

    def clone(self) -> MCTSSimEnvProtocol:
        """
        現在の状態から独立したコピーを返す（deepcopy ベース）。

        - deepcopy が TextIOWrapper（stdout / logger の stream 等）を辿って落ちないように、
          IO 系オブジェクトは memo に固定して「コピーしない」。
        - Match と Player を別々に deepcopy すると参照整合性が壊れるため、
          まず root player を deepcopy し、その player_copy.match を match_copy として採用する。
          （player→match の参照で一括コピーされるため、参照整合性が保たれる）
        - それでも失敗する場合に「どこ（stage）で」「何が（path/type）deepcopy を壊したか」を
          ログで特定できるように診断を出す（初回のみ詳細、以降は短縮）。
        - clone 後はシミュレーション用途として、ログ出力や一部副作用を抑制する設定を入れる。
        """
        import copy
        import io
        import sys
        import types
        import traceback

        def _seed_memo_io(memo, obj):
            # よくある属性名から IO を拾って memo に固定する（コピー禁止）
            for attr in ("_fp", "fp", "file", "stream", "out", "_out", "stdout", "_stdout", "stderr", "_stderr"):
                try:
                    v = getattr(obj, attr, None)
                except Exception:
                    v = None
                if isinstance(v, io.IOBase):
                    memo[id(v)] = v

        def _is_shareable_uncopyable(x):
            # deepcopy が苦手だが参照共有で問題になりにくいもの（関数/モジュール/ロック等）
            try:
                if isinstance(x, io.IOBase):
                    return True
                if isinstance(x, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
                    return True
                if isinstance(x, (types.MethodType, types.BuiltinMethodType)):
                    return True
            except Exception:
                pass
            # _thread.lock などは型が取りにくいので名前で拾う
            try:
                tn = type(x).__name__
                if "lock" in tn.lower():
                    return True
            except Exception:
                pass
            return False

        def _scan_paths(root, max_depth=4, max_items=64):
            # 失敗時の「本質」特定用: deepcopy を壊しやすい物の path を収集
            out = []
            seen = set()
            stack = [(root, "root", 0)]
            while stack and len(out) < int(max_items):
                x, path, d = stack.pop()
                if x is None:
                    continue
                xid = id(x)
                if xid in seen:
                    continue
                seen.add(xid)

                if _is_shareable_uncopyable(x):
                    try:
                        r = repr(x)
                    except Exception:
                        r = "<repr failed>"
                    if len(r) > 160:
                        r = r[:160] + "..."
                    out.append((path, type(x).__name__, r))
                    continue

                if d >= int(max_depth):
                    continue

                try:
                    if isinstance(x, dict):
                        for k, v in list(x.items())[:32]:
                            stack.append((v, f"{path}[{repr(k)}]", d + 1))
                        continue
                    if isinstance(x, (list, tuple, set)):
                        for i, v in enumerate(list(x)[:32]):
                            stack.append((v, f"{path}[{i}]", d + 1))
                        continue
                except Exception:
                    pass

                try:
                    dx = getattr(x, "__dict__", None)
                except Exception:
                    dx = None
                if isinstance(dx, dict):
                    for k, v in list(dx.items())[:48]:
                        stack.append((v, f"{path}.{k}", d + 1))

            return out

        def _diag_once(sig, msg, detail_lines):
            # 同じ失敗を延々と出さない（初回だけ詳細）
            try:
                d = getattr(self, "_mcts_clone_diag", None)
            except Exception:
                d = None
            if not isinstance(d, dict):
                d = {"sig_counts": {}}
                try:
                    self._mcts_clone_diag = d
                except Exception:
                    pass

            c = int(d["sig_counts"].get(sig, 0))
            d["sig_counts"][sig] = c + 1

            if c == 0:
                print(msg)
                for ln in detail_lines:
                    print(ln)
            else:
                # 2回目以降は短く
                print(msg + f" (repeat={c+1})")

        memo = {}

        # stdout/stderr（ConsoleTee 等で _fp を持つことがある）
        try:
            memo[id(sys.stdout)] = sys.stdout
            _seed_memo_io(memo, sys.stdout)
        except Exception:
            pass
        try:
            memo[id(sys.stderr)] = sys.stderr
            _seed_memo_io(memo, sys.stderr)
        except Exception:
            pass

        # match / player / logger 周辺の IO を拾って memo 化
        for obj in (
            self._match,
            self._player,
            getattr(self._player, "logger", None),
            getattr(self._match, "starting_player", None),
            getattr(self._match, "second_player", None),
            getattr(getattr(self._match, "starting_player", None), "logger", None),
            getattr(getattr(self._match, "second_player", None), "logger", None),
        ):
            if obj is None:
                continue
            try:
                if isinstance(obj, io.IOBase):
                    memo[id(obj)] = obj
                _seed_memo_io(memo, obj)
            except Exception:
                pass

        # ★追加: 事前スキャンして「deepcopy しない方が良い物」を memo に固定
        try:
            for root_obj in (self._player, self._match):
                for pth, tnm, _rp in _scan_paths(root_obj, max_depth=4, max_items=64):
                    # path は診断用。memo は「オブジェクト参照」をキーに固定するだけ。
                    pass
                # scan しながら実体も拾う（2nd pass: objectをたどり memo に入れる）
                seen = set()
                stack = [root_obj]
                while stack:
                    x = stack.pop()
                    if x is None:
                        continue
                    xid = id(x)
                    if xid in seen:
                        continue
                    seen.add(xid)

                    if _is_shareable_uncopyable(x):
                        memo[id(x)] = x
                        continue

                    try:
                        if isinstance(x, dict):
                            for v in list(x.values())[:32]:
                                stack.append(v)
                            continue
                        if isinstance(x, (list, tuple, set)):
                            for v in list(x)[:32]:
                                stack.append(v)
                            continue
                    except Exception:
                        pass

                    try:
                        dx = getattr(x, "__dict__", None)
                    except Exception:
                        dx = None
                    if isinstance(dx, dict):
                        for v in list(dx.values())[:48]:
                            stack.append(v)
        except Exception:
            pass

        match_copy = None
        player_copy = None

        # 1) まずは root player 起点で deepcopy（player→match の参照で Match も一括コピーされる）
        try:
            player_copy = copy.deepcopy(self._player, memo)
            match_copy = getattr(player_copy, "match", None)
            if match_copy is None:
                raise RuntimeError("MatchPlayerSimEnv.clone: player_copy.match is None.")
        except Exception as e1:
            sig = f"clone_stage1:{type(e1).__name__}:{str(e1)[:120]}"
            tb = traceback.format_exc().splitlines()[-12:]
            scan = _scan_paths(self._player, max_depth=4, max_items=20)
            detail = ["[MCTS_ENV][CLONE_FAIL] stage=player_deepcopy", f"  exc={type(e1).__name__}: {e1}"]
            detail += ["  tb=" + ln for ln in tb]
            if scan:
                detail.append("  scan(uncloneables) top:")
                for pth, tnm, rp in scan[:12]:
                    detail.append(f"    - {pth} :: {tnm} :: {rp}")
            _diag_once(sig, f"[MCTS_ENV][CLONE_FAIL] stage=player_deepcopy exc={type(e1).__name__}: {e1}", detail)

            # 2) フォールバック: match を deepcopy してから player を引く（従来経路）
            try:
                match_copy = copy.deepcopy(self._match, memo)
            except Exception as e2:
                sig2 = f"clone_stage2:{type(e2).__name__}:{str(e2)[:120]}"
                tb2 = traceback.format_exc().splitlines()[-12:]
                scan2 = _scan_paths(self._match, max_depth=4, max_items=20)
                detail2 = ["[MCTS_ENV][CLONE_FAIL] stage=match_deepcopy", f"  exc={type(e2).__name__}: {e2}"]
                detail2 += ["  tb=" + ln for ln in tb2]
                if scan2:
                    detail2.append("  scan(uncloneables) top:")
                    for pth, tnm, rp in scan2[:12]:
                        detail2.append(f"    - {pth} :: {tnm} :: {rp}")
                _diag_once(sig2, f"[MCTS_ENV][CLONE_FAIL] stage=match_deepcopy exc={type(e2).__name__}: {e2}", detail2)

                memo2 = dict(memo)
                for p in (getattr(self._match, "starting_player", None), getattr(self._match, "second_player", None)):
                    if p is None:
                        continue
                    try:
                        lg = getattr(p, "logger", None)
                    except Exception:
                        lg = None
                    if lg is not None:
                        memo2[id(lg)] = None

                match_copy = copy.deepcopy(self._match, memo2)

                # logger を作り直す（battle_logger が deepcopy 不可でもここで復旧）
                try:
                    from ..battle_logger import BattleLogger
                    for p in (getattr(match_copy, "starting_player", None), getattr(match_copy, "second_player", None)):
                        if p is None:
                            continue
                        try:
                            p.logger = BattleLogger(p)
                        except Exception:
                            pass
                except Exception:
                    pass

            # match_copy から「同一プレイヤー」を引く（参照整合性を保つ）
            root_name = self._root_player_name
            try:
                p1 = getattr(match_copy, "starting_player", None)
                p2 = getattr(match_copy, "second_player", None)
                if root_name is not None:
                    if p1 is not None and getattr(p1, "name", None) == root_name:
                        player_copy = p1
                    elif p2 is not None and getattr(p2, "name", None) == root_name:
                        player_copy = p2
                if player_copy is None:
                    player_copy = p1 if p1 is not None else p2
            except Exception:
                player_copy = getattr(match_copy, "starting_player", None)

        if match_copy is None or player_copy is None:
            raise RuntimeError(f"MatchPlayerSimEnv.clone: failed to clone match/player. match_copy={type(match_copy).__name__ if match_copy is not None else None} player_copy={type(player_copy).__name__ if player_copy is not None else None}")

        # match_copy 側に同名 player が居るなら、そちらを優先して採用（参照の一体性）
        root_name = self._root_player_name
        try:
            p1 = getattr(match_copy, "starting_player", None)
            p2 = getattr(match_copy, "second_player", None)
            if root_name is not None:
                if p1 is not None and getattr(p1, "name", None) == root_name:
                    player_copy = p1
                elif p2 is not None and getattr(p2, "name", None) == root_name:
                    player_copy = p2
        except Exception:
            pass

        # シミュレーション中はログを出さない（stdout/ファイルへの副作用防止）
        try:
            match_copy.log_mode = False
        except Exception:
            pass

        # ★ MCTS シミュレーションであることを明示（Player.select_action 側で参照）
        try:
            setattr(match_copy, "_is_mcts_simulation", True)
        except Exception:
            pass
        for k in ("log_file", "ml_log_file"):
            try:
                setattr(match_copy, k, None)
            except Exception:
                pass
        try:
            match_copy.use_reward_shaping = False
            match_copy.reward_shaping = None
        except Exception:
            pass

        # 整合性（念のため）
        if getattr(player_copy, "match", None) is not match_copy:
            try:
                player_copy.match = match_copy
            except Exception:
                pass
        for p in (getattr(match_copy, "starting_player", None), getattr(match_copy, "second_player", None)):
            if p is None:
                continue
            if getattr(p, "match", None) is not match_copy:
                try:
                    p.match = match_copy
                except Exception:
                    pass

        return MatchPlayerSimEnv(match_copy, player_copy)


    def _get_forced_actions_raw(self) -> Optional[Any]:
        m = self._match
        for nm in ("forced_actions", "_forced_actions", "forced_action_list", "forced_queue"):
            try:
                v = getattr(m, nm, None)
            except Exception:
                v = None
            if v is not None:
                return v
        return None

    def _forced_is_active(self) -> bool:
        fa = self._get_forced_actions_raw()
        if fa is None:
            return False
        try:
            if isinstance(fa, (list, tuple)):
                return len(fa) > 0
        except Exception:
            return False
        try:
            return bool(fa)
        except Exception:
            return False

    def _resolve_player_ref(self, ref: Any) -> Optional[Player]:
        m = self._match
        if ref is None:
            return None
        try:
            if isinstance(ref, Player):
                return ref
        except Exception:
            pass

        p1 = getattr(m, "starting_player", None)
        p2 = getattr(m, "second_player", None)

        try:
            if ref is p1 or ref is p2:
                return ref
        except Exception:
            pass

        try:
            if isinstance(ref, str):
                if p1 is not None and getattr(p1, "name", None) == ref:
                    return p1
                if p2 is not None and getattr(p2, "name", None) == ref:
                    return p2
        except Exception:
            pass

        try:
            if isinstance(ref, int):
                if int(ref) == 0:
                    return p1
                if int(ref) == 1:
                    return p2
        except Exception:
            pass

        return None

    def _get_forced_player(self) -> Optional[Player]:
        m = self._match

        for nm in ("forced_player", "_forced_player", "forced_player_obj"):
            try:
                ref = getattr(m, nm, None)
            except Exception:
                ref = None
            p = self._resolve_player_ref(ref)
            if p is not None:
                return p

        for nm in ("forced_player_name", "_forced_player_name", "forced_owner_name", "forced_side_name"):
            try:
                ref = getattr(m, nm, None)
            except Exception:
                ref = None
            p = self._resolve_player_ref(ref)
            if p is not None:
                return p

        for nm in ("forced_player_slot", "_forced_player_slot"):
            try:
                slot = getattr(m, nm, None)
            except Exception:
                slot = None
            if slot in ("starting_player", "second_player"):
                try:
                    return getattr(m, slot, None)
                except Exception:
                    pass

        return None

    def _get_current_player(self) -> Player:
        """
        Match.play と同じ規則で「現在手番プレイヤー」を返す。

        forced_actions が有効な間は forced 側プレイヤーを優先する（turn は進めない）。
        """
        m = self._match

        if self._forced_is_active():
            fp = self._get_forced_player()
            if fp is not None:
                return fp

        try:
            t = int(getattr(m, "turn", 0) or 0)
        except Exception:
            t = 0

        p = getattr(m, "starting_player", None) if (t % 2 == 0) else getattr(m, "second_player", None)
        if p is None:
            raise RuntimeError("MatchPlayerSimEnv: match.starting_player/second_player is missing.")
        return p

    def root_player_name(self) -> Optional[str]:
        return self._root_player_name

    def current_player_name(self) -> Optional[str]:
        try:
            return getattr(self._get_current_player(), "name", None)
        except Exception:
            return None

    def value_to_root(self, value_current_player: float) -> float:
        """
        モデルが「現在手番プレイヤー視点」の value を返す前提で、
        ルート（root player）視点に変換して返す。
        """
        try:
            v = float(value_current_player)
        except Exception:
            v = 0.0

        root = self._root_player_name
        cur = self.current_player_name()
        if root is None or cur is None:
            return v
        return v if (cur == root) else -v

    def _coerce_5int(self, vec: Any, action_obj: Any = None, player: Any = None) -> Optional[List[int]]:
        """
        vec を len=5 の int ベクトルに正規化する。
        - list/tuple/np.ndarray などを許容
        - Enum/tuple 値なども int に揃える
        """
        if vec is None:
            return None

        def _safe_repr(x: Any, limit: int = 480) -> str:
            try:
                s = repr(x)
            except Exception:
                try:
                    s = f"<repr failed: {type(x).__name__}>"
                except Exception:
                    s = "<repr failed>"
            if len(s) > int(limit):
                s = s[: int(limit)] + "..."
            return s

        def _action_brief(a: Any) -> str:
            if a is None:
                return "None"
            try:
                parts = [f"type={type(a).__name__}"]
                try:
                    nm = getattr(a, "name", None)
                    if nm is not None:
                        parts.append(f"name={nm}")
                except Exception:
                    pass
                try:
                    at = getattr(a, "action_type", None)
                    if at is not None:
                        parts.append(f"action_type={at}")
                except Exception:
                    pass
                return " ".join(parts)
            except Exception:
                return _safe_repr(a)

        # numpy
        try:
            import numpy as np  # noqa: F401
        except Exception:
            np = None

        if np is not None:
            try:
                if isinstance(vec, np.ndarray):
                    vec = vec.reshape(-1).tolist()
            except Exception:
                pass

        if isinstance(vec, tuple):
            vec = list(vec)

        if not isinstance(vec, list) or len(vec) != 5:
            try:
                print(
                    "[MCTS_ENV][COERCE_5INT_FAIL] reason=not_list_or_len_mismatch"
                    f" player={getattr(player, 'name', None) if player is not None else None}"
                    f" vec_type={type(vec).__name__ if vec is not None else None}"
                    f" vec_len={len(vec) if isinstance(vec, (list, tuple)) else None}"
                    f" vec={_safe_repr(vec)}"
                    f" action={_action_brief(action_obj)}",
                    flush=True,
                )
            except Exception:
                pass
            return None

        out: List[int] = []
        for x in vec:
            try:
                if isinstance(x, int):
                    out.append(int(x))
                    continue
                if hasattr(x, "value"):
                    xv = getattr(x, "value")
                    if isinstance(xv, int):
                        out.append(int(xv))
                        continue
                    if isinstance(xv, (tuple, list)) and len(xv) > 0 and isinstance(xv[0], int):
                        out.append(int(xv[0]))
                        continue
                # それ以外は int 変換を試す
                out.append(int(x))
            except Exception as e:
                try:
                    print(
                        "[MCTS_ENV][COERCE_5INT_FAIL] reason=int_cast_failed"
                        f" player={getattr(player, 'name', None) if player is not None else None}"
                        f" x_type={type(x).__name__ if x is not None else None}"
                        f" x={_safe_repr(x)}"
                        f" vec={_safe_repr(vec)}"
                        f" action={_action_brief(action_obj)}"
                        f" exc={type(e).__name__}:{e}",
                        flush=True,
                    )
                except Exception:
                    pass
                return None

        return out

    def _convert_legal_actions_5int(self, actions: List[Any], player: Player) -> List[List[int]]:
        """
        match.converter.convert_legal_actions(actions, player=...) を唯一の正として 5-int を生成する。
        失敗/不整合は RuntimeError で停止（フォールバック禁止）。

        A: converter が受け取っている actions の実体（要素の型/形）を、失敗時に必ずダンプする
        B: convert_legal_actions の返り値 ids（5-int になっているか）を、失敗時に必ずダンプする
        """
        m = self._match
        conv = getattr(m, "converter", None)
        if conv is None or not hasattr(conv, "convert_legal_actions"):
            raise RuntimeError("MatchPlayerSimEnv: _actions is missing (cannot build 5-int ids).")

        def _safe_repr(x: Any, limit: int = 480) -> str:
            try:
                s = repr(x)
            except Exception:
                try:
                    s = f"<repr failed: {type(x).__name__}>"
                except Exception:
                    s = "<repr failed>"
            if len(s) > int(limit):
                s = s[: int(limit)] + "..."
            return s

        def _try_serialize(a: Any) -> Any:
            try:
                fn = getattr(a, "serialize", None)
                if callable(fn):
                    try:
                        return fn(player=player)
                    except TypeError:
                        try:
                            return fn(player)
                        except Exception:
                            return None
                    except Exception:
                        return None
            except Exception:
                return None
            return None

        def _action_brief(a: Any) -> str:
            if a is None:
                return "None"
            try:
                parts = [f"type={type(a).__name__}"]
                try:
                    nm = getattr(a, "name", None)
                    if nm is not None:
                        parts.append(f"name={nm}")
                except Exception:
                    pass
                try:
                    at = getattr(a, "action_type", None)
                    if at is not None:
                        parts.append(f"action_type={at}")
                except Exception:
                    pass
                try:
                    ser = _try_serialize(a)
                    if ser is not None:
                        parts.append(f"serialize={_safe_repr(ser, limit=360)}")
                except Exception:
                    pass
                return " ".join(parts)
            except Exception:
                return _safe_repr(a)

        def _actions_sample(lst: Any, k: int = 5) -> List[str]:
            try:
                if not isinstance(lst, list):
                    return [f"<not_list type={type(lst).__name__} repr={_safe_repr(lst, limit=360)}>"]
                return [_action_brief(a) for a in lst[: int(k)]]
            except Exception:
                return ["<actions_sample failed>"]

        def _ids_sample(lst: Any, k: int = 8) -> str:
            try:
                if isinstance(lst, tuple):
                    try:
                        lst = list(lst)
                    except Exception:
                        pass
                if not isinstance(lst, list):
                    return f"<not_list type={type(lst).__name__} repr={_safe_repr(lst, limit=360)}>"
                return _safe_repr(lst[: int(k)], limit=520)
            except Exception:
                return "<ids_sample failed>"

        def _dump_header(tag: str, ids_obj: Any = None, extra: str = "") -> None:
            try:
                print(
                    f"[MCTS_ENV][LA5][{tag}]"
                    f" player={getattr(player, 'name', None)}"
                    f" conv={type(conv).__name__}"
                    f" actions_type={type(actions).__name__ if actions is not None else None}"
                    f" actions_len={len(actions) if isinstance(actions, list) else 'NA'}"
                    f" actions_sample={_safe_repr(_actions_sample(actions), limit=1600)}"
                    f" ids_type={type(ids_obj).__name__ if ids_obj is not None else None}"
                    f" ids_len={len(ids_obj) if isinstance(ids_obj, list) else 'NA'}"
                    f" ids_sample={_ids_sample(ids_obj)}"
                    f"{extra}",
                    flush=True,
                )
            except Exception:
                pass

        try:
            try:
                ids = conv.convert_legal_actions(actions, player=player)
            except TypeError:
                ids = conv.convert_legal_actions(actions)
        except Exception as e:
            _dump_header(
                "CONVERT_EXCEPTION",
                ids_obj=None,
                extra=f" exc={type(e).__name__}:{_safe_repr(e, limit=360)}",
            )
            raise

        if ids is None:
            _dump_header("CONVERT_NONE", ids_obj=ids)
            raise RuntimeError("MatchPlayerSimEnv: converter.convert_legal_actions returned None.")

        # numpy/tuple 等を含みうるので 1つずつ正規化
        out: List[List[int]] = []
        try:
            if isinstance(ids, tuple):
                ids = list(ids)
        except Exception:
            pass

        if not isinstance(ids, list):
            _dump_header("CONVERT_NOT_LIST", ids_obj=ids)
            raise RuntimeError("MatchPlayerSimEnv: converter.convert_legal_actions must return a list-like object.")

        if len(ids) != len(actions):
            _dump_header(
                "LEN_MISMATCH",
                ids_obj=ids,
                extra=f" len_actions={len(actions) if isinstance(actions, list) else 'NA'} len_ids={len(ids) if isinstance(ids, list) else 'NA'}",
            )
            raise RuntimeError(
                "MatchPlayerSimEnv._convert_legal_actions_5int: len mismatch between actions and converted ids. "
                "This indicates converter.convert_legal_actions output is inconsistent with the input actions. "
                f"src={__file__} turn={getattr(self._match, 'turn', None)} player={getattr(player, 'name', None)} "
                f"len_actions={len(actions) if isinstance(actions, list) else 'NA'} len_ids={len(ids) if isinstance(ids, list) else 'NA'}"
            )

        for i, vec in enumerate(ids):
            v = self._coerce_5int(vec, action_obj=actions[i], player=player)
            if v is None:
                try:
                    a_i = actions[i] if isinstance(actions, list) and i < len(actions) else None

                    ser_raw = _try_serialize(a_i) if a_i is not None else None
                    ser_5 = None
                    if ser_raw is not None:
                        ser_5 = self._coerce_5int(ser_raw, action_obj=a_i, player=player)

                    vec_eq_ser_raw = False
                    vec_eq_ser_5 = False
                    try:
                        vec_eq_ser_raw = (vec == ser_raw)
                    except Exception:
                        vec_eq_ser_raw = False
                    try:
                        vec_eq_ser_5 = (vec == ser_5)
                    except Exception:
                        vec_eq_ser_5 = False

                    print(
                        "[MCTS_ENV][LA5][COERCE_FAIL]"
                        f" player={getattr(player, 'name', None)}"
                        f" conv={type(conv).__name__}"
                        f" i={i} len_actions={len(actions)}"
                        f" action_i={_action_brief(a_i)}"
                        f" vec_type={type(vec).__name__ if vec is not None else None}"
                        f" vec={_safe_repr(vec)}"
                        f" serialize_raw={_safe_repr(ser_raw, limit=360)}"
                        f" serialize_5={_safe_repr(ser_5, limit=360)}"
                        f" vec_eq_serialize_raw={vec_eq_ser_raw}"
                        f" vec_eq_serialize_5={vec_eq_ser_5}",
                        flush=True,
                    )
                    # 直前の周辺も併記（重いので先頭数件だけ）
                    print(
                        "[MCTS_ENV][LA5][COERCE_FAIL_CTX]"
                        f" actions_sample={[ _action_brief(a) for a in (actions[:5] if isinstance(actions, list) else []) ]}"
                        f" ids_sample={_safe_repr(ids[:5] if isinstance(ids, list) else ids)}",
                        flush=True,
                    )
                except Exception:
                    pass
                raise RuntimeError("MatchPlayerSimEnv: converter.convert_legal_actions did not yield a 5-int id vector.")
            out.append(v)

        try:
            if __import__("os").getenv("MCTS_ENV_LA5_LOG", "0") == "1":
                c = int(getattr(self, "_la5_dbg_count", 0) or 0)
                if c < 20:
                    setattr(self, "_la5_dbg_count", c + 1)
                    print(
                        f"[MCTS_ENV][LA5][OK] turn={int(getattr(self._match, 'turn', 0) or 0)} "
                        f"player={getattr(player, 'name', type(player).__name__)} "
                        f"n_actions={int(len(actions))} ids_sample={out[:5]}",
                        flush=True,
                    )
        except Exception:
            pass

        return out

    def _start_turn_for_player(self, player: Player) -> None:
        """
        次手番の開始処理（ドロー等）。
        シグネチャ違いを吸収するが、失敗は停止（フォールバック禁止）。
        """
        try:
            player.setup_turn(self._match, viewing_player=None)
            return
        except TypeError:
            pass

        player.setup_turn(self._match)

    def get_obs_vec(self) -> List[float]:
        """
        現在手番プレイヤー視点の obs_vec（list[float]）を返す。

        ルール:
        - Match.build_public_state_for_ui(viewer=current_player) を「唯一の正」として使用する。
        - obs_vec が無い場合は match.encoder.encode_state(sd) を試し、それでも無ければ停止する。
        """
        cur = self._get_current_player()
        m = self._match

        if not hasattr(m, "build_public_state_for_ui"):
            raise RuntimeError("MatchPlayerSimEnv.get_obs_vec: match.build_public_state_for_ui is missing.")

        sd = m.build_public_state_for_ui(viewer=cur)
        if not isinstance(sd, dict):
            raise RuntimeError("MatchPlayerSimEnv.get_obs_vec: build_public_state_for_ui did not return a dict.")

        obs = sd.get("obs_vec", None)
        if obs is None:
            try:
                obs = sd.get("full_obs_vec", None)
                if obs is not None:
                    try:
                        import os
                        k = "_mcts_env_obs_fallback_full_budget"
                        b = getattr(m, k, None)
                        if b is None:
                            b = int(os.getenv("MCTS_ENV_OBS_FALLBACK_FULL_BUDGET", "6") or "6")
                        b = int(b)
                        if b > 0:
                            setattr(m, k, b - 1)
                            vlen = None
                            try:
                                vlen = len(obs)
                            except Exception:
                                vlen = None
                            print(
                                f"[MCTS_ENV][OBS_FALLBACK_FULL] turn={getattr(m,'turn',None)} cur={getattr(cur,'name',None)} "
                                f"type={type(obs).__name__} len={vlen}",
                                flush=True,
                            )
                    except Exception:
                        pass
            except Exception:
                obs = None

        if obs is None:
            enc = getattr(m, "encoder", None)
            try:
                fn = getattr(enc, "encode_state", None) if enc is not None else None
                if callable(fn):
                    try:
                        fn(sd, player=cur)
                    except TypeError:
                        try:
                            fn(sd, cur)
                        except Exception:
                            fn(sd)
            except Exception:
                pass

            try:
                obs = sd.get("obs_vec", None)
            except Exception:
                obs = None

            if obs is None and enc is not None and callable(enc):
                try:
                    enc(sd, None)
                except TypeError:
                    try:
                        enc(public_state=sd, legal_actions=None)
                    except TypeError:
                        try:
                            enc(player=sd, legal_actions=None)
                        except Exception:
                            try:
                                enc(sd)
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    pass

            try:
                obs = sd.get("obs_vec", None)
            except Exception:
                obs = None

            if obs is None:
                for k in ("obs_vec", "full_obs_vec", "obs_vec_az", "az_obs_vec", "observation_vec", "obs", "x"):
                    try:
                        v = sd.get(k, None)
                    except Exception:
                        v = None
                    if v is not None:
                        obs = v
                        break

        if obs is None:
            try:
                ks = sorted(list(sd.keys()))
            except Exception:
                ks = None
            raise RuntimeError(
                "MatchPlayerSimEnv.get_obs_vec: obs_vec is None (encoder not producing it). "
                f"keys={ks} encoder={type(getattr(m, 'encoder', None)).__name__ if getattr(m, 'encoder', None) is not None else None} src={__file__}"
            )

        # list[float] へ正規化
        try:
            if isinstance(obs, tuple):
                obs = list(obs)
            # numpy
            try:
                import numpy as np
                if isinstance(obs, np.ndarray):
                    obs = obs.reshape(-1).tolist()
            except Exception:
                pass
        except Exception:
            pass

        if not isinstance(obs, list) or not obs:
            raise RuntimeError("MatchPlayerSimEnv.get_obs_vec: obs_vec must be a non-empty list.")

        out = []
        for x in obs:
            try:
                out.append(float(x))
            except Exception:
                raise RuntimeError("MatchPlayerSimEnv.get_obs_vec: obs_vec contains a non-numeric value.")

        return out

    def _refresh_action_cache(self, player: Player) -> None:
        """
        現在状態の合法手を列挙し、5-int と Action の対応表をキャッシュする。

        forced_actions がある場合は forced_actions を優先し、
        forced 中は turn を進めない前提で「forced のみ」から 5-int -> Action 対応を作る。

        - converter 出力に重複 5-int が存在する場合は原則停止（デコード不能を明確化）。
          どうしても許容する場合は env: MCTS_ENV_ALLOW_DUP_IDS=1（先勝ち）で回避。
        """
        import os

        m = self._match
        try:
            turn = int(getattr(m, "turn", 0) or 0)
        except Exception:
            turn = None

        forced_raw = self._get_forced_actions_raw()
        forced_active = self._forced_is_active()
        forced_len = None
        try:
            if isinstance(forced_raw, (list, tuple)):
                forced_len = int(len(forced_raw))
        except Exception:
            forced_len = None

        actions = []
        ids = []

        if forced_active and isinstance(forced_raw, (list, tuple)):
            # forced_actions が Action 群（実行体）か、5-int 群（ID）かを判定
            all_5int_like = True
            forced_ids = []
            for x in list(forced_raw):
                v = self._coerce_5int(x, action_obj=None, player=player)
                if v is None:
                    all_5int_like = False
                    break
                forced_ids.append(v)

            if all_5int_like:
                # forced_actions が 5-int 群の場合:
                # 通常の gather_actions+converter で Action を用意し、その中から forced のみフィルタする
                all_actions = player.gather_actions(self._match) or []
                all_ids = self._convert_legal_actions_5int(all_actions, player) if all_actions else []

                full_lookup = {}
                for i, vec in enumerate(all_ids):
                    full_lookup[tuple(vec)] = all_actions[i]

                actions = []
                ids = []
                for v in forced_ids:
                    a = full_lookup.get(tuple(v), None)
                    if a is None:
                        raise RuntimeError(
                            "MatchPlayerSimEnv._refresh_action_cache: forced 5-int not found in current legal actions. "
                            f"turn={turn} player={getattr(player,'name',None)} forced_id={v}"
                        )
                    actions.append(a)
                    ids.append(v)
            else:
                # forced_actions が Action 群（実行体）の場合: forced_actions 自体を actions として採用
                actions = list(forced_raw)
                ids = self._convert_legal_actions_5int(actions, player) if actions else []
        else:
            actions = player.gather_actions(self._match) or []
            ids = self._convert_legal_actions_5int(actions, player) if actions else []

        lookup = {}
        dup_keys = []
        for i, vec in enumerate(ids):
            k = tuple(vec)
            if k in lookup:
                dup_keys.append(k)
            else:
                lookup[k] = actions[i]

        allow_dup = (os.getenv("MCTS_ENV_ALLOW_DUP_IDS", "0") == "1")
        if dup_keys and not allow_dup:
            try:
                from collections import Counter
                top = Counter(dup_keys).most_common(8)
            except Exception:
                top = [(dup_keys[0], len(dup_keys))]

            try:
                print(
                    "[MCTS_ENV][LA5][DUP_IDS]"
                    f" turn={turn} player={getattr(player,'name',None)}"
                    f" forced_active={forced_active}"
                    f" forced_len={forced_len}"
                    f" dup_top={top}",
                    flush=True,
                )
            except Exception:
                pass

            raise RuntimeError("MatchPlayerSimEnv: duplicate 5-int ids detected in converter output.")

        self._la_cache_turn = turn
        self._la_cache_player_name = getattr(player, "name", None)
        self._la_cache_actions = actions
        self._la_cache_ids = ids
        self._la_cache_lookup = lookup
        self._la_cache_forced_len = forced_len
        self._la_cache_forced_player_name = getattr(self._get_forced_player(), "name", None) if forced_active else None

    def legal_actions(self) -> List[Any]:
        """
        現在手番プレイヤーにとっての合法手 ID（5-int）リストを返す。

        forced_actions がある場合は forced_actions を優先し、forced 中は turn を進めない前提で返す。
        """
        cur = self._get_current_player()

        self._refresh_action_cache(cur)

        try:
            ids = getattr(self, "_la_cache_ids", None)
        except Exception:
            ids = None

        return ids if isinstance(ids, list) else []

    def step(self, action: Any) -> None:
        """
        指定された action（5-int ID）を 1手適用して内部状態を進める。

        forced 中の規則:
        - forced_actions が残っている間は turn を進めない
        - forced_actions を消化し切ったタイミングでのみ、can_continue=False なら turn を進める
        """
        forced_before = self._forced_is_active()

        cur = self._get_current_player()

        target = self._coerce_5int(action, action_obj=None, player=cur)
        if target is None:
            raise RuntimeError("MatchPlayerSimEnv.step: action must be a 5-int id vector.")

        pre_turn = None
        try:
            pre_turn = int(getattr(self._match, "turn", 0) or 0)
        except Exception:
            pre_turn = None

        now_turn = None
        try:
            now_turn = int(getattr(self._match, "turn", 0) or 0)
        except Exception:
            now_turn = None

        forced_len_now = None
        try:
            fr = self._get_forced_actions_raw()
            if isinstance(fr, (list, tuple)):
                forced_len_now = int(len(fr))
        except Exception:
            forced_len_now = None

        forced_player_name_now = getattr(self._get_forced_player(), "name", None) if forced_before else None

        cache_ok = False
        try:
            cache_ok = (
                getattr(self, "_la_cache_turn", None) == now_turn
                and getattr(self, "_la_cache_player_name", None) == getattr(cur, "name", None)
                and isinstance(getattr(self, "_la_cache_lookup", None), dict)
                and getattr(self, "_la_cache_forced_len", None) == forced_len_now
                and getattr(self, "_la_cache_forced_player_name", None) == forced_player_name_now
            )
        except Exception:
            cache_ok = False

        if not cache_ok:
            self._refresh_action_cache(cur)

        lookup = None
        actions = None
        ids = None
        try:
            lookup = getattr(self, "_la_cache_lookup", None)
            actions = getattr(self, "_la_cache_actions", None)
            ids = getattr(self, "_la_cache_ids", None)
        except Exception:
            lookup = None
            actions = None
            ids = None

        chosen = None
        try:
            if isinstance(lookup, dict):
                chosen = lookup.get(tuple(target), None)
        except Exception:
            chosen = None

        if chosen is None:
            # 1回だけ再構築して再試行（forced 切替や gather_actions の揺れを吸収）
            self._refresh_action_cache(cur)

            try:
                lookup = getattr(self, "_la_cache_lookup", None)
                actions = getattr(self, "_la_cache_actions", None)
                ids = getattr(self, "_la_cache_ids", None)
            except Exception:
                lookup = None
                actions = None
                ids = None

            try:
                if isinstance(lookup, dict):
                    chosen = lookup.get(tuple(target), None)
            except Exception:
                chosen = None

        if chosen is None:
            def _safe_repr(x: Any, limit: int = 640) -> str:
                try:
                    s = repr(x)
                except Exception:
                    s = f"<repr failed: {type(x).__name__}>"
                if len(s) > int(limit):
                    s = s[: int(limit)] + "..."
                return s

            def _try_serialize(a: Any) -> Any:
                try:
                    fn = getattr(a, "serialize", None)
                    if callable(fn):
                        try:
                            return fn(player=cur)
                        except TypeError:
                            try:
                                return fn(cur)
                            except Exception:
                                return None
                        except Exception:
                            return None
                except Exception:
                    return None
                return None

            def _action_brief(a: Any) -> str:
                if a is None:
                    return "None"
                try:
                    parts = [f"type={type(a).__name__}"]
                    try:
                        nm = getattr(a, "name", None)
                        if nm is not None:
                            parts.append(f"name={nm}")
                    except Exception:
                        pass
                    try:
                        at = getattr(a, "action_type", None)
                        if at is not None:
                            parts.append(f"action_type={at}")
                    except Exception:
                        pass
                    try:
                        ser = _try_serialize(a)
                        if ser is not None:
                            parts.append(f"serialize={_safe_repr(ser, limit=240)}")
                    except Exception:
                        pass
                    return " ".join(parts)
                except Exception:
                    return _safe_repr(a, limit=240)

            try:
                print(
                    "[MCTS_ENV][LA5][STEP_NO_MATCH]"
                    f" player={getattr(cur, 'name', None)}"
                    f" turn={getattr(self._match, 'turn', None)}"
                    f" forced_before={forced_before}"
                    f" target={_safe_repr(target, limit=120)}"
                    f" cache_ok={cache_ok}"
                    f" actions_len={len(actions) if isinstance(actions, list) else 'NA'}"
                    f" ids_len={len(ids) if isinstance(ids, list) else 'NA'}"
                    f" actions_sample={_safe_repr([_action_brief(a) for a in (actions[:5] if isinstance(actions, list) else [])], limit=980)}"
                    f" ids_sample={_safe_repr((ids[:10] if isinstance(ids, list) else ids), limit=520)}",
                    flush=True,
                )
            except Exception:
                pass

            raise RuntimeError(
                "MatchPlayerSimEnv.step: could not find the Action object for the given 5-int id (cache+converter-based). "
                "This indicates policy output is not in legal_actions, or converter output is unstable."
            )

        # 実行（ログ等の副作用は clone() 側で可能な限り無効化している前提）
        cur.act_and_regather_actions(self._match, chosen)

        # 以降の状態ではキャッシュは古いので無効化
        try:
            self._la_cache_turn = None
            self._la_cache_player_name = None
            self._la_cache_actions = None
            self._la_cache_ids = None
            self._la_cache_lookup = None
            self._la_cache_forced_len = None
            self._la_cache_forced_player_name = None
        except Exception:
            pass

        # 終局ならここで終了（turn の進行は不要）
        if self.is_terminal():
            return

        forced_after = self._forced_is_active()

        # forced 中は turn を進めない（forced を消化し切った時だけ advance 判定へ）
        if forced_after:
            return

        # ターンが切り替わるなら match.turn を進め、次手番の開始処理を行う
        try:
            can_cont = bool(getattr(cur, "can_continue", True))
        except Exception:
            can_cont = True

        if not can_cont:
            # forced が開始されていた場合は「消化し切ったタイミング」でのみ進める
            if forced_before and not forced_after:
                pass

            # Match.play_one_match の規則: ターン終了時に match.turn を必ず +1 する
            try:
                now_turn = int(getattr(self._match, "turn", 0) or 0)
            except Exception:
                now_turn = None

            if pre_turn is None or now_turn is None:
                raise RuntimeError("MatchPlayerSimEnv.step: cannot read match.turn for turn-advance check.")

            if now_turn != pre_turn:
                raise RuntimeError(
                    "MatchPlayerSimEnv.step: match.turn changed during a single action, "
                    "but Match.play_one_match only advances turn at the end of a turn. "
                    f"pre_turn={pre_turn} now_turn={now_turn}"
                )

            try:
                self._match.turn = pre_turn + 1
            except Exception:
                raise RuntimeError("MatchPlayerSimEnv.step: failed to advance match.turn on turn end.")

            nxt = self._get_current_player()

            # 次手番の開始処理（ドロー等）
            self._start_turn_for_player(nxt)

            # 念のため（can_continue を参照する実装があっても壊れないように）
            try:
                setattr(nxt, "can_continue", True)
            except Exception:
                pass

    def is_terminal(self) -> bool:
        """
        現在の状態が終局かどうかを返す。

        - match.game_over が True なら終局
        - turn > 100 は TURN_LIMIT として終局扱い
        - match._infer_end_reason() がある場合、DECK_OUT / TURN_LIMIT を補完して終局扱い
        """
        m = self._match

        if bool(getattr(m, "game_over", False)):
            return True

        try:
            t = int(getattr(m, "turn", 0) or 0)
        except Exception:
            t = 0

        if t > 100:
            try:
                if not getattr(m, "_end_reason", None):
                    setattr(m, "_end_reason", "TURN_LIMIT")
            except Exception:
                pass
            return True

        try:
            fn = getattr(m, "_infer_end_reason", None)
            if callable(fn):
                r = fn()
                if isinstance(r, str) and r.upper() in ("DECK_OUT", "TURN_LIMIT"):
                    try:
                        if not getattr(m, "_end_reason", None):
                            setattr(m, "_end_reason", r.upper())
                    except Exception:
                        pass
                    return True
        except Exception:
            pass

        return False

    def result(self) -> float:
        """
        ルート視点での最終リターン（例: 勝ち=1.0, 引き分け=0.0, 負け=-1.0）を返す。

        将来的には「どのプレイヤー視点か」を AlphaZeroMCTSPolicy 側と揃えた上で、
        match.winner / _end_reason などから値を算出する。
        現時点では match.winner とルートプレイヤー名から +1/-1/0 を算出する。
        """
        if not self.is_terminal():
            raise RuntimeError("MatchPlayerSimEnv.result called on non-terminal state.")

        root_name = self._root_player_name
        winner = getattr(self._match, "winner", None)

        if winner is None or root_name is None:
            return 0.0

        # winner が name 文字列でない実装もありうるため、可能な限り name に寄せて比較する
        try:
            if isinstance(winner, str):
                wname = winner
            else:
                wname = getattr(winner, "name", None)
        except Exception:
            wname = None

        if wname is None:
            return 0.0

        return 1.0 if wname == root_name else -1.0
