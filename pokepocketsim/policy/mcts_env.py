from __future__ import annotations

from typing import Any, List

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
        現在の手番プレイヤーの gather_actions(match) を呼び、各 Action を serialize(player) で ID 化して返す。
        （MCTS 内の key 比較を安定化するため、list は tuple 化する）
    - step(action):
        現在の合法手を再列挙し、serialize(player) が一致する Action を特定して act_and_regather_actions で 1 手進める。
    - is_terminal():
        match.game_over を参照して終局判定を行う。
    - result():
        match.winner とルートプレイヤー名から +1 / -1 / 0 を返す（非終局では例外）。

    注意:
    - deepcopy ベースの clone は重くなり得るため、MCTS を有効化する場合は性能・副作用の検証が必要。
    - clone() はログ抑制や一部フラグ無効化を行うが、完全な副作用ゼロを保証するものではない。
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

    def legal_actions(self) -> List[Any]:
        """
        現在手番プレイヤーにとっての合法手 ID リストを返す。

        実装:
        - Player.gather_actions(match) で Action 群を取得
        - 各 Action を serialize(self._player) で ID 化
        - MCTS 内部の比較/辞書キーに使いやすいよう、list は tuple に正規化して返す
        """
        actions = self._player.gather_actions(self._match)
        if not actions:
            return []

        legal_ids: List[Any] = []
        for a in actions:
            try:
                vec = a.serialize(self._player)
            except Exception:
                # シリアライズに失敗したアクションは MCTS からは扱えないため除外する。
                continue

            # ★ MCTS 内部で hash/key 比較に使えるよう tuple 正規化
            if isinstance(vec, list):
                try:
                    vec = tuple(vec)
                except Exception:
                    pass

            legal_ids.append(vec)

        return legal_ids

    def step(self, action: Any) -> None:
        """
        指定された action を 1 手適用して内部状態を進める。

        ここでの action は legal_actions() が返した ID ベクトルの 1 要素を想定する。
        現在の合法手を gather_actions で再列挙し、serialize(self._player) の結果が
        action と一致する Action を特定して act_and_regather_actions を用いて実行する。
        """
        actions = self._player.gather_actions(self._match)
        if not actions:
            return

        # ★ list/tuple の表現差で一致判定が落ちないよう正規化
        action_key = action
        if isinstance(action, list):
            try:
                action_key = tuple(action)
            except Exception:
                pass

        selected_action = None
        for a in actions:
            try:
                vec = a.serialize(self._player)
            except Exception:
                continue

            vec_key = vec
            if isinstance(vec, list):
                try:
                    vec_key = tuple(vec)
                except Exception:
                    pass

            if vec_key == action_key:
                selected_action = a
                break

        if selected_action is None:
            raise ValueError("MatchPlayerSimEnv.step: action not found in current legal_actions().")

        # 盤面更新とログ出力を含め、実環境と同じ経路で 1 手進める。
        # 戻り値（次の合法手群）はここでは利用せず、次回の legal_actions() 呼び出し時に
        # gather_actions で再度列挙する前提とする。
        self._player.act_and_regather_actions(self._match, selected_action)

    def is_terminal(self) -> bool:
        """
        現在の状態が終局かどうかを返す。

        将来的には match.game_over や winner, _end_reason を参照して
        判定する実装として、現時点では match.game_over をそのまま返す。
        """
        m = self._match
        return bool(getattr(m, "game_over", False))

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

        if not winner or root_name is None:
            return 0.0

        return 1.0 if winner == root_name else -1.0
