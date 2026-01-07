from __future__ import annotations
from typing import Any, List, Optional


class OnlineMixedPolicy:
    """
    メイン方策(main_policy)とサブ方策(fallback_policy)をオンラインで混ぜるラッパー。

    - Player.select_action からは select_action_index_online(...) が呼ばれる。
    - 内部では main / fallback が select_action_index[_online] を持っていれば再利用し、
      どちらも使えない／エラー時はランダムにフォールバックする。
    """

    def __init__(
        self,
        main_policy: Optional[Any] = None,
        fallback_policy: Optional[Any] = None,
        mix_prob: float = 0.5,
        rng: Optional[Any] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        """
        mix_prob:
            0.0   → 常に fallback_policy（あれば）を使う
            1.0   → 常に main_policy（あれば）を使う
            0〜1 → mix_prob の確率で main、それ以外で fallback を使う
        rng:
            乱数生成器 (random.Random 互換) があれば渡す。None なら内部で random モジュールを使う。
        """
        self.main_policy = main_policy
        self.fallback_policy = fallback_policy
        self.mix_prob = float(mix_prob)
        self.rng = rng
        self.model_dir = model_dir

        # メイン方策が実際に使われているかを検知するためのカウンタ
        self._main_success_calls = 0
        self._main_failed_calls = 0
        self._fallback_calls = 0

        # --- 統計用カウンタ（main=モデル, fallback=サブ, random=完全ランダム） ---
        self.stats_from_model = 0
        self.stats_from_fallback = 0
        self.stats_from_random = 0
        self.stats_errors = 0
        self.stats_total = 0
        self.last_source: Optional[str] = None

        # --- 追加：Player 側が参照する“正式API” ---
        self.last_decide_info: dict = {}
        self._last_decide_info: dict = self.last_decide_info

        # --- 追加：終局 [SUMMARY] 用の集計 ---
        self._sum_total = 0
        self._sum_used_pi = 0
        self._sum_used_q = 0
        self._sum_skip_q = 0
        self._sum_last_reason = None

    def _as_list_vec(self, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        try:
            tl = getattr(v, "tolist", None)
        except Exception:
            tl = None
        if callable(tl):
            try:
                vv = tl()
            except Exception:
                vv = None
            if isinstance(vv, list):
                return vv
            if isinstance(vv, tuple):
                return list(vv)
        return None

    def _dbg_budget_print(self, key: str, msg: str, default_budget: int = 8) -> None:
        """
        key ごとに出力回数を制限する（ログ氾濫防止）。
        """
        try:
            k = f"_dbg_budget__{key}"
            b = getattr(self, k, None)
            if b is None:
                b = int(default_budget)
            b = int(b)
            if b <= 0:
                return
            setattr(self, k, b - 1)
        except Exception:
            return
        try:
            print(msg, flush=True)
        except Exception:
            pass

    def _short_repr(self, v, max_chars: int = 240) -> str:
        """
        巨大/危険な repr を抑制しつつ、何が来たか分かる短い文字列にする。
        """
        try:
            s = repr(v)
        except Exception as e:
            s = f"<repr_fail:{e!r}>"
        try:
            if s is None:
                s = "None"
            if len(s) > int(max_chars):
                s = s[: int(max_chars)] + "...(truncated)"
        except Exception:
            pass
        return s

    def _format_list_head_tail(self, items: Any, full: bool = False) -> str:
        import os

        if not isinstance(items, list):
            return self._short_repr(items, max_chars=1200)

        mode = str(os.getenv("MCTS_ENV_LA5_LIST_MODE", "headtail")).strip().lower()
        if full or mode == "full":
            return self._short_repr(items, max_chars=6000)

        try:
            k = int(os.getenv("MCTS_ENV_LA5_LIST_K", "6") or "6")
        except Exception:
            k = 6

        if k <= 0 or len(items) <= 2 * k:
            return self._short_repr(items, max_chars=2400)

        head = items[:k]
        tail = items[-k:]
        return f"{self._short_repr(head, max_chars=2400)}...{self._short_repr(tail, max_chars=2400)}(len={len(items)})"

    def _trace_enabled(self) -> bool:
        import os

        return str(os.getenv("MCTS_ENV_TRACE", "0")).strip() == "1"

    def _next_la5_event_id(self) -> str:
        import time

        try:
            c = int(getattr(self, "_la5_event_counter", 0) or 0) + 1
        except Exception:
            c = 1
        try:
            setattr(self, "_la5_event_counter", c)
        except Exception:
            pass
        return f"la5_online_{int(time.time() * 1000)}_{int(c)}"

    def _hash_vecs(self, vecs: Any) -> Optional[str]:
        from .trace_utils import hash_vecs

        return hash_vecs(vecs)

    def _hash_vec(self, vec: Any) -> Optional[str]:
        return self._hash_vecs(vec)

    def _value_outline(self, v, sample_n: int = 6) -> dict:
        """
        値の「型・長さ・先頭サンプル・dictキー等」を安全に要約して返す。
        """
        out = {}
        try:
            out["type"] = type(v).__name__
        except Exception:
            out["type"] = "unknown"

        try:
            out["is_none"] = int(v is None)
        except Exception:
            out["is_none"] = 0

        # len
        try:
            out["len"] = int(len(v))  # type: ignore[arg-type]
        except Exception:
            out["len"] = None

        # dict keys
        try:
            if isinstance(v, dict):
                ks = []
                try:
                    for k in v.keys():
                        ks.append(type(k).__name__)
                        if len(ks) >= int(sample_n):
                            break
                except Exception:
                    ks = []
                out["dict_key_types"] = ks

                kk = []
                try:
                    for k in v.keys():
                        kk.append(self._short_repr(k, max_chars=80))
                        if len(kk) >= int(sample_n):
                            break
                except Exception:
                    kk = []
                out["dict_keys_head"] = kk
        except Exception:
            pass

        # list/tuple first elems + nested hint
        try:
            if isinstance(v, (list, tuple)):
                head = []
                types = []
                try:
                    for x in v[: int(sample_n)]:
                        head.append(self._short_repr(x, max_chars=80))
                        try:
                            types.append(type(x).__name__)
                        except Exception:
                            types.append("unknown")
                except Exception:
                    head = []
                    types = []
                out["head"] = head
                out["elem_types_head"] = types

                # 1xN / NxM っぽいか
                try:
                    if len(v) == 1 and isinstance(v[0], (list, tuple)):
                        out["shape_hint"] = f"1x{len(v[0])}"
                    elif len(v) > 0 and all(isinstance(x, (list, tuple)) for x in v[: min(len(v), 8)]):
                        w0 = None
                        ok = True
                        for x in v[: min(len(v), 8)]:
                            if w0 is None:
                                w0 = len(x)
                            elif len(x) != w0:
                                ok = False
                                break
                        if w0 is not None:
                            out["shape_hint"] = f"{len(v)}x{w0}" if ok else f"{len(v)}x(var)"
                except Exception:
                    pass
        except Exception:
            pass

        # fallback repr
        try:
            out["repr"] = self._short_repr(v)
        except Exception:
            out["repr"] = "<repr_fail>"

        return out

    def _dbg_unexpected_value(self, key: str, v, reason: str, state_dict: Optional[dict] = None, extra: Optional[dict] = None, default_budget: int = 8) -> None:
        """
        「本来読みたい形態ではない」値が来たときに、
        何が来たか分かるように概要をログ & state_dict に保存する。
        """
        try:
            info = self._value_outline(v)
        except Exception:
            info = {"type": type(v).__name__ if v is not None else "NoneType", "repr": "<outline_fail>"}

        if extra is not None:
            try:
                if isinstance(extra, dict):
                    info.update(extra)
            except Exception:
                pass

        self._dbg_budget_print(
            f"unexpected__{key}",
            f"[ONLINE_MIX][UNEXPECTED] key={key} reason={reason} info={self._short_repr(info, max_chars=500)}",
            default_budget=default_budget,
        )

        if not isinstance(state_dict, dict):
            return

        try:
            rec = {"key": str(key), "reason": str(reason)}
            try:
                rec.update(info)
            except Exception:
                pass

            buf = state_dict.get("online_mix_unexpected", None)
            if not isinstance(buf, list):
                buf = []
                state_dict["online_mix_unexpected"] = buf

            # 肥大化防止（最大 50 件）
            if len(buf) < 50:
                buf.append(rec)
        except Exception:
            pass

    def _coerce_numeric_vec(self, v, target_dim: Optional[int] = None):
        """
        v を「1次元の数値list[float]」へ強制変換する。
        失敗時は (None, reason) を返す。
        """
        vv = self._as_list_vec(v)
        if vv is None:
            return None, "not_list"
        try:
            # [[...]] / [[[...]]] のような 1xN ネストを剥く
            guard = 0
            while isinstance(vv, list) and len(vv) == 1 and isinstance(vv[0], (list, tuple)) and guard < 4:
                vv = list(vv[0]) if isinstance(vv[0], tuple) else vv[0]
                guard += 1
        except Exception:
            pass

        try:
            import numpy as np
            arr = np.asarray(vv, dtype=np.float32).reshape(-1)
            out = arr.tolist()
        except Exception:
            return None, "astype_float32_fail"

        if not isinstance(out, list) or not out:
            return None, "empty"

        if target_dim is not None:
            try:
                if len(out) != int(target_dim):
                    return None, f"dim_mismatch:{len(out)}!={int(target_dim)}"
            except Exception:
                return None, "dim_check_fail"

        # NaN/inf の混入を弾く（numpy float は Python float に変換済み）
        try:
            import math
            for x in out:
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    return None, "non_numeric_elem"
                if not math.isfinite(float(x)):
                    return None, "non_finite"
        except Exception:
            return None, "finite_check_fail"

        return out, "ok"

    def _normalize_pi_to_len(self, pi, n: int):
        """
        π を長さ n の list[float] に正規化する。
        対応:
          - list/tuple/np.ndarray
          - dict {idx:prob}
        """
        if pi is None:
            return None, "pi_none"

        # dict → list
        if isinstance(pi, dict):
            out = [0.0] * int(n)
            try:
                for k, v in pi.items():
                    try:
                        i = int(k)
                        if 0 <= i < int(n):
                            out[i] = float(v)
                    except Exception:
                        continue
            except Exception:
                return None, "pi_dict_iter_fail"
            return out, "ok_dict"

        vv = self._as_list_vec(pi)
        if vv is None:
            return None, "pi_not_seq"

        try:
            out = [float(x) for x in vv]
        except Exception:
            return None, "pi_float_cast_fail"

        try:
            if len(out) != int(n):
                return None, f"pi_len_mismatch:{len(out)}!={int(n)}"
        except Exception:
            return None, "pi_len_check_fail"

        return out, "ok_seq"

    def get_obs_vec(self, state_dict=None, actions=None, player=None):
        """
        PhaseD-Q（outer）から見える位置に obs_vec getter を用意する。
        優先順位:
          1) state_dict["obs_vec"]
          2) main_policy.get_obs_vec / encode_obs_vec
          3) player.match.encoder から生成（encode_obs_vec/encode_state など）
        """
        # 1) 既にあるならそれを返す
        if isinstance(state_dict, dict):
            vv = self._as_list_vec(state_dict.get("obs_vec", None))
            if vv is not None:
                return vv

        # 2) main_policy に委譲（ここが主ルート）
        mp = getattr(self, "main_policy", None)
        if mp is not None:
            fn = getattr(mp, "get_obs_vec", None)
            if callable(fn):
                vv = self._as_list_vec(fn(state_dict=state_dict, actions=actions, player=player))
                if vv is not None:
                    return vv
            fn = getattr(mp, "encode_obs_vec", None)
            if callable(fn):
                vv = self._as_list_vec(fn(state_dict=state_dict, actions=actions, player=player))
                if vv is not None:
                    return vv

        # 3) match.encoder から生成（最後の保険）
        m = getattr(player, "match", None) if player is not None else None
        enc = getattr(m, "encoder", None) if m is not None else None
        if enc is None:
            return None

        for meth in ("encode_obs_vec", "encode_obs", "encode_for_player", "encode", "build_obs_vec", "make_obs_vec", "get_obs_vec"):
            fn = getattr(enc, meth, None)
            if not callable(fn):
                continue
            for args in ((m, player, actions), (m, player), (player, actions), (player,), (m, actions), (m,), ()):
                try:
                    out = fn(*args)
                except TypeError:
                    continue
                except Exception:
                    out = None
                vv = self._as_list_vec(out)
                if vv is not None:
                    return vv

        fn = getattr(enc, "encode_state", None)
        if callable(fn) and m is not None and player is not None:
            try:
                me_fn = getattr(player, "public_state", None)
                me = me_fn() if callable(me_fn) else None

                opp = None
                ps = getattr(m, "public_state", None)
                players = ps.get("players") if isinstance(ps, dict) else None
                if not isinstance(players, list) or not players:
                    sp = getattr(m, "starting_player", None)
                    sp2 = getattr(m, "second_player", None)
                    players = [x for x in (sp, sp2) if x is not None]
                for p0 in players or []:
                    if p0 is not None and p0 is not player:
                        opp_fn = getattr(p0, "public_state", None)
                        if callable(opp_fn):
                            opp = opp_fn()
                        break

                feat = {"me": me if isinstance(me, dict) else {}, "opp": opp if isinstance(opp, dict) else {}}
                out = fn(feat)

                try:
                    import numpy as np
                    arr = np.asarray(out, dtype=np.float32).reshape(-1)
                    return arr.tolist()
                except Exception:
                    return self._as_list_vec(out)
            except Exception:
                return None

        return None

    def encode_obs_vec(self, state_dict=None, actions=None, player=None):
        # PhaseD-Q 側の探索に引っかかりやすい別名
        return self.get_obs_vec(state_dict=state_dict, actions=actions, player=player)

    # --------------------------------------------------
    # 内部ヘルパ: ソース種別を記録
    # --------------------------------------------------
    def _record_source(self, source: Optional[str]) -> None:
        self.last_source = source
        self.stats_total += 1
        if source == "model":
            self.stats_from_model += 1
        elif source == "fallback":
            self.stats_from_fallback += 1
        elif source == "random":
            self.stats_from_random += 1

    def _maybe_raise_if_main_never_used(self) -> None:
        """
        main_policy が一度も成功していないのに試合が進んでいる場合、
        設計ミス/配線ミスの早期発見用に例外化できるフック。

        既定では何もしない。環境変数 STRICT_MAIN_NEVER_USED=1 のときのみ有効。
        """
        try:
            import os
            if str(os.getenv("STRICT_MAIN_NEVER_USED", "0")).strip() != "1":
                return
        except Exception:
            return

        try:
            min_total = int(os.getenv("STRICT_MAIN_NEVER_USED_MIN_TOTAL", "50"))
        except Exception:
            min_total = 50

        if int(getattr(self, "stats_total", 0)) < int(min_total):
            return

        if int(getattr(self, "_main_success_calls", 0)) <= 0:
            raise RuntimeError(
                f"main_policy never used: stats_total={getattr(self, 'stats_total', 0)} "
                f"main_failed={getattr(self, '_main_failed_calls', 0)} "
                f"fallback_calls={getattr(self, '_fallback_calls', 0)}"
            )

    # --------------------------------------------------
    # 内部ヘルパ: 任意の policy から「インデックス」を1つもらう
    # --------------------------------------------------
    def _select_from_policy(self, policy: Any, state_dict: dict, actions: List[Any], player: Any, kind: str) -> Optional[int]:
        if policy is None or not actions:
            return None

        try:
            if hasattr(policy, "select_action_index_online"):
                try:
                    idx = policy.select_action_index_online(state_dict, actions, player=player, return_pi=True)
                except TypeError:
                    try:
                        idx = policy.select_action_index_online(state_dict, actions, player=player)
                    except TypeError:
                        idx = policy.select_action_index_online(state_dict, actions, player)
            elif hasattr(policy, "select_action_index"):
                try:
                    idx = policy.select_action_index(state_dict, actions, player=player)
                except TypeError:
                    idx = policy.select_action_index(state_dict, actions, player)
            else:
                return None

            pi = None
            if isinstance(idx, tuple) and len(idx) == 2:
                idx, pi = idx

            # ★追加: 戻り値に pi が載らない実装でも、state_dict / policy 属性から π を回収する
            _pi_from = None
            if pi is None and isinstance(state_dict, dict):
                try:
                    pi = (
                        state_dict.get("mcts_pi", None)
                        or state_dict.get("pi", None)
                        or state_dict.get("_mcts_pi", None)
                        or state_dict.get("root_pi", None)
                        or state_dict.get("policy_pi", None)
                    )
                    if pi is not None:
                        _pi_from = "state_dict"
                except Exception:
                    pi = None

            if pi is None:
                for _attr in (
                    "mcts_pi", "_mcts_pi",
                    "pi", "_pi",
                    "last_pi", "_last_pi",
                    "last_mcts_pi", "_last_mcts_pi",
                    "root_pi", "_root_pi",
                    "policy_pi", "_policy_pi",
                    "last_policy_pi", "_last_policy_pi",
                    "last_root_pi", "_last_root_pi",
                    "mcts_probs", "_mcts_probs",
                ):
                    try:
                        _v = getattr(policy, _attr, None)
                    except Exception:
                        _v = None
                    if _v is not None:
                        pi = _v
                        _pi_from = f"attr:{_attr}"
                        break

            # ★追加: policy.mcts / policy._mcts の内側も探索（実装差吸収）
            if pi is None:
                for _mcts_attr in ("mcts", "_mcts"):
                    try:
                        _m = getattr(policy, _mcts_attr, None)
                    except Exception:
                        _m = None
                    if _m is None:
                        continue
                    for _attr in ("pi", "_pi", "root_pi", "_root_pi", "last_pi", "_last_pi", "last_root_pi", "_last_root_pi"):
                        try:
                            _v = getattr(_m, _attr, None)
                        except Exception:
                            _v = None
                        if _v is not None:
                            pi = _v
                            _pi_from = f"{_mcts_attr}.{_attr}"
                            break
                    if pi is not None:
                        break

            # ★追加: π を actions 長に正規化できるかチェック（できない場合は理由をログ）
            _pi_norm = None
            _pi_norm_reason = None
            try:
                _pi_norm, _pi_norm_reason = self._normalize_pi_to_len(pi, len(actions))
            except Exception:
                _pi_norm = None
                _pi_norm_reason = "pi_norm_exception"

            if kind == "main":
                if _pi_norm is None:
                    self._dbg_budget_print(
                        "pi_missing",
                        f"[ONLINE_MIX][PI][WARN] kind=main pi_unusable reason={_pi_norm_reason} pi_from={_pi_from} actions={len(actions)} policy={getattr(policy, '__class__', type(policy)).__name__}",
                        default_budget=10,
                    )
                    self._dbg_unexpected_value(
                        "pi",
                        pi,
                        str(_pi_norm_reason),
                        state_dict=state_dict if isinstance(state_dict, dict) else None,
                        extra={"pi_from": _pi_from, "actions": len(actions), "policy": getattr(policy, '__class__', type(policy)).__name__},
                        default_budget=10,
                    )
                else:
                    self._dbg_budget_print(
                        "pi_ok",
                        f"[ONLINE_MIX][PI] kind=main pi_ok from={_pi_from} n={len(_pi_norm)} policy={getattr(policy, '__class__', type(policy)).__name__}",
                        default_budget=4,
                    )

            # 以降は正規化済み π を採用（あれば）
            if _pi_norm is not None:
                pi = _pi_norm

            # ★追加: main の π 受領状況（有無/型/長さ）を state_dict に必ず残す
            if isinstance(state_dict, dict) and kind == "main":
                _pi_present = 0
                _pi_len = -1
                _pi_type = "none"
                try:
                    if pi is not None:
                        _pi_present = 1
                        _pi_type = type(pi).__name__
                        try:
                            _pi_len = len(pi)
                        except Exception:
                            _pi_len = -1
                except Exception:
                    _pi_present = 0
                    _pi_len = -1
                    _pi_type = "err"

                try:
                    state_dict["mcts_pi_present"] = int(_pi_present)
                    state_dict["mcts_pi_len"] = int(_pi_len)
                    state_dict["mcts_pi_type"] = str(_pi_type)
                    if _pi_from is not None:
                        state_dict["mcts_pi_from"] = str(_pi_from)
                except Exception:
                    pass

            try:
                idx_i = int(idx)
            except Exception:
                return None

            if 0 <= idx_i < len(actions):
                # ★追加: main の π が無い/不正なら one-hot で必ず用意（PhaseD-Q が必ず回収できるように）
                if kind == "main" and _pi_norm is None:
                    try:
                        pi = [0.0] * int(len(actions))
                        pi[idx_i] = 1.0
                        _pi_from = "onehot_idx"
                        _pi_norm = pi
                        _pi_norm_reason = "onehot"
                    except Exception:
                        pass

                if _pi_norm is not None:
                    pi = _pi_norm

                # ★追加: OnlineMixedPolicy 自身にも π を残す（外側の attr 探索対策）
                try:
                    if kind == "main" and pi is not None:
                        self.last_pi = pi
                        self.mcts_pi = pi
                        self.last_mcts_pi = pi
                except Exception:
                    pass

                # ★追加: main の π 受領状況（有無/型/長さ）を state_dict に必ず残す（onehot 補完後に上書き）
                if isinstance(state_dict, dict) and kind == "main":
                    try:
                        state_dict["mcts_pi_present"] = int(1 if pi is not None else 0)
                        state_dict["mcts_pi_len"] = int(len(pi)) if pi is not None else -1
                        state_dict["mcts_pi_type"] = type(pi).__name__ if pi is not None else "none"
                        if _pi_from is not None:
                            state_dict["mcts_pi_from"] = str(_pi_from)
                    except Exception:
                        pass

                # ★追加: PhaseD-Q 側が参照する “純MCTSの選択idx/pi” を state_dict に保持
                try:
                    if isinstance(state_dict, dict) and kind == "main":
                        state_dict["mcts_idx"] = idx_i
                        if pi is not None:
                            state_dict["mcts_pi"] = pi
                            state_dict["pi"] = pi
                except Exception:
                    pass
                return idx_i
        except Exception as e:
            import os
            import time
            import traceback
            from ..debug_dump import write_debug_dump
            self.stats_errors += 1
            policy_name = getattr(policy, "__class__", type(policy)).__name__
            game_id = None
            turn = None
            player_name = None
            forced_active = None
            forced_len = None
            try:
                m = getattr(player, "match", None) if player is not None else None
                game_id = getattr(m, "game_id", None) if m is not None else None
                turn = getattr(m, "turn", None) if m is not None else None
                if m is not None:
                    fa = getattr(m, "forced_actions", None)
                    if isinstance(fa, (list, tuple)):
                        forced_len = int(len(fa))
                        forced_active = forced_len > 0
            except Exception:
                game_id = None
                turn = None
                forced_active = None
                forced_len = None
            try:
                player_name = getattr(player, "name", None) if player is not None else None
            except Exception:
                player_name = None
            print(
                "[ONLINE_MIX][ERROR]"
                f" game_id={game_id}"
                f" turn={turn}"
                f" player={player_name}"
                f" forced_active={forced_active}"
                f" forced_len={forced_len}"
                f" policy={policy_name}"
                f" kind={kind}"
                f" mix_prob={float(getattr(self, 'mix_prob', 0.0))}"
                f" main_policy={getattr(getattr(self, 'main_policy', None), '__class__', type(None)).__name__}"
                f" fallback_policy={getattr(getattr(self, 'fallback_policy', None), '__class__', type(None)).__name__}"
                f" fallback_calls={int(getattr(self, '_fallback_calls', 0) or 0)}"
                f" failed_in=select_action_index(_online) err={e!r}",
                flush=True,
            )
            try:
                event_id = None
                try:
                    event_id = getattr(policy, "last_step_no_match_event_id", None)
                except Exception:
                    event_id = None
                if event_id is None:
                    try:
                        event_id = getattr(policy, "_last_step_no_match_event_id", None)
                    except Exception:
                        event_id = None
                if event_id is None:
                    try:
                        m = getattr(player, "match", None) if player is not None else None
                        event_id = getattr(m, "_last_step_no_match_event_id", None) if m is not None else None
                    except Exception:
                        event_id = None
                if isinstance(state_dict, dict):
                    state_dict["last_step_no_match_event_id"] = event_id
                    state_dict["last_step_no_match_error"] = str(e)
            except Exception:
                pass
            try:
                forced_active = None
                forced_len = None
                turn_ctx = None
                player_ctx = None
                game_id_ctx = None
                if isinstance(state_dict, dict):
                    turn_ctx = state_dict.get("turn", None)
                try:
                    m = getattr(player, "match", None) if player is not None else None
                    game_id_ctx = getattr(m, "game_id", None) if m is not None else None
                    turn_ctx = getattr(m, "turn", None) if m is not None else None
                    player_ctx = getattr(player, "name", None) if player is not None else None
                    fa = getattr(m, "forced_actions", None) if m is not None else None
                    if isinstance(fa, (list, tuple)):
                        forced_len = int(len(fa))
                        forced_active = forced_len > 0
                except Exception:
                    game_id_ctx = None
                    turn_ctx = None
                    player_ctx = None
                    forced_active = None
                    forced_len = None

                legal_actions_serialized = []
                try:
                    for i, a in enumerate(actions or []):
                        vec = None
                        try:
                            fn = getattr(a, "to_id_vec", None)
                            if callable(fn):
                                try:
                                    vec = fn(player=player)
                                except TypeError:
                                    vec = fn(player)
                        except Exception:
                            vec = None
                        if vec is None:
                            try:
                                fn = getattr(a, "serialize", None)
                                if callable(fn):
                                    try:
                                        vec = fn(player=player)
                                    except TypeError:
                                        vec = fn(player)
                            except Exception:
                                vec = None
                        if isinstance(vec, tuple):
                            vec = list(vec)
                        if not isinstance(vec, list):
                            vec = [0, 0, 0, 0, 0]
                        if len(vec) < 5:
                            vec = list(vec) + [0] * (5 - len(vec))
                        if len(vec) > 5:
                            vec = list(vec)[:5]

                        legal_actions_serialized.append(
                            {
                                "i": i,
                                "action_type": getattr(a, "action_type", None),
                                "name": getattr(a, "name", None),
                                "vec": vec,
                            }
                        )
                except Exception:
                    legal_actions_serialized = []

                err_type = type(e).__name__
                err_msg = str(e)
                payload = {
                    "error_type": err_type,
                    "error_message": err_msg,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "run_context": {
                        "game_id": game_id_ctx,
                        "turn": turn_ctx,
                        "player": player_ctx,
                        "forced_actions_active": forced_active,
                    },
                    "action_context": {
                        "selected_vec": None,
                        "selected_source": str(kind),
                        "legal_actions_serialized": legal_actions_serialized,
                    },
                    "mcts_context": None,
                    "traceback": traceback.format_exception(type(e), e, e.__traceback__),
                }
                import os

                dump_path = write_debug_dump(payload)
                try:
                    cwd = os.getcwd()
                    rel = os.path.relpath(dump_path, cwd)
                except Exception:
                    cwd = None
                    rel = dump_path
                print(f"[DEBUG_DUMP] wrote: {rel} cwd={cwd}", flush=True)
            except Exception:
                pass
            if os.getenv("STRICT_POLICY_ERROR") == "1":
                # モデル使用が前提なのに失敗した場合は、環境変数で即エラー終了も選べる
                raise
            return None

        return None


    # --------------------------------------------------
    # Player.select_action から呼ばれる入口
    # --------------------------------------------------
    def select_action_index_online(self, state_dict: dict, actions: List[Any], player: Any = None) -> int:
        """
        actions: Player.gather_actions(...) が返す Action のリスト
        戻り値: その中から選ぶ action のインデックス
        """
        if not actions:
            try:
                self._sum_total = int(getattr(self, "_sum_total", 0)) + 1
            except Exception:
                pass
            try:
                self.last_source = "no_actions"
            except Exception:
                pass
            try:
                self.last_decide_info = {
                    "mode": "online_mix",
                    "source": "no_actions",
                    "selected_idx": 0,
                    "rnd": None,
                    "mix_prob": float(getattr(self, "mix_prob", 0.0)),
                    "p": float(getattr(self, "mix_prob", 0.0)),
                    "why": "no_actions",
                }
                self._last_decide_info = self.last_decide_info
            except Exception:
                pass
            return 0

        try:
            self._sum_total = int(getattr(self, "_sum_total", 0)) + 1
            self._sum_used_main = int(getattr(self, "_sum_used_main", 0))
            self._sum_used_fallback = int(getattr(self, "_sum_used_fallback", 0))
            self._sum_used_random = int(getattr(self, "_sum_used_random", 0))
        except Exception:
            pass

        # ★追加: PhaseD-Q (wrapper) が参照する obs_vec / action_candidates_vec を可能な範囲で補完
        try:
            if isinstance(state_dict, dict):
                m = getattr(player, "match", None) if player is not None else None

                converter = None

                # ★main 優先: main_policy / fallback_policy が converter を持っていれば最優先で使う
                for _pol_src in (getattr(self, "main_policy", None), getattr(self, "fallback_policy", None)):
                    if _pol_src is None:
                        continue
                    try:
                        converter = getattr(_pol_src, "converter", None)
                        if converter is None:
                            converter = getattr(_pol_src, "action_converter", None)
                    except Exception:
                        converter = None
                    if converter is not None:
                        break

                # 次点: match 側
                if converter is None and m is not None:
                    try:
                        converter = getattr(m, "converter", None)
                        if converter is None:
                            converter = getattr(m, "action_converter", None)
                    except Exception:
                        converter = None

                # 次点: state_dict に入っているもの
                if converter is None:
                    try:
                        converter = state_dict.get("converter", None) or state_dict.get("action_converter", None)
                    except Exception:
                        converter = None

                # 最後: globals
                if converter is None:
                    converter = globals().get("converter", None)

                # ★確定した converter を state_dict に保持（以降の層で outer を参照しないように）
                try:
                    if converter is not None:
                        state_dict["converter"] = converter
                        state_dict["action_converter"] = converter
                except Exception:
                    pass

                try:
                    if player is not None and "player" not in state_dict:
                        state_dict["player"] = player
                    if m is not None:
                        state_dict.setdefault("_match", m)
                        state_dict.setdefault("match", m)
                except Exception:
                    pass

                # obs_vec を作れそうなら埋める（match.encoder 経由など）
                # ★追加: main_policy 側の想定 obs_dim（例: 2448）に一致するものだけ採用する
                _target_obs_dim = None
                try:
                    _target_obs_dim = getattr(self.main_policy, "obs_dim", None)
                    if _target_obs_dim is None:
                        _target_obs_dim = getattr(self.main_policy, "_obs_dim", None)
                except Exception:
                    _target_obs_dim = None

                def _as_list_vec(_v):
                    if _v is None:
                        return None
                    if isinstance(_v, list):
                        return _v
                    if isinstance(_v, tuple):
                        return list(_v)
                    try:
                        _tl = getattr(_v, "tolist", None)
                    except Exception:
                        _tl = None
                    if callable(_tl):
                        try:
                            _vv = _tl()
                        except Exception:
                            _vv = None
                        if isinstance(_vv, list):
                            return _vv
                        if isinstance(_vv, tuple):
                            return list(_vv)
                    return None

                def _flatten_1xn(_v):
                    # [[...]] → [...] を吸収（encoder が 1xN を返すパターン対策）
                    try:
                        if isinstance(_v, tuple):
                            _v = list(_v)

                        # 1×N が多重にネストされて返るケース（例: [[[...]]]]）も剥く
                        _guard = 0
                        while isinstance(_v, list) and len(_v) == 1 and isinstance(_v[0], (list, tuple)) and _guard < 4:
                            _v = list(_v[0]) if isinstance(_v[0], tuple) else _v[0]
                            _guard += 1

                        # 2D(list of list) で target_dim に一致するならフラット化
                        if _target_obs_dim is not None and isinstance(_v, list) and _v and all(isinstance(x, (list, tuple)) for x in _v):
                            total = 0
                            ok = True
                            for x in _v:
                                try:
                                    total += len(x)
                                except Exception:
                                    ok = False
                                    break
                            if ok and total == int(_target_obs_dim):
                                out = []
                                for x in _v:
                                    try:
                                        out.extend(list(x) if isinstance(x, tuple) else x)
                                    except Exception:
                                        return _v
                                return out
                    except Exception:
                        pass
                    return _v

                def _dim_ok(_v):
                    if not isinstance(_v, (list, tuple)):
                        return False
                    try:
                        _v2 = _flatten_1xn(list(_v) if isinstance(_v, tuple) else _v)
                    except Exception:
                        return False
                    try:
                        if isinstance(_v2, (list, tuple)) and len(_v2) <= 0:
                            return False
                    except Exception:
                        return False
                    if _target_obs_dim is None:
                        return True
                    try:
                        return len(_v2) == int(_target_obs_dim)
                    except Exception:
                        return False

                def _try_call_for_obs(_obj, _meth_names):
                    if _obj is None:
                        return None
                    for _mn in _meth_names:
                        try:
                            _fn = getattr(_obj, _mn, None)
                        except Exception:
                            _fn = None
                        if not callable(_fn):
                            continue

                        # できるだけ「情報が多い引数」から当てに行く
                        for _args in ((state_dict, actions, player), (state_dict, actions), (m, player, actions), (m, player), (player, actions), (player,), ()):
                            try:
                                _out = _fn(*_args)
                            except TypeError:
                                continue
                            except Exception:
                                _out = None

                            try:
                                if isinstance(_out, dict):
                                    _out = _out.get("obs_vec", None)
                            except Exception:
                                pass

                            _out = _as_list_vec(_out)
                            _out = _flatten_1xn(_out) if _out is not None else None
                            if _out is not None and _dim_ok(_out):
                                return _out
                    return None

                _has_obs = ("obs_vec" in state_dict)
                if _has_obs:
                    try:
                        _v0 = state_dict.get("obs_vec", None)
                    except Exception:
                        _v0 = None

                    try:
                        if isinstance(_v0, dict):
                            _v0 = _v0.get("obs_vec", None)
                    except Exception:
                        pass

                    _v0 = _as_list_vec(_v0)
                    _v0 = _flatten_1xn(_v0) if _v0 is not None else None

                    _v0_empty = False
                    try:
                        _v0_empty = (isinstance(_v0, (list, tuple)) and len(_v0) <= 0)
                    except Exception:
                        _v0_empty = True

                    if _v0 is None or _v0_empty or (_target_obs_dim is not None and (not _dim_ok(_v0))):
                        try:
                            del state_dict["obs_vec"]
                        except Exception:
                            state_dict["obs_vec"] = None

                # ★ 追加: obs_vec が無ければ encoder / converter から生成して注入
                if ("obs_vec" not in state_dict) or (state_dict.get("obs_vec", None) is None):
                    _ov = None

                    # player を拾う（スコープに player が無い実装もあるので保険）
                    try:
                        _p = state_dict.get("player", None)
                    except Exception:
                        _p = None
                    if _p is None:
                        try:
                            _p = player
                        except Exception:
                            _p = None

                    # match を拾う（encoder が match 依存の可能性に備える）
                    try:
                        _mt = state_dict.get("_match", None) or state_dict.get("match", None)
                    except Exception:
                        _mt = None
                    if _mt is None:
                        try:
                            _mt = getattr(converter, "_match", None) or getattr(converter, "match", None)
                        except Exception:
                            _mt = None
                    if _mt is None:
                        try:
                            _mt = getattr(m, "_match", None) or getattr(m, "match", None)
                        except Exception:
                            _mt = None

                    # encoder 取得
                    try:
                        _enc = getattr(m, "encoder", None)
                    except Exception:
                        _enc = None
                    if _enc is None:
                        try:
                            _enc = getattr(_mt, "encoder", None)
                        except Exception:
                            _enc = None

                    # 呼び出し候補（実装差を吸収）
                    _names = (
                        "encode_obs_vec", "encode_obs", "make_obs_vec", "get_obs_vec",
                        "encode_player_obs", "encode_player", "encode_state", "encode_public",
                        "obs_vec"
                    )

                    # まず encoder → ダメなら converter の順
                    for _src in (_enc, converter):
                        if _src is None:
                            continue

                        for _nm in _names:
                            try:
                                _fn = getattr(_src, _nm, None)
                            except Exception:
                                _fn = None
                            if not callable(_fn):
                                continue

                            # 1) keyword で試す（match / player の両対応）
                            if _ov is None:
                                try:
                                    if _mt is not None and _p is not None:
                                        _ov = _fn(match=_mt, player=_p)
                                    elif _p is not None:
                                        _ov = _fn(player=_p)
                                except Exception:
                                    _ov = None

                            # 2) positional で試す
                            if _ov is None:
                                try:
                                    if _mt is not None and _p is not None:
                                        _ov = _fn(_mt, _p)
                                    elif _p is not None:
                                        _ov = _fn(_p)
                                except Exception:
                                    _ov = None

                            if _ov is not None:
                                break

                        if _ov is not None:
                            break

                    # 得られたら正規化して投入（空・次元不正は捨てる）
                    _ov = _as_list_vec(_ov)
                    _ov = _flatten_1xn(_ov) if _ov is not None else None

                    _ov_empty = False
                    try:
                        _ov_empty = (isinstance(_ov, (list, tuple)) and len(_ov) <= 0)
                    except Exception:
                        _ov_empty = True

                    if _ov is not None and (not _ov_empty) and (_target_obs_dim is None or _dim_ok(_ov)):
                        _ov2, _ov_reason = self._coerce_numeric_vec(_ov, target_dim=_target_obs_dim)
                        if _ov2 is not None:
                            state_dict["obs_vec"] = _ov2
                            try:
                                state_dict["obs_vec_numeric"] = 1
                                state_dict["obs_vec_len"] = int(len(_ov2))
                                state_dict["obs_vec_reason"] = str(_ov_reason)
                            except Exception:
                                pass
                            try:
                                if not getattr(m, "_obs_inject_once", False):
                                    setattr(m, "_obs_inject_once", True)
                                    print(f"[ONLINE_MIX][OBS] inject_ok len={len(_ov2)} reason={_ov_reason} src={'encoder' if _enc is not None else 'converter'}", flush=True)
                            except Exception:
                                pass
                        else:
                            try:
                                state_dict["obs_vec"] = None
                                state_dict["obs_vec_numeric"] = 0
                                state_dict["obs_vec_reason"] = str(_ov_reason)
                            except Exception:
                                pass
                            self._dbg_budget_print(
                                "obs_bad",
                                f"[ONLINE_MIX][OBS][WARN] inject_bad reason={_ov_reason} target_dim={_target_obs_dim}",
                                default_budget=10,
                            )
                            self._dbg_unexpected_value(
                                "obs_vec",
                                _ov,
                                str(_ov_reason),
                                state_dict=state_dict if isinstance(state_dict, dict) else None,
                                extra={"target_dim": _target_obs_dim},
                                default_budget=10,
                            )
                        try:
                            # PhaseD-Q 側が “obs_vec がある” と誤認しないように
                            if state_dict.get("obs_vec", None) is None and "obs_vec" in state_dict:
                                pass
                        except Exception:
                            pass
                    else:
                        try:
                            if not getattr(m, "_obs_inject_fail_once", False):
                                setattr(m, "_obs_inject_fail_once", True)
                                _k = []
                                try:
                                    _k = list(state_dict.keys())[:20]
                                except Exception:
                                    _k = []
                                print(f"[OBS_INJECT] fail mt={'Y' if _mt is not None else 'N'} p={'Y' if _p is not None else 'N'} keys={_k}")
                        except Exception:
                            pass

                _cur_obs = None
                try:
                    _cur_obs = state_dict.get("obs_vec", None)
                except Exception:
                    _cur_obs = None
                _cur_obs = _as_list_vec(_cur_obs)
                _cur_obs = _flatten_1xn(_cur_obs) if _cur_obs is not None else None
                _cur_empty = False
                try:
                    _cur_empty = (isinstance(_cur_obs, (list, tuple)) and len(_cur_obs) <= 0)
                except Exception:
                    _cur_empty = True

                if ("obs_vec" not in state_dict) or (_cur_obs is None) or _cur_empty:
                    obs_vec = None
                    m = getattr(player, "match", None) if player is not None else None

                    # ★追加: まず main_policy（AlphaZeroMCTSPolicy）側で 2448 を作れるならそれを最優先
                    obs_vec = _try_call_for_obs(
                        self.main_policy,
                        ("encode_obs_vec", "encode_obs", "build_obs_vec", "make_obs_vec", "get_obs_vec", "make_input", "build_input", "obs_vec"),
                    )

                    # ★次点: main_policy が encoder を持っていればそれも試す
                    if obs_vec is None:
                        _enc_main = None
                        try:
                            _enc_main = getattr(self.main_policy, "encoder", None)
                            if _enc_main is None:
                                _enc_main = getattr(self.main_policy, "_encoder", None)
                        except Exception:
                            _enc_main = None
                        obs_vec = _try_call_for_obs(
                            _enc_main,
                            ("encode_obs_vec", "encode_obs", "encode_for_player", "encode", "build_obs_vec", "make_obs_vec", "get_obs_vec"),
                        )

                    # ★最後: match.encoder（従来）も試す
                    if obs_vec is None:
                        enc = getattr(m, "encoder", None) if m is not None else None
                        obs_vec = _try_call_for_obs(
                            enc,
                            ("encode_obs_vec", "encode_obs", "encode_for_player", "encode"),
                        )
                        def _dbg_obs_enabled():
                            try:
                                import os
                                import sys
                                v = os.getenv("PHASED_Q_DEBUG_OBS", None)
                                if v is not None:
                                    return (str(v).strip() == "1")
                                p = ""
                                try:
                                    p = sys.argv[0] if sys.argv and sys.argv[0] else ""
                                except Exception:
                                    p = ""
                                p = p.replace("\\", "/").lower()
                                return ("ai vs ai.py" in p) or ("ai_vs_ai.py" in p) or p.endswith("/ai vs ai.py") or p.endswith("/ai_vs_ai.py")
                            except Exception:
                                return False

                        # ★追加: encode_state ベースのフォールバック（ai vs ai.py と同じ流儀）
                        if obs_vec is None:
                            fn = getattr(enc, "encode_state", None) if enc is not None else None
                            if callable(fn):
                                try:
                                    # ★追加: player が dict 等でも動くように、まず state_dict をそのまま encode_state に渡す
                                    #        （legal_actions は可能なら追加しておく）
                                    la_ids = []
                                    conv2 = None
                                    if m is not None:
                                        conv2 = getattr(m, "converter", None)
                                        if conv2 is None:
                                            conv2 = getattr(m, "action_converter", None)

                                    if conv2 is not None:
                                        fn_la = getattr(conv2, "convert_legal_actions", None)
                                        if callable(fn_la):
                                            try:
                                                la_ids = fn_la(actions or [])
                                            except Exception:
                                                la_ids = []

                                    if not la_ids:
                                        for a in actions or []:
                                            try:
                                                if hasattr(a, "to_id_vec"):
                                                    la_ids.append(a.to_id_vec())
                                                else:
                                                    la_ids.append(a if isinstance(a, list) else [int(a)])
                                            except Exception:
                                                continue

                                    feat0 = {}
                                    try:
                                        if isinstance(state_dict, dict):
                                            feat0.update(state_dict)
                                    except Exception:
                                        feat0 = {}

                                    if isinstance(la_ids, list) and la_ids:
                                        feat0["legal_actions"] = la_ids

                                    out0 = fn(feat0)
                                    try:
                                        if isinstance(out0, dict):
                                            out0 = out0.get("obs_vec", None)
                                    except Exception:
                                        pass

                                    try:
                                        import numpy as _np
                                        arr0 = _np.asarray(out0, dtype=_np.float32).reshape(-1)
                                        obs_vec = arr0.tolist()
                                    except Exception:
                                        obs_vec = out0 if isinstance(out0, list) else None

                                    obs_vec = _as_list_vec(obs_vec)
                                    obs_vec = _flatten_1xn(obs_vec) if obs_vec is not None else None
                                    if obs_vec is not None and (not _dim_ok(obs_vec)):
                                        obs_vec = None

                                    if obs_vec is None:
                                        try:
                                            import os
                                            if os.getenv("PHASED_Q_DEBUG_OBS") == "1":
                                                enc_name = getattr(enc, "__class__", type(enc)).__name__
                                                out_type = getattr(out0, "__class__", type(out0)).__name__
                                                print(f"[OBS_INJECT][WARN] encode_state produced no usable obs_vec: enc={enc_name} out_type={out_type} target_dim={_target_obs_dim}")
                                        except Exception:
                                            pass
                                except Exception:
                                    obs_vec = None

                            def _dbg_obs_print_once(_msg):
                                if not _dbg_obs_enabled():
                                    return
                                try:
                                    k = "_phased_q_obs_dbg_budget"
                                    b = getattr(self, k, None)
                                    if b is None:
                                        b = 5
                                    b = int(b)
                                    if b <= 0:
                                        return
                                    setattr(self, k, b - 1)
                                except Exception:
                                    pass
                                try:
                                    print(_msg)
                                except Exception:
                                    pass

                            if obs_vec is None and player is not None and callable(fn):
                                try:
                                    me_fn = getattr(player, "public_state", None)
                                    me = me_fn() if callable(me_fn) else None

                                    opp = None
                                    if m is not None:
                                        ps = getattr(m, "public_state", None)
                                        players = None
                                        if isinstance(ps, dict):
                                            players = ps.get("players")
                                        if not isinstance(players, list) or not players:
                                            sp = getattr(m, "starting_player", None)
                                            sp2 = getattr(m, "second_player", None)
                                            players = [x for x in (sp, sp2) if x is not None]
                                        for p0 in players or []:
                                            if p0 is not None and p0 is not player:
                                                opp_fn = getattr(p0, "public_state", None)
                                                if callable(opp_fn):
                                                    opp = opp_fn()
                                                break

                                    feat = {"me": me if isinstance(me, dict) else {}, "opp": opp if isinstance(opp, dict) else {}}
                                    if isinstance(la_ids, list) and la_ids:
                                        feat["legal_actions"] = la_ids

                                    out = fn(feat)
                                    try:
                                        if isinstance(out, dict):
                                            out = out.get("obs_vec", None)
                                    except Exception:
                                        pass

                                    try:
                                        import numpy as _np
                                        arr = _np.asarray(out, dtype=_np.float32).reshape(-1)
                                        obs_vec = arr.tolist()
                                    except Exception:
                                        obs_vec = out if isinstance(out, list) else None

                                    obs_vec = _as_list_vec(obs_vec)
                                    obs_vec = _flatten_1xn(obs_vec) if obs_vec is not None else None
                                    if obs_vec is not None and (not _dim_ok(obs_vec)):
                                        obs_vec = None
                                except Exception as _e:
                                    obs_vec = None
                                    try:
                                        enc_name = getattr(enc, "__class__", type(enc)).__name__
                                        _dbg_obs_print_once(f"[OBS_INJECT][WARN] encode_state failed: enc={enc_name} err={_e!r}")
                                    except Exception:
                                        pass

                    if obs_vec is not None:
                        _ov2, _ov_reason = self._coerce_numeric_vec(obs_vec, target_dim=_target_obs_dim)
                        if _ov2 is not None:
                            state_dict["obs_vec"] = _ov2
                            try:
                                state_dict["obs_vec_numeric"] = 1
                                state_dict["obs_vec_len"] = int(len(_ov2))
                                state_dict["obs_vec_reason"] = str(_ov_reason)
                            except Exception:
                                pass
                        else:
                            try:
                                state_dict["obs_vec"] = None
                                state_dict["obs_vec_numeric"] = 0
                                state_dict["obs_vec_reason"] = str(_ov_reason)
                            except Exception:
                                pass
                            self._dbg_budget_print(
                                "obs_bad2",
                                f"[ONLINE_MIX][OBS][WARN] final_bad reason={_ov_reason} target_dim={_target_obs_dim}",
                                default_budget=10,
                            )
                            self._dbg_unexpected_value(
                                "obs_vec",
                                obs_vec,
                                str(_ov_reason),
                                state_dict=state_dict if isinstance(state_dict, dict) else None,
                                extra={"target_dim": _target_obs_dim},
                                default_budget=10,
                            )

                # action_candidates_vec を作れそうなら埋める（converter 経由など）
                if "action_candidates_vec" not in state_dict or state_dict.get("action_candidates_vec", None) is None:
                    cand = None
                    cand_src = None
                    m = getattr(player, "match", None) if player is not None else None
                    conv = None
                    if m is not None:
                        conv = getattr(m, "converter", None)
                        if conv is None:
                            conv = getattr(m, "action_converter", None)

                    # ★正式API: 5-int を必ず作る（main を優先）
                    la_ids = None

                    def _coerce_5int_rows(_rows):
                        out = []
                        for r0 in _rows or []:
                            try:
                                r = list(r0) if isinstance(r0, tuple) else (r0[:] if isinstance(r0, list) else None)
                            except Exception:
                                r = None
                            if not isinstance(r, list):
                                continue
                            # 長さ調整（不足は0埋め、過剰は先頭5つ）
                            if len(r) < 5:
                                r = r + [0] * (5 - len(r))
                            elif len(r) > 5:
                                r = r[:5]
                            # int 強制（失敗は0）
                            rr = []
                            for x in r:
                                try:
                                    rr.append(int(x))
                                except Exception:
                                    rr.append(0)
                            out.append(rr)
                        return out

                    # 1) converter.convert_legal_actions があれば最優先（5-int を期待）
                    if conv is not None:
                        fn = getattr(conv, "convert_legal_actions", None)
                        if callable(fn):
                            try:
                                la_ids = fn(actions or [])
                            except Exception:
                                la_ids = None

                    # 2) 無ければ actions から to_id_vec を回収
                    if la_ids is None:
                        tmp = []
                        for a in actions or []:
                            try:
                                fn = getattr(a, "to_id_vec", None)
                            except Exception:
                                fn = None
                            if callable(fn):
                                try:
                                    tmp.append(fn())
                                    continue
                                except Exception:
                                    pass
                            if isinstance(a, (list, tuple)):
                                tmp.append(list(a) if isinstance(a, tuple) else a)
                                continue
                        la_ids = tmp

                    la_5 = _coerce_5int_rows(la_ids)
                    event_id = self._next_la5_event_id()
                    la_5_hash = self._hash_vecs(la_5)
                    obs_vec = None
                    try:
                        obs_vec = state_dict.get("obs_vec", None)
                    except Exception:
                        obs_vec = None
                    state_fingerprint = self._hash_vec(obs_vec) if obs_vec is not None else None

                    # ★必ず state_dict に 5-int を積む（main が受け取る正式入口）
                    try:
                        state_dict["legal_actions_5"] = la_5
                        state_dict["legal_actions_vec"] = la_5
                        state_dict["legal_actions"] = la_5
                        state_dict["la5_event_id"] = event_id
                        state_dict["legal_actions_vec_hash"] = la_5_hash
                        state_dict["trace_state_fingerprint"] = state_fingerprint
                    except Exception:
                        pass

                    if self._trace_enabled():
                        try:
                            game_id = None
                            turn = None
                            player_name = None
                            forced_active = None
                            forced_len = None
                            try:
                                m = getattr(player, "match", None) if player is not None else None
                                game_id = getattr(m, "game_id", None) if m is not None else None
                                turn = getattr(m, "turn", None) if m is not None else None
                                fa = getattr(m, "forced_actions", None) if m is not None else None
                                if isinstance(fa, (list, tuple)):
                                    forced_len = int(len(fa))
                                    forced_active = forced_len > 0
                            except Exception:
                                game_id = None
                                turn = None
                                forced_active = None
                                forced_len = None
                            try:
                                player_name = getattr(player, "name", None) if player is not None else None
                            except Exception:
                                player_name = None
                            import json

                            trace_payload = {
                                "event_id": event_id,
                                "game_id": game_id,
                                "turn": turn,
                                "player": player_name,
                                "forced_active": forced_active,
                                "forced_len": forced_len,
                                "state_fingerprint_online": state_fingerprint,
                                "state_fingerprint_env": None,
                                "n_actions": len(actions) if isinstance(actions, list) else None,
                                "n_ids": len(la_5) if isinstance(la_5, list) else None,
                                "legal_actions_vec_hash": la_5_hash,
                                "legal_actions_vec": la_5 if isinstance(la_5, list) else None,
                            }
                            print(
                                "[ONLINE_MIX][LA5][TRACE_A]"
                                f" event_id={event_id}"
                                f" game_id={game_id}"
                                f" turn={turn}"
                                f" player={player_name}"
                                f" forced_active={forced_active}"
                                f" forced_len={forced_len}"
                                f" state_fingerprint={state_fingerprint}"
                                f" n_actions={len(actions) if isinstance(actions, list) else 'NA'}"
                                f" n_ids={len(la_5) if isinstance(la_5, list) else 'NA'}"
                                f" legal_actions_vec_hash={la_5_hash}"
                                f" legal_actions_vec_full={self._format_list_head_tail(la_5, full=False)}"
                                f" trace_json={json.dumps(trace_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)}",
                                flush=True,
                            )
                        except Exception:
                            pass

                    # ★main が欲しい candidate 次元に合わせて作る（cand_dim=5 を最優先）
                    _target_cand_dim = None
                    try:
                        _target_cand_dim = getattr(self.main_policy, "cand_dim", None)
                        if _target_cand_dim is None:
                            _target_cand_dim = getattr(self.main_policy, "_cand_dim", None)
                    except Exception:
                        _target_cand_dim = None

                    cand = None
                    cand_src = None

                    if _target_cand_dim is None or int(_target_cand_dim) == 5:
                        cand = la_5
                        cand_src = "la5"
                    elif int(_target_cand_dim) == 32:
                        _emb = None
                        try:
                            _emb = globals().get("_embed_legal_actions_32d", None)
                        except Exception:
                            _emb = None
                        if callable(_emb):
                            try:
                                cand = _emb(la_5)
                                cand_src = "embed:_embed_legal_actions_32d"
                            except Exception:
                                cand = None

                    # 形・次元の最終チェック（崩れていたら載せない）
                    try:
                        if isinstance(cand, list) and len(cand) != len(actions):
                            _cand_before = cand
                            cand = None
                            self._dbg_unexpected_value(
                                "cand_vecs",
                                _cand_before,
                                f"cand_len_mismatch:{len(_cand_before)}!={len(actions)}",
                                state_dict=state_dict if isinstance(state_dict, dict) else None,
                                extra={"cand_src": cand_src, "actions": len(actions)},
                                default_budget=10,
                            )
                    except Exception:
                        pass

                    try:
                        if isinstance(cand, list) and cand:
                            v0 = cand[0]
                            if isinstance(v0, (list, tuple)):
                                if _target_cand_dim is not None and len(v0) != int(_target_cand_dim):
                                    _cand_before = cand
                                    cand = None
                                    self._dbg_unexpected_value(
                                        "cand_vecs",
                                        _cand_before,
                                        f"cand_dim_mismatch:{len(v0)}!={int(_target_cand_dim)}",
                                        state_dict=state_dict if isinstance(state_dict, dict) else None,
                                        extra={"cand_src": cand_src, "actions": len(actions), "target_dim": _target_cand_dim},
                                        default_budget=10,
                                    )
                            else:
                                _cand_before = cand
                                cand = None
                                self._dbg_unexpected_value(
                                    "cand_vecs",
                                    _cand_before,
                                    "cand_elem_not_seq",
                                    state_dict=state_dict if isinstance(state_dict, dict) else None,
                                    extra={"cand_src": cand_src, "actions": len(actions), "elem_type": type(v0).__name__},
                                    default_budget=10,
                                )
                    except Exception:
                        pass

                    if cand is not None:
                        state_dict["action_candidates_vec"] = cand
                        state_dict["action_candidates_vecs"] = cand
                        state_dict["cand_vecs"] = cand
                        try:
                            state_dict["cand_src"] = str(cand_src) if cand_src is not None else "unknown"
                            state_dict["cand_dim"] = int(len(cand[0])) if isinstance(cand[0], (list, tuple)) else -1
                        except Exception:
                            pass

                    # ★追加: actions からも id_vec を回収できるようにする（converter が無い/失敗時）
                    if la_ids is None:
                        la_ids = []
                        for a in actions or []:
                            try:
                                fn = getattr(a, "to_id_vec", None)
                            except Exception:
                                fn = None
                            if callable(fn):
                                try:
                                    la_ids.append(fn())
                                    continue
                                except Exception:
                                    pass
                            if isinstance(a, (list, tuple)):
                                la_ids.append(list(a) if isinstance(a, tuple) else a)
                                continue

                    # ★追加: 5-int（または Action 由来）を 32D に埋める
                    if cand is None and isinstance(la_ids, list) and la_ids:
                        _emb = None
                        try:
                            _emb = globals().get("_embed_legal_actions_32d", None)
                        except Exception:
                            _emb = None
                        if callable(_emb):
                            try:
                                cand = _emb(la_ids)
                                cand_src = "embed:_embed_legal_actions_32d"
                            except Exception:
                                cand = None

                    # 形・次元の最終チェック（崩れていたら載せない）
                    try:
                        if isinstance(cand, list) and len(cand) != len(actions):
                            _cand_before = cand
                            cand = None
                            self._dbg_unexpected_value(
                                "cand_vecs",
                                _cand_before,
                                f"cand_len_mismatch:{len(_cand_before)}!={len(actions)}",
                                state_dict=state_dict if isinstance(state_dict, dict) else None,
                                extra={"cand_src": cand_src, "actions": len(actions)},
                                default_budget=10,
                            )
                    except Exception:
                        pass

                    try:
                        if isinstance(cand, list) and cand:
                            v0 = cand[0]
                            if isinstance(v0, (list, tuple)) and len(v0) != 32:
                                _cand_before = cand
                                cand = None
                                self._dbg_unexpected_value(
                                    "cand_vecs",
                                    _cand_before,
                                    f"cand_dim_mismatch:{len(v0)}!=32",
                                    state_dict=state_dict if isinstance(state_dict, dict) else None,
                                    extra={"cand_src": cand_src, "actions": len(actions)},
                                    default_budget=10,
                                )
                    except Exception:
                        pass

                    if cand is not None:
                        # uniq比を計算（候補潰れ検出）
                        uniq = None
                        try:
                            sigs = set()
                            for row in cand:
                                if isinstance(row, tuple):
                                    row = list(row)
                                if isinstance(row, list):
                                    try:
                                        sigs.add(tuple(int(x) for x in row))
                                    except Exception:
                                        sigs.add(tuple(round(float(x), 6) for x in row))
                            uniq = (len(sigs), len(cand))
                        except Exception:
                            uniq = None

                        state_dict["action_candidates_vec"] = cand
                        state_dict["action_candidates_vecs"] = cand
                        state_dict["cand_vecs"] = cand
                        state_dict["cand_vecs_32d"] = cand
                        try:
                            state_dict["cand_src"] = str(cand_src) if cand_src is not None else "unknown"
                            if uniq is not None:
                                state_dict["cand_uniq"] = f"{int(uniq[0])}/{int(uniq[1])}"
                                state_dict["cand_uniq_ratio"] = (float(uniq[0]) / float(uniq[1])) if uniq[1] else 0.0
                        except Exception:
                            pass

                        if la_5:
                            state_dict.setdefault("legal_actions_vec", la_5)
                            state_dict.setdefault("legal_actions_5", la_5)

                        # 代表ログ（数回だけ）
                        if uniq is not None:
                            self._dbg_budget_print(
                                "cand",
                                f"[ONLINE_MIX][CAND] ok n={len(cand)} dim={(len(cand[0]) if isinstance(cand[0], (list, tuple)) else -1)} uniq={uniq[0]}/{uniq[1]} src={cand_src}",
                                default_budget=8,
                            )
                        else:
                            self._dbg_budget_print(
                                "cand2",
                                f"[ONLINE_MIX][CAND][WARN] ok_but_no_uniq n={len(cand)} src={cand_src}",
                                default_budget=6,
                            )
        except Exception:
            pass

        import random

        rnd = self.rng.random() if self.rng is not None else random.random()

        # ★追加: 手ごとの決定ソース/idx を state_dict と last_decide_info に確実に残す（printはここではしない）
        def _set_online_mix_trace(_source, _idx, _why=None):
            try:
                self.last_source = str(_source)
            except Exception:
                pass

            # ★追加: PhaseD-Q が必ず π を拾えるように、state_dict["pi"] を常に用意する
            try:
                if isinstance(state_dict, dict):
                    n = int(len(actions)) if actions is not None else 0
                    if n > 0:
                        pi0 = None
                        try:
                            pi0 = (
                                state_dict.get("pi", None)
                                or state_dict.get("mcts_pi", None)
                                or state_dict.get("pi_mcts", None)
                                or state_dict.get("policy_pi", None)
                            )
                        except Exception:
                            pi0 = None

                        pi_norm = None
                        try:
                            pi_norm, _ = self._normalize_pi_to_len(pi0, n)
                        except Exception:
                            pi_norm = None

                        if pi_norm is None:
                            # 最低保証: one-hot（これで uniform fallback を避けられる）
                            try:
                                pi_norm = [0.0] * int(n)
                                pi_norm[int(_idx)] = 1.0
                            except Exception:
                                pi_norm = None

                        if pi_norm is not None:
                            try:
                                state_dict["pi"] = pi_norm
                                # main のときは mcts_pi としても残す（外側が mcts_pi を見に行く実装差対策）
                                if str(_source) == "main":
                                    state_dict["mcts_pi"] = pi_norm
                                    state_dict["policy_pi"] = pi_norm
                                    state_dict["mcts_pi_present"] = 1
                                    state_dict["mcts_pi_len"] = int(len(pi_norm))
                                    state_dict["mcts_pi_type"] = type(pi_norm).__name__
                                else:
                                    # fallback/random では「MCTSπではない」ことを明示
                                    state_dict["mcts_pi_present"] = 0
                                    state_dict["mcts_pi_len"] = int(len(pi_norm))
                            except Exception:
                                pass

                            # outer が attr を見に行く実装もあるので self 側にも残す
                            try:
                                self.last_pi = pi_norm
                                self.pi = pi_norm
                            except Exception:
                                pass
            except Exception:
                pass

            try:
                self.last_decide_info = {
                    "mode": "online_mix",
                    "source": str(_source),
                    "selected_idx": int(_idx),
                    "rnd": float(rnd),
                    "mix_prob": float(self.mix_prob),
                    "p": float(self.mix_prob),
                    "why": (str(_why) if _why is not None else None),
                }
                if isinstance(state_dict, dict):
                    try:
                        self.last_decide_info["obs_len"] = state_dict.get("obs_vec_len", None)
                        self.last_decide_info["obs_ok"] = state_dict.get("obs_vec_numeric", None)
                        self.last_decide_info["cand_uniq"] = state_dict.get("cand_uniq", None)
                        self.last_decide_info["cand_src"] = state_dict.get("cand_src", None)
                        self.last_decide_info["mcts_pi_present"] = state_dict.get("mcts_pi_present", None)
                        self.last_decide_info["mcts_pi_len"] = state_dict.get("mcts_pi_len", None)
                    except Exception:
                        pass
                self._last_decide_info = self.last_decide_info
            except Exception:
                pass

            try:
                if str(_source) == "main":
                    self._sum_used_main = int(getattr(self, "_sum_used_main", 0)) + 1
                elif str(_source) == "fallback":
                    self._sum_used_fallback = int(getattr(self, "_sum_used_fallback", 0)) + 1
                elif str(_source) == "random":
                    self._sum_used_random = int(getattr(self, "_sum_used_random", 0)) + 1
            except Exception:
                pass

            if not isinstance(state_dict, dict):
                return
            try:
                state_dict["online_mix_source"] = str(_source)
                state_dict["online_mix_selected_idx"] = int(_idx)
                state_dict["online_mix_rnd"] = float(rnd)
                state_dict["online_mix_p"] = float(self.mix_prob)
                if _why is not None:
                    state_dict["online_mix_why"] = str(_why)
            except Exception:
                pass

            if self._trace_enabled():
                try:
                    event_id = None
                    generated_new = 0
                    state_fingerprint = None
                    state_fingerprint_env = None
                    la_5_hash = None
                    la_5 = None
                    try:
                        event_id = state_dict.get("la5_event_id", None)
                        state_fingerprint = state_dict.get("trace_state_fingerprint", None)
                        state_fingerprint_env = state_dict.get("env_state_fingerprint", None)
                        la_5_hash = state_dict.get("legal_actions_vec_hash", None)
                        la_5 = state_dict.get("legal_actions_5", None)
                    except Exception:
                        event_id = None
                        state_fingerprint = None
                        state_fingerprint_env = None
                        la_5_hash = None
                        la_5 = None
                    if event_id is None:
                        event_id = self._next_la5_event_id()
                        generated_new = 1
                        try:
                            state_dict["la5_event_id"] = event_id
                        except Exception:
                            pass
                    selected_vec = None
                    try:
                        if isinstance(la_5, list) and 0 <= int(_idx) < len(la_5):
                            selected_vec = la_5[int(_idx)]
                    except Exception:
                        selected_vec = None
                    import json

                    trace_payload = {
                        "event_id": event_id,
                        "generated_new": int(generated_new),
                        "selection_source": _source,
                        "selected_idx": int(_idx),
                        "selected_vec": selected_vec,
                        "state_fingerprint_online": state_fingerprint if state_fingerprint is not None else "NA",
                        "state_fingerprint_env": state_fingerprint_env if state_fingerprint_env is not None else "NA",
                        "legal_actions_vec_hash": la_5_hash,
                    }
                    print(
                        "[ONLINE_MIX][TRACE_B]"
                        f" event_id={event_id}"
                        f" generated_new={generated_new}"
                        f" selection_source={_source}"
                        f" selected_idx={int(_idx)}"
                        f" selected_vec={self._format_list_head_tail(selected_vec, full=True)}"
                        f" state_fingerprint_online={state_fingerprint if state_fingerprint is not None else 'NA'}"
                        f" state_fingerprint_env={state_fingerprint_env if state_fingerprint_env is not None else 'NA'}"
                        f" legal_actions_vec_hash={la_5_hash}"
                        f" trace_json={json.dumps(trace_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)}",
                        flush=True,
                    )
                except Exception:
                    pass

            if str(_source) == "random" and _why is not None:
                try:
                    game_id = None
                    turn = None
                    player_name = None
                    forced_active = None
                    forced_len = None
                    try:
                        m = getattr(player, "match", None) if player is not None else None
                        game_id = getattr(m, "game_id", None) if m is not None else None
                        turn = getattr(m, "turn", None) if m is not None else None
                        fa = getattr(m, "forced_actions", None) if m is not None else None
                        if isinstance(fa, (list, tuple)):
                            forced_len = int(len(fa))
                            forced_active = forced_len > 0
                    except Exception:
                        game_id = None
                        turn = None
                        forced_active = None
                        forced_len = None
                    try:
                        player_name = getattr(player, "name", None) if player is not None else None
                    except Exception:
                        player_name = None
                    event_id = None
                    try:
                        if isinstance(state_dict, dict):
                            event_id = state_dict.get("last_step_no_match_event_id", None)
                    except Exception:
                        event_id = None
                    if event_id is None:
                        try:
                            event_id = getattr(m, "_last_step_no_match_event_id", None) if m is not None else None
                        except Exception:
                            event_id = None
                    print(
                        "[ONLINE_MIX][RANDOM_FALLBACK]"
                        f" game_id={game_id}"
                        f" turn={turn}"
                        f" player={player_name}"
                        f" forced_active={forced_active}"
                        f" forced_len={forced_len}"
                        f" n_actions={len(actions) if isinstance(actions, list) else 'NA'}"
                        f" reason={_why}"
                        f" event_id={event_id}",
                        flush=True,
                    )
                except Exception:
                    pass

            # ログは環境変数でON/OFF（デフォルトOFF：Player側で出す想定）
            try:
                import os
                if str(os.getenv("ONLINE_MIX_TRACE", "0")).strip() != "1":
                    return
            except Exception:
                return

            _t = None
            _pl = None
            try:
                _t = state_dict.get("t", None) or state_dict.get("turn", None) or state_dict.get("turn_i", None) or state_dict.get("step", None) or state_dict.get("ply", None)
            except Exception:
                _t = None
            try:
                _pl = state_dict.get("player_name", None) or state_dict.get("player", None)
            except Exception:
                _pl = None

            _mpp = None
            _mpl = None
            _mpt = None
            try:
                _mpp = state_dict.get("mcts_pi_present", None)
                _mpl = state_dict.get("mcts_pi_len", None)
                _mpt = state_dict.get("mcts_pi_type", None)
            except Exception:
                _mpp = None
                _mpl = None
                _mpt = None

            _obs_ok = None
            _obs_len = None
            _obs_reason = None
            _cand_uniq = None
            _cand_src = None
            try:
                _obs_ok = state_dict.get("obs_vec_numeric", None)
                _obs_len = state_dict.get("obs_vec_len", None)
                _obs_reason = state_dict.get("obs_vec_reason", None)
                _cand_uniq = state_dict.get("cand_uniq", None)
                _cand_src = state_dict.get("cand_src", None)
            except Exception:
                _obs_ok = None
                _obs_len = None
                _obs_reason = None
                _cand_uniq = None
                _cand_src = None

            try:
                deck_rem = state_dict.get("deck_remaining", None) or state_dict.get("deck_remain", None) or state_dict.get("deck_left", None) or state_dict.get("deck_count", None) or state_dict.get("deck_len", None)
            except Exception:
                deck_rem = None
            try:
                prize_rem = state_dict.get("prize_remaining", None) or state_dict.get("prize_left", None) or state_dict.get("prizes_left", None) or state_dict.get("prize_count", None) or state_dict.get("prize_len", None)
            except Exception:
                prize_rem = None

            try:
                la_len = int(len(actions)) if actions is not None else -1
            except Exception:
                la_len = -1

            print(
                f"[ONLINE_MIX][STEP] t={_t if _t is not None else 'NA'} player={_pl if _pl is not None else 'NA'}"
                f" source={_source} idx={int(_idx)} la_len={la_len}"
                f" rnd={float(rnd):.6f} p={float(self.mix_prob):.6f}"
                f" mcts_pi_present={_mpp} mcts_pi_len={_mpl} mcts_pi_type={_mpt}"
                f" obs_ok={_obs_ok} obs_len={_obs_len} obs_reason={_obs_reason}"
                f" cand_uniq={_cand_uniq} cand_src={_cand_src}"
                f" deck={deck_rem} prize={prize_rem}"
                f"{(' why=' + str(_why)) if _why is not None else ''}"
                ,
                flush=True,
            )

        # --- 1) main / fallback のどちらも設定されていない → 完全ランダム ---
        if self.main_policy is None and self.fallback_policy is None:
            self._record_source("random")
            self._maybe_raise_if_main_never_used()
            _idx = random.randint(0, len(actions) - 1)
            _set_online_mix_trace("random", _idx, _why="no_main_no_fallback")
            return _idx

        # --- 2) mix_prob のクリップ ---
        p = self.mix_prob
        if p <= 0.0:
            # 0.0 以下 → ひたすら fallback (無ければ main → それも無ければランダム)
            idx = self._select_from_policy(self.fallback_policy, state_dict, actions, player, "fallback")
            if idx is not None:
                self._fallback_calls += 1
                self._record_source("fallback")
                _set_online_mix_trace("fallback", idx)
                return idx
            idx = self._select_from_policy(self.main_policy, state_dict, actions, player, "main")
            if idx is not None:
                self._main_success_calls += 1
                self._record_source("model")
                _set_online_mix_trace("main", idx)
                return idx
            self._main_failed_calls += 1
            self._record_source("random")
            self._maybe_raise_if_main_never_used()
            _idx = random.randint(0, len(actions) - 1)
            _set_online_mix_trace("random", _idx, _why="p<=0_both_failed")
            return _idx
        if p >= 1.0:
            # 1.0 以上 → ひたすら main (無ければ fallback → それも無ければランダム)
            idx = self._select_from_policy(self.main_policy, state_dict, actions, player, "main")
            if idx is not None:
                self._main_success_calls += 1
                self._record_source("model")
                _set_online_mix_trace("main", idx)
                return idx
            self._main_failed_calls += 1
            idx = self._select_from_policy(self.fallback_policy, state_dict, actions, player, "fallback")
            if idx is not None:
                self._fallback_calls += 1
                self._record_source("fallback")
                _set_online_mix_trace("fallback", idx)
                return idx
            self._record_source("random")
            self._maybe_raise_if_main_never_used()
            _idx = random.randint(0, len(actions) - 1)
            _set_online_mix_trace("random", _idx, _why="p>=1_both_failed")
            return _idx

        # --- 3) 0 < p < 1 の通常ケース：オンライン混合 ---
        # rnd < p なら main → fallback、そうでなければ fallback → main の順にトライ
        if rnd < p:
            # 先に main を呼ぶ
            idx = self._select_from_policy(self.main_policy, state_dict, actions, player, "main")
            if idx is not None:
                self._main_success_calls += 1
                self._record_source("model")
                _set_online_mix_trace("main", idx)
                return idx
            self._main_failed_calls += 1
            idx = self._select_from_policy(self.fallback_policy, state_dict, actions, player, "fallback")
            if idx is not None:
                self._fallback_calls += 1
                self._record_source("fallback")
                _set_online_mix_trace("fallback", idx)
                return idx
        else:
            # 先に fallback を呼ぶ
            idx = self._select_from_policy(self.fallback_policy, state_dict, actions, player, "fallback")
            if idx is not None:
                self._fallback_calls += 1
                self._record_source("fallback")
                _set_online_mix_trace("fallback", idx)
                return idx
            idx = self._select_from_policy(self.main_policy, state_dict, actions, player, "main")
            if idx is not None:
                self._main_success_calls += 1
                self._record_source("model")
                _set_online_mix_trace("main", idx)
                return idx
            self._main_failed_calls += 1

        # どちらも失敗したときの最終フォールバック
        self._record_source("random")
        self._maybe_raise_if_main_never_used()
        _idx = random.randint(0, len(actions) - 1)
        _set_online_mix_trace("random", _idx, _why="both_failed")
        return _idx


    def log_summary(self, player=None, match=None, reason=None):
        out = None
        try:
            if match is not None and hasattr(match, "log_print"):
                out = match.log_print
        except Exception:
            out = None
        if out is None:
            out = print

        total = int(getattr(self, "_sum_total", 0))
        main_n = int(getattr(self, "_sum_used_main", 0))
        fb_n = int(getattr(self, "_sum_used_fallback", 0))
        rnd_n = int(getattr(self, "_sum_used_random", 0))

        main_ok = int(getattr(self, "_main_success_calls", 0))
        main_fail = int(getattr(self, "_main_failed_calls", 0))
        fb_calls = int(getattr(self, "_fallback_calls", 0))
        mp = float(getattr(self, "mix_prob", 0.0))

        out(
            f"[SUMMARY] policy=online_mix total={total} main={main_n} fallback={fb_n} random={rnd_n} "
            f"main_ok={main_ok} main_fail={main_fail} fallback_calls={fb_calls} mix_prob={mp} reason={reason}"
        )

        # 下位ポリシー側にも summary があれば呼ぶ（任意）
        for pol in (getattr(self, "main_policy", None), getattr(self, "fallback_policy", None)):
            if pol is None:
                continue
            for fn_name in ("log_summary", "print_summary", "dump_summary", "summary"):
                fn = getattr(pol, fn_name, None)
                if callable(fn):
                    try:
                        fn(player=player, match=match, reason=reason)
                    except TypeError:
                        try:
                            fn(player, match)
                        except TypeError:
                            fn()
                    break
