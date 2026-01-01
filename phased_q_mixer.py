#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhaseD-Q (CQL) evaluation + online π/Q mixing wrapper extracted from ai vs ai.py.

This file is auto-extracted to slim down the main runner script.
"""

import os, json, random
import struct, zlib
import numpy as np
import d3rlpy

# Phase D CQL Q(s,a) ローダ＆評価ヘルパ
# ============================================================

# Phase D CQL の成果物パス
PHASED_Q_LEARNABLE_PATH = r"D:\date\phaseD_cql_run_p1\learnable_phaseD_cql.d3"
PHASED_Q_META_PATH      = r"D:\date\phaseD_cql_run_p1\train_phaseD_cql_meta.json"

# Phase D Q を ai vs ai で使うかどうか
USE_PHASED_Q: bool = True

# 内部キャッシュ
_PHASED_Q_ALGO = None
_PHASED_Q_OBS_DIM: int | None = None
_PHASED_Q_ACTION_DIM: int | None = None
_PHASED_Q_LAST_FULL_OBS_VEC: list[float] | None = None

# ★ 追加: Phase D Q をオンラインで π にミックスするかどうか
PHASED_Q_MIX_ENABLED: bool = bool(int(os.getenv("PHASED_Q_MIX_ENABLED", "1")))
# ★ 追加: π と Q のブレンド率（0.0=純粋π, 1.0=純粋Q）
PHASED_Q_MIX_LAMBDA: float = float(os.getenv("PHASED_Q_MIX_LAMBDA", "0.30"))
# ★ 追加: Q 側の softmax 温度（大きいほど分布がなだらかになる）
PHASED_Q_MIX_TEMPERATURE: float = float(os.getenv("PHASED_Q_MIX_TEMPERATURE", "1.0"))

# ★ 追加: Q が候補間でほぼ同一かどうかの判定閾値（q_span < eps を “flat” とみなす）
PHASED_Q_Q_FLAT_EPS: float = float(os.getenv("PHASED_Q_Q_FLAT_EPS", "1e-6"))
# ★ 追加: “終盤”判定（自分の山札残枚数 <= threshold）
PHASED_Q_LATE_DECK_TH: int = int(os.getenv("PHASED_Q_LATE_DECK_TH", "10"))
# ★ 追加: mix_changed の詳細ログを出すか（デフォルトは終盤だけ）
PHASED_Q_LOG_MIX_CHANGED: bool = bool(int(os.getenv("PHASED_Q_LOG_MIX_CHANGED", "1")))
PHASED_Q_LOG_MIX_CHANGED_ALL: bool = bool(int(os.getenv("PHASED_Q_LOG_MIX_CHANGED_ALL", "0")))
# ★ 追加: action 表示を詰める最大文字数（改行は除去）
PHASED_Q_ACTION_STR_MAX: int = int(os.getenv("PHASED_Q_ACTION_STR_MAX", "160"))

# ★ 追加: どこが壊れているか切り分ける DIAG ログ
PHASED_Q_DIAG_LOG: bool = bool(int(os.getenv("PHASED_Q_DIAG_LOG", "1")))
PHASED_Q_DIAG_EVERY: int = int(os.getenv("PHASED_Q_DIAG_EVERY", "200"))  # 0 なら周期ログ無効
PHASED_Q_DIAG_ON_FLAT: bool = bool(int(os.getenv("PHASED_Q_DIAG_ON_FLAT", "1")))
PHASED_Q_DIAG_ON_LOW_UNIQ: bool = bool(int(os.getenv("PHASED_Q_DIAG_ON_LOW_UNIQ", "1")))
PHASED_Q_DIAG_LOW_UNIQ_TH: float = float(os.getenv("PHASED_Q_DIAG_LOW_UNIQ_TH", "0.20"))
PHASED_Q_DIAG_ROWS: int = int(os.getenv("PHASED_Q_DIAG_ROWS", "3"))

PHASED_Q_DECIDE_LOG: bool = bool(int(os.getenv("PHASED_Q_DECIDE_LOG", "1")))
PHASED_Q_DECIDE_LOG_DIFF_ONLY: bool = bool(int(os.getenv("PHASED_Q_DECIDE_LOG_DIFF_ONLY", "0")))


def phaseD_q_load_if_needed() -> None:
    """
    Phase D CQL の learnable .d3 を lazy にロードし、グローバルにキャッシュする。
    ロードに失敗した場合は _PHASED_Q_ALGO は None のまま。
    """
    global _PHASED_Q_ALGO, _PHASED_Q_OBS_DIM, _PHASED_Q_ACTION_DIM

    if _PHASED_Q_ALGO is not None:
        return
    if not USE_PHASED_Q:
        return

    if not os.path.exists(PHASED_Q_LEARNABLE_PATH):
        print(f"[PhaseD Q] learnable not found: {PHASED_Q_LEARNABLE_PATH}", flush=True)
        return

    try:
        # d3rlpy v2 系では save_model(...) と対になるのは load_learnable(...)
        learnable = d3rlpy.load_learnable(PHASED_Q_LEARNABLE_PATH)

        # Learnable から CQL アルゴリズムを復元する
        try:
            from d3rlpy.algos import DiscreteCQL  # あなたが使っている CQL クラスに合わせて変更
            algo = DiscreteCQL.from_learnable(learnable)
        except Exception:
            # うまく復元できなかった場合はいったん learnable をそのまま保持
            algo = learnable
    except Exception as e:
        print(f"[PhaseD Q] failed to load .d3: {PHASED_Q_LEARNABLE_PATH} err={e!r}", flush=True)
        return

    _PHASED_Q_ALGO = algo

    # メタから obs_dim / action_dim を拾っておく（チェック用）
    try:
        with open(PHASED_Q_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        _PHASED_Q_OBS_DIM = int(meta.get("obs_dim", 0))
        _PHASED_Q_ACTION_DIM = int(meta.get("action_dim", 0))
    except Exception:
        _PHASED_Q_OBS_DIM = None
        _PHASED_Q_ACTION_DIM = None

    _pv = False
    try:
        _pv = callable(getattr(_PHASED_Q_ALGO, "predict_value", None))
    except Exception:
        _pv = False

    print(
        f"[PhaseD Q] loaded CQL learnable from {PHASED_Q_LEARNABLE_PATH} "
        f"(obs_dim={_PHASED_Q_OBS_DIM} action_dim={_PHASED_Q_ACTION_DIM} "
        f"algo={type(_PHASED_Q_ALGO).__name__} predict_value={1 if _pv else 0})",
        flush=True,
    )

def phaseD_q_evaluate(
    obs_vec: list[float] | np.ndarray,
    action_candidates_vec: list[list[float]] | np.ndarray,
) -> np.ndarray | None:
    """
    1 状態 obs_vec と、その状態での action_candidates_vec (K x action_dim) を受け取り、
      Q(s, a_k) の配列 (K,) を返す。

    失敗時や USE_PHASED_Q=False のときは None を返す。
    """
    if not USE_PHASED_Q:
        return None

    phaseD_q_load_if_needed()
    if _PHASED_Q_ALGO is None:
        return None

    obs = np.asarray(obs_vec, dtype=np.float32)
    cands = np.asarray(action_candidates_vec, dtype=np.float32)

    if obs.ndim == 1:
        # shape: (obs_dim,) -> (1, obs_dim)
        obs = obs[None, :]

    if obs.ndim != 2:
        print(f"[PhaseD Q] invalid obs shape: {obs.shape}")
        return None

    if cands.ndim != 2:
        print(f"[PhaseD Q] invalid cands shape: {cands.shape}")
        return None

    # 事前に記録しておいた obs_dim / action_dim と食い違う場合はあえて使わない
    if _PHASED_Q_OBS_DIM is not None and obs.shape[1] != _PHASED_Q_OBS_DIM:
        print(
            f"[PhaseD Q] obs_dim mismatch: got={obs.shape[1]} "
            f"expected={_PHASED_Q_OBS_DIM}"
        )

        allow_pad = bool(int(os.getenv("PHASED_Q_ALLOW_OBS_PAD", "0")))
        if not allow_pad:
            return None

        # ★暫定: mismatch をパディング/切り詰めで通す（デバッグ用）
        try:
            exp = int(_PHASED_Q_OBS_DIM)
            got = int(obs.shape[1])
            if got < exp:
                pad = np.zeros((obs.shape[0], exp - got), dtype=obs.dtype)
                obs = np.concatenate([obs, pad], axis=1)
            else:
                obs = obs[:, :exp]
            print(f"[PhaseD Q] ⚠️ obs adjusted for debug: {got} -> {obs.shape[1]}")
        except Exception:
            return None

    if _PHASED_Q_ACTION_DIM is not None and cands.shape[1] != _PHASED_Q_ACTION_DIM:
        print(
            f"[PhaseD Q] action_dim mismatch: got={cands.shape[1]} "
            f"expected={_PHASED_Q_ACTION_DIM}"
        )

        allow_pad = bool(int(os.getenv("PHASED_Q_ALLOW_ACTION_PAD", "0")))
        if not allow_pad:
            return None

        # ★暫定: mismatch をパディング/切り詰めで通す（デバッグ用）
        try:
            exp = int(_PHASED_Q_ACTION_DIM)
            got = int(cands.shape[1])
            if got < exp:
                pad = np.zeros((cands.shape[0], exp - got), dtype=cands.dtype)
                cands = np.concatenate([cands, pad], axis=1)
            else:
                cands = cands[:, :exp]
            print(f"[PhaseD Q] ⚠️ action adjusted for debug: {got} -> {cands.shape[1]}")
        except Exception:
            return None

    # obs を候補数 K にブロードキャストして (K, obs_dim) にする
    num_actions = int(cands.shape[0])
    if num_actions <= 0:
        return None
    obs_batch = np.repeat(obs, num_actions, axis=0)

    try:
        _pv = getattr(_PHASED_Q_ALGO, "predict_value", None)
    except Exception:
        _pv = None

    if not callable(_pv):
        if not bool(getattr(_PHASED_Q_ALGO, "_phased_q_no_predict_logged", False)):
            print(f"[PhaseD Q] algo has no predict_value: {type(_PHASED_Q_ALGO).__name__}", flush=True)
            try:
                setattr(_PHASED_Q_ALGO, "_phased_q_no_predict_logged", True)
            except Exception:
                pass
        return None

    # d3rlpy の CQL は predict_value(x, action) で Q(s,a) を返す
    try:
        values = _pv(obs_batch, cands)
    except Exception as e:
        print(f"[PhaseD Q] predict_value failed: {e!r}")
        return None

    # values は shape (K,) を想定
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] != num_actions:
        print(
            f"[PhaseD Q] unexpected values shape: {values.shape}, "
            f"num_actions={num_actions}"
        )
        return None

    return values

def phaseD_mix_pi_with_q(pi, q_values):
    """
    Phase D の Q 値と元の π をブレンドして新しい方策分布を返す。

    - pi, q_values は同じ長さの 1 次元配列を想定
    - 何かおかしければ安全側として None を返す（呼び出し側で base_pi にフォールバック）
    """
    try:
        # numpy 配列に変換（必ず 1 次元へ）
        pi_arr = np.asarray(pi, dtype=np.float64).reshape(-1)
        q_arr = np.asarray(q_values, dtype=np.float64).reshape(-1)

        # 次元一致 & 空チェック
        if pi_arr.size <= 0 or q_arr.size <= 0:
            return None
        if pi_arr.size != q_arr.size:
            return None

        # 有限値チェック（NaN/inf が混じると softmax が壊れる）
        if not np.all(np.isfinite(pi_arr)) or not np.all(np.isfinite(q_arr)):
            return None

        # π を安全に正規化（負値は 0 扱い、総和 0 なら一様）
        pi_arr = np.maximum(pi_arr, 0.0)
        pi_sum = float(pi_arr.sum())
        if not np.isfinite(pi_sum) or pi_sum <= 0.0:
            pi_arr = np.ones_like(pi_arr, dtype=np.float64) / float(pi_arr.size)
        else:
            pi_arr = pi_arr / pi_sum

        # Q から softmax 分布を作る（温度付き）
        tau = max(PHASED_Q_MIX_TEMPERATURE, 1e-6)
        # 数値安定化のため最大値を引いてから softmax
        q_shift = q_arr - np.max(q_arr)
        q_soft = np.exp(q_shift / tau)
        s = float(q_soft.sum())
        if not np.isfinite(s) or s <= 0.0:
            return None
        q_soft = q_soft / s

        # λ で π と Q-softmax を線形補間
        lam = min(max(PHASED_Q_MIX_LAMBDA, 0.0), 1.0)
        mixed = (1.0 - lam) * pi_arr + lam * q_soft

        # 念のためもう一度正規化
        msum = float(mixed.sum())
        if not np.isfinite(msum) or msum <= 0.0:
            return None
        mixed = mixed / msum

        try:
            try:
                _dbg = bool(globals().get("LOG_DEBUG_DETAIL", False)) or bool(int(os.getenv("PHASED_Q_MIX_DEBUG", "0")))
            except Exception:
                _dbg = bool(globals().get("LOG_DEBUG_DETAIL", False))
            if _dbg:
                _k = int(min(3, mixed.size))
                if _k > 0:
                    _mix_idx = list(np.argsort(mixed)[- _k:][::-1])
                    _pi_idx  = list(np.argsort(pi_arr)[- _k:][::-1])
                    _q_idx   = list(np.argsort(q_soft)[- _k:][::-1])
                    _mix_top = ",".join([f"{int(i)}:{float(mixed[int(i)]):.6f}" for i in _mix_idx])
                    _pi_top  = ",".join([f"{int(i)}:{float(pi_arr[int(i)]):.6f}" for i in _pi_idx])
                    _q_top   = ",".join([f"{int(i)}:{float(q_soft[int(i)]):.6f}" for i in _q_idx])
                    _chosen = int(np.argmax(mixed))
                    _chosen_p = float(mixed[_chosen]) if 0 <= _chosen < int(mixed.size) else float("nan")
                    print(f"[PhaseD-Q][MIX_CORE] lam={float(lam):.6f} tau={float(tau):.6f} n_cand={int(mixed.size)} "
                          f"pi_top={_pi_top} q_top={_q_top} mix_top={_mix_top} chosen={_chosen}:{_chosen_p:.6f}",
                          flush=True)
        except Exception:
            pass

        return mixed.astype(np.float32).tolist()
    except Exception:
        # 例外が出た場合は None を返して安全側に倒す
        return None


def _normalize_base_pi(pi, n):
    if isinstance(pi, dict) and "pi" in pi:
        pi = pi["pi"]

    try:
        _tl = getattr(pi, "tolist", None)
    except Exception:
        _tl = None
    if callable(_tl):
        try:
            pi = _tl()
        except Exception:
            pi = None

    if isinstance(pi, (list, tuple)):
        out = []
        for x in list(pi)[:n]:
            try:
                out.append(float(x))
            except Exception:
                out.append(0.0)
        if len(out) < n:
            out += [0.0] * (n - len(out))
        return out

    if isinstance(pi, dict):
        out = [0.0] * n
        for k, v in pi.items():
            if isinstance(k, int) and 0 <= k < n:
                try:
                    out[k] = float(v)
                except Exception:
                    pass
        return out

    return None


def _topk_pairs(arr, k=3):
    try:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
    except Exception:
        return "NA"
    try:
        kk = int(min(int(k), int(a.size)))
    except Exception:
        kk = 3
    if kk <= 0 or int(a.size) <= 0:
        return "NA"
    try:
        idx = list(np.argsort(a)[-kk:][::-1])
        return ",".join([f"{int(i)}:{float(a[int(i)]):.6f}" for i in idx])
    except Exception:
        return "NA"


def wrap_select_action_with_phased_q(pol, tag="phased", extra_roots=None):
    """
    pol の select_action_* を PhaseD-Q でラップして差し替える。

    - 返り値: pol（破壊的にメソッドを差し替える）
    """

    def _as_list(x):
        if x is None:
            return None
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        try:
            tl = getattr(x, "tolist", None)
        except Exception:
            tl = None
        if callable(tl):
            try:
                y = tl()
            except Exception:
                y = None
            if isinstance(y, list):
                return y
            if isinstance(y, tuple):
                return list(y)
        return None

    def _is_numeric_vec(v):
        if not isinstance(v, (list, tuple)) or len(v) <= 0:
            return False
        try:
            for x in v:
                float(x)
            return True
        except Exception:
            return False

    def _normalize_cand_vecs_32d(cand, n_expect):
        if cand is None:
            return None

        # dict {idx: vec} / {any: vec} を rows(list) に落とす
        if isinstance(cand, dict):
            try:
                _keys = list(cand.keys())
                if _keys and all(isinstance(k, int) for k in _keys):
                    cand = [cand[k] for k in sorted(_keys)]
                else:
                    cand = list(cand.values())
            except Exception:
                return None

        if not isinstance(cand, (list, tuple)):
            try:
                _tl = getattr(cand, "tolist", None)
            except Exception:
                _tl = None
            if callable(_tl):
                try:
                    cand = _tl()
                except Exception:
                    cand = None

        if not isinstance(cand, (list, tuple)) or not cand:
            return None
        try:
            if len(cand) != n_expect:
                return None
        except Exception:
            return None

        out = []
        for row in cand:
            if isinstance(row, dict):
                try:
                    _rk = list(row.keys())
                    if _rk and all(isinstance(k, int) for k in _rk):
                        row = [row[k] for k in sorted(_rk)]
                    else:
                        row = list(row.values())
                except Exception:
                    return None

            if not isinstance(row, (list, tuple)):
                try:
                    _tl2 = getattr(row, "tolist", None)
                except Exception:
                    _tl2 = None
                if callable(_tl2):
                    try:
                        row = _tl2()
                    except Exception:
                        row = None

            if isinstance(row, (list, tuple)):
                if len(row) == 32:
                    try:
                        out.append([float(x) for x in row])
                    except Exception:
                        return None
                    continue
                if len(row) == 17:
                    try:
                        rr = [float(x) for x in row] + [0.0] * (32 - 17)
                    except Exception:
                        return None
                    out.append(rr)
                    continue
                return None
            return None
        return out

    def _make_cand_vecs_32d(la_list, kwargs, ep_name=None):
        def _la_sig(a):
            try:
                try:
                    if isinstance(a, np.ndarray):
                        return f"ndarray(shape={tuple(a.shape)} dtype={getattr(a, 'dtype', None)})"
                except Exception:
                    pass
                if isinstance(a, (list, tuple)):
                    return f"seq(len={len(a)} head={list(a)[:6]})"
                if isinstance(a, dict):
                    try:
                        _ks_all = list(a.keys())
                        ks = sorted(_ks_all, key=lambda x: (str(type(x)), str(x)))[:8]
                    except Exception:
                        ks = list(a.keys())[:8]
                    ss = []
                    for k in ks[:4]:
                        try:
                            ss.append(f"{k}={str(a.get(k))[:24]}")
                        except Exception:
                            ss.append(f"{k}=<?>")
                    return f"dict(keys={ks} samp={','.join(ss)})"
                for _attr in ("action_type", "type", "kind", "id", "card_id", "src", "dst", "target", "index"):
                    try:
                        if hasattr(a, _attr):
                            v = getattr(a, _attr)
                            return f"{type(a).__name__}.{_attr}={str(v)[:48]}"
                    except Exception:
                        continue
                return f"{type(a).__name__}:{str(a)[:80]}"
            except Exception:
                return "<??>"

        def _cand_stats(cand):
            try:
                n = int(len(cand))
                if n <= 0:
                    return 0, 0, 0.0
                uniq = len({tuple(round(float(x), 6) for x in row) for row in cand})
                r = float(uniq) / float(n) if n > 0 else 0.0
                return n, int(uniq), float(r)
            except Exception:
                return 0, 0, 0.0

        def _crc32_row32(row):
            try:
                xs = [float(x) for x in list(row)[:32]]
                b = struct.pack("<%sf" % len(xs), *xs)
                return f"{(zlib.crc32(b) & 0xffffffff):08x}"
            except Exception:
                return "--------"

        def _row_minmax32(row):
            try:
                xs = [float(x) for x in list(row)[:32]]
                return (float(min(xs)), float(max(xs)))
            except Exception:
                return (0.0, 0.0)

        def _maybe_log_probe(msg):
            try:
                _c = int(getattr(pol, "_phased_q_cand_probe_count", 0))
            except Exception:
                _c = 0
            if _c >= 20:
                return
            try:
                setattr(pol, "_phased_q_cand_probe_count", _c + 1)
            except Exception:
                pass

            try:
                _take = min(3, int(len(la_list)))
                _sigs = [_la_sig(la_list[i]) for i in range(_take)]
                _sig_uniq = len(set(_sigs))
            except Exception:
                _sigs = []
                _sig_uniq = -1

            try:
                _tag2 = getattr(pol, "_phased_q_tag", None) or getattr(pol, "phased_q_tag", None) or tag
            except Exception:
                _tag2 = tag

            print(
                f"[PhaseD-Q][CAND_PROBE] tag={_tag2} ep={ep_name if ep_name is not None else 'NA'} "
                f"la_len={int(len(la_list))} la_sig_uniq={int(_sig_uniq)} "
                f"sigs0={_sigs}",
                flush=True,
            )
            if msg:
                print(msg, flush=True)

        def _pick_best(cands):
            best = (None, None, -1.0, -1)
            for src, cand in cands:
                if cand is None:
                    continue
                n, uq, r = _cand_stats(cand)
                if n <= 0:
                    continue
                if (uq > best[3]) or (uq == best[3] and r > best[2]):
                    best = (src, cand, r, uq)
            return best

        tried = []
        cands_try = []

        def _try_one(src, raw):
            cand = _normalize_cand_vecs_32d(raw, len(la_list))
            if cand is None:
                tried.append(f"{src}:none")
                return
            _n, _uq, _r = _cand_stats(cand)
            tried.append(f"{src}:{_uq}/{_n}({_r:.3f})")
            cands_try.append((src, cand))

            # ★追加: converter_32d が候補間で同一化している瞬間を短いログで確定させる
            try:
                if str(src) == "converter_32d" and bool(PHASED_Q_DIAG_LOG):
                    if int(_n) > 1 and int(_uq) <= 1:
                        _take = min(3, int(len(cand)))
                        _crcs = ",".join([_crc32_row32(cand[i]) for i in range(_take)])
                        _vmin0, _vmax0 = _row_minmax32(cand[0])
                        _maybe_log_probe(
                            f"[PhaseD-Q][CONV32_FLAT] uniq={int(_uq)}/{int(_n)} r={float(_r):.3f} "
                            f"crc0={_crcs} mnmx={float(_vmin0):.1f}/{float(_vmax0):.1f}"
                        )

                        # ★追加: uniq=1 のときに、入力 legal_actions 側が区別できているかを併記して原因切り分け
                        try:
                            _la_n = int(len(la_list))
                        except Exception:
                            _la_n = None

                        _la_head = []
                        _la_uq = None
                        try:
                            _take_la = min(6, int(len(la_list)))
                            _la_head = [la_list[i] for i in range(_take_la)]
                        except Exception:
                            _la_head = []

                        _types_head = []
                        _types_uq = None
                        try:
                            _types = []
                            for _a in la_list:
                                if isinstance(_a, (list, tuple)) and len(_a) > 0:
                                    _types.append(int(_a[0]))
                                else:
                                    _types.append(None)
                            _types_head = _types[:min(12, int(len(_types)))]
                            _types_uq = len(set([t for t in _types if t is not None]))
                        except Exception:
                            _types_head = []
                            _types_uq = None

                        try:
                            _la_uq = len(set([tuple(_a) if isinstance(_a, list) else tuple(_a) if isinstance(_a, tuple) else _a for _a in la_list]))
                        except Exception:
                            _la_uq = None

                        _vary = []
                        try:
                            if isinstance(cand, list) and len(cand) > 0 and isinstance(cand[0], list):
                                _D = int(len(cand[0]))
                                _eps = 1e-9
                                for _d in range(_D):
                                    _mn = None
                                    _mx = None
                                    for _row in cand:
                                        try:
                                            _v = float(_row[_d])
                                        except Exception:
                                            _v = None
                                        if _v is None:
                                            continue
                                        if _mn is None or _v < _mn:
                                            _mn = _v
                                        if _mx is None or _v > _mx:
                                            _mx = _v
                                    if _mn is None or _mx is None:
                                        continue
                                    if (_mx - _mn) > _eps:
                                        _vary.append(_d)
                        except Exception:
                            _vary = []

                        _maybe_log_probe(
                            f"[PhaseD-Q][CONV32_FLAT_DETAIL] la={_la_n} la_uq={_la_uq} "
                            f"types_uq={_types_uq} vary_dims={len(_vary)} vary_idx={_vary[:8]} "
                            f"la_head={_la_head} types_head={_types_head}"
                        )
            except Exception:
                pass

        # 0) 第2引数 la_list 自体が 32D/17D 候補ベクトルなら、それを最優先で試す
        _try_one("arg_la_list", la_list)

        # 1) 呼び出し元 / state_dict マージ済み kwargs から拾う（キーを増やす）
        for _k in (
            "action_candidates_vec",
            "action_candidates_vecs",
            "cand_vecs",
            "candidates_vec",
            "legal_actions_vecs",
            "legal_actions_vec",
            "legal_actions_19d",
        ):
            if _k in kwargs:
                _try_one(f"kwargs.{_k}", kwargs.get(_k, None))

        # 2) converter から作る（32D/17D の両方を試す）
        conv = kwargs.get("converter", None) or kwargs.get("action_converter", None)
        if conv is None:
            try:
                conv = getattr(pol, "converter", None) or getattr(pol, "action_converter", None)
            except Exception:
                conv = None

        if conv is not None:
            fn32 = getattr(conv, "convert_legal_actions_32d", None)
            if callable(fn32):
                try:
                    cand2 = fn32(la_list)
                except Exception:
                    cand2 = None
                _try_one("converter_32d", cand2)
            else:
                tried.append("converter_32d:missing")

            fn = getattr(conv, "convert_legal_actions", None)
            if callable(fn):
                try:
                    cand3 = fn(la_list)
                except Exception:
                    cand3 = None
                _try_one("converter", cand3)
            else:
                tried.append("converter:missing")
        else:
            tried.append("converter:none")

        src_best, cand_best, r_best, uq_best = _pick_best(cands_try)

        if cand_best is None:
            _maybe_log_probe("[PhaseD-Q][CAND_WARN] no_candidate_vecs_32d_from_any_source tried=%s" % ("|".join(tried)))
            return None, "none"

        try:
            _warn_th = float(os.getenv("PHASED_Q_CAND_WARN_UNIQ_TH", "0.80"))
        except Exception:
            _warn_th = 0.80

        if float(r_best) <= float(_warn_th):
            _maybe_log_probe(
                "[PhaseD-Q][CAND_WARN] best_src=%s best_uniq=%d best_ratio=%.3f tried=%s"
                % (str(src_best), int(uq_best), float(r_best), "|".join(tried))
            )

        return cand_best, str(src_best)

    _known_eps = (
        "select_action_index_online",
        "select_action_index",
        "select_action",
    )
    _callable_eps = []
    for _ep in _known_eps:
        try:
            _fn = getattr(pol, _ep, None)
        except Exception:
            _fn = None
        if callable(_fn):
            _callable_eps.append(_ep)

    if not _callable_eps:
        print(f"[PhaseD-Q][WRAP] tag={tag} pol_id={id(pol)} class={type(pol).__name__} methods=NONE", flush=True)
        return pol

    def _wrap_one(ep_name):
        orig = getattr(pol, ep_name)

        def _phased_q_emit_summary(reason="game_over", reset=True, extra_roots=None):
            if bool(getattr(pol, "_phased_q_summary_emitted", False)):
                return
            setattr(pol, "_phased_q_summary_emitted", True)

            roots = [pol]
            try:
                if isinstance(extra_roots, (list, tuple)):
                    for _r in extra_roots:
                        if _r is not None and _r not in roots:
                            roots.append(_r)
            except Exception:
                pass

            def _iter_policies(_root):
                _seen = set()
                _stack = [_root]
                while _stack:
                    _p = _stack.pop()
                    if _p is None:
                        continue
                    _pid = id(_p)
                    if _pid in _seen:
                        continue
                    _seen.add(_pid)
                    yield _p

                    for _attr in (
                        "main_policy", "fallback_policy",
                        "main", "fallback",
                        "policy", "inner", "wrapped",
                        "main_pol", "fallback_pol",
                    ):
                        try:
                            _ch = getattr(_p, _attr, None)
                        except Exception:
                            _ch = None
                        if _ch is None:
                            continue
                        if isinstance(_ch, (list, tuple)):
                            for _c in _ch:
                                if _c is not None:
                                    _stack.append(_c)
                        else:
                            _stack.append(_ch)

            agg = {
                "calls_total": 0,
                "calls_q_eval_ok": 0,
                "calls_q_eval_none": 0,
                "calls_mix_applied": 0,
                "mix_fallback_base_pi": 0,

                "calls_ep_select_action_index_online": 0,
                "calls_ep_not_select_action_index_online": 0,

                "mix_changed": 0,
                "mix_same": 0,
                "mix_mcts_idx_none": 0,

                "skip_obs_not_numeric": 0,
                "skip_la_list_missing": 0,
                "skip_la_list_empty": 0,
                "skip_cand_vecs_missing": 0,

                "pi_changed": 0,
                "pi_l1_n": 0,
                "pi_l1_sum": 0.0,

                "la_len_n": 0,
                "la_len_sum": 0.0,
                "la_len_min": None,
                "la_len_max": None,

                "q_span_n": 0,
                "q_span_sum": 0.0,
                "q_span_min": None,
                "q_span_max": None,
                "q_flat_count": 0,

                "q_std_n": 0,
                "q_std_sum": 0.0,

                "cand_uniq_n": 0,
                "cand_uniq_ratio_sum": 0.0,
                "cand_all_same": 0,

                "pi_present": 0,
                "pi_entropy_n": 0,
                "pi_entropy_sum": 0.0,
                "pi_uniform_like": 0,

                "pi_none_from_ret": 0,
                "pi_from_state_dict": 0,
                "pi_fallback_uniform": 0,

                "mcts_idx_n": 0,
                "mcts_idx0": 0,

                "late_calls": 0,
                "late_q_used": 0,
                "late_mix_changed": 0,
            }

            visited = []
            try:
                _seen2 = set()
                for _root in roots:
                    for _obj in _iter_policies(_root):
                        _oid = id(_obj)
                        if _oid in _seen2:
                            continue
                        _seen2.add(_oid)
                        visited.append(_obj)

                        st = getattr(_obj, "_phased_q_stats", None)
                        if not isinstance(st, dict):
                            continue

                        for _k in (
                            "calls_total",
                            "calls_q_eval_ok",
                            "calls_q_eval_none",
                            "calls_mix_applied",
                            "mix_fallback_base_pi",
                            "calls_ep_select_action_index_online",
                            "calls_ep_not_select_action_index_online",
                            "mix_changed",
                            "mix_same",
                            "mix_mcts_idx_none",
                            "skip_obs_not_numeric",
                            "skip_la_list_missing",
                            "skip_la_list_empty",
                            "skip_cand_vecs_missing",
                            "pi_changed",
                            "pi_l1_n",
                            "la_len_n",

                            "q_span_n",
                            "q_flat_count",
                            "q_std_n",
                            "cand_uniq_n",
                            "cand_all_same",

                            "pi_present",
                            "pi_entropy_n",
                            "pi_uniform_like",
                            "pi_none_from_ret",
                            "pi_from_state_dict",
                            "pi_fallback_uniform",

                            "mcts_idx_n",
                            "mcts_idx0",
                            "late_calls",
                            "late_q_used",
                            "late_mix_changed",
                        ):

                            try:
                                agg[_k] = int(agg.get(_k, 0)) + int(st.get(_k, 0) or 0)
                            except Exception:
                                pass

                        try:
                            agg["pi_l1_sum"] = float(agg.get("pi_l1_sum", 0.0)) + float(st.get("pi_l1_sum", 0.0) or 0.0)
                        except Exception:
                            pass

                        try:
                            agg["la_len_sum"] = float(agg.get("la_len_sum", 0.0)) + float(st.get("la_len_sum", 0.0) or 0.0)
                        except Exception:
                            pass

                        try:
                            agg["q_span_sum"] = float(agg.get("q_span_sum", 0.0)) + float(st.get("q_span_sum", 0.0) or 0.0)
                        except Exception:
                            pass

                        try:
                            agg["q_std_sum"] = float(agg.get("q_std_sum", 0.0)) + float(st.get("q_std_sum", 0.0) or 0.0)
                        except Exception:
                            pass

                        try:
                            agg["cand_uniq_ratio_sum"] = float(agg.get("cand_uniq_ratio_sum", 0.0)) + float(st.get("cand_uniq_ratio_sum", 0.0) or 0.0)
                        except Exception:
                            pass

                        try:
                            agg["pi_entropy_sum"] = float(agg.get("pi_entropy_sum", 0.0)) + float(st.get("pi_entropy_sum", 0.0) or 0.0)
                        except Exception:
                            pass

                        try:
                            _mn = st.get("la_len_min", None)
                            if _mn is not None:
                                _mn = int(_mn)
                                agg["la_len_min"] = _mn if agg["la_len_min"] is None else min(int(agg["la_len_min"]), _mn)
                        except Exception:
                            pass

                        try:
                            _mx = st.get("la_len_max", None)
                            if _mx is not None:
                                _mx = int(_mx)
                                agg["la_len_max"] = _mx if agg["la_len_max"] is None else max(int(agg["la_len_max"]), _mx)
                        except Exception:
                            pass

                        try:
                            _qmn = st.get("q_span_min", None)
                            if _qmn is not None:
                                _qmn = float(_qmn)
                                agg["q_span_min"] = _qmn if agg["q_span_min"] is None else min(float(agg["q_span_min"]), _qmn)
                        except Exception:
                            pass

                        try:
                            _qmx = st.get("q_span_max", None)
                            if _qmx is not None:
                                _qmx = float(_qmx)
                                agg["q_span_max"] = _qmx if agg["q_span_max"] is None else max(float(agg["q_span_max"]), _qmx)
                        except Exception:
                            pass
            except Exception:
                visited = []

            calls_total = int(agg.get("calls_total", 0) or 0)
            if calls_total <= 0:
                return

            calls_q_eval_ok = int(agg.get("calls_q_eval_ok", 0) or 0)
            calls_q_eval_none = int(agg.get("calls_q_eval_none", 0) or 0)
            calls_mix_applied = int(agg.get("calls_mix_applied", 0) or 0)
            mix_fallback_base_pi = int(agg.get("mix_fallback_base_pi", 0) or 0)

            sk_obs = int(agg.get("skip_obs_not_numeric", 0) or 0)
            sk_la_missing = int(agg.get("skip_la_list_missing", 0) or 0)
            sk_la_empty = int(agg.get("skip_la_list_empty", 0) or 0)
            sk_cand = int(agg.get("skip_cand_vecs_missing", 0) or 0)
            ep_online = int(agg.get("calls_ep_select_action_index_online", 0) or 0)
            ep_other  = int(agg.get("calls_ep_not_select_action_index_online", 0) or 0)

            la_n = int(agg.get("la_len_n", 0) or 0)
            la_sum = float(agg.get("la_len_sum", 0.0) or 0.0)
            la_min = agg.get("la_len_min", None)
            la_max = agg.get("la_len_max", None)
            la_avg = (la_sum / float(la_n)) if la_n > 0 else 0.0

            mix_changed = int(agg.get("mix_changed", 0) or 0)
            mix_same = int(agg.get("mix_same", 0) or 0)
            mix_mcts_idx_none = int(agg.get("mix_mcts_idx_none", 0) or 0)
            mix_change_rate = (float(mix_changed) / float(calls_q_eval_ok)) if calls_q_eval_ok > 0 else 0.0

            pi_changed = int(agg.get("pi_changed", 0) or 0)
            pi_l1_n = int(agg.get("pi_l1_n", 0) or 0)
            pi_l1_sum = float(agg.get("pi_l1_sum", 0.0) or 0.0)
            pi_l1_avg = (pi_l1_sum / float(pi_l1_n)) if pi_l1_n > 0 else 0.0
            pi_change_rate = (float(pi_changed) / float(pi_l1_n)) if pi_l1_n > 0 else 0.0

            q_span_n = int(agg.get("q_span_n", 0) or 0)
            q_span_sum = float(agg.get("q_span_sum", 0.0) or 0.0)
            q_span_min = agg.get("q_span_min", None)
            q_span_max = agg.get("q_span_max", None)
            q_span_avg = (q_span_sum / float(q_span_n)) if q_span_n > 0 else 0.0

            q_flat = int(agg.get("q_flat_count", 0) or 0)
            q_flat_rate = (float(q_flat) / float(calls_q_eval_ok)) if calls_q_eval_ok > 0 else 0.0

            pi_present = int(agg.get("pi_present", 0) or 0)
            pi_present_rate = (float(pi_present) / float(calls_total)) if calls_total > 0 else 0.0

            q_std_n = int(agg.get("q_std_n", 0) or 0)
            q_std_sum = float(agg.get("q_std_sum", 0.0) or 0.0)
            q_std_avg = (q_std_sum / float(q_std_n)) if q_std_n > 0 else 0.0

            cand_uniq_n = int(agg.get("cand_uniq_n", 0) or 0)
            cand_uniq_ratio_sum = float(agg.get("cand_uniq_ratio_sum", 0.0) or 0.0)
            cand_uniq_ratio_avg = (cand_uniq_ratio_sum / float(cand_uniq_n)) if cand_uniq_n > 0 else 0.0
            cand_all_same = int(agg.get("cand_all_same", 0) or 0)
            cand_all_same_rate = (float(cand_all_same) / float(cand_uniq_n)) if cand_uniq_n > 0 else 0.0

            print(
                f"[PhaseD-Q][SUMMARY] tag={tag} reason={reason}"
                f" calls_total={calls_total}"
                f" ep_online={ep_online} ep_other={ep_other}"
                f" q_eval_ok={calls_q_eval_ok} q_eval_none={calls_q_eval_none}"
                f" mix_applied={calls_mix_applied} mix_change_rate={mix_change_rate:.3f}"
                f" q_span_avg={q_span_avg:.6f} q_flat_rate={q_flat_rate:.3f}"
                f" cand_uniq_ratio_avg={cand_uniq_ratio_avg:.3f} cand_all_same_rate={cand_all_same_rate:.3f}"
                ,
                flush=True,
            )
            print(
                f"[PhaseD-Q][SUMMARY_DETAIL] tag={tag}"
                f" mix_fallback_base_pi={mix_fallback_base_pi}"
                f" mix_changed={mix_changed} mix_same={mix_same} mix_mcts_idx_none={mix_mcts_idx_none}"
                f" pi_changed={pi_changed} pi_change_rate={pi_change_rate:.3f} pi_l1_avg={pi_l1_avg:.6f}"
                f" skip_obs_not_numeric={sk_obs} skip_la_missing={sk_la_missing}"
                f" skip_la_empty={sk_la_empty} skip_cand_missing={sk_cand}"
                f" la_len_avg={la_avg:.2f} la_len_min={la_min} la_len_max={la_max}"
                f" q_span_min={q_span_min} q_span_max={q_span_max} q_std_avg={q_std_avg:.6f}"
                f" pi_present_rate={pi_present_rate:.3f}"
                ,
                flush=True,
            )

            try:
                for _obj in visited:
                    setattr(_obj, "_phased_q_summary_emitted", True)
            except Exception:
                pass

            if reset:
                try:
                    for _obj in visited:
                        st = getattr(_obj, "_phased_q_stats", None)
                        if isinstance(st, dict):
                            st.clear()
                except Exception:
                    pass

        if not hasattr(pol, "phaseD_q_emit_summary"):
            setattr(pol, "phaseD_q_emit_summary", _phased_q_emit_summary)
        if not hasattr(pol, "emit_phased_q_summary_once"):
            setattr(pol, "emit_phased_q_summary_once", _phased_q_emit_summary)

        if not getattr(pol, "_phased_q_stats_atexit_registered", False):
            try:
                import atexit as _atexit
                _atexit.register(lambda: _phased_q_emit_summary(reason="atexit", reset=False))
                setattr(pol, "_phased_q_stats_atexit_registered", True)
            except Exception:
                setattr(pol, "_phased_q_stats_atexit_registered", True)

        def wrapped(*args, **kwargs):
            _LOG_DETAIL = bool(globals().get("LOG_DEBUG_DETAIL", False))
            _DIAG = bool(globals().get("PHASED_Q_DIAG_LOG", False)) or _LOG_DETAIL

            def _diag(line):
                if not _DIAG:
                    return
                try:
                    print(line, flush=True)
                except Exception:
                    pass

            def _diag_kv(reason, **kv):
                if not _DIAG:
                    return
                try:
                    parts = [f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason={reason}"]
                    for k, v in kv.items():
                        try:
                            sv = str(v)
                        except Exception:
                            sv = "<err>"
                        if len(sv) > 240:
                            sv = sv[:240] + "..."
                        parts.append(f"{k}={sv}")
                    _diag(" ".join(parts))
                except Exception:
                    pass

            def _should_periodic_diag(call_i):
                try:
                    every = int(PHASED_Q_DIAG_EVERY)
                except Exception:
                    every = 0
                if every <= 0:
                    return False
                try:
                    return (int(call_i) % int(every)) == 0
                except Exception:
                    return False

            st = getattr(pol, "_phased_q_stats", None)
            if not isinstance(st, dict):
                st = {}
                setattr(pol, "_phased_q_stats", st)

            if not bool(st.get("diag_orig_logged", 0)):
                st["diag_orig_logged"] = 1
                try:
                    _qname = getattr(orig, "__qualname__", None)
                except Exception:
                    _qname = None
                try:
                    _mod = getattr(orig, "__module__", None)
                except Exception:
                    _mod = None
                try:
                    _name = getattr(orig, "__name__", None)
                except Exception:
                    _name = None
                _diag_kv(
                    "orig_info",
                    pol_class=type(pol).__name__,
                    pol_id=id(pol),
                    orig_module=_mod,
                    orig_name=_name,
                    orig_qualname=_qname,
                )

            st["calls_total"] = int(st.get("calls_total", 0)) + 1

            if _should_periodic_diag(st.get("calls_total", 0)):
                _diag_kv(
                    "periodic_snapshot",
                    calls_total=st.get("calls_total", 0),
                    calls_q_eval_ok=st.get("calls_q_eval_ok", 0),
                    calls_q_eval_none=st.get("calls_q_eval_none", 0),
                    calls_mix_applied=st.get("calls_mix_applied", 0),
                    pi_none_from_ret=st.get("pi_none_from_ret", 0),
                    pi_from_state_dict=st.get("pi_from_state_dict", 0),
                    pi_fallback_uniform=st.get("pi_fallback_uniform", 0),
                    cand_all_same=st.get("cand_all_same", 0),
                    q_flat_count=st.get("q_flat_count", 0),
                )

            if ep_name == "select_action_index_online":
                st["calls_ep_select_action_index_online"] = int(st.get("calls_ep_select_action_index_online", 0)) + 1
            else:
                st["calls_ep_not_select_action_index_online"] = int(st.get("calls_ep_not_select_action_index_online", 0)) + 1

            ret = orig(*args, **kwargs)

            if not bool(USE_PHASED_Q) or not bool(PHASED_Q_MIX_ENABLED):
                return ret

            if ep_name != "select_action_index_online":
                st["skip_ep_select_action_index_online"] = int(st.get("skip_ep_select_action_index_online", 0)) + 1
                return ret

            if isinstance(ret, tuple) and len(ret) == 2:
                base_out, pi = ret
            else:
                base_out, pi = ret, None

            state_dict = None
            if ep_name == "select_action_index_online":
                if len(args) >= 1 and isinstance(args[0], dict):
                    state_dict = args[0]
                if state_dict is None:
                    state_dict = kwargs.get("state_dict", None) if isinstance(kwargs.get("state_dict", None), dict) else None
            else:
                state_dict = kwargs.get("state_dict", None) if isinstance(kwargs.get("state_dict", None), dict) else None

            kw2 = dict(kwargs)
            if isinstance(state_dict, dict):
                for _k, _v in state_dict.items():
                    if _k not in kw2:
                        kw2[_k] = _v
            kwargs = kw2

            if pi is None:
                st["pi_none_from_ret"] = int(st.get("pi_none_from_ret", 0)) + 1
                if isinstance(state_dict, dict):
                    for _k in ("pi", "pi_mcts", "mcts_pi", "policy_pi"):
                        if _k in state_dict:
                            pi = state_dict.get(_k, None)
                            st["pi_from_state_dict"] = int(st.get("pi_from_state_dict", 0)) + 1
                            break

                if pi is None:
                    st["pi_missing_after_state_dict"] = int(st.get("pi_missing_after_state_dict", 0)) + 1
                    try:
                        _keys = list(state_dict.keys()) if isinstance(state_dict, dict) else None
                        _khead = _keys[:12] if isinstance(_keys, list) else None
                    except Exception:
                        _khead = None
                    _diag(
                        f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason=pi_missing"
                        f" state_dict_is_dict={int(isinstance(state_dict, dict))}"
                        f" state_keys_head={_khead}"
                    )

            obs_vec = None
            la_list = None
            _obs_src = "unset"
            _la_src = "unset"

            if ep_name == "select_action_index_online":
                obs_vec = _as_list((state_dict.get("obs_vec", None) if isinstance(state_dict, dict) else None))
                if obs_vec is not None:
                    _obs_src = "state_dict.obs_vec"
                if obs_vec is None:
                    obs_vec = _as_list((state_dict.get("obs", None) if isinstance(state_dict, dict) else None))
                    if obs_vec is not None:
                        _obs_src = "state_dict.obs"
                if obs_vec is None:
                    obs_vec = _as_list((state_dict.get("public_obs_vec", None) if isinstance(state_dict, dict) else None))
                    if obs_vec is not None:
                        _obs_src = "state_dict.public_obs_vec"
                if obs_vec is None:
                    obs_vec = _as_list((state_dict.get("full_obs_vec", None) if isinstance(state_dict, dict) else None))
                    if obs_vec is not None:
                        _obs_src = "state_dict.full_obs_vec"

                if len(args) >= 2 and la_list is None:
                    la_list = args[1]
                    if la_list is not None:
                        _la_src = "args[1]"

                if la_list is None and isinstance(state_dict, dict):
                    for _k in ("legal_actions", "legal_actions_list", "legal_actions_vec", "legal_actions_vecs", "legal_actions_19d", "la_list"):
                        if _k in state_dict and state_dict.get(_k, None) is not None:
                            la_list = state_dict.get(_k, None)
                            _la_src = f"state_dict.{_k}"
                            break
            else:
                if len(args) >= 1:
                    obs_vec = _as_list(args[0])
                if len(args) >= 2:
                    la_list = args[1]

                if not _is_numeric_vec(obs_vec):
                    obs_vec = _as_list(
                        kwargs.get("obs_vec", None)
                        or kwargs.get("obs", None)
                        or kwargs.get("public_obs_vec", None)
                        or kwargs.get("full_obs_vec", None)
                    )

                if la_list is None:
                    la_list = (
                        kwargs.get("legal_actions", None)
                        or kwargs.get("legal_actions_list", None)
                        or kwargs.get("legal_actions_vec", None)
                        or kwargs.get("legal_actions_vecs", None)
                        or kwargs.get("legal_actions_19d", None)
                        or kwargs.get("la_list", None)
                    )

            if not _is_numeric_vec(obs_vec):
                st["skip_obs_not_numeric"] = int(st.get("skip_obs_not_numeric", 0)) + 1
                _diag(
                    f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason=obs_not_numeric"
                    f" obs_src={_obs_src} obs_type={type(obs_vec).__name__}"
                    f" state_dict_is_dict={int(isinstance(state_dict, dict))}"
                )
                return ret

            if la_list is None:
                st["skip_la_list_missing"] = int(st.get("skip_la_list_missing", 0)) + 1
                _diag(
                    f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason=la_list_missing"
                    f" la_src={_la_src} args_len={int(len(args))}"
                    f" state_dict_is_dict={int(isinstance(state_dict, dict))}"
                )
                return ret

            la_list = la_list if isinstance(la_list, list) else list(la_list)

            try:
                _llen = int(len(la_list))
                st["la_len_n"] = int(st.get("la_len_n", 0)) + 1
                st["la_len_sum"] = float(st.get("la_len_sum", 0.0)) + float(_llen)
                _mn = st.get("la_len_min", None)
                _mx = st.get("la_len_max", None)
                st["la_len_min"] = _llen if _mn is None else min(int(_mn), _llen)
                st["la_len_max"] = _llen if _mx is None else max(int(_mx), _llen)
            except Exception:
                pass

            if not la_list:
                st["skip_la_list_empty"] = int(st.get("skip_la_list_empty", 0)) + 1
                _diag(
                    f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason=la_list_empty"
                    f" la_src={_la_src} obs_src={_obs_src}"
                    f" la_type={type(la_list).__name__} la_len=0"
                )
                return ret

            cand_vecs_f, cand_src = _make_cand_vecs_32d(la_list, kwargs, ep_name=ep_name)
            if cand_vecs_f is None:
                st["skip_cand_vecs_missing"] = int(st.get("skip_cand_vecs_missing", 0)) + 1
                _diag(
                    f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason=cand_vecs_missing"
                    f" cand_src={cand_src} la_src={_la_src} la_len={int(len(la_list))}"
                )
                return ret

            try:
                _n = int(len(cand_vecs_f))
                _uq = 0
                if _n > 0:
                    _uq = len({tuple(round(float(x), 6) for x in row[:32]) for row in cand_vecs_f})
                _r = (float(_uq) / float(_n)) if _n > 0 else 0.0

                st["cand_uniq_n"] = int(st.get("cand_uniq_n", 0)) + 1
                st["cand_uniq_ratio_sum"] = float(st.get("cand_uniq_ratio_sum", 0.0)) + float(_r)
                if int(_n) > 1 and int(_uq) <= 1:
                    st["cand_all_same"] = int(st.get("cand_all_same", 0)) + 1

                if bool(PHASED_Q_DIAG_ON_LOW_UNIQ) and float(_r) <= float(PHASED_Q_DIAG_LOW_UNIQ_TH):
                    _diag_kv(
                        "cand_low_uniq",
                        cand_src=cand_src,
                        la_len=len(la_list),
                        cand_n=_n,
                        cand_uniq=_uq,
                        cand_ratio=f"{_r:.3f}",
                    )
            except Exception:
                pass

            try:
                obs_vec_f = [float(x) for x in obs_vec]
            except Exception:
                st["skip_obs_float_cast"] = int(st.get("skip_obs_float_cast", 0)) + 1
                _diag(
                    f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason=obs_float_cast_failed"
                    f" obs_src={_obs_src} obs_type={type(obs_vec).__name__}"
                    f" obs_len={int(len(obs_vec)) if isinstance(obs_vec, list) else -1}"
                )
                return ret

            q_vals = phaseD_q_evaluate(obs_vec_f, cand_vecs_f)
            if q_vals is None:
                st["calls_q_eval_none"] = int(st.get("calls_q_eval_none", 0)) + 1
                _diag(
                    f"[PhaseD-Q][DIAG] tag={tag} ep={ep_name} reason=q_eval_none"
                    f" obs_len={int(len(obs_vec_f))} la_len={int(len(la_list))} cand_src={cand_src}"
                )
                return ret

            st["calls_q_eval_ok"] = int(st.get("calls_q_eval_ok", 0)) + 1

            base_pi = _normalize_base_pi(pi, len(la_list))
            if not isinstance(base_pi, list) or len(base_pi) != len(la_list):
                st["pi_fallback_uniform"] = int(st.get("pi_fallback_uniform", 0)) + 1
                n = len(la_list)
                base_pi = [1.0 / float(n)] * n

            # ★ 追加: base_pi を float & 非負 & 正規化（np.asarray(dtype=float64) で落ちないようにする）
            try:
                w = []
                s = 0.0
                for p in base_pi:
                    try:
                        pf = float(p)
                    except Exception:
                        pf = 0.0
                    if not (pf > 0.0):
                        pf = 0.0
                    w.append(pf)
                    s += pf
                if not (s > 0.0):
                    raise ValueError("base_pi_sum0")
                base_pi = [x / s for x in w]
            except Exception:
                st["pi_fallback_uniform"] = int(st.get("pi_fallback_uniform", 0)) + 1
                n = len(la_list)
                base_pi = [1.0 / float(n)] * n

            try:
                st["pi_present"] = int(st.get("pi_present", 0)) + (1 if pi is not None else 0)
            except Exception:
                pass

            try:
                a = np.asarray(base_pi, dtype=np.float64).reshape(-1)
                if a.size > 0 and np.all(np.isfinite(a)):
                    eps = 1e-12
                    aa = np.maximum(a, 0.0)
                    s = float(aa.sum())
                    if s > 0.0:
                        aa = aa / s
                    ent = float(-np.sum(aa * np.log(np.maximum(aa, eps))))
                    st["pi_entropy_n"] = int(st.get("pi_entropy_n", 0)) + 1
                    st["pi_entropy_sum"] = float(st.get("pi_entropy_sum", 0.0)) + float(ent)

                    try:
                        if float(np.max(aa) - np.min(aa)) < 1e-6:
                            st["pi_uniform_like"] = int(st.get("pi_uniform_like", 0)) + 1
                    except Exception:
                        pass
            except Exception:
                pass

            mixed_pi = phaseD_mix_pi_with_q(base_pi, q_vals)
            if mixed_pi is None:
                st["mix_fallback_base_pi"] = int(st.get("mix_fallback_base_pi", 0)) + 1
                mixed_pi = base_pi

            try:
                if not isinstance(mixed_pi, list) or len(mixed_pi) != len(la_list):
                    raise ValueError("mixed_pi_bad_shape")
                w = []
                s = 0.0
                for p in mixed_pi:
                    try:
                        pf = float(p)
                    except Exception:
                        pf = 0.0
                    if not (pf > 0.0):
                        pf = 0.0
                    w.append(pf)
                    s += pf
                if not (s > 0.0):
                    raise ValueError("mixed_pi_sum0")
                mixed_pi = [x / s for x in w]
            except Exception:
                st["mix_fallback_base_pi"] = int(st.get("mix_fallback_base_pi", 0)) + 1
                mixed_pi = base_pi

            import random as _random
            new_idx = _random.choices(range(len(la_list)), weights=mixed_pi, k=1)[0]

            st["calls_mix_applied"] = int(st.get("calls_mix_applied", 0)) + 1
            st["calls_q_used"] = int(st.get("calls_q_used", 0)) + 1

            _mcts_selected = None
            _mcts_sel_reason = None
            try:
                if isinstance(base_out, (int, np.integer)):
                    _mcts_selected = int(base_out)
                    _mcts_sel_reason = "base_out_int"
                else:
                    _mcts_sel_reason = "base_out_not_int"

                    if isinstance(la_list, list):
                        for _i, _a in enumerate(la_list):
                            try:
                                if _a == base_out:
                                    _mcts_selected = int(_i)
                                    _mcts_sel_reason = "match_eq"
                                    break
                            except Exception:
                                continue

                        if _mcts_selected is None:
                            for _i, _a in enumerate(la_list):
                                try:
                                    if _a is base_out:
                                        _mcts_selected = int(_i)
                                        _mcts_sel_reason = "match_is"
                                        break
                                except Exception:
                                    continue
            except Exception:
                _mcts_selected = None
                _mcts_sel_reason = "exception"

            _mcts_top = _topk_pairs(base_pi, k=3)
            _q_top = _topk_pairs(q_vals, k=3)
            _mix_top = _topk_pairs(mixed_pi, k=3)

            _mcts_argmax = None
            _mix_argmax = None
            try:
                _mcts_argmax = int(np.argmax(np.asarray(base_pi, dtype=np.float64)))
            except Exception:
                try:
                    _mcts_argmax = int(max(range(len(base_pi)), key=lambda i: float(base_pi[i])))
                except Exception:
                    _mcts_argmax = None
            try:
                _mix_argmax = int(np.argmax(np.asarray(mixed_pi, dtype=np.float64)))
            except Exception:
                try:
                    _mix_argmax = int(max(range(len(mixed_pi)), key=lambda i: float(mixed_pi[i])))
                except Exception:
                    _mix_argmax = None

            _mcts_ref = _mcts_selected if _mcts_selected is not None else _mcts_argmax

            _pi_l1 = None
            try:
                if isinstance(base_pi, list) and isinstance(mixed_pi, list) and len(base_pi) == len(mixed_pi):
                    _pi_l1 = float(sum(abs(float(mixed_pi[i]) - float(base_pi[i])) for i in range(len(base_pi))))
            except Exception:
                _pi_l1 = None

            if _pi_l1 is not None:
                st["pi_l1_n"] = int(st.get("pi_l1_n", 0)) + 1
                st["pi_l1_sum"] = float(st.get("pi_l1_sum", 0.0)) + float(_pi_l1)
                if _pi_l1 > 1e-12:
                    st["pi_changed"] = int(st.get("pi_changed", 0)) + 1

            if _mcts_ref is None:
                st["mix_mcts_idx_none"] = int(st.get("mix_mcts_idx_none", 0)) + 1
                st["mix_mcts_idx_none_reason_base_out_not_int"] = int(st.get("mix_mcts_idx_none_reason_base_out_not_int", 0)) + (1 if _mcts_sel_reason == "base_out_not_int" else 0)
                st["mix_mcts_idx_none_reason_exception"] = int(st.get("mix_mcts_idx_none_reason_exception", 0)) + (1 if _mcts_sel_reason == "exception" else 0)
                st["mix_mcts_idx_none_reason_no_match"] = int(st.get("mix_mcts_idx_none_reason_no_match", 0)) + (1 if _mcts_sel_reason in ("base_out_not_int",) and _mcts_selected is None else 0)

                try:
                    _bpi = np.asarray(base_pi, dtype=np.float64).reshape(-1)
                    _bpi_len = int(_bpi.size)
                    _bpi_sum = float(_bpi.sum()) if _bpi.size > 0 else 0.0
                    _bpi_min = float(np.min(_bpi)) if _bpi.size > 0 else 0.0
                    _bpi_max = float(np.max(_bpi)) if _bpi.size > 0 else 0.0
                    _bpi_fin = int(np.all(np.isfinite(_bpi))) if _bpi.size > 0 else 0
                except Exception:
                    _bpi_len = -1
                    _bpi_sum = 0.0
                    _bpi_min = 0.0
                    _bpi_max = 0.0
                    _bpi_fin = 0

                try:
                    _bo = repr(base_out)
                    if len(_bo) > 240:
                        _bo = _bo[:240] + "..."
                except Exception:
                    _bo = "<repr_err>"

                # 乱発を避ける（最初の20回だけ詳細を出す）
                try:
                    _c = int(st.get("diag_mcts_ref_none_printed", 0))
                except Exception:
                    _c = 0

                if _c < 20:
                    st["diag_mcts_ref_none_printed"] = _c + 1

                    try:
                        _keys = list(state_dict.keys()) if isinstance(state_dict, dict) else None
                        _khead = _keys[:24] if isinstance(_keys, list) else None
                    except Exception:
                        _khead = None

                    _diag_kv(
                        "mcts_ref_none",
                        base_out_type=type(base_out).__name__,
                        base_out_repr=_bo,
                        mcts_sel_reason=_mcts_sel_reason,
                        mcts_selected=_mcts_selected,
                        mcts_argmax=_mcts_argmax,
                        base_pi_len=_bpi_len,
                        base_pi_sum=f"{_bpi_sum:.6g}",
                        base_pi_min=f"{_bpi_min:.6g}",
                        base_pi_max=f"{_bpi_max:.6g}",
                        base_pi_all_finite=_bpi_fin,
                        pi_type=type(pi).__name__ if pi is not None else "None",
                        pi_from_ret=int(1 if (isinstance(ret, tuple) and len(ret) == 2) else 0),
                        la_src=_la_src,
                        obs_src=_obs_src,
                        state_keys_head=_khead,
                    )

                if _should_periodic_diag(st.get("calls_total", 0)):
                    _diag_kv(
                        "periodic_snapshot",
                        calls_total=st.get("calls_total", 0),
                        mix_mcts_idx_none=st.get("mix_mcts_idx_none", 0),
                        pi_none_from_ret=st.get("pi_none_from_ret", 0),
                        pi_from_state_dict=st.get("pi_from_state_dict", 0),
                        pi_fallback_uniform=st.get("pi_fallback_uniform", 0),
                    )
            else:
                st["mcts_idx_n"] = int(st.get("mcts_idx_n", 0)) + 1
                if int(_mcts_ref) == 0:
                    st["mcts_idx0"] = int(st.get("mcts_idx0", 0)) + 1
                if int(new_idx) != int(_mcts_ref):
                    st["mix_changed"] = int(st.get("mix_changed", 0)) + 1
                else:
                    st["mix_same"] = int(st.get("mix_same", 0)) + 1

            _q_span = 0.0
            _q_flat = 0
            _q_std = 0.0
            try:
                _qv = np.asarray(q_vals, dtype=np.float64).reshape(-1)
                if _qv.size > 0:
                    _q_span = float(np.max(_qv) - np.min(_qv))
                    _q_std = float(np.std(_qv))
                    _q_flat = 1 if _q_span < float(PHASED_Q_Q_FLAT_EPS) else 0

                    st["q_span_n"] = int(st.get("q_span_n", 0)) + 1
                    st["q_span_sum"] = float(st.get("q_span_sum", 0.0)) + float(_q_span)
                    _mn = st.get("q_span_min", None)
                    _mx = st.get("q_span_max", None)
                    st["q_span_min"] = float(_q_span) if _mn is None else min(float(_mn), float(_q_span))
                    st["q_span_max"] = float(_q_span) if _mx is None else max(float(_mx), float(_q_span))

                    st["q_std_n"] = int(st.get("q_std_n", 0)) + 1
                    st["q_std_sum"] = float(st.get("q_std_sum", 0.0)) + float(_q_std)

                    if int(_q_flat) == 1:
                        st["q_flat_count"] = int(st.get("q_flat_count", 0)) + 1
                        if bool(PHASED_Q_DIAG_ON_FLAT):
                            _diag_kv(
                                "q_flat",
                                q_span=f"{_q_span:.6g}",
                                q_std=f"{_q_std:.6g}",
                                cand_src=cand_src,
                                la_len=len(la_list),
                            )
            except Exception:
                _q_span = 0.0
                _q_flat = 0
                _q_std = 0.0

            def _compact_action_str(a):
                try:
                    s = str(a)
                except Exception:
                    s = "<action>"
                try:
                    s = s.replace("\r", " ").replace("\n", " ")
                except Exception:
                    pass
                try:
                    mx = int(PHASED_Q_ACTION_STR_MAX)
                except Exception:
                    mx = 160
                if mx > 0 and len(s) > mx:
                    s = s[:mx] + "..."
                return s

            _chosen_action_str = _compact_action_str(la_list[new_idx]) if 0 <= int(new_idx) < int(len(la_list)) else "<action>"

            _decide_log = bool(PHASED_Q_DECIDE_LOG)
            _diff_only = bool(PHASED_Q_DECIDE_LOG_DIFF_ONLY)

            _decide_target_ok = True
            try:
                if isinstance(tag, str) and tag.endswith(".outer"):
                    mp = getattr(pol, "main_policy", None)
                    if mp is not None:
                        _decide_target_ok = False
            except Exception:
                _decide_target_ok = True

            if _decide_log and _decide_target_ok:
                if (not _diff_only) or (_mcts_ref is None) or (int(new_idx) != int(_mcts_ref)):
                    _line = (
                        f"[DECIDE_PRE] tag={tag} ep={ep_name} cand_src={cand_src}"
                        f" la_len={int(len(la_list))}"
                        f" lam={float(PHASED_Q_MIX_LAMBDA):.3f} tau={float(PHASED_Q_MIX_TEMPERATURE):.3f}"
                        f" mcts_sel={_mcts_selected} mcts_ref={_mcts_ref} mcts_argmax={_mcts_argmax} mix_argmax={_mix_argmax} sampled={int(new_idx)}"
                        f" q_span={float(_q_span):.6f} q_flat={int(_q_flat)}"
                        f" mcts_top3={_mcts_top} q_top3={_q_top} mix_top3={_mix_top}"
                    )
                    try:
                        setattr(pol, "_last_decide_pre_line", _line)
                    except Exception:
                        pass

            if _decide_log and _decide_target_ok:
                if (not _diff_only) or (_mcts_ref is None) or (int(new_idx) != int(_mcts_ref)):
                    _line = f"[DECIDE_POST] tag={tag} ep={ep_name} chosen_idx={int(new_idx)} action={_chosen_action_str}"
                    try:
                        setattr(pol, "_last_decide_post_line", _line)
                    except Exception:
                        pass

            if _decide_log and _decide_target_ok:
                if (_mcts_ref is not None) and (int(new_idx) != int(_mcts_ref)):
                    _mcts_action_str = _compact_action_str(la_list[int(_mcts_ref)]) if 0 <= int(_mcts_ref) < int(len(la_list)) else "<action>"
                    _line = (
                        f"[DECIDE_DIFF] tag={tag} ep={ep_name} mcts_idx={int(_mcts_ref)} mix_idx={int(new_idx)}"
                        f" mcts_action={_mcts_action_str} mix_action={_chosen_action_str}"
                    )
                    try:
                        setattr(pol, "_last_decide_diff_line", _line)
                    except Exception:
                        pass

            if _LOG_DETAIL:
                print(
                    f"[PhaseD-Q][MIX] tag={tag} ep={ep_name} cand_src={cand_src}"
                    f" lam={float(PHASED_Q_MIX_LAMBDA):.3f} tau={float(PHASED_Q_MIX_TEMPERATURE):.3f}"
                    f" mix_idx={int(new_idx)}"
                    f" mcts_top3={_mcts_top} q_top3={_q_top} mix_top3={_mix_top}",
                    flush=True,
                )

            _out_action = la_list[new_idx]
            try:
                if ep_name in ("select_action_index_online", "select_action_index"):
                    _out_action = int(new_idx)
            except Exception:
                _out_action = la_list[new_idx]

            if isinstance(ret, tuple) and len(ret) == 2:
                if isinstance(pi, dict):
                    _pi = dict(pi)
                    _pi["pi"] = mixed_pi
                    _pi["pi_base_mcts"] = base_pi
                    _pi["phaseD_q_values"] = [float(x) for x in q_vals]
                    _pi["phaseD_q_used"] = True
                    _pi["phaseD_mix"] = {
                        "lambda": float(PHASED_Q_MIX_LAMBDA),
                        "temperature": float(PHASED_Q_MIX_TEMPERATURE),
                    }
                    return _out_action, _pi
                return _out_action, mixed_pi

            return _out_action

        return wrapped


    for _ep in _callable_eps:
        setattr(pol, _ep, _wrap_one(_ep))

    pol._phased_q_wrapped = True
    pol.phased_q_wrapped = True
    pol._phased_q_tag = tag
    pol.phased_q_tag = tag

    print(f"[PhaseD-Q][WRAP] tag={tag} pol_id={id(pol)} class={type(pol).__name__} methods={','.join(_callable_eps)}", flush=True)
    try:
        setattr(pol, "_phased_q_wrap_log_emitted", True)
    except Exception:
        pass

    return pol


# 旧名互換（ai_vs_ai 側で _wrap_* を使っていても動くように）
_wrap_select_action_with_phased_q = wrap_select_action_with_phased_q
