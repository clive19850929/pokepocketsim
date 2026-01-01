# policy_trace_monkeypatch.py
# -*- coding: utf-8 -*-
"""
POLICY_TRACE=1 で ModelPolicy.select_action をラップし、以下を出力します。
- 合法手のエンコード coverage（成功/総数）
- エンコード失敗の例（最大3件）
- 可能なら Q の上位 k 手（k=POLICY_TOPK, 既定5）
使い方:
  1) 本ファイルをプロジェクトの import パス上に置く（ai vs ai.py と同じ階層など）
  2) ai vs ai.py の先頭付近に `import policy_trace_monkeypatch` を1行追加
     ※ 下の「追加ファイル②: sitecustomize.py」を使えば ai vs ai.py の修正は不要です
  3) 実行時に環境変数 POLICY_TRACE=1 を設定（例: PowerShell → `$env:POLICY_TRACE=1`）
"""
import os
import sys
import types
import atexit
from collections import deque

if os.getenv("POLICY_TRACE", "0") != "1":
    # トレース無効
    pass
else:
    # どの場所にあっても対象モジュールを見つける
    pol_mod = None
    for name in ("pokepocketsim.policy.d3rlpy_policy", "d3rlpy_policy"):
        try:
            pol_mod = __import__(name, fromlist=["*"])
            break
        except Exception:
            continue

    if pol_mod is not None and hasattr(pol_mod, "ModelPolicy"):
        ModelPolicy = pol_mod.ModelPolicy
        _TOPK = int(os.getenv("POLICY_TOPK", "5"))
        _PRINT_FAIL_EXAMPLES = 3

        def _encode_vec_safe(self, a):
            # 可能な限り安全に encode_action_vec を呼ぶ
            try:
                if hasattr(self, "_encode_action_vec") and callable(self._encode_action_vec):
                    return self._encode_action_vec(a)
            except Exception:
                pass
            try:
                enc = getattr(getattr(self, "encoder", None), "encode_action_vec", None)
                if enc is not None:
                    return enc(a)
            except Exception:
                pass
            # どうにもならなければ失敗
            raise RuntimeError("no action encoder available")

        def _score_q_safe(self, obs_batch, act_batch):
            # 可能なら Q を返す。失敗したら None
            # 期待形状: obs_batch -> (N, obs_dim), act_batch -> (N, act_dim)
            import numpy as _np
            try:
                if hasattr(self, "predict_q") and callable(self.predict_q):
                    return _np.asarray(self.predict_q(obs_batch, act_batch)).reshape(-1)
            except Exception:
                pass
            try:
                # d3rlpy の actor 出力とコサイン類似度で近似スコア
                actor = getattr(getattr(self, "model", None), "policy", None)
                if actor is not None and hasattr(actor, "predict"):
                    a_star = _np.asarray(actor.predict(obs_batch)).reshape(len(obs_batch), -1)
                    # act_batch: (N, A), a_star: (N, A)
                    # cos = (a·b)/(|a||b|)
                    dot = (act_batch * a_star).sum(axis=1)
                    na = _np.linalg.norm(act_batch, axis=1) + 1e-8
                    nb = _np.linalg.norm(a_star, axis=1) + 1e-8
                    return (dot / (na * nb)).reshape(-1)
            except Exception:
                pass
            return None

        def _wrap_select_action(orig_fn):
            def _wrapped(self, state, legal_actions):
                import numpy as _np
                # coverage を測定
                ok = 0
                fails = []
                encs = []
                for a in (legal_actions or []):
                    try:
                        v = _encode_vec_safe(self, a)
                        encs.append(_np.asarray(v).reshape(-1))
                        ok += 1
                    except Exception as ex:
                        if len(fails) < _PRINT_FAIL_EXAMPLES:
                            fails.append(str(a)[:80])
                total = len(legal_actions or [])
                if total > 0:
                    cov = 100.0 * ok / max(1, total)
                    msg = f"[POLICY_TRACE] coverage: {ok}/{total} ({cov:.1f}%)"
                    if fails:
                        msg += " | fails(example)=" + "; ".join(fails)
                    print(msg)

                # Q 上位 k
                try:
                    if ok > 0 and _TOPK > 0:
                        # state はすでにベクトルで来る前提。dict の場合はエンコーダがあれば使用
                        obs_vec = None
                        if isinstance(state, dict):
                            encs_fn = getattr(getattr(self, "encoder", None), "encode_state", None)
                            if encs_fn is not None:
                                obs_vec = _np.asarray(encs_fn(state)).reshape(1, -1)
                        if obs_vec is None:
                            obs_vec = _np.asarray(state, dtype="float32").reshape(1, -1)
                        act_batch = _np.stack(encs, axis=0)
                        obs_batch = _np.repeat(obs_vec, repeats=act_batch.shape[0], axis=0)

                        qs = _score_q_safe(self, obs_batch, act_batch)
                        if qs is not None:
                            idx = _np.argsort(-qs)[: min(_TOPK, qs.shape[0])]
                            preview = ", ".join([f"{i}:{qs[i]:.3f}" for i in idx.tolist()])
                            print(f"[POLICY_TRACE] top{len(idx)} q: {preview}")
                except Exception:
                    pass

                # 本来の選択
                result = orig_fn(self, state, legal_actions)
                try:
                    action = result
                    pi = None
                    if isinstance(result, tuple) and len(result) == 2:
                        action, pi = result
                    if pi is not None:
                        setattr(self, "_policy_trace_last_pi", pi)
                    return action
                except Exception:
                    return result
            return _wrapped

        # select_action / select_action_index のどちらでもラップする
        if hasattr(ModelPolicy, "select_action") and callable(ModelPolicy.select_action):
            ModelPolicy.select_action = _wrap_select_action(ModelPolicy.select_action)
        elif hasattr(ModelPolicy, "select_action_index") and callable(ModelPolicy.select_action_index):
            ModelPolicy.select_action_index = _wrap_select_action(ModelPolicy.select_action_index)
    else:
        # 見つからない場合は沈黙
        pass

# ============================================================
# AlphaZeroMCTSPolicy 用: MCTS π の簡易トレースと終了時サマリ
# ============================================================

if os.getenv("POLICY_TRACE", "0") == "1":
    # 直近の数十ステップぶんだけ保持（最後にまとめて出す）
    _AZ_TRACE_MAX = 64
    _AZ_TRACE_RECENT = deque(maxlen=_AZ_TRACE_MAX)
    _AZ_TRACE_STATS = {
        "calls": 0,
        "pi_present": 0,
        "pi_missing": 0,
        "len_match": 0,
        "len_mismatch": 0,
    }

    try:
        _az_mod = __import__("pokepocketsim.policy.az_mcts_policy", fromlist=["AlphaZeroMCTSPolicy"])
        AlphaZeroMCTSPolicy = getattr(_az_mod, "AlphaZeroMCTSPolicy", None)
    except Exception:
        AlphaZeroMCTSPolicy = None

    if AlphaZeroMCTSPolicy is not None and hasattr(AlphaZeroMCTSPolicy, "select_action") and callable(AlphaZeroMCTSPolicy.select_action):

        def _wrap_az_select_action(orig_fn):
            def _wrapped(self, state, legal_actions):
                # 本来の選択をまず実行
                res = orig_fn(self, state, legal_actions)

                # 結果から π を取得（(action, pi) 形式 or self.last_pi 想定）
                pi = None
                if isinstance(res, tuple) and len(res) == 2:
                    _, pi = res
                else:
                    pi = getattr(self, "last_pi", None)

                la_len = len(legal_actions or [])
                has_pi = isinstance(pi, (list, tuple))
                pi_len = len(pi) if has_pi else 0
                pi_sum = float(sum(pi)) if has_pi else 0.0
                nz = int(sum(1 for v in pi if abs(v) > 1e-8)) if has_pi else 0

                # 統計更新
                _AZ_TRACE_STATS["calls"] += 1
                if has_pi:
                    _AZ_TRACE_STATS["pi_present"] += 1
                    if pi_len == la_len:
                        _AZ_TRACE_STATS["len_match"] += 1
                    else:
                        _AZ_TRACE_STATS["len_mismatch"] += 1
                else:
                    _AZ_TRACE_STATS["pi_missing"] += 1

                # 直近トレースの保存
                _AZ_TRACE_RECENT.append({
                    "la_len": la_len,
                    "pi_len": pi_len,
                    "pi_sum": pi_sum,
                    "nz": nz,
                })

                return res

            return _wrapped

        # AlphaZeroMCTSPolicy.select_action をラップ
        AlphaZeroMCTSPolicy.select_action = _wrap_az_select_action(AlphaZeroMCTSPolicy.select_action)

        def _dump_az_trace_summary():
            """プロセス終了時に AlphaZeroMCTSPolicy の π サマリを出力する。"""
            if _AZ_TRACE_STATS["calls"] == 0:
                return

            print(
                "[POLICY_TRACE_SUMMARY] AlphaZeroMCTSPolicy "
                f"calls={_AZ_TRACE_STATS['calls']} "
                f"pi_present={_AZ_TRACE_STATS['pi_present']} "
                f"pi_missing={_AZ_TRACE_STATS['pi_missing']} "
                f"len_match={_AZ_TRACE_STATS['len_match']} "
                f"len_mismatch={_AZ_TRACE_STATS['len_mismatch']}"
            )

            # 直近のトレース（最大 10 件）を詳細表示
            recent = list(_AZ_TRACE_RECENT)[-10:]
            for i, t in enumerate(recent, 1):
                print(
                    "[POLICY_TRACE_SUMMARY] "
                    f"#{i} la_len={t['la_len']} "
                    f"pi_len={t['pi_len']} "
                    f"sum={t['pi_sum']:.3f} "
                    f"nz={t['nz']}"
                )

        # プロセス終了時にサマリを出力
        atexit.register(_dump_az_trace_summary)
