# pokepocketsim/policy/d3rlpy_policy.py
from __future__ import annotations
from typing import List, Optional, Callable
import os
import numpy as np
import d3rlpy
import torch

from pokepocketsim.policy.state_encoder import StateEncoder


class ModelPolicy:
    """
    d3rlpy の連続 CQL モデルを使う方策（仕様に準拠）。
    - candidates（五要素配列）を共通エンコーダで学習時と同一の action_dim にエンコード
    - Q(s, a) を一括推論して argmax の“位置 i”を返す（返すのはインデックス）
    """

    def __init__(
        self,
        model_path: str,
        id_map_path: Optional[str] = None,   # 互換のため残すが未使用（連続では不要）
        scaler_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device

        # learnable / model の互換ローダ（d3rlpy バージョン差を吸収）
        self.algo, self._loader_api = self._load_d3rlpy_any(model_path, device=device)
        print(f"[ModelPolicy] loaded via {self._loader_api}: {model_path}")

        # state エンコーダ
        self.encoder = StateEncoder(scaler_path=scaler_path)

        # 外部から注入される “五要素→action_vec” の唯一エンコーダ
        self._enc_action: Optional[Callable[[List[int]], np.ndarray]] = None

        # 連続アクターでも sampling は使わない想定
        self.deterministic = True

        # GPUQServer クライアント（あれば優先）
        self._gpu_client = None

        # 学習時の action 次元（取得できる範囲で推定）
        self._expected_action_dim: Optional[int] = None
        try:
            impl = getattr(self.algo, "impl", None)
            if impl is not None:
                self._expected_action_dim = getattr(impl, "action_size", None)
                if self._expected_action_dim is None and hasattr(impl, "q_func"):
                    self._expected_action_dim = getattr(impl.q_func, "action_size", None)
        except Exception:
            # 取得不可でも動作は継続（ランタイムにチェックする）
            pass

        # デバッグ出力制御（最初の N 回だけ 1 行トレース）
        self._dbg_evaltrace = os.getenv("DEBUG_EVALTRACE", "1") == "1"
        self._dbg_first_n   = int(os.getenv("DEBUG_FIRST_N_EVALS", "5"))
        self._dbg_emitted   = 0
        # 直近の評価経路（GPU / LOCAL / POLICY(...) / FALLBACK_ZERO）
        self._last_eval_src = ""
        # 実戦の初回だけ強制でトレースを出したいとき（既定オン）
        self._dbg_force_match_trace = os.getenv("DEBUG_FORCE_MATCH_TRACE", "1") == "1"
        # set_action_encoder のラップで 1 回だけ出すためのフラグ
        self._enc_action_wrap_logged = 0
        # 追加: 状態・行動エンコードの可視化
        self._dbg_state_stats = os.getenv("DEBUG_STATE_STATS", "1") == "1"
        self._dbg_actenc_logged = False

    def _load_d3rlpy_any(self, path: str, device: str = "cpu"):
        """
        d3rlpy のバージョン差を吸収して learnable/model をロードする。
        1) d3rlpy.load_learnable を優先
        2) d3rlpy.load_model があれば試す
        3) 旧パッケージ内の別名 location も探索
        戻り値: (obj, api_name)
        """
        # まずは load_learnable を試す（device キーワードが無い版にも対応）
        try:
            loader = getattr(d3rlpy, "load_learnable", None)
            if loader is not None:
                try:
                    return loader(str(path), device=device), "d3rlpy.load_learnable"
                except TypeError:
                    return loader(str(path)), "d3rlpy.load_learnable"
        except Exception:
            pass

        # 次に load_model（存在する環境のみ）
        try:
            loader = getattr(d3rlpy, "load_model", None)
            if loader is not None:
                try:
                    return loader(str(path), device=device), "d3rlpy.load_model"
                except TypeError:
                    return loader(str(path)), "d3rlpy.load_model"
        except Exception:
            pass

        # モジュール内の別名（古い配置）も試す
        try:
            from importlib import import_module
            for modname, fname in [
                ("d3rlpy", "load_learnable"),
                ("d3rlpy.serialization", "load_learnable"),
                ("d3rlpy.serialize", "load_learnable"),
            ]:
                try:
                    mod = import_module(modname)
                    fn = getattr(mod, fname, None)
                    if fn is not None:
                        try:
                            return fn(str(path), device=device), f"{modname}.{fname}"
                        except TypeError:
                            return fn(str(path)), f"{modname}.{fname}"
                except Exception:
                    continue
        except Exception:
            pass

        raise RuntimeError(f"No compatible d3rlpy loader found for: {path}")

    # ===== 外部注入系 =====
    def attach_gpu_client(self, client) -> None:
        """GPUQServer クライアントを注入（あれば evaluate_q で優先）。"""
        self._gpu_client = client

    def set_action_encoder(self, fn: Callable[[List[int]], np.ndarray]) -> None:
        """五要素配列 → 学習時と同一次元の行動ベクトルに変換する関数を注入。"""
        def _wrapped(five: List[int]) -> np.ndarray:
            vec = fn(five)
            # 初回だけ、次元・ノルム・NaN/Inf・次元ミスマッチを可視化
            try:
                v = np.asarray(vec, dtype=np.float32).reshape(-1)
                if self._enc_action_wrap_logged < 1:
                    print(f"[ENCCHK] five={five[:5]} out_dim={v.size} norm={float(np.linalg.norm(v)):.3g} head10={v[:min(10, v.size)].round(6).tolist()}")
                    if self._expected_action_dim is not None and v.size != self._expected_action_dim:
                        print(f"[ENCWARN] dim_mismatch: encoded={v.size} expected={self._expected_action_dim}")
                    if np.isnan(v).any() or np.isinf(v).any():
                        print("[ENCWARN] contains NaN/Inf")
                    self._enc_action_wrap_logged += 1
            except Exception:
                pass
            return vec
        self._enc_action = _wrapped

    def set_deterministic(self, flag: bool) -> None:
        self.deterministic = bool(flag)

    def to_device(self, device: str) -> None:
        self.device = device

    # ===== ヘルパ =====
    @staticmethod
    def _ensure_2d(x: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
        return np.ascontiguousarray(arr)

    def _encode_candidates_to_action_batch(self, candidates: List[List[int]]) -> np.ndarray:
        """
        candidates: List[List[int]]（各要素は五要素配列想定。足りなければ 0 パディング）
        return    : (N, action_dim)
        """
        assert self._enc_action is not None, "action encoder not set (set_action_encoder を呼んでください)"
        if not candidates:
            return np.empty((0, 0), dtype=np.float32)

        a_list = []
        for c in candidates:
            five = (c[:5] + [0, 0, 0, 0, 0])[:5]     # 五要素に揃える
            a = self._enc_action(five)              # -> (A,)
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            a_list.append(a)

        a_mat = np.vstack(a_list).astype(np.float32, copy=False)  # (N, A)

        # 初回だけ行動エンコードのばらつきを出力（多様性が無いとQがフラット化しやすい）
        if not self._dbg_actenc_logged:
            try:
                col_std = np.std(a_mat, axis=0)
                diverse_dims = int(np.sum(col_std > 1e-6))
                print(
                    f"[ACTENC] N={a_mat.shape[0]} act_dim={a_mat.shape[1]} expected={self._expected_action_dim} "
                    f"diverse_dims={diverse_dims} col_std_head={col_std[:min(10, col_std.size)].round(6).tolist()}"
                )
            except Exception as _:
                pass
            self._dbg_actenc_logged = True

        # 期待 action 次元が分かっていれば明快に検証
        if self._expected_action_dim is not None and a_mat.shape[1] != self._expected_action_dim:
            raise ValueError(
                f"[ModelPolicy] action dim mismatch: encoded={a_mat.shape[1]} "
                f"expected={self._expected_action_dim}. "
                f"（学習時の ACTION_VEC_DIM と実行時のエンコーダ出力が一致していません）"
            )

        return a_mat

    # ===== Q(s,a) を一括評価 =====
    def evaluate_q(self, obs_batch: np.ndarray, act_batch: np.ndarray) -> np.ndarray:
            """
            obs_batch: (B, obs_dim)
            act_batch: (B, action_dim)
            return   : (B,)
            """
            obs_batch = self._ensure_2d(obs_batch, "obs_batch")
            act_batch = self._ensure_2d(act_batch, "act_batch")

            # バッチ次元の揃え（state が1行なら N にリピート）
            if obs_batch.shape[0] != act_batch.shape[0]:
                if obs_batch.shape[0] == 1:
                    obs_batch = np.repeat(obs_batch, act_batch.shape[0], axis=0)
                else:
                    raise ValueError(
                        f"[ModelPolicy] batch size mismatch: obs_batch={obs_batch.shape}, act_batch={act_batch.shape}"
                    )

            # action 次元の検証
            if self._expected_action_dim is not None and act_batch.shape[1] != self._expected_action_dim:
                raise ValueError(
                    f"[ModelPolicy] action dim mismatch: act_batch={act_batch.shape[1]} "
                    f"expected={self._expected_action_dim}"
                )

            # 追加: 候補行動の多様性を初回だけ可視化（エンジンレスでも見える）
            try:
                if self._dbg_evaltrace and self._dbg_emitted < self._dbg_first_n:
                    col_std = np.std(act_batch, axis=0)
                    diverse_dims = int(np.sum(col_std > 1e-6))
                    norms = np.linalg.norm(act_batch, axis=1)
                    print(
                        f"[ACTSTAT] N={act_batch.shape[0]} act_dim={act_batch.shape[1]} "
                        f"diverse_dims={diverse_dims} cand_norm_mean={float(np.mean(norms)):.3g} "
                        f"col_std_head={col_std[:min(10, col_std.size)].round(6).tolist()}"
                    )
            except Exception:
                pass

            # 評価経路ごとの 1 行トレース（最初の N 回のみ）
            def _trace(src: str, q_arr, note: str = "") -> None:
                try:
                    if self._dbg_evaltrace and self._dbg_emitted < self._dbg_first_n:
                        qv = np.asarray(q_arr, dtype=np.float32).reshape(-1)
                        flat = float(np.max(qv) - np.min(qv)) < 1e-8 if qv.size > 0 else True
                        print(
                            f"[EVALTRACE/{src}] obs={obs_batch.shape} act={act_batch.shape} "
                            f"q.shape={qv.shape} head={qv[:min(5, qv.size)].tolist()} var={float(np.var(qv) if qv.size else 0.0):.6g} "
                            f"flat={flat} {note}"
                        )
                        self._dbg_emitted += 1
                except Exception:
                    pass

            # --- 経路1: GPU クライアント（q_values） ---
            if self._gpu_client is not None:
                try:
                    q = self._gpu_client.q_values(obs_batch, act_batch)
                    _trace("GPU", q)
                    self._last_eval_src = "GPU"
                    q = np.asarray(q, dtype=np.float32).reshape(-1)
                    if np.isnan(q).any():
                        print("[EVALWARN] GPU q contains NaN -> falling back to LOCAL/BC", flush=True)
                        raise FloatingPointError("NaN in GPU q")
                    return q
                except Exception as _e:
                    print("[EVALWARN] GPU path failed:", _e)

            # --- 経路2: ローカルの predict_value（CQL等） ---
            try:
                with torch.no_grad():
                    q = self.algo.predict_value(obs_batch, act_batch)
                _trace("LOCAL", q)
                self._last_eval_src = "LOCAL"
                q = np.asarray(q, dtype=np.float32).reshape(-1)
                if np.isnan(q).any():
                    raise FloatingPointError("NaN in LOCAL q")
                return q
            except AttributeError:
                # algo に predict_value が無い（BCなど）→ 次の経路へ
                pass
            except Exception as _e:
                print("[EVALWARN] LOCAL predict_value failed:", _e)

            # --- 経路3: policy_score（BC等、Q頭が無い場合の擬似Q）---
            # 方針: 予測アクション a_hat を出し、候補 a との類似度をスコア化
            #   SCORE_MODE=cosine なら cos(a, a_hat)、既定は -||a - a_hat||^2
            try:
                score_mode = os.getenv("SCORE_MODE", "l2")
                with torch.no_grad():
                    # d3rlpy の actor は algo.predict で取り出せる想定（1件のみ）
                    a_hat = self.algo.predict(obs_batch)[0]
                a_hat = np.asarray(a_hat, dtype=np.float32).reshape(1, -1)
                if a_hat.shape[1] != act_batch.shape[1]:
                    raise ValueError(f"[policy_score] dim mismatch: pred={a_hat.shape[1]} cand={act_batch.shape[1]}")

                if score_mode.lower().startswith("cos"):
                    # コサイン類似度
                    denom = (np.linalg.norm(act_batch, axis=1) * np.linalg.norm(a_hat, axis=1)).reshape(-1) + 1e-9
                    q = (act_batch @ a_hat.T).reshape(-1) / denom
                    # a_hat と候補の距離/類似度を一度だけ可視化
                    try:
                        if self._dbg_evaltrace and self._dbg_emitted < self._dbg_first_n:
                            sims = q.copy().reshape(-1)
                            print(
                                f"[POLICY] mode=cos a_hat_norm={float(np.linalg.norm(a_hat)):.3g} "
                                f"sim_span={float(np.max(sims)-np.min(sims)):.3g} sim_head={sims[:min(5, sims.size)].round(6).tolist()}"
                            )
                    except Exception:
                        pass
                    _trace("POLICY(cos)", q)
                    self._last_eval_src = "POLICY(cos)"
                else:
                    # 負の二乗距離
                    diff = act_batch - a_hat
                    q = -np.sum(diff * diff, axis=1)
                    # a_hat と候補の距離を一度だけ可視化
                    try:
                        if self._dbg_evaltrace and self._dbg_emitted < self._dbg_first_n:
                            l2s = np.sqrt(np.sum(diff * diff, axis=1))
                            print(
                                f"[POLICY] mode=l2 a_hat_norm={float(np.linalg.norm(a_hat)):.3g} "
                                f"l2_span={float(np.max(l2s)-np.min(l2s)):.3g} l2_head={l2s[:min(5, l2s.size)].round(6).tolist()}"
                            )
                    except Exception:
                        pass
                    _trace("POLICY(l2)", q)
                    self._last_eval_src = "POLICY(l2)"

                q = np.asarray(q, dtype=np.float32).reshape(-1)
                if np.isnan(q).any():
                    raise FloatingPointError("NaN in policy_score")
                return q
            except Exception as _e:
                # ここまで落ちたら実行継続はできるが、集計器的には Evals=0 の原因になる
                print("[EVALFATAL] policy_score path failed:", _e)
                # 最低限の長さは返す（全ゼロは argmax=0 になる）
                self._last_eval_src = "FALLBACK_ZERO"
                return np.zeros((act_batch.shape[0],), dtype=np.float32)

    # ===== 中核：候補集合から i を選ぶ =====
    def select_from_candidates(self, state_obj: dict, candidates: List[List[int]]) -> int:
        assert self._enc_action is not None, "action encoder not set (set_action_encoderを呼んでください)"
        if not candidates:
            # legal_actions が空の監査ログ（最初の数回のみ冗長出力）
            if os.getenv("LOG_EMPTY_LEGAL", "1") == "1" and self._dbg_emitted < self._dbg_first_n:
                try:
                    me = state_obj.get("me", {})
                    benchN = len(me.get("bench", []) or [])
                    handN  = len(me.get("hand", []) or [])
                    print(f"[LEGAL] empty legal_actions: hand={handN} bench={benchN}")
                except Exception:
                    print("[LEGAL] empty legal_actions")
                self._dbg_emitted += 1
            return 0

        # 初回ターンだけ確実にトレースを出す（環境変数で無効化可）
        if self._dbg_force_match_trace and self._dbg_emitted < max(self._dbg_first_n, 1):
            try:
                self._dbg_evaltrace = True
                self._dbg_first_n = max(self._dbg_first_n, 1)
                # 候補の五要素と数を軽量プレビュー
                head = min(3, len(candidates))
                print(f"[CANDS] count={len(candidates)} five_head={ [ (c[:5] + [0,0,0,0,0])[:5] for c in candidates[:head] ] }")
            except Exception:
                pass

        # ✅ 状態は学習時と同じ入力のみでエンコード（legal_actionsは混ぜない）
        s = self.encoder.encode_state(state_obj).astype(np.float32, copy=False)
        s = s.reshape(1, -1)

        a_mat = self._encode_candidates_to_action_batch(candidates)
        q = self.evaluate_q(s, a_mat)

        # 状態ベクトルの統計を初回だけ出す（スケーラの二重適用/定数化の兆候を検知）
        if self._dbg_state_stats and self._dbg_emitted < self._dbg_first_n:
            try:
                v = s[0]
                print(
                    f"[STATE] dim={s.shape[1]} min={float(np.min(v)):.3g} max={float(np.max(v)):.3g} "
                    f"mean={float(np.mean(v)):.3g} std={float(np.std(v)):.3g} "
                    f"head10={v[:min(10, v.size)].round(6).tolist()}"
                )
            except Exception as _:
                pass

        idx = int(np.argmax(q))

        # デバッグ（必要時のみ）
        if os.getenv("DEBUG_CANDIDATES", "0") == "1":
            head = min(3, len(candidates))
            print(f"[CANDS] L={len(candidates)}  head five_ints={candidates[:head]}")
            print(f"[CANDS] head a_mat head10={[a_mat[i,:10].round(6).tolist() for i in range(head)]}")
            print(f"[CANDS] head Q={q[:head].round(6).tolist()}  -> pick idx={idx}")

        # フラット検査（環境変数で有効化。既定: 1 = 有効）
        if os.getenv("DEBUG_FLAT", "1") == "1":
            self._debug_check_flatness(s, a_mat, q)

        return idx

    def _debug_check_flatness(self, s: np.ndarray, a_mat: np.ndarray, q: np.ndarray) -> None:
        try:
            first_n = int(os.getenv("DEBUG_FIRST_N_FLAT", "5"))
            if getattr(self, "_dbg_flat_emitted", None) is None:
                self._dbg_flat_emitted = 0

            qv = np.asarray(q, dtype=np.float32).reshape(-1)
            if qv.size == 0:
                return
            q_span = float(np.max(qv) - np.min(qv))
            is_q_flat = q_span < 1e-7

            # 候補行動行列の多様性を調査
            amat = np.asarray(a_mat, dtype=np.float32)
            if amat.ndim != 2 or amat.shape[0] == 0:
                return
            col_std = np.std(amat, axis=0)                          # 次元ごとのばらつき
            diverse_dims = int(np.sum(col_std > 1e-6))              # 動いている次元数
            mean_pair_l2 = 0.0
            mean_pair_cos = 0.0
            if amat.shape[0] >= 2:
                i0, i1 = 0, 1
                v0, v1 = amat[i0], amat[i1]
                diff = v0 - v1
                mean_pair_l2 = float(np.sqrt(np.sum(diff * diff)))
                denom = (np.linalg.norm(v0) * np.linalg.norm(v1)) + 1e-9
                mean_pair_cos = float(np.dot(v0, v1) / denom)

            # policy_score を使う時は a_hat との距離も確認（学習器が定数化していないか）
            a_hat_info = ""
            try:
                if os.getenv("SCORE_MODE", "").lower() in ("", "l2", "cosine", "cos"):
                    with torch.no_grad():
                        a_hat = self.algo.predict(np.asarray(s, dtype=np.float32))[0]
                    a_hat = np.asarray(a_hat, dtype=np.float32).reshape(1, -1)
                    if a_hat.shape[1] == amat.shape[1]:
                        diff = amat - a_hat
                        l2s = np.sqrt(np.sum(diff * diff, axis=1))
                        l2_span = float(np.max(l2s) - np.min(l2s))
                        a_hat_info = f" a_hat_l2_span={l2_span:.3e} a_hat_l2_head={l2s[:min(3, l2s.size)].round(6).tolist()}"
                    else:
                        a_hat_info = f" a_hat_dim_mismatch pred={a_hat.shape[1]} cand={amat.shape[1]}"
            except Exception as _:
                pass

            # 1行要約
            print(
                f"[FLATCHK] q_span={q_span:.3e} is_q_flat={is_q_flat} "
                f"cands={amat.shape[0]} act_dim={amat.shape[1]} diverse_dims={diverse_dims} "
                f"pair_l2={mean_pair_l2:.3e} pair_cos={mean_pair_cos:.6f}{a_hat_info}"
            )

            # フラットのとき、最初の数回だけ詳細を出す
            if is_q_flat and self._dbg_flat_emitted < first_n:
                head = min(3, amat.shape[0])
                print(f"[FLATDET] q_head={qv[:head].round(6).tolist()}")
                print(f"[FLATDET] col_std_head={col_std[:min(10, col_std.size)].round(6).tolist()} "
                      f"(>1e-6 dims = {diverse_dims})")
                # 候補先頭3本の先頭10次元
                rows = [amat[i, :min(10, amat.shape[1])].round(6).tolist() for i in range(head)]
                print(f"[FLATDET] a_mat_rows_head10={rows}")
                self._dbg_flat_emitted += 1

        except Exception as e:
            print("[FLATERR]", e)

    # ===== 互換API =====
    def select_action_index(self, state_obj: dict, candidates: List[List[int]], **_) -> int:
        """index を返す（仕様：返すのは i）。"""
        return self.select_from_candidates(state_obj, candidates)

    def select_action(self, state_obj: dict, candidates: List[List[int]], **_) -> int:
        """後方互換：index を返す。"""
        return self.select_from_candidates(state_obj, candidates)

    def choose_action(self, state_obj: dict, candidates: List[List[int]], **_) -> List[int]:
        """行動そのものが欲しい場合。"""
        idx = self.select_from_candidates(state_obj, candidates)
        return candidates[idx]
