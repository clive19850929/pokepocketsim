import json
import os

import numpy as np

import d3rlpy


# ============================================================
# Phase D CQL Q(s,a) ローダ＆評価ヘルパ（分離版）
# ============================================================

# Phase D CQL の成果物パス（必要なら環境変数で上書き可能）
PHASED_Q_LEARNABLE_PATH = os.getenv(
    "PHASED_Q_LEARNABLE_PATH",
    r"D:\date\phaseD_cql_run_p1\learnable_phaseD_cql.d3",
)
PHASED_Q_META_PATH = os.getenv(
    "PHASED_Q_META_PATH",
    r"D:\date\phaseD_cql_run_p1\train_phaseD_cql_meta.json",
)

# Phase D Q を使うかどうか（既定は True 相当）
USE_PHASED_Q: bool = bool(int(os.getenv("USE_PHASED_Q", "1")))

# 内部キャッシュ
_PHASED_Q_ALGO = None
_PHASED_Q_OBS_DIM: int | None = None
_PHASED_Q_ACTION_DIM: int | None = None
_PHASED_Q_LAST_FULL_OBS_VEC: list[float] | None = None

# π と Q の混合設定（既定は現行と同じ）
PHASED_Q_MIX_ENABLED: bool = bool(int(os.getenv("PHASED_Q_MIX_ENABLED", "1")))
PHASED_Q_MIX_LAMBDA: float = float(os.getenv("PHASED_Q_MIX_LAMBDA", "0.30"))
PHASED_Q_MIX_TEMPERATURE: float = float(os.getenv("PHASED_Q_MIX_TEMPERATURE", "1.0"))


class PhaseDQBundle:
    def __init__(
        self,
        learnable_path: str | None = None,
        meta_path: str | None = None,
        use_phased_q: bool | None = None,
        mix_enabled: bool | None = None,
        mix_lambda: float | None = None,
        mix_temperature: float | None = None,
    ):
        self.learnable_path = PHASED_Q_LEARNABLE_PATH if learnable_path is None else learnable_path
        self.meta_path      = PHASED_Q_META_PATH if meta_path is None else meta_path
        self.use_phased_q   = USE_PHASED_Q if use_phased_q is None else bool(use_phased_q)

        self.mix_enabled     = PHASED_Q_MIX_ENABLED if mix_enabled is None else bool(mix_enabled)
        self.mix_lambda      = PHASED_Q_MIX_LAMBDA if mix_lambda is None else float(mix_lambda)
        self.mix_temperature = PHASED_Q_MIX_TEMPERATURE if mix_temperature is None else float(mix_temperature)

        self._algo = None
        self._obs_dim: int | None = None
        self._action_dim: int | None = None

    def load_if_needed(self) -> None:
        global _PHASED_Q_LAST_FULL_OBS_VEC

        if self._algo is not None:
            return
        if not self.use_phased_q:
            return

        if not os.path.exists(self.learnable_path):
            print(f"[PhaseD Q] learnable not found: {self.learnable_path}", flush=True)
            return

        try:
            # d3rlpy v2 系では save_model(...) と対になるのは load_learnable(...)
            learnable = d3rlpy.load_learnable(self.learnable_path)

            # Learnable から CQL アルゴリズムを復元する
            try:
                from d3rlpy.algos import DiscreteCQL  # あなたが使っている CQL クラスに合わせて変更
                algo = DiscreteCQL.from_learnable(learnable)
            except Exception:
                # うまく復元できなかった場合はいったん learnable をそのまま保持
                algo = learnable
        except Exception as e:
            print(f"[PhaseD Q] failed to load .d3: {self.learnable_path} err={e!r}", flush=True)
            return

        self._algo = algo

        # メタから obs_dim / action_dim を拾っておく（チェック用）
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._obs_dim = int(meta.get("obs_dim", 0))
            self._action_dim = int(meta.get("action_dim", 0))
        except Exception:
            self._obs_dim = None
            self._action_dim = None

        print(
            f"[PhaseD Q] loaded CQL learnable from {self.learnable_path} "
            f"(obs_dim={self._obs_dim} action_dim={self._action_dim})",
            flush=True,
        )

    def q_values(
        self,
        obs_vec: list[float] | np.ndarray,
        action_candidates_vec: list[list[float]] | np.ndarray,
    ) -> np.ndarray | None:
        """
        1 状態 obs_vec と、その状態での action_candidates_vec (K x action_dim) を受け取り、
          Q(s, a_k) の配列 (K,) を返す。

        失敗時や use_phased_q=False のときは None を返す。
        """
        if not self.use_phased_q:
            return None

        self.load_if_needed()
        if self._algo is None:
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

        if self._obs_dim is not None and obs.shape[1] != self._obs_dim:
            print(
                f"[PhaseD Q] obs_dim mismatch: got={obs.shape[1]} "
                f"expected={self._obs_dim}"
            )

            allow_pad = bool(int(os.getenv("PHASED_Q_ALLOW_OBS_PAD", "0")))
            if not allow_pad:
                return None

            # ★暫定: mismatch をパディング/切り詰めで通す（デバッグ用）
            try:
                exp = int(self._obs_dim)
                got = int(obs.shape[1])
                if got < exp:
                    pad = np.zeros((obs.shape[0], exp - got), dtype=obs.dtype)
                    obs = np.concatenate([obs, pad], axis=1)
                else:
                    obs = obs[:, :exp]
                print(f"[PhaseD Q] ⚠️ obs adjusted for debug: {got} -> {obs.shape[1]}")
            except Exception:
                return None

        if self._action_dim is not None and cands.shape[1] != self._action_dim:
            print(
                f"[PhaseD Q] action_dim mismatch: got={cands.shape[1]} "
                f"expected={self._action_dim}"
            )
            return None

        # obs を候補数 K にブロードキャストして (K, obs_dim) にする
        num_actions = cands.shape[0]
        obs_batch = np.repeat(obs, num_actions, axis=0)

        # d3rlpy の CQL は predict_value(x, action) で Q(s,a) を返す
        try:
            values = self._algo.predict_value(obs_batch, cands)
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

    def mix_pi(self, pi, q_values):
        """
        Phase D の Q 値と元の π をブレンドして新しい方策分布を返す。

        - pi, q_values は同じ長さの 1 次元配列を想定
        - 何かおかしければ安全側として元の pi をそのまま返す
        """
        if not self.mix_enabled:
            return pi

        try:
            # numpy 配列に変換
            pi_arr = np.asarray(pi, dtype=np.float64)
            q_arr = np.asarray(q_values, dtype=np.float64)

            # 1 次元配列 & 次元一致チェック
            if pi_arr.ndim != 1 or q_arr.ndim != 1:
                return pi
            if pi_arr.shape[0] != q_arr.shape[0]:
                return pi

            # Q から softmax 分布を作る（温度付き）
            tau = max(float(self.mix_temperature), 1e-6)
            # 数値安定化のため最大値を引いてから softmax
            q_shift = q_arr - np.max(q_arr)
            q_soft = np.exp(q_shift / tau)
            s = float(q_soft.sum())
            if not np.isfinite(s) or s <= 0.0:
                return pi
            q_soft = q_soft / s

            # λ で π と Q-softmax を線形補間
            lam = min(max(float(self.mix_lambda), 0.0), 1.0)
            mixed = (1.0 - lam) * pi_arr + lam * q_soft
            try:
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


            # 念のためもう一度正規化
            msum = float(mixed.sum())
            if not np.isfinite(msum) or msum <= 0.0:
                return pi
            mixed = mixed / msum

            return mixed.astype(np.float32).tolist()
        except Exception:
            # 例外が出た場合は元の π をそのまま返して安全側に倒す
            return pi


_DEFAULT_BUNDLE = PhaseDQBundle()


def phaseD_q_load_if_needed() -> None:
    global _PHASED_Q_ALGO, _PHASED_Q_OBS_DIM, _PHASED_Q_ACTION_DIM

    _DEFAULT_BUNDLE.load_if_needed()

    _PHASED_Q_ALGO = _DEFAULT_BUNDLE._algo
    _PHASED_Q_OBS_DIM = _DEFAULT_BUNDLE._obs_dim
    _PHASED_Q_ACTION_DIM = _DEFAULT_BUNDLE._action_dim


def phaseD_q_evaluate(
    obs_vec: list[float] | np.ndarray,
    action_candidates_vec: list[list[float]] | np.ndarray,
) -> np.ndarray | None:
    return _DEFAULT_BUNDLE.q_values(obs_vec, action_candidates_vec)


def phaseD_mix_pi_with_q(pi, q_values):
    return _DEFAULT_BUNDLE.mix_pi(pi, q_values)
