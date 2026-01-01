#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
d3rlpy_dataset_all.npz から CQL を学習し、run_p1 に成果物を保存します。

保存物:
  - learnable.d3（d3rlpy v2 の学習一式）
  - model_final.d3（weights-only）
  - scaler.npz（z-score の mean/std）
  - train_meta.json（学習時の指紋と監査情報）
  - obs_mask.npy（観測のゼロ分散次元マスク）

方針（A〜Eの要件を満たすプリセット）:
  A. 安定化リラン（ep=1〜2 は強めに保守：フェーズ別に設定）
     • Phase1: cw=50 / nas=12 / τ=0.005 / critic_dropout=0.50
               actor_lr=1e-5, critic_lr=2e-6, temp_lr=3e-4, alpha_lr=3e-5
               soft_q_backup=True, max_q_backup=False
     • Phase2: cw=30 / nas=20 / τ=0.004 / critic_dropout=0.40
               actor_lr=2e-5, critic_lr=1e-5, temp_lr=3e-4
     • Phase3: cw=20 / nas=24 / τ=0.003 / critic_dropout=0.40
               actor_lr=3e-5, critic_lr=1e-5, temp_lr=1e-4
  B. 観測は d3rlpy の StandardScaler を使用（学習側で統計を推定）
     • scaler.npz にも mean/std を保存（推論系と整合）
     • 行動は MinMaxActionScaler で明示クリップ（min=0, max=1 を保証）
  C. ポリシー揺れ抑制
     • initial_temperature=0.10, alpha_threshold=0.70
  D. 自動調整ルール（各エポック末の評価で機械的に反映）
     • TD スパイク（z95>3）→ critic_lr *=0.7, cw +=2, nas=min(nas+4, 32)
     • Qスケール大（ratio>2.0）→ cw +=2, critic_lr *=0.8
     • Qスケール小（ratio<0.5）→ cw=max(cw-1, 2), actor_lr *=1.2
     • Δpolicy>0.06 → actor_lr *=0.7, temp_lr *=0.7
     • 安定域達成（td_p95≤3 & 0.7≤ratio≤1.5 が2連続）→ cw-=2, nas=max(nas-4, 8)
  E. チェックポイント選定（3条件を満たすものを優先保存）
     1) td_p95 ≤ 3.5
     2) value_scale ratio ∈ [0.7, 1.5]
     3) policy_std ≥ 0.10 かつ Δpolicy ≤ 0.06
"""

# ===== Speed presets（最上部で設定）=====
import os
# CPU スレッドの過剰並列を抑制（Windowsで特に効く）
try:
    from pokepocketsim.action import ACTION_SCHEMAS
except Exception:
    ACTION_SCHEMAS = {}
    print("[WARN] pokepocketsim.action.ACTION_SCHEMAS を読み込めませんでした（空辞書で継続します）。")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 速い検証用の軽量モード（FAST_MODE=1 で有効）
FAST_MODE = bool(int(os.getenv("FAST_MODE", "1")))  # 1=軽量, 0=通常

from datetime import datetime
import signal, sys
import shutil
import numpy as np
import torch
import d3rlpy   # v2 を前提
import json
import csv

# 既定の割引率（PBRSと整合）
DEFAULT_GAMMA = float(os.getenv("CQL_GAMMA", "0.99"))

# === AE 条件・上限（自動チューニング目標） ===
AE_TD_MAX    = 3.5
AE_VS_LO     = 0.7
AE_VS_HI     = 1.5
AE_PSTD_MIN  = 0.10
AE_DPOL_MAX  = 0.06
CW_MAX       = 48.0 if FAST_MODE else 96.0   # FAST は保守の行き過ぎを抑制

# === 収束寄りの再開プリセット（推奨） =========================
# ・freeze_actor_steps を 5,000–10,000（ここでは 7,500）に設定
# ・actor_update_interval=4 で actor 更新を間引く
# ・critic_dropout=0.60 に固定（過剰な値発散を抑える）
# ・Q/TD をやさしくクリップ（50）
# ・温度の学習率は一旦 1e-4 に下げる（過度な温度暴走の抑制）
# ・soft_q_backup=True を固定（値スケール比 r が正常域に戻るまで max に戻さない）
RECOVERY_PRESET = {
    "freeze_actor_steps": 0,        # ★ actor を最初から動かす（勾配枯渇を回避）
    "actor_update_interval": 2,
    "critic_dropout": 0.40,         # 0.60 → 0.40（Qの表現力を戻す）
    "q_clip_abs": (30.0 if FAST_MODE else 50.0),
    "td_clip_abs": (30.0 if FAST_MODE else 50.0),

    "conservative_weight": 20.0,    # 32.0 → 20.0（CQLの保守圧を緩める）
    "n_action_samples": 24,
    "critic_lr": 2.0e-6,
    "temp_lr": 1.0e-4,

    "soft_q_backup": True,
    "max_q_backup": False,
}

# ===[ 追加 ]========================================================
# フェーズ自動ゲート（Phase1を安定まで継続し、安定で昇格／不安定で降格）
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
import math

Phase = Literal["phase1_stabilize", "phase2_growth", "phase3_converge"]

@dataclass
class EpochMetrics:
    epoch: int
    z95: float
    td_p95: float
    q_p50_ratio: float   # |Q|_p50 / base_p50
    delta_policy: float  # Δpolicy
    policy_std: float

def _isfinite(*xs):
    return all(math.isfinite(x) for x in xs)

class StabilityGate:
    def __init__(
        self,
        min_epochs_phase1: int = 3,
        consecutive_ok: int = 2,
        consecutive_ng: int = 2,
        q_ratio_low: float = 0.9,
        q_ratio_high: float = 1.8,
        z95_max: float = 3.2,
        dpolicy_max: float = 0.06,
        pstd_min: float = 0.05,
        td_p95_eps_trend: float = +0.02,
    ):
        self.min_epochs_phase1 = min_epochs_phase1
        self.consecutive_ok = consecutive_ok
        self.consecutive_ng = consecutive_ng
        self.q_ratio_low = q_ratio_low
        self.q_ratio_high = q_ratio_high
        self.z95_max = z95_max
        self.dpolicy_max = dpolicy_max
        self.pstd_min = pstd_min
        self.td_p95_eps_trend = td_p95_eps_trend
        self._ok_streak = 0
        self._ng_streak = 0

    def reset_streaks(self):
        self._ok_streak = 0
        self._ng_streak = 0

    def check_stable_pair(self, prev: EpochMetrics, cur: EpochMetrics) -> bool:
        if not _isfinite(prev.z95, prev.td_p95, prev.q_p50_ratio, prev.delta_policy, prev.policy_std,
                         cur.z95, cur.td_p95, cur.q_p50_ratio, cur.delta_policy, cur.policy_std):
            return False
        cond_z  = (cur.z95 <= self.z95_max)
        cond_q  = (self.q_ratio_low <= cur.q_p50_ratio <= self.q_ratio_high)
        cond_dp = (cur.delta_policy <= self.dpolicy_max)
        cond_ps = (cur.policy_std >= self.pstd_min)
        cond_td = (cur.td_p95 <= prev.td_p95 * (1.0 + self.td_p95_eps_trend))
        return cond_z and cond_q and cond_dp and cond_ps and cond_td

    def update_streaks(self, is_ok: bool):
        if is_ok:
            self._ok_streak += 1
            self._ng_streak = 0
        else:
            self._ng_streak += 1
            self._ok_streak = 0

    @property
    def ok_ready(self) -> bool:
        return self._ok_streak >= self.consecutive_ok

    @property
    def ng_trigger(self) -> bool:
        return self._ng_streak >= self.consecutive_ng


class PhaseScheduler:
    """
    Phase1: 最小エポック達成 & 安定連続OK → Phase2
    Phase2: 安定連続OK → Phase3 ／ 不安定連続 → Phase1
    Phase3: 不安定連続 → Phase2
    """
    def __init__(self):
        self.phase: Phase = "phase1_stabilize"
        self.gate = StabilityGate(
            min_epochs_phase1=2,   # ← 3 → 2（早めに抜けやすく）
            consecutive_ok=2,
            consecutive_ng=2,
            z95_max=3.2,          # ← 3.0 → 3.2（初期のノイズを少し許容）
        )
        self.history: List[EpochMetrics] = []

    def _phase_params(self, phase: Phase) -> Dict:
        if phase == "phase1_stabilize":
            return dict(
                conservative_weight=26.0, n_action_samples=12,   # 仕様A
                actor_lr=1.0e-5, critic_lr=2.0e-6,
                temp_lr=3.0e-4, alpha_lr=3.0e-5,
                initial_temperature=0.16, initial_alpha=0.10,    # 仕様C
                tau=0.005,                                        # 仕様A
                critic_dropout=0.40,
                actor_update_interval=(4 if FAST_MODE else 2),  # ← FAST は間引きを強めに
                batch_size=128,
                q_clip_abs=100.0, td_clip_abs=100.0,
                **_force_backup_mode("soft"),                     # Phase1は soft backup（仕様A）
            )

        if phase == "phase2_growth":
            return dict(
                conservative_weight=30.0, n_action_samples=20,   # 仕様A
                actor_lr=1.5e-5, critic_lr=5.0e-6,
                temp_lr=3.0e-4, alpha_lr=3.5e-5,
                initial_temperature=0.10, initial_alpha=0.12,
                tau=0.004,                                        # 仕様A
                critic_dropout=0.45,
                actor_update_interval=4,
                batch_size=128,
                **_force_backup_mode("max"),
            )

        return dict(  # phase3_converge
            conservative_weight=20.0, n_action_samples=24,       # 仕様A
            actor_lr=2.0e-5, critic_lr=8.0e-6,
            temp_lr=1.0e-4, alpha_lr=4.0e-5,
            initial_temperature=0.10, initial_alpha=0.14,
            tau=0.003,                                           # 仕様A
            critic_dropout=0.40,
            actor_update_interval=2,
            batch_size=192,
            **_force_backup_mode("max"),
        )


    def get_current_params(self) -> Dict:
        return self._phase_params(self.phase)

    def _maybe_promote_from_phase1(self) -> Optional[Phase]:
        if len(self.history) < 2:
            return None
        prev, cur = self.history[-2], self.history[-1]
        if cur.epoch < self.gate.min_epochs_phase1:
            self.gate.reset_streaks()
            return None
        is_ok = self.gate.check_stable_pair(prev, cur)
        self.gate.update_streaks(is_ok)
        if self.gate.ok_ready:
            self.gate.reset_streaks()
            return "phase2_growth"
        return None

    def _maybe_move_phase2_or_3(self) -> Optional[Phase]:
        if len(self.history) < 2:
            return None
        prev, cur = self.history[-2], self.history[-1]
        is_ok = self.gate.check_stable_pair(prev, cur)
        self.gate.update_streaks(is_ok)
        if self.gate.ng_trigger:
            self.gate.reset_streaks()
            if self.phase == "phase3_converge":
                return "phase2_growth"
            elif self.phase == "phase2_growth":
                return "phase1_stabilize"
            return None
        if self.phase == "phase2_growth" and self.gate.ok_ready:
            self.gate.reset_streaks()
            return "phase3_converge"
        return None

    def update(self, metrics: EpochMetrics) -> Dict:
        self.history.append(metrics)
        new_phase: Optional[Phase] = None
        if self.phase == "phase1_stabilize":
            new_phase = self._maybe_promote_from_phase1()
        else:
            new_phase = self._maybe_move_phase2_or_3()
        if new_phase and new_phase != self.phase:
            self.phase = new_phase
            self.gate.reset_streaks()
        params = self._phase_params(self.phase)
        return {"phase": self.phase, "params": params}
# ===[ /追加 ]======================================================


class HistoryLogger:
    """
    1) JSONL: そのままの辞書を1行ずつ追記（完全ログ）
       <RUN_DIR>/full_log.jsonl

    2) CSV: よく見る主要項目を整形して追記（集計しやすい）
       <RUN_DIR>/full_log.csv
    """
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.run_dir, "full_log.jsonl")
        self.csv_path   = os.path.join(self.run_dir, "full_log.csv")

        self.csv_fields = [
            "epoch", "step", "score", "ts",
            "m_td_p95", "m_td_z95", "m_vs_ratio", "m_vs_ratio_ema",
            "m_dpolicy", "m_dpolicy_ema", "m_policy_std", "m_policy_std_ema",
            "m_beh_p50", "m_beh_p90", "m_temp",
            "c_phase", "c_cw", "c_nas", "c_actor_lr", "c_critic_lr",
            "c_temp_lr", "c_alpha_lr", "c_aui", "c_tau",
            "c_soft_q_backup", "c_max_q_backup", "c_batch_size",
            "note"
        ]

        self._csv_file_exists = os.path.exists(self.csv_path)
        self._csv_fh = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=self.csv_fields)
        if not self._csv_file_exists:
            self._csv_writer.writeheader()

        self._jsonl_fh = open(self.jsonl_path, "a", encoding="utf-8")

    def write(self, *, epoch: int, step: int, metrics: dict, current: dict,
              score: float, ts: str, note: str = ""):
        rec = {
            "epoch": int(epoch),
            "step": int(step),
            "score": float(score),
            "ts": ts,
            "metrics": metrics or {},
            "current": current or {},
            "note": note,
        }
        self._jsonl_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._jsonl_fh.flush()

        m = metrics or {}
        c = current or {}
        row = {
            "epoch": epoch,
            "step": step,
            "score": score,
            "ts": ts,
            "m_td_p95":          m.get("td_p95"),
            "m_td_z95":          m.get("td_z95"),
            "m_vs_ratio":        m.get("vs_ratio"),
            "m_vs_ratio_ema":    m.get("vs_ratio_ema"),
            "m_dpolicy":         m.get("dpolicy"),
            "m_dpolicy_ema":     m.get("dpolicy_ema"),
            "m_policy_std":      m.get("policy_std"),
            "m_policy_std_ema":  m.get("policy_std_ema"),
            "m_beh_p50":         m.get("beh_p50"),
            "m_beh_p90":         m.get("beh_p90"),
            "m_temp":            m.get("temp"),
            "c_phase":           c.get("name") or c.get("phase"),
            "c_cw":              c.get("conservative_weight"),
            "c_nas":             c.get("n_action_samples"),
            "c_actor_lr":        c.get("actor_lr"),
            "c_critic_lr":       c.get("critic_lr"),
            "c_temp_lr":         c.get("temp_lr"),
            "c_alpha_lr":        c.get("alpha_lr"),
            "c_aui":             c.get("actor_update_interval"),
            "c_tau":             c.get("tau"),
            "c_soft_q_backup":   c.get("soft_q_backup"),
            "c_max_q_backup":    c.get("max_q_backup"),
            "c_batch_size":      c.get("batch_size"),
            "note":              note,
        }
        self._csv_writer.writerow(row)
        self._csv_fh.flush()

    def close(self):
        try:
            self._csv_fh.close()
        except Exception:
            pass
        try:
            self._jsonl_fh.close()
        except Exception:
            pass

class TrainLoggerMixin:
    """
    既存の Trainer / Runner クラスに混ぜて使うための Mixin。
    - __init__ で run_dir を渡して初期化
    - self._log_progress(...) を学習エポック末に呼ぶ
    - self.write_summary(...) を最後に呼ぶ
    """
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.hist_logger = HistoryLogger(self.run_dir)
        self._last_epoch_snapshot = {}
        self._epochs_all = []  # 全エポック履歴（summary.json に同梱）

    @staticmethod
    def _jsonable(v):
        import numpy as np
        if isinstance(v, (int, float, np.integer, np.floating, bool, type(None))):
            try:
                return float(v)
            except Exception:
                return v
        return v

    def _log_progress(self, epoch, step, metrics, current, score,
                      tag="epoch_end", note=""):
        # メモリ保持（最後に write_summary で利用）
        self._last_epoch_snapshot.clear()
        self._last_epoch_snapshot.update({
            "tag": tag,
            "epoch": int(epoch),
            "step": int(step),
            "metrics": {k: self._jsonable(v) for k, v in (metrics or {}).items()},
            "current": {k: self._jsonable(v) for k, v in (current or {}).items()},
            "score": self._jsonable(score),
            "ts": datetime.now().isoformat(timespec="seconds"),
            "note": note,
        })
        # 追加: 全エポック履歴にも追記（CSVに依存せず summary.json へ同梱）
        try:
            self._epochs_all.append({
                "tag": tag,
                "epoch": int(epoch),
                "step": int(step),
                "metrics": {k: self._jsonable(v) for k, v in (metrics or {}).items()},
                "current": {k: self._jsonable(v) for k, v in (current or {}).items()},
                "score": self._jsonable(score),
                "ts": self._last_epoch_snapshot["ts"],
                "note": note,
            })
        except Exception:
            pass
        # 永続化（完全ログ）
        self.hist_logger.write(
            epoch=epoch,
            step=step,
            metrics=(metrics or {}),
            current=(current or {}),
            score=score,
            ts=self._last_epoch_snapshot["ts"],
            note=note,
        )

    def get_last_snapshot(self):
        return dict(self._last_epoch_snapshot)

    # 学習完了時に要約を書き出したい場合（任意）
    def write_summary(self, *, win_k: int, win_n: int,
                      best_epoch: int, best_step: int,
                      best_epoch_ae: int, best_step_ae: int):
        import math, json as _json

        def _confint_95_binom(k: int, n: int):
            if n <= 0:
                return (0.0, 0.0, 0.0)
            p = k / n
            se = math.sqrt(p * (1 - p) / n)
            lo, hi = max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
            return (p, lo, hi)

        last = self.get_last_snapshot()
        cur = (last.get("current") or {})
        met = (last.get("metrics") or {})
        p, lo, hi = _confint_95_binom(win_k, win_n)

        summary = {
            "win_rate": {"k": win_k, "n": win_n, "p": p, "ci95": [lo, hi]},
            "best": {
                "score_min": float(last.get("score") or float("nan")),
                "epoch": int(best_epoch or 0),
                "step": int(best_step or 0),
                "ae_epoch": int(best_epoch_ae or 0),
                "ae_step": int(best_step_ae or 0),
            },
            "final_metrics": {
                "td_p95": float(met.get("td_p95") or float("nan")),
                "vs_ratio_ema": float(met.get("vs_ratio_ema") or float("nan")),
                "dpolicy_ema": float(met.get("dpolicy_ema") or float("nan")),
                "policy_std_ema": float(met.get("policy_std_ema") or float("nan")),
                "beh_p90": float(met.get("beh_p90") or float("nan")),
                "td_z95": float(met.get("td_z95") or float("nan")),
            },
            "final_hparams": {
                "phase": cur.get("name") or cur.get("phase") or "unknown",
                "conservative_weight": float(cur.get("conservative_weight") or float("nan")),
                "n_action_samples": int(cur.get("n_action_samples") or 0),
                "actor_lr": float(cur.get("actor_lr") or float("nan")),
                "critic_lr": float(cur.get("critic_lr") or float("nan")),
                "temp_lr": float(cur.get("temp_lr") or float("nan")),
                "alpha_lr": float(cur.get("alpha_lr") or float("nan")),
                "actor_update_interval": int(cur.get("actor_update_interval") or 0),
                "tau": float(cur.get("tau") or float("nan")),
                "soft_q_backup": bool(cur.get("soft_q_backup", False)),
                "max_q_backup": bool(cur.get("max_q_backup", False)),
                "batch_size": int(cur.get("batch_size") or 0),
            },
            # ここから追加: 全エポック履歴をサマリーに同梱
            "epochs": (list(self._epochs_all) or None),
            "epochs_meta": {
                "count": int(len(self._epochs_all)),
            },
        }

        out_json = os.path.join(self.run_dir, "summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            _json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

        print("\n=== SUMMARY (Final Only) ===")
        print(f"WinRate(raw terminal): {win_k}/{win_n} = {p:.3f}  [95% CI: {lo:.3f}–{hi:.3f}]")
        print(f"Best score@epoch={summary['best']['epoch']} step={summary['best']['step']} | "
              f"AE-best@epoch={summary['best']['ae_epoch']} step={summary['best']['ae_step']}")
        print("Final metrics:",
              f"td_p95={summary['final_metrics']['td_p95']:.3f},",
              f"vs_ratio_ema={summary['final_metrics']['vs_ratio_ema']:.2f},",
              f"dpolicy_ema={summary['final_metrics']['dpolicy_ema']:.4f},",
              f"policy_std_ema={summary['final_metrics']['policy_std_ema']:.4f},",
              f"beh_p90={summary['final_metrics']['beh_p90']:.4f}")
        print("Final hparams:",
              f"phase={summary['final_hparams']['phase']},",
              f"cw={summary['final_hparams']['conservative_weight']},",
              f"nas={summary['final_hparams']['n_action_samples']},",
              f"aui={summary['final_hparams']['actor_update_interval']},",
              f"lr(a/c/t)={summary['final_hparams']['actor_lr']:.1e}/"
              f"{summary['final_hparams']['critic_lr']:.1e}/"
              f"{summary['final_hparams']['temp_lr']:.1e},",
              f"tau={summary['final_hparams']['tau']},",
              f"maxQ={summary['final_hparams']['max_q_backup']}")
        print(f"summary.json written: {out_json}")

    def close_logger(self):
        try:
            self.hist_logger.close()
        except Exception:
            pass

try:
    from artifact_fingerprint import (  # 学習時指紋（任意）
        compute_schema_sha, compute_vocab_sha, compute_scaler_sha
    )
except Exception:
    compute_schema_sha = compute_vocab_sha = compute_scaler_sha = None
    print("[WARN] artifact_fingerprint を読み込めませんでした（指紋機能は無効化）。")

from sklearn.model_selection import train_test_split
from d3rlpy.algos import CQLConfig
from d3rlpy.datasets import MDPDataset
from d3rlpy.logging import CombineAdapterFactory, FileAdapterFactory, TensorboardAdapterFactory
from d3rlpy.metrics import AverageValueEstimationEvaluator
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
try:
    from d3rlpy.models.q_functions import MinQFunctionFactory  # 一部版で提供
    _q_factory = MinQFunctionFactory(share_encoder=False)
except Exception:
    _q_factory = MeanQFunctionFactory(share_encoder=False)
from d3rlpy.models.optimizers import AdamFactory
# --- d3rlpy の配置差分に強い動的インポート（PyrightのreportMissingImportsも回避） ---
import importlib, contextlib
from typing import Any, Optional

MinMaxActionScaler: Optional[Any] = None
StandardScaler:    Optional[Any] = None

try:
    # v2: まず __init__ 直下を試す（2.8.1 では StandardScaler が無いため StandardObservationScaler を別名で受ける）
    from d3rlpy.preprocessing import MinMaxActionScaler as _MM, StandardObservationScaler as _SS  # type: ignore
    MinMaxActionScaler, StandardScaler = _MM, _SS
except Exception:
    # ランタイムで存在するものを順に探す（静的importを避けてPyright警告を出さない）
    try:
        _pre = importlib.import_module("d3rlpy.preprocessing")
    except Exception:
        _pre = None

    if _pre is not None:
        MinMaxActionScaler = getattr(_pre, "MinMaxActionScaler", None)
        # まずは新名（StandardObservationScaler）を探し、無ければ旧名（StandardScaler）を探す
        StandardScaler     = getattr(_pre, "StandardObservationScaler", None) or getattr(_pre, "StandardScaler", None)

    if MinMaxActionScaler is None:
        with contextlib.suppress(Exception):
            MinMaxActionScaler = importlib.import_module("d3rlpy.preprocessing.action_scalers").MinMaxActionScaler  # type: ignore

    if StandardScaler is None:
        # 新名の配置（scalers.StandardObservationScaler）→ 旧名（scalers.StandardScaler）の順で探す
        with contextlib.suppress(Exception):
            StandardScaler = importlib.import_module("d3rlpy.preprocessing.scalers").StandardObservationScaler  # type: ignore
    if StandardScaler is None:
        with contextlib.suppress(Exception):
            StandardScaler = importlib.import_module("d3rlpy.preprocessing.scalers").StandardScaler  # type: ignore

    if (MinMaxActionScaler is None) or (StandardScaler is None):
        print("[WARN] MinMaxActionScaler/StandardScaler をインポートできません。d3rlpy の版を確認してください。")



# PyTorch スレッドと TF32（Ampere+ で高速化、精度影響は小）
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass
# ========================================

# 保存先（ai vs ai と合わせる）
RUN_DIR = r"d3rlpy_logs/run_p1"   # ← p2 を学習するときは run_p2 に変更
ID_MAP_SRC = "id_action_map.json" # ある場合のみコピー（任意）
ACTION_TYPES_SRC = "action_types.json"  # ある場合のみコピー（one-hot 母集合の固定）

# === zero-variance auto drop settings ===
OBS_ZERO_VAR_EPS = float(os.getenv("OBS_ZERO_VAR_EPS", "1e-12"))  # ゼロ分散判定しきい値
OBS_MASK_PATH = os.getenv("OBS_MASK_PATH", os.path.join(RUN_DIR, "obs_mask.npy"))  # マスク保存先

# しきい値 <= 0 のときはドロップ無効（マスク=全True）
OBS_ZERO_VAR_DISABLE = (OBS_ZERO_VAR_EPS <= 0.0)

TYPE_SCHEMAS = { 7: ["stack_index"] }  # 進化(EVOLVE)
MAX_ARGS = 3

def try_export_learnable(algo, out_path: str) -> bool:
    """学習済みインスタンスの一式を保存（v2 は save()、一部版は save_learnable()）。"""
    try:
        if hasattr(algo, "save"):
            algo.save(out_path)  # v2 正式API
            print(f"learnable を保存: {out_path}")
            return True
        if hasattr(algo, "save_learnable"):
            algo.save_learnable(out_path)  # 互換API
            print(f"learnable を保存: {out_path}")
            return True
        print("この d3rlpy 版には save / save_learnable がないため learnable.d3 は作成しません。")
        return False
    except Exception as e:
        print(f"learnable 保存をスキップ（{type(e).__name__}: {e}）")
        return False

def _ensure_v2():
    ver = getattr(d3rlpy, "__version__", "0")
    if not str(ver).startswith("2"):
        raise RuntimeError(f"d3rlpy v2.x が必要です（現在: {ver}）。仮想環境を確認してください。")

def _size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return -1.0

# === [追加] backup の一貫性ユーティリティ =========================
def _force_backup_mode(prefer: str = "max"):
    """
    d3rlpy の CQL は soft_q_backup と max_q_backup が排他。
    prefer='max' なら max_q_backup=True / soft_q_backup=False を全域で強制。
    """
    prefer = (prefer or "max").lower()
    if prefer not in ("max", "soft"):
        prefer = "max"
    soft = (prefer == "soft")
    return {"soft_q_backup": bool(soft), "max_q_backup": bool(not soft)}
# ================================================================

# ▼▼ d3rlpy互換 + Torch/Numpy両対応の PostScaledObservationScaler（差し替え） ▼▼
class PostScaledObservationScaler:
    """
    d3rlpy の ObservationScaler ラッパー。
    - built フラグを持ち、fit/fit_with_transition_picker は一度だけ実行（idempotent）。
    - 2回目以降に d3rlpy から呼ばれても no-op にして Assertion を回避。
    - transform / reverse_transform は形状ガード＆Torch/NumPy両対応。
    """

    def __init__(self, base, clip_min=-5.0, clip_max=5.0, post=None):
        self.base = base
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.post = post
        self._post_warned = False
        self.built = False  # d3rlpy 互換のフラグ

    # --- d3rlpy の再フィット要求に対しても idempotent に振る舞う ---
    def fit(self, *args, **kwargs):
        if self.built:
            return
        if hasattr(self.base, "fit"):
            self.base.fit(*args, **kwargs)
        self.built = True

    def fit_with_transition_picker(self, *args, **kwargs):
        if self.built:
            return
        if hasattr(self.base, "fit_with_transition_picker"):
            self.base.fit_with_transition_picker(*args, **kwargs)
        else:
            # フォールバック（fit があれば呼ぶ）
            if hasattr(self.base, "fit"):
                self.base.fit(*args, **kwargs)
        self.built = True

    def build_with_dataset(self, dataset):
        # d3rlpy が呼ぶ場合に備えた補助（無い実装もあるため存在チェック）
        if hasattr(self.base, "build_with_dataset"):
            out = self.base.build_with_dataset(dataset)
            self.built = bool(getattr(self.base, "built", True))
            return out
        self.built = True
        return self

    def reset(self):
        """再フィットを許可したい場合に手動で呼ぶ"""
        self.built = False
        if hasattr(self.base, "reset"):
            try:
                self.base.reset()
            except Exception:
                pass

    def transform(self, x):
        y = self.base.transform(x)

        # Torch か NumPy かを判定
        try:
            import torch
            is_tensor = isinstance(y, torch.Tensor)
        except Exception:
            is_tensor = False

        if is_tensor:
            y = y.clamp(min=self.clip_min, max=self.clip_max)
            if self.post is not None:
                post = torch.as_tensor(self.post, dtype=y.dtype, device=y.device)
                # 形状ガード
                if post.ndim == 0 or (post.ndim == 1 and post.shape[0] == y.shape[-1]):
                    y = y * post
                else:
                    if not self._post_warned:
                        print(f"[OBS SCALE][WARN] post_scale shape {tuple(post.shape)} "
                              f"!= last-dim {y.shape[-1]} — skipping post-scale.")
                        self._post_warned = True
            return y
        else:
            import numpy as np
            y = np.clip(y, self.clip_min, self.clip_max)
            if self.post is not None:
                post = self.post
                # 形状ガード
                if np.ndim(post) == 0 or (np.ndim(post) == 1 and post.shape[0] == y.shape[-1]):
                    y = y * post
                else:
                    if not self._post_warned:
                        print(f"[OBS SCALE][WARN] post_scale shape {post.shape} "
                              f"!= last-dim {y.shape[-1]} — skipping post-scale.")
                        self._post_warned = True
            return y

    def reverse_transform(self, x):
        y = x

        # Torch か NumPy かを判定
        try:
            import torch
            is_tensor = isinstance(y, torch.Tensor)
        except Exception:
            is_tensor = False

        if self.post is not None:
            if is_tensor:
                post = torch.as_tensor(self.post, dtype=y.dtype, device=y.device)
                if post.ndim == 0 or (post.ndim == 1 and post.shape[0] == y.shape[-1]):
                    y = y / torch.clamp(post, min=1e-6)
                else:
                    if not self._post_warned:
                        print(f"[OBS SCALE][WARN] post_scale shape {tuple(post.shape)} "
                              f"!= last-dim {y.shape[-1]} — skipping reverse post-scale.")
                        self._post_warned = True
            else:
                import numpy as np
                post = self.post
                if np.ndim(post) == 0 or (np.ndim(post) == 1 and post.shape[0] == y.shape[-1]):
                    y = y / np.maximum(post, 1e-6)
                else:
                    if not self._post_warned:
                        print(f"[OBS SCALE][WARN] post_scale shape {post.shape} "
                              f"!= last-dim {y.shape[-1]} — skipping reverse post-scale.")
                        self._post_warned = True

        if hasattr(self.base, "reverse_transform"):
            return self.base.reverse_transform(y)
        return y

    # base 側に未実装の属性/メソッドをフォワード
    def __getattr__(self, name):
        return getattr(self.base, name)

# ----------------------- データ読み込み ----------------------- #
def load_dataset(path: str):
    print(f"データセット '{path}' を読み込み中...")
    data = np.load(path)

    obs = data["observations"].astype(np.float32, copy=False)
    act = data["actions"].astype(np.float32, copy=False)
    rew = data["rewards"].astype(np.float32, copy=False)
    ter = data["terminals"].astype(np.bool_,   copy=False)

    # ★ 勝率計算用に「生の報酬」を退避
    rew_raw = rew.copy()

    print(f"  遷移数 : {len(obs)}")
    print(f"  状態次元 : {obs.shape[1]}")
    if act.ndim == 1:
        # —— 本スクリプトは (K + MAX_ARGS + 1) 連続ベクトル前提。離散 npz は明示エラー。
        raise RuntimeError(
            "このスクリプトは連続行動（one-hot+args+mask = K+3+1）前提です。"
            "離散行動 npz が渡されています。エンコード済み npz を再生成してください。"
        )
    else:
        print(f"  行動ベクトル: {act.shape}（連続） min={act.min():.3f}, max={act.max():.3f}")

    # === obs_mask の読み込み（前処理で作成済みを期待） ===
    src_mask = os.getenv("OBS_MASK_SRC", os.path.join(os.path.dirname(path), "obs_mask.npy"))
    if not os.path.exists(src_mask):
        raise RuntimeError(f"[FATAL] obs_mask が見つかりません: {src_mask}")
    try:
        obs_mask = np.load(src_mask, allow_pickle=False).astype(np.bool_)
    except Exception as e:
        raise RuntimeError(f"[FATAL] obs_mask の読み込みに失敗: {src_mask} ({e})")
    if obs_mask.ndim != 1 or obs_mask.shape[0] != obs.shape[1]:
        raise RuntimeError(f"[FATAL] obs_mask 形状不一致: obs_dim={obs.shape[1]} vs mask={obs_mask.shape} path={src_mask}")

    print(f"[OBS-MASK] loaded from: {src_mask} (len={obs_mask.shape[0]})")

    scaler_path = os.path.join(os.path.dirname(path), "scaler.npz")
    clip_min = -10.0   # 既定フォールバック
    clip_max =  10.0
    post_scale = None

    if os.path.exists(scaler_path):
        sc = np.load(scaler_path)
        mean, std = sc["mean"], sc["std"]
        if mean.shape != (obs.shape[1],) or std.shape != (obs.shape[1],):
            raise RuntimeError(
                f"[FATAL] scaler 形状不一致: mean={mean.shape} std={std.shape} vs obs_dim={obs.shape[1]} path={scaler_path}"
            )
        # ★ 追加読み込み：clip / post
        if "clip_min" in sc.files: clip_min = float(sc["clip_min"])
        if "clip_max" in sc.files: clip_max = float(sc["clip_max"])
        if "post_scale" in sc.files: post_scale = sc["post_scale"].astype(np.float32)
        print(f"[SCALER] loaded: {scaler_path} | clip=[{clip_min},{clip_max}] | post={'yes' if post_scale is not None else 'no'}")
    else:
        print(f"[SCALER] not found (ok): {scaler_path}")

    # 報酬のセンタリング & スケーリング（学習安定化）
    r_med = float(np.median(rew))
    np.subtract(rew, r_med, out=rew)
    r_std = float(rew.std()) if rew.std() > 1e-12 else 1.0
    # 目標分散は 0.08（環境変数で変更可能）。0 に落ちないようガード。
    target_std = max(1e-6, float(os.getenv("CQL_REWARD_STD", "0.08")))
    scale = max(r_std / target_std, 1e-6)
    np.divide(rew, scale, out=rew)
    # （任意）外れ値を軽くクリップ
    np.clip(rew, -1.0, 1.0, out=rew)

    print(f"  reward centered by median={r_med:.4f}")
    print(f"  reward scaled: std {r_std:.4f} -> {target_std:.4f} (scale={scale:.3f})")
    # ★ 追加: obs_mask を観測に適用（学習は縮約後で行う）
    if obs_mask.dtype != np.bool_:
        obs_mask = obs_mask.astype(np.bool_)
    n_dropped = int((~obs_mask).sum())
    print(f"[OBS-MASK] mask true={int(obs_mask.sum())} / {obs_mask.shape[0]}  (drop={n_dropped})")
    if n_dropped > 0:
        obs = obs[:, obs_mask]

    # 追加: マスク適用後にも“ほぼ定数”が残っていないか軽く監査（警告のみ）
    try:
        v_after = np.nanvar(obs, axis=0)
        rem_small = int(np.sum(v_after <= max(1e-10, float(OBS_ZERO_VAR_EPS))))
        if rem_small > 0:
            print(f"[OBS-MASK][WARN] mask 適用後も small-var(≤max(1e-10,EPS)) が {rem_small} 次元残っています。"
                  " mask 生成のしきい値や一致を再確認してください。")
    except Exception as _e:
        print(f"[OBS-MASK][WARN] post-mask variance check failed: {type(_e).__name__}: {_e}")

    # ★★★ 二段マスク（自動ドロップ）★★★
    #   一次マスク後の分散で改めて判定し、残存 small-var をその場で落とす。
    try:
        eps2 = float(os.getenv("OBS_ZERO_VAR_EPS2", "1e-10"))  # 二段目のしきい値（環境変数で変更可）
        v_after = np.nanvar(obs, axis=0)
        mask2 = (v_after > max(1e-10, eps2))
        drop2 = int((~mask2).sum())
        if drop2 > 0:
            print(f"[OBS-MASK][AUTO] second-stage drop: small-var ≤ {max(1e-10, eps2):.1e} → {drop2} 次元を追加ドロップ")
            # いまの（一次マスク後の）座標系で削る
            obs = obs[:, mask2]
            # 元（全体）座標のマスクへ反映（obs_mask が True の場所に mask2 を戻し書き）
            full_mask = obs_mask.copy()
            full_mask_idx = np.where(obs_mask)[0]
            full_mask[full_mask_idx[~mask2]] = False
            obs_mask = full_mask
        else:
            print("[OBS-MASK][AUTO] second-stage drop: 追加ドロップなし")
    except Exception as _e:
        print(f"[OBS-MASK][AUTO][WARN] second-stage masking failed: {type(_e).__name__}: {_e}")

    # ★ post_scale も（最終）obs_mask で縮約して観測次元と合わせる（重要）
    if post_scale is not None:
        try:
            if post_scale.shape[0] != obs_mask.shape[0]:
                print(
                    f"[SCALER][WARN] post_scale length {post_scale.shape[0]} != mask length {obs_mask.shape[0]} "
                    f"(will try best-effort)"
                )
            post_scale = post_scale[obs_mask.astype(bool)]
        except Exception as e:
            print(f"[SCALER][WARN] aligning post_scale with obs_mask failed: {e} → disable post_scale")
            post_scale = None

    # 観測の標準化統計は保存のみ（実際の標準化は d3rlpy の StandardScaler に任せる）
    mean = obs.mean(axis=0, keepdims=True)
    std  = np.maximum(obs.std(axis=0, keepdims=True), 1e-6)
    mean_1d = mean.squeeze().astype(np.float32)
    std_1d  = std.squeeze().astype(np.float32)



    # エピソード分割
    end_idxs = np.where(ter)[0].tolist()
    if not end_idxs or end_idxs[-1] != len(ter) - 1:
        end_idxs.append(len(ter) - 1)
    starts = [0] + [i + 1 for i in end_idxs[:-1]]
    episodes = list(zip(starts, end_idxs))  # (start, end) で end は終端 index

    epi_ids = np.arange(len(episodes))
    epi_train, epi_test = train_test_split(epi_ids, test_size=0.2, random_state=42)

    def _concat(epi_list):
        idxs = []
        for k in epi_list:
            s, e = episodes[k]
            idxs.extend(range(s, e + 1))
        idxs = np.array(idxs, dtype=np.int64)
        return obs[idxs], act[idxs], rew[idxs], ter[idxs]

    # ★ テスト側の「生の終端報酬」を抽出
    def _end_rewards(epi_idx_list):
        last = []
        for k in epi_idx_list:
            _, e = episodes[k]
            last.append(float(rew_raw[e]))
        return np.array(last, dtype=np.float32)

    obs_tr, act_tr, rew_tr, ter_tr = _concat(epi_train)
    obs_te, act_te, rew_te, ter_te = _concat(epi_test)
    test_last_raw = _end_rewards(epi_test)

    # === 観測の z-score 前処理（d3rlpy.StandardScaler 不在時のフォールバック） ===
    if StandardScaler is None:
        obs_tr = (obs_tr - mean) / std
        obs_te = (obs_te - mean) / std
        # ▼▼▼ B案: z-score の後に clip → * post_scale（追加） ▼▼▼
        np.clip(obs_tr, clip_min, clip_max, out=obs_tr)
        np.clip(obs_te, clip_min, clip_max, out=obs_te)
        if post_scale is not None:
            obs_tr *= post_scale
            obs_te *= post_scale
        # ▲▲▲ ここまで追加 ▲▲▲
        print("[OBS SCALE][fallback] applied numpy z-score (+ clip & post_scale) to observations")

    # ★ 行動の Min-Max は [0,1] に統一（クリップ保証）
    a_min = np.zeros_like(act_tr[0], dtype=np.float32)
    a_max = np.ones_like(act_tr[0],  dtype=np.float32)

    train_ds = MDPDataset(obs_tr, act_tr, rew_tr, ter_tr)
    test_ds  = MDPDataset(obs_te, act_te, rew_te, ter_te)

    print(f"  episodes: total={len(episodes)}, train={len(epi_train)}, test={len(epi_test)}")
    # ★ 追加: 観測マスクでドロップされた次元数
    n_dropped = int((~obs_mask).sum())
    # ★ 返却値に clip/post を追加（呼び出し側でラッパースケーラを作る）
    return (train_ds, test_ds, mean_1d, std_1d,
            a_min, a_max, test_last_raw, obs_mask, n_dropped,
            clip_min, clip_max, post_scale)


def _enable_critic_grad_clip(algo, max_norm: float = 1.0) -> bool:
    """
    critic(Q) の Optimizer.step に clip_grad_norm_ を挟む（多重ラップ防止付き）。
    """
    try:
        import torch
        import torch.nn as nn

        impl = getattr(algo, "impl", None)
        if impl is None:
            print("[CLIP][WARN] algo.impl が見つかりません。")
            return False

        # critic/Q のパラメータ収集
        params = []

        def _collect_from(obj):
            nonlocal params
            if obj is None:
                return
            if isinstance(obj, nn.Module):
                params += list(obj.parameters())
            elif isinstance(obj, (list, tuple)):
                for m in obj:
                    if isinstance(m, nn.Module):
                        params += list(m.parameters())

        for nm in ("q_func", "q_funcs", "q_function", "q_functions", "critic", "critics"):
            _collect_from(getattr(impl, nm, None))

        uniq = []
        seen = set()
        for p in params:
            pid = id(p)
            if pid not in seen:
                uniq.append(p); seen.add(pid)
        params = uniq

        if not params:
            print("[CLIP][WARN] critic(Q) のパラメータが見つかりませんでした。")
            return False

        wrapped = 0

        def _wrap_optimizer(opt):
            nonlocal wrapped
            if opt is None or not hasattr(opt, "step"):
                return
            if getattr(opt, "_clip_wrapped", False):
                return
            _orig = opt.step
            def _step_with_clip(*args, __orig=_orig, __params=params, **kwargs):
                torch.nn.utils.clip_grad_norm_(__params, max_norm)
                return __orig(*args, **kwargs)
            opt.step = _step_with_clip
            opt._clip_wrapped = True
            wrapped += 1

        for name in dir(impl):
            low = name.lower()
            if ("optim" in low) and (("q" in low) or ("critic" in low)):
                try:
                    _wrap_optimizer(getattr(impl, name, None))
                except Exception:
                    pass

        try:
            _opts = getattr(impl, "optimizers", None)
            if isinstance(_opts, (list, tuple)):
                for opt in _opts:
                    _wrap_optimizer(opt)
        except Exception:
            pass

        if wrapped > 0:
            print(f"[CLIP] Critic grad clip 有効化: max_norm={max_norm} | 対象optimizer={wrapped}個 | params={len(params)}個")
            return True

        print("[CLIP][WARN] critic 用 optimizer が見つかりませんでした。")
        return False

    except Exception as e:
        print(f"[CLIP][WARN] 失敗: {e}")
        return False

def _enable_actor_grad_clip(algo, max_norm: float = 1.0) -> bool:
    """policy(=actor) の Optimizer.step 前に clip_grad_norm_ を挟む（多重ラップ防止付き）。"""
    try:
        import torch
        import torch.nn as nn

        impl = getattr(algo, "impl", None)
        if impl is None:
            print("[CLIP][WARN] algo.impl が見つかりません。")
            return False

        params = []
        def _collect_from(obj):
            nonlocal params
            if obj is None:
                return
            if isinstance(obj, nn.Module):
                params += list(obj.parameters())
            elif isinstance(obj, (list, tuple)):
                for m in obj:
                    if isinstance(m, nn.Module):
                        params += list(m.parameters())

        for nm in ("policy", "_policy", "actor", "actors", "pi", "policy_network"):
            _collect_from(getattr(impl, nm, None))

        uniq, seen = [], set()
        for p in params:
            if id(p) not in seen:
                uniq.append(p); seen.add(id(p))
        params = uniq

        if not params:
            print("[CLIP][WARN] actor(policy) のパラメータが見つかりませんでした。")
            return False

        wrapped = 0
        def _wrap_optimizer(opt):
            nonlocal wrapped
            if opt is None or not hasattr(opt, "step"):
                return
            if getattr(opt, "_clip_wrapped", False):
                return
            _orig = opt.step
            def _step_with_clip(*args, __orig=_orig, __params=params, **kwargs):
                torch.nn.utils.clip_grad_norm_(__params, max_norm)
                return __orig(*args, **kwargs)
            opt.step = _step_with_clip
            opt._clip_wrapped = True
            wrapped += 1

        for name in dir(impl):
            low = name.lower()
            if ("optim" in low) and any(k in low for k in ("actor", "policy", "pi")):
                try:
                    _wrap_optimizer(getattr(impl, name, None))
                except Exception:
                    pass

        try:
            _opts = getattr(impl, "optimizers", None)
            if isinstance(_opts, (list, tuple)):
                for opt in _opts:
                    _wrap_optimizer(opt)
        except Exception:
            pass

        if wrapped > 0:
            print(f"[CLIP] Actor  grad clip 有効化: max_norm={max_norm} | 対象optimizer={wrapped}個 | params={len(params)}個")
            return True

        print("[CLIP][WARN] actor 用 optimizer が見つかりませんでした。")
        return False

    except Exception as e:
        print(f"[CLIP][WARN] 失敗(actor): {e}")
        return False

def _set_optimizer_lr(opt, lr: float) -> int:
    """torch.optim.* または d3rlpy impl 内の optimizer に対して学習率を一括設定。"""
    if opt is None:
        return 0
    try:
        if hasattr(opt, "param_groups"):
            for g in opt.param_groups:
                g["lr"] = float(lr)
            return 1
    except Exception:
        pass
    return 0

def _try_set_attr(obj, name: str, value) -> bool:
    try:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return True
    except Exception:
        pass
    return False
def _apply_phase_to_algo(algo, phase: dict) -> None:
    """
    d3rlpy v2 の CQL に段階的パラメータを適用する + 重要ハイパラ（τ/backup/dropout/actor_update_interval）も同期。
    """
    impl = getattr(algo, "impl", None)

    # CQL の保守項・サンプル数
    cw = float(phase["conservative_weight"])
    nas = int(phase["n_action_samples"])
    _try_set_attr(getattr(algo, "config", None), "conservative_weight", cw)
    _try_set_attr(getattr(algo, "config", None), "n_action_samples", nas)
    if impl is not None:
        _try_set_attr(impl, "conservative_weight", cw)
        _try_set_attr(impl, "n_action_samples", nas)

    # 温度初期値
    it = float(phase["initial_temperature"])
    ia = float(phase["initial_alpha"])
    _try_set_attr(getattr(algo, "config", None), "initial_temperature", it)
    _try_set_attr(getattr(algo, "config", None), "initial_alpha", ia)
    if impl is not None:
        _try_set_attr(impl, "initial_temperature", it)
        _try_set_attr(impl, "initial_alpha", ia)

    # 学習率（freeze_actor_steps>0 の間は actor を停止 = lr 0）
    freeze_steps = int(phase.get("freeze_actor_steps", 0) or 0)
    a_lr = 0.0 if freeze_steps > 0 else float(phase["actor_lr"])
    c_lr = float(phase["critic_lr"])
    t_lr = float(phase["temp_lr"])
    alr = float(phase["alpha_lr"])
    for nm, v in (
        ("actor_learning_rate", a_lr),
        ("critic_learning_rate", c_lr),
        ("temp_learning_rate", t_lr),
        ("alpha_learning_rate", alr),
    ):
        _try_set_attr(getattr(algo, "config", None), nm, v)

    if impl is not None:
        for name in dir(impl):
            low = name.lower()
            try:
                obj = getattr(impl, name)
            except Exception:
                continue
            if obj is None:
                continue
            set_to = None
            if "actor" in low or "policy" in low or low.endswith("_pi"):
                set_to = a_lr
            elif "critic" in low or "q" in low:
                set_to = c_lr
            elif "temp" in low or "temperat" in low or "entropy" in low:
                set_to = t_lr
            elif "alpha" in low:
                set_to = alr
            if set_to is not None:
                _set_optimizer_lr(obj, set_to)
        try:
            opts = getattr(impl, "optimizers", None)
            if isinstance(opts, (list, tuple)):
                for opt in opts:
                    _set_optimizer_lr(opt, c_lr)
        except Exception:
            pass

    # τ / backup / dropout を同期（config と encoder に反映）
    tau = float(phase.get("tau", 0.005))
    _try_set_attr(getattr(algo, "config", None), "tau", tau)
    _try_set_attr(impl, "tau", tau)

    # backup 方式はフェーズ定義を尊重（未指定なら現状維持）
    if "soft_q_backup" in phase:
        _try_set_attr(getattr(algo, "config", None), "soft_q_backup", bool(phase["soft_q_backup"]))
        _try_set_attr(impl, "soft_q_backup", bool(phase["soft_q_backup"]))
    if "max_q_backup" in phase:
        _try_set_attr(getattr(algo, "config", None), "max_q_backup", bool(phase["max_q_backup"]))
        _try_set_attr(impl, "max_q_backup", bool(phase["max_q_backup"]))


    cdrop = float(phase.get("critic_dropout", 0.20))
    try:
        enc = getattr(getattr(algo, "config", None), "critic_encoder_factory", None)
        if isinstance(enc, VectorEncoderFactory):
            enc.dropout_rate = cdrop
    except Exception:
        pass

    # ★ 追加: actor_update_interval を同期
    aui = int(phase.get("actor_update_interval", 1))
    _try_set_attr(getattr(algo, "config", None), "actor_update_interval", aui)
    if impl is not None:
        _try_set_attr(impl, "actor_update_interval", aui)

    bs = int(phase.get("batch_size", 256))
    _try_set_attr(getattr(algo, "config", None), "batch_size", bs)
    name = phase.get("name") or phase.get("phase") or "unknown"
    sqb = bool(phase.get("soft_q_backup", getattr(getattr(algo, "config", None), "soft_q_backup", False)))
    mqb = bool(phase.get("max_q_backup",  getattr(getattr(algo, "config", None), "max_q_backup",  False)))
    print(
        f"[PHASE] {name} | cw={cw} nas={nas} "
        f"| lr(actor/critic/temp/alpha)={a_lr:.1e}/{c_lr:.1e}/{t_lr:.1e}/{alr:.1e} "
        f"| init_temp/alpha={it:.2f}/{ia:.2f} | tau={tau:.4f} | "
        f"soft_q_backup={sqb} max_q_backup={mqb} | critic_dropout={cdrop:.2f} "
        f"| actor_update_interval={aui} | batch_size={bs}"
    )

    # 温度の下限/上限制御（policy_stdの枯渇や過大化を防ぐ）
    _raise_temperature_min(algo, min_temp=float(os.getenv("MIN_TEMP", "0.08")))
    _cap_temperature(algo,  max_temp=float(os.getenv("MAX_TEMP", "0.22")))

# ▼▼ ここに追加 ▼▼
def _cap_temperature(algo, max_temp: float = 0.22) -> None:
    """impl 内の温度/対数温度を安全に上限クリップ。"""
    try:
        impl = getattr(algo, "impl", None) or getattr(algo, "_impl", None)
        if impl is None:
            return
        import math, torch
        # 直接温度
        for name in ("temperature", "temp", "alpha", "entropy_temperature"):
            if hasattr(impl, name):
                t = getattr(impl, name)
                if isinstance(t, torch.Tensor):
                    with torch.no_grad():
                        t.data.clamp_(max=max_temp)
                elif isinstance(t, (float, int)):
                    setattr(impl, name, min(float(t), max_temp))
        # 対数温度
        for name in ("log_temperature", "log_temp", "log_alpha"):
            if hasattr(impl, name):
                lt = getattr(impl, name)
                if hasattr(lt, "data"):
                    with torch.no_grad():
                        lt.data.clamp_(max=float(math.log(max_temp)))
    except Exception:
        pass

def _raise_temperature_min(algo, min_temp: float = 0.05) -> None:
    """
    impl 内の温度/対数温度を min_temp 以上に“下限引上げ”する。
    - 可能なら log パラメータを直接上書き（勾配には影響しないよう no_grad）。
    """
    try:
        import math, torch
        impl = getattr(algo, "impl", None) or getattr(algo, "_impl", None)
        if impl is None:
            return

        def _as_tensor1(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(float(x), dtype=torch.float32)

        # 直接温度（数値 or Tensor）
        for name in ("temperature", "temp", "alpha", "entropy_temperature"):
            if hasattr(impl, name):
                v = getattr(impl, name)
                try:
                    cur = float(v.item() if hasattr(v, "item") else v)
                    if cur < min_temp:
                        with torch.no_grad():
                            if isinstance(v, torch.Tensor):
                                v.data.copy_(_as_tensor1(min_temp))
                            else:
                                setattr(impl, name, float(min_temp))
                except Exception:
                    pass

        # 対数温度
        for name in ("log_temperature", "log_temp", "log_alpha"):
            if hasattr(impl, name):
                v = getattr(impl, name)
                try:
                    with torch.no_grad():
                        tgt = float(math.log(min_temp))
                        if hasattr(v, "data"):
                            cur = float(v.detach().cpu().item())
                            if cur < tgt:
                                v.data.copy_(_as_tensor1(tgt))
                except Exception:
                    pass
    except Exception:
        pass
# ▲▲ ここに追加 ▲▲

def _maybe_set_value_clips(algo, q_abs: float = None, td_abs: float = None) -> None:
    """
    実装がサポートしていれば、Q値やTD/targetのクリップ上限を設定。
    未対応版では無視（副作用なし）。
    """
    impl = getattr(algo, "impl", None) or getattr(algo, "_impl", None)
    if impl is None:
        return
    try:
        if q_abs is not None:
            for name in ("q_clip_abs", "q_value_clip", "q_clip_value"):
                _try_set_attr(impl, name, float(q_abs))
        if td_abs is not None:
            for name in ("td_clip_abs", "target_clip_abs", "backup_clip_abs"):
                _try_set_attr(impl, name, float(td_abs))
    except Exception:
        pass

def _select_phase(epoch: int) -> Dict:
    """
    エポック番号に応じて既定のフェーズ設定(dict)を返すユーティリティ。
    既定の境界: 1-4 → phase1, 5-8 → phase2, 9+ → phase3
    呼び出し側では max(ep, 5) / max(ep, 9) で最低境界を担保済み。
    """
    # しきい値に応じてフェーズ名を決定
    if epoch >= 9:
        phase: Phase = "phase3_converge"
    elif epoch >= 5:
        phase = "phase2_growth"
    else:
        phase = "phase1_stabilize"

    # 既定値は PhaseScheduler の定義を流用（既存ロジックと整合）
    sch = PhaseScheduler()
    sch.phase = phase
    params = sch.get_current_params()
    params = {**params, "name": phase}
    return params


# ----------------------- モデル生成（安定プリセット） ----------------------- #
def create_model(device: str, observation_scaler, action_scaler):
    GAMMA = DEFAULT_GAMMA

    actor_enc  = VectorEncoderFactory(
        hidden_units=[256, 256], activation="relu", use_layer_norm=True
    )
    critic_enc = VectorEncoderFactory(
        hidden_units=[256, 256],
        activation="relu",
        use_layer_norm=True,
        dropout_rate=0.40,  # A：過適合緩和を強めに
    )

    # d3rlpy v2: 学習率は Config 側で指定。Factory には lr を渡さない。
    adam = dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
    actor_opt = AdamFactory(**adam)
    critic_opt = AdamFactory(**adam)
    temp_opt  = AdamFactory(**adam)
    alpha_opt = AdamFactory(**adam)

    # 初期フェーズ(Phase1)は soft backup を既定に（後続フェーズで _apply_phase_to_algo が上書き）
    bkp = _force_backup_mode(os.getenv("INIT_BACKUP_MODE", "soft"))
    cfg = CQLConfig(
        batch_size=256,
        gamma=GAMMA,
        n_critics=2,
        soft_q_backup=bkp["soft_q_backup"],
        max_q_backup =bkp["max_q_backup"],

        conservative_weight=36.0,
        n_action_samples=32,

        initial_temperature=0.10,
        alpha_threshold=0.70,
        initial_alpha=0.10,

        actor_learning_rate=3e-5,
        critic_learning_rate=7e-5,
        temp_learning_rate=3e-4,   # 0.0 → 3e-4（探索回復）
        alpha_learning_rate=3e-5,

        actor_optim_factory=actor_opt,
        critic_optim_factory=critic_opt,
        temp_optim_factory=temp_opt,
        alpha_optim_factory=alpha_opt,

        actor_encoder_factory=actor_enc,
        critic_encoder_factory=critic_enc,

        q_func_factory=_q_factory,

        observation_scaler=observation_scaler,
        action_scaler=action_scaler,

        tau=0.002,
    )

    return cfg.create(device=device)
def _tdcombo_call_impl(self, algo, episodes):
    """
    TD-Proxy と各種スナップショット指標をまとめて計算して返す実体。
    - 依存: self.td_obs, self.td_nxt, self.s0, self.probe_obs, self.behavior_s, self.behavior_a
    - 参照: self.gamma, self._ema_update, self._safe_alpha, self._transform_action, 各種 OK 範囲
    """
    import numpy as _np

    self._epoch += 1
    rng = self.rng

    # ========== TD-Proxy 用サンプリング ==========
    n_all = int(self.td_obs.shape[0]) if hasattr(self, "td_obs") else 0
    n_pick = min(int(self.td_sample_n), n_all) if n_all > 0 else 0

    if n_pick > 0:
        idx = rng.choice(n_all, size=n_pick, replace=False)
        s_batch   = self.td_obs[idx].astype(_np.float32, copy=False)
        sn_batch  = self.td_nxt[idx].astype(_np.float32, copy=False)

        # π(s), Q(s, π(s))
        a_s  = _np.asarray(algo.predict(s_batch), dtype=_np.float32)
        q_s  = _np.asarray(algo.predict_value(s_batch, a_s), dtype=_np.float32).reshape(-1)

        # π(s'), Q(s', π(s'))
        a_sn = _np.asarray(algo.predict(sn_batch), dtype=_np.float32)
        q_sn = _np.asarray(algo.predict_value(sn_batch, a_sn), dtype=_np.float32).reshape(-1)

        # TD-Proxy（報酬情報がないため |Q(s,π)-γ Q(s',π')| を proxy として使う）
        td_proxy = _np.abs(q_s - (self.gamma * q_sn))

        td_p50 = float(_np.percentile(td_proxy, 50))
        td_p90 = float(_np.percentile(td_proxy, 90))
        td_p95 = float(_np.percentile(td_proxy, 95))

        # 標準化 z-score の 95%（スパイク検知用）
        mu  = float(_np.median(td_proxy))
        sd  = float(td_proxy.std()) + 1e-6
        z95 = float(_np.percentile(_np.abs((td_proxy - mu) / sd), 95.0))
    else:
        td_p50 = td_p90 = td_p95 = z95 = _np.nan

    # EMA 更新
    td95_ema = self._ema_update("td_p95", td_p95)

    # ========== 値スケールと比率 ==========
    if getattr(self, "probe_obs", _np.zeros((0, 1), _np.float32)).shape[0] > 0:
        a_probe = _np.asarray(algo.predict(self.probe_obs), dtype=_np.float32)
        q_probe = _np.asarray(algo.predict_value(self.probe_obs, a_probe), dtype=_np.float32).reshape(-1)
        vs_p50  = float(_np.percentile(_np.abs(q_probe), 50))
    else:
        vs_p50 = _np.nan

    # 基準 vs を初回に固定（フロア 0.1 でゼロ割と暴走を防止）
    if getattr(self, "_vs_base", None) is None and _np.isfinite(vs_p50) and vs_p50 > 0:
        self._vs_base = float(max(abs(vs_p50), 0.1))

    if getattr(self, "_vs_base", 0.0) > 0:
        base = float(max(self._vs_base or 0.0, 0.1))
        r = (float(abs(vs_p50)) / base) if _np.isfinite(vs_p50) else _np.nan
        vs_ratio = float(min(max(r, 0.1), 10.0))
    else:
        vs_ratio = _np.nan

    ratio_ema = self._ema_update("vs_ratio", vs_ratio)

    # --- しつこく上限張り付きなら自動リベース ---
    if _np.isfinite(ratio_ema) and ratio_ema >= 3.0 and _np.isfinite(vs_p50) and vs_p50 > 0:  # ← 変更: 9.5 → 3.0
            self._vs_base = float(max(abs(vs_p50), 0.1))

    # ========== 初期状態の価値（参考） ==========
    if getattr(self, "s0", _np.zeros((0, 1), _np.float32)).shape[0] > 0:
        a0 = _np.asarray(algo.predict(self.s0), dtype=_np.float32)
        q0 = _np.asarray(algo.predict_value(self.s0, a0), dtype=_np.float32).reshape(-1)
        vhat_init = float(_np.mean(q0))
    else:
        vhat_init = _np.nan

    # ========== ポリシー変動量と分散 ==========
    if getattr(self, "probe_obs", _np.zeros((0, 1), _np.float32)).shape[0] > 0:
        a_cur = _np.asarray(algo.predict(self.probe_obs), dtype=_np.float32)

        if getattr(self, "_prev_policy", None) is None:
            dpolicy = 0.0
        else:
            diff = a_cur - self._prev_policy
            dpolicy = float(_np.mean(_np.linalg.norm(diff, axis=1)))

        # 次回用に保持
        self._prev_policy = a_cur.copy()

        # policy_std: 各次元の標準偏差の平均
        policy_std = float(_np.mean(_np.std(a_cur, axis=0)))
    else:
        dpolicy = policy_std = _np.nan

    dpol_ema = self._ema_update("dpolicy", dpolicy)
    pstd_ema = self._ema_update("policy_std", policy_std)

    # ========== 挙動ポリシーとの距離 ==========
    if (getattr(self, "behavior_s", _np.zeros((0, 1), _np.float32)).shape[0] > 0 and
        getattr(self, "behavior_a", _np.zeros((0, 1), _np.float32)).shape[0] > 0):
        a_pred = _np.asarray(algo.predict(self.behavior_s), dtype=_np.float32)
        a_pred_t = self._transform_action(a_pred)
        a_beh_t  = self._transform_action(self.behavior_a)
        beh_d    = _np.linalg.norm(a_pred_t - a_beh_t, axis=1)

        beh_p50 = float(_np.percentile(beh_d, 50))
        beh_p90 = float(_np.percentile(beh_d, 90))
    else:
        beh_p50 = beh_p90 = _np.nan

    # ========== α / 温度 ==========
    temp_val = float(self._safe_alpha(algo))

    # ========== Caution判定（スパイク・スケール逸脱・探索不足 等） ==========
    bad_flags = []
    # TDスパイク
    if _np.isfinite(z95) and z95 > 3.0:
        bad_flags.append(True)
    # VS比の逸脱
    if _np.isfinite(vs_ratio) and (vs_ratio > self.vs_ratio_ok[1] or vs_ratio < self.vs_ratio_ok[0]):
        bad_flags.append(True)
    # Δpolicy 過大
    if _np.isfinite(dpol_ema) and dpol_ema > 0.06:
        bad_flags.append(True)
    # 探索不足（policy_std 低）
    if _np.isfinite(pstd_ema) and pstd_ema < getattr(self, "policy_std_ok", 0.10):
        bad_flags.append(True)
    # 挙動距離過大
    if _np.isfinite(beh_p90) and beh_p90 > getattr(self, "beh_ok_threshold", 0.25):
        bad_flags.append(True)

    if any(bad_flags):
        self._caution_streak += 1
    else:
        # 徐々に減衰させる（ゼロまで）
        self._caution_streak = max(0, int(self._caution_streak) - 1)

    # 返却（NaNは呼び出し側で保護される想定だが、0.0に落として返す項目も用意）
    def _nz(x):
        return float(x) if _np.isfinite(x) else 0.0

    return {
        "td_p50": _nz(td_p50),
        "td_p90": _nz(td_p90),
        "td_p95": _nz(td_p95),
        "td_p95_ema": _nz(td95_ema),
        "td_z95": _nz(z95),

        "temp": _nz(temp_val),
        "vs_p50": _nz(vs_p50),
        "vs_ratio": _nz(vs_ratio),
        "vs_ratio_ema": _nz(ratio_ema),
        "vhat_init": _nz(vhat_init),

        "dpolicy": _nz(dpolicy),
        "dpolicy_ema": _nz(dpol_ema),

        "beh_p50": _nz(beh_p50),
        "beh_p90": _nz(beh_p90),

        "policy_std": _nz(policy_std),
        "policy_std_ema": _nz(pstd_ema),

        "caution_streak": float(self._caution_streak),
    }

def show_win_rate_from_raw_terminal(last_rewards_raw: np.ndarray):
    n_all = int(last_rewards_raw.size)
    n_win = int((last_rewards_raw > 0).sum())
    wr = (n_win / n_all) if n_all > 0 else 0.0
    print(f"テスト勝率(生報酬の終端): {n_win}/{n_all} = {wr:.3f}")

def main():
    print("d3rlpy ポケモンカードゲーム学習開始")
    _ensure_v2()

    dataset_path = os.getenv("D3RLPY_DATASET", r"D:\date\d3rlpy_dataset_all.npz")
    if not os.path.exists(dataset_path):
        print(f"データセット '{dataset_path}' が見つかりません。先に生成してください。")
        return

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ★ act の min/max と “生の終端報酬”、観測マスク情報も受け取る
    train_ds, test_ds, mean, std, a_min, a_max, test_last_raw, obs_mask, n_dropped, clip_min, clip_max, post_scale = load_dataset(dataset_path)

    # === 成果物として obs_mask と scaler を RUN_DIR に保存 ===
    os.makedirs(RUN_DIR, exist_ok=True)

    # obs_mask
    try:
        np.save(OBS_MASK_PATH, obs_mask.astype(np.bool_))
        print(f"[OBS-MASK] saved to: {OBS_MASK_PATH}")
        print(f"[OBS-MASK] note: training loads mask from {os.getenv('OBS_MASK_SRC', os.path.join(os.path.dirname(dataset_path), 'obs_mask.npy'))}")
    except Exception as e:
        print(f"[OBS-MASK][WARN] save failed: {e}")

    scaler_out = os.path.join(RUN_DIR, "scaler.npz")
    try:
        np.savez(
            scaler_out,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            clip_min=np.float32(clip_min),
            clip_max=np.float32(clip_max),
            post_scale=(post_scale.astype(np.float32) if post_scale is not None else None),
        )
        print(f"[SCALER] stats (mean/std + clip + post) saved to: {scaler_out}")
    except Exception as e:
        print(f"[SCALER][WARN] save failed: {e}")

    # --- 監査: 状態次元チェック（297）
    expected_state_dim = None
    try:
        _vmap = None
        _candidates = [
            os.path.join(os.path.dirname(dataset_path), "v2", "card_id2idx.json"),
            os.path.join(os.path.dirname(dataset_path), "card_id2idx.json"),
            "card_id2idx.json",
        ]
        for _p in _candidates:
            if os.path.exists(_p):
                with open(_p, "r", encoding="utf-8") as _f:
                    _vmap = json.load(_f)
                print(f"[CHECK] vocab ok: size={len(_vmap)} (V-1) from {_p}")
                break
        if _vmap is None:
            raise FileNotFoundError("card_id2idx.json not found in candidates")
        V = int(len(_vmap) + 1)  # 0 は PAD
        expected_state_dim = 6 * V + 203
    except Exception as _e:
        print(f"[CHECK][WARN] 語彙の読み込みに失敗（expected_state_dim を算出できません）: {type(_e).__name__}: {_e}")

    state_dim = None
    try:
        state_dim = int(train_ds.observations.shape[1])  # v1スタイル
    except Exception:
        try:
            if getattr(train_ds, "episodes", None):
                state_dim = int(train_ds.episodes[0].observations.shape[1])  # v2スタイル
        except Exception:
            state_dim = None

    if state_dim is not None and expected_state_dim is not None:
        expected_after_mask = int(expected_state_dim) - int(n_dropped or 0)
        print(f"[CHECK] vocab_size(V-1)={len(_vmap)}, "
              f"expected_state_dim(before mask)={expected_state_dim}, "
              f"dropped={int(n_dropped or 0)}, "
              f"expected_after_mask={expected_after_mask}, "
              f"actual_state_dim={state_dim}")
        if state_dim == expected_after_mask:
            print("[CHECK] 状態次元は expected_after_mask と一致しています。")
        else:
            print(
                f"[CHECK][WARN] 状態次元不一致: expected_after_mask={expected_after_mask}, "
                f"actual={state_dim}（フェーズCでは警告のみ・続行）"
            )
    elif state_dim is not None:
        print(f"[CHECK] state_dim={state_dim}（expected_state_dim の算出に失敗したため一致検証はスキップ）")
    else:
        print(f"[CHECK][WARN] 状態次元の検証をスキップ: 形状を推定できませんでした。")


    # --- 監査: 行動次元 = K + (MAX_ARGS=3) + 1 ---
    action_dim = None
    try:
        action_dim = int(train_ds.actions.shape[1])
    except Exception:
        try:
            if getattr(train_ds, "episodes", None):
                action_dim = int(train_ds.episodes[0].actions.shape[1])
        except Exception:
            action_dim = None

    if action_dim is not None:
        K_infer = max(0, action_dim - (MAX_ARGS + 1))
        print(f"[CHECK] action_dim={action_dim} → inferred_K={K_infer} (should be len(action_types))")
        try:
            with open("action_types.json", "r", encoding="utf-8") as _af:
                _atypes = json.load(_af)
            K_file = int(len(_atypes))
            print(f"[CHECK] action_types.json found: K_file={K_file}")
            assert K_infer == K_file, f"行動one-hot次元K不一致: inferred={K_infer}, file={K_file}"
        except FileNotFoundError:
            print("[CHECK] action_types.json が見つかりません（inferred_K のみで続行）")
        except Exception as _e:
            print(f"[CHECK][WARN] 行動次元の検証に失敗（スキップ）: {type(_e).__name__}: {_e}")

        if action_dim <= 5:
            raise RuntimeError(
                "actions の次元が 5 でした。五要素そのままが保存されています。\n"
                "学習・推論一貫のため『五要素→(K+3+1)次元』にエンコードした npz を再生成してください。"
            )
    else:
        print(f"[CHECK][WARN] 行動次元の検証をスキップ: 形状を推定できませんでした。")

    # ====== 観測/行動スケーラ ======
    # 観測：StandardScaler に B案 post_scale を合成（z → clip → *post）
    if StandardScaler is not None:
        base_scaler = StandardScaler()
        observation_scaler = PostScaledObservationScaler(base_scaler, clip_min, clip_max, post_scale)
        print("[OBS SCALE] StandardScaler(+post_scale) を使用します（z → clip → *post）")
    else:
        observation_scaler = None  # ← Fallback は load_dataset の numpy 前処理（z→clip→*post）で対応済み
        print("[WARN] StandardScaler 不在: 事前の numpy 前処理（z→clip→*post）で対応します")

    # 行動：Min-Max（[0,1] に統一してクリップ保証）
    if MinMaxActionScaler is not None:
        print(f"[ACTION SCALE] force min=0 / max=1 for all {action_dim} dims（クリップ保証）")
        action_scaler = MinMaxActionScaler(minimum=a_min, maximum=a_max)
    else:
        action_scaler = None
        print("[WARN] action_scaler=None で継続します（不安定化の恐れ）")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    algo = create_model(device, observation_scaler, action_scaler)

    # 先に一度だけ明示 build（d3rlpy v2 では fit 内でも build されるが、impl を確実に用意する）
    try:
        algo.build_with_dataset(train_ds)
    except Exception:
        # 一部バージョンに build_with_dataset が無い場合は、そのまま続行（fit 時に build される）
        pass

    # build 後に勾配クリップを有効化（impl が存在するので WARN が出ない）
    _enable_critic_grad_clip(algo, max_norm=1.0)
    _enable_actor_grad_clip(algo,  max_norm=1.0)


    # ★ ログルートを先に確実に作成（File/Tensorboard が存在前提で動く環境対策）
    os.makedirs("d3rlpy_logs", exist_ok=True)

    if FAST_MODE:
        logger = FileAdapterFactory(root_dir="d3rlpy_logs")
    else:
        logger = CombineAdapterFactory(
            [FileAdapterFactory(root_dir="d3rlpy_logs"), TensorboardAdapterFactory(root_dir="d3rlpy_logs")]
        )


    # === TDComboEvaluator（健全性監視 + スナップショット） ===
    class TDComboEvaluator:
        """
        TD-Proxy（健全性）とスナップショット指標（V̂_init / Δpolicy / Behavior距離 / policy_std）を
        1回のEvaluator呼び出しでまとめて出力。Cautionが連続したら早期停止（例外）を投げる。
        """
        def __init__(
            self,
            episodes,
            gamma=0.99,
            td_sample_n=2048,
            probe_n=1024,
            behavior_n=1024,
            alpha_ok_range=(0.05, 0.80),
            vs_ratio_ok=(0.5, 2.0),
            early_stop_streak=20,  # 寛容気味
            seed=42,
            action_min=None,
            action_max=None,
            beh_ok_threshold=0.25,
            policy_std_ok=0.10,  # C の目標値
        ):
            self.episodes = episodes
            self.gamma = gamma
            self.td_sample_n = td_sample_n
            self.probe_n = probe_n
            self.behavior_n = behavior_n
            self.alpha_ok_range = alpha_ok_range
            self.vs_ratio_ok = vs_ratio_ok
            self.early_stop_streak = int(early_stop_streak)
            self.rng = np.random.RandomState(seed)
            self.a_min = None if action_min is None else np.asarray(action_min, dtype=np.float32)
            self.a_max = None if action_max is None else np.asarray(action_max, dtype=np.float32)
            self.beh_ok_threshold = float(beh_ok_threshold)
            self.policy_std_ok  = float(policy_std_ok)

            # —— 事前サンプル集合 —— #
            obs_list, next_list = [], []
            for ep in self.episodes:
                T = len(ep)
                if T < 2:
                    continue
                for t in range(T - 1):
                    obs_list.append(ep.observations[t])
                    next_list.append(ep.observations[t + 1])
            self.td_obs = np.asarray(obs_list, dtype=np.float32) if obs_list else np.zeros((0, 1), np.float32)
            self.td_nxt = np.asarray(next_list, dtype=np.float32) if obs_list else np.zeros((0, 1), np.float32)

            s0 = []
            for ep in self.episodes:
                if len(ep) > 0:
                    s0.append(ep.observations[0])
            self.s0 = np.asarray(s0, dtype=np.float32) if s0 else np.zeros((0, 1), np.float32)

            all_obs = []
            for ep in self.episodes:
                if len(ep) > 0:
                    all_obs.append(ep.observations)
            all_obs = np.concatenate(all_obs, axis=0) if all_obs else np.zeros((0, 1), np.float32)
            if all_obs.shape[0] > self.probe_n:
                idx = self.rng.choice(all_obs.shape[0], size=self.probe_n, replace=False)
                self.probe_obs = all_obs[idx]
            else:
                self.probe_obs = all_obs

            s_b, a_b = [], []
            for ep in self.episodes:
                if len(ep) == 0:
                    continue
                s_b.append(ep.observations)
                a_b.append(ep.actions)
            if s_b:
                s_b = np.concatenate(s_b, axis=0)
                a_b = np.concatenate(a_b, axis=0).astype(np.float32)
                if s_b.shape[0] > self.behavior_n:
                    idx = self.rng.choice(s_b.shape[0], size=self.behavior_n, replace=False)
                    self.behavior_s = s_b[idx].astype(np.float32)
                    self.behavior_a = a_b[idx].astype(np.float32)
                else:
                    self.behavior_s = s_b.astype(np.float32)
                    self.behavior_a = a_b.astype(np.float32)
            else:
                self.behavior_s = np.zeros((0, 1), np.float32)
                self.behavior_a = np.zeros((0, 1), np.float32)

            self._epoch = 0
            self._prev_p95 = None
            self._vs_base = None
            self._prev_policy = None
            self._caution_streak = 0
            self.best_epoch = 0
            self.best_td_p95 = float("inf")

            # === EMA（判定の短期ノイズ抑制）===
            self._ema = {}        # {"td_p95":..., "vs_ratio":..., "dpolicy":..., "policy_std":...}
            self._ema_beta = 0.2  # 0.2〜0.3 目安

        def _ema_update(self, key, x):
            if not np.isfinite(x):
                return x
            v = self._ema.get(key)
            if v is None:
                self._ema[key] = float(x)
                return x
            v = (1 - self._ema_beta) * v + self._ema_beta * float(x)
            self._ema[key] = v
            return v

        # ========= ここから追記: ラッパー =========
        def __call__(self, algo, episodes):
            # クラス外に置いた実装(_tdcombo_call_impl)へ委譲
            return _tdcombo_call_impl(self, algo, episodes)
        # ========= 追記ここまで =========

        def _safe_alpha(self, algo) -> float:
            """
            impl 内の temperature / alpha / log_* を安全に読む。
            見つかった順に使い、log の場合は exp で戻す。
            """
            try:
                import numpy as _np
                impl = getattr(algo, "impl", None) or getattr(algo, "_impl", None)
                if impl is None:
                    return float("nan")

                for name in ("temperature", "temp", "alpha", "entropy_temperature"):
                    v = getattr(impl, name, None)
                    if v is not None:
                        return float(v.item() if hasattr(v, "item") else v)

                for name in ("log_temperature", "log_temp", "log_alpha"):
                    v = getattr(impl, name, None)
                    if v is not None:
                        val = float(v.item() if hasattr(v, "item") else v)
                        return float(_np.exp(val))
            except Exception:
                pass
            return float("nan")

        def _value_scale_p50(self, algo) -> float:
            if self.probe_obs.shape[0] == 0:
                return float("nan")
            a = algo.predict(self.probe_obs)
            q = algo.predict_value(self.probe_obs, a).reshape(-1)
            return float(np.percentile(np.abs(q), 50))

        def _transform_action(self, a: np.ndarray) -> np.ndarray:
            try:
                if (self.a_min is not None) and (self.a_max is not None):
                    mn = self.a_min.astype(np.float32)
                    mx = self.a_max.astype(np.float32)
                    a  = a.astype(np.float32)
                    a  = np.clip(a, mn, mx)
                    denom = np.maximum(mx - mn, 1e-6)
                    a  = (a - mn) / denom
                    return a
            except Exception:
                pass
            return a.astype(np.float32)

    def _as_scalar(ev, key: str):
        def _wrapped(algo, episodes):
            v = ev(algo, episodes)
            return float(v.get(key, 0.0)) if isinstance(v, dict) else float(v)
        return _wrapped

    td_n   = 512  if FAST_MODE else 2048
    prob_n = 512  if FAST_MODE else 1024
    beh_n  = 512  if FAST_MODE else 1024

    # === ここから追加: action_dimに応じた動的な挙動距離しきい値 ===
    def _default_beh_threshold(dim: int) -> float:
        # Uniform[0,1]^d の平均距離 ~ sqrt(d/6) をベースに係数1.2、最低0.25
        import math
        return max(1.2 * math.sqrt(dim / 6.0), 0.25)

    try:
        ACTION_DIM = (
            len(a_min) if (a_min is not None) else
            (action_dim if ("action_dim" in locals() and isinstance(action_dim, int))
             else int(train_ds.actions.shape[1]))
        )
    except Exception:
        ACTION_DIM = int(train_ds.actions.shape[1])

    BEH_OK = _default_beh_threshold(ACTION_DIM)
    print(f"[EVAL] beh_ok_threshold set to {BEH_OK:.3f} for action_dim={ACTION_DIM}")
    # === 追加ここまで ===


    _td_eval = TDComboEvaluator(
        test_ds.episodes,
        gamma=DEFAULT_GAMMA,
        td_sample_n=td_n,
        probe_n=prob_n,
        behavior_n=beh_n,
        early_stop_streak=20,
        action_min=a_min,
        action_max=a_max,
        beh_ok_threshold=BEH_OK,
        policy_std_ok=0.05,
    )

    if FAST_MODE:
        evaluators = {}
    else:
        try:
            evaluators = {
                "value_scale": AverageValueEstimationEvaluator(episodes=test_ds.episodes),
                "td_p95": _as_scalar(_td_eval, "td_p95"),
            }
        except Exception as e:
            print(f"[EVAL][WARN] value_scale evaluator disabled: {e}")
            evaluators = {"td_p95": _as_scalar(_td_eval, "td_p95")}

    # === ANCHOR: ここに挿入（main()内・インデント4スペース）===
    def steps_for_phase(phase_name: str, epoch: int) -> int:
        """
        フェーズに応じて1エポックのステップ数を返す。
        Phase1は短め（ep<=2はさらに短め）、Phase2は標準、Phase3は長め。
        """
        name = (phase_name or "").lower()
        if FAST_MODE:
            if "phase1" in name:
                return 1200 if epoch <= 2 else 1800
            if "phase2" in name:
                return 2500
            return 3500  # phase3_converge（FAST）
        else:
            if "phase1" in name:
                return 3000 if epoch <= 2 else 4000
            if "phase2" in name:
                return 6000
            return 8000  # phase3_converge

    exp_name = f"CQL_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"\n=== CQL | 学習開始 ===")

    # 動的ステップ制御: 総ステップと最大エポック上限（whileループ側で使用）
    MAX_TOTAL_STEPS = 40_000 if FAST_MODE else 150_000
    MAX_EPOCHS_HARD = 200

    # ← 追加：未定義参照を避けるための既定値（後で毎エポック上書き）
    EPOCHS = MAX_EPOCHS_HARD
    N_STEPS_PER_EPOCH = 5_000  # デフォ。各エポックの先頭で steps_for_phase に置き換える

    os.makedirs(RUN_DIR, exist_ok=True)
    best_path_tmp = os.path.join(RUN_DIR, "_best_tmp.d3")
    best_path_ae  = os.path.join(RUN_DIR, "_best_AE_tmp.d3")  # （E）三条件を満たすベスト

    best_eval = TDComboEvaluator(
        test_ds.episodes,
        gamma=DEFAULT_GAMMA,
        td_sample_n=td_n,
        probe_n=prob_n,
        behavior_n=beh_n,
        early_stop_streak=0,
        action_min=a_min,
        action_max=a_max,
        beh_ok_threshold=BEH_OK,
        policy_std_ok=0.05,
    )

# === 進捗ログ（メモリのみ・ファイル出力なし） ===
    _last_epoch_snapshot = {}  # 最終サマリー専用に最新だけ保持
    _epochs_all = []          # ← 追加: 全エポック履歴を保持して summary.json に同梱する

    def _jsonable(v):
        if isinstance(v, (int, float, np.integer, np.floating, bool)):
            try:
                return float(v)
            except Exception:
                return v
        return v

    def _log_progress(epoch, step, metrics, current, score, tag="epoch_end"):
        # ファイルへは書かず、最後のスナップショットだけ保持
        _last_epoch_snapshot.clear()
        _last_epoch_snapshot.update({
            "tag": tag,
            "epoch": int(epoch),
            "step": int(step),
            "metrics": {k: _jsonable(v) for k, v in (metrics or {}).items()},
            "current": {k: _jsonable(v) for k, v in (current or {}).items()},
            "score": _jsonable(score),
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        # ← 追加: 全履歴にも追記（CSVに頼らずサマリーへ出せるようにする）
        try:
            _epochs_all.append({
                "tag": tag,
                "epoch": int(epoch),
                "step": int(step),
                "metrics": {k: _jsonable(v) for k, v in (metrics or {}).items()},
                "current": {k: _jsonable(v) for k, v in (current or {}).items()},
                "score": _jsonable(score),
                "ts": _last_epoch_snapshot["ts"],
            })
        except Exception:
            pass

    # 最後に参照するためのヘルパ
    def _get_last_snapshot():
        return dict(_last_epoch_snapshot)

    # ← 追加: 全エポック履歴を返すヘルパ
    def _get_epochs_all():
        return list(_epochs_all)

    class AutoTuner:
        """
        AE条件達成に向けて、metrics に応じて current(dict) を“その場で”更新する軽量チューナ。
        - 低 policy_std / 低温度: 温度底上げ・temp/alpha lr強化・AUI=1・cw/nasを段階的に緩め・soft backupへ
        - z95 スパイク: cw/nas を強める（保守寄り）
        - vs_ratio: 大→保守↑、小→保守↓＋actor_lr↑
        - Δpolicy 高: actor/temp を絞る
        - cw は常に上限 (CW_MAX) でクリップ
        """
        def __init__(self,
                     cw_max: float = CW_MAX,
                     pstd_target: float = AE_PSTD_MIN,
                     temp_floor: float = 0.05,
                     temp_lr_max: float = 1e-3,
                     hold_soft_until_ratio_ok: bool = True):
            self.cw_max = float(cw_max)
            self.pstd_target = float(pstd_target)
            self.temp_floor = float(temp_floor)
            self.temp_lr_max = float(temp_lr_max)
            self.low_pstd_streak = 0
            # 比率が正常域に戻るまで soft_q_backup を維持するか
            self.hold_soft_until_ratio_ok = bool(hold_soft_until_ratio_ok)
            self._ratio_ok_streak = 0      # ← 追加: ratio 正常化の連続回数

        def _clip(self, v, lo, hi):
            return float(max(lo, min(hi, v)))

        def update(self, *, algo, current: dict, metrics: dict) -> str:
            # 実測値
            z95   = float(metrics.get("td_z95", float("nan")))
            ratio = float(metrics.get("vs_ratio_ema", float("nan")))
            dpol  = float(metrics.get("dpolicy_ema", float("nan")))
            pstd  = float(metrics.get("policy_std_ema", float("nan")))
            temp  = float(metrics.get("temp", float("nan")))

            # 低 policy_std をカウント（目標の半分 or 0.03 を下限）
            if (pstd == pstd) and (pstd < max(0.5 * self.pstd_target, 0.03)):
                self.low_pstd_streak += 1
            else:
                self.low_pstd_streak = 0

            changed = []

            # === フェーズ連動の soft/max 方針 ===
            phase_name = str(current.get("name") or current.get("phase") or "").lower()
            if "phase1" in phase_name:
                # Phase1 は ratio が安定化するまで soft を保持
                self.hold_soft_until_ratio_ok = True
            else:
                # Phase2/3 は ratio 正常域(0.7〜1.5)が“2連続”で入れば soft 解除
                ratio_ok = (np.isfinite(ratio) and 0.7 <= ratio <= 1.5)
                streak = int(current.get("_ratio_ok_streak", 0))
                streak = (streak + 1) if ratio_ok else 0
                current["_ratio_ok_streak"] = streak
                self.hold_soft_until_ratio_ok = (streak < 2)

            # === 回復モード（hold_soft_until_ratio_ok）: 比率が正常域に戻るまで soft を維持 ===
            if self.hold_soft_until_ratio_ok:
                if not (np.isfinite(ratio) and 0.7 <= ratio <= 1.5):
                    current["soft_q_backup"] = True
                    current["max_q_backup"]  = False
                    # 明示的に保持していることを記録
                    if "soft_hold" not in changed:
                        changed.append("soft_hold")

            # ====== 追加: value-scale オーバーシュート・ガード ======
            if (ratio == ratio) and ratio > 3.0:
                is_phase1 = "phase1" in str(current.get("name") or current.get("phase") or "").lower()

                # 危険域（>3.0）が続く場合のカウンタ
                streak = int(current.get("_ratio_hi_streak", 0))
                streak = (streak + 1) if ratio > 3.0 else 0
                current["_ratio_hi_streak"] = streak

                # overshoot 中は常に soft を維持（max へは退避しない）
                current["soft_q_backup"] = True
                current["max_q_backup"]  = False
                current["_force_max_backup"] = False

                current["conservative_weight"] = min(
                    float(current.get("conservative_weight", 24.0)) + (8.0 if ratio > 10.0 else 4.0),
                    self.cw_max
                )
                current["n_action_samples"] = min(
                    int(current.get("n_action_samples", 24)) + (8 if ratio > 10.0 else 4),
                    32
                )
                # クリティックの lr を落として暴走を抑える
                current["critic_lr"] = self._clip(float(current.get("critic_lr", 1e-5)) * 0.5, 3e-6, 3e-4)
                # 価値のクリップを強める（値スパイクの抑制）
                current["q_clip_abs"]  = 30.0
                current["td_clip_abs"] = 30.0
                # actor 更新を間引き（値安定を優先）
                current["actor_update_interval"] = max(4, int(current.get("actor_update_interval", 2)))
                # dropout を少し強める
                current["critic_dropout"] = min(0.60, float(current.get("critic_dropout", 0.50)) + 0.05)
                # 追加: ターゲット更新をより保守的に
                current["tau"] = 0.0015
                changed.append("overshoot_guard")

            # -- 探索不足レスキュー（連続2回 or 温度が極端に低い）
            if self.low_pstd_streak >= 2 or ((temp == temp) and temp < 0.02):
                _raise_temperature_min(algo, self.temp_floor)
                _cap_temperature(algo, max_temp=0.30)
                current["temp_lr"]  = self._clip(float(current.get("temp_lr", 3e-4))  * 2.0, 1e-5, self.temp_lr_max)
                current["alpha_lr"] = self._clip(float(current.get("alpha_lr", 3e-5)) * 1.3, 1e-6, 3e-3)
                # 追加: 探索回復のため actor_lr の下限を底上げ
                current["actor_lr"] = max(float(current.get("actor_lr", 1e-5)), 1.0e-5)
                current["actor_update_interval"] = 1
                # 追加: 高 cw 域では戻しを強める
                _cw = float(current.get("conservative_weight", 24.0))
                current["conservative_weight"] = max(_cw - (6.0 if _cw >= 36.0 else 4.0), 12.0)
                current["n_action_samples"]    = max(int(current.get("n_action_samples", 24)) - 4, 12)
                _try_set_attr(getattr(algo, "config", None), "alpha_threshold", 0.50)
                _try_set_attr(getattr(algo, "impl",   None), "alpha_threshold", 0.50)

                # 回復モードでは常に soft を維持
                if self.hold_soft_until_ratio_ok or (not (ratio == ratio) or (0.5 <= ratio <= 1.5)):
                    current["soft_q_backup"] = True
                    current["max_q_backup"]  = False
                changed.append("explore_boost")

            # -- z95 スパイク時は保守強化
            if (z95 == z95) and z95 > 4.5:
                current["conservative_weight"] = min(float(current.get("conservative_weight", 24.0)) + 4.0, self.cw_max)
                current["n_action_samples"]    = min(int(current.get("n_action_samples", 24)) + 4, 32)
                changed.append("spike_guard")

            # -- value-scale 調整（通常帯の微調整）
            if (ratio == ratio) and ratio > AE_VS_HI:
                current["conservative_weight"] = min(float(current.get("conservative_weight", 24.0)) + 2.0, self.cw_max)
                changed.append("vs_big→cw+2")
            elif (ratio == ratio) and ratio < AE_VS_LO:
                current["conservative_weight"] = max(float(current.get("conservative_weight", 24.0)) - 1.0, 12.0)
                current["actor_lr"] = self._clip(float(current.get("actor_lr", 2e-5)) * 1.2, 3e-6, 3e-4)
                changed.append("vs_small→cw-1, actor_lr×1.2")

            # -- Δpolicy 高
            if (dpol == dpol) and dpol > AE_DPOL_MAX:
                current["actor_lr"] = self._clip(float(current.get("actor_lr", 2e-5)) * 0.7, 3e-6, 3e-4)
                current["temp_lr"]  = self._clip(float(current.get("temp_lr", 1e-4)) * 0.7, 1e-6, self.temp_lr_max)
                changed.append("dpolicy_high→lrs×0.7")

            # 常に cw を上限クリップ
            current["conservative_weight"] = min(float(current.get("conservative_weight", 24.0)), self.cw_max)

            # —— ratio 正常化の連続判定（解放バルブ）——
            if (ratio == ratio) and (0.7 <= ratio <= 1.8) and (z95 == z95) and (z95 <= 3.2):
                self._ratio_ok_streak += 1
            else:
                self._ratio_ok_streak = 0

            if (self._ratio_ok_streak >= 2) or (np.isfinite(pstd) and pstd >= 0.10):
                # backup を soft に戻し、保守を少し緩め、AUI も戻す
                current["soft_q_backup"] = True
                current["max_q_backup"]  = False
                current["conservative_weight"] = max(float(current.get("conservative_weight", 24.0)) - 2.0, 16.0)
                current["n_action_samples"]    = max(int(current.get("n_action_samples", 24)) - 4, 12)
                current["actor_update_interval"] = max(2, int(current.get("actor_update_interval", 2)))
                # 探索緩め設定を既定へ戻す
                _try_set_attr(getattr(algo, "config", None), "alpha_threshold", 0.70)
                _try_set_attr(getattr(algo, "impl",   None), "alpha_threshold", 0.70)
                changed.append("overshoot_release")

            return ", ".join(changed)

    # —— SIGINT 時にも暫定ベストを確定保存 —— #
    def _graceful_exit(sig, frame):
        try:
            if os.path.exists(best_path_ae):
                shutil.copy2(best_path_ae, os.path.join(RUN_DIR, "learnable.d3"))
                print("[BEST] SIGINT: AE 条件ベストを確定保存")
            elif os.path.exists(best_path_tmp):
                shutil.copy2(best_path_tmp, os.path.join(RUN_DIR, "learnable.d3"))
                print("[BEST] SIGINT: 暫定ベストを確定保存")
        finally:
            sys.exit(130)
    signal.signal(signal.SIGINT, _graceful_exit)

    current = _select_phase(1)
    # 収束寄りの再開プリセットを適用（ep1 開始前）
    current.update(RECOVERY_PRESET)

    # （任意）開始メタ
    _log_progress(0, 0, {"note": "run_start"},
                  {**current, "device": device, "gamma": DEFAULT_GAMMA, "seed": 42},
                  float("nan"), "run_start")

    # 安定域カウント（D）
    from collections import deque
    stable_hist = deque(maxlen=3)

    # ★ AE 早期停止のガード（連続回数と actor 更新回数でゲート）
    AE_CONSEC_REQ = 4
    MIN_ACTOR_UPDATES_BEFORE_AE = 10_000
    ae_ok_consec = 0
    actor_updates_done = 0

    # ベスト記録
    best_score = float("inf")
    best_epoch = 0
    best_step = 0

    best_score_ae = float("inf")
    best_epoch_ae = 0
    best_step_ae = 0

    patience = max(40, EPOCHS // 2)
    no_improve = 0

    z95_bad_streak = 0

    autotuner = AutoTuner(
        cw_max=CW_MAX,
        pstd_target=AE_PSTD_MIN,
        temp_floor=0.12,                # ← 変更: 探索温度の下限を底上げ
        temp_lr_max=5e-4,
        hold_soft_until_ratio_ok=True,  # ← 変更: vs_ratioが正常域に戻るまで soft を保持
    )

    # ★ 追加：フェーズスケジューラ
    scheduler = PhaseScheduler()
    scheduler.phase = "phase1_stabilize"
    epoch = 0
    total_steps_done = 0
    while (total_steps_done < MAX_TOTAL_STEPS) and (epoch < MAX_EPOCHS_HARD):
        epoch += 1
        # === 動的ステップ数決定（フェーズ・エポック依存） ===
        phase_name_for_step = str(current.get("name") or current.get("phase") or "")
        n_ep_steps = int(steps_for_phase(phase_name_for_step, epoch))
        # 総ステップ上限でガード
        remain = int(MAX_TOTAL_STEPS - total_steps_done)
        if remain <= 0:
            print("[STOP] Reached MAX_TOTAL_STEPS.")
            break
        if n_ep_steps > remain:
            n_ep_steps = remain

        # フェーズ適用
        for k in ("conservative_weight","n_action_samples","actor_lr","critic_lr","temp_lr",
                  "alpha_lr","initial_temperature","initial_alpha","tau","soft_q_backup",
                  "max_q_backup","critic_dropout","freeze_actor_steps","actor_update_interval",
                  "batch_size", "q_clip_abs", "td_clip_abs"):
            current.setdefault(k, None)
        current["freeze_actor_steps"] = int(current.get("freeze_actor_steps") or 0)
        _apply_phase_to_algo(algo, current)
        _maybe_set_value_clips(
            algo,
            q_abs=(current.get("q_clip_abs") or (30.0 if FAST_MODE else 50.0)),
            td_abs=(current.get("td_clip_abs") or (30.0 if FAST_MODE else 50.0)),
        )

        algo.fit(
            dataset=train_ds,
            n_steps=n_ep_steps,
            n_steps_per_epoch=n_ep_steps,
            experiment_name=exp_name,
            logger_adapter=logger,
            save_interval=10**9,
            evaluators=(evaluators if not FAST_MODE else {}),
            show_progress=(not FAST_MODE),
        )
        total_steps_done += n_ep_steps

        # ★ actor の更新回数を概算で積算（freeze 期間を差し引き）
        pre_freeze = int(current.get("freeze_actor_steps") or 0)
        aui = max(1, int(current.get("actor_update_interval", 1)))
        updates_this_epoch = max(0, n_ep_steps - pre_freeze) // aui
        actor_updates_done += updates_this_epoch

        # === 追加: 上限到達で安全に終了 ===
        if total_steps_done >= MAX_TOTAL_STEPS:
            print("[STOP] Reached MAX_TOTAL_STEPS.")
            break

        # —— ベスト更新判定（score と AE 条件の両軸） —— #
        try:
            _raise_temperature_min(algo, min_temp=float(os.getenv("MIN_TEMP", "0.08")))  # 評価直前に底上げ
            metrics = best_eval(algo, test_ds.episodes)
            td    = float(metrics.get("td_p95",  np.inf))
            ratio = float(metrics.get("vs_ratio_ema", np.inf))   # ★ EMA を使用
            dpol  = float(metrics.get("dpolicy_ema",  np.inf))   # ★
            beh90 = float(metrics.get("beh_p90",  np.inf))
            pstd  = float(metrics.get("policy_std_ema", np.inf)) # ★

            pen = 0.0
            if np.isfinite(ratio): pen += 1e3 * max(0.0, ratio - 2.0)
            if np.isfinite(dpol):  pen += 1e3 * max(0.0, dpol  - 0.06)
            if np.isfinite(beh90): pen += 1e3 * max(0.0, beh90 - BEH_OK)  # ← 閾値を揃える

            score = td + pen
        except Exception as e:
            print(f"[BEST] 評価に失敗（epoch={epoch}, step={total_steps_done}）: {e}")
            td = float("inf"); score = float("inf")
            ratio = dpol = beh90 = pstd = float("nan")

        print(
            f"[BEST] epoch={epoch} step={total_steps_done} "
            f"td_p95={td:.6f} ratio={ratio:.2f} dpolicy={dpol:.4f} beh90={beh90:.4f} pstd={pstd:.4f} score={score:.6f}"
        )
        _log_progress(epoch, total_steps_done, metrics, current, score)  # ← 最新だけメモリ保持

        # ★★★ 早期停止条件を緩和＋複合化 ★★★
        z95_now     = float(metrics.get("td_z95", np.inf))
        pstd_now    = float(metrics.get("policy_std_ema", np.inf))
        td_p95_now  = float(metrics.get("td_p95", np.inf))

        # 直近の p95 を current に保持して「上昇（+2%以上）」を判定
        prev_td_p95 = current.get("_prev_td_p95", np.nan)
        trend_up    = (np.isfinite(td_p95_now) and np.isfinite(prev_td_p95)
                       and td_p95_now > prev_td_p95 * 1.02)
        if np.isfinite(td_p95_now):
            current["_prev_td_p95"] = td_p95_now  # 次回用に更新

            # 早期停止の判定
            if (np.isfinite(z95_now) and z95_now > 4.5
                    and trend_up
                    and np.isfinite(pstd_now) and pstd_now < 0.05):
                z95_bad_streak += 1
            else:
                z95_bad_streak = 0

            if z95_bad_streak >= 5:
                print("[EARLY-STOP] z95>4.5 & trend↑ & policy_std<0.05 が5連続 → 強制停止")
                break

            # === AutoTuner による自動調整（AE条件に寄せる） ===
            auto_report = autotuner.update(algo=algo, current=current, metrics=metrics)

            # 反映（共通）
            _apply_phase_to_algo(algo, current)
            _maybe_set_value_clips(
                algo,
                q_abs=(current.get("q_clip_abs") or 50.0),
                td_abs=(current.get("td_clip_abs") or 50.0)
            )
            if auto_report:
                print(f"[AUTOTUNE] {auto_report} | cw={current.get('conservative_weight')}, "
                      f"nas={current.get('n_action_samples')}, aui={current.get('actor_update_interval')}, "
                      f"lr(a/c/t)={current.get('actor_lr'):.1e}/{current.get('critic_lr'):.1e}/{current.get('temp_lr'):.1e}")

            # === 追加：フェーズスケジューラ反映（昇降格） ===
            prev_phase = str(current.get("name") or current.get("phase") or "")
            sch_metrics = EpochMetrics(
                epoch=epoch,
                z95=float(metrics.get("td_z95", float("inf"))),
                td_p95=float(metrics.get("td_p95", float("inf"))),
                q_p50_ratio=float(metrics.get("vs_ratio_ema", float("nan"))),
                delta_policy=float(metrics.get("dpolicy_ema", float("nan"))),
                policy_std=float(metrics.get("policy_std_ema", float("nan"))),
            )
            phase_info = scheduler.update(sch_metrics)   # 昇降格（ヒステリシス付き）

            # === フェーズ切替時のみ、安全なサブセットを新既定へ合わせる ===
            if prev_phase != phase_info["phase"]:
                base = phase_info["params"]
                for k in ("tau", "critic_dropout", "actor_update_interval", "batch_size"):
                    current[k] = base.get(k, current.get(k))
                # backup 方針もフェーズ既定へ合わせる（Phase1=soft / Phase2,3=max）
                if "phase1" in phase_info["phase"]:
                    current.update(_force_backup_mode("soft"))
                else:
                    current.update(_force_backup_mode("max"))

            # AutoTunerの変更は尊重し、空きを埋める
            for k, v in phase_info["params"].items():
                if k not in current:
                    current[k] = v
            current["name"] = phase_info["phase"]

            # 昇格直後（Phase2入り）で actor のフリーズを解除（序盤だけ守って早めに動かす）
            if prev_phase != phase_info["phase"] and "phase2" in phase_info["phase"]:
                current["freeze_actor_steps"] = 0

            # 粘着化された max backup フラグが立っていれば最終的に上書き
            if current.get("_force_max_backup", False):
                current["soft_q_backup"] = False
                current["max_q_backup"]  = True
            if current.get("_ratio_ok_streak", 0) >= 2:
                current["_force_max_backup"] = False

            _apply_phase_to_algo(algo, current)
            _maybe_set_value_clips(
                algo,
                q_abs=(current.get("q_clip_abs") or 50.0),
                td_abs=(current.get("td_clip_abs") or 50.0)
            )
            print(f"[PHASE][APPLIED] now={current['name']}")

            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_step = total_steps_done
                ok = try_export_learnable(algo, best_path_tmp)

                if ok:
                    print(f"[BEST] 更新: epoch={epoch} (step={total_steps_done}) → learnable を保存: {best_path_tmp}")
                else:
                    print("[BEST] スナップショット保存に失敗（次エポックで再挑戦）")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[EARLY-STOP] no improvement for {no_improve} epochs → stop.")
                    break

            # （E）AE 三条件
            PSTD_AE = 0.10  # 仕様E: policy_std ≥ 0.10
            ae_ok = (
                (np.isfinite(td)    and td    <= 3.5) and
                (np.isfinite(ratio) and 0.7   <= ratio <= 1.5) and
                (np.isfinite(pstd)  and pstd  >= PSTD_AE) and
                (np.isfinite(dpol)  and dpol  <= 0.06)
            )
            if ae_ok and score < best_score_ae:
                best_score_ae = score
                best_epoch_ae = epoch
                best_step_ae = total_steps_done
                ok = try_export_learnable(algo, best_path_ae)
                if ok:
                    print(f"[BEST][AE] 更新: epoch={epoch} (step={total_steps_done}) → learnable を保存: {best_path_ae}")

            # —— D: 自動調整ルール —— #
            # 安定域達成チェック（td_p95≤3 かつ 0.7≤ratio≤1.5）
            is_stable = (np.isfinite(td) and td <= 3.0) and (np.isfinite(ratio) and 0.7 <= ratio <= 1.5)
            stable_hist.append(bool(is_stable))

            # ルール適用（学習率は下限/上限でガード）
            def _clip(v, lo, hi):
                return float(max(lo, min(hi, v)))

            # TD スパイク
            if np.isfinite(metrics.get("td_z95", np.inf)) and metrics["td_z95"] > 3.0:
                current["critic_lr"] = _clip(current["critic_lr"] * 0.5, 1e-6, 3e-4)
                current["conservative_weight"] = min(current["conservative_weight"] + 4.0, 64.0)
                current["n_action_samples"] = min(int(current["n_action_samples"]) + 4, 32)

            # Q スケール大/小（cwの上限だけ 64 に拡張）
            if np.isfinite(ratio) and ratio > 2.0:
                current["conservative_weight"] = min(current["conservative_weight"] + 2.0, 64.0)
                current["critic_lr"] = _clip(current["critic_lr"] * 0.8, 3e-6, 3e-4)
            elif np.isfinite(ratio) and ratio < 0.5:
                current["conservative_weight"] = max(current["conservative_weight"] - 1.0, 2.0)
                current["actor_lr"] = _clip(current["actor_lr"] * 1.2, 3e-6, 3e-4)

            # Δpolicy>0.06 → 学習率を倍率で絞る（仕様D）
            if np.isfinite(dpol) and dpol > 0.06:
                current["actor_lr"] = _clip(current["actor_lr"] * 0.7, 3e-6, 3e-4)
                current["temp_lr"]  = _clip(current["temp_lr"]  * 0.7, 1e-6, 3e-3)

            # 2/3 以上で安定扱い → 少し保守を緩める
            if sum(stable_hist) >= 2:
                current["conservative_weight"] = max(current["conservative_weight"] - 2.0, 16.0)
                current["n_action_samples"] = max(int(current["n_action_samples"]) - 4, 24)
                stable_hist.clear()

            # AE（仕様E）達成の判定
            ae_ok = (
                (np.isfinite(td)    and td    <= 3.5) and
                (np.isfinite(ratio) and 0.7   <= ratio <= 1.5) and
                (np.isfinite(pstd)  and pstd  >= 0.10) and
                (np.isfinite(dpol)  and dpol  <= 0.06)
            )
            ae_ok_consec = (ae_ok_consec + 1) if ae_ok else 0

            # ★ actor が十分に更新されるまで AE 早期停止を抑止
            if (actor_updates_done >= MIN_ACTOR_UPDATES_BEFORE_AE) and (ae_ok_consec >= AE_CONSEC_REQ):
                print(f"[EARLY-STOP] AE条件を{AE_CONSEC_REQ}連続達成 "
                      f"+ actor_updates={actor_updates_done}≥{MIN_ACTOR_UPDATES_BEFORE_AE} → 終了")
                break

            def _any_bad(*vals):
                return any((not np.isfinite(v)) or np.isnan(v) or np.isinf(v) for v in vals)

            if _any_bad(td, ratio, dpol, beh90, pstd):
                print("[SAFETY] NaN/Inf detected in metrics → halve lrs & increase cw once.")
                current["actor_lr"]  = max(current["actor_lr"] * 0.5, 1e-6)
                current["critic_lr"] = max(current["critic_lr"] * 0.5, 1e-6)
                current["conservative_weight"] = min(current["conservative_weight"] + 2.0, 32.0)
                try:
                    if os.path.exists(best_path_ae):
                        shutil.copy2(best_path_ae, os.path.join(RUN_DIR, "learnable.d3"))
                        print("[SAFETY] rolled back to AE-best.")
                    elif os.path.exists(best_path_tmp):
                        shutil.copy2(best_path_tmp, os.path.join(RUN_DIR, "learnable.d3"))
                        print("[SAFETY] rolled back to best.")
                except Exception as _e:
                    print(f"[SAFETY][WARN] rollback failed: {_e}")

            # ← ここでフリーズ残量をエポック単位で減算（0で自動解除）
            if int(current.get("freeze_actor_steps", 0)) > 0:
                current["freeze_actor_steps"] = max(
                    0,
                    int(current["freeze_actor_steps"]) - n_ep_steps
                )

            # 固定間隔の軽量チェックポイント
            CKPT_EVERY = 5
            if (epoch % CKPT_EVERY) == 0:
                ck = os.path.join(RUN_DIR, f"ckpt_ep{epoch:03d}.d3")  # ← タイポ修正
                try:
                    algo.save_model(ck)  # weights-only
                    print(f"[CKPT] saved lightweight checkpoint: {ck}")
                except Exception as e:
                    print(f"[CKPT][WARN] save_model failed (skip): {e}")

    # —— 学習終了後の保存確定 —— #
    show_win_rate_from_raw_terminal(test_last_raw)

    learnable_path = os.path.join(RUN_DIR, "learnable.d3")
    model_path = os.path.join(RUN_DIR, "model_final.d3")
    learnable_final_path = os.path.join(RUN_DIR, "learnable_final.d3")

    # ❶ AEベスト ≫ 通常ベスト ≫ 現在モデル の優先で learnable を確定保存
    try:
        if os.path.exists(best_path_ae):
            shutil.copy2(best_path_ae, learnable_path)
            print(f"[BEST][FINAL] AE 条件ベストを確定保存 → {learnable_path}")
        elif os.path.exists(best_path_tmp):
            shutil.copy2(best_path_tmp, learnable_path)
            print(f"[BEST][FINAL] 暫定ベストを確定保存 → {learnable_path}")
        else:
            try_export_learnable(algo, learnable_path)
    except Exception as e:
        print(f"[BEST][FINAL][WARN] 確定保存に失敗: {e}")

    # ❷ weights-only も最終保存
    try:
        algo.save_model(model_path)
        print(f"[WEIGHTS] saved: {model_path} ({_size_mb(model_path):.1f} MB)")
    except Exception as e:
        print(f"[WEIGHTS][WARN] save_model failed (skip): {e}")

    # ❷' 最終状態の learnable も別名で保存（温度の下限を持ち上げてから）
    try:
        _raise_temperature_min(algo, min_temp=0.05)  # temp=0.0 で固着している場合の救済
    except Exception:
        pass
    try:
        try_export_learnable(algo, learnable_final_path)
    except Exception as e:
        print(f"[WEIGHTS][WARN] save learnable_final failed (skip): {e}")

    # === 学習完了サマリー（最終出力のみ） ====================
    def _confint_95_binom(k: int, n: int):
        import math
        if n <= 0:
            return (0.0, 0.0, 0.0)
        p = k / n
        se = math.sqrt(p * (1 - p) / n)
        lo, hi = max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
        return (p, lo, hi)

    def _safe_float(x, default=float("nan")):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    # エポックごとの詳細ファイルは作らず、最後にだけ要約を表示＋train_meta.jsonへ保存
    def write_summary(run_dir: str,
                      test_last_raw: np.ndarray,
                      best_epoch: int, best_step: int,
                      best_epoch_ae: int, best_step_ae: int,
                      learnable_path: str, model_path: str, meta_path: str):
        # 勝率（テストの生終端報酬ベース）
        n = int(test_last_raw.size)
        k = int((test_last_raw > 0).sum())
        p, lo, hi = _confint_95_binom(k, n)
        # 直近スナップショット
        last = _get_last_snapshot()
        final_metrics = last.get("metrics", {}) if isinstance(last.get("metrics"), dict) else {}
        final_current = last.get("current", {}) if isinstance(last.get("current"), dict) else {}

        # 付帯成果物の相対パスを揃える（存在しない場合は None）
        def _rel(path: str) -> str:
            try:
                return os.path.relpath(path, run_dir) if os.path.exists(path) else None
            except Exception:
                return None

        learnable_rel       = _rel(learnable_path)
        learnable_final_rel = _rel(os.path.join(run_dir, "learnable_final.d3"))
        model_rel           = _rel(model_path)
        scaler_rel          = _rel(os.path.join(run_dir, "scaler.npz"))
        obs_mask_rel        = _rel(os.getenv("OBS_MASK_PATH", os.path.join(run_dir, "obs_mask.npy")))
        summary_json_rel    = _rel(os.path.join(run_dir, "summary.json"))
        full_csv_rel        = _rel(os.path.join(run_dir, "full_log.csv"))
        full_jsonl_rel      = _rel(os.path.join(run_dir, "full_log.jsonl"))

        # ファイルサイズ（MB）
        sizes = {
            "learnable_mb":       _size_mb(learnable_path),
            "learnable_final_mb": _size_mb(os.path.join(run_dir, "learnable_final.d3")),
            "model_final_mb":     _size_mb(model_path),
            "scaler_mb":          _size_mb(os.path.join(run_dir, "scaler.npz")),
            "obs_mask_mb":        _size_mb(os.getenv("OBS_MASK_PATH", os.path.join(run_dir, "obs_mask.npy"))),
        }

        # 勝率
        win = {"k": k, "n": n, "p": p, "ci95": [lo, hi]}

        # 最終メトリクス（安全に取り出し）
        def _gf(key, default=float("nan")):
            try:
                v = final_metrics.get(key, default)
                return float(v) if v is not None else default
            except Exception:
                return default

        metrics_final = {
            "td_p95":        _gf("td_p95"),
            "td_z95":        _gf("td_z95"),
            "vs_ratio_ema":  _gf("vs_ratio_ema"),
            "dpolicy_ema":   _gf("dpolicy_ema"),
            "policy_std_ema":_gf("policy_std_ema"),
            "beh_p90":       _gf("beh_p90"),
            "temp":          _gf("temp"),
        }

        # 最終ハイパラ
        def _gc(key, default=None):
            try:
                v = final_current.get(key, default)
                return (float(v) if isinstance(v, (int, float)) else v)
            except Exception:
                return default

        hparams_final = {
            "phase":                 str(final_current.get("name") or final_current.get("phase") or "unknown"),
            "conservative_weight":   _gc("conservative_weight"),
            "n_action_samples":      int(final_current.get("n_action_samples") or 0),
            "actor_lr":              _gc("actor_lr"),
            "critic_lr":             _gc("critic_lr"),
            "temp_lr":               _gc("temp_lr"),
            "alpha_lr":              _gc("alpha_lr"),
            "tau":                   _gc("tau"),
            "actor_update_interval": int(final_current.get("actor_update_interval") or 0),
            "critic_dropout":        _gc("critic_dropout"),
            "soft_q_backup":         bool(final_current.get("soft_q_backup", False)),
            "max_q_backup":          bool(final_current.get("max_q_backup", False)),
            "batch_size":            int(final_current.get("batch_size") or 0),
        }

        # 可能なら指紋（artifact_fingerprint）
        fingerprints = {}
        try:
            if compute_scaler_sha and scaler_rel:
                fingerprints["scaler_sha"] = compute_scaler_sha(os.path.join(run_dir, scaler_rel))
        except Exception:
            pass
        try:
            if compute_vocab_sha:
                # 既知の候補から vocab を探す
                _cands = [
                    os.path.join(os.path.dirname(os.getenv("D3RLPY_DATASET", "")), "v2", "card_id2idx.json"),
                    os.path.join(os.path.dirname(os.getenv("D3RLPY_DATASET", "")), "card_id2idx.json"),
                    "card_id2idx.json",
                ]
                for _p in _cands:
                    if _p and os.path.exists(_p):
                        fingerprints["vocab_sha"] = compute_vocab_sha(_p)
                        break
        except Exception:
            pass
        try:
            if compute_schema_sha and os.path.exists("action_types.json"):
                fingerprints["schema_sha"] = compute_schema_sha("action_types.json")
        except Exception:
            pass

        # obs_mask の監査（残存 True 次元）
        try:
            _mask_path = os.getenv("OBS_MASK_PATH", os.path.join(run_dir, "obs_mask.npy"))
            if os.path.exists(_mask_path):
                _mask = np.load(_mask_path, allow_pickle=False).astype(np.bool_)
                mask_info = {"len": int(_mask.size), "true": int(_mask.sum()), "drop": int((~_mask).sum())}
            else:
                mask_info = None
        except Exception:
            mask_info = None

        meta = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run": {
                "dir": run_dir,
                "fast_mode": bool(int(os.getenv("FAST_MODE", "1"))),
                "device": final_current.get("device"),
                "gamma": final_current.get("gamma", DEFAULT_GAMMA),
                "seed": 42,
                "dataset": os.getenv("D3RLPY_DATASET"),
                "exp_name": os.getenv("D3RLPY_EXPNAME", "CQL"),
                "total_steps_done": int(last.get("step") or 0),
            },
            "win_rate": win,
            "best": {
                "score_min": _safe_float(last.get("score")),
                "epoch": int(best_epoch or 0),
                "step": int(best_step or 0),
                "ae_epoch": int(best_epoch_ae or 0),
                "ae_step": int(best_step_ae or 0),
            },
            "metrics_final": metrics_final,
            "hparams_final": hparams_final,
            "artifacts": {
                "learnable": learnable_rel,
                "learnable_final": learnable_final_rel,
                "model_final": model_rel,
                "scaler": scaler_rel,
                "obs_mask": obs_mask_rel,
                "summary_json": summary_json_rel,
                "full_log_csv": full_csv_rel,
                "full_log_jsonl": full_jsonl_rel,
            },
            "filesize_mb": sizes,
            "fingerprints": fingerprints or None,
            "obs_mask_info": mask_info,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

        print("\n=== TRAIN META ===")
        print(f"WinRate: {k}/{n} = {p:.3f}  [95% CI: {lo:.3f}–{hi:.3f}]")

        # 欠損に強い取り出し（KeyError回避）
        _mf  = metrics_final or {}
        _td95 = float(_mf.get("td_p95", float("nan")))
        _z95  = float(_mf.get("td_z95", float("nan")))
        _vsr  = float(_mf.get("vs_ratio_ema", float("nan")))
        _dpe  = float(_mf.get("dpolicy_ema", float("nan")))
        _pse  = float(_mf.get("policy_std_ema", float("nan")))

        print(
            f"Final td_p95={_td95:.3f}, z95={_z95:.2f}, "
            f"vs_ratio_ema={_vsr:.2f}, dpolicy_ema={_dpe:.4f}, "
            f"policy_std_ema={_pse:.4f}"
        )
        print(f"Saved train_meta.json → {meta_path}")

# === 追加: summary.json を必ず保存（ファストモードでも） ===
        try:
            _last = _get_last_snapshot() if ' _get_last_snapshot' in globals() or '_get_last_snapshot' in locals() else {}
            _cur  = (_last.get("current") or {}) if isinstance(_last, dict) else {}
            _met  = (_last.get("metrics") or {}) if isinstance(_last, dict) else {}

            # === 追加: エポックごとのコンパクトサマリーを構築（full_log.csv から） ===
            _epochs_compact, _epochs_meta = [], {"source": os.path.join(RUN_DIR, "full_log.csv"), "stride": 1, "count": 0}
            try:
                _csv_path = os.path.join(RUN_DIR, "full_log.csv")
                if os.path.exists(_csv_path):
                    import csv
                    with open(_csv_path, "r", encoding="utf-8") as _fh:
                        _rows = list(csv.DictReader(_fh))
                    _epochs_meta["count"] = len(_rows)
                    _max_keep = 400
                    _stride = max(1, len(_rows) // _max_keep)  # 多すぎる場合は間引き
                    _epochs_meta["stride"] = _stride
                    for i, r in enumerate(_rows):
                        if (i % _stride) != 0:
                            continue
                        def _f(key, default=None, cast=float):
                            v = r.get(key)
                            if v is None or v == "":
                                return default
                            try:
                                return cast(v)
                            except Exception:
                                try:
                                    return cast(v.replace(",", ""))
                                except Exception:
                                    return default
                        _epochs_compact.append({
                            "epoch": _f("epoch", 0, int),
                            "step": _f("step", 0, int),
                            "score": _f("score"),
                            "td_p95": _f("m_td_p95"),
                            "td_z95": _f("m_td_z95"),
                            "vs_ratio_ema": _f("m_vs_ratio_ema"),
                            "dpolicy_ema": _f("m_dpolicy_ema"),
                            "policy_std_ema": _f("m_policy_std_ema"),
                            "beh_p90": _f("m_beh_p90"),
                            "temp": _f("m_temp"),
                            "phase": r.get("c_phase"),
                            "cw": _f("c_cw"),
                            "nas": _f("c_nas", 0, int),
                            "aui": _f("c_aui", 0, int),
                            "tau": _f("c_tau"),
                            "soft_q_backup": (r.get("c_soft_q_backup") in ("True", "true", "1", 1, True)),
                            "max_q_backup": (r.get("c_max_q_backup") in ("True", "true", "1", 1, True)),
                            "ts": r.get("ts"),
                            "note": r.get("note"),
                        })
            except Exception:
                pass

            # 追加: 全エポック履歴を summary.json に同梱（CSV 不要）
            try:
                epochs_all = _get_epochs_all()
            except Exception:
                epochs_all = None
            if not epochs_all:
                # フォールバック: full_log.jsonl から復元
                try:
                    _jsonl = os.path.join(run_dir, 'full_log.jsonl')
                    if os.path.exists(_jsonl):
                        epochs_all = []
                        with open(_jsonl, 'r', encoding='utf-8') as _fh:
                            for _line in _fh:
                                try:
                                    _rec = json.loads(_line)
                                    epochs_all.append({
                                        'epoch': int(_rec.get('epoch', 0)),
                                        'step': int(_rec.get('step', 0)),
                                        'metrics': _rec.get('metrics', {}),
                                        'current': _rec.get('current', {}),
                                        'score': _rec.get('score', None),
                                        'ts': _rec.get('ts', ''),
                                        'note': _rec.get('note', ''),
                                    })
                                except Exception:
                                    pass
                except Exception:
                    epochs_all = []
            summary = {
                            "win_rate": {"k": int(k), "n": int(n), "p": float(p), "ci95": [float(lo), float(hi)]},
                            "final_metrics": {
                                "td_p95": float(_met.get("td_p95", _td95)),
                                "td_z95": float(_met.get("td_z95", _z95)),
                                "vs_ratio_ema": float(_met.get("vs_ratio_ema", _vsr)),
                                "dpolicy_ema": float(_met.get("dpolicy_ema", _dpe)),
                                "policy_std_ema": float(_met.get("policy_std_ema", _pse)),
                                "beh_p90": float(_met.get("beh_p90", float("nan"))),
                            },
                            "final_hparams": {
                                "phase": _cur.get("name") or _cur.get("phase") or "unknown",
                                "conservative_weight": float(_cur.get("conservative_weight", float("nan"))),
                                "n_action_samples": int(_cur.get("n_action_samples", 0)),
                                "actor_update_interval": int(_cur.get("actor_update_interval", 0)),
                                "actor_lr": float(_cur.get("actor_lr", float("nan"))),
                                "critic_lr": float(_cur.get("critic_lr", float("nan"))),
                                "temp_lr": float(_cur.get("temp_lr", float("nan"))),
                                "tau": float(_cur.get("tau", float("nan"))),
                                "soft_q_backup": bool(_cur.get("soft_q_backup", False)),
                                "max_q_backup": bool(_cur.get("max_q_backup", False)),
                            },
                            "epochs": (_get_epochs_all() if ('_get_epochs_all' in globals() or '_get_epochs_all' in locals()) else []),  # ← 追加
                            "epochs_ordered": (  # ← 追加: epoch, step で昇順ソートしたビュー
                                sorted(_get_epochs_all(), key=lambda r: (r.get("epoch", 0), r.get("step", 0)))
                                if ('_get_epochs_all' in globals() or '_get_epochs_all' in locals()) else []
                            ),
                            "artifacts": {
                                "run_dir": RUN_DIR,
                                "learnable": learnable_path,
                                "weights": model_path,
                                "train_meta": meta_path,
                                "progress_csv": os.path.join(RUN_DIR, "progress.csv"),
                            },
                        }

            # 追加: epochs を同梱
            summary['epochs'] = epochs_all if epochs_all is not None else []
            _summary_path = os.path.join(RUN_DIR, "summary.json")
            with open(_summary_path, "w", encoding="utf-8") as _f:
                json.dump(summary, _f, ensure_ascii=False, indent=2, sort_keys=True)
            print(f"summary.json written: {_summary_path}")
        except Exception as _e:
            print(f"[SUMMARY][WARN] failed to write summary.json: {type(_e).__name__}: {_e}")

    # 付帯ファイルのコピー（存在すれば）
    for src in (ID_MAP_SRC, ACTION_TYPES_SRC):
        try:
            if src and os.path.exists(src):
                dst = os.path.join(RUN_DIR, os.path.basename(src))
                shutil.copy2(src, dst)
                print(f"[COPY] {src} → {dst}")
        except Exception as e:
            print(f"[COPY][WARN] {src}: {e}")

    # train_meta.json を書く
    meta_out = os.path.join(RUN_DIR, "train_meta.json")
    write_summary(
        run_dir=RUN_DIR,
        test_last_raw=test_last_raw,
        best_epoch=best_epoch, best_step=best_step,
        best_epoch_ae=best_epoch_ae, best_step_ae=best_step_ae,
        learnable_path=learnable_path, model_path=model_path, meta_path=meta_out
    )

    print("\n=== DONE ===")
    print(f"Artifacts: {RUN_DIR}")
    return


if __name__ == "__main__":
    main()
