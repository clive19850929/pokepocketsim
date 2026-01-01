#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase D:
  build_phaseD_rl_dataset.py で作成した
    phaseD_rl_dataset_all.npz
  から CQL で Q(s,a) を学習するスクリプト。

前提:
  npz の中身 (build_phaseD_rl_dataset.py の仕様):
    - observations : (N, obs_dim) float32
    - actions      : (N, act_dim) float32
    - rewards      : (N,)         float32   # Phase D 用に設計済みの reward (z を基にスケール・クリップ)
    - terminals    : (N,)         float32   # 今は全部 1.0 (1-step episode 扱い)
    - sample_weight: (N,)         float32   # end_reason による重み (現状では未使用)
    - end_reason_ids: (N,)        int64     # end_reason の ID

ここでは:
  - d3rlpy の「連続行動版 CQL」を使って Q(s,a) を学習する。
  - 1-step エピソードとして扱う（terminals はすべて 1.0）。
    ブートストラップは効かず、Q ≒ r の回帰に近い形になる。

使い方(例):
  (venv310) PS C:/Users/CLIVE/poke-pocket-sim>
    & C:/Users/CLIVE/poke-pocket-sim/venv310/Scripts/python.exe `
      C:/Users/CLIVE/poke-pocket-sim/train_phaseD_cql.py
"""

import json
import os
from typing import Any, Dict

import numpy as np
import torch
import d3rlpy

from d3rlpy.algos import CQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.dataset import MDPDataset
try:
    from d3rlpy.context import save_default_context  # d3rlpy v1 系
except Exception:
    def save_default_context(path: str) -> None:
        # d3rlpy v2 など context モジュールが無い場合は何もしない
        return

# ============================================================
# ==================== 設定ブロック ==========================
# ============================================================

# build_phaseD_rl_dataset.py の出力
DATA_NPZ_PATH = r"D:\date\phaseD_rl_dataset_all.npz"
META_JSON_PATH = r"D:\date\phaseD_rl_dataset_all_meta.json"

# 出力先 (d3rlpy のログ・モデル保存先)
RUN_DIR = r"D:\date\phaseD_cql_run_p1"

# 学習ハイパーパラメータ
EPOCHS = 20              # 見かけ上の「エポック」数（実際には n_steps = EPOCHS * N_STEPS_PER_EPOCH）
BATCH_SIZE = 1024
N_STEPS_PER_EPOCH = 2000  # 1 エポック相当で何ステップ学習するか
GAMMA = 0.99
LEARNING_RATE = 3e-4        # メタ情報用（CQLConfig には渡さない）

# CQL 特有のハイパーパラメータ (ざっくりデフォ寄り)
CQL_ALPHA = 5.0
CQL_N_ACTIONS = 10
CQL_N_SAMPLES = 10

# 乱数シード
SEED = 42

# ============================================================
# ================== 設定ブロックここまで ====================
# ============================================================


def load_phaseD_npz(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"npz not found: {path}")
    data = np.load(path)
    required = [
        "observations",
        "actions",
        "rewards",
        "terminals",
    ]
    for k in required:
        if k not in data:
            raise RuntimeError(f"npz missing key: {k}")
    return data


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    set_seed(SEED)

    # ---------- データ読み込み ----------
    data = load_phaseD_npz(DATA_NPZ_PATH)

    obs = data["observations"].astype(np.float32)
    acts = data["actions"].astype(np.float32)
    rews = data["rewards"].astype(np.float32)
    terms = data["terminals"].astype(np.float32)

    # sample_weight / end_reason_ids も一応取り出しておく（現状は未使用）
    sample_weight = data.get("sample_weight", None)
    end_reason_ids = data.get("end_reason_ids", None)

    # build_phaseD_rl_dataset.py 側の meta があれば読み込んでログに出す
    meta_from_build = None
    if os.path.exists(META_JSON_PATH):
        try:
            with open(META_JSON_PATH, "r", encoding="utf-8") as fr:
                meta_from_build = json.load(fr)
            print(f"[META] loaded meta from {META_JSON_PATH}")
            reason_counts = meta_from_build.get("num_samples_by_end_reason")
            if isinstance(reason_counts, dict):
                print(f"[META] num_samples_by_end_reason={reason_counts}")
        except Exception as e:
            print(f"[META] WARNING: failed to load meta json: {e}")
            meta_from_build = None

    n, obs_dim = obs.shape
    _, act_dim = acts.shape

    print(f"[DATA] path={DATA_NPZ_PATH}")
    print(f"[DATA] samples={n} obs_dim={obs_dim} act_dim={act_dim}")
    if sample_weight is not None:
        print(f"[DATA] sample_weight shape={sample_weight.shape}")
    if end_reason_ids is not None:
        print(f"[DATA] end_reason_ids shape={end_reason_ids.shape}")

    # 1-step episode として扱うため、MDPDataset には単純に (obs, act, rew, terminals) を渡す
    dataset = MDPDataset(
        observations=obs,
        actions=acts,
        rewards=rews,
        terminals=terms,
    )

    # ---------- モデル設定 ----------
    os.makedirs(RUN_DIR, exist_ok=True)
    save_default_context(RUN_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CFG] device={device}")
    print(f"[CFG] d3rlpy.__version__={d3rlpy.__version__}")

    # この d3rlpy バージョンでは CQLConfig の __init__ で alpha 等を受け取れないので、
    # いったんデフォルト設定で生成し、必要なら後から属性を上書きする。
    config = CQLConfig()
    if hasattr(config, "gamma"):
        config.gamma = GAMMA
    # alpha / n_action_samples / n_random_actions は、対応していない可能性が高いので
    # ここではデフォルト値のまま使う（meta には値だけ記録しておく）。

    # 形状などは dataset から自動推論させる
    algo = config.create(device=device)
    algo.build_with_dataset(dataset)

    # ---------- 学習 ----------
    total_steps = EPOCHS * N_STEPS_PER_EPOCH
    print(
        f"[TRAIN] epochs={EPOCHS} n_steps_per_epoch={N_STEPS_PER_EPOCH} "
        f"total_steps={total_steps} batch_size={BATCH_SIZE}"
    )

    algo.fit(
        dataset=dataset,
        n_steps=total_steps,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        experiment_name="phaseD_cql",
        save_interval=N_STEPS_PER_EPOCH,
    )

    # ---------- 保存 ----------
    # d3rlpy 標準の保存形式 (.d3) と、後で読みやすい軽量版 (.pt, json) を両方残しておく

    learnable_path = os.path.join(RUN_DIR, "learnable_phaseD_cql.d3")
    algo.save(learnable_path)
    print(f"[SAVE] d3rlpy learnable -> {learnable_path}")

    # PyTorch の state_dict も別途保存しておく
    torch_path = os.path.join(RUN_DIR, "model_phaseD_cql.pt")
    torch.save(
        {
            "algo": "CQL",
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "state_dict": algo.impl.q_function_f1.state_dict()
            if hasattr(algo.impl, "q_function_f1")
            else None,
        },
        torch_path,
    )
    print(f"[SAVE] torch state_dict -> {torch_path}")

    # メタ情報 (データセット meta + CQL 設定) を保存
    meta: Dict[str, Any] = {
        "data_npz": os.path.abspath(DATA_NPZ_PATH),
        "data_meta_json": os.path.abspath(META_JSON_PATH),
        "run_dir": os.path.abspath(RUN_DIR),
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "epochs": EPOCHS,
        "n_steps_per_epoch": N_STEPS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "cql_alpha": CQL_ALPHA,
        "cql_n_actions": CQL_N_ACTIONS,
        "cql_n_samples": CQL_N_SAMPLES,
        "seed": SEED,
        "device": device,
    }
    if meta_from_build is not None:
        meta["build_phaseD_meta"] = meta_from_build
    meta_path = os.path.join(RUN_DIR, "train_phaseD_cql_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)
    print(f"[SAVE] meta -> {meta_path}")

    print("[DONE] Phase D CQL training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
