#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase B で生成した自己対戦教師データ (s_t, π_t, z) から、
policy+value 一体ネットワークを教師ありで学習するスクリプト。

入力:
  - prepare_selfplay_supervised.py で作成した JSONL:
      {
        "obs_vec": [...],                      # List[float],  長さ = obs_dim
        "action_candidates_vec": [[...], ...], # List[List[float]], 各候補の特徴ベクトル
        "pi": [...],                           # List[float], len == num_candidates
        "z": float,                            # 最終勝敗ラベル (+1 / -1 / 0)
        "end_reason": "PRIZE_OUT" など        # 終局理由ラベル（オプション）
      }

出力:
  - 学習済みモデル (torch.save した .pt ファイル)

使い方:
  python train_selfplay_supervised.py
  （パラメータはファイル先頭の「設定ブロック」で変更）
"""

import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# ==================== 設定ブロック ==========================
# ============================================================
# 入力データ（selfplay 教師データ JSONL）
DATA_PATH = r"D:\date\selfplay_supervised_dataset.jsonl"

# 出力モデルファイル
OUT_PATH = r"D:\date\selfplay_supervised_pv_gen000.pt"

# 学習ハイパーパラメータ
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
VALUE_COEF = 1.0

# 使用するサンプル数の上限（-1 なら全行）
MAX_SAMPLES = -1

# DataLoader workers（Windows では 0〜2 を推奨）
NUM_WORKERS = 0

# CUDA で AMP(混合精度)を使うかどうか
USE_AMP = False

# 乱数シード
SEED = 42

# 1 エポックあたりに学習で使うバッチ数の上限
#   - -1 ならそのエポック内の全バッチを使用
#   - 例えば 500 にすると、「ランダムシャッフルされた先頭 500 バッチ」だけを使う軽量モード
MAX_STEPS_PER_EPOCH = 0  # 0 以下なら「全バッチ使う」

# ミニバッチ単位の進捗ログを何バッチごとに出すか
LOG_INTERVAL = 1000

# end_reason に応じたサンプル重み（end_reason が無い場合は 1.0）
USE_END_REASON_WEIGHTS = True
END_REASON_WEIGHT_TABLE = {
    "PRIZE_OUT":   1.0,
    "BASICS_OUT":  1.0,
    "DECK_OUT":    0.1,
    "TIMEOUT":     0.5,
    "SUDDEN_DEATH": 0.7,
    "CONCEDE":     0.5,
    "UNKNOWN":     0.5,
}
DEFAULT_SAMPLE_WEIGHT = 1.0
# ============================================================
# ================== 設定ブロックここまで ====================
# ============================================================

class JSONLSelfplayDataset(Dataset):
    def __init__(self, path: str, max_samples: int = -1):
        self.path = path
        self.offsets: List[int] = []
        self._fh = None

        first_valid = None
        exp_obs_dim = None
        exp_cand_dim = None

        with open(path, "r", encoding="utf-8") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                obs = obj.get("obs_vec")
                cand = obj.get("action_candidates_vec")
                pi = obj.get("pi")
                z = obj.get("z")
                if not isinstance(obs, list) or not isinstance(cand, list) or not isinstance(pi, list):
                    continue
                if len(cand) == 0 or len(pi) == 0 or len(cand) != len(pi):
                    continue
                if not isinstance(z, (int, float)):
                    continue

                first_cand = cand[0]
                if not isinstance(first_cand, list) or not first_cand:
                    continue

                if exp_obs_dim is None or exp_cand_dim is None:
                    exp_obs_dim = len(obs)
                    exp_cand_dim = len(first_cand)
                else:
                    if len(obs) != exp_obs_dim:
                        continue
                    if len(first_cand) != exp_cand_dim:
                        continue

                self.offsets.append(off)
                if first_valid is None:
                    first_valid = obj
                if max_samples > 0 and len(self.offsets) >= max_samples:
                    break

        if not self.offsets or first_valid is None or exp_obs_dim is None or exp_cand_dim is None:
            raise RuntimeError(f"no valid samples found in {path}")

        self.obs_dim = int(exp_obs_dim)
        self.cand_dim = int(exp_cand_dim)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._fh is None:
            self._fh = open(self.path, "r", encoding="utf-8")
        self._fh.seek(self.offsets[idx])
        line = self._fh.readline()
        return json.loads(line)

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs_list: List[torch.Tensor] = []
    cand_list: List[torch.Tensor] = []
    pi_list: List[torch.Tensor] = []
    z_list: List[torch.Tensor] = []
    w_list: List[torch.Tensor] = []

    for obj in batch:
        obs = torch.tensor(obj["obs_vec"], dtype=torch.float32)
        cands = torch.tensor(obj["action_candidates_vec"], dtype=torch.float32)
        pi = torch.tensor(obj["pi"], dtype=torch.float32)
        z = torch.tensor([float(obj["z"])], dtype=torch.float32)

        weight_value = DEFAULT_SAMPLE_WEIGHT
        if USE_END_REASON_WEIGHTS:
            end_reason = obj.get("end_reason")
            if isinstance(end_reason, str) and end_reason:
                key = end_reason.upper()
                weight_value = float(END_REASON_WEIGHT_TABLE.get(key, DEFAULT_SAMPLE_WEIGHT))
        w = torch.tensor([weight_value], dtype=torch.float32)

        obs_list.append(obs)
        cand_list.append(cands)
        pi_list.append(pi)
        z_list.append(z)
        w_list.append(w)

    obs_batch = torch.stack(obs_list, dim=0)
    z_batch = torch.cat(z_list, dim=0)
    sample_weight = torch.cat(w_list, dim=0)
    return {
        "obs": obs_batch,
        "cands": cand_list,
        "pi": pi_list,
        "z": z_batch,
        "sample_weight": sample_weight,
    }

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, cand_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.obs_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.policy_fc1 = nn.Linear(hidden_dim + cand_dim, hidden_dim)
        self.policy_fc2 = nn.Linear(hidden_dim, 1)

        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor, cand_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        h = self.activation(self.obs_fc1(obs))
        h = self.activation(self.obs_fc2(h))

        logits_list: List[torch.Tensor] = []
        for i, cands in enumerate(cand_list):
            hi = h[i]
            num_actions = cands.size(0)
            hi_rep = hi.unsqueeze(0).expand(num_actions, -1)
            x = torch.cat([hi_rep, cands], dim=1)
            x = self.activation(self.policy_fc1(x))
            logits = self.policy_fc2(x).squeeze(-1)
            logits_list.append(logits)

        v = self.activation(self.value_fc1(h))
        v = self.value_fc2(v).squeeze(-1)
        return logits_list, v

def compute_loss(
    logits_list: List[torch.Tensor],
    values: torch.Tensor,
    pi_list: List[torch.Tensor],
    z_batch: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    value_coef: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_losses: List[torch.Tensor] = []
    value_losses: List[torch.Tensor] = []

    for i, (logits, pi, z) in enumerate(zip(logits_list, pi_list, z_batch)):
        log_probs = torch.log_softmax(logits, dim=-1)
        pi_norm = pi / (pi.sum() + 1e-8)
        policy_loss = -(pi_norm * log_probs).sum()

        value_loss = (values[i] - z).pow(2)

        if sample_weight is not None:
            w = sample_weight[i]
            policy_loss = policy_loss * w
            value_loss = value_loss * w

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

    if sample_weight is not None:
        w_sum = float(sample_weight.sum().item())
        if w_sum > 0.0:
            policy_loss_mean = torch.stack(policy_losses).sum() / sample_weight.sum()
            value_loss_mean = torch.stack(value_losses).sum() / sample_weight.sum()
        else:
            policy_loss_mean = torch.stack(policy_losses).mean()
            value_loss_mean = torch.stack(value_losses).mean()
    else:
        policy_loss_mean = torch.stack(policy_losses).mean()
        value_loss_mean = torch.stack(value_losses).mean()

    total_loss = policy_loss_mean + value_coef * value_loss_mean
    return total_loss, policy_loss_mean, value_loss_mean

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    set_seed(SEED)

    dataset = JSONLSelfplayDataset(DATA_PATH, max_samples=MAX_SAMPLES)
    print(
        f"[DATA] path={DATA_PATH} samples={len(dataset)} "
        f"obs_dim={dataset.obs_dim} cand_dim={dataset.cand_dim}"
    )
    if USE_END_REASON_WEIGHTS:
        print("[CFG] USE_END_REASON_WEIGHTS=True")
        print(f"[CFG] END_REASON_WEIGHT_TABLE={END_REASON_WEIGHT_TABLE}")
        print(f"[CFG] DEFAULT_SAMPLE_WEIGHT={DEFAULT_SAMPLE_WEIGHT}")
    else:
        print("[CFG] USE_END_REASON_WEIGHTS=False (all samples weight=1.0)")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = PolicyValueNet(dataset.obs_dim, dataset.cand_dim, hidden_dim=HIDDEN_DIM)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    global_step = 0
    num_batches_per_epoch = len(loader)
    print(f"[TRAIN] batches_per_epoch={num_batches_per_epoch} batch_size={BATCH_SIZE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_pl = 0.0
        running_vl = 0.0
        epoch_steps = 0

        max_steps = MAX_STEPS_PER_EPOCH if MAX_STEPS_PER_EPOCH > 0 else num_batches_per_epoch

        for batch_idx, batch in enumerate(loader, start=1):
            if batch_idx > max_steps:
                break

            obs = batch["obs"].to(device)
            cands = [x.to(device) for x in batch["cands"]]
            pi_list = [x.to(device) for x in batch["pi"]]
            z_batch = batch["z"].to(device)
            sample_weight = batch.get("sample_weight")
            if sample_weight is not None:
                sample_weight = sample_weight.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_list, values = model(obs, cands)
                total_loss, pl, vl = compute_loss(
                    logits_list,
                    values,
                    pi_list,
                    z_batch,
                    sample_weight=sample_weight,
                    value_coef=VALUE_COEF,
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(total_loss.item())
            running_pl += float(pl.item())
            running_vl += float(vl.item())
            global_step += 1
            epoch_steps += 1

            if (batch_idx % LOG_INTERVAL == 0) or (batch_idx == max_steps):
                avg_loss = running_loss / max(epoch_steps, 1)
                avg_pl = running_pl / max(epoch_steps, 1)
                avg_vl = running_vl / max(epoch_steps, 1)
                print(
                    f"[EPOCH {epoch}] step={global_step} "
                    f"batch={batch_idx}/{num_batches_per_epoch} "
                    f"loss={avg_loss:.6f} policy={avg_pl:.6f} value={avg_vl:.6f}"
                )

        epoch_loss = running_loss / max(epoch_steps, 1)
        epoch_pl = running_pl / max(epoch_steps, 1)
        epoch_vl = running_vl / max(epoch_steps, 1)
        print(
            f"[EPOCH {epoch} DONE] steps_in_epoch={epoch_steps} "
            f"loss={epoch_loss:.6f} policy={epoch_pl:.6f} value={epoch_vl:.6f}"
        )

    out_path = OUT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": dataset.obs_dim,
            "cand_dim": dataset.cand_dim,
            "hidden_dim": HIDDEN_DIM,
        },
        out_path,
    )
    print(f"[SAVE] model -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
