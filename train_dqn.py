import json
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DQNConfig  # or DiscreteCQLConfig
from dataset_loader import load_jsonl_dataset
import os

DATA_PATH = "log/all_matches_ml_converted.jsonl"
SCALER_PATH = os.path.join("D:\\date", "scaler.npz")  # ★ B案: 前処理で保存した scaler.npz の場所

def load_scaler(path):
    sc = np.load(path)
    mean = sc["mean"].astype(np.float32)
    std = sc["std"].astype(np.float32)
    clip_min = float(sc["clip_min"])
    clip_max = float(sc["clip_max"])
    post = sc["post_scale"].astype(np.float32) if "post_scale" in sc.files else None
    return mean, std, clip_min, clip_max, post

def preprocess(obs_batch, mean, std, clip_min, clip_max, post):
    y = (obs_batch.astype(np.float32) - mean) / np.maximum(std, 1e-6)
    y = np.clip(y, clip_min, clip_max)
    if post is not None:
        y = y * post  # ★ B案の心臓部：z-score の後に重み付け
    return y

mean, std, clip_min, clip_max, post = load_scaler(SCALER_PATH)

print("データセット読み込み中...")
dataset = load_jsonl_dataset(DATA_PATH)
observations = np.array(dataset['observations'])
actions = np.array(dataset['actions'])
rewards = np.array(dataset['rewards'])
terminals = np.array(dataset['terminals'])

# データ分割
split = int(len(observations) * 0.9)

# === ★ B案: 学習/評価ともに同じ前処理を適用 ===
train_obs = preprocess(observations[:split], mean, std, clip_min, clip_max, post)
test_obs  = preprocess(observations[split:], mean, std, clip_min, clip_max, post)

train_dataset = MDPDataset(
    observations=train_obs,
    actions=actions[:split],
    rewards=rewards[:split],
    terminals=terminals[:split]
)
test_dataset = MDPDataset(
    observations=test_obs,
    actions=actions[split:],
    rewards=rewards[split:],
    terminals=terminals[split:]
)


max_choices = 0
with open('log/all_matches_ml_converted.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        max_choices = max(max_choices, len(record['legal_actions']))
print("全ターンでの最大選択肢数:", max_choices)

algo = DQNConfig().create(device="cpu", enable_ddp=False)

print("学習開始...")

steps = len(train_obs) * 10  # 訓練データ数 × 任意エポック
algo.fit(
    train_dataset,
    n_steps=steps,
)

# ↓↓↓ バリデーション精度の計算（B案前処理済みデータを使用）
print("バリデーション推論・精度計算中...")
test_observations = test_obs
test_actions = actions[split:]

preds = algo.predict(test_observations)
accuracy = (preds == test_actions).mean()
print(f"Validation Accuracy: {accuracy:.4f}")

algo.save_model("dqn_model.pt")
print("学習完了・モデル保存しました。")
