# --- ここからファイル全体（修正後・完全版） ---
"""
盤面状態ベクトル → 将来リターンを推定する ValueNet 学習スクリプト（Windows + CUDA 想定）
- torch.compile など未使用（この環境向けに削除）
- エピソード単位で学習/検証を分割してリークを防止（前処理の episode/split ファイルを利用）
- 早期終了 / HuberLoss / weight decay / LRウォームアップ+Cosine / 勾配クリップ で汎化を強化
- 大規模 npz をメモリマップ + カスタム Dataset で省メモリ読み込み
- scaler.npz があれば (x-mean)/std の正規化を適用
- 追加: オートチューニングモード（複数ハイパラを自動で試行し最良を採用）
- 追加: データ変更ガード（データ入替え/取り違えを検知して警告＆停止）
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from dataclasses import dataclass
import shutil

from value_net import ValueNet

# ======== 切り替えフラグ（True/Falseで制御） ========
OBS_MODE_PARTIAL = True
APPLY_OBS_MASK   = True
ENABLE_DISTILL   = True

def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return bool(int(v))
    except Exception:
        return default

OBS_MODE_PARTIAL = _env_flag("VALNET_PARTIAL", OBS_MODE_PARTIAL)
APPLY_OBS_MASK   = _env_flag("VALNET_APPLY_MASK", APPLY_OBS_MASK)
ENABLE_DISTILL   = _env_flag("VALNET_DISTILL", ENABLE_DISTILL)

PAIRED_FULL_STATES_KEY = os.environ.get("VALNET_FULL_KEY", "full_states")
teacher_env = os.environ.get("VALNET_TEACHER", "")
if teacher_env and (os.path.isabs(teacher_env) or os.sep in teacher_env):
    TEACHER_MODEL_PATH = teacher_env
else:
    TEACHER_MODEL_PATH = os.path.join("data", teacher_env or "value_full.pt")
DISTILL_EPOCHS         = int(os.environ.get("VALNET_DISTILL_EPOCHS", "5"))
DISTILL_LR             = float(os.environ.get("VALNET_DISTILL_LR", "1e-4"))
DISTILL_WD             = float(os.environ.get("VALNET_DISTILL_WD", "1e-6"))
DISTILL_CLIP           = float(os.environ.get("VALNET_DISTILL_CLIP", "1.0"))

# === Data Guard default behavior ==========================================
# "strict" : 不一致なら例外停止（厳格）
# "update" : 不一致なら基準(guard)を書き換えて続行（←既定）
# "force"  : チェックはするが必ず続行（ログだけ）
DATA_GUARD_MODE = "update"

def _write_data_guard(guard_path: str, meta: dict):
    """現在のデータメタをガードファイルに保存（既存はタイムスタンプ付きバックアップ）"""
    import json, shutil, datetime, os
    if os.path.exists(guard_path):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(guard_path, guard_path + f".bak.{ts}")
    os.makedirs(os.path.dirname(guard_path), exist_ok=True)
    with open(guard_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DATA-GUARD] 基準を更新しました -> {guard_path}")


# ======== オートチューニングON/OFF ========
AUTOTUNE = bool(int(os.environ.get("VALNET_AUTOTUNE", "1")))  # 既定: ON
# 許容する「改善」とみなすしきい値（val loss がこの値だけ下がれば改善）
AUTOTUNE_TOL = float(os.environ.get("VALNET_AUTOTUNE_TOL", "1e-4"))

print("[MODE] OBS_MODE_PARTIAL =", OBS_MODE_PARTIAL)
print("[MODE] APPLY_OBS_MASK   =", APPLY_OBS_MASK)
print("[MODE] ENABLE_DISTILL   =", ENABLE_DISTILL)
print("[MODE] AUTOTUNE        =", AUTOTUNE)
if ENABLE_DISTILL:
    print("[MODE] DISTILL: teacher =", TEACHER_MODEL_PATH, " key_in_npz =", PAIRED_FULL_STATES_KEY)

# ========================= 基本設定 =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 読み込みファイル名と期待次元 ===
NPZ_PATH = os.environ.get("VALNET_NPZ", os.environ.get("NPZ_PATH", r"D:\date\state_return_all.npz"))
EXPECTED_D = int(os.environ.get("VALNET_EXPECTED_D", "2448"))  # 期待する状態次元（既定2448）

# --- V1モード（完全情報・蒸留OFF・マスクOFF）なら v1 ディレクトリを自動参照（環境変数未指定時のみ） ---
if (not os.environ.get("VALNET_NPZ") and not os.environ.get("NPZ_PATH")
    and (not OBS_MODE_PARTIAL) and (not APPLY_OBS_MASK) and (not ENABLE_DISTILL)):
    _base = os.path.dirname(os.path.abspath(NPZ_PATH))
    _cand = os.path.join(_base, "v1", "state_return_all.npz")
    if os.path.exists(_cand):
        NPZ_PATH = _cand
        print(f"[AUTO] V1 mode detected -> NPZ_PATH={NPZ_PATH}")
    else:
        _cand2 = r"D:\date\v1\state_return_all.npz"
        if os.path.exists(_cand2):
            NPZ_PATH = _cand2
            print(f"[AUTO] V1 mode detected -> NPZ_PATH={NPZ_PATH}")

DATA_DIR = os.path.dirname(NPZ_PATH)
EP_IDS_PATH   = os.path.join(DATA_DIR, "episode_ids.npy")
EP_IDX_PATH   = os.path.join(DATA_DIR, "transition_episode_idx.npy")
SPLITS_PATH   = os.path.join(DATA_DIR, "splits.npz")

SCALER_PATH_CANDIDATES = [
    os.path.join(DATA_DIR, "scaler.npz"),
    os.path.join(DATA_DIR, "scaler_value.npz"),
]
OBS_MASK_PATH = os.path.join(DATA_DIR, "obs_mask.npy")
VOCAB_PATH = os.path.join(DATA_DIR, "card_id2idx.json")

def _load_vocab(path):
    import json
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            d = json.load(f)
            return {int(k): int(v) for k, v in d.items()}
        except Exception:
            return {}

vocab = _load_vocab(VOCAB_PATH)
if len(vocab) < 10:
    raise RuntimeError(
        f"[FATAL] 語彙が見つからない/小さすぎます: {VOCAB_PATH} "
        "（先に前処理で語彙を構築してから ValueNet を学習してください）"
    )
print(f"[CHECK] vocab ok: size={len(vocab)} from {VOCAB_PATH}")

# 再現性・ハイパラ（デフォルトは“基準線”用、オートチューン時は上書きされる）
SEED        = int(os.environ.get("VALNET_SEED", "42"))
BATCH_SIZE  = int(os.environ.get("VALNET_BS", "2048"))
EPOCHS      = int(os.environ.get("VALNET_EPOCHS", "40"))
LR          = float(os.environ.get("VALNET_LR", "3e-4"))
WEIGHT_DECAY= float(os.environ.get("VALNET_WD", "1e-4"))
CLIP_GRAD   = float(os.environ.get("VALNET_CLIP", "1.0"))
HUBER_DELTA = float(os.environ.get("VALNET_HUBER_DELTA", "1.0"))
PATIENCE    = int(os.environ.get("VALNET_PATIENCE", "5"))
VAL_RATIO_FALLBACK = float(os.environ.get("VALNET_VAL_RATIO", "0.2"))

LAMBDA_BELLMAN = float(os.environ.get("VALNET_LMB", "0.1"))
GAMMA          = float(os.environ.get("VALNET_GAMMA", "0.99"))

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
if DEVICE.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

np.random.seed(SEED)
torch.manual_seed(SEED)

# === 保存先 / 評価専用フラグ（EVAL_ONLY）/ 事前ロード ===
RUN_DIR    = os.environ.get("VALNET_RUN_DIR", os.path.join(DATA_DIR, "runs"))
os.makedirs(RUN_DIR, exist_ok=True)
EVAL_ONLY  = bool(int(os.environ.get("VALNET_EVAL_ONLY", "0")))
LOAD_MODEL = os.environ.get("VALNET_LOAD_MODEL", "")

# ========================= データ読み込み =========================
bundle = np.load(NPZ_PATH, mmap_mode="r")
required = {"states", "returns"}
missing = required.difference(set(bundle.files))
if missing:
    raise RuntimeError(f"[FATAL] NPZに必要キーがありません: 欠落={missing} / files={bundle.files} / path={NPZ_PATH}")

states_mm  : np.memmap = bundle["states"]
returns_mm : np.memmap = bundle["returns"]

N, D = int(states_mm.shape[0]), int(states_mm.shape[1])
print(f"[DATA] states shape = {states_mm.shape}, returns shape = {returns_mm.shape}")
print(f"[DEBUG] EXPECTED_D={EXPECTED_D}")

# ================== 厳密整合チェック ==================
if returns_mm.shape[0] != N:
    raise RuntimeError(f"[FATAL] データ件数不一致: states_N={N} vs returns_N={returns_mm.shape[0]}")
if D != EXPECTED_D:
    if APPLY_OBS_MASK:
        # マスク有効次元で判定（mask は states 長以上であること）
        mask_path = os.path.join(os.path.dirname(NPZ_PATH), "obs_mask.npy")
        try:
            obs_mask = np.load(mask_path)
        except Exception as e:
            raise RuntimeError(f"[FATAL] state_dim={D} と EXPECTED_D={EXPECTED_D} が不一致かつ、"
                               f"APPLY_OBS_MASK=True ですが obs_mask の読込に失敗しました: {mask_path}  err={e}")
        if obs_mask.ndim != 1:
            raise RuntimeError(f"[FATAL] obs_mask の次元が不正です: ndim={obs_mask.ndim} path={mask_path}")
        if obs_mask.shape[0] < D:
            raise RuntimeError(f"[FATAL] obs_mask の長さが states より短いです: mask={obs_mask.shape[0]}  states={D}  path={mask_path}")

        effective_d = int(np.count_nonzero(obs_mask[:D]))

        if effective_d == EXPECTED_D:
            print(f"[DATA-GUARD] 物理D={D} と EXPECTED_D={EXPECTED_D} は不一致ですが、"
                  f"obs_mask 有効次元={effective_d} のため続行します。mask={mask_path}")
        else:
            # 蒸留ONの場合は、学生入力Dはそのまま、教師入力のみ基準次元へ右ゼロパディングして続行を許容
            if ENABLE_DISTILL:
                teacher_base_d = int(os.environ.get("TEACHER_BASE_D", "305"))
                if D <= teacher_base_d:
                    print(f"[DATA-GUARD] 有効次元={effective_d} と EXPECTED_D={EXPECTED_D} が不一致ですが、"
                          f"ENABLE_DISTILL=True のため教師入力を teacher_base_d={teacher_base_d} に右ゼロパディングして続行します。学生入力D={D}")
                    # 後段（教師入力構築時）で参照するフラグ
                    PAD_TEACHER_TO_BASE_D = True
                else:
                    raise RuntimeError(f"[FATAL] 学生入力D={D} が教師基準次元より大きいです: teacher_base_d={teacher_base_d}")
            else:
                raise RuntimeError(f"[FATAL] 物理D={D}≠EXPECTED_D={EXPECTED_D} かつ obs_mask 有効次元={effective_d} も一致せず、"
                                   f"ENABLE_DISTILL=False のため続行不可。mask={mask_path} NPZ_PATH={NPZ_PATH}")
    else:
        raise RuntimeError(f"[FATAL] state_dim={D} が期待 {EXPECTED_D} と不一致。NPZ_PATH={NPZ_PATH}")

# scaler の形状チェック（mean/std 必須・D と一致）
scaler_path = next((p for p in SCALER_PATH_CANDIDATES if os.path.exists(p)), "")
if not scaler_path:
    raise RuntimeError(f"[FATAL] scaler が見つかりません: candidates={SCALER_PATH_CANDIDATES}")
sc = np.load(scaler_path)
mean, std = sc["mean"], sc["std"]
if mean.shape != (D,) or std.shape != (D,):
    raise RuntimeError(f"[FATAL] scaler 形状不一致: mean={mean.shape} std={std.shape} vs D={D}")
print(f"[INFO] scaler loaded: {scaler_path}")

# obs_mask があれば形状チェック（存在しなくても良いが、あれば D と一致必須）
if os.path.exists(OBS_MASK_PATH):
    obs_mask_arr = np.load(OBS_MASK_PATH).astype(np.bool_)
    if obs_mask_arr.ndim != 1 or obs_mask_arr.shape[0] != D:
        raise RuntimeError(f"[FATAL] obs_mask 形状不一致: {obs_mask_arr.shape} vs D={D}  path={OBS_MASK_PATH}")
    print(f"[INFO] obs_mask loaded: {OBS_MASK_PATH} (len={obs_mask_arr.size})")
else:
    print(f"[INFO] obs_mask not found (optional): {OBS_MASK_PATH}")
# =====================================================

# ===== DATA CHANGE GUARD =====
import json as _json_guard
import hashlib as _hashlib_guard
import time as _time_guard

GUARD_PATH = os.path.join(DATA_DIR, "dataset_guard.json")

def _sha256_head(path: str, head_mb: int = 16) -> str:
    h = _hashlib_guard.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(head_mb * 1024 * 1024))
        return h.hexdigest()
    except Exception:
        return ""

def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _json_guard.load(f)
    except Exception:
        return None

current_guard = {
    "npz_path": os.path.abspath(NPZ_PATH),
    "npz_hash_head16mb": _sha256_head(NPZ_PATH, 16),
    "states_shape": list(states_mm.shape),
    "returns_len": int(returns_mm.shape[0]),
    "expected_d": int(EXPECTED_D),
    "has_ep_idx": bool(os.path.exists(EP_IDX_PATH)),
    "ep_idx_len": int(np.load(EP_IDX_PATH).shape[0]) if os.path.exists(EP_IDX_PATH) else 0,
    "has_splits": bool(os.path.exists(SPLITS_PATH)),
    "scaler_path": next((p for p in SCALER_PATH_CANDIDATES if os.path.exists(p)), ""),
    "obs_mask_path": OBS_MASK_PATH if os.path.exists(OBS_MASK_PATH) else "",
    "mtime_npz": os.path.getmtime(NPZ_PATH) if os.path.exists(NPZ_PATH) else 0.0,
    "created_at": _time_guard.strftime("%Y-%m-%d %H:%M:%S"),
}

# 基本整合チェック（ここで致命的不整合は落とす）
problems = []
if current_guard["returns_len"] != states_mm.shape[0]:
    problems.append(f"returns 長さが states と不一致: states_N={states_mm.shape[0]} vs returns_N={current_guard['returns_len']}")
if current_guard["has_ep_idx"] and current_guard["ep_idx_len"] != states_mm.shape[0]:
    problems.append(f"transition_episode_idx.npy の長さが不一致: ep_idx={current_guard['ep_idx_len']} vs N={states_mm.shape[0]}")
if problems:
    detail = "\n  - ".join(problems)
    raise RuntimeError(f"[DATA-GUARD] 整合性問題を検出:\n  - {detail}\n[DATA-GUARD] 停止します。")

# 既存ガードと比較
prev_guard = _load_json(GUARD_PATH)
diffs = []
if prev_guard:
    def _cmp(key, label=None):
        a, b = prev_guard.get(key), current_guard.get(key)
        if a != b:
            diffs.append(f"{label or key}: prev={a} / now={b}")
    _cmp("npz_path", "NPZパス")
    _cmp("npz_hash_head16mb", "NPZハッシュ(先頭16MB)")
    _cmp("states_shape", "states形状")
    _cmp("returns_len", "returns件数")
    _cmp("expected_d", "期待次元EXPECTED_D")
    _cmp("has_ep_idx", "ep_idx有無")
    _cmp("ep_idx_len", "ep_idx長さ")

if not prev_guard:
    print("[DATA-GUARD] 既存ガードが見つかりません（初回）。この状態を正準として保存します。")
    _write_data_guard(GUARD_PATH, current_guard)
elif diffs:
    msg = "[DATA-GUARD] 前回とデータが異なります:\n  - " + "\n  - ".join(diffs)
    print(msg)
    if DATA_GUARD_MODE == "strict":
        raise RuntimeError(msg + "\n[DATA-GUARD] MODE=strict のため停止します。")
    elif DATA_GUARD_MODE == "force":
        print("[DATA-GUARD] MODE=force -> 基準無視で続行します（非推奨）。")
    elif DATA_GUARD_MODE == "update":
        _write_data_guard(GUARD_PATH, current_guard)
        print("[DATA-GUARD] MODE=update -> 新しいデータを基準として採用して続行します。")
    else:
        raise RuntimeError(msg + "\n[DATA-GUARD] 未知のモードのため停止します。")
else:
    print("[DATA-GUARD] データは基準と一致しています。")
# ===== /DATA CHANGE GUARD =====


# エピソードindex（あれば利用）
ep_idx_arr = np.load(EP_IDX_PATH).astype(np.int64) if os.path.exists(EP_IDX_PATH) else None

def build_next_and_done(N: int, ep_idx: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    next_idx = np.minimum(np.arange(N) + 1, N - 1).astype(np.int64)
    done = np.zeros(N, dtype=np.float32)
    if ep_idx is None:
        done[-1] = 1.0
        return next_idx, done
    same = (ep_idx[1:] == ep_idx[:-1])
    next_idx[:-1] = np.where(same, np.arange(N - 1) + 1, np.arange(N - 1))
    next_idx[-1]  = N - 1
    done[:-1] = (~same).astype(np.float32)
    done[-1]  = 1.0
    return next_idx, done

NEXT_IDX, DONE_VEC = build_next_and_done(N, ep_idx_arr)

# ---------- scaler 読み込み（あれば使用） ----------
scaler = None
for p in SCALER_PATH_CANDIDATES:
    if os.path.exists(p):
        s = np.load(p)
        mean = s["mean"].astype(np.float32, copy=False)
        std  = s["std"].astype(np.float32, copy=False)
        print(f"[DEBUG] scaler path={p}  mean.shape={mean.shape}  std.shape={std.shape}  D={D}")
        if mean.shape[0] != D or std.shape[0] != D:
            raise RuntimeError(f"[FATAL] scaler 形状不一致: mean/std={mean.shape}/{std.shape}, state_dim={D} @ {p}")
        scaler = {
            "mean": torch.from_numpy(mean),
            "std":  torch.from_numpy(std),
            "clip_min": float(s.get("clip_min", -5.0)),
            "clip_max": float(s.get("clip_max",  5.0)),
        }
        if DEVICE.type == "cuda":
            scaler["mean"] = scaler["mean"].to(DEVICE, non_blocking=True)
            scaler["std"]  = scaler["std"].to(DEVICE, non_blocking=True)
        print(f"[INFO] scaler loaded: {p}")
        break

def apply_scaler(x: torch.Tensor) -> torch.Tensor:
    if scaler is None:
        return x
    eps = 1e-6
    z = (x - scaler["mean"]) / (scaler["std"] + eps)
    return torch.clamp(z, scaler["clip_min"], scaler["clip_max"])

# ---------- 観測マスクのロード（必要時） ----------
obs_mask_tensor = None
if APPLY_OBS_MASK and os.path.exists(OBS_MASK_PATH):
    _mask = np.load(OBS_MASK_PATH).astype(np.float32)
    print(f"[DEBUG] obs_mask shape={_mask.shape}  D={D}")
    if _mask.shape[0] != D:
        raise RuntimeError(f"[FATAL] obs_mask 形状不一致: mask={_mask.shape} vs state_dim={D} @ {OBS_MASK_PATH}")
    obs_mask_tensor = torch.from_numpy(_mask).to(DEVICE)
    print(f"[INFO] obs_mask loaded: {OBS_MASK_PATH}")

# ---------- 分割情報の読み込み ----------
def load_split_indices() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    rng = np.random.default_rng(SEED)
    has_ep_ids = os.path.exists(EP_IDS_PATH)
    has_ep_idx = os.path.exists(EP_IDX_PATH)
    has_splits = os.path.exists(SPLITS_PATH)
    if has_ep_ids and has_ep_idx and has_splits:
        ep_ids = np.load(EP_IDS_PATH, allow_pickle=True)
        ep_idx = np.load(EP_IDX_PATH).astype(np.int64)
        splits = np.load(SPLITS_PATH, allow_pickle=True)
        ep_id_to_idx = {eid: i for i, eid in enumerate(ep_ids.tolist())}
        train_ep_ids = splits["train_ep_ids"]
        val_ep_ids   = splits["val_ep_ids"]
        test_ep_ids  = splits["test_ep_ids"] if "test_ep_ids" in splits.files else None
        train_idx_set = set(ep_id_to_idx[e] for e in train_ep_ids if e in ep_id_to_idx)
        val_idx_set   = set(ep_id_to_idx[e] for e in val_ep_ids   if e in ep_id_to_idx)
        test_idx_set  = set(ep_id_to_idx[e] for e in test_ep_ids  if (test_ep_ids is not None and e in ep_id_to_idx)) \
                        if test_ep_ids is not None else set()
        train_mask = np.isin(ep_idx, list(train_idx_set))
        val_mask   = np.isin(ep_idx, list(val_idx_set))
        test_mask  = np.isin(ep_idx, list(test_idx_set)) if test_idx_set else None
        train_idx = np.nonzero(train_mask)[0]
        val_idx   = np.nonzero(val_mask)[0]
        test_idx  = np.nonzero(test_mask)[0] if test_mask is not None else None
        print(f"[SPLIT] from files  train={train_idx.size}  val={val_idx.size}" +
              (f"  test={test_idx.size}" if test_idx is not None else ""))
        return train_idx, val_idx, test_idx
    if has_ep_idx:
        ep_idx = np.load(EP_IDX_PATH).astype(np.int64)
        uniq_eps = np.unique(ep_idx)
        rng.shuffle(uniq_eps)
        n_val = max(1, int(math.ceil(len(uniq_eps) * VAL_RATIO_FALLBACK)))
        val_eps = set(uniq_eps[:n_val].tolist())
        train_eps = set(uniq_eps[n_val:].tolist())
        train_mask = np.isin(ep_idx, list(train_eps))
        val_mask   = np.isin(ep_idx, list(val_eps))
        train_idx = np.nonzero(train_mask)[0]
        val_idx   = np.nonzero(val_mask)[0]
        print(f"[SPLIT] random by episodes (fallback)  train={train_idx.size}  val={val_idx.size}")
        return train_idx, val_idx, None
    all_idx = np.arange(N, dtype=np.int64)
    rng.shuffle(all_idx)
    n_val = max(1, int(math.ceil(N * VAL_RATIO_FALLBACK)))
    val_idx = all_idx[:n_val]
    train_idx = all_idx[n_val:]
    print(f"[SPLIT] random by transitions (fallback)  train={train_idx.size}  val={val_idx.size}")
    return train_idx, val_idx, None

train_idx, val_idx, test_idx = load_split_indices()

# ========================= Dataset 実装（省メモリ） =========================
class NpzArrayDataset(Dataset):
    """メモリマップ配列を参照し、(s, r, s_next, done) を返す"""
    def __init__(self,
                 states_mm: np.ndarray,
                 returns_mm: np.ndarray,
                 next_idx: np.ndarray,
                 done_vec: np.ndarray,
                 indices: np.ndarray):
        self.states   = states_mm
        self.returns  = returns_mm
        self.next_idx = next_idx
        self.done     = done_vec
        self.indices  = indices.astype(np.int64, copy=False)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        j = int(self.next_idx[i])
        s      = torch.from_numpy(np.array(self.states[i], copy=False)).to(torch.float32)
        r      = torch.tensor(float(self.returns[i]), dtype=torch.float32)
        s_next = torch.from_numpy(np.array(self.states[j], copy=False)).to(torch.float32)
        d      = torch.tensor(float(self.done[i]), dtype=torch.float32)  # 1.0=terminal, 0.0=non-terminal
        return s, r, s_next, d

train_ds = NpzArrayDataset(states_mm, returns_mm, NEXT_IDX, DONE_VEC, train_idx)
val_ds   = NpzArrayDataset(states_mm, returns_mm, NEXT_IDX, DONE_VEC, val_idx)

pin_mem = (DEVICE.type == "cuda")
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin_mem,
    num_workers=0,
    persistent_workers=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_mem,
    num_workers=0,
    persistent_workers=False,
)

# ========================= 保存名ルール =========================
def _derive_save_name() -> str:
    if not OBS_MODE_PARTIAL:
        return "value_full.pt"
    base = "value_partial"
    if not APPLY_OBS_MASK:
        base += "_nomask"
    if ENABLE_DISTILL:
        base += "_distilled"
    return base + ".pt"

# ========================= 蒸留フェーズ（任意 / オプション） =========================
if ENABLE_DISTILL:
    teacher_base_d = int(os.environ.get("TEACHER_BASE_D", "305"))
    if os.path.exists(TEACHER_MODEL_PATH):
        # 教師ネットは teacher_base_d（例: 305）で構築
        teacher = ValueNet(state_dim=teacher_base_d).to(DEVICE)
        teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE, weights_only=True))
        teacher.eval()

        # 教師入力ソース:
        #   1) NPZに full_states があれば最優先
        #   2) なければ states を使用（列不足なら右ゼロパディングで teacher_base_d に拡張）
        if PAIRED_FULL_STATES_KEY in bundle.files:
            teacher_src_mm: np.memmap = bundle[PAIRED_FULL_STATES_KEY]
            src_d = int(teacher_src_mm.shape[1])
            if src_d > teacher_base_d:
                raise RuntimeError(f"[FATAL] full_states 列数が教師基準次元を超えています: full={src_d} base={teacher_base_d}")
            use_states_as_teacher = False
        else:
            teacher_src_mm = states_mm
            src_d = D
            use_states_as_teacher = True
            print("[DISTILL] full_states が見つからないため、states を教師入力として使用します。")

        class DistillDataset(Dataset):
            def __init__(self, s_mm, t_mm, indices, pad_to):
                self.s = s_mm
                self.t = t_mm
                self.idx = indices.astype(np.int64, copy=False)
                self.pad_to = int(pad_to)
                self.src_d = int(t_mm.shape[1])
            def __len__(self): return int(self.idx.shape[0])
            def __getitem__(self, i):
                k = int(self.idx[i])
                s  = torch.from_numpy(np.array(self.s[k], copy=False)).to(torch.float32)
                t  = torch.from_numpy(np.array(self.t[k], copy=False)).to(torch.float32)
                # 教師入力は必要なら右ゼロパディングで pad_to に揃える
                if self.src_d < self.pad_to:
                    pad_w = self.pad_to - self.src_d
                    t = F.pad(t, (0, pad_w), mode="constant", value=0.0)
                return s, t

        distill_ds = DistillDataset(states_mm, teacher_src_mm, train_idx, teacher_base_d)
        distill_loader = DataLoader(
            distill_ds, batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=pin_mem, num_workers=0, persistent_workers=False
        )
        net_tmp = ValueNet(state_dim=D).to(DEVICE)  # 蒸留は student 初期化（学生は D）
        opt_d = torch.optim.AdamW(net_tmp.parameters(), lr=DISTILL_LR, weight_decay=DISTILL_WD)
        scaler_amp = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

        for ep in range(1, max(1, DISTILL_EPOCHS) + 1):
            net_tmp.train()
            total = 0.0
            for s, fs in distill_loader:
                s  = s.to(DEVICE, non_blocking=True)
                fs = fs.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    s  = apply_scaler(s)
                    # 教師側スケーリング：
                    # - teacher_base_d == D なら同じスケーラでOK（今回 305→OK）
                    # - それ以外のときは “無変換” で可（必要なら教師用scalerを別途導入）
                    if teacher_base_d == D:
                        fs = apply_scaler(fs)
                    if obs_mask_tensor is not None and OBS_MODE_PARTIAL and APPLY_OBS_MASK:
                        s = s * obs_mask_tensor  # 生徒側のみ部分観測
                    with torch.no_grad():
                        t = teacher(fs)
                    p = net_tmp(s)
                    loss = F.smooth_l1_loss(p, t, reduction="mean")
                opt_d.zero_grad(set_to_none=True)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(net_tmp.parameters(), max_norm=DISTILL_CLIP)
                scaler_amp.step(opt_d)
                scaler_amp.update()
                total += float(loss.item()) * s.size(0)
            avg = total / len(distill_loader.dataset)
            if use_states_as_teacher:
                print(f"[distill] epoch {ep:02d}  train {avg:.4f}  (teacher_input=states -> pad to {teacher_base_d})")
            else:
                print(f"[distill] epoch {ep:02d}  train {avg:.4f}  (teacher_input=full_states)")
    else:
        print("[WARN] 蒸留スキップ: teacher モデルが見つかりません。")


# ========================= オートチューニング対応の学習関数 =========================
@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    batch_size: int
    lambda_bellman: float
    epochs: int
    patience: int
    warmup_epochs: int = 5
    tag: str = ""  # 保存名に含めるラベル

def train_once(cfg: TrainConfig) -> dict:
    """
    与えたハイパラで1回学習を実行し、ベストの val_loss / 保存パスなどを返す
    DataLoader は上で作ったものを使う（batch_size は再構築）
    """
    global LAMBDA_BELLMAN
    LAMBDA_BELLMAN_orig = LAMBDA_BELLMAN
    LAMBDA_BELLMAN = cfg.lambda_bellman

    # ローダを cfg.batch_size で再構築
    pin_mem = (DEVICE.type == "cuda")
    train_loader_local = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        pin_memory=pin_mem, num_workers=0, persistent_workers=False
    )
    val_loader_local = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=pin_mem, num_workers=0, persistent_workers=False
    )

    # モデル・最適化器
    net = ValueNet(state_dim=D).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # スケジューラ（ウォームアップ+Cosine）
    def lr_lambda(current_epoch):
        if current_epoch < cfg.warmup_epochs:
            return float(current_epoch + 1) / float(max(1, cfg.warmup_epochs))
        progress = (current_epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # 保存先（タグ付き）
    def _derive_save_name_with_tag() -> str:
        base = _derive_save_name().replace(".pt", "")
        if cfg.tag:
            base += f"_{cfg.tag}"
        return base + ".pt"

    best_path_trial = os.path.join(os.environ.get("VALNET_RUN_DIR", DATA_DIR), _derive_save_name_with_tag())
    os.makedirs(os.path.dirname(best_path_trial), exist_ok=True)
    best_val = float("inf")
    no_improve = 0

    def run_epoch(loader: DataLoader, train: bool) -> tuple[float, float, float]:
        if train: net.train()
        else: net.eval()
        total_loss = total_sup = total_bell = 0.0
        total_count = 0
        for s, y, s_next, done in loader:
            s      = s.to(DEVICE, non_blocking=True)
            y      = y.to(DEVICE, non_blocking=True)
            s_next = s_next.to(DEVICE, non_blocking=True)
            done   = done.to(DEVICE, non_blocking=True)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                s      = apply_scaler(s)
                s_next = apply_scaler(s_next)
                if obs_mask_tensor is not None and APPLY_OBS_MASK:
                    s      = s * obs_mask_tensor
                    s_next = s_next * obs_mask_tensor
                v_s = net(s)
                sup_loss = F.huber_loss(v_s, y, delta=HUBER_DELTA, reduction="mean")
                with torch.no_grad():
                    v_sp = net(s_next)
                nonterm = (done < 0.5)
                if nonterm.any():
                    delta = GAMMA * v_sp[nonterm] - v_s[nonterm]
                    bellman_loss = F.huber_loss(delta, torch.zeros_like(delta),
                                                delta=HUBER_DELTA, reduction="mean")
                else:
                    bellman_loss = v_s.new_tensor(0.0)
                loss = sup_loss + LAMBDA_BELLMAN * bellman_loss
            if train:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=CLIP_GRAD)
                scaler_amp.step(opt)
                scaler_amp.update()
            bs = s.size(0)
            total_loss += float(loss.item()) * bs
            total_sup  += float(sup_loss.item()) * bs
            total_bell += float(bellman_loss.item()) * bs
            total_count += bs
        denom = max(1, total_count)
        return total_loss / denom, total_sup / denom, total_bell / denom

    # 学習ループ
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_sup, train_bell = run_epoch(train_loader_local, train=True)
        val_loss,   val_sup,   val_bell   = run_epoch(val_loader_local,   train=False)
        sched.step()
        print(f"[{cfg.tag or 'trial'}] epoch {epoch:02d}  "
              f"train {train_loss:.4f} (sup {train_sup:.4f} / bell {train_bell:.4f})  "
              f"val {val_loss:.4f} (sup {val_sup:.4f} / bell {val_bell:.4f})  "
              f"lr {sched.get_last_lr()[0]:.2e}  LMB={LAMBDA_BELLMAN:g}  BS={cfg.batch_size}")

        if val_loss + 1e-7 < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(net.state_dict(), best_path_trial)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[{cfg.tag or 'trial'}][EARLY STOP] no improvement for {cfg.patience} epochs.")
                break

    # 評価（ベストを読み直し）
    state_dict = torch.load(best_path_trial, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    preds, targets = [], []
    with torch.no_grad():
        for s, y, s_next, done in val_loader_local:
            s = s.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                s = apply_scaler(s)
                if obs_mask_tensor is not None and APPLY_OBS_MASK:
                    s = s * obs_mask_tensor
                p = net(s).detach().cpu().numpy()
            preds.append(p.reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    corr = float(np.corrcoef(y_true, pred)[0, 1])

    # LAMBDA_BELLMAN を復帰
    LAMBDA_BELLMAN = LAMBDA_BELLMAN_orig

    return {
        "best_val": float(best_val),
        "r2": r2,
        "corr": corr,
        "model_path": best_path_trial,
        "cfg": cfg,
    }

# ========================= オートチューニング実行ブロック =========================
if AUTOTUNE:
    # 10試行プラン（前回ログと同一構成）
    trial_plan = [
        # 1) 基準線：純教師あり（Bellman補助なし）
        TrainConfig(lr=2e-4,   weight_decay=1e-4, batch_size=1024, lambda_bellman=0.00,
                    epochs=max(40, EPOCHS), patience=PATIENCE, warmup_epochs=3, tag="base"),
        # 2) 弱いBellman補助
        TrainConfig(lr=2e-4,   weight_decay=1e-4, batch_size=1024, lambda_bellman=0.02,
                    epochs=max(40, EPOCHS), patience=PATIENCE, warmup_epochs=3, tag="lmb002"),
        # 3) 正則化強化
        TrainConfig(lr=1.5e-4, weight_decay=3e-4, batch_size=1024, lambda_bellman=0.02,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=3, tag="lmb002_reg"),
        # 4) Bellman↑ + BS↓
        TrainConfig(lr=1.5e-4, weight_decay=3e-4, batch_size=768,  lambda_bellman=0.05,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=3, tag="lmb005_bs768"),
        # 5) WDちょい弱 + 短ウォームアップ
        TrainConfig(lr=1.5e-4, weight_decay=2e-4, batch_size=1024, lambda_bellman=0.02,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=2, tag="lmb002_wd2e4_wu2"),
        # 6) WD強め + 短ウォームアップ
        TrainConfig(lr=1.5e-4, weight_decay=4e-4, batch_size=1024, lambda_bellman=0.02,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=2, tag="lmb002_wd4e4_wu2"),
        # 7) LRを少し下げる
        TrainConfig(lr=1.3e-4, weight_decay=3e-4, batch_size=1024, lambda_bellman=0.02,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=2, tag="lmb002_lr1p3"),
        # 8) LRを少し上げる
        TrainConfig(lr=1.7e-4, weight_decay=3e-4, batch_size=1024, lambda_bellman=0.02,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=2, tag="lmb002_lr1p7"),
        # 9) Bellman弱め（0.01）
        TrainConfig(lr=1.5e-4, weight_decay=3e-4, batch_size=1024, lambda_bellman=0.01,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=2, tag="lmb001"),
        # 10) Bellmanやや強め（0.03）
        TrainConfig(lr=1.5e-4, weight_decay=3e-4, batch_size=1024, lambda_bellman=0.03,
                    epochs=max(60, EPOCHS), patience=PATIENCE, warmup_epochs=2, tag="lmb003"),
    ]

    best = None
    for i, cfg in enumerate(trial_plan, 1):
        print(f"\n[AUTOTUNE] Trial {i}/{len(trial_plan)}: {cfg}\n")
        res = train_once(cfg)
        print(f"[AUTOTUNE] Trial {i} result: val={res['best_val']:.6f}, R2={res['r2']:.4f}, corr={res['corr']:.4f}")
        if (best is None) or (res["best_val"] + AUTOTUNE_TOL < best["best_val"]):
            best = res
        else:
            # 悪化または改善なし → そのまま次手へ
            pass

    # 最良モデルを正式な保存名にコピー（従来名）
    final_path = os.path.join(os.environ.get("VALNET_RUN_DIR", DATA_DIR), _derive_save_name())
    if best and os.path.abspath(best["model_path"]) != os.path.abspath(final_path):
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.copy2(best["model_path"], final_path)
    print(f"\n[AUTOTUNE] [OK] best val={best['best_val']:.6f}, tag={best['cfg'].tag}, saved to {final_path}")
    print(f"[AUTOTUNE] R2={best['r2']:.4f}, corr={best['corr']:.4f}")

    # --- 追加: 採用版以外の試行チェックポイント(.pt)を削除 ---
    try:
        run_dir = os.path.dirname(final_path)
        base = _derive_save_name().replace(".pt", "")
        for fname in os.listdir(run_dir):
            if not fname.endswith(".pt"):
                continue
            fpath = os.path.join(run_dir, fname)
            if os.path.abspath(fpath) == os.path.abspath(final_path):
                continue
            # タグ付き（base_*.pt）のみを削除対象にする
            if fname.startswith(base + "_"):
                os.remove(fpath)
        print(f"[CLEANUP] removed other trial checkpoints in {run_dir}")
    except Exception as e:
        print(f"[CLEANUP][WARN] {e}")

else:
    # === 単発学習（従来どおり） ===
    if EVAL_ONLY and LOAD_MODEL:
        try:
            # 評価のみ：自己完結の val ループを定義
            def _eval_only_once(model_path: str) -> tuple[float, float, float]:
                net_eval = ValueNet(state_dim=D).to(DEVICE)
                sd = torch.load(model_path, map_location=DEVICE, weights_only=True)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                net_eval.load_state_dict(sd, strict=False)
                net_eval.eval()
                total_loss = total_sup = total_bell = 0.0
                total_count = 0
                with torch.no_grad():
                    for s, y, s_next, done in val_loader:
                        s      = s.to(DEVICE, non_blocking=True)
                        y      = y.to(DEVICE, non_blocking=True)
                        s_next = s_next.to(DEVICE, non_blocking=True)
                        done   = done.to(DEVICE, non_blocking=True)
                        s      = apply_scaler(s)
                        s_next = apply_scaler(s_next)
                        if obs_mask_tensor is not None and APPLY_OBS_MASK:
                            s      = s * obs_mask_tensor
                            s_next = s_next * obs_mask_tensor
                        v_s = net_eval(s)
                        sup_loss = F.huber_loss(v_s, y, delta=HUBER_DELTA, reduction="mean")
                        v_sp = net_eval(s_next)
                        nonterm = (done < 0.5)
                        if nonterm.any():
                            delta = GAMMA * v_sp[nonterm] - v_s[nonterm]
                            bellman_loss = F.huber_loss(delta, torch.zeros_like(delta),
                                                        delta=HUBER_DELTA, reduction="mean")
                        else:
                            bellman_loss = v_s.new_tensor(0.0)
                        loss = sup_loss + LAMBDA_BELLMAN * bellman_loss
                        bs = s.size(0)
                        total_loss += float(loss.item()) * bs
                        total_sup  += float(sup_loss.item()) * bs
                        total_bell += float(bellman_loss.item()) * bs
                        total_count += bs
                denom = max(1, total_count)
                return total_loss / denom, total_sup / denom, total_bell / denom

            val_loss, val_sup, val_bell = _eval_only_once(LOAD_MODEL)
            meta_path = os.path.join(os.environ.get("VALNET_RUN_DIR", DATA_DIR), "train_meta.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                _json_guard.dump({"best_val_loss": float(val_loss), "model_path": LOAD_MODEL}, mf, ensure_ascii=False, indent=2)
            print(f"[EVAL-ONLY] val_loss={val_loss:.6f}  saved meta to {meta_path}")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[EVAL-ONLY][WARN] evaluation failed: {e}")

    default_cfg = TrainConfig(
        lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
        lambda_bellman=LAMBDA_BELLMAN, epochs=EPOCHS,
        patience=PATIENCE, warmup_epochs=5, tag="manual"
    )
    res = train_once(default_cfg)
    final_path = os.path.join(os.environ.get("VALNET_RUN_DIR", "data"), _derive_save_name())
    if os.path.abspath(res["model_path"]) != os.path.abspath(final_path):
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.copy2(res["model_path"], final_path)
    print(f"\n[OK] saved best model to {final_path} (val_loss={res['best_val']:.6f})")

    # --- 追加: 採用版以外の試行チェックポイント(.pt)を削除 ---
    try:
        run_dir = os.path.dirname(final_path)
        base = _derive_save_name().replace(".pt", "")
        for fname in os.listdir(run_dir):
            if not fname.endswith(".pt"):
                continue
            fpath = os.path.join(run_dir, fname)
            if os.path.abspath(fpath) == os.path.abspath(final_path):
                continue
            if fname.startswith(base + "_"):
                os.remove(fpath)
        print(f"[CLEANUP] removed other trial checkpoints in {run_dir}")
    except Exception as e:
        print(f"[CLEANUP][WARN] {e}")

    # write meta for orchestrator
    meta_path = os.path.join(os.environ.get("VALNET_RUN_DIR", "data"), "train_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            _json_guard.dump({"best_val_loss": float(res["best_val"]), "model_path": final_path}, mf, ensure_ascii=False, indent=2)
        print(f"[META] wrote {meta_path}")
    except Exception as e:
        print(f"[META][WARN] failed to write meta: {e}")
    print(f"[VAL] 決定係数(R2): {res['r2']:.4f}")
    print(f"[VAL] 相関係数: {res['corr']:.4f}")
# --- ここまで（修正後） ---
