#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bootstrap_initial_models.py

JSONL（公開/完全情報）から
  1) データ監査 & split(train/val)
  2) scaler.npz と 指紋(schema_sha / vocab_sha / scaler_sha)
  3) Value: 完全情報(教師)→公開(生徒) 蒸留
  4) Policy: 候補softmaxによる BC
  5) 成果物配置（run_p1/ と runs_v3/）

に必要な最小限をまとめて実行します。

前提ファイル:
  - D:\\date\\ai_vs_ai_match_all_ids.jsonl            (公開)
  - D:\\date\\ai_vs_ai_match_all_private_ids.jsonl    (完全情報)
  - action_types.json / card_id2idx.json が作業ディレクトリか同階層に存在（任意だが強く推奨）

出力:
  - d3rlpy_logs/run_p1/
      - policy_bc.pt
      - scaler.npz
      - train_meta.json
      - (存在すればコピー) action_types.json, card_id2idx.json
      - train_ids.txt / val_ids.txt
  - runs_v3/
      - value_full.pt
      - value_partial_distilled.pt

実行例:
  python bootstrap_initial_models.py
"""

from __future__ import annotations

import os
import sys
import json
import math
import random
import shutil
import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud


# ====== パス設定 ======
PUBLIC_JSONL  = r"D:\date\ai_vs_ai_match_all_ids.jsonl"
PRIVATE_JSONL = r"D:\date\ai_vs_ai_match_all_private_ids.jsonl"
RUN_DIR   = os.path.join("d3rlpy_logs", "run_p1")
VALUE_DIR = os.path.join("runs_v3")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(VALUE_DIR, exist_ok=True)

ACTION_TYPES_SRC = "action_types.json"   # 指紋材料
VOCAB_SRC        = "card_id2idx.json"    # 指紋材料
MAX_ARGS = int(os.getenv("POKE_MAX_ARGS", "3"))

# 乱数
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# スレッド控えめ（重たい環境向け）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ====== 監査ユーティリティ ======
def has_minus_one(x: Any) -> bool:
    """リスト/ネスト配列に -1 が含まれているか簡易チェック（許容外混入検知）"""
    try:
        if isinstance(x, (list, tuple)):
            for a in x:
                if has_minus_one(a):
                    return True
        elif isinstance(x, np.ndarray):
            return np.any(x == -1)
        elif isinstance(x, (int, float)):
            return x == -1
    except Exception:
        return False
    return False

# === [ADD] 観測キー自動検出 ===
def _looks_1d_numeric_list(v, min_len: int = 8) -> bool:
    if not isinstance(v, list):
        return False
    if len(v) < min_len:
        return False
    head = v[:min(16, len(v))]
    return all(isinstance(x, (int, float)) for x in head)

def detect_obs_key(g2: dict, preferred: tuple = ("obs_vec","observation","obs","state_vec","state","x","features")) -> Optional[str]:
    counts = {}
    seen = 0
    for entries in g2.values():
        for e in entries:
            for k in preferred:
                v = e.get(k)
                if _looks_1d_numeric_list(v):
                    counts[k] = counts.get(k, 0) + 1
            for k, v in e.items():
                if k in {"pi","legal_actions","action_candidates_vec","legal_actions_19d","action","action_result","game_id"}:
                    continue
                if _looks_1d_numeric_list(v):
                    counts[k] = counts.get(k, 0) + 1
            seen += 1
            if seen >= 2000:
                break
        if seen >= 2000:
            break
    if not counts:
        return None
    return max(counts, key=counts.get)


def trim_entries_after_canonical_terminal(entries: List[Dict]) -> List[Dict]:
    """終局後のダミー行が混じるケースの簡易トリム（'done'/ 'terminal' / 'is_terminal' を見る）"""
    out = []
    seen_terminal = False
    for e in entries:
        term = bool(e.get("done") or e.get("terminal") or e.get("is_terminal"))
        if not seen_terminal:
            out.append(e)
            if term:
                seen_terminal = True
        else:
            pass
    return out


# ====== 指紋（train_bc.py と互換の最小材料のみ） ======
def _read_action_types_normalized(path: str) -> Optional[List[Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            try:
                items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
            except Exception:
                items = sorted(obj.items(), key=lambda x: str(x[0]))
                items = [(k, v) for k, v in items]
            return [v for _, v in items]
    except Exception:
        pass
    return None


def schema_sha_from_action_types(action_types_path: str, max_args: int) -> tuple[Optional[str], Optional[dict]]:
    types = _read_action_types_normalized(action_types_path)
    if types is None:
        return None, None
    spec = {"TYPE_SCHEMAS": types, "MAX_ARGS": int(max_args)}
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), spec


def vocab_fp_from_json(path: str) -> tuple[Optional[str], Optional[int]]:
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None, None
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        sha = hashlib.sha256(blob).hexdigest()
        return sha, len(obj)
    except Exception:
        return None, None


def scaler_fp(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        keys = sorted(list(data.files))
        h = hashlib.sha256()
        for k in keys:
            h.update(k.encode())
            h.update(data[k].tobytes())
        return h.hexdigest()
    except Exception:
        return None


# ====== JSONL ロード ======
def read_jsonl_grouped_by_game(path: str) -> Dict[str, List[Dict]]:
    g2 = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            e = json.loads(ln)
            gid = e.get("game_id") or e.get("state_before", {}).get("game_id")
            if not gid:
                continue
            g2.setdefault(str(gid), []).append(e)
    def key_turn(e):
        for k in ("turn", "ply", "step", "t", "idx"):
            if k in e:
                return e[k]
        return 0
    for gid in list(g2.keys()):
        g2[gid] = trim_entries_after_canonical_terminal(sorted(g2[gid], key=key_turn))
    return g2


# ====== スプリット & スケーラ ======
def split_train_val(g2: Dict[str, List[Dict]], ratio: float = 0.9) -> Tuple[List[str], List[str]]:
    gids = list(g2.keys())
    random.shuffle(gids)
    cut = int(len(gids) * ratio)
    return gids[:cut], gids[cut:]


# === [REPLACE] もとの make_scaler_from_train を丸ごと置き換え ===
def make_scaler_from_train(g2: Dict[str, List[Dict]], train_ids: List[str], save_to: str, obs_key: str) -> Tuple[np.ndarray, np.ndarray]:
    obs = []
    for gid in train_ids:
        for e in g2[gid]:
            v = e.get(obs_key)
            if _looks_1d_numeric_list(v, min_len=1):
                obs.append(v)
    X = np.asarray(obs, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        raise RuntimeError(f"{obs_key} が見つからず scaler を作れません。")
    mean = X.mean(0).astype(np.float32)
    std  = X.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    np.savez(save_to, mean=mean, std=std)
    return mean, std


# ====== Value: MLP ======
class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


def load_pairs_from_jsonl(path: str, key_vec: str, key_z: str = "z") -> Tuple[torch.Tensor, torch.Tensor]:
    X: List[List[float]] = []
    Y: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            e = json.loads(ln)
            v = e.get(key_vec)
            z = e.get(key_z, 0)
            if isinstance(v, list) and len(v) > 0:
                X.append(v)
                Y.append(float(z))
    X = torch.tensor(np.asarray(X, dtype=np.float32))
    Y = torch.tensor(np.asarray(Y, dtype=np.float32)).unsqueeze(1)
    return X, Y

def try_build_alignment_key(e: Dict) -> Optional[str]:
    gid = e.get("game_id") or e.get("state_before", {}).get("game_id")
    if gid is None:
        return None
    for k in ("turn", "ply", "turn_index", "ply_index", "step", "t", "idx"):
        if k in e:
            return f"{gid}#{int(e[k])}"
    return None

# === [REPLACE] もとの build_teacher_targets_for_public を丸ごと置き換え ===
def build_teacher_targets_for_public(public_path: str, private_path: str, teacher: nn.Module, private_full_key: str) -> Optional[torch.Tensor]:
    # private の辞書（game_id#turn -> obs_full_vec 相当）
    priv_map: Dict[str, List[float]] = {}
    with open(private_path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            e = json.loads(ln)
            key = try_build_alignment_key(e)
            if key is None:
                continue
            v = e.get(private_full_key)
            if isinstance(v, list) and len(v) > 0:
                priv_map[key] = v

    batch: List[List[float]] = []
    to_predict: List[int] = []
    outputs: Dict[int, float] = {}

    def flush_batch():
        if not batch:
            return
        with torch.no_grad():
            X = torch.tensor(np.asarray(batch, dtype=np.float32), device=DEVICE)
            y = teacher(X).squeeze(1).cpu().numpy().tolist()
        for idx, val in zip(to_predict, y):
            outputs[idx] = float(val)
        batch.clear(); to_predict.clear()

    with open(public_path, "r", encoding="utf-8") as f:
        for idx, ln in enumerate(f):
            if not ln.strip():
                continue
            e = json.loads(ln)
            key = try_build_alignment_key(e)
            if key is None:
                continue
            v = priv_map.get(key)
            if v is None:
                continue
            batch.append(v)
            to_predict.append(idx)
            if len(batch) >= 4096:
                flush_batch()
        flush_batch()

    if not outputs:
        return None

    Yteach: List[float] = []
    with open(public_path, "r", encoding="utf-8") as f:
        for idx, ln in enumerate(f):
            if idx in outputs:
                Yteach.append(outputs[idx])
    if not Yteach:
        return None
    return torch.tensor(np.asarray(Yteach, dtype=np.float32)).unsqueeze(1)


# ====== Policy: 候補softmax BC ======
class BiTower(nn.Module):
    """score(s,a)=phi(s)・psi(a) を候補上で計算"""
    def __init__(self, d_obs: int, d_act: int = 19, h: int = 512):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_obs, 1024),
            nn.ReLU(),
            nn.Linear(1024, h),
        )
        self.psi = nn.Sequential(
            nn.Linear(d_act, 256),
            nn.ReLU(),
            nn.Linear(256, h),
        )

    def forward(self, obs: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        # obs:[B,D], cands:[B,N,19]
        s = self.phi(obs)[:, None, :]     # [B,1,H]
        a = self.psi(cands)               # [B,N,H]
        scores = (s * a).sum(-1)          # [B,N]
        return scores


def extract_bc_triplets(g2: Dict[str, List[Dict]], gids: List[str], obs_key: str = "obs_vec") -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    (obs_vec, candidates[N,19], target_idx) のリストを作る
    教師は 1) pi があれば argmax、2) なければ action_result を候補に照合
    """
    items: List[Tuple[np.ndarray, np.ndarray, int]] = []
    skipped = 0
    for gid in gids:
        for e in g2[gid]:
            obs = e.get(obs_key)
            cands = e.get("action_candidates_vec") or e.get("legal_actions_19d")
            if not (isinstance(obs, list) and isinstance(cands, list) and len(cands) > 0):
                continue

            tgt_idx: Optional[int] = None
            pi = e.get("pi")
            if isinstance(pi, list) and len(pi) == len(cands) and abs(sum(float(x) for x in pi) - 1.0) < 1e-3:
                tgt_idx = int(np.argmax(np.asarray(pi, dtype=np.float32)))
            else:
                act = None
                ar = e.get("action_result") or e.get("action") or {}
                if isinstance(ar, dict):
                    act = ar.get("action") or ar.get("action_19d") or ar.get("chosen") or ar.get("chosen_19d")
                if isinstance(act, list) and len(act) == 19:
                    for i, cand in enumerate(cands):
                        try:
                            if list(cand) == list(act):
                                tgt_idx = i
                                break
                        except Exception:
                            pass

            if tgt_idx is None:
                skipped += 1
                continue

            items.append(
                (np.asarray(obs, dtype=np.float32),
                 np.asarray(cands, dtype=np.float32),
                 int(tgt_idx))
            )

    if len(items) == 0:
        raise RuntimeError("BC 用の (obs, candidates, target) が 0 件でした。データのフィールド名をご確認ください。")
    print(f"[BC] usable={len(items)} | skipped={skipped}")
    return items


# ====== 進捗表示の軽いユーティリティ ======
def mb(nbytes: float) -> float:
    return nbytes / (1024 * 1024)


# ====== メイン ======
def main() -> None:
    print(f"[RUN] device={DEVICE}")

    # 1) データ読み込み＆監査
    print("[DATA] loading & auditing ...")
    g2_pub = read_jsonl_grouped_by_game(PUBLIC_JSONL)
    g2_prv = read_jsonl_grouped_by_game(PRIVATE_JSONL)

    # -1 混入ざっくりチェック（違反が多ければここで中断する運用を推奨）
    probe_fields = ("obs_vec", "action_candidates_vec", "obs_full_vec")
    bad_count = 0
    for gsrc, tag in ((g2_pub, "public"), (g2_prv, "private")):
        for gid, entries in gsrc.items():
            for e in entries:
                for k in probe_fields:
                    if k in e and has_minus_one(e[k]):
                        bad_count += 1
                        break
    if bad_count > 0:
        print(f"[AUDIT][WARN] -1 の混入が {bad_count} 件見つかりました（終局以降や欠損の可能性）。必要に応じて前処理で除去してください。")

    # 2) split（公開データで）
    train_ids, val_ids = split_train_val(g2_pub, 0.9)
    with open(os.path.join(RUN_DIR, "train_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(RUN_DIR, "val_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids))
    print(f"[SPLIT] games: train={len(train_ids)} | val={len(val_ids)}")

    # === [ADD] 観測キーの自動検出（環境変数で上書き可） ===
    PUB_OBS_KEY   = os.getenv("OBS_KEY_PUBLIC") or detect_obs_key(g2_pub, ("obs_vec","observation","obs","state_vec","state","x","features"))
    PRIV_FULL_KEY = os.getenv("OBS_FULL_KEY_PRIVATE") or detect_obs_key(g2_prv, ("obs_full_vec","full_obs_vec","private_vec","state_full_vec","obs_private_vec"))
    if PUB_OBS_KEY is None:
        raise RuntimeError("公開JSONLから観測ベクトルのキーを自動検出できませんでした。環境変数 OBS_KEY_PUBLIC で指定してください。")
    print(f"[DETECT] public obs key  = {PUB_OBS_KEY}")
    if PRIV_FULL_KEY is None:
        print("[DETECT][WARN] private(完全情報)の観測キーを検出できませんでした。蒸留はOFFになります。")
    else:
        print(f"[DETECT] private full key = {PRIV_FULL_KEY}")

    # 3) scaler（公開 train の観測ベクトル）
    scaler_path = os.path.join(RUN_DIR, "scaler.npz")
    mean, std = make_scaler_from_train(g2_pub, train_ids, scaler_path, PUB_OBS_KEY)
    print(f"[SCALE] scaler.npz saved -> {scaler_path} | obs_dim={mean.shape[0]}")

    # 付随ファイルコピー（あれば）
    for src in (ACTION_TYPES_SRC, VOCAB_SRC):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(RUN_DIR, os.path.basename(src)))
            print(f"[COPY] {src} -> {RUN_DIR}")

    # 指紋
    at_in_run = os.path.join(RUN_DIR, os.path.basename(ACTION_TYPES_SRC))
    at_src = at_in_run if os.path.exists(at_in_run) else ACTION_TYPES_SRC
    schema_sha, schema_spec = schema_sha_from_action_types(at_src, MAX_ARGS)

    vocab_in_run = os.path.join(RUN_DIR, os.path.basename(VOCAB_SRC))
    vocab_sha, vocab_size = vocab_fp_from_json(vocab_in_run if os.path.exists(vocab_in_run) else VOCAB_SRC)
    scaler_sha = scaler_fp(scaler_path)

    # 4) Value: 教師（完全情報）
    if PRIV_FULL_KEY is not None:
        print("[VALUE] training teacher (full obs -> z) ...")
        Xt, Yt = load_pairs_from_jsonl(PRIVATE_JSONL, PRIV_FULL_KEY, "z")
        Ds = Xt.shape[1]
        teacher = MLP(Ds).to(DEVICE)
        opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
        mse = nn.MSELoss()
        teacher.train()
        for ep in range(6):
            pred = teacher(Xt.to(DEVICE))
            loss = mse(pred, Yt.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"[VALUE][teacher] ep={ep} mse={float(loss):.5f}")
        torch.save({"state_dict": teacher.state_dict(), "dim": Ds}, os.path.join(VALUE_DIR, "value_full.pt"))
        print(f"[SAVE] value_full.pt -> {VALUE_DIR}")
    else:
        teacher = None
        print("[VALUE][teacher] SKIP (private full key not found)")

    # 4') Value: 生徒（公開 -> z, + 蒸留）
    print("[VALUE] training student (public obs -> z, with distill) ...")
    Xp, Yp = load_pairs_from_jsonl(PUBLIC_JSONL, PUB_OBS_KEY, "z")
    student = MLP(Xp.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    if teacher is not None and PRIV_FULL_KEY is not None:
        Yteach = build_teacher_targets_for_public(PUBLIC_JSONL, PRIVATE_JSONL, teacher, private_full_key=PRIV_FULL_KEY)
        use_distill = Yteach is not None and len(Yteach) == len(Xp)
    else:
        Yteach = None
        use_distill = False
    print(f"[VALUE] distill={'ON' if use_distill else 'OFF'} (aligned={0 if Yteach is None else len(Yteach)})")

    student.train()
    EPOCHS = 6
    BS = 4096
    mse = nn.MSELoss()
    idxs = np.arange(len(Xp))
    for ep in range(EPOCHS):
        np.random.shuffle(idxs)
        running = 0.0
        for i in range(0, len(idxs), BS):
            b = idxs[i:i+BS]
            xb = Xp[b].to(DEVICE)
            zb = Yp[b].to(DEVICE)
            pb = student(xb)
            if use_distill:
                tb = Yteach[b].to(DEVICE)
                loss = 0.7 * mse(pb, tb) + 0.3 * mse(pb, zb)
            else:
                loss = mse(pb, zb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
            running += float(loss) * len(b)
        print(f"[VALUE][student] ep={ep} mse={running/len(Xp):.6f}")
    torch.save({"state_dict": student.state_dict(), "dim": Xp.shape[1]}, os.path.join(VALUE_DIR, "value_partial_distilled.pt"))
    print(f"[SAVE] value_partial_distilled.pt -> {VALUE_DIR}")

    # 5) Policy: 候補softmax BC（公開 train のみで学習）
    print("[BC] building triplets (train only) ...")
    try:
        items = extract_bc_triplets(g2_pub, train_ids, obs_key=PUB_OBS_KEY)
    except RuntimeError as e:
        print(f"[BC][WARN] skip BC training: {e}")
        items = []

    if items:
        D_obs = items[0][0].shape[0]
        A_dim = items[0][1].shape[1]  # 19 を想定
        model = BiTower(D_obs, A_dim).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        CE = nn.CrossEntropyLoss(label_smoothing=0.05)

        class TripletDS(tud.Dataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                o, c, t = self.items[idx]
                return o, c, t

        def collate(batch):
            obs = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
            padN = max(len(b[1]) for b in batch)
            A = batch[0][1].shape[1]
            C = np.full((len(batch), padN, A), 0.0, np.float32)
            mask = np.full((len(batch), padN), -1e9, np.float32)
            tgt = []
            for bi, (_, c, t) in enumerate(batch):
                C[bi, :len(c), :] = c
                mask[bi, :len(c)] = 0.0
                tgt.append(t)
            return obs, torch.tensor(C, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), torch.tensor(tgt, dtype=torch.long)

        ds = TripletDS(items)
        BATCH = 512
        loader = tud.DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False, collate_fn=collate)

        EPOCHS_BC = 8
        model.train()
        for ep in range(EPOCHS_BC):
            total = 0.0
            count = 0
            for obs, C, mask, tgt in loader:
                obs = obs.to(DEVICE); C = C.to(DEVICE); mask = mask.to(DEVICE); tgt = tgt.to(DEVICE)
                scores = model(obs, C) + mask
                loss = CE(scores, tgt)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                total += float(loss) * len(obs)
                count += len(obs)
            print(f"[BC] ep={ep} loss={total/max(1,count):.6f}")

        torch.save(model.state_dict(), os.path.join(RUN_DIR, "policy_bc.pt"))
        print(f"[SAVE] policy_bc.pt -> {RUN_DIR}")
    else:
        print("[BC] no usable triplets; policy_bc.pt will NOT be created.")

    # 6) train_meta.json（指紋固定）
    meta = {
        "algo": "BC(bi-tower)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE,
        "seed": SEED,
        "fingerprints": {
            "schema_sha": schema_sha,
            "vocab_sha": vocab_sha,
            "vocab_size": vocab_size,
            "scaler_sha": scaler_sha,
        },
        "schema_sha": schema_sha,
        "vocab_sha": vocab_sha,
        "vocab_size": vocab_size,
        "scaler_sha": scaler_sha,
        "max_args": MAX_ARGS,
    }
    with open(os.path.join(RUN_DIR, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, sort_keys=True, indent=2)
    if schema_sha:
        print(f"[FPRINT] schema_sha(saved) = {schema_sha}")

    try:
        sz_pol = os.path.getsize(os.path.join(RUN_DIR, "policy_bc.pt"))
        sz_scl = os.path.getsize(os.path.join(RUN_DIR, "scaler.npz"))
        print(f"[DONE] run_dir={RUN_DIR}")
        print(f"       size(policy_bc.pt): {mb(sz_pol):.2f} MB")
        print(f"       size(scaler.npz)  : {mb(sz_scl):.2f} MB")
        print(f"[DONE] value_dir={VALUE_DIR}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
