# pokepocketsim/policy/state_encoder.py
from __future__ import annotations
import numpy as np, json, os

OBS_BENCH_MAX = 8
HP_MAX   = 500.0
DECK_MAX = 60.0

OBS_DIM_TARGET = 2448

class StateEncoder:
    def __init__(self, scaler_path: str, target_dim: int = OBS_DIM_TARGET):
        self.scaler = {}
        if os.path.exists(scaler_path):
            data = np.load(scaler_path, allow_pickle=False)
            self.scaler["mean"] = data.get("mean", None)
            self.scaler["std"]  = data.get("std", None)
            # 前処理が出す clip_* が無い環境でも互換で動く
            self.scaler["clip_min"] = data.get("clip_min", None)
            self.scaler["clip_max"] = data.get("clip_max", None)

        # 語彙は card_id2idx.json と同ディレクトリから読む
        vocab_path = os.path.join(os.path.dirname(scaler_path), "card_id2idx.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        self.V = int(len(m) + 1)  # 0=PAD

        self.base_dim = 6*self.V + 195
        self.target_dim = int(target_dim)

        self.expected_dim = self.target_dim

    def _pad_or_trunc(self, x: np.ndarray, dim: int) -> np.ndarray:
        if x.shape[0] == dim:
            return x
        if x.shape[0] > dim:
            return x[:dim]
        y = np.zeros(dim, dtype=np.float32)
        y[:x.shape[0]] = x
        return y

    def _safe_zscore_and_clip(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_min, clip_max) -> np.ndarray:
        std = np.maximum(std, 1e-6)
        x = (x - mean.astype(np.float32)) / std.astype(np.float32)
        if clip_min is not None and clip_max is not None:
            x = np.clip(x, clip_min, clip_max)
        return x

    def encode_state(self, s: dict) -> np.ndarray:
        """
        実装詳細は既存のままで OK。ここでは次元の形だけ保証し、スケーリングを適用。
        """
        # --- 既存実装で 6V+195 の順に並ぶようにエンコード ---
        x = np.zeros(self.base_dim, dtype=np.float32)

        # …（既存の multi-hot / 数値特徴 / private(me/opp) の詰め処理）…

        mean = self.scaler.get("mean")
        std  = self.scaler.get("std")
        clip_min = self.scaler.get("clip_min")
        clip_max = self.scaler.get("clip_max")

        # --- z-score 正規化（mean/std があれば）＋任意のclip ---
        # scaler 次元に応じて「baseで正規化→pad」か「pad→targetで正規化」を安全に選ぶ
        if mean is not None and std is not None:
            try:
                mean = mean.astype(np.float32)
                std  = std.astype(np.float32)

                if mean.shape[0] == x.shape[0] and std.shape[0] == x.shape[0]:
                    x = self._safe_zscore_and_clip(x, mean, std, clip_min, clip_max)
                    x = self._pad_or_trunc(x, self.target_dim)
                    return x

                if mean.shape[0] == self.target_dim and std.shape[0] == self.target_dim:
                    x = self._pad_or_trunc(x, self.target_dim)
                    x = self._safe_zscore_and_clip(x, mean, std, clip_min, clip_max)
                    return x

                print(f"[ENCODER] ⚠️ scaler dim mismatch: mean={int(mean.shape[0])} std={int(std.shape[0])} base={int(self.base_dim)} target={int(self.target_dim)} -> skip zscore/clip")
            except Exception as e:
                print(f"[ENCODER] ⚠️ zscore/clip failed: {e} -> skip zscore/clip")

        x = self._pad_or_trunc(x, self.target_dim)
        return x
