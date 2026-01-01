# ===== 修正後（clip/scale を追加、Δ をクリップしてから加算） =====
import torch
import json
import numpy as np
from typing import Callable, Any, Dict
import sys
import os
try:
    from .my_mcc_sampler import mcc_sampler
except Exception:
    from my_mcc_sampler import mcc_sampler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_d3rlpy_data import D3RLPyDataPreprocessor

# 追加インポート（他ファイルは変更しません）
from typing import List, Optional
from collections import Counter


# ─────────────────────────────────────────────────────────
# 1) 実行時の状態dict（学習時と同スキーマ）
# ─────────────────────────────────────────────────────────
def _drop_private_keys_me(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d or {})

    # 自分でも「未知」のはずの情報は落とす（MCC で me_private に補完させる）
    out.pop("prize_cards", None)
    out.pop("prize_enum", None)

    # deck が dict のケース（{"cards":[...]} 等）を想定して cards を落とす
    deck = out.get("deck", None)
    if isinstance(deck, dict):
        deck = dict(deck)
        deck.pop("cards", None)
        out["deck"] = deck
    else:
        out.pop("deck", None)

    return out

def _drop_private_keys_opp(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d or {})

    # 相手の非公開（中身）は落とす：手札・山札・プライズ
    out.pop("hand", None)
    out.pop("hand_enum", None)

    out.pop("prize_cards", None)
    out.pop("prize_enum", None)

    deck = out.get("deck", None)
    if isinstance(deck, dict):
        deck = dict(deck)
        deck.pop("cards", None)
        out["deck"] = deck
    else:
        out.pop("deck", None)

    return out

def build_runtime_state_dict(player, match) -> Dict[str, Any]:
    me  = player.serialize()
    opp = player.opponent.serialize() if player.opponent else {}

    me  = _drop_private_keys_me(me)
    opp = _drop_private_keys_opp(opp) if opp is not None else opp

    # ★ stadium を学習時キーに合わせる
    if "stadium" not in me and "active_stadium" in me:
        me["stadium"] = me["active_stadium"]
    if opp is not None and "stadium" not in opp and "active_stadium" in opp:
        opp["stadium"] = opp["active_stadium"]

    # ★ hand_count / discard_pile_count が無ければ補完
    me.setdefault("hand_count", len(me.get("hand", [])))
    me.setdefault("discard_pile_count", len(me.get("discard_pile", [])))
    if opp is not None:
        opp.setdefault("hand_count", opp.get("hand_count", 0))
        opp.setdefault("discard_pile_count", len(opp.get("discard_pile", [])) if "discard_pile" in opp else 0)

    return {
        "me":  me,
        "opp": opp,
        "turn": getattr(match, "turn", 0),
    }


# d3rlpy 前処理器（語彙は既存 card_id2idx.json を読み込む実装のはず）
_preproc = D3RLPyDataPreprocessor(entries=[])

def encode_state(player, match) -> np.ndarray:
    state_dict = build_runtime_state_dict(player, match)
    return _preproc.encode_state(state_dict)

def encode_state_from_dict(state_dict: Dict[str, Any]) -> np.ndarray:
    return _preproc.encode_state(state_dict)

# ─────────────────────────────────────────────────────────
# 1.5) MCC 用 ctx ビルダー（このファイル内だけで完結）
# ─────────────────────────────────────────────────────────
def _enum_from_card(obj) -> int:
    try:
        enum_ = getattr(obj, "card_enum", None)
        if isinstance(enum_, int):
            return enum_
        if hasattr(enum_, "value"):
            v = enum_.value
            if isinstance(v, (tuple, list)) and v:
                return int(v[0])
            if isinstance(v, int):
                return v
        return int(getattr(obj, "id", 0)) or 0
    except Exception:
        return 0

def _collect_zone_enums(zone, *, include_stack_bottom: bool) -> List[int]:
    out: List[int] = []
    if not zone:
        return out
    for c in zone:
        if isinstance(c, list):
            if not c:
                continue
            if include_stack_bottom:
                for obj in c:
                    cid = _enum_from_card(obj)
                    if cid:
                        out.append(cid)
            else:
                obj = c[-1]
                cid = _enum_from_card(obj)
                if cid:
                    out.append(cid)
        else:
            cid = _enum_from_card(c)
            if cid:
                out.append(cid)
    return out

def _count_zone_enums(zone, *, top_only: bool, include_stack_bottom: bool) -> Counter:
    if top_only and include_stack_bottom:
        include_stack_bottom = False
    return Counter(_collect_zone_enums(zone, include_stack_bottom=include_stack_bottom))

def _merge_counters(*counters: Counter) -> Counter:
    out = Counter()
    for c in counters:
        if isinstance(c, Counter):
            out.update(c)
    return out

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def build_mcc_ctx(player, match) -> Dict[str, Any]:
    """
    覗き見禁止で、mcc_sampler が必要とする ctx を組み立てる。
    - 自分: initial_deck_enums（あれば）＋公開情報のカウント
    - 相手: 公開情報（トップのみ）＋プライズ既知/未知＋手札枚数
    - prior: match.opp_archetypes または match.opp_card_pool_counts を自動参照
    """
    p = player
    ctx: Dict[str, Any] = {
        "match": match,
        "player": p,
        "allow_peek_me": False,
        "allow_peek_opp": False,
    }

    # 自分：既知60枚（なければ現状の総和でフォールバック）
    my_known = getattr(p, "initial_deck_enums", None)
    if isinstance(my_known, list) and len(my_known) >= 60:
        ctx["my_known_decklist_enums"] = [int(x) for x in my_known]
    else:
        deck_ids   = _collect_zone_enums(getattr(p.deck, "cards", []), include_stack_bottom=True)
        hand_ids   = _collect_zone_enums(getattr(p, "hand", []), include_stack_bottom=True)
        prize_ids  = _collect_zone_enums(getattr(p, "prize_cards", []), include_stack_bottom=True)
        active_ids = _collect_zone_enums(getattr(p, "active_card", []), include_stack_bottom=True)
        bench_ids  = []
        for stk in getattr(p, "bench", []) or []:
            bench_ids.extend(_collect_zone_enums(stk, include_stack_bottom=True))
        trash_ids  = _collect_zone_enums(getattr(p, "discard_pile", []), include_stack_bottom=True)
        ctx["my_known_decklist_enums"] = deck_ids + hand_ids + prize_ids + active_ids + bench_ids + trash_ids

    # 自分：公開カウント（進化スタックの下まで全採用）
    my_hand   = _count_zone_enums(getattr(p, "hand", []), top_only=False, include_stack_bottom=True)
    my_trash  = _count_zone_enums(getattr(p, "discard_pile", []), top_only=False, include_stack_bottom=True)
    my_active = _count_zone_enums(getattr(p, "active_card", []), top_only=False, include_stack_bottom=True)
    my_bench  = Counter()
    for stk in getattr(p, "bench", []) or []:
        my_bench += _count_zone_enums(stk, top_only=False, include_stack_bottom=True)
    my_revealed_prize  = _count_zone_enums(getattr(p, "revealed_prize_cards", []), top_only=False, include_stack_bottom=True)
    my_revealed_search = _count_zone_enums(getattr(p, "revealed_cards", []), top_only=False, include_stack_bottom=True)
    my_public = _merge_counters(my_hand, my_trash, my_active, my_bench, my_revealed_prize, my_revealed_search)
    ctx["my_public_enum_counts"] = {int(cid): int(n) for cid, n in my_public.items() if cid != 0 and n > 0}

    # 相手：公開観測（トップのみ）
    opp = getattr(p, "opponent", None)
    if opp is not None:
        opp_trash = _count_zone_enums(getattr(opp, "discard_pile", []), top_only=False, include_stack_bottom=False)
        opp_active_top = _count_zone_enums(getattr(opp, "active_card", []), top_only=True, include_stack_bottom=False)
        opp_bench_tops = Counter()
        for stk in getattr(opp, "bench", []) or []:
            opp_bench_tops += _count_zone_enums(stk, top_only=True, include_stack_bottom=False)
        opp_revealed_prize  = _count_zone_enums(getattr(opp, "revealed_prize_cards", []), top_only=False, include_stack_bottom=False)
        opp_revealed_search = _count_zone_enums(getattr(opp, "revealed_cards", []), top_only=False, include_stack_bottom=False)
        obs = _merge_counters(opp_trash, opp_active_top, opp_bench_tops, opp_revealed_prize, opp_revealed_search)
        ctx["opp_observed_enum_counts"] = {int(cid): int(n) for cid, n in obs.items() if cid != 0 and n > 0}

        # プライズ既知/未知
        known_prizes: List[int] = []
        try:
            for c in getattr(opp, "revealed_prize_cards", []) or []:
                if isinstance(c, list) and c:
                    cid = _enum_from_card(c[-1])
                else:
                    cid = _enum_from_card(c)
                if cid:
                    known_prizes.append(cid)
        except Exception:
            pass
        ctx["opp_prize_known_enums"] = known_prizes
        ctx["opp_prize_unknown"] = max(0, 6 - len(known_prizes))

        # 手札枚数のみ
        ctx["opp_hand_size"] = _safe_len(getattr(opp, "hand", []))

    # prior: アーキタイプ or カードプール（あれば使う）
    arch = getattr(match, "opp_archetypes", None)
    pool = getattr(match, "opp_card_pool_counts", None)
    if isinstance(arch, list) and arch:
        # 形式は [{ "enums":[…60], "weight":0.6 }, ...] を想定
        cleaned = []
        for a in arch:
            if not isinstance(a, dict):
                continue
            enums = a.get("enums") or []
            if isinstance(enums, list) and len(enums) >= 60:
                w = float(a.get("weight", 1.0))
                cleaned.append({"enums": [int(x) for x in enums], "weight": w})
        if cleaned:
            ctx["opp_archetypes"] = cleaned
    elif isinstance(pool, dict) and pool:
        ctx["opp_card_pool_counts"] = {int(cid): int(n) for cid, n in pool.items() if int(n) > 0}

    return ctx


# ─────────────────────────────────────────────────────────
# 2) PBRS 本体
# ─────────────────────────────────────────────────────────
class PotentialBasedRewardShaping:
    """
    Potential-Based Reward Shaping with ValueNet.
    - on_step_start(player, match)  手番開始時に Φ(s) を記録
    - on_step_end(player, match, base_reward) 遷移確定で R + scale*(γΦ(s') − Φ(s)) を返す
    """

    def __init__(
        self,
        value_net: torch.nn.Module,
        encode_state_func: Callable[[Any, Any], np.ndarray] = encode_state,
        gamma: float = 0.99,
        device: str = "cpu",
        scale: float = 1.5,          # ← 追加: 差分のスケール
        clip: float | None = None,   # ← 追加: クリップ上限（絶対値）
        encode_state_from_dict_func: Callable[[Dict[str, Any]], np.ndarray] = encode_state_from_dict,
        mcc_sampler: Callable[[Dict[str, Any], Any], Dict[str, Any]] | None = None,
    ):
        self.value_net = value_net.eval().to(device)
        self.encode_state = encode_state_func
        self.encode_state_from_dict = encode_state_from_dict_func
        self.gamma = gamma
        self.device = device
        self.scale = scale
        self.clip = clip
        self.prev_phi: Dict[Any, float] = {}
        self.mcc_sampler = mcc_sampler

        # ★ 追加: Δ 整形のためのターゲットとバッファ
        self.target_std = 0.35
        self.percentile = 99.5
        self.safe_tau_min = 0.3
        self.safe_tau_max = 0.8
        self.scale_min = 0.5
        self.scale_max = 5.0
        self._delta_buf: List[float] = []
        self._delta_buf_maxlen = 10000

        # ★ 追加: Value 校正（a,b）の読込（ENV か 既定パス）
        try:
            calib_path = r"C:\Users\CLIVE\poke-pocket-sim\data\value_calibration.json"
            self._calib_a = None
            self._calib_b = None
            if os.path.isfile(calib_path):
                with open(calib_path, "r", encoding="utf-8") as f:
                    calib = json.load(f)
                a = float(calib.get("a", 1.0))
                b = float(calib.get("b", 0.0))
                self._calib_a = a
                self._calib_b = b
                print(f"[PBRS] Loaded calibration: a={a:.6f}, b={b:.6f} from {calib_path}")
            else:
                print(f"[PBRS] Calibration file not found: {calib_path} (use raw V)")
        except Exception as e:
            print(f"[PBRS][WARN] Failed to load calibration: {e}")
            self._calib_a = None
            self._calib_b = None

    # 公開情報のみで Φ を出す
    def _phi_public(self, player, match) -> float:
        vec = self.encode_state(player, match)
        if vec is None:
            return 0.0
        x = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v = float(self.value_net(x).item())
        if self._calib_a is not None and self._calib_b is not None:
            v = self._calib_a * v + self._calib_b
        return v

    # MCC で K サンプルの期待値 Φ を出す
    def _phi_mcc(self, player, match) -> float:
        samples = int(getattr(match, "mcc_samples", 0) or 0)
        if samples <= 0:
            return self._phi_public(player, match)

        top_k = int(getattr(match, "mcc_top_k", 0) or 0)

        # 基本の公開状態
        base_state = build_runtime_state_dict(player, match)

        phi_vals = []
        for _ in range(samples):
            # 1) MCC 用 ctx（覗き見禁止）を組み立て
            ctx = build_mcc_ctx(player, match)
            ctx["top_k"] = top_k

            # 2) 非公開を補完
            if self.mcc_sampler is not None:
                completed = self.mcc_sampler(base_state, ctx)
            else:
                completed = base_state

            # 3) 完全情報（に近い）状態をベクトル化
            vec = self.encode_state_from_dict(completed)
            if vec is None:
                continue
            x = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 4) ValueNetでスカラー
            with torch.no_grad():
                v = float(self.value_net(x).item())
            if self._calib_a is not None and self._calib_b is not None:
                v = self._calib_a * v + self._calib_b
            phi_vals.append(v)

        if not phi_vals:
            return 0.0
        return float(sum(phi_vals) / len(phi_vals))

    def _phi(self, player, match) -> float:
        if getattr(match, "use_mcc", False):
            return self._phi_mcc(player, match)
        return self._phi_public(player, match)

    def on_step_start(self, player, match):
        self.prev_phi[player] = self._phi(player, match)

    # ★ 追加: 非終端Δをセンタリング→クリップ→std正規化
    def _normalize_delta(self, raw_delta: float) -> float:
        # バッファ更新
        self._delta_buf.append(float(raw_delta))
        if len(self._delta_buf) > self._delta_buf_maxlen:
            del self._delta_buf[0:len(self._delta_buf) - self._delta_buf_maxlen]

        arr = np.asarray(self._delta_buf, dtype=np.float32)
        if arr.size < 8:
            return float(raw_delta)

        med = float(np.median(arr))
        centered = float(raw_delta) - med

        tau = float(np.percentile(np.abs(arr - med), self.percentile))
        tau = float(np.clip(tau, self.safe_tau_min, self.safe_tau_max))
        centered = float(np.clip(centered, -tau, tau))

        std_clip = float(np.std(np.clip(arr - med, -tau, tau)))
        scale = self.target_std / max(std_clip, 1e-8)
        scale = float(np.clip(scale, self.scale_min, self.scale_max))

        return centered * scale

    def on_step_end(self, player, match, base_reward: float) -> float:
        # 1) この遷移が終端かどうか（battle_logger から伝達）
        is_terminal_transition = bool(
            getattr(match, "_transition_done", False) or getattr(match, "game_over", False)
        )

        if is_terminal_transition:
            # φ(s') = 0 とみなす ⇒ Δ = γ*0 - φ_prev = -φ_prev
            phi_prev = self.prev_phi.get(player, 0.0)
            delta = -phi_prev
            if self.clip is not None:
                delta = float(np.clip(delta, -self.clip, self.clip))
            shaped = base_reward + self.scale * delta
            # エピソード終了なので φ をリセット
            self.prev_phi[player] = 0.0
            return shaped

        # 2) それ以外は通常の PBRS
        phi_next = self._phi(player, match)
        phi_prev = self.prev_phi.get(player, 0.0)
        delta = self.gamma * phi_next - phi_prev
        # ★ 非終端のみ：medianセンタリング→99.5%クリップ→std=0.35正規化
        delta = self._normalize_delta(delta)
        shaped = base_reward + self.scale * delta
        self.prev_phi[player] = phi_next
        return shaped

    def terminal_bonus(self, player, match, terminal_phi: float = 0.0) -> float:
        """
        終局行にも PBRS の最後の差分を載せるためのボーナス。
        返り値は scale * clip(gamma*terminal_phi - phi_prev).
        通常は終局のΦ(s_T)=0を基準にする（terminal_phi=0.0）。
        """
        # 直近のφ(s)がキャッシュに無ければ、今の公開状態から算出
        phi_prev = self.prev_phi.get(player, self._phi(player, match))
        delta = self.gamma * float(terminal_phi) - float(phi_prev)
        if self.clip is not None:
            delta = float(np.clip(delta, -self.clip, self.clip))
        # エピソード終了なのでキャッシュをリセット
        self.prev_phi[player] = 0.0
        return self.scale * delta


    # 互換用（既存コードが calculate_shaped_reward を呼んでいる場合）
    calculate_shaped_reward = on_step_end

    def reset(self, *args, **kwargs):
        self.prev_phi.clear()

def create_reward_shaping(
    value_net,
    encode_state_func=encode_state,
    gamma=0.99,
    device="cpu",
    scale=1.0,
    clip=None,
    encode_state_from_dict_func=encode_state_from_dict,
    mcc_sampler=mcc_sampler,
):
    return PotentialBasedRewardShaping(
        value_net,
        encode_state_func,
        gamma,
        device,
        scale=scale,
        clip=clip,
        encode_state_from_dict_func=encode_state_from_dict_func,
        mcc_sampler=mcc_sampler,
    )

def load_value_net(value_net_class, pt_path, device="cpu"):
    net = value_net_class()
    ckpt = torch.load(pt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            # dict だが中身が state_dict そのものの可能性
            state = ckpt
    else:
        state = ckpt

    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net