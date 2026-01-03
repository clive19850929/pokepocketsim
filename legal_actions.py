# -*- coding: utf-8 -*-
"""
legal_actions.py

ai vs ai.py から分離した「legal_actions（合法手）の抽出・整形・候補ベクトル化」関連ユーティリティ。

前提:
- ai vs ai.py 側で build_encoder_from_files(...) により生成される _encode_action_raw を、
  set_action_encoder(...) で本モジュールへ注入してから利用してください。
- worker.py が Windows spawn で __main__（ai vs ai.py）参照する設計の場合は、
  ai vs ai.py 側で本モジュールの関数を import して “同名で再公開” してください。

このファイルは「移行した関数本体は極力そのまま」維持し、依存するエンコーダだけ注入で解決します。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


def set_action_encoder(encode_action_raw: Callable[[Any], Any], action_vec_dim: int = 32) -> None:
    """
    ai vs ai.py で生成した共有エンコーダ（_encode_action_raw）を注入する。
    - encode_action_from_vec_32d は _encode_action_raw を参照するため、注入前に呼ばないこと。
    """
    global _encode_action_raw, ACTION_VEC_DIM
    _encode_action_raw = encode_action_raw
    try:
        ACTION_VEC_DIM = int(action_vec_dim)
    except Exception:
        ACTION_VEC_DIM = action_vec_dim  # type: ignore[assignment]


# 既定（注入で上書きされる）
ACTION_VEC_DIM = 32


def _safe_encode_obs_for_candidates(sb_src, encoder, la_ids=None):
    """obs_vec を『必ず』生成し、0次元だった場合はその場で警告を出す。"""
    try:
        me  = sb_src.get("me",  {}) if isinstance(sb_src, dict) else {}
        opp = sb_src.get("opp", {}) if isinstance(sb_src, dict) else {}
        feat = {"me": me, "opp": opp}
        if isinstance(la_ids, list) and la_ids:
            feat["legal_actions"] = la_ids

        out = encoder.encode_state(feat)
        try:
            import numpy as _np
            arr = _np.asarray(out, dtype=_np.float32).reshape(-1)
            if arr.size == 0:
                print("[OBS] ⚠️ encoder から 0次元ベクトルが返りました（scaler不在/不一致の可能性）")
            return arr.tolist()
        except Exception:
            if isinstance(out, list) and len(out) == 0:
                print("[OBS] ⚠️ obs_vec が [] です（encoder を確認してください）")
            return out if isinstance(out, list) else []
    except Exception as e:
        print(f"[OBS] encode_state failed (with legal_actions): {e}")
        # legal_actions 無しで再トライ
        try:
            out = encoder.encode_state({"me": me, "opp": opp})
            try:
                import numpy as _np
                arr = _np.asarray(out, dtype=_np.float32).reshape(-1)
                if arr.size == 0:
                    print("[OBS] ⚠️ encoder から 0次元ベクトルが返りました（fallback, no legal_actions）")
                return arr.tolist()
            except Exception:
                return out if isinstance(out, list) else []
        except Exception:
            return []


def _embed_legal_actions_32d(la_ids):  # pyright: ignore[reportUnusedFunction]
    outs = []
    try:
        import numpy as np
    except Exception:
        return []

    TARGET_DIM = 32

    def _zeros():
        return [0.0] * TARGET_DIM

    def _to_id_vec(a):
        if isinstance(a, list) and len(a) > 0:
            return a
        if isinstance(a, tuple) and len(a) > 0:
            return list(a)
        if isinstance(a, int):
            return [a]
        if isinstance(a, dict):
            v = a.get("id")
            if isinstance(v, int):
                return [v]
            return None
        return None

    src = (la_ids or [])
    for a in src:
        a_vec = _to_id_vec(a)
        if not isinstance(a_vec, list) or not a_vec:
            outs.append(_zeros())
            continue

        try:
            v = encode_action_from_vec_32d(a_vec)
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception:
            outs.append(_zeros())
            continue

        if arr.size < TARGET_DIM:
            pad = np.zeros(TARGET_DIM - arr.size, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.size > TARGET_DIM:
            arr = arr[:TARGET_DIM]

        # “全 -1” は無効候補として 0 に落とす（全滅はさせない）
        try:
            if arr.size == TARGET_DIM and bool(np.all(np.isfinite(arr))) and bool(np.all(np.abs(arr + 1.0) <= 1e-9)):
                outs.append(_zeros())
                continue
        except Exception:
            pass

        # NaN/Inf も 0 に落とす
        try:
            if not bool(np.all(np.isfinite(arr))):
                outs.append(_zeros())
                continue
        except Exception:
            outs.append(_zeros())
            continue

        outs.append(arr.astype(np.float32).tolist())

    return outs


def encode_action_from_vec_32d(five_ints):
    try:
        v = _encode_action_raw(five_ints)

        # _encode_action_raw の生出力（例: 17d）を Policy 期待次元（例: 32d）へ揃える
        if isinstance(v, np.ndarray):
            vv = v.reshape(-1).tolist()
        elif isinstance(v, (list, tuple)):
            vv = list(v)
        else:
            vv = [v]

        try:
            target_dim = int(globals().get("ACTION_VEC_DIM", 0) or 0)
        except Exception:
            target_dim = 0

        if target_dim > 0:
            if len(vv) < target_dim:
                vv = vv + [0] * (target_dim - len(vv))
            elif len(vv) > target_dim:
                vv = vv[:target_dim]

        return vv
    except NameError:
        raise RuntimeError(
            "Action encoder is not initialized yet. "
            "Call build_encoder_from_files(...) before using encode_action_from_vec_32d."
        )


def _attach_action_encoder_if_supported(pol):
    """
    モデル方策や GPU クライアント側にアクション埋め込み関数を渡せるフックが
    用意されている場合に差し込む（後方互換のためのベストエフォート）。
    """
    try:
        enc32 = globals().get("encode_action_from_vec_32d", None)
        enc19 = globals().get("encode_action_from_vec_19d", None)
        enc = enc32 if callable(enc32) else (enc19 if callable(enc19) else None)
        if enc is None:
            return

        if hasattr(pol, "set_action_encoder") and callable(getattr(pol, "set_action_encoder")):
            pol.set_action_encoder(enc)
        elif hasattr(pol, "action_encoder_fn"):
            pol.action_encoder_fn = enc
    except Exception:
        pass


def _pick_legal_actions(entry: dict):
    """
    候補手を取り出す優先順:
      1) top-level entry['legal_actions']
      2) action_result['legal_actions']
      3) action_result.substeps[*].legal_actions（後ろから）
      4) state_before / state_after の top-level 'legal_actions'
      5) state_before/after の me / opp の 'legal_actions'
      見つからなければ []
    """
    if not isinstance(entry, dict):
        return []

    # 1) top-level
    la = entry.get("legal_actions")
    if isinstance(la, list) and la:
        return la

    # 2) action_result 直下
    ar = entry.get("action_result") or {}
    if isinstance(ar, dict):
        la2 = ar.get("legal_actions")
        if isinstance(la2, list) and la2:
            return la2

        # 3) substeps を後ろから
        subs = ar.get("substeps")
        if isinstance(subs, list) and subs:
            for st in reversed(subs):
                if isinstance(st, dict):
                    la3 = st.get("legal_actions")
                    if isinstance(la3, list) and la3:
                        return la3

    # 4) state_before / state_after 直下
    for k in ("state_before", "state_after"):
        st = entry.get(k) or {}
        if isinstance(st, dict):
            la4 = st.get("legal_actions")
            if isinstance(la4, list) and la4:
                return la4

    # 5) state_* の me / opp 内
    for k in ("state_before", "state_after"):
        st = entry.get(k) or {}
        if isinstance(st, dict):
            for side in ("me", "opp"):
                s = st.get(side) or {}
                if isinstance(s, dict):
                    la5 = s.get("legal_actions")
                    if isinstance(la5, list) and la5:
                        return la5

    return []
