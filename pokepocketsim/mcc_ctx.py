# mcc_ctx.py
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional

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

def _count_zone_enums(zone, *, top_only: bool, include_stack_bottom: bool) -> Counter:
    """
    zone 内の枚数を enum カウントにして返す。
    - top_only=True  : 進化スタックは最上段のみを数える
    - include_stack_bottom=True: スタックの下まで全て数える
    """
    cnt = Counter()
    if not zone:
        return cnt
    for c in zone:
        if isinstance(c, list):
            if not c:
                continue
            if include_stack_bottom:
                for obj in c:
                    cid = _enum_from_card(obj)
                    if cid:
                        cnt[cid] += 1
            else:
                obj = c[-1]
                cid = _enum_from_card(obj)
                if cid:
                    cnt[cid] += 1
        else:
            cid = _enum_from_card(c)
            if cid:
                cnt[cid] += 1
    return cnt

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

def load_archetypes(json_obj_or_list) -> List[Dict[str, Any]]:
    """
    [{ "enums": [cid,...(60)], "weight": 0.6 }, ...] を返す。
    ファイルから読む場合は json.load(...) の戻り値を渡すだけ。
    """
    if isinstance(json_obj_or_list, list):
        out = []
        for a in json_obj_or_list:
            if not isinstance(a, dict):
                continue
            enums = a.get("enums") or []
            if isinstance(enums, list) and len(enums) >= 60:
                w = float(a.get("weight", 1.0))
                out.append({"enums": [int(x) for x in enums], "weight": w})
        return out
    return []

def load_card_pool_counts(json_obj_or_dict) -> Dict[int, int]:
    """
    { card_id: max_count } の辞書を返す。
    """
    out: Dict[int, int] = {}
    if isinstance(json_obj_or_dict, dict):
        for k, v in json_obj_or_dict.items():
            try:
                cid = int(k) if not isinstance(k, int) else k
                cnt = int(v)
                if cid != 0 and cnt > 0:
                    out[cid] = cnt
            except Exception:
                pass
    return out

def build_mcc_ctx(player, *,
                  forbid_peek: bool = True,
                  archetypes: Optional[List[Dict[str, Any]]] = None,
                  card_pool_counts: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """
    mcc_sampler に渡す ctx を構築して返す。
    - forbid_peek=True で allow_peek_* を False 固定
    - 自分: initial_deck_enums と公開カウントから bag を構成できる材料を入れる
    - 相手: アーキタイプ prior または カードプール を入れる
    """
    p = player
    m = getattr(p, "match", None)

    # -------- 覗き見フラグ --------
    ctx: Dict[str, Any] = {
        "match": m,
        "player": p,
        "allow_peek_me": (False if forbid_peek else False),
        "allow_peek_opp": (False if forbid_peek else False),
    }

    # -------- 自分側: 既知60枚（必須） --------
    my_known = getattr(p, "initial_deck_enums", None)
    if isinstance(my_known, list) and len(my_known) >= 60:
        ctx["my_known_decklist_enums"] = [int(x) for x in my_known]
    else:
        # フォールバック: 現在の山札のみ（完全ではないが最低限）
        try:
            cur = []
            for c in getattr(p.deck, "cards", []) or []:
                obj = c[-1] if isinstance(c, list) and c else c
                cur.append(_enum_from_card(obj))
            ctx["my_known_decklist_enums"] = cur
        except Exception:
            ctx["my_known_decklist_enums"] = []

    # 公開情報（自分はスタックの下まで確定で引ける）
    my_hand = _count_zone_enums(getattr(p, "hand", []), top_only=False, include_stack_bottom=True)
    my_trash = _count_zone_enums(getattr(p, "discard_pile", []), top_only=False, include_stack_bottom=True)
    my_active = _count_zone_enums(getattr(p, "active_card", []), top_only=False, include_stack_bottom=True)
    my_bench = _merge_counters(*[
        _count_zone_enums(stk, top_only=False, include_stack_bottom=True)
        for stk in (getattr(p, "bench", []) or [])
    ])
    # 公開プライズ・公開サーチ等が別管理ならここで加算
    my_revealed_prizes = _count_zone_enums(getattr(p, "revealed_prize_cards", []), top_only=False, include_stack_bottom=True)
    my_revealed_search = _count_zone_enums(getattr(p, "revealed_cards", []), top_only=False, include_stack_bottom=True)

    my_public = _merge_counters(my_hand, my_trash, my_active, my_bench, my_revealed_prizes, my_revealed_search)
    ctx["my_public_enum_counts"] = {int(cid): int(n) for cid, n in my_public.items() if cid != 0 and n > 0}

    # -------- 相手側: 観測（トップのみ/下は見えない） --------
    opp = getattr(p, "opponent", None)
    if opp is not None:
        opp_trash = _count_zone_enums(getattr(opp, "discard_pile", []), top_only=False, include_stack_bottom=False)
        opp_active_top = _count_zone_enums(getattr(opp, "active_card", []), top_only=True, include_stack_bottom=False)
        opp_bench_tops = _merge_counters(*[
            _count_zone_enums(stk, top_only=True, include_stack_bottom=False)
            for stk in (getattr(opp, "bench", []) or [])
        ])
        opp_tools = _merge_counters(
            _count_zone_enums(getattr(getattr(opp, "active_card", [None])[-1] if getattr(opp, "active_card", []) else None, "tools", []), top_only=False, include_stack_bottom=False)
        )
        opp_revealed_prizes = _count_zone_enums(getattr(opp, "revealed_prize_cards", []), top_only=False, include_stack_bottom=False)
        opp_revealed_search = _count_zone_enums(getattr(opp, "revealed_cards", []), top_only=False, include_stack_bottom=False)

        obs = _merge_counters(opp_trash, opp_active_top, opp_bench_tops, opp_tools, opp_revealed_prizes, opp_revealed_search)
        ctx["opp_observed_enum_counts"] = {int(cid): int(n) for cid, n in obs.items() if cid != 0 and n > 0}

        # プライズ既知/未知
        known_prizes = []
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
        total_prize_slots = 6
        ctx["opp_prize_unknown"] = max(0, total_prize_slots - len(known_prizes))

        # 手札枚数のみ（中身は見ない）
        ctx["opp_hand_size"] = _safe_len(getattr(opp, "hand", []))

    # -------- prior（どちらか必須） --------
    if isinstance(archetypes, list) and archetypes:
        ctx["opp_archetypes"] = archetypes
    elif isinstance(card_pool_counts, dict) and card_pool_counts:
        ctx["opp_card_pool_counts"] = {int(cid): int(n) for cid, n in card_pool_counts.items() if n > 0}

    return ctx
