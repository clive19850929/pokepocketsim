# helpers_initial_deck.py

from collections import Counter

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

def _collect_zone_enums(zone, *, include_stack_bottom: bool) -> list[int]:
    out = []
    if not zone:
        return out
    for c in zone:
        if isinstance(c, list):
            if include_stack_bottom:
                for obj in c:
                    cid = _enum_from_card(obj)
                    if cid:
                        out.append(cid)
            else:
                obj = c[-1] if c else None
                cid = _enum_from_card(obj) if obj else 0
                if cid:
                    out.append(cid)
        else:
            cid = _enum_from_card(c)
            if cid:
                out.append(cid)
    return out

def finalize_initial_deck_enums(player) -> None:
    """
    初期60枚リストを「現在見えている全ゾーン」から復元して1回だけ確定。
    既にロック済みなら何もしない。
    """
    if getattr(player, "initial_deck_locked", False):
        return

    deck_ids   = _collect_zone_enums(getattr(player.deck, "cards", []), include_stack_bottom=True)
    hand_ids   = _collect_zone_enums(getattr(player, "hand", []), include_stack_bottom=True)
    prize_ids  = _collect_zone_enums(getattr(player, "prize_cards", []), include_stack_bottom=True)
    active_ids = _collect_zone_enums(getattr(player, "active_card", []), include_stack_bottom=True)
    bench_ids  = []
    for stk in getattr(player, "bench", []) or []:
        bench_ids.extend(_collect_zone_enums(stk, include_stack_bottom=True))
    trash_ids  = _collect_zone_enums(getattr(player, "discard_pile", []), include_stack_bottom=True)

    all_ids = deck_ids + hand_ids + prize_ids + active_ids + bench_ids + trash_ids
    # 60枚の検証（足りない/多い場合も、そのまま保持して OK。上流の生成に依存）
    player.initial_deck_enums = list(all_ids)
    player.initial_deck_locked = True
