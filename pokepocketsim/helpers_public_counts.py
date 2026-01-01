# helpers_public_counts.py

from collections import Counter

def build_my_public_enum_counts(player) -> dict[int, int]:
    """
    自分の公開ゾーンを全て数えて {cid: cnt} を返す。
    進化スタックは“下”まで全部数える（自分だから完全情報）。
    """
    def _count(zone, include_stack_bottom=True):
        from helpers_initial_deck import _collect_zone_enums
        return Counter(_collect_zone_enums(zone, include_stack_bottom=include_stack_bottom))

    hand   = _count(getattr(player, "hand", []))
    active = _count(getattr(player, "active_card", []))
    bench  = Counter()
    for stk in getattr(player, "bench", []) or []:
        bench += _count(stk)
    trash  = _count(getattr(player, "discard_pile", []))
    revealed_prize  = _count(getattr(player, "revealed_prize_cards", []))
    revealed_search = _count(getattr(player, "revealed_cards", []))

    total = hand + active + bench + trash + revealed_prize + revealed_search
    return {int(cid): int(n) for cid, n in total.items() if cid != 0 and n > 0}
