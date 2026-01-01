from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .card import Card

def add_damage_counters(card: "Card", counters: int) -> None:
    """ダメカンを追加し、hp も同期する"""
    if card.max_hp is None:           # トレーナーなど
        return
    card.damage_counters += counters
    card.hp = card.max_hp - card.damage_counters * 10 

def remove_damage_counters(card: "Card", counters: int) -> None:
    """ダメカンを取り除き、hp も同期する"""
    if card.max_hp is None:
        return
    # ダメカンは0未満にならないようにする
    card.damage_counters = max(0, card.damage_counters - counters)
    card.hp = card.max_hp - card.damage_counters * 10
