# pokepocketsim/deck.py
import random
import uuid
from typing import List, Optional, Any, Type
from .card import Card
from .energy_cards import EnergyCard
from .protocols import ICard

class Deck:
    """山札クラス：ポケモン／トレーナー／エネルギーをすべて物理カードで管理"""

    def __init__(
        self, cards: Optional[List[Any]] = None
    ) -> None:
        self.uid: uuid.UUID = uuid.uuid4()
        self.cards: List[Any] = cards if cards is not None else []

    # ---------- 追加／内部ユーティリティ ---------- #

    def _add_card(self, card: Card) -> None:
        self.cards.append(card)

    def _add_item(self, card_class: Type) -> None:
        self.cards.append(card_class)

    # 旧インターフェース互換（Card のみ）
    def add_card(self, card: Card) -> None:
        self.add(card)

    # 新しい統合メソッド
    def add(self, card_or_item: Any) -> None:
        if isinstance(card_or_item, Card):
            self._add_card(card_or_item)
        else:
            self._add_item(card_or_item)

    # ---------- ドロー関連 ---------- #

    def draw_card(self) -> "ICard":
        if not self.cards:
            raise ValueError("Deck is empty")    # ← 山札切れを例外にする
        return self.cards.pop(0)


    # ---------- デバッグ表示 ---------- #
    def __repr__(self) -> str:
        return "Deck:\n" + "\n".join(str(card) for card in self.cards)

    def extend(self, items) -> None:
        """リストやイテラブルでカード群を追加"""
        for itm in items:
            self.add(itm)
    
    def shuffle(self) -> None:
        """山札をランダムにシャッフルする"""
        random.shuffle(self.cards)