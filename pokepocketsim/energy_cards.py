# pokepocketsim/energy_cards.py
from enum import Enum, auto
from .energy_card import EnergyCard   # ← クラスを直接インポート（相対インポート）


class EnergyCards(Enum):
    BASIC_PSYCHIC = auto()
    BASIC_FIRE    = auto()
    BASIC_WATER   = auto()
    BASIC_GRASS   = auto()
    BASIC_LIGHTNING = auto()
    BASIC_FIGHTING = auto()
    BASIC_DARK    = auto()
    BASIC_METAL   = auto()

    def create(self) -> EnergyCard:   # 返り値型ヒントもクラス
        """Enum -> 実体カードを生成"""
        mapping = {
            EnergyCards.BASIC_PSYCHIC: EnergyCard("psychic"),
            EnergyCards.BASIC_FIRE:    EnergyCard("fire"),
            EnergyCards.BASIC_WATER:   EnergyCard("water"),
            EnergyCards.BASIC_GRASS:   EnergyCard("grass"),
            EnergyCards.BASIC_LIGHTNING: EnergyCard("lightning"),
            EnergyCards.BASIC_FIGHTING: EnergyCard("fighting"),
            EnergyCards.BASIC_DARK:    EnergyCard("dark"),
            EnergyCards.BASIC_METAL:   EnergyCard("metal"),
        }
        return mapping[self]
