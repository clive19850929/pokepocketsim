# pokepocketsim/energy_card.py
from .card import Card


class EnergyCard(Card):
    """基本エネルギーカードを表すクラス"""

    def __init__(self, color: str) -> None:
        # name を "Psychic Energy" などに自動生成
        super().__init__(f"{color.title()} Energy")
        self.color: str = color
        self.is_energy: bool = True  # 後で判定に使う

    def __repr__(self) -> str:
        return f"{self.color.title()} Energy"
