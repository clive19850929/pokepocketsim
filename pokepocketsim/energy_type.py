"""
EnergyType
----------
ポケモンカードで使われるエネルギータイプを列挙で表します。
必要に応じて新しいタイプ（ドラゴン、フェアリー など）を追加してください。
"""

from enum import Enum

class EnergyType(Enum):
    GRASS     = "grass"      # くさ
    FIRE      = "fire"       # ほのお
    WATER     = "water"      # みず
    LIGHTNING = "lightning"  # でんき
    PSYCHIC   = "psychic"    # エスパー
    FIGHTING  = "fighting"   # かくとう
    DARK      = "dark"       # あく
    METAL     = "metal"      # はがね
    COLORLESS = "colorless"  # 無色
    DRAGON    = "dragon"     # ドラゴン

    def __str__(self) -> str:
        """Enum値をそのまま返す（例: 'lightning'）"""
        return self.value
