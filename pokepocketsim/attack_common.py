from enum import Enum
from typing import Dict, Any

class EnergyType(Enum):
    """Energy types in the game."""

    WATER = "water"
    FIRE = "fire"
    GRASS = "grass"
    LIGHTNING = "lightning"
    PSYCHIC = "psychic"
    FIGHTING = "fighting"
    DARKNESS = "darkness"
    METAL = "metal"
    COLORLESS = "colorless"
    ANY = "any"


# Attack data dictionary with damage values and energy costs
ATTACKS: Dict[str, Dict[str, Any]] = {

    "suitoru": {
        "name_ja": "すいとる",
        "damage": 10,
        "energy": {"colorless": 1},
        "has_side_effect": True
    },
    "energy_gift": {
        "name_ja": "エナジーギフト",
        "damage": None,
        "energy": {"colorless": 1},
        "has_side_effect": True
    },
    "wonder_cotton": {
        "name_ja": "ワンダーコットン",
        "damage": None,
        "energy": {"colorless": 1},
        "has_side_effect": True
    },
    "slash": {
        "name_ja": "きりさく",
        "damage": 50,
        "energy": {"colorless": 2},
        "has_side_effect": False
    },
    "blizzard_burst": {
        "name_ja": "ブリザードバースト",
        "damage": 130,
        "energy": {"water": 2, "colorless": 1},
        "has_side_effect": True
    },
    "voltage_burst": {
        "name_ja": "ボルテージバースト",
        "damage": 130,
        "energy": {"lightning": 2, "colorless": 1},
        "has_side_effect": True
    },
    "jittosuru": {
        "name_ja": "じっとする",
        "damage": 0,
        "energy": {"colorless": 1},
        "has_side_effect": True
    },
    "raitoninnguboru": {
        "name_ja": "ライトニングボール",
        "damage": 50,
        "energy": {"colorless": 1, "lightning": 2},
        "has_side_effect": False
    },
    "nakamawoyobu": {
        "name_ja": "なかまをよぶ",
        "damage": 0,
        "energy": {"colorless": 1},
        "has_side_effect": True
    },
    "batibati": {
        "name_ja": "バチバチ",
        "damage": 20,
        "energy": {"lightning": 1},
        "has_side_effect": False
    },
    "katakiuti": {
        "name_ja": "カタキウチ",
        "damage": 50,
        "energy": {"colorless": 1, "fighting": 1},
        "has_side_effect": True
    },
    "randkurasshu": {
        "name_ja": "ランドクラッシュ",
        "damage": 100,
        "energy": {"colorless": 1, "fighting": 2},
        "has_side_effect": False
    },
}
