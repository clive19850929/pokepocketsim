##新しくカードを定義する時は、必ずcards_enum.pyにカード名を定義する。
##定義するのはデータベースで定義するカードの大文字英語名（アンダーバーのみ使用可）、ID番号、日本名
##グッズのみ日本名+英語名（item.pyで定義するカードの英語名）が必要になる

from enum import Enum

class Cards(Enum):
# --- ポケモン（10001～49999） ---
    MEWTWO_EX    = (10001, "ミュウツーEX")
    RALTS        = (10002, "ラルトス")
    KIRLIA       = (10003, "キルリア")
    ZEKROM_EX    = (10005, "ゼクロムex")
    MONMEN       = (10006, "モンメン")
    ERUHUN_EX    = (10007, "エルフーンex")
    SHIBISHIRASU = (10008, "シビシラス")
    SHIBIBIRU    = (10009, "シビビール")
    EMONGA       = (10010, "エモンガ")
    TERAKION     = (10011, "テラキオン")

# --- グッズ（50001～69999） ---
    ULTRA_BALL   = (50001, "ハイパーボール", "UltraBall")
    POKEMON_CATCHER = (50002, "ポケモンキャッチャー", "PokemonCatcher")
    SWITCH       = (50003, "ポケモンいれかえ", "Switch")
    POKEGEAR3_0  = (50004, "ポケギア3.0", "Pokegear3.0")
    CRUSHING_HAMMER = (50005, "クラッシュハンマー", "CrushingHammer")

# --- サポート（70001～79999） ---
    NANJAMO      = (70001, "ナンジャモ")
    BOSS_ORDERS  = (70002, "ボスの指令")
    PROFESSORS_RESEARCH = (70003, "博士の研究")
    CROWN        = (70004, "クラウン")
    CHEREN       = (70005, "チェレン")
    TOUKO        = (70006, "トウコ")
    MAKOMO       = (70007, "マコモ")

# --- スタジアム（80001～89999） ---
    BOWL_TOWN    = (80001, "ボウルタウン")

# --- ポケモンのどうぐ（90001～99999） ---
    BALLOON      = (90001, "ふうせん")

# --- エネルギーカード（00001～09999） ---
    GRASS_ENERGY     = (1, "草エネルギー")
    FIRE_ENERGY      = (2, "炎エネルギー")
    WATER_ENERGY     = (3, "水エネルギー")
    LIGHTNING_ENERGY = (4, "雷エネルギー")
    PSYCHIC_ENERGY   = (5, "超エネルギー")
    FIGHTING_ENERGY  = (6, "闘エネルギー")
    DARK_ENERGY      = (7, "悪エネルギー")
    METAL_ENERGY     = (8, "鋼エネルギー")

class Attacks(Enum):
# --- ワザ名（100001～） ---
    JITTOSURU = (100001, "じっとする", "jittosuru")
    SLASH = (100002, "きりさく", "slash")
    VOLTAGE_BURST = (100003, "ボルテージバースト", "voltage_burst")
    BLIZZARD_BURST = (100004, "ブリザードバースト", "blizzard_burst")
    ENERGY_GIFT = (100005, "エナジーギフト", "energy_gift")
    RAITONINNGUBORU = (100006, "ライトニングボール", "raitoninnguboru")
    BATIBATI = (100007, "バチバチ", "batibati")
    NAKAMAWOYOBU = (100008, "なかまをよぶ", "nakamawoyobu")
    KATAKIUTI = (100009, "カタキウチ", "katakiuti")
    RANDKURASSHU = (100010, "ランドクラッシュ", "randkurasshu")
    SUITORU = (100011, "すいとる", "suitoru")

class Ability(Enum):
# --- 特性名（200001～） ---
    ELEKIDYNAMO = (200001, "エレキダイナモ", "ElekiDynamo")