# pokepocketsim/decks.py

from .cards_enum import Cards

# デッキレシピ群
deck01 = [
    #ポケモン
    (Cards.SHIBISHIRASU,      4),
    (Cards.SHIBIBIRU,         4),
    (Cards.EMONGA,         2),
    (Cards.TERAKION,         2),
    (Cards.ZEKROM_EX,      4),
    #グッズ
    (Cards.ULTRA_BALL,     4),
    (Cards.SWITCH,         4),
    (Cards.POKEMON_CATCHER,2),
    #ポケモンのどうぐ
    (Cards.BALLOON,     3),
    #サポーター
    (Cards.BOSS_ORDERS,    2),
    (Cards.MAKOMO,         2),
    (Cards.CROWN,        2),
    (Cards.CHEREN,          4),
    #スタジアム
    (Cards.BOWL_TOWN,      3),
    #エネルギー
    (Cards.LIGHTNING_ENERGY,13),
    (Cards.FIGHTING_ENERGY,5),
]

deck02 = [
    #ポケモン
    (Cards.SHIBISHIRASU,      4),
    (Cards.SHIBIBIRU,         4),
    (Cards.EMONGA,         2),
    (Cards.TERAKION,         2),
    (Cards.ZEKROM_EX,      4),
    #グッズ
    (Cards.ULTRA_BALL,     4),
    (Cards.SWITCH,         4),
    (Cards.POKEMON_CATCHER,2),
    #ポケモンのどうぐ
    (Cards.BALLOON,     3),
    #サポーター
    (Cards.BOSS_ORDERS,    2),
    (Cards.MAKOMO,         2),
    (Cards.CROWN,        2),
    (Cards.CHEREN,          4),
    #スタジアム
    (Cards.BOWL_TOWN,      3),
    #エネルギー
    (Cards.LIGHTNING_ENERGY,13),
    (Cards.FIGHTING_ENERGY,5),
]

# 名前付きで管理できるようdictにまとめるのがおすすめ
ALL_DECK_RECIPES = {
    "deck01": deck01,
    "deck02": deck02,
    # 今後追加するときはここに追記
}

# デッキ名と日本語のタイプ名も管理できる
DECK_TYPE_NAMES = {
    "deck01": "ゼクロムシビビールテラキオン",
    "deck02": "ゼクロムシビビールテラキオン",
    # 追加デッキ名...
}

def make_deck_from_recipe(recipe):
    from pokepocketsim.card import Card
    deck = []
    for card_enum, count in recipe:
        deck.extend([Card.create_card(card_enum) for _ in range(count)])
    if len(deck) != 60:
        print(f"警告: デッキの枚数が{len(deck)}枚です（60枚ではありません）")
    return deck[:60]
