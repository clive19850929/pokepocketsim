##スタジアムを新しく定義する時は、必ず効果内で自身のカードを場に出すように設定する
##スタジアムを新しく定義する時は、必ずcard.pyにインポートするカードを定義させる。
##スタジアムを新しく定義する時は、必ずcard.pyにdef create_cardにスタジアムカード名を定義する。
##スタジアムを新しく定義する時は、必ずcards_db.pyにスタジアムカード名を定義する。
##スタジアムを新しく定義する時は、必ずcards_enum.pyにスタジアムカード名を定義する。
"""
［注意！！］ベンチの特徴量を機械学習に適応させるため、ゼロの大空洞など、ベンチ数が8に変化する場合は、
prepare_d3rlpy_data.pyのself.STADIUM_BENCH8_IDSにカードIDを入力する！！
"""


import random
import uuid
from .card_base import CardBase
from .cards_enum import Cards
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .player import Player
    from .card import Card

class Stadium(CardBase):
    def __init__(self, card_enum: Cards, **kwargs):
        super().__init__(card_enum=card_enum, is_basic=False, **kwargs)
        self.is_stadium = True

    def __str__(self):
        return self.name


class BowlTown(Stadium):
    """
    【ボウルタウン】
    おたがいのプレイヤーは、自分の番ごとに1回、自分の山札から
    たねポケモン（「ルールを持つポケモン」をのぞく）を1枚選び、
    ベンチに出してよい。そして山札を切る。
    「ルールを持つポケモン」とは特別なポケモンで、現時点では is_ex=True のポケモン
    """
    def __init__(self, **kwargs):
        super().__init__(card_enum=Cards.BOWL_TOWN, **kwargs)
        self.name = "ボウルタウン"
        self.is_stadium = True

    def __str__(self):
        return self.name

    def player_able_to_use(self, player: "Player") -> bool:
        return True

    def use(self, player: "Player") -> bool:
        return True
    
    def apply_effect(self, player: "Player") -> None:
        player.has_used_stadium_effect = True
        
        if len(player.bench) >= 5:
            return
        
        # たねポケモン抽出
        basic_pokemon = [
            card for card in player.deck.cards
            if (hasattr(card, "is_basic") and card.is_basic and 
                hasattr(card, "is_ex") and not card.is_ex and
                hasattr(card, "hp") and card.hp is not None)
        ]
        
        # ログ用変数
        chosen_card = None
        selected_id = 0

        if not basic_pokemon:
            player.log_print("山札にたねポケモンがありませんでした")
        else:
            if player.is_bot:
                chosen_card = random.choice(basic_pokemon + [None])
                selected_id = int(chosen_card.card_enum.value[0]) if chosen_card else 0
                if chosen_card:
                    player.log_print(f"{player.name} はボウルタウンの効果で {chosen_card.name} をベンチに出しました")
                else:
                    player.log_print("AIはスキップしました")
            else:
                player.log_print("ボウルタウンの効果で山札からたねポケモンを1枚選んでベンチに出せます（スキップは0、カードID入力も可）:")
                for i, card in enumerate(basic_pokemon, 1):
                    player.log_print(f"{i}: {card.name}（ID: {card.card_enum.value[0]}）")
                player.log_print("番号またはカードIDを入力してください（0はスキップ）:")
                while True:
                    idx = input("番号またはカードID: ").strip()
                    if idx == "0":
                        chosen_card = None
                        selected_id = 0
                        player.log_print("ボウルタウンの効果をスキップしました")
                        break
                    # インデックス番号（1始まり）
                    if idx.isdigit():
                        idx_int = int(idx)
                        if 1 <= idx_int <= len(basic_pokemon):
                            chosen_card = basic_pokemon[idx_int-1]
                            selected_id = int(chosen_card.card_enum.value[0])
                            player.log_print(f"{chosen_card.name} をベンチに出しました")
                            break
                        # カードID直接指定
                        for c in basic_pokemon:
                            if int(c.card_enum.value[0]) == idx_int:
                                chosen_card = c
                                selected_id = int(chosen_card.card_enum.value[0])
                                player.log_print(f"{chosen_card.name} をベンチに出しました")
                                break
                        if chosen_card:
                            break
                    print("無効な入力です。")
        
        # --- 行動ログにカードIDのみ記録 ---
        if hasattr(player, "last_action") and player.last_action:
            player.last_action.extra = {
                "selected_id": selected_id   # ベンチに出したカードID、スキップは0
            }
        
        # 実際にベンチに追加
        if chosen_card:
            player.deck.cards.remove(chosen_card)
            chosen_card.entered_turn = getattr(player.match, 'turn', 1)
            player.bench.append([chosen_card])
        
        player.deck.shuffle()