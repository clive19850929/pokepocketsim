from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from .protocols import ICard
from .card_base import CardBase
from .cards_enum import Cards

if TYPE_CHECKING:
    from .player import Player
    from .attack import EnergyType
    from .condition import ConditionBase

TOOL_CLASS_DICT: dict[str, type] = {}

def register_tool(cls):
    """
    どうぐクラスを自動登録するデコレータ
    ・クラス名キー（Balloon 等）
    ・日本語名キー（JP_NAME があれば）  
    → TOOL_CLASS_DICT["ふうせん"] = Balloon
    """

    TOOL_CLASS_DICT[cls.__name__] = cls
    if hasattr(cls, "JP_NAME"):
        TOOL_CLASS_DICT[cls.JP_NAME] = cls
    return cls

class Tool(CardBase):
    """
    《ポケモンのどうぐ》共通クラス  
    - 生成時に Cards.* を渡して固定 ID / 名称をセット
    """
    def __init__(self, card_enum: Cards) -> None:
        super().__init__(card_enum)   # Item 側で card_id / name を設定
        self.is_tool = True           # どうぐフラグ
    
    def __str__(self) -> str:
        return self.name
    
    def can_attach_to(self, target) -> bool:
        """どうぐを付けることができるかチェック"""
        # ポケモンにしか付けられない
        if not hasattr(target, "hp") or target.hp is None:
            return False
        
        # 既にどうぐが付いている場合は付けられない
        if hasattr(target, "tools") and target.tools:
            return False
        
        return True
    
    def use(self, player: "Player", target) -> bool:
        """どうぐを使用してポケモンに付ける"""
        if not self.can_attach_to(target):
            return False
        
        if not hasattr(target, "tools"):
            target.tools = []
        target.tools.append(self)
        
        if self in player.hand:
            player.hand.remove(self)
        
        player.log_print(f"{player.name} は {self.name}（どうぐ）を {target.name} に付けた")
        return True


@register_tool
class Balloon(Tool):
    """ふうせん: ポケモンに付けるとにげるエネルギーコストが-2"""
    JP_NAME = "ふうせん"

    def __init__(self, **kwargs):
        super().__init__(Cards.BALLOON, **kwargs)

    def get_retreat_cost_reduction(self) -> int:
        return 2

    # 追加: 装着時フック
    def on_tool_attached(self, pokemon) -> None:
        """
        ポケモンに付いた瞬間に retreat_cost を減算して上書き。
        同種/別種のツールが複数あっても合算される。
        """
        # ベース値を一度だけ保持
        if not hasattr(pokemon, "_retreat_cost_base"):
            setattr(pokemon, "_retreat_cost_base", int(getattr(pokemon, "retreat_cost", 0)))

        # ツールごとの減少量テーブルに登録
        mods = getattr(pokemon, "_retreat_cost_mods", None)
        if mods is None:
            mods = {}
            setattr(pokemon, "_retreat_cost_mods", mods)
        mods[id(self)] = int(self.get_retreat_cost_reduction())

        # 合計減少量を反映して実体の retreat_cost を更新（ロガーはこの値を読む）
        base = int(getattr(pokemon, "_retreat_cost_base", 0))
        total_red = sum(int(v) for v in mods.values())
        pokemon.retreat_cost = max(0, base - total_red)

    # 追加: 解除時フック
    def on_tool_detached(self, pokemon) -> None:
        """外れたら自分の寄与分を削除して再計算。無くなればベースに戻す。"""
        mods = getattr(pokemon, "_retreat_cost_mods", None) or {}
        mods.pop(id(self), None)

        base = int(getattr(pokemon, "_retreat_cost_base", int(getattr(pokemon, "retreat_cost", 0))))
        if mods:
            total_red = sum(int(v) for v in mods.values())
            pokemon.retreat_cost = max(0, base - total_red)
        else:
            # もう減少要因が無ければベースに戻して掃除
            pokemon.retreat_cost = base
            if hasattr(pokemon, "_retreat_cost_mods"):
                delattr(pokemon, "_retreat_cost_mods")
            if hasattr(pokemon, "_retreat_cost_base"):
                delattr(pokemon, "_retreat_cost_base")


