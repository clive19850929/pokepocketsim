##グッズとサポートは効果をそれぞれのクラスで定義しているため、from .item import .supporterにインポートするカードを定義させる必要がある。

import random
import uuid
from .cards_enum import Cards
from .card_base import CardBase
from .attack import EnergyType  # Attackは循環インポート回避のため削除
from .ability import Ability
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Union, Callable, Type, cast
from collections import Counter
from .cards_db import CARDS_DICT


if TYPE_CHECKING:
    from .player import Player


class Card(CardBase):
    # ★ 1) hp / type / attacks / retreat_cost を Optional 化
    #    2) 任意の追加フラグを受け取れる **extra を追加
    def __init__(
        self,
        card_enum: Cards,
        hp: Optional[int] = None,
        type: Optional[Union[str, "EnergyType"]] = None,
        attacks: Optional[List[Callable]] = None,
        retreat_cost: Optional[int] = None,
        ability: Optional[Any] = None,
        weakness: Optional["EnergyType"] = None,
        is_basic: bool = False,
        is_ex: bool = False,
        evolves_from: Optional[Union["Cards", "Card"]] = None,
        **extra: Any,
    ) -> None:
        super().__init__(card_enum)
        # --- 固定 ID での管理 -----------------
        self.card_enum: Cards      = card_enum
        self.card_id: int          = card_enum.value[0]   # ← 決定論 ID
        self.name: str             = card_enum.value[1]   # 和名（または英名）
        # ★必要なら「区別用 uuid」も保持
        self.instance_id: uuid.UUID = uuid.uuid4()

        # ★ None は None のまま保持（0 にしない）
        self.max_hp: Optional[int] = hp
        self.hp: Optional[int] = hp
        self.damage_counters: int = 0
        self.type: Union[str, EnergyType, None] = type
        self.attacks: List[Callable] = attacks if attacks is not None else []   # ★
        self.retreat_cost: Optional[int] = retreat_cost
        self.modifiers: List[Any] = []
        self.ability: Optional[Any] = ability
        self.conditions: List[Any] = []
        self.weakness: Optional[EnergyType] = weakness
        self.is_basic: bool = is_basic
        self.is_ex: bool = is_ex
        self.has_used_ability: bool = False
        self.evolves_from: Optional[Union[Cards, "Card"]] = evolves_from
        self.can_evolve: bool = False
        self.attached_energies: list = []
        self.tools: list = []
        self.entered_turn: Optional[int] = None   # ★ 進化禁止判定用

        # ★ extra で渡された任意フィールドをそのまま属性化
        for k, v in extra.items():
            setattr(self, k, v)

        # ★ 3) 追加フィールド (例: is_energy=True) を動的に取り込む
        for key, value in extra.items():
            setattr(self, key, value)

    def add_condition(self, condition: Any) -> None:
        if any(isinstance(cond, condition.__class__) for cond in self.conditions):
            # TODO: sometimes the conditions do not get removed off benched pokemon
            return
        self.conditions.append(condition)

    def remove_condition(self, condition_name: str) -> None:
        self.conditions = [
            condition for condition in self.conditions if condition != condition_name
        ]

    def update_conditions(self) -> None:
        self.conditions = [
            condition for condition in self.conditions if not condition.rid()
        ]

    @staticmethod
    def add_energy(player: "Player", card: "Card", energy_card: "Card") -> None:
        if not hasattr(card, "attached_energies"):
            card.attached_energies = []
        card.attached_energies.append(energy_card)
        if player.print_actions:
            attached_names = [getattr(e, "name", str(e)) for e in card.attached_energies]
            counts = Counter(attached_names)
            jp_str = ", ".join(
                f"{name}×{count}" if count > 1 else f"{name}"
                for name, count in counts.items()
            )
            print(f"{card.name} の現在のエネルギー: {jp_str if jp_str else 'エネルギーなし'}")

    def remove_energy(self, energy_name: str) -> None:
        attached = getattr(self, "attached_energies", [])
        for i, card in enumerate(attached):
            if getattr(card, "name", None) == energy_name:
                attached.pop(i)
                break
        else:
            raise ValueError(f"Energy card '{energy_name}' not found in attached_energies.")

    def get_retreat_cost(self, owner=None) -> int:
        """
        現在のにげるコスト（どうぐ等の軽減込み）を返す。
        - どうぐが get_retreat_cost_reduction() を持っていれば合算して減算
        - 下限は 0
        """
        base = int(self.retreat_cost or 0)
        reduction = 0
        for t in getattr(self, "tools", []) or []:
            if hasattr(t, "get_retreat_cost_reduction"):
                try:
                    reduction += int(t.get_retreat_cost_reduction())
                except Exception:
                    pass
        return max(0, base - reduction)

    def get_effective_retreat_cost(self) -> int:
        return self.get_retreat_cost()
    
    def remove_retreat_cost_energy(self, player=None, cost: Optional[int] = None) -> None:
        """
        逃げるコスト分のエネルギーをトラッシュする。
        cost が与えられなければ get_retreat_cost() を用いる。
        """
        pay = self.get_retreat_cost(player) if cost is None else max(0, int(cost))
        if pay <= 0:
            return

        attached = getattr(self, "attached_energies", None)
        if attached is None:
            self.attached_energies = []
            attached = self.attached_energies

        if len(attached) < pay:
            if player and getattr(player, "print_actions", False):
                player.log_print("逃げるのに必要なエネルギーが足りません")
            return

        used = []

        if player is not None and not getattr(player, "is_bot", True):
            if getattr(player, "print_actions", False):
                player.log_print(f"逃げるために{pay}個のエネルギーを消費します。")
            for k in range(pay):
                print("どのエネルギーを消費しますか？")
                for i, e in enumerate(attached):
                    print(f"{i}: {getattr(e, 'name', str(e))}")
                hi = len(attached) - 1
                while True:
                    try:
                        sel = int(input(f"{k+1}個目のエネルギーを選んでください (0-{hi}): "))
                        if 0 <= sel <= hi:
                            break
                    except Exception:
                        pass
                    print("無効な番号です。")
                used.append(attached.pop(sel))
        else:
            import random
            for _ in range(pay):
                if not attached:
                    break
                sel = random.randrange(len(attached))
                used.append(attached.pop(sel))

        if player is not None:
            player.discard_pile.extend(used)
            if used and getattr(player, "print_actions", False):
                names = ", ".join(getattr(c, "name", str(c)) for c in used)
                player.log_print(f"逃げるコストとして {names} をトラッシュしました")



    def get_total_energy(self) -> int:
        return len(getattr(self, "attached_energies", []))

    def get_energy_count(self, type_name: str) -> int:
        attached = getattr(self, "attached_energies", [])
        return sum(1 for card in attached if getattr(card, "name", None) == type_name)

    def evolve(self, evolved_card_name: Cards) -> None:
        """
        Evolves this card into the given evolved card.

        Args:
            evolved_card_name (Cards): The card to evolve into.
        """
        # 循環 import 回避のためメソッド内で遅延インポート
        from .cards_db import CARDS_DICT

        if evolved_card_name not in CARDS_DICT:
            raise ValueError(f"Card {evolved_card_name} does not exist in CARDS_DICT.")

        evolved_card_info = CARDS_DICT[evolved_card_name]

        if (
            evolved_card_info["evolves_from"] is None
            or evolved_card_info["evolves_from"].value != self.name
        ):
            raise ValueError(
                f"{evolved_card_name.value} cannot evolve from {self.name}"
            )

        # 進化前のダメージ量を計算
        if self.max_hp is not None and self.hp is not None:
            damage = self.max_hp - self.hp
        else:
            damage = 0

        # evolved_card_name.valueはtupleなので、2番目の要素（日本語名）を取得
        if isinstance(evolved_card_name.value, tuple):
            self.name = evolved_card_name.value[1]  # 日本語名を取得
        else:
            self.name = str(evolved_card_name.value)
        self.max_hp = evolved_card_info["hp"]
        if self.max_hp is not None:
            self.hp = self.max_hp - damage  # 進化後もダメージを引き継ぐ
            # ダメカンも適切に同期
            self.damage_counters = damage // 10
        self.type = evolved_card_info["type"]
        self.attacks = evolved_card_info["attacks"]
        self.retreat_cost = evolved_card_info["retreat_cost"]
        self.ability = evolved_card_info["ability"]
        self.weakness = evolved_card_info["weakness"]
        self.is_basic = evolved_card_info["is_basic"]
        self.is_ex = evolved_card_info["is_ex"]
        self.evolves_from = evolved_card_info["evolves_from"]
        self.can_evolve = False

    def __repr__(self):
        # 手札表示時は show_attached_energies が False ならエネルギー情報を出さない
        show_attached = getattr(self, "show_attached_energies", True)
        if self.hp is not None:
            if not show_attached:
                return f"{self.name} (HP: {self.hp})"
            attached = getattr(self, "attached_energies", [])
            from collections import Counter
            counts = Counter(getattr(card, "name", str(card)) for card in attached)
            if counts:
                attached_str = ", ".join(
                    f"{name}×{count}" if count > 1 else f"{name}"
                    for name, count in counts.items()
                )
                return f"{self.name} (HP: {self.hp}) [{attached_str}]"
            else:
                return f"{self.name} (HP: {self.hp})"
        else:
            return self.name

    def serialize(self) -> Dict[str, Any]:
        ability_name = None
        if self.ability:
            ability_name = getattr(self.ability, "name", None)

        weakness_name = None
        if self.weakness:
            weakness_name = self.weakness.name

        evolves_from_name = None
        if self.evolves_from:
            if isinstance(self.evolves_from, Cards):
                evolves_from_name = self.evolves_from.value
            else:
                # If evolves_from is a Card object
                other_card = self.evolves_from
                evolves_from_name = other_card.name

        return {
            "name": self.name,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "type": str(self.type),
            "attached_energies": [e.name for e in getattr(self, "attached_energies", [])],
            "retreat_cost": self.get_retreat_cost(),
            "ability": ability_name,
            "weakness": weakness_name,
            "is_basic": self.is_basic,
            "is_ex": self.is_ex,
            "evolves_from": evolves_from_name,
            "can_evolve": self.can_evolve,
            "conditions": [cond.serialize() for cond in self.conditions],
        }

    """
    それぞれのトレーナーズはカード内で遅延 importしている。
    """

    @classmethod
    def create_card(cls, card_enum):
        from .cards_db import CARDS_DICT
        from .attack import Attack  # 遅延import

        card_info = CARDS_DICT[card_enum]

        # グッズの場合
        if card_info.get("is_item"):
            from .item import ITEM_CLASS_DICT
            cls_ = ITEM_CLASS_DICT.get(card_info["name"])
            card = cls_()
            card.card_enum = card_enum
            return card

        # どうぐの場合
        elif card_info.get("is_tool"):
            from .tool import TOOL_CLASS_DICT
            cls_ = TOOL_CLASS_DICT[card_info["name"]]
            card = cls_()
            card.card_enum = card_enum
            return card

        # サポーターの場合
        if card_info.get("is_supporter"):
            from .supporter import SUPPORTER_CLASS_DICT
            cls_ = SUPPORTER_CLASS_DICT[card_info["name"]]
            card = cls_()
            card.card_enum = card_enum
            return card

        # スタジアムの場合
        if card_info.get("is_stadium"):
            from .stadium import BowlTown, Stadium
            if card_info["name"] == "ボウルタウン":
                card = BowlTown()
            else:
                card = Stadium()
            card.card_enum = card_enum
            return card

        # それ以外（ポケモンカード）はstr→関数変換してから生成
        card_info = dict(card_info)  # dict化で編集可
        if "attacks" in card_info and card_info["attacks"] and isinstance(card_info["attacks"][0], str):
            card_info["attacks"] = [getattr(Attack, atk) for atk in card_info["attacks"]]
        card = cls(card_enum=card_enum, **card_info)
        card.card_enum = card_enum
        return card


