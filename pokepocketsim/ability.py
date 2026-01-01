import random
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Any,
    Dict,
    Callable,
    Union,
    cast,
)
import uuid
from .attack import EnergyType
from .protocols import ICard, IPlayer

if TYPE_CHECKING:
    from .player import Player
    from .card import Card


# TODO: fix multiple abilities
class Ability:

    class ElekiDynamo:
        def __init__(self) -> None:
            self.name: str = "エレキダイナモ"
            # 任意: 固定IDがあるならここにも保持
            self.ability_id: int = 200001

        def able_to_use(self, player: "Player") -> bool:
            from .cards_enum import Cards
            has_energy = any(getattr(card, "card_enum", None) == Cards.LIGHTNING_ENERGY
                             for card in player.discard_pile)
            has_bench = any(stack for stack in player.bench)
            return has_energy and has_bench

        def gather_actions(self, player: "Player", card_using_ability: "Card") -> List[Any]:
            from .action import Action, ActionType
            from .cards_enum import Cards

            if not self.able_to_use(player):
                return []

            # トラッシュの雷エネルギーは“束”として扱い、ターゲットごとに1アクションだけ出す
            bench_slots = [(idx, stack) for idx, stack in enumerate(player.bench) if stack]
            actions: List[Action] = []

            for bench_idx, stack in bench_slots:
                target = stack[-1]
                label = f"[特性] {self.name}：トラッシュの雷エネルギーを{target.name}に付ける"

                # payload は UUID を持たない（=どの雷でもよい）。energy_id は enum の数値IDを入れる。
                try:
                    energy_enum_id = int(Cards.LIGHTNING_ENERGY.value[0])
                except Exception:
                    energy_enum_id = 0

                payload = {
                    "action": [8, self.ability_id],
                    "ability_id": self.ability_id,
                    "energy_id": energy_enum_id,             # ← 束を表すため “enum の数値ID”
                    "target_id": str(getattr(target, "id")), # ← ターゲットは UUID で特定
                    "selected_target_bench_idx": bench_idx + 1,
                    "using_card_id": str(getattr(card_using_ability, "id")),  # ← 重複抑制で区別するため
                }

                def _act(player: "Player",
                         using_card_id=getattr(card_using_ability, "id"),
                         target_id=getattr(target, "id"),
                         payload=payload,
                         self=self):
                    return self.use(player, using_card_id, target_id, payload)

                a = Action(
                    label,
                    _act,
                    ActionType.USE_ABILITY,
                    can_continue_turn=True,
                    card_class=None,
                    extra=payload,
                )
                # 解析器向けに ability_id を明示
                a.ability_id = self.ability_id
                actions.append(a)

            return actions

        def use(self, player: "Player", using_card_id, target_id, payload: dict) -> dict:
            from .cards_enum import Cards
            from .card import Card

            # 能力を使う本人（ベンチ/バトル場いずれかにいる想定）
            using_stack = player.find_by_id(
                player.bench + ([player.active_card] if player.active_card else []),
                using_card_id
            )
            if using_stack is None:
                return {"error": "using_card_not_found", **payload}
            using_card = using_stack[-1]

            if getattr(using_card, "has_used_ability", False):
                return {"error": "already_used", **payload}

            # “どの雷でもよい”ので、最初に見つかった雷エネルギー1枚を使う
            energy_card = next(
                (c for c in player.discard_pile if getattr(c, "card_enum", None) == Cards.LIGHTNING_ENERGY),
                None
            )
            if energy_card is None:
                return {"error": "no_energy", **payload}

            target_stack = player.find_by_id(player.bench, target_id)
            if target_stack is None:
                return {"error": "target_not_found", **payload}
            target = target_stack[-1]

            Card.add_energy(player, target, energy_card)
            player.discard_pile.remove(energy_card)
            using_card.has_used_ability = True

            result = dict(payload)
            result["ok"] = True
            return result
