from typing import Protocol, Optional, List, Union, TYPE_CHECKING
from .condition import ConditionBase
from .energy_type import EnergyType


if TYPE_CHECKING:
    from .attack import EnergyType


class ICard(Protocol):
    type: str | EnergyType | None
    hp: int
    max_hp: int
    name: str

    def add_condition(self, condition: ConditionBase) -> None: ...


class IPlayer(Protocol):
    active_card: ICard
    bench: List[ICard]
    opponent: "IPlayer"

    def move_active_card_to_bench(self) -> None: ...
    def move_hand_to_deck_bottom_and_shuffle(self) -> None: ...
    def draw_cards(self, n: int) -> None: ...
    def choose_opponent_bench_card(self) -> ICard: ...
    def switch_active_with(self, card: ICard) -> None: ...
    @property
    def prize_left(self) -> int: ...

class Supporter(Protocol):
    name: str

    def card_able_to_use(self, card: ICard) -> bool: ...
