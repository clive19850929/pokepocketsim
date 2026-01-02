import random
from typing import Any, List, Sequence

class RandomPolicy:
    """
    何も学習していない完全ランダム方策。
    Player 側からは policy.select_action(state, legal_actions) を呼ぶだけで使える想定。
    """
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: Any, legal_actions: List[Sequence[int]], **_) -> Sequence[int] | None:
        if not legal_actions:
            return None
        return self._rng.choice(legal_actions)
