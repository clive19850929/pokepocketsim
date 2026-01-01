import uuid
from typing import Any

# 基底クラスだけ残す
class CardBase:
    def __init__(self, card_enum=None, **kwargs):
        self.id = uuid.uuid4()
        self.card_enum = card_enum
        self.name = card_enum.value[1]
        # 共通プロパティ初期化（必要に応じて追加）

    def serialize(self) -> dict:
        return {
            "id": str(self.id),
            "card_enum": str(self.card_enum),
            "name": self.name,
            # 必要に応じて他の共通プロパティも追記OK
        }