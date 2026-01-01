from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Dict, Any, cast, Union
from .protocols import ICard
import random
import uuid
from .card_base import CardBase
from .cards_enum import Cards

if TYPE_CHECKING:
    from .player import Player
    from .attack import EnergyType
    from .condition import ConditionBase

ITEM_CLASS_DICT = {}

def register_item(cls):
    ITEM_CLASS_DICT[cls.__name__] = cls
    if hasattr(cls, "JP_NAME"):
        ITEM_CLASS_DICT[cls.JP_NAME] = cls
    return cls

class Item(CardBase):
    """
    すべての《グッズ》の基底クラス。
    - card_enum : Cards のメンバーを必須で渡す
    - card_id   : card_enum.value[0] により固定
    - name      : card_enum.value[1]（和名）を自動セット
    """
    def __init__(self, card_enum: Cards, **kwargs) -> None:
        super().__init__(card_enum, **kwargs)   # Card 側で card_id / name などを設定
        self.is_item = True           # 共通フラグ
    
    def __str__(self) -> str:
        return self.name

@register_item
class Switch(Item):
    """バトル場とベンチを入れ替えるグッズ"""
    JP_NAME = "ポケモンいれかえ"

    def __init__(self, **kwargs):
        super().__init__(Cards.SWITCH, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, player: "Player") -> bool:
        """ベンチに1体以上いれば使用可"""
        return bool(getattr(player, "bench", None) and len(player.bench) > 0)

    def use(self, player: "Player") -> bool:
        """
        multi-step ログ(substeps)対応。
        substeps 例:
        [
          { "phase": "switch.select",
            "macro": [USE_ITEM, 50002, 0, 0, 0],
            "legal_actions": [[USE_ITEM, 50002, 1, 0, 0], [USE_ITEM, 50002, 2, 0, 0], ...],
            "action_index": k,
            "action_vec": [USE_ITEM, 50002, i, 0, 0],
            "bench_snapshot": [
                {"idx": 1, "id": 10009, "name":"○○", "hp": 90}, ...
            ]
          },
          { "phase": "switch.apply",
            "macro": [USE_ITEM, 50002, i, 0, 0],
            "selected_bench_idx": i,
            "old_active_id": 10005,
            "new_active_id": 10009
          }
        ]
        使えない場合は:
          { "phase": "switch.unusable", "macro": [USE_ITEM, 50002, 0, 0, 0] }
        """
        import random

        # --- ヘルパ ---
        def _as_int(x) -> int:
            try:
                if isinstance(x, int):
                    return x
                if hasattr(x, "value"):
                    v = x.value
                    if isinstance(v, int):
                        return v
                    if isinstance(v, (tuple, list)) and v:
                        return int(v[0])
                return int(x)
            except Exception:
                return 0

        from .action import ActionType
        t_use_item = _as_int(ActionType.USE_ITEM)
        item_id    = _as_int(getattr(self, "card_enum", None))

        substeps = []

        # 使用可否チェック
        if not self.card_able_to_use(player):
            if player.print_actions:
                player.log_print("ベンチにポケモンがいないのでポケモンいれかえは使えません")
            substeps.append({
                "phase": "switch.unusable",
                "macro": [t_use_item, item_id, 0, 0, 0],
            })
            self._log_action(player, selected_bench_idx=None, old_active_id=None, new_active_id=None, substeps=substeps)
            return False

        # --- Legal 作成（1..N のインデックスで表現）---
        legal = []
        bench_snapshot = []
        for i, stack in enumerate(player.bench):
            legal.append([t_use_item, item_id, i, 0, 0])
            top = stack[-1]
            bench_snapshot.append({
                "idx": i,
                "id": _as_int(getattr(top, "card_enum", None)),
                "name": getattr(top, "name", ""),
                "hp": getattr(top, "hp", None),
            })

        # --- 選択（AI: ランダム / 人間: 入力）---
        if getattr(player, "is_bot", False):
            idx = random.randint(0, len(player.bench) - 1)
            new_active = player.bench[idx]
            player.log_print(f"AIはベンチ{idx}（{new_active[-1].name}）を選択")
        else:
            # 人間向けの画面表示（任意）
            print("ベンチのどのポケモンをバトル場に出しますか？")
            for i, stack in enumerate(player.bench):
                pokemon = stack[-1]
                attached_energies = getattr(pokemon, "attached_energies", [])
                if attached_energies:
                    energy_names = [e.name for e in attached_energies]
                    from collections import Counter
                    counts = Counter(energy_names)
                    energy_str = ", ".join(f"{name}×{count}" if count > 1 else f"{name}" for name, count in counts.items())
                    print(f"{i}: {pokemon.name} (HP: {pokemon.hp}) [{energy_str}]")
                else:
                    print(f"{i}: {pokemon.name} (HP: {pokemon.hp})")
            while True:
                try:
                    raw = input("番号を選んでください: ").strip()
                    idx = int(raw)
                    if 0 <= idx < len(player.bench):
                        new_active = player.bench[idx]
                        break
                except Exception:
                    pass
                print("無効な入力です。")
            player.log_print(f"ユーザーはベンチ{idx}（{new_active[-1].name}）を選択")

        action_vec  = [t_use_item, item_id, idx, 0, 0]
        action_index = legal.index(action_vec) if action_vec in legal else 0

        # substep: select
        substeps.append({
            "phase": "switch.select",
            "macro": [t_use_item, item_id, 0, 0, 0],
            "legal_actions": legal,
            "action_index": action_index,
            "action_vec": action_vec,
            "bench_snapshot": bench_snapshot,
        })

        # --- 入れ替え ---
        old_active = player.active_card
        player.bench.remove(new_active)
        if old_active is not None:
            player.bench.append(old_active)
        player.active_card = new_active

        new_active_id = _as_int(getattr(new_active[-1], "card_enum", None))
        old_active_id = _as_int(getattr(old_active[-1], "card_enum", None)) if old_active else None

        # 互換: 以前のログ参照向け
        player._last_switch_info = {
            "selected_bench_idx": idx,
            "new_active_id": new_active_id,
            "old_active_id": old_active_id,
        }

        # substep: apply
        substeps.append({
            "phase": "switch.apply",
            "macro": [t_use_item, item_id, idx, 0, 0],
            "selected_bench_idx": idx,
            "old_active_id": old_active_id,
            "new_active_id": new_active_id,
        })

        if old_active is not None:
            player.log_print(f"{old_active[-1].name} と {new_active[-1].name} を入れ替えた！")
        else:
            player.log_print(f"{new_active[-1].name} をバトル場に出した！")

        # まとめて extra に格納
        self._log_action(
            player,
            selected_bench_idx=idx,
            old_active_id=old_active_id,
            new_active_id=new_active_id,
            substeps=substeps,
        )
        return True

    # --- ログ集約（last_action.extra に追記） ---
    def _log_action(self, player, selected_bench_idx, old_active_id, new_active_id, substeps):
        if hasattr(player, "last_action") and player.last_action:
            extra = getattr(player.last_action, "extra", {}) if hasattr(player.last_action, "extra") else {}
            if not isinstance(extra, dict):
                extra = {}
            extra.update({
                "selected_bench_idx": selected_bench_idx,  # 1..N / None
                "old_active_id": old_active_id,
                "new_active_id": new_active_id,
                "substeps": substeps,
            })
            player.last_action.extra = extra

    def serialize(self) -> str:
        return "Switch"


@register_item
class PokemonCatcher(Item):
    """ポケモンキャッチャー: コイントスで表なら相手のベンチポケモンをバトル場に出す"""
    JP_NAME = "ポケモンキャッチャー"

    def __init__(self, **kwargs):
        super().__init__(Cards.POKEMON_CATCHER, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, player: "Player") -> bool:
        """相手のベンチが1体以上いるときのみ使用可"""
        return bool(
            getattr(player, "opponent", None)
            and getattr(player.opponent, "bench", None)
            and len(player.opponent.bench) > 0
        )

    # --- 内部ヘルパ ---
    @staticmethod
    def _as_int(x) -> int:
        try:
            if isinstance(x, int):
                return x
            if hasattr(x, "value"):
                v = x.value
                if isinstance(v, int):
                    return v
                if isinstance(v, (tuple, list)) and v:
                    return int(v[0])
            return int(x)
        except Exception:
            return 0

    def _ensure_macro(self, player, macro_vec):
        """last_action.extra に macro を事前セット（後で別処理に上書きされにくくする）"""
        if hasattr(player, "last_action") and player.last_action:
            ex = getattr(player.last_action, "extra", {})
            if not isinstance(ex, dict):
                ex = {}
            # macro が無ければ入れる（あれば保持）
            ex.setdefault("macro", macro_vec)
            player.last_action.extra = ex

    def _merge_extra(self, player, **kwargs):
        """last_action.extra にキーをマージ（既存は温存しつつ更新）"""
        if hasattr(player, "last_action") and player.last_action:
            ex = getattr(player.last_action, "extra", {})
            if not isinstance(ex, dict):
                ex = {}
            ex.update({k: v for k, v in kwargs.items()})
            player.last_action.extra = ex

    def use(self, player: "Player") -> bool:
        """
        UltraBall / CrushingHammer / Pokegear と同じ multi-step ログ仕様に対応。
        substeps 例:
        [
        { "phase": "catcher.coin",   "macro": [USE_ITEM, 50002, 0, 0, 0], "coin": "heads" },
        { "phase": "catcher.select", "macro": [USE_ITEM, 50002, 0, 0, 0],
            "legal_actions": [[USE_ITEM, 50002, 1, 0, 0], [USE_ITEM, 50002, 2, 0, 0], ...],
            "action_index": k, "action_vec": [USE_ITEM, 50002, i, 0, 0],
            "bench_snapshot": [ {"idx":1,"id":10005,"name":"○○","hp":230}, ... ]
        },
        { "phase": "catcher.apply",  "macro": [USE_ITEM, 50002, i, 0, 0],
            "old_active_id": 10009, "new_active_id": 10005
        }
        ]
        tails の場合は select/apply は出ず coin だけを残す。
        """
        import random
        from .action import ActionType

        t_use_item = self._as_int(ActionType.USE_ITEM)
        # ★ 明示的に 50002 を取得（Enum → 数値）
        item_id = int(self.card_enum.value[0])

        substeps = []

        # まず macro を先行セット（あとから extra が一部上書きされても最低限の骨格が残る）
        self._ensure_macro(player, [t_use_item, item_id, 0, 0, 0])

        # 使用可否チェック
        if not self.card_able_to_use(player):
            if player.print_actions:
                player.log_print("相手のベンチにポケモンがいないのでポケモンキャッチャーは使えません")
            substeps.append({
                "phase": "catcher.unusable",
                "macro": [t_use_item, item_id, 0, 0, 0]
            })
            # ここで coin="none" を含めて確定的に書き込む
            self._merge_extra(
                player,
                coin="none",
                selected_bench_idx=None,
                old_active_id=None,
                new_active_id=None,
                substeps=substeps,
            )
            return False

        # --- コイントス ---
        coin = random.choice(['heads', 'tails'])
        coin_str = "heads" if coin == "heads" else "tails"
        if player.print_actions:
            player.log_print(f"ポケモンキャッチャーのコイントス: {'表' if coin == 'heads' else '裏'}")

        # substep: coin
        substeps.append({
            "phase": "catcher.coin",
            "macro": [t_use_item, item_id, 0, 0, 0],
            "coin": coin_str
        })

        # 裏なら効果なし（使った事実は True を返して残す）
        if coin == 'tails':
            if player.print_actions:
                player.log_print("コイントスは裏。効果なし。")
            # tails の結果を確定的に書き込む
            self._merge_extra(
                player,
                coin="tails",
                selected_bench_idx=None,
                old_active_id=None,
                new_active_id=None,
                substeps=substeps,
            )
            return True

        # --- 表：相手のベンチから 1 体選ぶ ---
        opp = player.opponent
        bench = getattr(opp, "bench", [])
        # legal_actions: [USE_ITEM, item_id, bench_idx(1..N), 0, 0]
        legal = []
        bench_snapshot = []
        for i, stack in enumerate(bench):
            legal.append([t_use_item, item_id, i, 0, 0])
            top = stack[-1]
            bench_snapshot.append({
                "idx": i,
                "id": self._as_int(getattr(top, "card_enum", None)),
                "name": getattr(top, "name", ""),
                "hp": getattr(top, "hp", None)
            })

        # 実際の選択
        if getattr(player, "is_bot", False):
            idx = random.randint(0, len(bench) - 1)
            target_stack = bench[idx]
            if player.print_actions:
                player.log_print(f"AIは相手のベンチ{idx}（{target_stack[-1].name}）を選択")
        else:
            print("相手のベンチのどのポケモンをバトル場に出しますか？")
            for i, stack in enumerate(bench):
                pokemon = stack[-1]
                attached_energies = getattr(pokemon, "attached_energies", [])
                if attached_energies:
                    energy_names = [e.name for e in attached_energies]
                    from collections import Counter
                    counts = Counter(energy_names)
                    energy_str = ", ".join(f"{name}×{count}" if count > 1 else f"{name}" for name, count in counts.items())
                    print(f"{i}: {pokemon.name} (HP: {pokemon.hp}) [{energy_str}]")
                else:
                    print(f"{i}: {pokemon.name} (HP: {pokemon.hp})")
            while True:
                try:
                    raw = input("番号を選んでください: ").strip()
                    idx0 = int(raw)
                    if 0 <= idx0 < len(bench):
                        target_stack = bench[idx]
                        break
                except Exception:
                    pass
                print("無効な入力です。")
            player.log_print(f"ユーザーは相手のベンチ{idx}（{target_stack[-1].name}）を選択")

        action_vec  = [t_use_item, item_id, idx, 0, 0]
        action_index = legal.index(action_vec) if action_vec in legal else 0

        # substep: select
        substeps.append({
            "phase": "catcher.select",
            "macro": [t_use_item, item_id, 0, 0, 0],
            "legal_actions": legal,
            "action_index": action_index,
            "action_vec": action_vec,
        })

        # --- 相手のバトル場と選択ベンチを入れ替え ---
        old_active = opp.active_card
        opp.bench.remove(target_stack)
        if old_active is not None:
            opp.bench.append(old_active)
        opp.active_card = target_stack

        new_active_id = self._as_int(getattr(target_stack[-1], "card_enum", None))
        old_active_id = self._as_int(getattr(old_active[-1], "card_enum", None)) if old_active else None

        # substep: apply（実行結果）
        substeps.append({
            "phase": "catcher.apply",
            "macro": [t_use_item, item_id, idx, 0, 0],
            "selected_bench_idx": idx,
            "old_active_id": old_active_id,
            "new_active_id": new_active_id
        })

        # 互換: 以前のログ併用のため（任意）
        player._last_switch_info = {
            "selected_bench_idx": idx,
            "new_active_id": new_active_id,
            "old_active_id": old_active_id,
            "coin": "heads"
        }

        if old_active is not None:
            player.log_print(f"相手の {old_active[-1].name} と {target_stack[-1].name} を入れ替えた！")
        else:
            player.log_print(f"相手の {target_stack[-1].name} をバトル場に出した！")

        # まとめて extra に格納（最後にもう一度確実に書き込む）
        self._merge_extra(
            player,
            coin="heads",
            selected_bench_idx=idx,
            old_active_id=old_active_id,
            new_active_id=new_active_id,
            substeps=substeps,
        )
        return True

    def serialize(self) -> str:
        return "PokemonCatcher"



@register_item
class Pokegear3_0(Item):
    """ポケギア3.0: 山札の上から7枚を見て、その中からサポートカード1枚を手札に加える（選ばないことも可能）"""
    JP_NAME = "ポケギア3.0"

    def __init__(self, **kwargs):
        super().__init__(Cards.POKEGEAR3_0, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, player: "Player") -> bool:
        # 山札が1枚以上あれば使用可
        return len(player.deck.cards) > 0

    def use(self, player: "Player") -> bool:
        """
        UltraBall / CrushingHammer と同じ multi-step ログ仕様に対応。
        substeps の形:
        [
        { "phase": "pokegear.look",
            "macro": [USE_ITEM, 50007, 0, 0, 0],
            "looked_ids": [・・最大7枚のenum id・・],
            "supporter_ids": [・・その中のサポートのenum id群（重複あり）・・],
            "supporter_unique": [・・重複除去・・]
        },
        { "phase": "pokegear.select",
            "macro": [USE_ITEM, 50007, 0, 0, 0],
            "legal_actions": [[USE_ITEM, 50007, 0, 0, sid], ..., [USE_ITEM, 50007, 0, 0, 0]],  # 最後は「選ばない」
            "legal_unique":  [[...], ...],           # 重複なし（解析用：任意）
            "unique_to_all": [[0,3], [1], ...],      # ユニーク -> 元の写像（解析用：任意）
            "action_index": k,                       # legal_actions 上のインデックス
            "unique_index": u,                       # legal_unique 上のインデックス
            "action_vec": [USE_ITEM, 50007, 0, 0, sid_or_0]
        }
        ]
        最終的に last_action.extra には、{ "selected_id": sid_or_0, "substeps": [...] } を格納する。
        """
        import random

        # --- ヘルパ ---
        def _as_int(x) -> int:
            try:
                if isinstance(x, int):
                    return x
                if hasattr(x, "value"):
                    v = x.value
                    if isinstance(v, int):
                        return v
                    if isinstance(v, (tuple, list)) and v:
                        return int(v[0])
                return int(x)
            except Exception:
                return 0

        from .action import ActionType
        t_use_item = _as_int(ActionType.USE_ITEM)
        item_id = _as_int(getattr(self, "card_enum", None))

        # substeps 蓄積
        substeps = []

        # 1) 見る枚数を決定
        cards_to_look = min(7, len(player.deck.cards))
        if cards_to_look == 0:
            # 山札が空で使用不可
            if player.print_actions:
                player.log_print("山札が空なのでポケギア3.0は使えません")
            # substep を残して終了
            substeps.append({
                "phase": "pokegear.look",
                "macro": [t_use_item, item_id, 0, 0, 0],
                "looked_ids": [],
                "supporter_ids": [],
                "supporter_unique": []
            })
            self._log_action(player, selected_id=0, substeps=substeps)
            return False

        looked_cards = player.deck.cards[:cards_to_look]
        looked_ids = []
        for c in looked_cards:
            looked_ids.append(_as_int(getattr(c, "card_enum", None)))

        supporter_cards = [c for c in looked_cards if getattr(c, "is_supporter", False)]
        supporter_ids = [_as_int(getattr(c, "card_enum", None)) for c in supporter_cards]

        # 重複を解析用にユニーク化
        sup_seen = {}
        supporter_unique = []
        for sid in supporter_ids:
            if sid not in sup_seen:
                sup_seen[sid] = True
                supporter_unique.append(sid)

        # substep: look
        substeps.append({
            "phase": "pokegear.look",
            "macro": [t_use_item, item_id, 0, 0, 0],
            "looked_ids": looked_ids,              # 上から見たカードID列（最大7）
            "supporter_ids": supporter_ids,        # 見えたサポートID列（重複あり）
            "supporter_unique": supporter_unique   # 重複なし（解析用）
        })

        # 2) 選択（任意。0=選ばない）
        # legal_actions 構築：サポート候補 + 「選ばない(0)」
        legal = []
        for sid in supporter_ids:
            legal.append([t_use_item, item_id, 0, 0, sid])
        # 選択しない
        legal.append([t_use_item, item_id, 0, 0, 0])

        # 重複なし版（解析用：任意）
        seen = {}
        legal_unique = []
        unique_to_all = []
        for i, vec in enumerate(legal):
            key = tuple(vec)
            if key not in seen:
                seen[key] = len(legal_unique)
                legal_unique.append(vec)
                unique_to_all.append([i])
            else:
                unique_to_all[seen[key]].append(i)

        # 実際の選択
        chosen_card = None
        selected_id = 0  # 既定は選ばない
        if not supporter_cards:
            # 候補なし → 自動で「選ばない」
            action_vec = [t_use_item, item_id, 0, 0, 0]
            action_index = legal.index(action_vec) if action_vec in legal else len(legal) - 1
            unique_index = seen.get(tuple(action_vec), 0)
            if player.print_actions:
                player.log_print("見たカードにサポートカードがありませんでした")
        else:
            if player.is_bot:
                # AI: 候補からランダムで 1 枚（または稀に選ばない）を選ぶ
                # ここでは単純に「候補からランダム選択」: 選ばない分岐は入れずにおく（必要なら確率を足す）
                chosen_card = random.choice(supporter_cards)
            else:
                print("見たカードからサポートカードを1枚選んでください（選ばない場合は 0 または空欄）:")
                for i, c in enumerate(supporter_cards, 1):
                    print(f"{i}: {c.name} (ID: {_as_int(getattr(c, 'card_enum', None))})")
                print("0: 選択しない")
                while True:
                    idx = input("番号またはカードIDで入力: ").strip()
                    if idx == "0" or idx == "":
                        chosen_card = None
                        break
                    try:
                        # まずインデックス（1始まり）
                        int_idx = int(idx)
                        if 1 <= int_idx <= len(supporter_cards):
                            chosen_card = supporter_cards[int_idx - 1]
                            break
                        # それ以外はカードIDとして探索
                        for c in supporter_cards:
                            if int_idx == _as_int(getattr(c, "card_enum", None)):
                                chosen_card = c
                                break
                        if chosen_card:
                            break
                    except (ValueError, IndexError):
                        # 文字列としてカードID比較
                        for c in supporter_cards:
                            if idx == str(_as_int(getattr(c, "card_enum", None))):
                                chosen_card = c
                                break
                        if chosen_card:
                            break
                    print("無効な入力です。")

            if chosen_card is not None:
                selected_id = _as_int(getattr(chosen_card, "card_enum", None))
                action_vec = [t_use_item, item_id, 0, 0, selected_id]
            else:
                action_vec = [t_use_item, item_id, 0, 0, 0]

            action_index = legal.index(action_vec) if action_vec in legal else len(legal) - 1
            unique_index = seen.get(tuple(action_vec), 0)

        # substep: select
        substeps.append({
            "phase": "pokegear.select",
            "macro": [t_use_item, item_id, 0, 0, 0],
            "legal_actions": legal,          # 重複あり
            "legal_unique": legal_unique,    # 重複なし（解析用：任意）
            "unique_to_all": unique_to_all,  # ユニーク -> 元の写像（解析用：任意）
            "action_index": action_index,    # legal_actions 上のインデックス
            "unique_index": unique_index,    # legal_unique 上のインデックス
            "action_vec": action_vec         # 実際に選ばれた 5 要素
        })

        # 3) 効果の適用
        if selected_id != 0 and chosen_card is not None:
            # 山札からそのカードを取り、手札へ
            if chosen_card in player.deck.cards:
                player.deck.cards.remove(chosen_card)
            player.hand.append(chosen_card)
            if player.print_actions:
                player.log_print(f"{chosen_card.name} を手札に加えました")
        else:
            if player.print_actions:
                player.log_print("サポートカードを選びませんでした")

        # 山札をシャッフル
        random.shuffle(player.deck.cards)
        if player.print_actions:
            player.log_print("山札をシャッフルしました")

        # 4) 最終 extra を集約
        self._log_action(player, selected_id=selected_id, substeps=substeps)
        return True

    def _log_action(self, player, selected_id: int, substeps):
        """
        UltraBall / CrushingHammer と同じ思想で last_action.extra に追記。
        """
        if hasattr(player, "last_action") and player.last_action:
            extra = getattr(player.last_action, "extra", {}) if hasattr(player.last_action, "extra") else {}
            if not isinstance(extra, dict):
                extra = {}
            extra["selected_id"] = selected_id       # 手札に加えたサポート（0=選ばない）
            extra["substeps"] = substeps             # multi-step 記録
            player.last_action.extra = extra



@register_item
class CrushingHammer(Item):
    """クラッシュハンマー: コインが表なら相手の場のポケモン1体からエネルギー1枚トラッシュ"""
    JP_NAME = "クラッシュハンマー"

    def __init__(self, **kwargs):
        super().__init__(Cards.CRUSHING_HAMMER, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, player: "Player") -> bool:
        if not player.opponent:
            return False
        for stack in self._get_all_opponent_pokemon(player):
            energies = getattr(stack[-1], "attached_energies", [])
            if energies:
                return True
        return False

    def use(self, player: "Player") -> bool:
        """
        出力仕様（UltraBall と同様の multi-step ログを出す）:
        - player.last_action.extra['substeps'] に、各ステップのサブログを配列で格納
          * 例:
            [
            { "phase": "crushinghammer.coin",
                "macro": [USE_ITEM, 50002, 0, 0, 0],
                "coin": "表", "coin_success": 1
            },
            { "phase": "crushinghammer.select",
                "macro": [USE_ITEM, 50002, 0, 0, 0],
                "legal_actions": [[USE_ITEM, 50002, t_idx, energy_id, 0], ...],   # 重複あり
                "legal_unique":  [[...], ...],    # 重複除去（任意、解析用）
                "unique_to_all": [[0, 3], [1], ...],
                "action_index": k,                 # legal_actions 上のインデックス
                "unique_index": u,                 # legal_unique 上のインデックス（任意）
                "action_vec":   [USE_ITEM, 50002, t_idx, energy_id, 0]
            }
            ]
        - 失敗（コイン裏/対象不在）でも coin ステップを残し、select ステップは出さない。
        - 最終結果として player.last_action.extra に
        { "target_index": t_idx or 99, "energy_id": energy_enum or 9999, "coin": "表/裏", "coin_success": 0/1 }
        を格納する。
        """
        # --- 小ヘルパ ---
        def _as_int(x) -> int:
            try:
                if isinstance(x, int):
                    return x
                if hasattr(x, "value"):
                    v = x.value
                    if isinstance(v, int):
                        return v
                    if isinstance(v, (tuple, list)) and v:
                        return int(v[0])
                return int(x)
            except Exception:
                return 0

        from .action import ActionType  # USE_ITEM を数値化して使う
        t_use_item = _as_int(ActionType.USE_ITEM)
        item_id = _as_int(getattr(self, "card_enum", None))

        # substeps 蓄積用
        substeps = []

        # 1) 事前チェック
        if not self.card_able_to_use(player):
            # 対象無し：簡易 substep を残して終了
            substeps.append({
                "phase": "crushinghammer.coin",
                "macro": [t_use_item, item_id, 0, 0, 0],
                "coin": "スキップ",
                "coin_success": 0
            })
            self._log_action(player, target_index=99, energy_id=9999, substeps=substeps, coin="スキップ", success=0)
            if player.print_actions:
                player.log_print("相手の場のポケモンにエネルギーが付いていないのでクラッシュハンマーは使えません")
            return False

        # 2) コイントス substep
        coin = random.choice(["表", "裏"])
        coin_success = 1 if coin == "表" else 0
        substeps.append({
            "phase": "crushinghammer.coin",
            "macro": [t_use_item, item_id, 0, 0, 0],
            "coin": coin,
            "coin_success": coin_success
        })

        if not coin_success:
            # 失敗：ここで確定
            if player.print_actions:
                player.log_print("効果は発動しませんでした")
            self._log_action(player, target_index=99, energy_id=9999, substeps=substeps, coin=coin, success=0)
            return False

        # 3) 対象列挙（相手のバトル場→ベンチ）
        all_targets = []
        # t_idx は「相手の場の並び index」(0:バトル場, 1..:ベンチ)
        for t_idx, stack in enumerate(self._get_all_opponent_pokemon(player)):
            energies = getattr(stack[-1], "attached_energies", [])
            for e_idx, e in enumerate(energies):
                energy_enum = _as_int(getattr(e, "card_enum", None))
                all_targets.append((t_idx, e_idx, stack, e, energy_enum))

        if not all_targets:
            # 念のため（理論上は card_able_to_use True なので到達しない）
            if player.print_actions:
                player.log_print("相手の場のどのポケモンにもエネルギーがありません")
            self._log_action(player, target_index=99, energy_id=9999, substeps=substeps, coin=coin, success=1)
            return False

        # 4) substep: select の legal_actions を 5 要素で列挙
        legal = []
        for (t_idx, e_idx, stack, e, energy_enum) in all_targets:
            legal.append([t_use_item, item_id, t_idx, energy_enum, 0])

        # 重複を残したままでも学習可能だが、解析用途にユニーク配列も併記
        seen = {}
        legal_unique = []
        unique_to_all = []
        for i, vec in enumerate(legal):
            key = tuple(vec)
            if key not in seen:
                seen[key] = len(legal_unique)
                legal_unique.append(vec)
                unique_to_all.append([i])
            else:
                unique_to_all[seen[key]].append(i)

        # 5) 選択（人間/AI）
        if player.is_bot:
            choice = random.randrange(len(all_targets))
            t_idx, e_idx, target_stack, energy_to_remove, energy_enum = all_targets[choice]
            if player.print_actions:
                player.log_print(
                    f"AIは index {t_idx} の {target_stack[-1].name} のエネルギー{e_idx}({energy_to_remove.name})を選択しました"
                )
        else:
            # 人間向けプロンプト
            print("トラッシュするエネルギーを選んでください（番号）:")
            for i, (t_idx, e_idx, stack, energy, _eid) in enumerate(all_targets):
                print(f"{i}: ポケモンindex {t_idx}（{stack[-1].name}, HP:{stack[-1].hp}）のエネルギーindex {e_idx}（{energy.name}）")
            while True:
                raw = input("番号: ").strip()
                if raw.isdigit() and 0 <= int(raw) < len(all_targets):
                    choice = int(raw)
                    break
                print("無効な入力です。")
            t_idx, e_idx, target_stack, energy_to_remove, energy_enum = all_targets[choice]
            if player.print_actions:
                player.log_print(
                    f"ユーザーは index {t_idx} の {target_stack[-1].name} のエネルギー{e_idx}({energy_to_remove.name})を選択しました"
                )

        action_vec = [t_use_item, item_id, t_idx, energy_enum, 0]
        action_index = legal.index(action_vec)
        unique_index = seen.get(tuple(action_vec), 0)

        # 6) substep: select を記録
        substeps.append({
            "phase": "crushinghammer.select",
            "macro": [t_use_item, item_id, 0, 0, 0],
            "legal_actions": legal,         # 重複あり
            "legal_unique": legal_unique,   # 重複なし（解析用：任意）
            "unique_to_all": unique_to_all, # ユニーク→元の写像（解析用：任意）
            "action_index": action_index,   # legal_actions のインデックス
            "unique_index": unique_index,   # legal_unique のインデックス
            "action_vec": action_vec        # 実際に選ばれた 5 要素
        })

        # 7) 効果の実行（エネルギーをトラッシュ）
        target_stack[-1].attached_energies.pop(e_idx)
        if hasattr(player.opponent, 'discard_pile'):
            player.opponent.discard_pile.append(energy_to_remove)
        if player.print_actions:
            player.log_print(f"index {t_idx} の {target_stack[-1].name} から {energy_to_remove.name} をトラッシュしました")

        # 8) まとめてログ（最終形）
        self._log_action(
            player,
            target_index=t_idx,
            energy_id=energy_enum,
            substeps=substeps,
            coin=coin,
            success=1
        )
        return True

    def _get_all_opponent_pokemon(self, player):
        stacks = []
        if player.opponent:
            if player.opponent.active_card:
                stacks.append(player.opponent.active_card)
            for stack in player.opponent.bench:
                stacks.append(stack)
        return stacks

    def _log_action(self, player, target_index=None, energy_id=None, substeps=None, coin=None, success=None):
        """
        UltraBall と同じ思想で、最後に last_action.extra に必要情報を集約する。
        - substeps: 上記 use() で構築した multi-step 記録（任意）
        - coin / success: コイントス結果の可視化（任意）
        - target_index / energy_id: 最終決定パラメータ
        ※ 互換性維持のため、target_index/energy_id だけの呼び出しでも動く。
        """
        if hasattr(player, "last_action") and player.last_action:
            extra = getattr(player.last_action, "extra", {}) if hasattr(player.last_action, "extra") else {}
            if not isinstance(extra, dict):
                extra = {}
            if substeps is not None:
                extra["substeps"] = substeps
            if target_index is not None:
                extra["target_index"] = target_index
            if energy_id is not None:
                extra["energy_id"] = energy_id
            if coin is not None:
                extra["coin"] = coin
            if success is not None:
                # 0/1 に正規化
                extra["coin_success"] = 1 if bool(success) else 0
            player.last_action.extra = extra



@register_item
class UltraBall(Item):
    """ハイパーボール: 手札2枚トラッシュで山札からポケモン1枚サーチ"""
    JP_NAME = "ハイパーボール"
    def __init__(self, **kwargs):
        super().__init__(Cards.ULTRA_BALL, **kwargs)
    
    def __str__(self):
        return self.name
    
    def card_able_to_use(self, player: "Player") -> bool:
        # ハイパーボール以外に手札が2枚以上必要
        other_cards = [c for c in player.hand if c != self]
        return len(other_cards) >= 2
    
    def use(self, player: "Player") -> bool:
        # --- substeps 蓄積バッファを初期化（なければ）---
        if not hasattr(player, "_pending_substeps"):
            player._pending_substeps = []

        # --- 大枠（macro）: [USE_ITEM=3, ULTRA_BALL=50001, 0, 0, 0] ---
        try:
            from .action import ActionType
            macro_vec = [int(ActionType.USE_ITEM), int(self.card_enum.value[0]), 0, 0, 0]
        except Exception:
            macro_vec = [3, 50001, 0, 0, 0]

        other_cards = [c for c in player.hand if c != self]
        if len(other_cards) < 2:
            print("ハイパーボール：トラッシュできる手札が2枚必要です")
            return False
    
        # --- 1) トラッシュ1枚目選択 ---
        if player.is_bot:
            legal1 = []
            for c in other_cards:
                try:
                    cid = int(c.card_enum.value[0])
                except Exception:
                    cid = 0
                legal1.append([macro_vec[0], macro_vec[1], cid, 0, 0])
            # ★ サブステップ開始：合法手提示直前
            player.begin_substep("ultraball.trash1", legal1)
            trash1 = random.choice(other_cards)
            action_index1 = other_cards.index(trash1)
            action_vec1   = legal1[action_index1]
            # ★ サブステップ終了：選択直後
            player.end_substep(action_vec1, action_index1)
        else:
            print("トラッシュするカード2枚を選んでください (番号またはカードID):")
            for idx, c in enumerate(other_cards, 1):
                print(f"[{idx}] {c.id}: {c}")
            def select_card(prompt, exclude=None):
                while True:
                    raw = input(prompt).strip()
                    try:
                        val = int(raw)
                    except ValueError:
                        print("数値を入力してください。")
                        continue
                    card = None
                    if 1 <= val <= len(other_cards): 
                        card = other_cards[val - 1]
                    else:
                        card = next((x for x in other_cards if x.id == val), None)
                    if card and (exclude is None or card != exclude):
                        return card
                    print("無効な指定です。")
            legal1 = []
            for c in other_cards:
                try:
                    cid = int(c.card_enum.value[0])
                except Exception:
                    cid = 0
                legal1.append([macro_vec[0], macro_vec[1], cid, 0, 0])
            # ★ サブステップ開始：合法手提示直前
            player.begin_substep("ultraball.trash1", legal1)
            trash1 = select_card("1枚目: ")
            action_index1 = other_cards.index(trash1)
            action_vec1   = legal1[action_index1]
            # ★ サブステップ終了：選択直後
            player.end_substep(action_vec1, action_index1)

        # --- 2) トラッシュ2枚目選択 ---
        remain_cards = [c for c in other_cards if c != trash1]
        if player.is_bot:
            trash2 = random.choice(remain_cards)
            legal2 = []
            try:
                t1 = int(trash1.card_enum.value[0])
            except Exception:
                t1 = 0
            for c in remain_cards:
                try:
                    cid = int(c.card_enum.value[0])
                except Exception:
                    cid = 0
                legal2.append([macro_vec[0], macro_vec[1], t1, cid, 0])
            # ★ サブステップ開始
            player.begin_substep("ultraball.trash2", legal2)
            action_index2 = remain_cards.index(trash2)
            action_vec2   = legal2[action_index2]
            # ★ サブステップ終了
            player.end_substep(action_vec2, action_index2)
        else:
            trash2 = select_card("2枚目: ", exclude=trash1)
            legal2 = []
            try:
                t1 = int(trash1.card_enum.value[0])
            except Exception:
                t1 = 0
            for c in remain_cards:
                try:
                    cid = int(c.card_enum.value[0])
                except Exception:
                    cid = 0
                legal2.append([macro_vec[0], macro_vec[1], t1, cid, 0])
            # ★ サブステップ開始
            player.begin_substep("ultraball.trash2", legal2)
            action_index2 = remain_cards.index(trash2)
            action_vec2   = legal2[action_index2]
            # ★ サブステップ終了
            player.end_substep(action_vec2, action_index2)

        # --- 実処理：トラッシュ実行 ---
        to_trash = [trash1, trash2]
        for c in to_trash:
            player.hand.remove(c)
            player.discard_pile.append(c)
            print(f"{c}をトラッシュしました")

        # 2. 山札からポケモンを1枚選ぶ
        pokemon_cards = [c for c in player.deck.cards if hasattr(c, "hp") and c.hp is not None]
        if not pokemon_cards:
            print("山札にポケモンがいません")
            selected_pokemon = None
            # legal_select は 0 のみ
            try:
                t1 = int(trash1.card_enum.value[0])
            except Exception:
                t1 = 0
            try:
                t2 = int(trash2.card_enum.value[0])
            except Exception:
                t2 = 0
            legal3 = [[macro_vec[0], macro_vec[1], t1, t2, 0]]
            action_index3 = 0
            action_vec3   = legal3[0]
        else:
            if player.is_bot:
                selected_pokemon = random.choice(pokemon_cards)
                # legal3 は pokemon_cards の順序で生成
                legal3 = []
                try:
                    t1 = int(trash1.card_enum.value[0])
                except Exception:
                    t1 = 0
                try:
                    t2 = int(trash2.card_enum.value[0])
                except Exception:
                    t2 = 0
                for c in pokemon_cards:
                    try:
                        pid = int(c.card_enum.value[0])
                    except Exception:
                        pid = 0
                    legal3.append([macro_vec[0], macro_vec[1], t1, t2, pid])
                action_index3 = pokemon_cards.index(selected_pokemon)
                action_vec3   = legal3[action_index3]
            else:
                print("山札から加えるポケモンを選んでください (番号またはカードID, 0でスキップ):")
                shown = []
                for idx, c in enumerate(pokemon_cards, 1):
                    if c.id not in [x.id for x in shown]:
                        shown.append(c)
                        print(f"[{idx}] {c.id}: {c}")
                while True:
                    raw = input("カード番号またはID, 0でスキップ: ").strip()
                    try:
                        val = int(raw)
                    except ValueError:
                        print("数値を入力してください。")
                        continue
                    if val == 0:
                        selected_pokemon = None
                        break
                    if 1 <= val <= len(shown):
                        selected_pokemon = shown[val-1]
                        break
                    else:
                        selected_pokemon = next((x for x in shown if x.id == val), None)
                        if selected_pokemon:
                            break
                    print("無効な指定です。")

                # legal3 は UI で提示した shown の順序で生成
                legal3 = []
                try:
                    t1 = int(trash1.card_enum.value[0])
                except Exception:
                    t1 = 0
                try:
                    t2 = int(trash2.card_enum.value[0])
                except Exception:
                    t2 = 0
                if shown:
                    for c in shown:
                        try:
                            pid = int(c.card_enum.value[0])
                        except Exception:
                            pid = 0
                        legal3.append([macro_vec[0], macro_vec[1], t1, t2, pid])
                else:
                    # shown が空のときもフォールバックで 0 を 1 件
                    legal3 = [[macro_vec[0], macro_vec[1], t1, t2, 0]]

                if selected_pokemon is None:
                    # スキップ時は 0 を選択肢として最後に追加して選択
                    if legal3 and legal3[-1][4] != 0:
                        legal3.append([macro_vec[0], macro_vec[1], t1, t2, 0])
                    action_index3 = len(legal3) - 1
                    action_vec3   = legal3[action_index3]
                else:
                    # shown の中での index
                    if shown:
                        action_index3 = shown.index(selected_pokemon)
                        action_vec3   = legal3[action_index3]
                    else:
                        action_index3 = 0
                        action_vec3   = legal3[0]

        # substep#3 を記録
        player.begin_substep("ultraball.select", legal3)
        player.end_substep(action_vec3, action_index3)
        
        if selected_pokemon:
            player.deck.cards.remove(selected_pokemon)
            player.hand.append(selected_pokemon)
            print(f"{selected_pokemon}を手札に加えました")
            selected_id = int(selected_pokemon.card_enum.value[0])
        else:
            print("何も加えませんでした")
            selected_id = 0
        
        random.shuffle(player.deck.cards)
        print("山札をシャッフルしました")

        # --- 行動ログ(extra)を個別IDで出力 ---
        if hasattr(player, "last_action") and player.last_action:
            player.last_action.extra = {
                "trashed_id1": int(trash1.card_enum.value[0]),
                "trashed_id2": int(trash2.card_enum.value[0]),
                "selected_id": selected_id
            }
        return True

    def serialize(self) -> str:
        return "UltraBall"
