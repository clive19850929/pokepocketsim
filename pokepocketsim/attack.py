# Dynamically created attack methods and their source code:
from __future__ import annotations
import math
from enum import Enum
from functools import wraps
from .condition import Condition
from .attack_common import EnergyType, ATTACKS
from .damage_utils import add_damage_counters, remove_damage_counters

from typing import (
    TYPE_CHECKING,
    Dict,
    Any,
    Callable,
    TypeVar,
    cast,
    Optional,
    List,
    Union,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .player import Player
    from .card import Card

# Type for attack functions
AttackFunc = Callable[..., None]
F = TypeVar("F", bound=AttackFunc)


def apply_damage(func: F) -> F:
    """ワザ実行後に『通常ダメージ』または『ダメカンを直接のせる』を処理するデコレータ"""
    @wraps(func)
    def wrapper(player: "Player", *args: Any, **kwargs: Any) -> None:
        attack_name = func.__name__

        # 攻撃側のアクティブが存在しない場合は何もしない
        if player.active_card is None:
            return None

        # まずワザ本体を実行
        func(player, *args, **kwargs)

        # 防御側がいない場合は以降不要
        if player.opponent is None or player.opponent.active_card is None:
            return None

        # --- 攻撃側・防御側（進化スタック対応：一番上のカード） --------------------------
        attacker_stack = player.active_card
        defender_stack = player.opponent.active_card
        attacker_top = attacker_stack[-1] if isinstance(attacker_stack, list) else attacker_stack
        defender_top = defender_stack[-1] if isinstance(defender_stack, list) else defender_stack

        # 攻撃情報を取得（ATTACKS に無い = ダメージもダメカンも与えないワザ）
        attack_info = ATTACKS.get(attack_name, {})
        if not attack_info:
            return None

        # -------------------------------------------------------------
        # ① ダメカンを直接のせるワザ  ─────────────────────────────
        #    （弱点・抵抗力・追加修正の影響を受けないのが公式ルール）
        # -------------------------------------------------------------
        if "counters" in attack_info:
            cnt = attack_info["counters"]
            old_hp = defender_top.hp
            add_damage_counters(defender_top, cnt)

            # 表示・ログ
            if player.print_actions:
                player.log_print(f"{defender_top.name} にダメカンを {cnt} 個のせた！")
                if old_hp is not None and defender_top.hp is not None:
                    player.log_print(f"{defender_top.name} のHP: {old_hp} → {defender_top.hp}")

            # きぜつ判定
            if defender_top.hp is not None and defender_top.hp <= 0:
                player.opponent.was_knocked_out_by_attack_last_turn = True
            return None  # ダメージ計算はスキップ

        # -------------------------------------------------------------
        # ② 通常ダメージワザ  ───────────────────────────────────
        # -------------------------------------------------------------
        if "damage" in attack_info:
            damage: int = attack_info["damage"]

            # --- かたきうち専用ボーナス -----------------
            if attack_name == "katakiuti" and getattr(player, "katakiuti_bonus_80", False):
                damage += 80                # ここで加算 → 後の弱点計算に含まれる
                player.katakiuti_bonus_80 = False   # 使ったらリセット
            # ------------------------------------------

            # ── 弱点・抵抗力補正
            attacker_type = str(attacker_top.type)
            damage = apply_type_effects(damage, attacker_type, defender_top)

            # ── 条件による修正
            attacker_conditions: List[str] = getattr(attacker_top, "conditions", [])
            defender_conditions: List[str] = getattr(defender_top, "conditions", [])

            if "Plus10DamageDealed" in attacker_conditions:
                damage += 10
            if "Plus30DamageDealed" in attacker_conditions:
                damage += 30
            if "Minus20DamageReceived" in defender_conditions:
                damage = max(0, damage - 20)

            # ── Player に直近ダメージを記録（任意）
            if hasattr(player, "_last_attack_damage"):
                player._last_attack_damage = damage

            # ── ダメカンへ変換して付与
            counters = math.ceil(damage / 10)          # 10 で割り切れない場合は繰り上げ
            old_hp = defender_top.hp
            add_damage_counters(defender_top, counters)

            # ── ログ出力
            if player.print_actions:
                player.log_print(f"{defender_top.name} に {damage} のダメージ！！")
                if old_hp is not None and defender_top.hp is not None:
                    player.log_print(f"{defender_top.name} のHP: {old_hp} → {defender_top.hp}")

            # ── きぜつ判定
            if defender_top.hp is not None and defender_top.hp <= 0:
                player.opponent.was_knocked_out_by_attack_last_turn = True

        # 「damage」も「counters」もどちらも含まれない場合は何もしない
        return None

    return cast(F, wrapper)


def apply_type_effects(damage: int, attacker_type: str, defender_card) -> int:
    """
    defender_cardのweakness属性を参照し、攻撃側タイプと一致すれば2倍にする。
    """
    attacker_type = str(attacker_type).lower()
    if hasattr(defender_card, "weakness") and defender_card.weakness:
        defender_weakness = defender_card.weakness.value if hasattr(defender_card.weakness, "value") else str(defender_card.weakness)
        if attacker_type == str(defender_weakness).lower():
            return int(damage * 2)
    return damage


class Attack:
    """Class containing attack methods and utilities."""

    @staticmethod
    def can_use_attack(card: "Card", attack_func: AttackFunc) -> bool:
        attack_name = attack_func.__name__
        if attack_name not in ATTACKS:
            return False

        attack_data = ATTACKS[attack_name]
        energy_cost = attack_data.get("energy", {})

        attached = getattr(card, "attached_energies", [])
        # 1. 各エネルギーのカウント（typeごと） Enum/strどちらも対応
        energy_counter = {}
        for e in attached:
            t = getattr(e, "type", None)
            if isinstance(t, Enum):
                t = t.value
            elif isinstance(t, str):
                t = t.lower()
            if t:
                energy_counter[t] = energy_counter.get(t, 0) + 1

        # 2. まずタイプ指定を消費
        for etype, need in energy_cost.items():
            if isinstance(etype, Enum):
                etype_str = etype.value
            else:
                etype_str = str(etype).lower()
            if etype_str == "colorless":
                continue
            if energy_counter.get(etype_str, 0) < need:
                return False
            energy_counter[etype_str] -= need

        # 3. 残りのエネルギー合計で無色を満たす
        colorless_need = energy_cost.get("colorless", 0)
        remaining = sum(energy_counter.values())
        if colorless_need > remaining:
            return False

        return True



    @staticmethod
    def attack_repr(name: str, damage: int, energy_cost: Dict[str, int]) -> str:
        """
        Create a string representation of an attack.

        Args:
            name: Name of the attack
            damage: Damage value of the attack
            energy_cost: Dictionary of energy costs by type

        Returns:
            String representation of the attack
        """
        energy_str = ", ".join(
            f"{count} {energy_type}" for energy_type, count in energy_cost.items()
        )
        return f"{name} ({damage} damage, {energy_str})"

    # ゼクロムexのボルテージバースト
    @staticmethod
    @apply_damage
    def voltage_burst(player: "Player") -> None:
        from .card import Card
        if not player.opponent:
            return
        side_taken = 6 - player.opponent.prize_left()
        extra_damage = 50 * side_taken
        # 反動ダメージ
        # ワザ関数の冒頭でトップカードを取得
        top_active = player.active_card[-1] if isinstance(player.active_card, list) and player.active_card else player.active_card
        if top_active and isinstance(top_active, Card) and hasattr(top_active, "hp") and top_active.hp is not None:
            add_damage_counters(top_active, 3)
            if player.print_actions:
                print(f"{getattr(top_active, 'name', '')} にも反動で30ダメージ！")
        # 追加ダメージ分だけ相手に加算
        if extra_damage > 0:
            top_opp = player.opponent.active_card[-1] if isinstance(player.opponent.active_card, list) and player.opponent.active_card else player.opponent.active_card
            if top_opp and isinstance(top_opp, Card) and hasattr(top_opp, "hp") and top_opp.hp is not None:
                add_damage_counters(top_opp, extra_damage // 10)
                if player.print_actions:
                    print(f"{getattr(top_opp, 'name', '')} に追加ダメージ {extra_damage}！")

    # ゼクロムexのきりさく
    @staticmethod
    @apply_damage
    def slash(player: "Player") -> None:
        from .card import Card
        pass

    # モンメンのすいとる
    @staticmethod
    @apply_damage
    def suitoru(player: "Player") -> None:
        from .card import Card
        # バトル場の自分のカード
        top_active = player.active_card[-1] if isinstance(player.active_card, list) and player.active_card else player.active_card
        if top_active and hasattr(top_active, "hp"):
            if isinstance(top_active, Card):
                max_hp = getattr(top_active, "max_hp", None)
                if max_hp is not None:
                    remove_damage_counters(top_active, 1)
                if player.print_actions and getattr(top_active, "name", None):
                    print(f"{getattr(top_active, 'name', '')} の すいとる！ HPが10回復！")

    # テラキオのカタキウチ
    @staticmethod
    @apply_damage
    def katakiuti(player: "Player") -> None:
        from .card import Card
        """
        追加 80 ダメージは apply_damage 内でまとめて計算したいので、
        ここではフラグを立てるだけにする。
        """
        # 前の相手ターンにきぜつしたか判定
        bonus = (
            getattr(player, "was_knocked_out_by_attack_last_turn", False)
        )

        # ログだけ出しておく
        if bonus and player.print_actions:
            print("かたきうち！前の相手の番にワザのダメージできぜつしたので80ダメージ追加！")

        # 80 ダメージを apply_damage に伝えるフラグ
        player.katakiuti_bonus_80 = bonus
        player.was_knocked_out_by_attack_last_turn = False

    # テラキオのランドクラッシュ
    @staticmethod
    @apply_damage
    def randkurasshu(player: "Player") -> None:
        from .card import Card
        pass

    # エモンガのなかまをよぶ
    @staticmethod
    @apply_damage
    def nakamawoyobu(player: "Player") -> None:
        from .card import Card
        deck = player.deck.cards
        bench_space = 5 - len(player.bench)
        if bench_space <= 0:
            if player.print_actions:
                print("ベンチに空きがありません。なかまをよぶは使用できません。")
            return  # 技の効果を発動しない
        basic_candidates = [c for c in deck if getattr(c, "is_basic", False)]
        if not basic_candidates:
            if player.print_actions:
                print("山札にたねポケモンがありません。")
            return
        max_select = min(2, bench_space, len(basic_candidates))
        selected = []
        import random
        if player.is_bot:
            n = random.randint(0, max_select)
            selected = random.sample(basic_candidates, n) if n > 0 else []
        else:
            # 山札内訳を表示（ハイパーボールと同様）
            pokemon_cards_all = [c for c in deck if hasattr(c, "hp") and c.hp is not None]
            trainer_cards = [
                c for c in deck
                if getattr(c, "is_item", False)
                    or getattr(c, "is_supporter", False)
                    or getattr(c, "is_stadium", False)
                    or getattr(c, "is_tool", False)
            ]
            energy_cards = [c for c in deck if getattr(c, "is_energy", False)]
            other_cards = [
                c for c in deck
                if c not in pokemon_cards_all and c not in trainer_cards and c not in energy_cards
            ]
            def counter(cards):
                names = [str(c) for c in cards]
                from collections import Counter
                return Counter(names)
            sections = [
                ("【ポケモン】", counter(pokemon_cards_all)),
                ("【トレーナーズ】", counter(trainer_cards)),
                ("【エネルギー】", counter(energy_cards)),
                ("【その他】", counter(other_cards))
            ]
            print("あなたの山札内訳:")
            for title, counts in sections:
                if counts:
                    print(title)
                    for name, n in counts.items():
                        print(f"  {name} ×{n}")
            remaining = basic_candidates.copy()
            for i in range(max_select):
                print(f"山札からベンチに出すたねポケモンを選んでください（スキップは0）: {i+1}枚目")
                print("0: 選ばない")
                for idx, c in enumerate(remaining, 1):
                    print(f"{idx}: {c}")
                while True:
                    s = input("番号: ").strip()
                    if s == "0" or s == "":
                        break
                    try:
                        idx = int(s) - 1
                        if 0 <= idx < len(remaining):
                            selected.append(remaining[idx])
                            del remaining[idx]
                            break
                    except Exception:
                        pass
                    print("無効な入力です。0または有効な番号を入力してください。")
                if s == "0" or s == "":
                    break
        for card in selected:
            player.bench.append([card])
            deck.remove(card)
            if player.print_actions:
                print(f"{card} をベンチに出しました。")
        random.shuffle(deck)
        if player.print_actions:
            print("山札をシャッフルしました。")


    # エモンガのバチバチ
    @staticmethod
    @apply_damage
    def batibati(player: "Player") -> None:
        from .card import Card
        pass

    # シビシラスのライトニングボール
    @staticmethod
    @apply_damage
    def raitoninnguboru(player: "Player") -> None:
        from .card import Card
        pass

    # エルフーンexのエナジーギフト
    @staticmethod
    @apply_damage
    def energy_gift(player: "Player") -> None:
        from .card import Card
        deck_cards = getattr(getattr(player, "deck", None), "cards", [])
        basic_energies = [card for card in deck_cards if getattr(card, "is_basic_energy", False)]
        if not basic_energies:
            if player.print_actions:
                print("山札に基本エネルギーがありません。")
            return
        # 最大3枚まで選ぶ
        selected_energies = basic_energies[:3]
        # 場の全ポケモン
        all_pokemon = [player.active_card] + list(getattr(player, "bench", []))
        all_pokemon = [p for p in all_pokemon if p is not None]
        import random
        for energy_card in selected_energies:
            # ポケモン選択（人間プレイヤーの場合はinput、AIならランダム）
            if not player.is_bot:
                print(f"どのポケモンに {energy_card.name} を付けますか？")
                for idx, poke in enumerate(all_pokemon):
                    if isinstance(poke, Card):
                        print(f"{idx}: {poke.name}")
                    else:
                        print(f"{idx}: {poke}")
                while True:
                    try:
                        choice = int(input("番号を入力: "))
                        if 0 <= choice < len(all_pokemon):
                            break
                    except Exception:
                        pass
                    print("無効な入力です。")
                target_pokemon = all_pokemon[choice]
            else:
                # AIの場合はランダム
                target_pokemon = random.choice(all_pokemon)
            if isinstance(target_pokemon, Card):
                target_pokemon.attached_energies.append(energy_card)
                if player.print_actions:
                    print(f"{energy_card.name} を {target_pokemon.name} に付けた。")
            deck_cards.remove(energy_card)
        # 山札をシャッフル
        random.shuffle(deck_cards)
        if player.print_actions:
            print("山札を切った。")

    # キュレムexのブリザードバースト
    @staticmethod
    @apply_damage
    def blizzard_burst(player: "Player") -> None:
        from .card import Card
        # --- 1) 相手がとったサイド枚数 ---
        if not player.opponent:
            return
        side_taken = 6 - player.opponent.prize_left()
        bench_damage = 10 * side_taken
        # --- 2) バトル場の自分のカード ---
        active_card = player.active_card[-1] if isinstance(player.active_card, list) and player.active_card else player.active_card
        # --- 3) バトル場の相手カード ---
        opponent_card = player.opponent.active_card[-1] if isinstance(player.opponent.active_card, list) and player.opponent.active_card else player.opponent.active_card
        # --- 4) 基本ダメージ（ATTACKS定義通り） ---
        base_damage = 130
        # --- 5) 相手バトル場にダメージ ---
        if opponent_card and hasattr(opponent_card, "hp"):
            if isinstance(opponent_card, Card):
                add_damage_counters(opponent_card, base_damage // 10)
                ac_name = getattr(active_card, "name", "")
                oc_name = getattr(opponent_card, "name", "")
                if player.print_actions:
                    print(f"{ac_name} の ブリザードバースト！ {oc_name} に {base_damage} ダメージ！")
        # --- 6) 相手ベンチ全員に追加ダメージ（弱点・抵抗力計算なし） ---
        for bench_card in getattr(player.opponent, "bench", []):
            # bench_cardがリスト（進化山）の場合はトップカードを参照
            top_bench = bench_card[-1] if isinstance(bench_card, list) and bench_card else bench_card
            if isinstance(top_bench, Card):
                top_bench.hp = int(top_bench.hp) if top_bench.hp is not None else 0
                add_damage_counters(top_bench, bench_damage // 10)
                bc_name = getattr(top_bench, "name", "")
                if player.print_actions:
                    print(f"{bc_name}（ベンチ）に {bench_damage} ダメージ！（弱点・抵抗力計算なし）")

    # シビシラスのじっとする
    @staticmethod
    @apply_damage
    def jittosuru(player: "Player") -> None:
        from .card import Card
        # バトル場の自分のカード
        top_active = player.active_card[-1] if isinstance(player.active_card, list) and player.active_card else player.active_card
        if isinstance(top_active, Card):
            if hasattr(top_active, "hp"):
                top_active.hp = int(top_active.hp) if top_active.hp is not None else 0
                # 最大HPを超えないように回復
                max_hp = getattr(top_active, "max_hp", None)
                if max_hp is not None:
                    from .damage_utils import remove_damage_counters
                    remove_damage_counters(top_active, 1)
                else:
                    top_active.hp += 10
                if player.print_actions and getattr(top_active, "name", None):
                    print(f"{getattr(top_active, 'name', '')} の じっとする！ HPが10回復！")

