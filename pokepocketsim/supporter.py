##サポートを新しく定義する時は、必ず効果内で自身のカードをトラッシュするように設定する
##サポートを新しく定義する時は、必ずcard.pyにインポートするカードを定義させる。
##サポートを新しく定義する時は、必ずcard.pyにdef create_cardにサポートカード名を定義する。
##サポートを新しく定義する時は、必ずcards_db.pyにサポートカード名を定義する。
##サポートを新しく定義する時は、必ずcards_enum.pyにサポートカード名を定義する。

from enum import Enum
import random
from typing import Any, Union, TYPE_CHECKING
from .condition import Condition, ConditionBase
from .protocols import ICard, IPlayer
from .card import Card
from .card_base import CardBase
from .cards_enum import Cards

if TYPE_CHECKING:
    from .player import Player
SUPPORTER_CLASS_DICT = {}

def register_supporter(cls):
    SUPPORTER_CLASS_DICT[cls.__name__] = cls
    if hasattr(cls, "JP_NAME"):
        SUPPORTER_CLASS_DICT[cls.JP_NAME] = cls
    return cls

class Supporter(CardBase):
    """
    サポーター基底クラス

    実装者向けの契約(Contract):
    - use(self, player) -> bool を実装すること。
      * True を返した場合（=効果が有効に解決した場合）は、
        「自身を手札から取り除き、player.discard_pile へ移動する」処理まで
        “各サポータークラス側で必ず行う”こと。
        例:
            if self in player.hand:
                player.hand.remove(self)
            player.discard_pile.append(self)
      * False を返した場合（=効果不成立）はカードは手札に残す。
    - 追加コストや例外的なルール（例: 自身をトラッシュしない特殊カード）がある場合も
      ここ(use)の中で完結させ、Player側での一律トラッシュは行わない。

    これにより、カードごとの差異をクラス内に閉じ込め、Player側の分岐を増やさない方針。
    """
    def use(self, player: "Player") -> bool:
        raise NotImplementedError
    def __init__(self, card_enum: Cards, **kwargs):
        super().__init__(card_enum=card_enum, is_basic=False, **kwargs)
        self.id: int = card_enum.value[0]
        self.card_enum: Cards = card_enum
        self.name: str = card_enum.value[1]
        self.is_supporter: bool = True
        self.is_basic: bool = False
    def __str__(self):
        return self.name

@register_supporter
class Nanjamo(Supporter):
    """
    【ナンジャモ】（Iono）
    - お互いのプレイヤーは自分の手札を山札の下に戻して切る
    - その後、自分の残りサイド枚数と同じ枚数だけ引く
    """
    JP_NAME = "ナンジャモ"
    def __init__(self, **kwargs):
        super().__init__(Cards.NANJAMO, **kwargs)
    def __str__(self):
        return self.name

    # 対象カードを取らないので常に True
    def card_able_to_use(self, card: "ICard") -> bool:
        return True
    def player_able_to_use(self, player: "Player") -> bool:
        # 特に追加条件なし
        return True
    def use(self, player: "Player") -> None:
        # …効果処理
        if self in player.hand:
            player.hand.remove(self)
        player.discard_pile.append(self)
        player.log_print(f"{self.name} をトラッシュに送りました。")

        # 自分と相手両方に同じ処理を行う（相手がNoneの場合も安全に）
        targets = [player]
        if player.opponent is not None:
            targets.append(player.opponent)
        for pl in targets:
            # 1) 手札を山札の下へ
            pl.move_hand_to_deck_bottom_and_shuffle()
            # 2) 残りサイド枚数だけドロー
            prize_cnt: int = pl.prize_left() if callable(pl.prize_left) else pl.prize_left
            pl.draw_cards(prize_cnt)
            # 表示の分岐
            if pl is player:
                hand_cards = ', '.join(str(c) for c in pl.hand)
                print(f"{pl.name} の新しい手札: {hand_cards}")
            else:
                print(f"{pl.name} の新しい手札: （非公開：{len(pl.hand)}枚）")


@register_supporter
class BossOrders(Supporter):
    """
    【ボスの指令】
    - 相手のベンチポケモン 1 匹を選び、バトルポケモンと入れ替える
    """
    JP_NAME = "ボスの指令"
    def __init__(self, **kwargs):
        super().__init__(Cards.BOSS_ORDERS, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, card: "ICard") -> bool:
        return True

    def player_able_to_use(self, player: "Player") -> bool:
        return (
            player.opponent is not None
            and hasattr(player.opponent, "bench")
            and len(player.opponent.bench) > 0
        )

    def use(self, player: "Player") -> None:
        if self in player.hand:
            player.hand.remove(self)
        player.discard_pile.append(self)
        player.log_print(f"{self.name} をトラッシュに送りました。")

        if (
            player.opponent is None
            or not hasattr(player.opponent, "bench")
            or not player.opponent.bench
        ):
            player.log_print("相手ベンチにポケモンがいません。ボスの指令は無効です")
            return

        # --- 相手のベンチから 1 匹選択 ---
        if player.is_bot:
            idx = random.randint(0, len(player.opponent.bench) - 1)
            chosen_stack = player.opponent.bench[idx]
            player.log_print(f"AIは相手のベンチ{idx}（{chosen_stack[-1].name}）を選択")
        else:
            print("相手のベンチポケモン一覧:")
            for i, stack in enumerate(player.opponent.bench):
                pokemon = stack[-1]
                attached_energies = getattr(pokemon, "attached_energies", [])
                from collections import Counter
                counts = Counter(getattr(card, "name", str(card)) for card in attached_energies)
                attached_str = ", ".join(
                    f"{name}×{count}" if count > 1 else f"{name}"
                    for name, count in counts.items()
                )
                energy_display = f" [{attached_str}]" if attached_str else ""
                print(f"{i}: {pokemon.name} (HP: {pokemon.hp}){energy_display}")
            idx = int(input("バトル場に出す番号を選んでください: "))
            chosen_stack = player.opponent.bench[idx]
            player.log_print(f"ユーザーは相手のベンチ{idx}（{chosen_stack[-1].name}）を選択")

        # 入れ替え
        old_active = player.opponent.active_card
        player.opponent.bench.remove(chosen_stack)
        if old_active is not None:
            player.opponent.bench.append(old_active)
        player.opponent.active_card = chosen_stack

        # --- ログ用 extra 情報の付与 ---
        if hasattr(player, "last_action"):
            player.last_action.extra = {
                "selected_bench_idx": idx,
                "new_active_id": chosen_stack[-1].card_enum.value[0],
                "old_active_id": old_active[-1].card_enum.value[0] if old_active else None
            }

        if old_active is not None:
            player.log_print(f"相手の {old_active[-1].name} と {chosen_stack[-1].name} を入れ替えた！")
        else:
            player.log_print(f"相手の {chosen_stack[-1].name} をバトル場に出した！")

        return idx, chosen_stack[-1].card_enum.value[0]


@register_supporter
class Professors_Research(Supporter):
    """
    【博士の研究】
    - 手札を全てトラッシュする。その後、山札から7枚カードを引く
    """
    JP_NAME = "博士の研究"
    def __init__(self, **kwargs):
        super().__init__(Cards.PROFESSORS_RESEARCH, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, card: "ICard") -> bool:
        # 手札全体を対象とするため常にTrue
        return True

    def player_able_to_use(self, player: "Player") -> bool:
        # 基本的にいつでも使える
        return True

    def use(self, player: "Player") -> None:
        # …効果処理
        if self in player.hand:
            player.hand.remove(self)
        player.discard_pile.append(self)
        print(f"{self.name} をトラッシュに送りました。")

        # 1) 手札を全てトラッシュ
        discard_count = len(player.hand)
        player.discard_pile.extend(player.hand)
        player.hand.clear()
        print(f"{player.name} の手札 {discard_count} 枚を全てトラッシュしました")
        # 2) 山札から7枚引く
        player.draw_cards(7)
        hand_cards = ', '.join(str(c) for c in player.hand)
        print(f"{player.name} の新しい手札: {hand_cards}")

@register_supporter
class Crown(Supporter):
    """
    【クラウン】
    - おたがいのプレイヤーは、それぞれ手札をすべて山札にもどしてシャッフルする。
    - その後、自分はコインを1回投げる。
        - オモテなら自分は5枚、相手は3枚、山札から引く。
        - ウラなら自分は3枚、相手は5枚、山札から引く。
    """
    JP_NAME = "クラウン"
    DEBUG = False
    def __init__(self, **kwargs):
        super().__init__(Cards.CROWN, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, card: "ICard") -> bool:
        # 対象カードを取らない
        return True

    def player_able_to_use(self, player: "Player") -> bool:
        # いつでも使える
        return True

    def use(self, player: "Player") -> None:

        # --- 再入防止: 進行中なら抑止して即 return ---
        if getattr(self, "_effect_resolving", False):
            if self.DEBUG:
                print("[DEBUG] crown: reentry suppressed")
            return
        self._effect_resolving = True

        # --- デバッグ用: 使用前スナップショット ---
        if self.DEBUG:
            _dbg_before = {
                "hand_ids": [id(c) for c in player.hand],
                "trash_ids": [id(c) for c in player.discard_pile],
                "hand_len": len(player.hand),
                "deck_len": len(player.deck.cards),
                "trash_len": len(player.discard_pile),
            }

            print(f"\n[DEBUG] クラウン使用前:")
            print(f"  手札: {[player.format_card_public(c, show_hp=True) for c in player.hand]}")   # 自分の手札はHP出してOK（開発時）
            print(f"  山札: {len(player.deck.cards)}枚")
            print(f"  トラッシュ: {[player.format_card_public(c, show_hp=False) for c in player.discard_pile]}")  # ← HP非表示

        # 1) クラウンの仕様を宣言
        try:
            player.log_print(f"{player.name} は {self.name}（サポーター）を使った")
        except Exception:
            print(f"{player.name} は {self.name}（サポーター）を使った")

        # 2) 使用したクラウンを手札から削除してトラッシュに追加
        if self in player.hand:
            player.hand.remove(self)
            player.discard_pile.append(self)
            print(f"{self.name} をトラッシュに送りました。id={id(self)}")
        else:
            # 同名カードが手札に複数ある場合のデバッグ
            found = False
            for card in player.hand:
                if isinstance(card, Crown):
                    found = True
                    if self.DEBUG:
                        print(f"  [DEBUG] クラウン同名インスタンス: {id(card)} (self={id(self)})")
            if not found and self.DEBUG:
                print("  [DEBUG] 手札にクラウンがありませんでした")
            player.discard_pile.append(self)
            print(f"{self.name} (self) をトラッシュに追加 id={id(self)}")

        if self.DEBUG:
            print(f"\n[DEBUG] クラウン使用後（トラッシュ処理直後）:")
            print(f"  手札: {[str(c) for c in player.hand]}")
            print(f"  山札: {len(player.deck.cards)}枚")
            print(f"  トラッシュ: {[str(c) for c in player.discard_pile]}")

        # 3) 残った自分の手札を山札に戻してシャッフル
        if player.hand:
            player.deck.cards.extend(player.hand)
            player.hand.clear()
            random.shuffle(player.deck.cards)
            print(f"{player.name} は残りの手札を山札に戻してシャッフルしました。")
        else:
            print(f"{player.name} は戻す手札がありませんでした。")

        # 3') 相手の手札も山札に戻してシャッフル（効果:「おたがい」）
        if player.opponent and player.opponent.hand:
            player.opponent.deck.cards.extend(player.opponent.hand)
            player.opponent.hand.clear()
            random.shuffle(player.opponent.deck.cards)
            print(f"{player.opponent.name} も手札を山札に戻してシャッフルしました。")
        elif player.opponent:
            print(f"{player.opponent.name} は戻す手札がありませんでした。")

        if self.DEBUG:
            print(f"\n[DEBUG] クラウン使用後（山札戻し直後）:")
            print(f"  手札: {[str(c) for c in player.hand]}")
            print(f"  山札: {len(player.deck.cards)}枚")
            print(f"  トラッシュ: {[str(c) for c in player.discard_pile]}")

        # 4) コインを投げてドロー処理（表:5枚、裏:3枚、相手と自分）
        coin = random.choice(['表', '裏'])
        draw_self = 5 if coin == '表' else 3
        draw_opp = 3 if coin == '表' else 5

        try:
            player.log_print(f"{self.name}: コインの結果 → {coin}")
        except Exception:
            print(f"{self.name}: コインの結果 → {coin}")

        # 自分と相手で引く枚数が逆になる場合
        player.draw_cards(draw_self)
        if player.opponent:
            player.opponent.draw_cards(draw_opp)

        if self.DEBUG:
            print(f"\n[DEBUG] 新しい手札ドロー後:")
            print(f"  手札: {[str(c) for c in player.hand]}")
            print(f"  山札: {len(player.deck.cards)}枚")
            print(f"  トラッシュ: {[str(c) for c in player.discard_pile]}")

        # ドロー内容の表示（オプション）
        # ← ここだけ条件分岐＋ログ出力を追加（最小変更）
        if player.is_bot:
            # 相手（CPU）が使用 → 相手=枚数, 自分=内容
            msg1 = f"{player.name} の手札: {len(player.hand)} 枚"
            try:
                player.log_print(msg1)
            except Exception:
                print(msg1)
            if player.opponent:
                msg2 = f"{player.opponent.name} の新しい手札: {', '.join(str(c) for c in player.opponent.hand)}"
                try:
                    player.log_print(msg2)
                except Exception:
                    print(msg2)
        else:
            # 自分（人間）が使用 → 自分=内容, 相手=枚数
            msg1 = f"{player.name} の新しい手札: {', '.join(str(c) for c in player.hand)}"
            try:
                player.log_print(msg1)
            except Exception:
                print(msg1)
            if player.opponent:
                msg2 = f"{player.opponent.name} の手札: {len(player.opponent.hand)} 枚"
                try:
                    player.log_print(msg2)
                except Exception:
                    print(msg2)

        # --- デバッグ整合性チェック（異常検出時のみ WARN を出す） ---
        if self.DEBUG:
            try:
                _dup_idx = [i for i, c in enumerate(player.discard_pile) if c is self]
                if len(_dup_idx) == 0:
                    print(f"[WARN] クラウンのトラッシュ記録が見つかりません（hand.remove → discard.append の欠落の可能性） id={id(self)}")
                elif len(_dup_idx) > 1:
                    print(f"[WARN] クラウンがトラッシュに重複しています: indices={_dup_idx} id={id(self)}")

                if self in player.hand:
                    print(f"[WARN] クラウンが手札に残存しています（二重移動失敗の可能性） id={id(self)}")

                if 'hand_ids' in locals().get('_dbg_before', {}):
                    if id(self) in _dbg_before["hand_ids"] and self not in player.discard_pile:
                        print(f"[WARN] クラウンが『手札から消え、かつトラッシュにも無い』状態を検出（消滅疑い） id={id(self)}")

                # 収支の大まかな整合（カード総枚数の変化は戻し＋ドローで変動し得るため、ここではスナップショット差の報告のみ）
                _after = {
                    "hand_len": len(player.hand),
                    "deck_len": len(player.deck.cards),
                    "trash_len": len(player.discard_pile),
                }
                if '_dbg_before' in locals():
                    print(f"[DEBUG] crown-recap: before={{hand:{_dbg_before['hand_len']}, deck:{_dbg_before['deck_len']}, trash:{_dbg_before['trash_len']}}} "
                          f"after={{hand:{_after['hand_len']}, deck:{_after['deck_len']}, trash:{_after['trash_len']}}}")
            except Exception as e:
                print(f"[DEBUG] crown invariant check error: {e!r}")
        self._effect_resolving = False


@register_supporter
class Cheren(Supporter):
    """
    【チェレン】
    - 自分の山札を3枚引く
    """
    JP_NAME = "チェレン"
    def __init__(self, **kwargs):
        super().__init__(Cards.CHEREN, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, card: "ICard") -> bool:
        # 対象カードを取らない
        return True

    def player_able_to_use(self, player: "Player") -> bool:
        # いつでも使える
        return True

    def use(self, player: "Player") -> None:
        # …効果処理
        if self in player.hand:
            player.hand.remove(self)
        player.discard_pile.append(self)
        player.log_print(f"{self.name} をトラッシュに送りました。")

        player.draw_cards(3)
        hand_cards = ', '.join(str(c) for c in player.hand)
        print(f"{player.name} は山札から3枚引いた")
        print(f"{player.name} の新しい手札: {hand_cards}")


@register_supporter
class Touko(Supporter):
    """
    【トウコ】
    - 自分の山札から「進化ポケモン」と「エネルギー」を1枚ずつ選び、相手に見せて、手札に加える。
    - その後、山札をシャッフルする。
    """
    JP_NAME = "トウコ"
    def __init__(self, **kwargs):
        super().__init__(Cards.TOUKO, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, card: "ICard") -> bool:
        return True

    def player_able_to_use(self, player: "Player") -> bool:
        return True

    def use(self, player: "Player") -> None:
        if self in player.hand:
            player.hand.remove(self)
        player.discard_pile.append(self)
        print(f"{self.name} をトラッシュに送りました。")

        evo_candidates = [c for c in player.deck.cards if hasattr(c, "is_basic") and c.is_basic is False]
        energy_candidates = [c for c in player.deck.cards if getattr(c, "is_energy", False)]

        chosen_evo, chosen_energy = None, None

        # 進化ポケモン選択
        if evo_candidates:
            if player.is_bot:
                idx = random.randint(0, len(evo_candidates))  # 0〜len→スキップ含む
                if idx != 0:
                    chosen_evo = evo_candidates[idx-1]
            else:
                print("山札から加える進化ポケモンを選んでください（インデックス or カードID）:")
                print("0: 選ばない")
                for i, c in enumerate(evo_candidates, 1):
                    print(f"{i}: {c} (ID: {int(c.card_enum.value[0])})")
                inp = input("番号 or カードID: ").strip()
                found = None
                if inp.isdigit():
                    num = int(inp)
                    if num == 0:
                        pass
                    elif 1 <= num <= len(evo_candidates):
                        found = evo_candidates[num-1]
                    else:
                        # カードID
                        for c in evo_candidates:
                            if int(c.card_enum.value[0]) == num:
                                found = c
                                break
                if found:
                    chosen_evo = found

        # エネルギーカード選択
        if energy_candidates:
            if player.is_bot:
                idx = random.randint(0, len(energy_candidates))  # 0〜len→スキップ含む
                if idx != 0:
                    chosen_energy = energy_candidates[idx-1]
            else:
                print("山札から加えるエネルギーカードを選んでください（インデックス or カードID）:")
                print("0: 選ばない")
                for i, c in enumerate(energy_candidates, 1):
                    print(f"{i}: {c} (ID: {int(c.card_enum.value[0])})")
                inp = input("番号 or カードID: ").strip()
                found = None
                if inp.isdigit():
                    num = int(inp)
                    if num == 0:
                        pass
                    elif 1 <= num <= len(energy_candidates):
                        found = energy_candidates[num-1]
                    else:
                        for c in energy_candidates:
                            if int(c.card_enum.value[0]) == num:
                                found = c
                                break
                if found:
                    chosen_energy = found

        # 手札に加える
        added_cards = []
        if chosen_evo:
            player.deck.cards.remove(chosen_evo)
            player.hand.append(chosen_evo)
            added_cards.append(str(chosen_evo))
        if chosen_energy:
            player.deck.cards.remove(chosen_energy)
            player.hand.append(chosen_energy)
            added_cards.append(str(chosen_energy))

        # 行動ログ：カードIDのみ
        if hasattr(player, "last_action") and player.last_action:
            player.last_action.extra = {
                "evo_id": int(chosen_evo.card_enum.value[0]) if chosen_evo else 0,
                "energy_id": int(chosen_energy.card_enum.value[0]) if chosen_energy else 0,
            }

        # 表示
        if added_cards:
            print(f"{player.name} は「{'、'.join(added_cards)}」を相手に見せて手札に加えた")
        else:
            print(f"{player.name} は何も加えなかった")

        random.shuffle(player.deck.cards)
        print("山札をシャッフルしました")


@register_supporter
class Makomo(Supporter):
    """
    【マコモ】
    - 自分のバトル場とベンチのポケモン全員のHPを、それぞれ40回復する。
    """
    JP_NAME = "マコモ"
    def __init__(self, **kwargs):
        super().__init__(Cards.MAKOMO, **kwargs)

    def __str__(self):
        return self.name

    def card_able_to_use(self, card: "ICard") -> bool:
        # 対象をとらないため常にTrue
        return True

    def player_able_to_use(self, player: "Player") -> bool:
        # 場に1体でも「HPが減っている or ダメカン>0」のポケモンがいれば使用可
        def is_damaged(poke) -> bool:
            if getattr(poke, "damage_counters", 0) > 0:
                return True
            hp = getattr(poke, "hp", None)
            mx = getattr(poke, "max_hp", None)
            return (hp is not None and mx is not None and hp < mx)

        if getattr(player, "active_card", None):
            if is_damaged(player.active_card[-1]):
                return True
        for st in getattr(player, "bench", []):
            if st and is_damaged(st[-1]):
                return True
        return False

    def use(self, player: "Player") -> None:
        targets = []
        # …効果処理

        # --- デバッグ: 使用前スナップショットと対象抽出（ダメージありのみ） ---
        def _is_damaged(poke) -> bool:
            if getattr(poke, "damage_counters", 0) > 0:
                return True
            hp = getattr(poke, "hp", None)
            mx = getattr(poke, "max_hp", None)
            return (hp is not None and mx is not None and hp < mx)

        if hasattr(player, "active_card") and player.active_card and _is_damaged(player.active_card[-1]):
            targets.append(player.active_card[-1])
        if hasattr(player, "bench") and player.bench:
            for stack in player.bench:
                if stack and _is_damaged(stack[-1]):
                    targets.append(stack[-1])

        print(f"\n[DEBUG] マコモ使用前: damaged_targets={len(targets)} / hand={len(player.hand)} deck={len(player.deck.cards)} trash={len(player.discard_pile)}")

        # 使用不可ケース（ダメージ対象が一体もいない）→ 消費せず終了
        if not targets:
            print("[DEBUG] マコモ: ダメージを受けたポケモンがいないため使用できません（トラッシュしません）")
            return

        # 消費はここで一度だけ（重複トラッシュ回避）
        if self in player.hand:
            player.hand.remove(self)
            if self not in player.discard_pile:
                player.discard_pile.append(self)
        else:
            if self not in player.discard_pile:
                player.discard_pile.append(self)
        print(f"{self.name} をトラッシュに送りました。")

        # --- 回復処理（対象のみ40回復） ---
        for poke in targets:
            # 表示の一貫性: 「前」の数値でラベル固定し、その後にHPを書き換える
            name = getattr(poke, "name", str(poke))
            before = getattr(poke, "hp", 0) or 0
            label = f"{name} (HP: {max(0, before)})"  # 0未満は0に丸めて表示
            # 実際の回復
            max_hp = getattr(poke, "max_hp", before)
            poke.hp = min(max_hp, before + 40)
            after = poke.hp
            print(f"{label} のHPを40回復（{before}→{after}）")

        print(f"{player.name} のバトル場とベンチのポケモン全員のHPを40回復した")

        # --- デバッグ: 使用後スナップショット＆異常検出 ---
        if self not in player.discard_pile:
            print("[WARN] マコモ: トラッシュ記録が見つかりません（消費漏れ疑い）")
        if self in player.hand:
            print("[WARN] マコモ: 消費後も手札に残存しています（remove失敗の可能性）")
        print(f"[DEBUG] マコモ使用後: hand={len(player.hand)} deck={len(player.deck.cards)} trash={len(player.discard_pile)}")
