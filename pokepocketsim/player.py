from __future__ import annotations
import random, copy
from typing import (
    List,
    Optional,
    TYPE_CHECKING,
    Dict,
    Any,
    Union,
    Type,
    Tuple,
    cast,
    Protocol,
)
import uuid
import json
from .action import ActionType, Action
from .card import Card, Cards
from .item import Item
from .battle_logger import BattleLogger
from .supporter import Supporter
from .attack import Attack
from .attack_common import EnergyType
from .protocols import ICard, IPlayer
from collections import Counter
import re
from collections import OrderedDict
from .damage_utils import add_damage_counters

MAX_BENCH = 5

if TYPE_CHECKING:
    from .deck import Deck
    from .match import Match
    from .action import Action

class Player:
    def __init__(self, name: str, deck: "Deck", is_bot: bool = True) -> None:
        self.logger: BattleLogger = BattleLogger(self)
        self.enforce_ex_no_evolve = True
        self.print_actions: bool = True
        self.name: str = name
        self.deck: "Deck" = deck
        self.is_bot: bool = is_bot
        self.mulligan_count: int = 0  # マリガン回数を追加
        # 山札をシャッフルしてから開幕マリガン用ドロー開始
        self.deck.shuffle()
        # --- 開幕手札 7 枚ドロー＋マリガン処理 ---
        self.hand: List[Any]
        while True:
            # 7 枚ドロー
            self.hand = [
                card
                for _ in range(min(7, len(self.deck.cards)))
                if (card := self.deck.draw_card()) is not None
            ]
            # ベーシックポケモンがあるかチェック
            has_basic = any(
                isinstance(card, Card) and card.is_basic
                for card in self.hand
            )
            if has_basic or not self.deck.cards:
                break
            # マリガン：手札を山札底に戻してシャッフル
            self.log_print(f"{self.name} の手札にベーシックポケモンがいないのでマリガンします")
            self.mulligan_count += 1  # マリガン回数をカウント
            self.deck.cards.extend(self.hand)
            self.hand.clear()
            self.deck.shuffle()
            # --- AI同士の対戦時、山札を引くかの選択肢と選択内容の出力は不要 ---
        self.log_print('---------------------------------------------------')  # ★ここ追加
        # --- マリガン終了 ---
        self.discard_pile: List[Card] = []
        self.bench: List[List[Card]] = []
        self.active_card: Optional[List[Card]] = None
        self.points: int = 0
        self.opponent: Optional["Player"] = None
        self.current_energy: Optional[str] = None
        self.has_used_trainer: bool = False
        self.has_added_energy: bool = False
        self.can_continue: bool = True
        self.id: uuid.UUID = uuid.uuid4()
        self.evaluate_actions: bool = False
        self.prize_cards: list[ICard] = []  # サイドカードは後で作成
        self.has_used_item: bool = False        # グッズ
        self.has_used_supporter: bool = False   # サポーター
        self.has_used_stadium: bool = False     # スタジアム
        self.has_used_tool: bool = False        # どうぐ
        self.has_retreat_this_turn: bool = False  # 1ターンに1回だけ逃げる可
        self.active_stadium: Optional[Any] = None  # 現在のスタジアムカード
        self.has_used_stadium_effect = False  # スタジアム効果の使用フラグ
        self._last_attack_damage: Optional[int] = None  # 最後の攻撃のダメージ
        self.has_attacked_this_turn: bool = False  # このターンでワザを使用したかどうか
        self.last_action_type: Optional[ActionType] = None  # 最後に実行されたアクションのタイプ
        self.match: Optional["Match"] = None  # linterエラー回避のため追加
        self._printed_state_before: bool = False
        self._last_coin_result: Optional[str] = None  # 最後のコイン判定結果を記録
        self._last_trashed_cards: Optional[List[str]] = None  # 最後にトラッシュしたカードを記録
        self._last_selected_pokemon: Optional[str] = None  # 最後に選択したポケモンを記録

        self.was_knocked_out_by_attack_last_turn: bool = False  # ワザで相手がきぜつしたかどうか
        self.katakiuti_bonus_80: bool = False  # かたきうちの80ダメージボーナスフラグ

        # ==== 追加：MLログ用ワーク領域 ====
        self._pending_substeps: list = []              # サブステップ一時バッファ
        self._current_substep: dict | None = None      # サブステップ編集中の一時領域
        self._last_legal_actions_before: list[list[int]] = []  # 直前の合法手（5整数ベクトル）
        self._pending_decide_lines: list[str] = []     # [DECIDE_*] を行動行直前に出すためのバッファ

    def _build_state_snapshot(self) -> Dict[str, Any]:
        return self.logger.build_state_snapshot()


    def begin_substep(self, phase: str, legal_actions: List[List[int]]):
        return self.logger.begin_substep(phase, legal_actions)

    def end_substep(self, action_vec: List[int], action_index: int):
        return self.logger.end_substep(action_vec, action_index)

    def apply_stadium_effects(self) -> None:
        """
        現在のスタジアムの効果を適用する
        """
        if self.active_stadium and hasattr(self.active_stadium, 'apply_effect'):
            self.active_stadium.apply_effect(self)

    def log_print(self, *args, **kwargs):
        try:
            m = getattr(self, "match", None)
            if m is not None and hasattr(m, "log_print"):
                return m.log_print(*args, **kwargs)
        except Exception:
            pass
        return self.logger.log_print(*args, **kwargs)
    
    def ml_log_print(self, *args, **kwargs):
        return self.logger.ml_log_print(*args, **kwargs)

    @property
    def active_card_and_bench(self) -> List[List[Card]]:
        stacks = []
        if self.active_card:
            stacks.append(self.active_card)
        stacks.extend(self.bench)
        return stacks

    def set_opponent(self, other: "Player") -> None:
        self.opponent = other        # Match 初期化時に必ず呼ぶ

    def start_turn(self, match: "Match", viewing_player=None) -> None:
        # ターン開始時に場の全ポケモンを進化可能にする（ただし1ターン目の禁止を反映）
        cur_turn = getattr(self.match, "turn", 0) if self.match else 0
        for stack in self.active_card_and_bench:
            if not stack:
                continue
            top = stack[-1]
            top.can_evolve = Player._eligible_to_evolve_now(self, top)
            if hasattr(top, "has_used_ability"):
                top.has_used_ability = False

        # ターン開始時にワザ・アクション種別フラグをリセット
        self.has_attacked_this_turn = False
        self.last_action_type = None
        self.has_added_energy = False
        self.has_used_supporter = False
        self.has_retreat_this_turn = False
        self.has_used_stadium = False
        self.has_used_stadium_effect = False

        # ★ 追加：ターン開始ヘッダをコンソールに明示（手番・山札/手札/トラッシュ/ベンチ/バトル場）
        try:
            if self.match:
                side = "先攻" if (self.match.turn % 2 == 0) else "後攻"
                header = f"----- {side} {self.match.turn}ターン開始: {self.name} -----"
                self.log_print(header)
                deck_n  = len(getattr(self.deck, "cards", []))
                hand_n  = len(getattr(self, "hand", []))
                trash_n = len(getattr(self, "discard_pile", []))
                bench_n = len(getattr(self, "bench", []))
                self.log_print(f"{self.name}: 山札 {deck_n} / 手札 {hand_n} / トラッシュ {trash_n} / ベンチ {bench_n}")
                if self.active_card and len(self.active_card) > 0:
                    top = self.active_card[-1]
                    self.log_print(f"バトル場: {getattr(top, 'name', '???')} (HP:{max(0, getattr(top, 'hp', 0) or 0)})")
                else:
                    self.log_print("バトル場: なし")
        except Exception:
            pass

        # ゲーム終了フラグが立っていれば何もしない
        if self.match and getattr(self.match, 'game_over', False):
            return

        # --- 強制ドロー（山札切れ＝敗北） ---
        # ドローは setup_turn() で1回だけ行うように統一
        pass

        # log_mode=True時はターン冒頭の全体状態出力を抑制する
        if not (self.match and getattr(self.match, 'log_mode', False)):
            pass  # ここでのprint_player_state()呼び出しを削除

        # [STATE_OBJ_BEFORE]出力フラグをリセット
        self._printed_state_before = False

        # ==== step-start hook (PBRS) ====
        # NOTE:
        #   on_step_start は act_and_regather_actions() の「各アクション直前」で呼ぶため、
        #   ここでは呼ばない（二重呼び出し防止）。

        # --- ここからアクションループ（毎回再列挙） ---
        # 既存の RL / best_actions フローは必要なら先に処理
        if hasattr(self, 'action_to_take') and getattr(self, 'action_to_take', None) is not None:
            actions = self.gather_actions(match)
            self.process_rl_actions(match, actions, self.action_to_take)
            return

        if hasattr(self, 'best_actions') and getattr(self, 'best_actions', None):
            self.process_best_actions(match, self.best_actions)
            return

        while True:
            # 1) 直近の盤面から合法手を最新化
            actions = self.gather_actions(match)

            # 行動が無ければ終了（安全策：END_TURNがあればそれを適用）
            if not actions:
                break

            # ★ 直前の合法手（5整数）をキャッシュ（学習ログ用）
            self._last_legal_actions_before = [a.serialize(self) for a in actions]
            if hasattr(self.logger, "last_legal_actions_before"):
                self.logger.last_legal_actions_before = self._last_legal_actions_before

            # 2) 現在の状態スナップショット
            #    - AI: MCC/学習向けに logger のスナップショット（me_private/opp_private 等）を優先
            #    - 人間: UI向け公開状態を優先
            if getattr(self, "is_bot", False):
                try:
                    state_dict = self._build_state_snapshot()
                except Exception:
                    state_dict = self.get_state_dict()
            else:
                try:
                    # 公開情報を UI 向けに整形（hand/トラッシュ等に "cat" などが付与される）
                    state_dict = match.build_public_state_for_ui(viewer=viewing_player)
                except Exception:
                    # 互換フォールバック
                    state_dict = self.get_state_dict()

            # 3) インデックス選択（人間は入力、CPUは bind_policy_to_player で差し替え済み）
            try:
                idx = int(self.select_action(state_dict, actions))
            except Exception:
                idx = 0
            if idx < 0 or idx >= len(actions):
                idx = 0

            selected_action = actions[idx]
            actions = self.act_and_regather_actions(match, selected_action)

            # 5) 対戦終了 or ターン終了なら抜ける
            if getattr(match, "game_over", False):
                break
            if selected_action.action_type in (ActionType.ATTACK, ActionType.END_TURN):
                break

        # ここまで来たらターン処理を返す
        return

    def compute_reward(self) -> tuple[float, int]:
        """
        報酬と done を一元的に返す。
        - self.match.game_over が立ったターンのみ done=1
        - 勝者が自分  … reward=+1
        - 敗者が自分  … reward=-1
        - 進行中      … reward=0, done=0
        """
        if self.match and getattr(self.match, "game_over", False):
            winner = getattr(self.match, "winner", None)
            return (1.0, 1) if winner == self.name else (-1.0, 1)
        return (0.0, 0)


    def print_player_state(self, actions=None, action_tuple2id=None):
        return self.logger.print_player_state(actions=actions, action_tuple2id=action_tuple2id)

    def handle_knockout_points(self) -> bool:
        result = False
        if (
            self.opponent is not None
            and self.opponent.active_card is not None
            and self.opponent.active_card[-1].hp is not None
            and self.opponent.active_card[-1].hp <= 0
        ):
            if self.print_actions:
                self.log_print(
                    f"{self.opponent.name} の {self.opponent.active_card[-1].name} は気絶した！"
                )
            # サイド取得
            # サイド取得
            prize_to_take = 2 if self.opponent.active_card[-1].is_ex else 1
            if self.print_actions:
                kind = "EXポケモン" if prize_to_take == 2 else "通常ポケモン"
                self.log_print(f"{self.name} は{kind}を気絶させたのでサイドカードを{prize_to_take}枚獲得します")
            self.points += prize_to_take

            # ★ forced_actions 対応：KO後のサイド取得は match 側 forced に切り替える（無ければ従来通り即時取得）
            self._set_forced_take_prize(getattr(self, "match", None), prize_to_take)

            # --- 気絶したポケモンと付随カードを全てトラッシュへ ---
            if self.opponent.active_card is not None:
                for card in self.opponent.active_card:
                    # 付随エネルギー
                    if hasattr(card, "attached_energies"):
                        self.opponent.discard_pile.extend(card.attached_energies)
                        card.attached_energies = []
                    # 付随どうぐ
                    tools = getattr(card, "tools", None)
                    if tools:
                        # ★追加: 外れる前にフックで副作用を巻き戻す（例: にげるコスト低下を解除）
                        for tool in list(tools):
                            if hasattr(tool, "on_detached"):
                                try:
                                    tool.on_detached(self.opponent, card)  # (player, pokemon)
                                except Exception:
                                    pass
                        # その後、通常通りトラッシュへ
                        self.opponent.discard_pile.extend(tools)
                        delattr(card, "tools")
                    self.opponent.discard_pile.append(card)
            # バトル場から除去
            self.opponent.active_card = None

            # --- ベンチが空なら勝利 ---
            if not self.opponent.bench or len(self.opponent.bench) == 0:
                if self.print_actions:
                    self.log_print(f"{self.opponent.name} の場から全てのポケモンがいなくなりました。{self.name} の勝利です！")
                if self.match:
                    setattr(self.match, "_end_reason", "bench_out_or_knockout")
                    self.match.game_over = True
                    self.match.winner = self.name
                    try:
                        if hasattr(self.opponent, "logger") and self.opponent.logger:
                            self.opponent.logger.log_terminal_step(reason="bench_out_or_knockout")
                    except Exception:
                        pass
                result = True
            else:
                # ベンチから昇格（人間は選択、AIはランダム/今後強化学習へ拡張可）
                self.opponent.promote_from_bench()
        return result

    def promote_from_bench(self):
        """
        ベンチからバトル場にポケモンを昇格する。
        人間は入力で、AIはランダムで選択（今後、強化学習対応もここに拡張）。
        """
        if not self.bench or len(self.bench) == 0:
            # ベンチが空＝何もできない
            return None
        if not self.is_bot:
            # 人間の場合
            while True:
                print("バトル場に出すポケモンをベンチ番号で選んでください：")
                for idx, stack in enumerate(self.bench, start=1):
                    top = stack[-1]
                    print(f"  [{idx}] {top.name} (HP: {self._display_hp(getattr(top, 'hp', 0))})")
                try:
                    choice_raw = int(input("ベンチ番号を入力: "))
                    if 1 <= choice_raw <= len(self.bench):
                        choice = choice_raw - 1
                        break
                    else:
                        print("番号が正しくありません。")
                except Exception:
                    print("整数で入力してください。")
            promoted = self.bench.pop(choice)
        else:
            # AI（現状はランダム、将来強化学習可）
            # 学習済みAIならここにポリシー/モデルによる選択ロジック
            if bool(getattr(getattr(self, "match", None), "_is_mcts_simulation", False)):
                choice = 0
            else:
                choice = random.randrange(len(self.bench))
            promoted = self.bench.pop(choice)
        self.active_card = promoted
        if hasattr(self, "log_print"):
            self.log_print(f"{self.name} はベンチから {promoted[-1].name} をバトル場に出しました。")
        return promoted

    def print_state_after(self, turn=None):
        return self.logger.print_state_after(turn=turn)

    def print_possible_actions(self, actions: List["Action"]) -> None:
        # 人間プレイヤーの場合は常に選択肢を表示
        if not self.is_bot:
            self.log_print("--------------------------------")   
            if self.print_actions:
                self.log_print(f"{self.name} の選択肢:")
                print(f"{self.name} の選択肢:")
                for i, action in enumerate(actions, start=1):
                    self.log_print(f"\t{i}: {action}")
                    print(f"\t{i}: {action}")
        # AIプレイヤーの場合はprintのみ
        else:
            print("--------------------------------")
            if self.print_actions:
                print(f"{self.name} の選択肢:")
                for i, action in enumerate(actions, start=1):
                    print(f"\t{i}: {action}")

    def _card_display_sort_key(self, card):
        """
        表示専用のカテゴリ順:
        0: ポケモン, 1: グッズ, 2: サポート, 3: どうぐ, 4: スタジアム, 5: エネルギー, 99: 不明
        ※ 既存属性のみ使用（内部状態やログは変更しない）
        """
        name = getattr(card, "name", "")
        # ポケモン判定: HPや攻撃を持つ / is_basic / 進化カード等
        if getattr(card, "is_basic", False) or hasattr(card, "hp") or hasattr(card, "attacks"):
            cat = 0
        elif getattr(card, "is_item", False):
            cat = 1
        elif getattr(card, "is_supporter", False):
            cat = 2
        elif getattr(card, "is_tool", False):
            cat = 3
        elif getattr(card, "is_stadium", False):
            cat = 4
        elif getattr(card, "is_energy", False) or "エネルギー" in name:
            cat = 5
        else:
            cat = 99
        return (cat, name)

    def _format_card_list_for_display(self, cards):
        """
        同名カードをまとめ、_card_display_sort_key で表示順に並べて文字列を返す
        """
        if not cards:
            return "なし"
        # 表示用にのみ並び替え（元のリスト順は不変更）
        sorted_cards = sorted(cards, key=self._card_display_sort_key)
        from collections import OrderedDict
        counts = OrderedDict()
        for c in sorted_cards:
            nm = getattr(c, "name", str(c))
            counts[nm] = counts.get(nm, 0) + 1
        return ", ".join(f"{n}×{cnt}" if cnt > 1 else n for n, cnt in counts.items())

    def _display_hp(self, hp):
        """
        表示専用HPの丸め（ログ整形専用）:
        - 数値なら 0 未満を 0 に丸める
        - None は 0 扱い
        - 数値化できない場合はそのまま返す（'?'
          などの表記を保持）
        """
        try:
            return max(0, int(hp))
        except Exception:
            return 0 if hp is None else hp

    def format_card_public(self, card, show_hp: bool = True) -> str:
        """公開情報としてカード名を整形して返す。
        トラッシュや山札の一覧では show_hp=False を渡してHPを隠す。
        """
        name = getattr(card, "name", str(card))
        if show_hp and hasattr(card, "hp") and hasattr(card, "max_hp"):
            return f"{name} (HP: {card.hp})"
        return name

    def _print_turn_snapshot_for_human(self):
        """
        人間プレイヤー向けの盤面スナップショット（手札/トラッシュは所望順で表示）
        ※ ログ用オブジェクトには触らない
        """
        # 自分側：バトル場
        print("現在のバトル場:")
        if self.active_card and len(self.active_card) > 0:
            top = self.active_card[-1]
            attached = getattr(top, "attached_energies", [])
            from collections import Counter
            counts = Counter(getattr(card, "name", str(card)) for card in attached)
            attached_str = ", ".join(f"{name}×{count}" if count > 1 else f"{name}" for name, count in counts.items())
            print(f"\t{top.name} (HP: {self._display_hp(getattr(top, 'hp', 0))}) [{attached_str}]")
        else:
            print("\tなし")

        # 自分側：ベンチ
        print("現在のベンチ:")
        from collections import Counter
        for idx, stack in enumerate(self.bench):
            if isinstance(stack, list) and len(stack) > 0:
                top = stack[-1]
                attached = getattr(top, "attached_energies", [])
                counts = Counter(getattr(card, "name", str(card)) for card in attached)
                attached_str = ", ".join(f"{name}×{count}" if count > 1 else f"{name}" for name, count in counts.items())
                print(f"\t{idx + 1}: {top.name} (HP: {self._display_hp(getattr(top, 'hp', 0))}) [{attached_str}]")

        # ★ 手札（所望順にソートして表示）
        print("現在の手札:")
        print(f"\t{self._format_card_list_for_display(self.hand)}")

        # ★ トラッシュ（所望順にソートして表示）
        print("現在のトラッシュ:")
        print(f"\t{self._format_card_list_for_display(getattr(self, 'discard_pile', []))}")

        # 相手側（公開情報のみ）
        if self.opponent:
            print("相手のバトル場:")
            if self.opponent.active_card and len(self.opponent.active_card) > 0:
                top = self.opponent.active_card[-1]
                attached = getattr(top, "attached_energies", [])
                counts = Counter(getattr(card, "name", str(card)) for card in attached)
                attached_str = ", ".join(f"{name}×{count}" if count > 1 else f"{name}" for name, count in counts.items())
                print(f"\t{top.name} (HP: {getattr(top,'hp',0)}) [{attached_str}]")
            else:
                print("\tなし")

            print("相手のベンチ:")
            for idx, stack in enumerate(self.opponent.bench):
                if isinstance(stack, list) and len(stack) > 0:
                    top = stack[-1]
                    attached = getattr(top, "attached_energies", [])
                    counts = Counter(getattr(card, "name", str(card)) for card in attached)
                    attached_str = ", ".join(f"{name}×{count}" if count > 1 else f"{name}" for name, count in counts.items())
                    print(f"\t{idx + 1}: {top.name} (HP: {getattr(top,'hp',0)}) [{attached_str}]")

    def process_best_actions(
        self, match: "Match", best_actions
    ):
        if best_actions:
            actions = self.gather_actions(match)
            # ★ 直前合法手を保持（レコードの一貫性を担保）
            self._last_legal_actions_before = [a.serialize(self) for a in actions]
            best_action = Action.find_action(actions, best_actions[0])
            self.act_and_regather_actions(match, best_action)
            best_actions.pop(0)
            return best_actions
        else:
            self.can_continue = False
            return []

    def process_user_actions(
        self, match: "Match", actions: List["Action"]
    ) -> List["Action"]:
        # ★ 既に終局しているならここで終了（以降の処理・表示を止める）
        if self.match and getattr(self.match, 'game_over', False):
            self.can_continue = False
            return []
        # -------------------------------
        #  色付け（ポケモン名のみ）準備
        # -------------------------------
        try:
            if not getattr(self.__class__, "_colorama_inited", False):
                # Colorama があれば使う（無ければ ImportError で素通り）
                from colorama import init, Fore, Style
                init(autoreset=True)
                self.__class__._colorama_inited = True
                self.__class__._POKE = Fore.CYAN + Style.BRIGHT
                self.__class__._RESET = Style.RESET_ALL
        except Exception:
            # 失敗時は無色
            self.__class__._POKE = ""
            self.__class__._RESET = ""
            self.__class__._colorama_inited = True
        POKE = getattr(self.__class__, "_POKE", "")
        RESET = getattr(self.__class__, "_RESET", "")
        def poke_name(name: str) -> str:
            return f"{POKE}{name}{RESET}" if POKE else name
        # -------------------------------
        #  表示用のヘルパ
        # -------------------------------
        from collections import Counter
        def energy_str(attached_list: List[Any]) -> str:
            counts = Counter(getattr(c, "name", str(c)) for c in (attached_list or []))
            if not counts:
                return ""
            return ", ".join(f"{k}×{v}" if v > 1 else k for k, v in counts.items())
        def bench_line(idx: int, stack: List[Any]) -> Optional[str]:
            if isinstance(stack, list) and stack:
                top = stack[-1]
                nm = poke_name(getattr(top, "name", "???"))
                hp = self._display_hp(getattr(top, "hp", 0))
                att = energy_str(getattr(top, "attached_energies", []))
                return f"\t{idx}: {nm} (HP: {hp}) [{att}]"
            return None
        # ★ 既存の表示ロジックを流用（ポケモン→グッズ→サポーター→どうぐ→スタジアム→エネルギー、同名まとめ）
        def join_card_names(cards: List[Any]) -> str:
            return self._format_card_list_for_display(list(cards))
        # スタジアム取得（自分→相手の順で存在する方）
        def current_stadium_name() -> str:
            st = getattr(self, "active_stadium", None)
            if not st and getattr(self, "opponent", None):
                st = getattr(self.opponent, "active_stadium", None)
            return getattr(st, "name", "なし") if st else "なし"
        # 盤面を（毎回）出すための関数
        def show_console_state():
            # --- 自分側の公開情報ヘッダ（山札/サイド/手札 枚数） ---
            my_deck_n  = len(getattr(self.deck, "cards", []))
            my_side_n  = len(getattr(self, "prize_cards", []))
            my_hand_n  = len(getattr(self, "hand", []))
            print(f"あなた: 山札 {my_deck_n} / サイド {my_side_n} / 手札 {my_hand_n}")
            # バトル場
            print("現在のバトル場:")
            if self.active_card and len(self.active_card) > 0:
                top = self.active_card[-1]
                nm = poke_name(getattr(top, "name", "???"))
                hp = self._display_hp(getattr(top, "hp", 0))
                att = energy_str(getattr(top, "attached_energies", []))
                print(f"\t{nm} (HP: {hp}) [{att}]")
            else:
                print("\tなし")
            # ベンチ
            print("現在のベンチ:")
            if getattr(self, "bench", None):
                for i, stack in enumerate(self.bench, start=1):
                    line = bench_line(i, stack)
                    if line:
                        print(line)
            # 手札（カテゴリソート＆集計表示）
            print("現在の手札:")
            print(f"\t{join_card_names(list(getattr(self, 'hand', [])))}")
            # トラッシュ（カテゴリソート＆集計表示）
            print("現在のトラッシュ:")
            print(f"\t{join_card_names(list(getattr(self, 'discard_pile', [])))}")
            # --- 相手側（公開情報） ---
            if self.opponent:
                opp_deck_n = len(getattr(self.opponent.deck, "cards", []))
                opp_side_n = len(getattr(self.opponent, "prize_cards", []))
                # 手札の枚数は原則公開情報（中身は非公開）
                opp_hand_n = len(getattr(self.opponent, "hand", []))
                print(f"相手: 山札 {opp_deck_n} / サイド {opp_side_n} / 手札 {opp_hand_n}")
                print("相手のバトル場:")
                if self.opponent.active_card and len(self.opponent.active_card) > 0:
                    top = self.opponent.active_card[-1]
                    nm = poke_name(getattr(top, "name", "???"))
                    hp = self._display_hp(getattr(top, "hp", 0))
                    att = energy_str(getattr(top, "attached_energies", []))
                    print(f"\t{nm} (HP: {hp}) [{att}]")
                else:
                    print("\tなし")
                print("相手のベンチ:")
                if getattr(self.opponent, "bench", None):
                    for i, stack in enumerate(self.opponent.bench, start=1):
                        line = bench_line(i, stack)
                        if line:
                            print(line)
                # 追加：相手のトラッシュ
                print("相手のトラッシュ:")
                print(f"\t{join_card_names(list(getattr(self.opponent, 'discard_pile', [])))}")
            # 最後に現在のスタジアム
            print(f"スタジアム: {current_stadium_name()}")
            print("--------------------------------")
        # 人間プレイヤーのターン開始時：盤面情報を出力
        show_console_state()
        # 直前の合法手（5整数）をキャッシュ（学習ログ用）
        self._last_legal_actions_before = [a.serialize(self) for a in actions]
        # === 入力ループ（行動→盤面を即再表示）===
        while actions and not (self.match and getattr(self.match, 'game_over', False)):
            # ★ 直前の合法手（5整数）をキャッシュ
            self._last_legal_actions_before = [a.serialize(self) for a in actions]
            selected_index = self.choose_action(actions, print_actions=True)
            selected_action = actions[selected_index]
            # 実行
            actions = self.act_and_regather_actions(match, selected_action)
            # 実行直後にも終局チェック
            if self.match and getattr(self.match, 'game_over', False):
                break
            # ★ ドローや盤面変化後の“最新状態”を再表示
            show_console_state()
            # ワザまたはターンエンドでターン終了
            if selected_action.action_type in [ActionType.ATTACK, ActionType.END_TURN]:
                break
        return actions

    def process_bot_actions(
        self, match: "Match", actions: List["Action"]
    ) -> List["Action"]:
        # 終局後にアクションを続けないためのガード
        while actions and not (self.match and getattr(self.match, 'game_over', False)):

            # actions と action_tuple2id を渡して状態出力
            self.print_player_state(actions, getattr(match, "action_tuple2id", None))
            self._last_legal_actions_before = [a.serialize(self) for a in actions]
            self.print_possible_actions(actions)
            if getattr(self.match, 'log_mode', False):
                self.log_print('--------------------------------')

            try:
                state_dict = self._build_state_snapshot()
            except Exception:
                state_dict = self.get_state_dict()
            action_obj = self.select_action(state_dict, actions)

            if isinstance(action_obj, int):
                selected_action = actions.pop(action_obj)
            elif isinstance(action_obj, Action):
                try:
                    idx = actions.index(action_obj)
                    selected_action = actions.pop(idx)
                except ValueError:
                    selected_action = actions.pop(0)
            else:
                import random
                selected_action = actions.pop(random.randint(0, len(actions) - 1))

            if getattr(self.match, 'log_mode', False):
                self.log_print(f"AI選択: {selected_action}")
                self.log_print(f"{self.name} 手札：{', '.join(str(c) for c in self.hand)}")

            actions = self.act_and_regather_actions(match, selected_action)

            # ★ 実行直後にも終局チェック（相手のきぜつ・サイド取り切り・山札切れ等で終了するため）
            if self.match and getattr(self.match, 'game_over', False):
                break

            # ワザまたはターンエンドで明示的に終了
            if selected_action.action_type in [ActionType.ATTACK, ActionType.END_TURN]:
                break

        # ---------- ループを抜けた後 ----------
        # ★ 既に終局しているなら強制 END_TURN は入れない（余計なログを出さない）
        if self.match and getattr(self.match, 'game_over', False):
            self.can_continue = False
            return []

        # ・END_TURN も ATTACK も行っていない場合だけ 強制 END_TURN を入れる
        if self.last_action_type not in (ActionType.END_TURN, ActionType.ATTACK):

            end_action = Action(
                "[ターンエンド] 自分の番を終わる",
                Player.end_turn,
                ActionType.END_TURN,
                can_continue_turn=False,
            )
            self.last_action = end_action
            self.last_action_type = ActionType.END_TURN

            # ★ 最新の合法手をEND_TURNのみとして記録
            self._last_legal_actions_before = [[int(ActionType.END_TURN), 0, 0, 0, 0]]
            self.logger.last_legal_actions_before = self._last_legal_actions_before

            # ★ act前の状態を記録
            pre_state = self._build_state_snapshot()

            # ★ 実行
            end_action.act(self)

            # ★ レコード出力
            action_result = self._build_action_result(end_action)
            post_state = self._build_state_snapshot()
            self._log_step(pre_state, self._last_legal_actions_before, action_result, post_state)

        # ターンは確実に終了したのでフラグを落とし、空リストを返す
        self.can_continue = False
        return []


    def process_rl_actions(
        self, match: "Match", actions: List["Action"], action_to_take: int
    ) -> List["Action"]:
        if actions:
            self._last_legal_actions_before = [a.serialize(self) for a in actions]
            if 0 <= action_to_take < len(actions):
                selected_action = actions.pop(action_to_take)
                actions = self.act_and_regather_actions(match, selected_action)
                return actions
            else:
                self.can_continue = False
                return []
        else:
            self.can_continue = False
            return []

    def act_and_regather_actions(self, match: "Match", action: "Action") -> List["Action"]:

        # === (A) すでに終局していたら即終了 ===
        if self.match and getattr(self.match, "game_over", False):
            self.can_continue = False
            return []

        # ① 実行直前スナップショット（legal_actions は直前に保持したものを使う）
        pre_state = self.logger.build_state_snapshot()
        # 直前の合法手は logger 側の保持を優先
        legal_actions_vec = getattr(self.logger, "last_legal_actions_before", [])
        if not legal_actions_vec:
            legal_actions_vec = getattr(self, "_last_legal_actions_before", [])

        # ★ 追加：直前の意思決定ログを“行動行の直前”に必ず出す
        try:
            pend = getattr(self, "_pending_decide_lines", None)
            if isinstance(pend, list) and pend:
                for line in pend:
                    self.log_print(line)
                pend.clear()
        except Exception:
            pass

        # ★ 追加：実行アクションをコンソールに即出力（人/CPU共通）
        try:
            turn_str = f"{self.match.turn}" if self.match else "?"
            self.log_print(f"{turn_str} {self.name}: {action}")
        except Exception:
            pass

        # ② 実行
        self.last_action = action
        self.last_action_type = action.action_type

        # ★ 追加：MCC/PBRS step start（実行前フック）
        try:
            m = (self.match if self.match is not None else match)
            rs = getattr(m, "reward_shaping", None) if m is not None else None
            if rs is not None and (getattr(m, "use_reward_shaping", False) or getattr(m, "use_mcc", False)):
                fn = getattr(rs, "on_step_start", None)
                if callable(fn):
                    try:
                        fn(player=self, match=m)
                    except TypeError:
                        try:
                            fn(self, m)
                        except TypeError:
                            fn()
        except Exception:
            pass

        self.can_continue = action.act(self)
        self.handle_knockout_points()

        # ③ アクション結果（substeps は begin/end_substep により注入済み）
        action_result = self.logger.build_action_result(action)

        # ★ 追加：MCC/PBRS step end（ここで MCC を回す想定）
        try:
            m = (self.match if self.match is not None else match)
            rs = getattr(m, "reward_shaping", None) if m is not None else None
            if rs is not None and (getattr(m, "use_reward_shaping", False) or getattr(m, "use_mcc", False)):
                base_reward = 0.0
                try:
                    if isinstance(action_result, dict):
                        base_reward = float(action_result.get("reward", 0.0))
                except Exception:
                    base_reward = 0.0

                fn = getattr(rs, "on_step_end", None)
                if callable(fn):
                    shaped = None
                    try:
                        shaped = fn(player=self, match=m, base_reward=base_reward)
                    except TypeError:
                        try:
                            shaped = fn(self, m, base_reward)
                        except TypeError:
                            try:
                                shaped = fn(self, m)
                            except TypeError:
                                shaped = fn()

                    try:
                        if isinstance(shaped, (tuple, list)) and shaped:
                            shaped_val = shaped[0]
                        else:
                            shaped_val = shaped

                        if isinstance(action_result, dict):
                            action_result.setdefault("reward_base", base_reward)
                            if isinstance(shaped_val, (int, float)):
                                action_result["reward_shaped"] = float(shaped_val)
                    except Exception:
                        pass
        except Exception:
            pass

        # ④ 実行直後スナップショット
        post_state = self.logger.build_state_snapshot()

        # ⑤ 1レコードで出力
        self.logger.log_step(pre_state, legal_actions_vec, action_result, post_state)

        # --- 追加：CPUターンで選択肢を出さない設定なら、実行結果だけ1行で出す ---
        # （例：「[ターンエンド] 自分の番を終わる」「[エネルギー] 雷エネルギー を バトル場 の 〇〇 に付ける」等）
        if self.is_bot and not self.print_actions:
            try:
                self.print_action_result(action)  # BattleLogger.print_action_result(...) を経由
            except Exception:
                pass

        # === (B) アクション結果で終局したら、ここで完全停止 ===
        if self.match and getattr(self.match, "game_over", False):
            # ★ 追加：終局時にポリシーサマリを1回だけ出す（PhaseD-Q 等）
            try:
                m = self.match
                if not getattr(m, "_policy_summary_logged", False):
                    setattr(m, "_policy_summary_logged", True)
                    pol = getattr(self, "policy", None)
                    if pol is not None:
                        for fn_name in ("log_summary", "print_summary", "dump_summary", "summary", "on_game_end", "on_match_end"):
                            fn = getattr(pol, fn_name, None)
                            if callable(fn):
                                try:
                                    fn(player=self, match=m, reason=getattr(m, "_end_reason", None))
                                except TypeError:
                                    try:
                                        fn(self, m)
                                    except TypeError:
                                        try:
                                            fn(m)
                                        except TypeError:
                                            fn()
                                break
            except Exception:
                pass

            self.can_continue = False
            return []

        # ⑥ 次の候補収集
        actions_next: List["Action"] = []
        if action.action_type == ActionType.SET_ACTIVE_CARD:
            pass
        elif self.can_continue:
            actions_next = self.gather_actions(match)
        return actions_next


    def _build_action_result(self, action: "Action") -> Dict[str, Any]:
        return self.logger.build_action_result(action)

    def _log_step(self, pre_state: Dict[str, Any], legal_actions: List[List[int]], action_result: Dict[str, Any], post_state: Dict[str, Any]):
        return self.logger.log_step(pre_state, legal_actions, action_result, post_state)

    def print_action_result(self, action: "Action"):
        return self.logger.print_action_result(action)

    def print_state_only(self):
        # バトル場のポケモン情報
        if self.active_card:
            top = self.active_card[-1]
            attached_energies = getattr(top, "attached_energies", [])
            energies = [e.name for e in attached_energies] if attached_energies else []
            conditions = [c.__class__.__name__ for c in getattr(top, "conditions", [])]
            active_pokemon_obj = {
                "name": top.name,
                "hp": max(0, top.hp if top.hp is not None else 0),
                "energies": energies if energies else [],
                "conditions": conditions
            }
        else:
            active_pokemon_obj = {"name": None, "hp": None, "energies": [], "conditions": []}
        bench_pokemon_list = []
        for idx in range(5):
            if idx < len(self.bench) and self.bench[idx]:
                stack = self.bench[idx]
                if stack:
                    top = stack[-1]
                    attached_energies = getattr(top, "attached_energies", [])
                    energies = [e.name for e in attached_energies] if attached_energies else []
                    bench_pokemon_list.append({
                        "name": top.name,
                        "hp": max(0, top.hp if top.hp is not None else 0),
                        "energies": energies if energies else []
                    })
                else:
                    bench_pokemon_list.append(None)
            else:
                bench_pokemon_list.append(None)
        
        # 手札のカード名リスト
        hand_names = [card.name for card in self.hand]
        
        reward_val, done_val = self.compute_reward()
        reward = reward_val  # 片方だけで良い場合はこれで
        
        log_obj = {
            "game_id": getattr(self.match, 'game_id', None),
            "turn": getattr(self.match, 'turn', 0),
            "player": self.name,
            "hand": hand_names,
            "hand_count": len(self.hand),
            "bench_count": len([p for p in self.bench if p]),
            "prize_count": len(self.prize_cards),
            "deck_count": len(self.deck.cards),
            "active_pokemon": active_pokemon_obj,
            "bench_pokemon": bench_pokemon_list,
            "discard_pile": [c.name for c in self.discard_pile],
            "discard_pile_count": len(self.discard_pile),
            "reward": reward
        }
        
        # PBRS使用時は整形報酬の詳細情報を追加
        if self.match and self.match.use_reward_shaping and self.match.reward_shaping:
            potential = self.match.reward_shaping.calculate_potential(self, self.match)
            log_obj["potential"] = potential
            log_obj["shaped_reward"] = reward  # 既に整形済みの報酬
        
        # アクション後の相手の状態を出力
        if self.opponent:
            # バトル場
            if self.opponent.active_card:
                top = self.opponent.active_card[-1]
                attached_energies = getattr(top, "attached_energies", [])
                energies = [e.name for e in attached_energies] if attached_energies else []
                conditions = [c.__class__.__name__ for c in getattr(top, "conditions", [])]
                opp_active_pokemon_obj = {
                    "name": top.name,
                    "hp": max(0, top.hp if top.hp is not None else 0),
                    "energies": energies if energies else [],
                    "conditions": conditions
                }
            else:
                opp_active_pokemon_obj = {"name": None, "hp": None, "energies": [], "conditions": []}
            
            # ベンチ（5枠固定、いない場合はnull）
            opp_bench_pokemon = []
            for idx in range(5):
                if idx < len(self.opponent.bench) and self.opponent.bench[idx]:
                    top = self.opponent.bench[idx][-1]
                    attached_energies = getattr(top, "attached_energies", [])
                    energies = [e.name for e in attached_energies] if attached_energies else []
                    opp_bench_pokemon.append({"name": top.name, "hp": max(0, top.hp if top.hp is not None else 0), "energies": energies if energies else []})
                else:
                    opp_bench_pokemon.append(None)
            
            opp_after_log_obj = {
                "game_id": getattr(self.match, 'game_id', None),
                "turn": getattr(self.match, 'turn', None),
                "player": self.opponent.name,
                "hand_count": len(self.opponent.hand),
                "bench_count": len(self.opponent.bench),
                "prize_count": len(self.opponent.prize_cards),
                "deck_count": len(self.opponent.deck.cards),
                "active_pokemon": opp_active_pokemon_obj,
                "bench_pokemon": opp_bench_pokemon,
                "discard_pile": [c.name for c in self.opponent.discard_pile],
                "discard_pile_count": len(self.opponent.discard_pile),
            }
            
            opp_after_output = "[STATE_OBJ_OPPONENT_AFTER] " + json.dumps(opp_after_log_obj, ensure_ascii=False)
            # print(opp_after_output, flush=True)  # ← 画面出力を抑制
            # ログファイルには出力しない

    def remove_item_from_hand(self, card_class: Optional[Type[Any]]) -> None:
        if card_class is None:
            return

        try:
            card_to_remove = next(
                card for card in self.hand if isinstance(card, card_class)
            )
            self.hand.remove(card_to_remove)
        except StopIteration:
            # Handle the case where no matching card is found
            if self.print_actions:
                self.log_print(f"No {card_class.__name__} card found in 手札 to remove")

    def prize_left(self) -> int:
        """残りサイド枚数を返す"""
        return len(self.prize_cards)
        # サイドを取ったときに 1 枚減らすメソッド例

    def _build_take_prize_actions(self) -> List["Action"]:
        acts: List["Action"] = []
        # ActionType.TAKE_PRIZE が未実装なら空（呼び出し側でフォールバックする）
        if not hasattr(ActionType, "TAKE_PRIZE"):
            return acts

        for i in range(len(self.prize_cards)):
            a = Action(
                f"[サイド取得] サイドカード{i + 1}番目を取る",
                lambda player=self, idx=i: player.take_prize(forced_idx=idx),
                ActionType.TAKE_PRIZE,
                can_continue_turn=True,
            )
            # serialize 側で使うなら自由に（必須でなければ無くてもOK）
            a.extra = {"selected_prize_idx": i + 1}
            acts.append(a)

        return acts

    def _set_forced_take_prize(self, match: "Match", count: int) -> None:
        """
        KO後のサイド取得を forced_actions に切り替える。
        TAKE_PRIZE が無い場合は従来通り即時取得へフォールバック。
        """
        m = match if match is not None else getattr(self, "match", None)

        if m is None or (not hasattr(ActionType, "TAKE_PRIZE")):
            for _ in range(int(count or 0)):
                if self.prize_cards:
                    self.take_prize()
            return

        try:
            setattr(m, "forced_player", self)
        except Exception:
            pass

        try:
            setattr(m, "forced_prize_remaining", int(count or 0))
        except Exception:
            pass

        m.forced_actions = self._build_take_prize_actions()

    def take_prize(self, forced_idx: Optional[int] = None) -> None:
        if not self.prize_cards:
            raise ValueError("既にサイドが残っていません")

        if forced_idx is not None:
            try:
                idx = int(forced_idx)
            except Exception:
                idx = 0
            if idx < 0 or idx >= len(self.prize_cards):
                idx = 0
        else:
            if self.is_bot:
                if bool(getattr(getattr(self, "match", None), "_is_mcts_simulation", False)):
                    idx = 0
                    if self.print_actions:
                        self.log_print(f"{self.name} はサイドカード{idx + 1}番目を固定選択しました（MCTSシミュレーション）")
                else:
                    idx = random.randint(0, len(self.prize_cards) - 1)
                    # AI同士の対戦時はサイドカード選択の詳細を表示
                    if self.print_actions:
                        self.log_print(f"{self.name} はサイドカード{idx + 1}番目をランダム選択しました")
            else:
                print("サイドカードを選んでください:")
                for i in range(1, len(self.prize_cards) + 1):
                    print(f"{i}: [裏向き]")
                while True:
                    try:
                        idx = int(input("番号: ")) - 1
                        if 0 <= idx < len(self.prize_cards):
                            break
                    except Exception:
                        pass
                    print("無効な入力です。")

        card = self.prize_cards.pop(idx)
        self.hand.append(card)
        self.log_print(f"サイドカードとして「{card}」を手札に加えました")
        # 残りサイド枚数を表示
        if self.print_actions:
            self.log_print(f"{self.name} の残りサイド枚数: {len(self.prize_cards)}枚")

        # forced_prize_remaining がある場合は decrement & 次の候補を更新/クリア
        try:
            m = getattr(self, "match", None)
            if m is not None and hasattr(m, "forced_prize_remaining"):
                rem = int(getattr(m, "forced_prize_remaining", 0) or 0)
                if rem > 0:
                    rem -= 1
                    setattr(m, "forced_prize_remaining", rem)
                    if rem > 0 and self.prize_cards:
                        m.forced_actions = self._build_take_prize_actions()
                    else:
                        m.forced_actions = []
                        try:
                            setattr(m, "forced_player", None)
                        except Exception:
                            pass
        except Exception:
            pass

        # --- サイド全取り勝利判定（TAKE_PRIZE 実行時に確定） ---
        if len(self.prize_cards) == 0 and self.match:
            if self.print_actions:
                self.log_print(f"{self.name} はサイドを全て取り切ったので勝利です！")
            setattr(self.match, "_end_reason", "prize_out")
            self.match.game_over = True
            self.match.winner = self.name
            try:
                if hasattr(self, "logger") and self.logger:
                    self.logger.log_terminal_step(reason="prize_out")
            except Exception:
                pass

    @staticmethod
    def remove_card_from_hand(player: "Player", card_id: uuid.UUID) -> None:
        card = Player.find_by_id(player.hand, card_id)
        if not card:
            raise ValueError(f"Card not found in hand.")
        player.hand.remove(card)

    def gather_actions(self, match: "Match") -> List["Action"]:
        # ★ 追加：終局ガード（最優先で返す）
        if self.match and getattr(self.match, 'game_over', False):
            return []

        # ★ 追加：forced_actions が有効なら、それだけを返す（KO後サイド取得など）
        try:
            if match is not None and getattr(match, "forced_player", None) is self:
                fa = getattr(match, "forced_actions", None)
                if isinstance(fa, list) and fa:
                    return fa
        except Exception:
            pass

        # --- enum/id 正規化ヘルパ（Enum/tuple値/名前→常に int に揃える）---
        def _as_card_id(enum_or_int, name_fallback=None) -> int:
            try:
                # すでに int
                if isinstance(enum_or_int, int):
                    return enum_or_int
                # Enum メンバー
                if hasattr(enum_or_int, "value"):
                    v = enum_or_int.value
                    if isinstance(v, int):
                        return v
                    if isinstance(v, (tuple, list)) and len(v) > 0:
                        return int(v[0])
                # その他の数値化可能な型
                return int(enum_or_int)
            except Exception:
                # 名前から Cards を引いて補完
                if name_fallback:
                    try:
                        from .cards_enum import Cards as CardsEnum
                        for c in CardsEnum:
                            vals = c.value
                            # vals = (id, ja_name, en_key?) の想定
                            if len(vals) >= 2 and vals[1] == name_fallback:
                                return int(vals[0])
                            if len(vals) >= 3 and vals[2] == name_fallback:
                                return int(vals[0])
                    except Exception:
                        pass
                return 0

        # --- ★追加：ベンチ番号は「同一性(is)」で引く（list.index の同値判定を避ける）---
        def _bench_index_by_identity(stack_obj) -> int:
            try:
                for _i, _st in enumerate(self.bench, start=1):
                    if _st is stack_obj:
                        return _i
            except Exception:
                pass
            return 0

        actions: List["Action"] = []
        if self.active_card is not None:
            # 3. 特性
            for stack in self.active_card_and_bench:
                card = stack[-1]
                if (
                    card.ability
                    and not card.has_used_ability
                    and hasattr(card.ability, "able_to_use")
                    and card.ability.able_to_use(self)
                ):
                    ability_actions = card.ability.gather_actions(self, card)

                    # --- フォールバック: ability_id と bench_idx を付与 ---
                    # stack がバトル場なら 0、ベンチなら 1..5 を推定（同一性で確実に）
                    if stack is self.active_card:
                        bench_idx_val = 0
                    else:
                        bench_idx_val = _bench_index_by_identity(stack)

                    for ability_action in ability_actions:
                        if ability_action.action_type == ActionType.USE_ABILITY:
                            # extra が無ければ dict を用意
                            if not hasattr(ability_action, "extra") or not isinstance(ability_action.extra, dict):
                                ability_action.extra = {}

                            # schema 側で要求されることが多い selected_bench_idx をフォールバック設定
                            ability_action.extra.setdefault("selected_bench_idx", bench_idx_val)
                            # 万一 bench_idx を参照する処理系にも配慮
                            ability_action.extra.setdefault("bench_idx", bench_idx_val)

                        actions.append(ability_action)

            # 4. エネルギー付与
            if not self.has_added_energy and self.active_card is not None:
                energy_cards = [c for c in self.hand if getattr(c, "is_energy", False)]
                for e_card in energy_cards:
                    eid = getattr(e_card, "id", None) or getattr(e_card, "card_id", None)  # 実行時用(UUID)
                    for target in self.active_card_and_bench:
                        tgt = target[-1]
                        tid = getattr(tgt, "id", None) or getattr(tgt, "card_id", None)  # 実行時用(UUID)
                        if target is self.active_card:
                            position = "バトル場"
                            bench_idx = 0
                        else:
                            idx = _bench_index_by_identity(target)
                            position = f"ベンチ{idx}"
                            bench_idx = idx
                        a = Action(
                            f"[エネルギー] {e_card.name} を {position} の {tgt.name} に付ける",
                            lambda player=self, tid=tid, eid=eid: self.attach_energy_card(tid, eid),
                            ActionType.ATTACH_ENERGY,
                            can_continue_turn=True
                        )
                        # ★ serialize 用（固定数値IDとベンチ番号）
                        enum_raw = getattr(e_card, "card_enum", None)
                        enum_id = _as_card_id(enum_raw, getattr(e_card, "name", None))
                        a.card_id = enum_id
                        # serialize 側フォールバック用に energy_id も保持
                        a.extra = {"bench_idx": bench_idx, "target_id": tid, "energy_id": enum_id}
                        actions.append(a)

            # 5. グッズ
            item_cards = [c for c in self.hand if hasattr(c, "is_item") and c.is_item]
            for item_card in item_cards:
                # ポケモンいれかえはベンチが空ならスキップ
                if getattr(item_card, 'name', '') == 'ポケモンいれかえ' and (not self.bench or len(self.bench) == 0):
                    continue
                # ポケモンキャッチャーは相手のベンチが空ならスキップ
                if getattr(item_card, 'name', '') == 'ポケモンキャッチャー' and (not self.opponent or not self.opponent.bench or len(self.opponent.bench) == 0):
                    continue
                # card_able_to_useがあれば使用可否を判定
                if hasattr(item_card, 'card_able_to_use'):
                    if not item_card.card_able_to_use(self):
                        continue  # 使用不可ならアクション追加しない

                iid = getattr(item_card, "id", None) or getattr(item_card, "card_id", None)  # 実行時用(UUID)
                a = Action(
                    f"[グッズ] {item_card.name} を使う",
                    lambda player=self, iid=iid: self.use_item_by_id(iid),
                    ActionType.USE_ITEM,
                    card_class=type(item_card),
                )
                # ★ serialize 用（固定数値ID）: enum/名前を確実に int 化
                a.card_id = _as_card_id(getattr(item_card, "card_enum", None), getattr(item_card, "name", None))
                actions.append(a)

            # 5.5. ポケモンのどうぐ
            tool_cards = [c for c in self.hand if hasattr(c, "is_tool") and c.is_tool]
            for tool_card in tool_cards:
                attachable_targets = []
                for i, stack in enumerate(self.active_card_and_bench):
                    top = stack[-1]
                    # 既にどうぐが付いているポケモンはスキップ
                    if hasattr(top, 'tools') and top.tools:
                        continue
                    if hasattr(tool_card, 'can_attach_to') and tool_card.can_attach_to(top):
                        position = "バトル場" if i == 0 else f"ベンチ{i}"
                        attachable_targets.append((top, position, i))

                for target, position, i in attachable_targets:
                    tool_id = getattr(tool_card, "id", None) or getattr(tool_card, "card_id", None)   # 実行時用(UUID)
                    target_id = getattr(target, "id", None) or getattr(target, "card_id", None)       # 実行時用(UUID)
                    a = Action(
                        f"[どうぐ] {tool_card.name} を {position} の {target.name} に付ける",
                        lambda player=self, tool_id=tool_id, target_id=target_id: self.attach_tool_by_id(tool_id, target_id),
                        ActionType.ATTACH_TOOL,
                        can_continue_turn=True,
                        card_class=type(tool_card),
                    )
                    # ★ serialize 用（固定数値IDと対象情報）
                    a.card_id = _as_card_id(getattr(tool_card, "card_enum", None), getattr(tool_card, "name", None))
                    a.extra = {"target_id": target_id, "bench_idx": i}
                    actions.append(a)

            # 6. サポーター
            if not self.has_used_supporter and not (match and match.turn == 0 and match.starting_player == self):
                supporter_cards = [c for c in self.hand if hasattr(c, "is_supporter") and c.is_supporter]
                for sup in supporter_cards:
                    if hasattr(sup, "player_able_to_use") and not sup.player_able_to_use(self):
                        continue
                    # 実行時に使う UUID 等
                    sid_local = getattr(sup, "id", None) or getattr(sup, "card_id", None)
                    sup_local = sup  # 遅延束縛対策（lambda 既定引数で捕捉）

                    a = Action(
                        f"[サポーター] {sup_local.name} を使う",
                        lambda player=self, sid=sid_local, s=sup_local: (
                            self.use_supporter_by_id(sid) if hasattr(self, "use_supporter_by_id") else self.use_supporter(s)
                        ),
                        ActionType.USE_SUPPORTER,
                        card_class=type(sup_local),
                    )
                    # ★ serialize 用（固定数値ID）: enum/名前を確実に int 化
                    a.card_id = _as_card_id(getattr(sup_local, "card_enum", None), getattr(sup_local, "name", None))
                    actions.append(a)

            # 7. スタジアム
            if not self.has_used_stadium:
                stadium_cards = [c for c in self.hand if hasattr(c, "is_stadium") and c.is_stadium]
                current = self._current_stadium()  # ← 追加：自分 or 相手、場にあるスタジアム
                for st in stadium_cards:
                    # 既存のスタジアムと同名なら候補に出さない（実行時エラーを未然に回避）
                    if current and getattr(current, "name", None) == getattr(st, "name", None):
                        continue
                    # すでに自分が同名を出しているケース（冪等ガード）
                    if self.active_stadium and self.active_stadium.name == st.name:
                        continue
                    if hasattr(st, "player_able_to_use") and not st.player_able_to_use(self):
                        continue

                    sid_local = getattr(st, "id", None) or getattr(st, "card_id", None)
                    st_local = st
                    a = Action(
                        f"[スタジアム] {st_local.name} を場に出す",
                        lambda player=self, sid=sid_local, st_obj=st_local:
                            self.use_stadium_by_id(sid) if hasattr(self, "use_stadium_by_id") else self.use_stadium(st_obj),
                        ActionType.PLAY_STADIUM,
                        card_class=type(st_local),
                    )
                    # serialize 用の card_id 付与（先頭で定義した _as_card_id をそのまま使う）
                    a.card_id = _as_card_id(getattr(st_local, "card_enum", None), getattr(st_local, "name", None))
                    actions.append(a)

            # 7.5. スタジアム効果の使用（ボウルタウン）
            st = self.active_stadium or (self.opponent.active_stadium if self.opponent else None)
            if st and hasattr(st, 'apply_effect') and not self.has_used_stadium_effect:
                if st.name == "ボウルタウン":
                    if len(self.bench) < 5:
                        basic_pokemon = [
                            c for c in self.deck.cards
                            if (hasattr(c, "is_basic") and c.is_basic and 
                                hasattr(c, "is_ex") and not c.is_ex and
                                hasattr(c, "hp") and c.hp is not None)
                        ]
                        if basic_pokemon:
                            # 事前に候補一覧を表示してから effect 本体を呼ぶラッパ
                            def _bowl_town_act(player=self, st_obj=st, candidates=basic_pokemon):
                                # 人間に見えるように候補一覧をコンソールへ表示
                                header = "【ボウルタウン】山札のたねポケモン候補:"
                                # ログにも少し残す（printは人間向け、log_printはログファイル向け）
                                try:
                                    player.log_print(header)
                                except Exception:
                                    print(header)
                                for i, c in enumerate(candidates, start=1):
                                    # UUID は長いので先頭8文字だけ添える（入力は番号でOK）
                                    cid = getattr(c, "id", None) or getattr(c, "card_id", None)
                                    cid_short = str(cid)[:8] if cid is not None else "-"
                                    line = f"  {i}: {c.name} (HP: {getattr(c,'hp',0)})  id={cid_short}…"
                                    try:
                                        player.log_print(line)
                                    except Exception:
                                        print(line)
                                if not candidates:
                                    msg = "（候補なし）"
                                    try:
                                        player.log_print(msg)
                                    except Exception:
                                        print(msg)
                                # 効果を適用（フラグ更新はスタジアム側で行う想定）
                                return st_obj.apply_effect(player)

                            a = Action(
                                "[スタジアム] ボウルタウンの効果を使用",
                                _bowl_town_act,
                                ActionType.STADIUM_EFFECT,
                            )
                            # ★ serialize 用（固定数値IDを付けられるなら付与）
                            if hasattr(st, "card_enum"):
                                a.card_id = _as_card_id(getattr(st, "card_enum", None), getattr(st, "name", None))
                            actions.append(a)

            # 7. 逃げる（ねむり/まひ等は“にげる不可”）
            if (
                not self.has_retreat_this_turn
                and self.active_card is not None
                and Player._cannot_retreat_reason(self.active_card[-1]) is None  # ← 追加
                and self.active_card[-1].get_total_energy() is not None
                and self.active_card[-1].get_effective_retreat_cost() is not None
                and self.active_card[-1].get_total_energy() >= self.active_card[-1].get_effective_retreat_cost()
                and len(self.bench) > 0
            ):
                a = Action(
                    f"[逃げる] バトル場の ({self.active_card[-1]})",
                    lambda player=self: Player.retreat(player),
                    ActionType.RETREAT,
                )
                # 逃げるは card_id 不要だが、固定数値ID（enum由来）に統一しておく（API: 5整数は全てint）
                top = self.active_card[-1]
                a.card_id = _as_card_id(getattr(top, "card_enum", None), getattr(top, "name", None))
                actions.append(a)

            # 8. ワザ
            # 先攻1ターン目はワザ宣言不可
            if not (match.turn == 0 and match.starting_player == self):
                if self.active_card is not None:
                    card = self.active_card[-1]
                    for attack in card.attacks:
                        result = Attack.can_use_attack(card, attack)
                        if result:
                            # attack_common.ATTACKSのname_jaを優先
                            attack_name = None
                            from .attack_common import ATTACKS
                            if attack.__name__ in ATTACKS and 'name_ja' in ATTACKS[attack.__name__]:
                                attack_name = ATTACKS[attack.__name__]['name_ja']
                            if not attack_name:
                                attack_name = getattr(attack, 'name_ja', None)
                            if not attack_name:
                                attack_name = getattr(attack, '__name__', str(attack))
                            a = Action(
                                f"[ワザ宣言] {card.name} の {attack_name}",
                                attack,
                                ActionType.ATTACK,
                                can_continue_turn=False,
                            )
                            # ★ serialize 用（攻撃IDを必ず入れる）
                            if not hasattr(a, "extra") or not isinstance(a.extra, dict):
                                a.extra = {}
                            attack_id_val = getattr(attack, "attack_id", None)
                            if not attack_id_val:
                                try:
                                    from .cards_enum import Attacks as AttacksEnum
                                    # ① 日本名で一致 ② なければ関数名(英キー)で一致
                                    attack_id_val = next(
                                        atk.value[0] for atk in AttacksEnum
                                        if atk.value[1] == attack_name or atk.value[2] == getattr(attack, "__name__", "")
                                    )
                                except Exception:
                                    attack_id_val = 0
                            a.extra["attack_id"] = _as_card_id(attack_id_val, attack_name)
                            if hasattr(card, "id"):
                                a.card_id = _as_card_id(getattr(card, "card_enum", None), getattr(card, "name", None))  # （任意）攻撃の“持ち主”
                            actions.append(a)

            # 9. ターンエンド
            if self.active_card is not None and len(actions) > 0:
                actions.append(
                    Action(
                        "[ターンエンド] 自分の番を終わる",
                        Player.end_turn,
                        ActionType.END_TURN,
                        can_continue_turn=False,
                    )
                )
        else:
            # 先攻1ターン目：アクティブを置く
            if not self.active_card:
                from .energy_card import EnergyCard
                for card in self.hand:
                    if isinstance(card, Card) and not isinstance(card, EnergyCard) and card.is_basic:
                        # ★ 実行時用(UUID)とserialize用(数値ID)を分ける
                        cid = getattr(card, "id", None) or getattr(card, "card_id", None)  # 実行時
                        enum_id = getattr(card, "card_enum", None)  # serialize
                        turn_now = match.turn
                        a = Action(
                            f"Set {card.name} as バトル場",
                            lambda player=self, cid=cid, turn=turn_now: Player.set_active_card_from_hand(
                                player, cid, turn
                            ),
                            ActionType.SET_ACTIVE_CARD,
                        )
                        # ★ serialize 用
                        a.card_id = _as_card_id(enum_id, getattr(card, "name", None))
                        actions.append(a)

        # --- アクション重複抑制 ---
        unique_actions = []
        seen = set()
        for a in actions:
            # 実行意味が同じ（[t, main, p3, p4, p5] が一致）なら重複とみなす
            key = tuple(a.serialize(self))

            # ★ 追加: 特性はターゲットで区別する
            if a.action_type == ActionType.USE_ABILITY:
                ex = getattr(a, "extra", {}) or {}
                # target_id があればそれを、無ければ bench_idx / selected_bench_idx を使う
                target_marker = ex.get("target_id") or ex.get("selected_target_bench_idx") \
                                or ex.get("bench_idx") or ex.get("selected_bench_idx")
                if target_marker is not None:
                    key = key + (str(target_marker),)
                # エネルギーも別物ならさらに区別しておく（任意だが安全）
                if "energy_id" in ex:
                    key = key + (str(ex["energy_id"]),)
                # ← ここを追加：同名の能力持ち（例: シビビール複数）を区別
                using_marker = ex.get("using_card_id") or ex.get("using_id")
                if using_marker is not None:
                    key = key + (str(using_marker),)


            if key not in seen:
                unique_actions.append(a)
                seen.add(key)
        actions = unique_actions

        # 何もできない場合でも必ずターンエンドを追加
        if not actions:
            actions.append(
                Action(
                    "[ターンエンド] 自分の番を終わる",
                    Player.end_turn,
                    ActionType.END_TURN,
                    can_continue_turn=False,
                )
            )
        return actions


    def end_turn(self):
        """
        ターンエンド:
        - can_continue を False に
        - ターン内フラグ／一時フラグをリセット
        ACTION_RESULT のログは print_action_result() 側で 1 回だけ出力する
        """
        self.can_continue = False

        # --- ターン内1回まで系（冪等リセット） ---
        self.has_added_energy = False
        self.has_used_supporter = False
        self.has_retreat_this_turn = False
        self.has_attacked_this_turn = False
        self.has_used_stadium = False
        self.has_used_stadium_effect = False
        self.has_used_tool = False

        # --- 一時的な効果フラグ（次ターンに持ち越さない） ---
        self.katakiuti_bonus_80 = False
        self.was_knocked_out_by_attack_last_turn = False

        return False    # Action.act() の戻り値


    def attach_energy_card(self, target_id, energy_card_id):
        stack = Player.find_by_id(self.active_card_and_bench, target_id)
        energy_card = Player.find_by_id(self.hand, energy_card_id)
        if stack and energy_card:
            top = stack[-1]
            # attached_energiesのみに統一
            if not hasattr(top, "attached_energies"):
                top.attached_energies = []
            top.attached_energies.append(energy_card)
            self.hand.remove(energy_card)
            self.has_added_energy = True
            # 追加: 現在のエネルギー表示
            if self.print_actions:
                from collections import Counter
                attached = getattr(top, "attached_energies", [])
                counts = Counter(getattr(card, "name", str(card)) for card in attached)
                attached_str = ", ".join(f"{name}×{count}" if count > 1 else f"{name}" for name, count in counts.items())
                self.log_print(f"{top.name} の現在のエネルギー: [{attached_str}]")

    def use_item(self, card):
        used = True
        if hasattr(card, 'use'):
            used = card.use(self)
            if used is False:
                self.log_print(f"{card.name}は無効だったので手札に残します")
                return False
        else:
            self.log_print(f"{card.name}（グッズ）の効果は未実装です")
            return False  # 未実装も手札に残す

        # 正常に使えた場合のみ手札からトラッシュ
        if card in self.hand:
            self.hand.remove(card)
            self.discard_pile.append(card)
        return True

    def use_item_by_id(self, card_id):
        card = Player.find_by_id(self.hand, card_id)
        if not card:
            self.log_print("手札にそのグッズがありません")
            return False
        return self.use_item(card)

    def use_supporter(self, card):
        # ─────────────────────────────────────────────────────────────
        # 【設計メモ】
        # ・サポーターの効果処理と「手札→トラッシュへの移動」は“各サポータークラスの use() 内”
        #   で行うのが本プロジェクトの方針。
        #   ここ(Player側)ではフラグ更新とログのみを行い、手札/トラッシュの操作はしない。
        #   └ 成功時:  サポーター側の use(self) が自身を手札から取り除き、トラッシュへ送る。
        #   └ 失敗時:  use(self) が False を返し、カードは手札に残る。
        # ・例外的に“自動でトラッシュしない”挙動をしたいサポーターも、各クラスの use() で明示的に制御する。
        # ・この方針により、カードごとの微細な効果解決順や追加コスト等をクラス内に閉じ込める。
        # ─────────────────────────────────────────────────────────────
        # 先攻1ターン目はサポーター禁止（公式ルール）
        m = getattr(self, "match", None)
        if m and getattr(m, "turn", 0) == 0 and getattr(m, "current_player", None) == self.name:
            self.log_print(f"{self.name} は先攻1ターン目のためサポーターは使えません")
            return False

        used = True
        if hasattr(card, 'use'):
            used = card.use(self)
            if used is False:
                self.log_print(f"{card.name}は無効だったので手札に残します")
                return False
        self.log_print(f"{self.name} は {card.name}（サポーター）を使った")
        self.has_used_supporter = True
        return True


    def _current_stadium(self):
        return self.active_stadium or (self.opponent.active_stadium if self.opponent else None)

    def use_stadium(self, card):
        """
        スタジアムカードを使用する
        - 自分の番に1枚しか使えない
        - 既存のスタジアムがある場合は、持ち主のトラッシュに置く
        - 同じ名前のスタジアムは出せない
        """
        # 既にスタジアムを使用済みかチェック
        if self.has_used_stadium:
            self.log_print(f"{self.name} は既にスタジアムを使用済みです")
            return False
        
        current = self._current_stadium()
        if current:
            if current.name == card.name:
                self.log_print(f"既に同名のスタジアム「{card.name}」が場に出ています")
                return False
            # ★ 既存スタジアムをオーナーのトラッシュへ
            owner = getattr(current, 'owner', None)
            if owner is None:
                owner = self if self.active_stadium is current else (self.opponent if self.opponent else self)
            owner.discard_pile.append(current)
            # 双方の参照をクリア
            if self.opponent and self.opponent.active_stadium is current:
                self.opponent.active_stadium = None
            if self.active_stadium is current:
                self.active_stadium = None
            if self.print_actions:
                self.log_print(f"既存のスタジアム「{current.name}」を{owner.name}のトラッシュに置きました")

        # 新規スタジアムを場へ
        if hasattr(card, 'use') and card.use(self) is False:
            self.log_print(f"{card.name}は無効だったので手札に残します")
            return False

        self.active_stadium = card
        card.owner = self
        if card in self.hand:
            self.hand.remove(card)
        self.has_used_stadium = True
        if self.print_actions:
            self.log_print(f"{self.name} は {card.name}（スタジアム）を場に出しました")
        return True

    def attach_tool(self, card, target):
        # 1) 事前バリデーション（副作用なし）
        if hasattr(card, "can_attach_to") and not card.can_attach_to(target):
            self.log_print(f"{getattr(card, 'name','どうぐ')} は {getattr(target,'name','ポケモン')} に付けられません")
            return False

        # 2) target.tools を用意
        if not hasattr(target, "tools") or target.tools is None:
            target.tools = []

        # 3) 冪等ガード：同一どうぐが既に付与済みなら何もしない
        if any((t is card) or (getattr(t, "id", None) == getattr(card, "id", None)) for t in target.tools):
            # 既に付いている（再入）ので何もしない
            self.log_print(f"{target.name} には既に {card.name} が付いています（冪等）")
            # 手札に同じ物が残っていれば不整合なので念のため除去（必要なければ削ってOK）
            if card in self.hand:
                self.hand.remove(card)
            return True

        # 4) 1体につき1枚制限
        if target.tools:
            self.log_print(f"{target.name} には既にどうぐが付いているので {card.name} は付けられません")
            return False

        # 5) 実付与（ここだけが唯一の付与ポイント）
        target.tools.append(card)
        if card in self.hand:
            self.hand.remove(card)
        self.has_used_tool = True
        self.log_print(f"{self.name} は {card.name}（どうぐ）を {target.name} に付けた")

        # 6) 付与後フック（副作用はここで）
        if hasattr(card, "on_attached"):
            try:
                card.on_attached(self, target)
            except Exception as e:
                self.log_print(f"[warn] on_attached で例外: {e}")

        return True

    
    def attach_tool_by_id(self, tool_id, target_id) -> bool:
        """IDでどうぐをポケモンに付ける。成功時 True を返す。"""
        tool_card = Player.find_by_id(self.hand, tool_id)
        flat_board = [card for stack in self.active_card_and_bench if stack for card in stack]
        target_card = Player.find_by_id(flat_board, target_id)

        if tool_card is None:
            self.log_print("どうぐカードが見つかりません")
            return False
        if target_card is None:
            self.log_print("対象ポケモンが見つかりません")
            return False

        # 以降の検証・付与・副作用は attach_tool() に一本化
        return self.attach_tool(tool_card, target_card)

    @staticmethod
    def _eligible_to_evolve_now(player: "Player", top: "Card") -> bool:
        """
        公式ルールに沿った進化可否（この瞬間）を判定する安全ガード。
        - 自分の最初の番（先攻: turn=0 / 後攻: turn=1）は進化禁止
        - そのターンに場に出た個体（entered_turn == 現在ターン）は進化禁止
        - カード側の can_evolve フラグも尊重
        """
        m = getattr(player, "match", None)
        turn = getattr(m, "turn", 0)

        # EX/進化先なしは常に不可（設定に応じて）
        try:
            enforce_ex0 = bool(getattr(player, "enforce_ex_no_evolve", getattr(m, "enforce_ex_no_evolve", False)))
        except Exception:
            enforce_ex0 = False
        if enforce_ex0 and bool(getattr(top, "is_ex", False)):
            return False
        evols = getattr(top, "evolves_to", [])
        if not evols:
            return False

        # 自分の最初の番は禁止（先攻:0 / 後攻:1）
        if turn in (0, 1):
            return False

        # 場に出たばかりは禁止（entered_turn が未設定でも安全側に倒す）
        entered = getattr(top, "entered_turn", None)
        if entered is None or entered == turn:
            return False

        return True


    @staticmethod
    def evolve_and_remove_from_hand(
        player: "Player", stack_id: int, evolution_card_id: uuid.UUID
    ) -> None:
        # stack_idはactive_card_and_benchのインデックス
        try:
            stack = player.active_card_and_bench[stack_id]
            evolution_card = Player.find_by_id(player.hand, evolution_card_id)
            if not evolution_card:
                raise ValueError("Evolution card not found in hand.")

            # 進化前のポケモン（現在のトップ）
            pre_evolution = stack[-1]

            # ★ 最終ガード：この瞬間に進化できるかを一元判定
            if not Player._eligible_to_evolve_now(player, pre_evolution):
                if hasattr(player, "log_print"):
                    player.log_print("このターンは進化できません（最初の番/出したばかり/再進化防止）")
                return

            # ★ 系統チェック：evolves_from が一致しない進化は不可
            if getattr(evolution_card, "evolves_from", None) != getattr(pre_evolution, "card_enum", None):
                raise ValueError("この組み合わせでは進化できません")

            # 進化カードを山の上に重ねる
            stack.append(evolution_card)
            player.hand.remove(evolution_card)

            # 進化後のポケモン（新しいトップ）
            evolved = stack[-1]

            # ★ 進化前の在場ターンを引き継ぐ（将来の進化可否のため）
            evolved.entered_turn = getattr(pre_evolution, "entered_turn", getattr(player.match, "turn", 0))

            # 1. エネルギーを“移す”
            if hasattr(pre_evolution, "attached_energies") and pre_evolution.attached_energies:
                evolved.attached_energies = pre_evolution.attached_energies
                pre_evolution.attached_energies = []
                player.log_print(f"エネルギーを引き継ぎました: {[e.name for e in evolved.attached_energies]}")

            # 2. どうぐを“移す”
            if hasattr(pre_evolution, "tools") and pre_evolution.tools:
                tools = list(pre_evolution.tools)

                # ★追加: 進化前ポケモンから一旦デタッチ（副作用の巻き戻し）
                for tool in tools:
                    if hasattr(tool, "on_detached"):
                        try:
                            tool.on_detached(player, pre_evolution)  # (player, pokemon)
                        except Exception:
                            pass

                # 実体としては同じどうぐを進化後へ載せ替え
                evolved.tools = tools
                try:
                    delattr(pre_evolution, "tools")
                except Exception:
                    pre_evolution.tools = []

                # ★追加: 進化後ポケモンに再アタッチ（副作用を再適用）
                for tool in tools:
                    if hasattr(tool, "on_attached"):
                        try:
                            tool.on_attached(player, evolved)  # (player, pokemon)
                        except Exception:
                            pass

                player.log_print(f"どうぐを引き継ぎました: {[t.name for t in evolved.tools]}")

            # 3. ダメカンを“移す”
            if hasattr(pre_evolution, "damage_counters"):
                evolved.damage_counters = getattr(pre_evolution, "damage_counters", 0)
                pre_evolution.damage_counters = 0
                if getattr(evolved, "max_hp", None) is not None:
                    evolved.hp = evolved.max_hp - (evolved.damage_counters * 10)
                player.log_print(f"ダメカンを引き継ぎました: {evolved.damage_counters}個")

            # 4. 状態異常を“移す”
            if hasattr(pre_evolution, "conditions") and pre_evolution.conditions:
                evolved.conditions = pre_evolution.conditions
                pre_evolution.conditions = []
                player.log_print(f"状態異常を引き継ぎました: {[c.__class__.__name__ for c in evolved.conditions]}")

            player.log_print(f"進化: {pre_evolution.name} → {evolution_card.name}")

            # 進化直後は進化不可にする（次のターンまで再進化できない）
            evolved.can_evolve = False

        except IndexError:
            raise ValueError("進化元ポケモンが見つかりません")


    @staticmethod
    def add_card_to_bench(player: "Player", card_id: "uuid.UUID | str", turn: int) -> bool:
        """
        手札の card_id（UUID/文字列）に一致するカードをベンチに出す。
        ベンチが満杯（5体）なら False を返す。成功時 True。
        """
        import uuid as _uuid

        # UUID / 文字列を問わず比較できるよう正規化関数
        def _norm(x):
            try:
                if isinstance(x, _uuid.UUID):
                    return str(x)
            except Exception:
                pass
            return str(x)

        # 手札から対象カードを検索
        target_norm = _norm(card_id)
        card = None
        for c in player.hand:
            cid = getattr(c, "id", None) or getattr(c, "card_id", None)
            if cid is not None and _norm(cid) == target_norm:
                card = c
                break

        if card is None:
            # 以前は ValueError を投げていたが、落とさずに False を返す
            if getattr(player, 'print_actions', False):
                player.log_print("WARN: add_card_to_bench: Card not found in 手札")
            return False

        # ベンチ上限チェック（必要ならスタジアム等で 8 へ拡張するロジックに差し替え）
        BENCH_MAX = 5
        if len(player.bench) >= BENCH_MAX:
            if getattr(player, 'print_actions', False):
                player.log_print("WARN: add_card_to_bench: Bench is full")
            return False

        # ベンチに出す（進化スタックはリスト）
        card.entered_turn = turn  # このターンに場に出た印を残す（当ターン進化禁止用）
        card.can_evolve = False
        # ★ 追加：場に出した直後は特性未使用に統一
        try:
            setattr(card, "has_used_ability", False)
        except Exception:
            pass
        player.bench.append([card])
        player.hand.remove(card)

        # ログ（任意）
        if getattr(player, 'print_actions', False):
            player.log_print("現在のベンチ:")
            from collections import Counter
            for idx, stack in enumerate(player.bench):
                if isinstance(stack, list) and stack:
                    top = stack[-1]
                    attached = getattr(top, "attached_energies", [])
                    counts = Counter(getattr(ec, "name", str(ec)) for ec in attached)
                    attached_str = ", ".join(f"{n}×{cnt}" if cnt > 1 else f"{n}" for n, cnt in counts.items())
                    player.log_print(f"\t{idx + 1}: {top.name} (HP: {player._display_hp(getattr(top,'hp',0))}) [{attached_str}]")

        # 追加行動可（Action.can_continue_turn=True が既定だが、明示で True にしても可）
        return True


    @staticmethod
    def retreat(player: "Player") -> None:
        if player.active_card is None:
            raise ValueError("No バトル場 to retreat")
        # 追加: ねむり/まひ なら“にげる不可”
        reason = Player._cannot_retreat_reason(player.active_card[-1])
        if reason is not None:
            if player.print_actions:
                player.log_print(f"{player.active_card[-1].name} は『{reason}』のため、このターンはにげられません")
            return  # Action 側の can_continue_turn 設定に任せる

        effective_retreat_cost = player.active_card[-1].get_effective_retreat_cost()
        if effective_retreat_cost is None or player.active_card[-1].get_total_energy() < effective_retreat_cost:
            raise ValueError(f"Not enough energy to retreat {player.active_card[-1].name}")

        # ▼支払ったエネルギーの“内訳”を受け取る（list を返す仕様に）
        spent_energies = player.active_card[-1].remove_retreat_cost_energy(player) or []

        # ▼コンソールに内訳ログを出す（UX向上）
        if player.print_actions:
            spent_str = ", ".join(getattr(c, "name", str(c)) for c in spent_energies) or "なし"
            player.log_print(f"[逃げる] コスト支払い: {spent_str}")

        # （学習ログにも残したい場合）
        if hasattr(player, "last_action") and player.last_action:
            ex = getattr(player.last_action, "extra", {}) if isinstance(getattr(player.last_action, "extra", {}), dict) else {}
            ex.update({
                "retreat_spent_energy_ids": [int(getattr(c, "card_enum", getattr(c, "enum", None)).value[0]) 
                                             if getattr(getattr(c, "card_enum", getattr(c, "enum", None)), "value", None) else None
                                             for c in spent_energies]
            })
            player.last_action.extra = ex

        if not player.bench or len(player.bench) == 0:
            raise ValueError("No cards in the bench to switch with")

        # --- ベンチ選択 ---
        if not player.is_bot:
            print("ベンチのどのポケモンをバトル場に出しますか？")
            for i, stack in enumerate(player.bench, start=1):
                top = stack[-1]
                print(f"{i}: {top.name} (HP: {player._display_hp(getattr(top, 'hp', 0))})")
            while True:
                try:
                    idx_raw = int(input("番号を選んでください: "))
                    if 1 <= idx_raw <= len(player.bench):
                        idx = idx_raw - 1
                        break
                except Exception:
                    pass
                print("無効な入力です。")
        else:
            idx = random.randint(0, len(player.bench) - 1)

        new_active_card = player.bench.pop(idx)   # ベンチから新しいアクティブカードをpop
        old_active_card = player.active_card
        player.active_card = new_active_card
        player.bench.append(old_active_card)      # もともとのバトル場をベンチに追加

        # --- ログ出力（AI学習向けのsubstepsを残す） ---
        if hasattr(player, "last_action") and player.last_action:
            # 1始まりに変換
            selected_bench_idx = idx + 1
            # 逃げる行動：[5, 0, 0, 0, 0]
            # 選択したベンチのインデックス：[5, selected_bench_idx, 0, 0, 0]
            substeps = [
                {"phase": "retreat.action", "macro": [5, 0, 0, 0, 0]},
                {"phase": "retreat.select", "macro": [5, 0, 0, 0, 0], "selected_bench_idx": selected_bench_idx, "action_vec": [5, selected_bench_idx, 0, 0, 0]}
            ]
            # extraにマージ（既存のextraがdictでなければ新規）
            ex = getattr(player.last_action, "extra", {}) if hasattr(player.last_action, "extra") else {}
            if not isinstance(ex, dict):
                ex = {}
            ex.update({
                "selected_bench_idx": selected_bench_idx,
                "retreat_action_vec": [5, selected_bench_idx, 0, 0, 0],
                "substeps": substeps
            })
            player.last_action.extra = ex

        if player.print_actions:
            player.log_print(
                f"{old_active_card[-1].name} をベンチに戻し、{player.active_card[-1].name} をバトル場に出した！"
            )
        player.has_retreat_this_turn = True  # 逃げたフラグ


    def move_active_card_to_bench(self) -> None:
        if self.active_card is None:
            raise ValueError("No バトル場 to move to bench")
        if not self.bench:
            raise ValueError("No cards in the bench to switch with")

        old_active_card = self.active_card
        self.bench.append(self.active_card)

        # Ensure the new バトル場 is different from the old one
        eligible_cards = [stack for stack in self.bench if stack != old_active_card]
        if eligible_cards:
            self.active_card = random.choice(eligible_cards)
            self.bench.remove(self.active_card)
            if self.print_actions:
                self.log_print(
                    f"{old_active_card[-1].name} retreated, {self.active_card[-1].name} set as active"
                )
        else:
            raise ValueError("No eligible cards in bench to set as active")

    @staticmethod
    def set_active_card_from_hand(player: "Player", card_id: uuid.UUID, turn: int) -> None:
        card = Player.find_by_id(player.hand, card_id)
        if card and card in player.hand:
            if player.print_actions:
                player.log_print(f"Setting バトル場 from 手札 to {card.name}")
            card.entered_turn = turn  # ★ 場に出したターンを記録
            card.can_evolve = False
            # ★ 追加：場に出した直後は特性未使用に統一
            try:
                setattr(card, "has_used_ability", False)
            except Exception:
                pass
            player.active_card = [card]
            player.hand.remove(card)
        else:
            raise ValueError(f"Card not found in 手札 or invalid")

    def set_active_card_from_bench(self, card: List[Card]) -> None:
        if card not in self.bench:
            raise ValueError(f"Card {card[-1].name if isinstance(card, list) else card.name} not in bench")

        if self.print_actions:
            self.log_print(f"Setting バトル場 from bench to {card[-1].name if isinstance(card, list) else card.name}")
        self.active_card = card
        self.bench.remove(card)

    def setup_turn(self, match: "Match", viewing_player=None, log_mode=False):
        # ドロー処理
        drawn_card = self.deck.draw_card() if self.deck.cards else None
        if drawn_card is not None:
            self.hand.append(drawn_card)
            if not log_mode:
                if viewing_player is not None and viewing_player == self:
                    if not self.is_bot:
                        print(f"{self.name} は山札から「{drawn_card}」を引きました（残り山札は{len(self.deck.cards)}枚）")
                    else:
                        self.log_print(f"{self.name} は山札から「{drawn_card}」を引きました（残り山札は{len(self.deck.cards)}枚）")
                else:
                    if not self.is_bot:
                        print(f"{self.name} はカードをドローしました（残り山札は{len(self.deck.cards)}枚）")
                    else:
                        self.log_print(f"{self.name} はカードをドローしました（残り山札は{len(self.deck.cards)}枚）")
        else:
            # ★ここでのみ敗北を確定させる（ターン開始時のドローができない＝敗北）
            if self.match:
                self.match.game_over = True
                self.match.winner = self.opponent.name if self.opponent else None
                setattr(self.match, "_end_reason", "deck_out")  # ← 追加：終局理由を明示
            self.can_continue = False  # ← 追加：以降の処理を止める
            self.log_print(f"{self.name} はターン開始時に山札からドローできませんでした。{self.opponent.name if self.opponent else ''} の勝利です！")  # ← 常にログ出力
            try:
                # 終局レコードをこの場で出す（deck_out）— 二重出力は logger 側フラグで抑止
                if hasattr(self, "logger") and self.logger and not getattr(self.logger, "_terminal_logged", False):
                    self.logger.log_terminal_step(reason="deck_out")
            except Exception:
                pass

            # ★ 追加：deck_out でもポリシーサマリを1回だけ出す（PhaseD-Q 等）
            try:
                m = self.match
                if m is not None and not getattr(m, "_policy_summary_logged", False):
                    setattr(m, "_policy_summary_logged", True)
                    pol = getattr(self, "policy", None)
                    if pol is not None:
                        for fn_name in ("log_summary", "print_summary", "dump_summary", "summary", "on_game_end", "on_match_end"):
                            fn = getattr(pol, fn_name, None)
                            if callable(fn):
                                try:
                                    fn(player=self, match=m, reason=getattr(m, "_end_reason", None))
                                except TypeError:
                                    try:
                                        fn(self, m)
                                    except TypeError:
                                        try:
                                            fn(m)
                                        except TypeError:
                                            fn()
                                break
            except Exception:
                pass

            return  # 以降のターン処理をスキップ

    # --- 追加: ねむり/まひ などの“にげる不可”判定 ---
    @staticmethod
    def _cannot_retreat_reason(card) -> Optional[str]:
        """
        バトル場ポケモンが「ねむり / まひ」等で“にげる”できないなら、その理由文字列を返す。
        できるなら None を返す。
        """
        conds = getattr(card, "conditions", []) or []
        names = set()
        for c in conds:
            # クラス名と文字列表記の両方に対応（ログ実装に合わせる）
            if hasattr(c, "__class__"):
                names.add(c.__class__.__name__)
            if isinstance(c, str):
                names.add(c)

        # ねむり
        if ("Sleep" in names) or ("ねむり" in names):
            return "ねむり"
        # まひ（英語表記の揺れも吸収）
        if ("Paralysis" in names) or ("Paralyzed" in names) or ("まひ" in names):
            return "まひ"

        return None


    @staticmethod
    def find_by_id(objects: List[Any], target_id: "uuid.UUID | str | int | None") -> Optional[Any]:
        # 文字列 payload（JSON）や UUID オブジェクトのどちらでも照合できるように正規化
        def _norm(x: Any) -> str:
            try:
                import uuid as _uuid
                if isinstance(x, _uuid.UUID):
                    return str(x)
            except Exception:
                pass
            return str(x)

        if target_id is None:
            return None
        tnorm = _norm(target_id)

        for obj in objects:
            # ベンチなどの進化スタック（list）対応：一致したら「スタック全体」を返す
            if isinstance(obj, list):
                for card in obj:
                    cid = getattr(card, "id", None) or getattr(card, "card_id", None)
                    if cid is not None and _norm(cid) == tnorm:
                        return obj
            else:
                cid = getattr(obj, "id", None) or getattr(obj, "card_id", None)
                if cid is not None and _norm(cid) == tnorm:
                    return obj
        return None


    def serialize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "手札": [card.serialize() for card in self.hand],
            # 進化山（List[List[Card]]）は各山ごとにリストでserialize
            "bench": [[card.serialize() for card in stack] for stack in self.bench],
            # バトル場も進化山（List[Card]）としてserialize
            "active_card": [card.serialize() for card in self.active_card] if self.active_card else None,
            "deck": [card.serialize() for card in self.deck.cards],
            "discard_pile": [card.serialize() for card in self.discard_pile],
            "points": self.points,
            "has_used_trainer": self.has_used_trainer,
            "last_action": getattr(self, "last_action", None),   # 例: [3, 50001, 10012, 10034, 10002]
        }

    def __repr__(self) -> str:
        return f"Player({self.name}, 手札: {len(self.hand)} cards, バトル場: {self.active_card}, Bench: {len(self.bench)} cards, Points: {self.points})"

    def get_state_dict(self):
        """
        print_public_snapshot() が期待する形:
        {
            "turn": int,                                 # 0始まり（表示時は +1 して出す）
            "current_player": 0 or 1,                    # 絶対サイド: 0=先攻(starting), 1=後攻(second)
            "current_player_name": str,                  # 現在手番プレイヤーの名前
            "active_stadium": 任意,
            "me":  {...},                                # 自分（人なら手札内容を公開）
            "opp": {...},                                # 相手（手札は枚数のみ）
        }
        """
        m = getattr(self, "match", None)

        def _serialize_top_of_stack(stack):
            if not stack:
                return None
            top = stack[-1]
            if hasattr(top, "serialize_public"):
                try:
                    return top.serialize_public()
                except Exception:
                    pass
            return {
                "name": getattr(top, "name", None),
                "hp":   getattr(top, "hp", None),
            }

        def _pub(p, full_hand: bool):
            # active
            active_pub = _serialize_top_of_stack(getattr(p, "active_card", []))

            # bench
            bench_pub = []
            for st in getattr(p, "bench", []):
                pub = _serialize_top_of_stack(st)
                if pub is not None:
                    bench_pub.append(pub)

            # discard
            discard_src = getattr(p, "discard_pile", [])
            discard_pub = []
            for c in discard_src:
                if hasattr(c, "serialize_public"):
                    try:
                        discard_pub.append(c.serialize_public())
                        continue
                    except Exception:
                        pass
                discard_pub.append(getattr(c, "name", None))

            deck_obj = getattr(p, "deck", None)
            deck_cards = getattr(deck_obj, "cards", []) if deck_obj is not None else []
            base = {
                "active_pokemon": active_pub,
                "bench_pokemon": bench_pub,
                "discard_pile":  discard_pub,
                "deck_count":     len(deck_cards),
                "prize_count":    len(getattr(p, "prize_cards", [])),
            }
            if full_hand:
                hand_pub = []
                for c in getattr(p, "hand", []):
                    if hasattr(c, "serialize_public"):
                        try:
                            hand_pub.append(c.serialize_public())
                            continue
                        except Exception:
                            pass
                    hand_pub.append(getattr(c, "name", None))
                base["hand"] = hand_pub
            else:
                base["hand_count"] = len(getattr(p, "hand", []))
            return base

        # 人側の手札は内容を公開、CPU 側は枚数のみ
        me  = _pub(self, full_hand=not getattr(self, "is_bot", True))
        opp = _pub(getattr(self, "opponent", None), full_hand=False) if getattr(self, "opponent", None) else {}

        # --- 手番（絶対サイド）と名前を付与 ---
        turn_idx = getattr(m, "turn", 0) if m is not None else 0  # 0始まり
        sp = getattr(m, "starting_player", None) if m is not None else None
        sec = getattr(m, "second_player", None) if m is not None else None
        current_side = 0 if (turn_idx % 2 == 0) else 1  # 0=先攻, 1=後攻（絶対）
        current_name = (sp.name if current_side == 0 else sec.name) if (sp and sec) else None

        # スタジアム（自分→相手の順で存在する方）
        st = getattr(self, "active_stadium", None)
        if not st and getattr(self, "opponent", None):
            st = getattr(self.opponent, "active_stadium", None)
        if st and hasattr(st, "serialize_public"):
            try:
                st_pub = st.serialize_public()
            except Exception:
                st_pub = getattr(st, "name", None)
        else:
            st_pub = getattr(st, "name", None) if st else None

        return {
            "turn": turn_idx,
            "current_player": current_side,        # 絶対サイド
            "current_player_name": current_name,   # 表示はこれを使う
            "active_stadium": st_pub,
            "me": me,
            "opp": opp,
            "game_id": getattr(m, "game_id", None),
        }

    # --- 手札を山札の下へ戻してシャッフル ---
    def move_hand_to_deck_bottom_and_shuffle(self) -> None:
        # 手札のカードをすべて山札末尾へ
        # hand は List[Card]、deck.cards は List[Card] を想定
        self.deck.cards.extend(self.hand)
        self.hand.clear()
        random.shuffle(self.deck.cards)

    # --- n 枚ドロー ---
    def draw_cards(self, n: int, match=None) -> None:
        """
        効果による複数ドロー用。
        - 山札が尽きてもこの場では敗北しない（公式同様、次の自分のターン開始ドローで引けないと敗北）。
        - 引ける分だけ引いて、引けなければ打ち切る。
        """
        for _ in range(n):
            card = None
            try:
                card = self.deck.draw_card()
            except ValueError:
                # 山札が空：この場では敗北にしない
                card = None

            if card is None:
                # これ以上引けないので打ち切り（敗北判定はターン開始時に行う）
                if self.print_actions:
                    self.log_print(f"{self.name} はこれ以上ドローできません（山札切れ）。敗北判定は次の自分のターン開始時に行われます。")
                break

            # 通常通り手札へ
            self.hand.append(card)

    # --- AI用ベンチ選好スコア（低HP・EX・エネルギー持ち・進化を優先）---
    def _opponent_bench_score(self, card) -> float:
        hp = getattr(card, "hp", 0) or 0
        is_ex = 1 if getattr(card, "is_ex", False) else 0
        energy_n = len(getattr(card, "attached_energies", []) or [])
        stage_bonus = 15 if not getattr(card, "is_basic", False) else 0
        # 大物(=EX)・育ってる(=エネ多い/進化)を優先し、残りHPが低いほど優先
        # タイブレーク用の微小ノイズを足す（MCTSシミュレーション中は決定論にする）
        import random
        m = getattr(self, "match", None)
        noise = 0.0 if bool(getattr(m, "_is_mcts_simulation", False)) else (random.random() * 0.01)
        return (is_ex * 200) + (energy_n * 12) + stage_bonus - hp + noise

    # --- ベンチインデックス選択（ポリシーフック→既定ヒューリスティック）---
    def _select_opponent_bench_index(self, stacks: List[List["Card"]]) -> int:
        # 1) 外部から差し替え可能なフック（例: 学習済みモデル）
        #    使い方: player.select_opponent_bench_index_policy = lambda self, stacks: 0
        if hasattr(self, "select_opponent_bench_index_policy") and callable(self.select_opponent_bench_index_policy):
            try:
                idx = int(self.select_opponent_bench_index_policy(self, stacks))
                if 0 <= idx < len(stacks):
                    return idx
            except Exception:
                pass
        # 2) 既定：ヒューリスティック
        scores = [self._opponent_bench_score(s[-1]) for s in stacks]
        return max(range(len(stacks)), key=lambda i: scores[i])


    # --- 相手ベンチから 1 匹選択（人間 or AI） ---
    def choose_opponent_bench_card(self) -> Card:
        # ① 相手がいない／ベンチが空 → 例外
        if self.opponent is None or not self.opponent.bench:
            raise ValueError("相手のベンチが空です")

        # ② ベンチが1体だけなら自動選択
        if len(self.opponent.bench) == 1:
            chosen = self.opponent.bench[0][-1]
            if self.print_actions:
                self.log_print(f"相手ベンチが1体のため自動選択: {chosen.name}")
            return chosen

        # ③ 人間 or AI で分岐
        if not self.is_bot:
            # 既存の人間入力ルート
            self.log_print("相手のベンチ:")
            for i, stack in enumerate(self.opponent.bench, 1):
                self.log_print(f"{i}: {stack[-1].name}")
            while True:
                try:
                    idx = int(input("番号: ")) - 1
                    if 0 <= idx < len(self.opponent.bench):
                        return self.opponent.bench[idx][-1]
                except Exception:
                    pass
                self.log_print("無効な入力です。")
        else:
            # --- AI：ヒューリスティック（またはフック）で選択 ---
            # 空でないスタックだけに絞りつつ、元のベンチ番号との対応表を保持
            bench_map = [i for i, s in enumerate(self.opponent.bench) if s]
            stacks    = [self.opponent.bench[i] for i in bench_map]
            ai_local_idx = self._select_opponent_bench_index(stacks)  # 0..len(stacks)-1
            orig_idx = bench_map[ai_local_idx]
            chosen_stack = self.opponent.bench[orig_idx]
            chosen_card  = chosen_stack[-1]

            if self.print_actions:
                self.log_print(f"AIは相手のベンチ{orig_idx + 1}（{chosen_card.name}）を選択")

            # 学習ログ用：直前アクションに選択結果を埋め込む（任意）
            if hasattr(self, "last_action") and getattr(self, "last_action", None):
                ex = getattr(self.last_action, "extra", {})
                if not isinstance(ex, dict):
                    ex = {}
                ex.update({
                    "selected_opponent_bench_idx": orig_idx + 1,  # 1始まりで保持
                    "selected_opponent_bench_name": getattr(chosen_card, "name", None),
                })
                self.last_action.extra = ex

            return chosen_card


    def move_hand_to_deck_and_shuffle(self) -> None:
        """
        手札のカードをすべて山札に戻し、山札をシャッフルする
        """
        self.deck.cards.extend(self.hand)
        self.hand.clear()
        self.deck.shuffle()

    def print_player_state_after(self):
        import json
        # [STATE_OBJ_OPPONENT_AFTER]として相手の状態を出力
        if not self.match or not self.opponent:
            return
        opp = self.opponent
        if opp.active_card:
            top = opp.active_card[-1]
            attached_energies = getattr(top, "attached_energies", [])
            energies = [e.name for e in attached_energies] if attached_energies else []
            conditions = [c.__class__.__name__ for c in getattr(top, "conditions", [])]
            opp_active_pokemon_obj = {
                "name": top.name,
                "hp": max(0, top.hp if top.hp is not None else 0),
                "energies": energies if energies else [],
                "conditions": conditions
            }
        else:
            opp_active_pokemon_obj = {"name": None, "hp": None, "energies": [], "conditions": []}
        opp_bench_pokemon = []
        for idx in range(5):
            if idx < len(opp.bench) and opp.bench[idx]:
                top = opp.bench[idx][-1]
                attached_energies = getattr(top, "attached_energies", [])
                energies = [e.name for e in attached_energies] if attached_energies else []
                opp_bench_pokemon.append({"name": top.name, "hp": max(0, top.hp if top.hp is not None else 0), "energies": energies if energies else []})
            else:
                opp_bench_pokemon.append(None)
        opp_log_obj = {
            "game_id": getattr(self.match, 'game_id', None),
            "turn": getattr(self.match, 'turn', None),
            "player": opp.name,
            "hand_count": len(opp.hand),
            "bench_count": len(opp.bench),
            "prize_count": len(opp.prize_cards),
            "deck_count": len(opp.deck.cards),
            "active_pokemon": opp_active_pokemon_obj,
            "bench_pokemon": opp_bench_pokemon,
            "discard_pile": [c.name for c in opp.discard_pile],
            "discard_pile_count": len(opp.discard_pile),
            "done": self.compute_reward()[1]
        }
        opp_json_output = "[STATE_OBJ_OPPONENT_AFTER] " + json.dumps(opp_log_obj, ensure_ascii=False)
        # print(opp_json_output, flush=True)  # ← 画面出力を抑制
        # ログファイルには出力しない

    def choose_action(self, actions, print_actions=True):
        if print_actions:
            for i, action in enumerate(actions, start=1):
                print(f"{i}: {action}")
        while True:
            try:
                raw = int(input("アクション番号を選んでください: "))
                if 1 <= raw <= len(actions):
                    return raw - 1  # 内部は0始まりで扱う
            except Exception:
                pass
            print(f"無効な入力です。1〜{len(actions)}の番号で入力してください。")

    def select_action(self, state_dict, actions):
        if not self.is_bot:
            # 人間は対話入力
            return self.choose_action(actions, print_actions=True)

        """
        actions: 5整数ベクトルの合法手リスト
        返り値: 選んだ action のインデックス
        """
        # ★ MCTS シミュレーション中は、重いオンライン混合を回さず deterministic に先頭を返す
        try:
            m = getattr(self, "match", None)
            if bool(getattr(m, "_is_mcts_simulation", False)):
                return 0 if actions else 0
        except Exception:
            pass

        # 1) オンライン混合ポリシーがあればそちらを最優先
        if hasattr(self, "policy") and self.policy is not None:
            # (a) 専用メソッド select_action_index_online があれば、オンライン混合として呼び出す
            try:
                if hasattr(self.policy, "select_action_index_online"):
                    sd2 = state_dict
                    try:
                        if isinstance(state_dict, dict):
                            sd2 = dict(state_dict)
                    except Exception:
                        sd2 = state_dict

                    try:
                        if isinstance(sd2, dict):
                            sd2.setdefault("player_name", getattr(self, "name", None))
                            try:
                                sd2.setdefault("t", int(getattr(getattr(self, "match", None), "turn", 0)))
                            except Exception:
                                pass
                            sd2.setdefault("_match", getattr(self, "match", None))
                    except Exception:
                        pass

                    la_19d = None
                    try:
                        if isinstance(actions, (list, tuple)) and actions and hasattr(actions[0], "serialize"):
                            la_19d = [a.serialize(self) for a in actions]
                        elif isinstance(actions, (list, tuple)) and actions and isinstance(actions[0], (list, tuple)):
                            ok = True
                            for _x in actions[0]:
                                if not isinstance(_x, int):
                                    ok = False
                                    break
                            if ok:
                                la_19d = list(actions)
                        elif isinstance(getattr(self, "_last_legal_actions_before", None), list) and self._last_legal_actions_before:
                            la_19d = list(self._last_legal_actions_before)
                        elif isinstance(getattr(getattr(self, "logger", None), "last_legal_actions_before", None), list) and self.logger.last_legal_actions_before:
                            la_19d = list(self.logger.last_legal_actions_before)
                    except Exception:
                        la_19d = None

                    # --- ★追加：正式APIガード（legal_actions_19d は「5要素すべてint」を強制） ---
                    try:
                        def _is_5ints(v):
                            if not isinstance(v, (list, tuple)) or len(v) != 5:
                                return False
                            for _y in v:
                                if isinstance(_y, bool) or (not isinstance(_y, int)):
                                    return False
                            return True

                        if la_19d is not None:
                            fixed = []
                            bad_i = None
                            bad_v = None
                            for i, v in enumerate(la_19d):
                                if _is_5ints(v):
                                    fixed.append([int(x) for x in v])
                                    continue
                                # 形だけ合っていれば int 化を試みる（失敗したら破棄）
                                try:
                                    if isinstance(v, (list, tuple)) and len(v) == 5:
                                        vv = []
                                        ok2 = True
                                        for x in v:
                                            if isinstance(x, bool):
                                                ok2 = False
                                                break
                                            vv.append(int(x))
                                        if ok2 and _is_5ints(vv):
                                            fixed.append(vv)
                                            continue
                                except Exception:
                                    pass
                                bad_i = i
                                bad_v = v
                                break

                            if bad_i is not None:
                                try:
                                    self.log_print(f"[API_VIOLATION] legal_actions_19d invalid at i={bad_i}: {type(bad_v)} {bad_v}")
                                except Exception:
                                    pass
                                la_19d = None
                            else:
                                la_19d = fixed
                    except Exception:
                        la_19d = None
                    # --- ★追加ここまで ---

                    try:
                        if isinstance(sd2, dict):
                            if la_19d is not None:
                                sd2.setdefault("legal_actions_19d", la_19d)
                                sd2.setdefault("legal_actions_vecs", la_19d)
                                sd2.setdefault("legal_actions_vec", la_19d)
                                sd2.setdefault("legal_actions_list", la_19d)
                                sd2.setdefault("la_list", la_19d)
                                sd2.setdefault("legal_actions", la_19d)

                                sd2.setdefault("action_candidates_vec", la_19d)
                                sd2.setdefault("action_candidates_vecs", la_19d)
                                sd2.setdefault("cand_vecs", la_19d)
                            else:
                                sd2.setdefault("legal_actions", list(range(len(actions))))
                    except Exception:
                        pass

                    try:
                        if isinstance(sd2, dict) and isinstance(sd2.get("obs_vec", None), (list, tuple)) and sd2.get("obs_vec", None):
                            _ov = sd2.get("obs_vec", None)
                            _out = []
                            ok = True
                            for _x in _ov:
                                try:
                                    _out.append(float(_x))
                                except Exception:
                                    ok = False
                                    break
                            if ok:
                                sd2["obs_vec"] = _out
                                sd2.setdefault("public_obs_vec", _out)
                                sd2.setdefault("full_obs_vec", _out)
                    except Exception:
                        pass

                    try:
                        if isinstance(sd2, dict) and (not isinstance(sd2.get("obs_vec", None), (list, tuple)) or not sd2.get("obs_vec", None)):
                            enc = None
                            try:
                                enc = getattr(self.policy, "encoder", None) or getattr(self.policy, "state_encoder", None)
                            except Exception:
                                enc = None
                            if enc is None:
                                try:
                                    m = getattr(self, "match", None)
                                    enc = getattr(m, "encoder", None) if m is not None else None
                                except Exception:
                                    enc = None

                            _vec = None
                            if enc is not None:
                                try:
                                    fn = getattr(enc, "encode_state", None)
                                    _vec = fn(sd2) if callable(fn) else None
                                except Exception:
                                    _vec = None

                            if _vec is None:
                                conv = None
                                try:
                                    conv = getattr(self.policy, "converter", None) or getattr(self.policy, "action_converter", None)
                                except Exception:
                                    conv = None
                                if conv is None:
                                    try:
                                        m = getattr(self, "match", None)
                                        conv = getattr(m, "converter", None) or getattr(m, "action_converter", None) if m is not None else None
                                    except Exception:
                                        conv = None
                                if conv is None and isinstance(sd2, dict):
                                    try:
                                        _m2 = sd2.get("_match", None)
                                        conv = getattr(_m2, "converter", None) or getattr(_m2, "action_converter", None) if _m2 is not None else None
                                    except Exception:
                                        conv = None

                                if conv is not None:
                                    try:
                                        fn = getattr(conv, "encode_state", None)
                                        _vec = fn(sd2) if callable(fn) else None
                                    except Exception:
                                        _vec = None
                                    if _vec is None:
                                        try:
                                            fn = getattr(conv, "convert_state", None)
                                            _vec = fn(sd2) if callable(fn) else None
                                        except Exception:
                                            _vec = None
                                    if _vec is None:
                                        try:
                                            fn = getattr(conv, "build_obs", None)
                                            _vec = fn(sd2) if callable(fn) else None
                                        except Exception:
                                            _vec = None

                            try:
                                if hasattr(_vec, "tolist"):
                                    _vec = _vec.tolist()
                            except Exception:
                                pass

                            if isinstance(_vec, dict):
                                for _k in ("obs_vec", "obs", "public_obs_vec", "full_obs_vec", "x", "vec"):
                                    if _k in _vec:
                                        _vec = _vec[_k]
                                        break
                                try:
                                    if hasattr(_vec, "tolist"):
                                        _vec = _vec.tolist()
                                except Exception:
                                    pass

                            if isinstance(_vec, (list, tuple)) and _vec:
                                _out = []
                                ok = True
                                for _x in _vec:
                                    try:
                                        _out.append(float(_x))
                                    except Exception:
                                        ok = False
                                        break
                                if ok:
                                    sd2["obs_vec"] = _out
                                    sd2.setdefault("public_obs_vec", _out)
                                    sd2.setdefault("full_obs_vec", _out)
                    except Exception:
                        pass

                    # ★追加：MCC（hidden information completion）を policy 呼び出し直前に実行
                    try:
                        m = getattr(self, "match", None)
                        if m is not None and bool(getattr(m, "use_mcc", False)) and isinstance(sd2, dict):
                            mcc_sampler = getattr(getattr(m, "reward_shaping", None), "mcc_sampler", None)

                            if callable(mcc_sampler):
                                ctx = {
                                    "match": m,
                                    "player": self,
                                    "allow_peek_me": bool(getattr(m, "mcc_allow_peek_me", True)),
                                    "allow_peek_opp": bool(getattr(m, "mcc_allow_peek_opp", True)),
                                }
                                top_k = getattr(m, "mcc_top_k", None)
                                if top_k is not None:
                                    try:
                                        ctx["top_k"] = int(top_k)
                                    except Exception:
                                        pass

                                try:
                                    sd_for_mcc = dict(sd2)
                                except Exception:
                                    sd_for_mcc = sd2

                                mcc_state = mcc_sampler(sd_for_mcc, ctx)

                                try:
                                    sd2["_mcc"] = mcc_state
                                    if isinstance(mcc_state, dict):
                                        if "me_private" in mcc_state:
                                            sd2["me_private"] = mcc_state.get("me_private")
                                        if "opp_private" in mcc_state:
                                            sd2["opp_private"] = mcc_state.get("opp_private")
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # ★ 追加：DECIDEログはここで“作るだけ”。実際の出力は act_and_regather_actions の直前で行う
                    try:
                        pend = getattr(self, "_pending_decide_lines", None)
                        if isinstance(pend, list):
                            pend.clear()
                            t_now = int(getattr(getattr(self, "match", None), "turn", 0))
                            pend.append(f"[DECIDE_PRE] t={t_now} player={getattr(self, 'name', None)} policy={self.policy.__class__.__name__} n_actions={len(actions)}")
                    except Exception:
                        pass

                    idx = self.policy.select_action_index_online(sd2, actions, player=self)
                    if isinstance(idx, int) and 0 <= idx < len(actions):
                        try:
                            pend = getattr(self, "_pending_decide_lines", None)
                            if isinstance(pend, list):
                                try:
                                    avec = actions[idx].serialize(self) if hasattr(actions[idx], "serialize") else None
                                except Exception:
                                    avec = None
                                pend.append(f"[DECIDE_POST] idx={idx} action_type={getattr(actions[idx], 'action_type', None)} vec={avec}")

                                info = None
                                for _k in ("last_decide_info", "_last_decide_info", "last_decision", "_last_decision", "decide_info"):
                                    if hasattr(self.policy, _k):
                                        info = getattr(self.policy, _k)
                                        break
                                if isinstance(info, dict):
                                    parts = []
                                    for k in ("mode", "source", "mix_prob", "q_w", "pi_w", "used_q", "used_pi", "skip_reason"):
                                        if k in info:
                                            parts.append(f"{k}={info.get(k)}")
                                    if parts:
                                        pend.append("[DECIDE_DIFF] " + " ".join(parts))
                                elif isinstance(info, str) and info:
                                    pend.append("[DECIDE_DIFF] " + info)
                        except Exception:
                            pass
                        return idx
            except Exception:
                # 失敗しても既存ロジックにフォールバック
                pass

            # (b) 従来の select_action_index もそのまま利用（ポリシー側で混合済みでもよい）
            try:
                # policy は「インデックス」を返すようにする
                try:
                    m = getattr(self, "match", None)
                    if m is not None and bool(getattr(m, "use_mcc", False)) and isinstance(state_dict, dict):
                        mcc_sampler = getattr(getattr(m, "reward_shaping", None), "mcc_sampler", None)

                        if callable(mcc_sampler):
                            ctx = {
                                "match": m,
                                "player": self,
                                "allow_peek_me": bool(getattr(m, "mcc_allow_peek_me", True)),
                                "allow_peek_opp": bool(getattr(m, "mcc_allow_peek_opp", True)),
                            }
                            top_k = getattr(m, "mcc_top_k", None)
                            if top_k is not None:
                                try:
                                    ctx["top_k"] = int(top_k)
                                except Exception:
                                    pass

                            try:
                                sd_for_mcc = dict(state_dict)
                            except Exception:
                                sd_for_mcc = state_dict

                            mcc_state = mcc_sampler(sd_for_mcc, ctx)

                            try:
                                state_dict["_mcc"] = mcc_state
                                if isinstance(mcc_state, dict):
                                    if "me_private" in mcc_state:
                                        state_dict["me_private"] = mcc_state.get("me_private")
                                    if "opp_private" in mcc_state:
                                        state_dict["opp_private"] = mcc_state.get("opp_private")
                            except Exception:
                                pass
                except Exception:
                    pass

                idx = self.policy.select_action_index(state_dict, actions, player=self)
                if isinstance(idx, int) and 0 <= idx < len(actions):
                    return idx
            except Exception:
                # 例外時はランダムにフォールバック
                pass

        import random
        return random.randint(0, len(actions) - 1)


    def print_state_opponent(self):
        # 相手の状態を[STATE_OBJ_OPPONENT]として出力
        import json
        if not self.opponent or not self.match:
            return
        # バトル場
        if self.opponent.active_card:
            top = self.opponent.active_card[-1]
            attached_energies = getattr(top, "attached_energies", [])
            energies = [e.name for e in attached_energies] if attached_energies else []
            conditions = [c.__class__.__name__ for c in getattr(top, "conditions", [])]
            opp_active_pokemon_obj = {
                "name": top.name,
                "hp": max(0, top.hp if top.hp is not None else 0),
                "energies": energies if energies else [],
                "conditions": conditions
            }
        else:
            opp_active_pokemon_obj = {"name": None, "hp": None, "energies": [], "conditions": []}
        # ベンチ
        opp_bench_pokemon = []
        for idx in range(5):
            if idx < len(self.opponent.bench) and self.opponent.bench[idx]:
                top = self.opponent.bench[idx][-1]
                attached_energies = getattr(top, "attached_energies", [])
                energies = [e.name for e in attached_energies] if attached_energies else []
                opp_bench_pokemon.append({"name": top.name, "hp": max(0, top.hp if top.hp is not None else 0), "energies": energies if energies else []})
            else:
                opp_bench_pokemon.append(None)
        opp_log_obj = {
            "game_id": getattr(self.match, 'game_id', None) if self.match else None,
            "turn": getattr(self.match, 'turn', 0),
            "player": self.opponent.name,
            "hand_count": len(self.opponent.hand),
            "bench_count": len(self.opponent.bench),
            "prize_count": len(self.opponent.prize_cards),
            "deck_count": len(self.opponent.deck.cards),
            "active_pokemon": opp_active_pokemon_obj,
            "bench_pokemon": opp_bench_pokemon,
            "discard_pile": [c.name for c in self.opponent.discard_pile],
            "discard_pile_count": len(self.opponent.discard_pile),
            "done": self.compute_reward()[1]
        }
        opp_json_output = "[STATE_OBJ_OPPONENT] " + json.dumps(opp_log_obj, ensure_ascii=False)
        if self.match and hasattr(self.match, 'log_file') and self.match.log_file:
            self.log_print(opp_json_output)


    def pokemon_check(self):
        """ターン終了後の状態異常・特殊状態の処理（公式ルール準拠）"""
        if self.active_card is None or not self.active_card:
            return
        top = self.active_card[-1]
        conditions = getattr(top, "conditions", [])
        # 状態異常は複数同時に存在しない仕様（基本：どれか1つだけ）
        # ただし、「どく＋やけど」だけは両立することがある
        # →まずどく・やけどを優先的にチェック
        condition_names = [c.__class__.__name__ if hasattr(c, "__class__") else str(c) for c in conditions]
        # --- 毒処理（ダメカン方式） ---
        if "Poison" in condition_names or "どく" in condition_names:
            add_damage_counters(top, 1)  # ダメカン1個載せる（10ダメージ分）
            self.log_print(f"{self.name}のバトル場は『どく』でダメカン1個（残りHP:{self._display_hp(getattr(top, 'hp', 0))}）")
        # --- やけど処理（ダメカン2個＋コイントスで解除判定） ---
        if "Burn" in condition_names or "Burned" in condition_names or "やけど" in condition_names:
            add_damage_counters(top, 2)
            self.log_print(f"{self.name}のバトル場は『やけど』でダメカン2個（残りHP:{self._display_hp(getattr(top, 'hp', 0))}）")
            coin = random.choice(["heads", "tails"])
            if coin == "heads":
                # やけど解除
                new_conditions = [c for c in conditions if not (
                    (hasattr(c, "__class__") and c.__class__.__name__ in ["Burn", "Burned"])
                    or (isinstance(c, str) and c == "やけど")
                )]
                top.conditions = new_conditions
                self.log_print(f"コイントス：表 → 『やけど』回復")
            else:
                self.log_print(f"コイントス：裏 → 『やけど』継続")
        # --- ねむり処理（コイントス） ---
        if "Sleep" in condition_names or "ねむり" in condition_names:
            coin = random.choice(["heads", "tails"])
            if coin == "heads":
                new_conditions = [c for c in conditions if not (
                    (hasattr(c, "__class__") and c.__class__.__name__ == "Sleep")
                    or (isinstance(c, str) and c == "ねむり")
                )]
                top.conditions = new_conditions
                self.log_print(f"コイントス：表 → 『ねむり』回復")
            else:
                self.log_print(f"コイントス：裏 → 『ねむり』継続")
        # --- まひ処理（自動回復） ---
        if "Paralysis" in condition_names or "Paralyzed" in condition_names or "まひ" in condition_names:
            new_conditions = [c for c in conditions if not (
                (hasattr(c, "__class__") and c.__class__.__name__ in ["Paralysis", "Paralyzed"])
                or (isinstance(c, str) and c == "まひ")
            )]
            top.conditions = new_conditions
            self.log_print(f"{self.name}のバトル場は『まひ』が自動で治る")
        # --- 気絶判定 ---
        if top.hp <= 0:
            self.log_print(f"{self.name}のバトル場がポケモンチェックで気絶！")
            # ★ 相手にサイドを与える
            if self.opponent:
                self.opponent.handle_knockout_points()

# NOTE:
#   setup_battle_and_bench は class Player の中に「定義」されているのではなく、
#   このモジュール末尾で関数として定義し、最後に
#     Player.setup_battle_and_bench = setup_battle_and_bench
#   で Player に動的にバインド（モンキーパッチ）しています。
#   そのため「class Player 内を検索しても見つからない」のが正常です。
def setup_battle_and_bench(self, opponent, viewing_player=None):
    # log_modeによる出力抑制をやめ、AI vs AIと同じく全出力を許可
    if viewing_player is None:
        viewing_player = self

    # --- 表示 ---
    if viewing_player == self:
        print(f"{self.name} の手札一覧:")
        if self.hand:
            hand_str = ', '.join(f"{c.name}" for c in self.hand)
            print(hand_str)
        else:
            print("なし")
    else:
        if hasattr(self, 'match') and self.match and self.match.turn == 0:
            if not self.is_bot:
                print(f"{self.name} の手札一覧:")
                if self.hand:
                    hand_str = ', '.join(f"{c.name}" for c in self.hand)
                    print(hand_str)
                else:
                    print("なし")
            else:
                pass

    basic_pokemons = [card for card in self.hand if getattr(card, "is_basic", False)]
    if not basic_pokemons:
        if not (self.match and getattr(self.match, 'log_mode', False)) and viewing_player == self:
            print(f"{self.name} の手札にたねポケモンがいません（後でマリガン判定）")
        return

    # ここで現在のターン番号を取得（初期セットアップ時は 0）
    turn_now = getattr(self.match, "turn", 0)

    # バトル場選択の表示
    if viewing_player == self:
        print(f"{self.name} の手札にあるたねポケモン:")
        for i, card in enumerate(basic_pokemons, start=1):
            print(f" {i}: {card}")
        if self.is_bot:
            print(f"AI選択肢: {[f'{i}: {card.name}' for i, card in enumerate(basic_pokemons, start=1)]}")
            idx = random.randint(0, len(basic_pokemons) - 1)
            print(f"{self.name} はバトル場に出すたねポケモンの番号として {idx + 1}（{basic_pokemons[idx].name}）をランダム選択しました")
        else:
            while True:
                try:
                    idx_raw = int(input(f"{self.name} はバトル場に出すたねポケモンの番号を入力してください: "))
                    if 1 <= idx_raw <= len(basic_pokemons):
                        idx = idx_raw - 1
                        break
                except Exception:
                    pass
                print("有効な番号を入力してください")
    else:
        if self.is_bot:
            idx = random.randint(0, len(basic_pokemons) - 1)
        else:
            idx = 0  # 人間プレイヤーがviewing_playerでなければ自動で0番

    active = basic_pokemons[idx]
    # ★ 初期配置にも entered_turn と can_evolve を明示
    active.entered_turn = turn_now
    active.can_evolve = False

    self.active_card = [active]
    self.hand.remove(active)
    if viewing_player == self:
        print(f"{self.name} はバトル場に {active} をセット")

    bench_candidates = [poke for j, poke in enumerate(basic_pokemons) if j != idx]
    for poke in bench_candidates:
        if viewing_player == self:
            if self.is_bot:
                res = random.choice(['y', 'n'])
                print(f"AI選択肢: ['y', 'n']")
                print(f"{self.name} は「{poke}」をベンチに出しますか？ (y/n): {res}（ランダム選択）")
            else:
                while True:
                    res = input(f"{self.name} は「{poke}」をベンチに出しますか？ (y/n): ")
                    if res.lower() in ['y', 'n']:
                        break
                    print("y または n を入力してください")
        else:
            if self.is_bot:
                res = random.choice(['y', 'n'])
            else:
                res = 'n'

        if res.lower() == 'y':
            # ★ ベンチに置く初期配置にも entered_turn / can_evolve を付与
            poke.entered_turn = turn_now
            poke.can_evolve = False
            try:
                setattr(poke, "has_used_ability", False)
            except Exception:
                pass

            self.bench.append([poke])
            if poke in self.hand:
                self.hand.remove(poke)
            if viewing_player == self:
                print(f"{self.name} は「{poke}」をベンチに出しました")
        else:
            if viewing_player == self:
                print(f"{self.name} は「{poke}」をベンチに出しませんでした")

    try:
        m = getattr(self, "match", None)
        if self.is_bot and viewing_player != self:
            out = m.log_print if m and hasattr(m, "log_print") else print

            if self.active_card:
                out(f"{self.name} はバトル場に たねポケモンを裏向きでセット")
            else:
                out(f"{self.name} はバトル場に たねポケモンをセットできませんでした")

            if self.bench:
                out(f"{self.name} はベンチに {len(self.bench)} 体を裏向きでセット")
            else:
                out(f"{self.name} はベンチに出しませんでした")
    except Exception:
        pass

Player.setup_battle_and_bench = setup_battle_and_bench

def parse_action_text(text):
    if "ベンチに出す" in text:
        m = re.search(r"\] (.+) をベンチに出す", text)
        return ["play_bench", m.group(1)] if m else ["unknown", text]
    elif "特性" in text:
        # 特性のテキストから特性名を抽出
        # パターン: "[特性] 特性名：効果"
        m = re.search(r"\[特性\] (.+?)：(.+)", text)
        if m:
            return ["ability", m.group(1), m.group(2)]
        else:
            # フォールバック: 特性の部分だけを抽出
            m = re.search(r"\[特性\] (.+)", text)
            return ["ability", m.group(1), ""] if m else ["unknown", text]
    elif "エネルギー" in text and "付ける" in text:
        m = re.search(r"\] (.+) を (.+) の (.+) に付ける", text)
        return ["attach_energy", m.group(1), m.group(2), m.group(3)] if m else ["unknown", text]
    elif "サポーター" in text and "使う" in text:
        m = re.search(r"\] (.+) を使う", text)
        return ["use_supporter", m.group(1)] if m else ["unknown", text]
    elif "どうぐ" in text and "付ける" in text:
        m = re.search(r"\] (.+) を (.+) の (.+) に付ける", text)
        return ["attach_tool", m.group(1), m.group(2), m.group(3)] if m else ["unknown", text]
    elif "自分の番を終わる" in text:
        return ["end_turn"]
    elif "ワザ宣言" in text:
        m = re.search(r"\] (.+) の (.+)", text)
        return ["attack", m.group(1), m.group(2)] if m else ["unknown", text]
    elif "進化" in text:
        m = re.search(r"\] (.+) の (.+) を (.+) に進化", text)
        return ["evolve", m.group(2), m.group(3)] if m else ["unknown", text]
    elif "グッズ" in text and "使う" in text:
        m = re.search(r"\] (.+) を使う", text)
        return ["use_item", m.group(1)] if m else ["unknown", text]
    elif "逃げる" in text:
        return ["retreat"]
    elif "スタジアム" in text and "場に出す" in text:
        m = re.search(r"\] (.+) を場に出す", text)
        return ["stadium", m.group(1)] if m else ["unknown", text]
    elif "スタジアム" in text and "使用" in text:
        m = re.search(r"\] (.+)の効果を使用", text)
        return ["stadium_effect", m.group(1)] if m else ["unknown", text]
    else:
        return ["unknown", text]