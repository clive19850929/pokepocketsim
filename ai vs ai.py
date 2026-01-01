# jsonファイルに変換した時は[-1]など変換にミスした選択肢が無いか確認する
"""
================================================================================
ai vs ai.py : 自己対戦ランナー（MCTS(pi) + PhaseD-Q(CQL Q) の混合）/ 学習ログ生成
--------------------------------------------------------------------------------
このスクリプトは poke-pocket-sim の自己対戦を連続実行し、以下を行います。

  1) 方策の構築（per-process / 遅延ロード）
     - マルチプロセス worker 内で build_policy() を呼び、ワーカー専用の policy を生成する
       （policy_p1 / policy_p2 が未定義でも安全に build_policy へフォールバック）
     - online_mix の場合:
         - main_policy : AlphaZeroMCTSPolicy（MCTS により π を生成）
         - fallback / Q側 : PhaseD-Q（d3rlpy CQL の Q 値）を混合（OnlineMixedPolicy 等）
       ※ PhaseD-Q の混合パラメータ（λ/温度/topk/モデルパス等）は
          「online_mix 実装（現行: ai vs ai.py 内の policy/wrapper 群）」側で管理される
       ※ MCTS のシミュレーション中は Player.select_action のガードにより
          “重い online_mix を回さず deterministic で先頭を返す” 経路がある

  2) ログの統合（1ゲーム=1ログ + 構造化ログを優先）
     - stdout/stderr を {GAMELOG_DIR}/{game_id}.log に集約（GameLogContext）
       ※ 既に console_tee が同一パスへ tee 済みの場合は二重 tee を回避する
     - ML用の構造化ログは {tmp or PER_MATCH_LOG_DIR}/ai_vs_ai_match_{pid}_{n}.ml.jsonl を使用し、
       “残骸誤読”防止のため毎試合必ず空ファイルに初期化してから書き始める
     - 解析（parse_log_file）は可能なら常に .ml.jsonl を最優先し、
       うまく取れない場合のみ .log へフォールバックする

  3) 行動決定ログの“試合中”可視化（DECIDE pending → 行動行直前 flush）
     - Player.select_action 側では [DECIDE_PRE]/[DECIDE_POST]/[DECIDE_DIFF] を
       “その場で出力せず” self._pending_decide_lines に積むだけにする
     - Player.act_and_regather_actions の冒頭で pending を必ず flush してから
       “行動行（例: '36 p1: ...'）” を出す
       → 「試合中にモデルが呼ばれている」ことをログ上で定量確認できる

  4) 終局時 SUMMARY の“1回だけ”制御
     - 終局検知時（Player.act_and_regather_actions の game_over 分岐）に
       match._policy_summary_logged を用いて SUMMARY を 1ゲーム1回だけ出す
       （PhaseD-Q / online_mix の log_summary / on_game_end 等を探索して呼ぶ）

  5) 学習データ(JSONL)の生成（worker → writer のバッチ集約）
     - raw / ids / private_ids を “batch” として queue に送る（集中 writer が統合出力）
     - raw:
         - LOG_FULL_INFO=True なら完全情報、False なら公開版（privates 除去）を出力
         - end_reason（PRIZE_OUT / BASICS_OUT / DECK_OUT ...）を併記
         - 1ゲーム1行の game_summary（record_type="game_summary"）を raw に追記する
     - ids:
         - 公開情報ベースの整数ID化（converter.convert_record）
         - pi は「元ログに raw π があれば優先」、無ければ onehot 等で補完する
         - action_candidates_vec（32d等）は _embed_legal_actions_32d で付与する
         - obs_vec は EMIT_OBS_VEC_FOR_CANDIDATES により build_obs_partial_vec 等で付与する
     - private_ids:
         - keep_private=True の完全情報ID化に加え、attach_fullvec_fields により
           obs_full_vec 等を付与する（必要な後段学習のため）
     - デッキアウト除外:
         - SKIP_DECKOUT_LOGGING=True の場合、デッキアウト判定を “未加工全行” で先に行い、
           該当試合は early-continue で writer に流さない（容量/品質対策）

--------------------------------------------------------------------------------
MCC（Monte-Carlo Completion）に関する現行の前提と注意
--------------------------------------------------------------------------------
  - worker は match.use_mcc / match.mcc_samples / match.mcc_every / match.cpu_workers_mcc 等を設定する
  - さらに USE_MCC or USE_REWARD_SHAPING の場合、reward_shaping_obj を Match に渡す
  - 試合中の MCC 呼び出し回数は、mcc_debug_snapshot().total_calls の差分で計測し、
    [MCC_CALLS] game_id=... calls=... として出力する

  - 重要（現状の不具合/要注意点）:
      USE_MCC=True でも calls=0 になる場合がある。
      典型原因は「MCC を呼ぶ箇所が use_reward_shaping 判定に縛られており、
      use_mcc=True だけでは reward_shaping 側が実行されない」こと。
      → MCC を有効化するには “Match/Player のフック条件” が
        use_reward_shaping OR use_mcc を見るように統一されている必要がある。

--------------------------------------------------------------------------------
関連ファイル（役割と関係）
--------------------------------------------------------------------------------

[ ai vs ai.py ] ・・・ オーケストレーター（本ファイル）
  - 対戦ループ、マルチプロセス worker、集中 writer、JSONL出力、集計、デバッグログ制御
  - worker 内で Deck を作り、Player/Match を組み立てて試合を回す
  - build_policy() で online_mix / az_mcts / random 等を組み立て、Match に差し込む
  - converter / encoder を Match / policy に接続し、obs_vec や候補ベクトル生成を支える
  - MCC_CALLS（mcc_debug_snapshot 差分）を game_id 単位で表示できるようにする


[ az_mcts_policy.py ] ・・・ AlphaZeroMCTSPolicy（MCTS(pi) 生成）
  - Selfplay supervised PV モデル（例: selfplay_supervised_pv_gen000.pt）をロードして
    盤面→候補→(policy,value) を出し、MCTS で π を作る中核
  - online_mix の main_policy 側として利用される想定

[ phased_q_mixer.py ] ・・・ （旧）PhaseD-Q 分離版（現行は未使用の可能性）
  - 現行運用では PhaseD-Q の混合は online_mix 実装（ai vs ai.py 内の wrapper）に寄っている
  - 将来 “PhaseD-Q を分離管理” する場合の置き場候補だが、現状は import 経路に無い可能性が高い

[ match.py / player.py / action.py ] ・・・ シミュレータ本体（進行/合法手/行動実行）
  - Match : ターン進行・合法手列挙・勝敗決定・ログファイル出力（.log/.ml.jsonl）
            worker から use_mcc / mcc_* / converter / encoder 等を注入される
  - Player: 行動決定（policy呼び出し）と行動実行の起点
            - select_action: pending decide lines を積む（出力はしない）
            - act_and_regather_actions: pending を flush して “行動行直前”に DECIDE を出す
            - 終局時: SUMMARY を 1回だけ出す（match._policy_summary_logged）
  - Action: 行動表現（to_id_vec 等）と変換の起点

[ battle_logger.py ] ・・・ 対戦ログの記録（通常ログ + ML用の構造化ログの土台）
  - Player から委譲されて使われる想定のロガー
  - state/action/合法手/substeps 等を蓄積し、.ml.jsonl 出力の材料を提供
  - 「終局後に追記しない」等のガードでログ整合性を担保する

[ action_space.py ] ・・・ 行動空間の定義（ML用の ID/ベクトル化の基盤）
  - ActionType / action schema / action_vec の次元定義など
  - 「legal_actions の ID化」「候補ベクトル（32d等）埋め」の下支え

[ cards_enum.py ] ・・・ カードID（enum）定義の中核
[ cards_db.py ] ・・・ カードDB（カード個別定義・効果の紐付け）
[ deck.py / decks.py ] ・・・ デッキ構築とレシピ管理（自己対戦の入力）
[ helpers_initial_deck.py ] ・・・ 初期デッキ列/ゾーン列の enum 化ユーティリティ
[ helpers_public_counts.py ] ・・・ 公開情報カウント（見えている枚数の推定補助）
[ card_base.py / card.py ] ・・・ カード実体と共通基盤
[ ability.py / supporter.py / item.py / tool.py / stadium.py ] ・・・ 効果処理の各モジュール
[ attack.py / attack_common.py / damage_utils.py ] ・・・ ワザ処理（ダメージ/特殊処理/整合性）
[ generator_attack.py ] ・・・ （開発補助）攻撃定義の生成器（ランタイム必須ではない場合あり）
[ protocols.py ] ・・・ （補助）型/Protocol 定義（import 経路次第）

[ my_mcc_sampler.py / mcc_ctx.py ] ・・・ MCC（Monte-Carlo Completion）一式
  - base_state から me_private/opp_private を補完して擬似完全情報状態をサンプルする
  - 呼び出し回数/統計のスナップショットを持ち、MCC が “実際に回っているか” を定量確認しやすい
  - ※ 現状の自己対戦で USE_MCC=True でも calls=0 の場合は
     「フック条件が use_reward_shaping に縛られている」疑いが強い

[ reward_shaping.py ] ・・・ PBRS / 進捗特徴量 / MCCフックの集約候補
  - Phase C 用の shaped reward / Φ(s) / 進捗特徴量（deck危険度・ターン罰など）を管理しやすい
  - 現行の自己対戦で MCC を動かす場合は、Match/Player 側の呼び出し条件と整合が必要

[ game_log.py ] ・・・ GameLogContext（stdout/stderr を game_id.log へ集約）
  - {GAMELOG_DIR}/{game_id}.log に “試合ログ + デバッグログ” を一本化
  - console_tee が同一パスに tee 済みなら二重teeを避ける

[ console_tee.py / ai vs ai.py 内蔵tee ] ・・・ 起動直後からの tee ラッパ（任意/併用）
  - 早い段階から stdout/stderr を tee したい場合に利用
  - GameLogContext と併用する際は「同一パス二重書き込み」を避ける

--------------------------------------------------------------------------------
学習フロー（B → D のスクリプト群）
--------------------------------------------------------------------------------

[ prepare_selfplay_supervised.py ] ・・・ Phase B 教師データ JSONL 化（s_t, π_t, z）
  - 入力: ai_vs_ai_match_all_ids.jsonl 等（ids ログ）
  - 出力: selfplay_supervised_dataset.jsonl
    形式: {"obs_vec","action_candidates_vec","pi","z","end_reason", ...}

[ train_selfplay_supervised.py ] ・・・ Phase B 教師あり学習（Policy+Value）
  - 入力: selfplay_supervised_dataset.jsonl
  - 出力: selfplay_supervised_pv_gen000.pt（PVネット）
  - end_reason で sample_weight を付与でき、DECK_OUT を弱める設計が可能

[ build_phaseD_rl_dataset.py ] ・・・ Phase D 用 RL データセット(.npz) 構築
  - 入力: ai_vs_ai_match_all_private_ids.jsonl 等（private_ids ログ）
  - 出力: phaseD_rl_dataset_all.npz / phaseD_rl_dataset_all_meta.json
  - end_reason による sample_weight / reward 設計をここで管理

[ train_phaseD_cql.py ] ・・・ Phase D: CQL 学習（Q(s,a)）
  - 入力: phaseD_rl_dataset_all.npz
  - 出力: learnable_phaseD_cql.d3（d3rlpy learnable）等
  - PhaseD-Q 側は、この learnable を読み込んで “Q推定” を行う

--------------------------------------------------------------------------------
動作確認の“最低ライン”（ログでチェック）
--------------------------------------------------------------------------------
  - 起動直後:
      [POLICY_SPEC] に online_mix / SELFPLAY_ALPHAZERO_MODE / USE_MCTS_POLICY が出る
  - 試合中（行動の直前）:
      [DECIDE_PRE] / [DECIDE_POST] / [DECIDE_DIFF] が行動行の直前に出る
      → 「MCTS(pi) と PhaseD-Q(Q) を使った判断が実際に呼ばれている」証拠
  - 終局:
      [PhaseD-Q][SUMMARY] が 1ゲーム1回だけ出る（Player の once 制御が効いている）
  - MCC:
      [MCC_CALLS] game_id=... calls=... が出る（calls>0 が理想）
      ※ calls=0 の場合は “MCCフック条件の誤り” を疑う（use_reward_shaping 縛り等）
================================================================================
"""
"""
================================================================================
Phase C 概要（PBRS + MCC + “DECK_OUT を減らす”ための橋渡し）
--------------------------------------------------------------------------------
Phase C の狙い
  - 目的1: DECK_OUT に流れやすい自己対戦を抑制し、「勝ちに行く短期の目的」を学習に混ぜる
  - 目的2: 部分観測（公開情報）でも安定して学べる表現・ラベルを作る（= Phase B と Phase D の間を埋める）
  - 目的3: 隠れ情報（山札/手札）をそのまま覗かずに、MCC で “それっぽい完全情報候補” をサンプルして学習を安定化

入力（Phase C が触るログ/表現）
  - ids ログ（公開情報ベース）:
      obs_vec / action_candidates_vec / pi / z / end_reason ...
  - private_ids ログ（完全情報ベース）:
      obs_full_vec 等（必要なら）/ keep_private=True の状態
  - deck / 初期デッキ列（helpers_initial_deck）や公開カウント（helpers_public_counts）:
      「見えている情報」と「見えていない情報」の境界を作る材料

中核コンポーネント
  - MCC（my_mcc_sampler.py / mcc_ctx.py）
      base_state（公開情報 + 一部確定情報）に対して、
      me_private/opp_private を “候補分布”として補完し、擬似完全情報状態を複数サンプルする
      → 1状態に対して K 個の completion を作り、期待値/分散を持った学習信号を作れる

  - PBRS / 中間報酬（reward_shaping.py）
      Potential Φ(s) を設計し、 shaped_r = r + γΦ(s') - Φ(s) を付与する
      - DECK_OUT を誘発する長期戦に対して、ターンペナルティ/停滞ペナルティ等を段階的に導入
      - prize差・盤面優位・山札危険度など “勝ちに繋がる進捗” を薄く加点（やりすぎない）
      ※ 重要: PBRS は理論上 “最適方策を保つ” 形で報酬を足せる（ただし Φ の設計は慎重に）

Phase C を自己対戦に“直接”入れるときの注意（現行の落とし穴）
  - USE_MCC=True でも、フック条件が use_reward_shaping に縛られていると MCC が呼ばれず calls=0 になる
  - したがって、自己対戦で MCC を動かすなら
      「reward_shaping を呼ぶ条件 = use_reward_shaping OR use_mcc」
    が Match/Player 側で統一されている必要がある

Phase C の成果物（出力）
  - completion 平均/分散を持つ value ターゲット（任意）
  - end_reason ベースの重み付きサンプル（DECK_OUT を弱める）
  - Phase D（CQL）に渡す RL dataset の改善材料（reward/weight の再設計）

--------------------------------------------------------------------------------
Phase C は「自己対戦を壊さずに、勝ちに行く学習信号を足す」ための調整フェーズ。
Phase B（PV）と Phase D（CQL）をつなぐ “現場の補強工事” として位置づける。
================================================================================
"""

import random, uuid, json, os, torch, time, warnings, tempfile, d3rlpy, numpy as np
import sys, atexit
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)


# --- console tee (stdout/stderr -> single game_id.log) ---
try:
    from console_tee import _tee_console_start, _tee_console_stop
except Exception:
    def _tee_console_start(log_path: str, enable: bool = True) -> None:
        return

    def _tee_console_stop() -> None:
        return

from collections import Counter


from multiprocessing import Process, Queue, Event, cpu_count
from pokepocketsim.player import Player
from pokepocketsim.action import ActionType
from pokepocketsim.action import ACTION_SCHEMAS
from pokepocketsim.deck import Deck
from pokepocketsim.decks import make_deck_from_recipe, ALL_DECK_RECIPES, DECK_TYPE_NAMES
# ↓ 重複/衝突を避けるため、cards_enum 側だけを使う
from pokepocketsim.cards_enum import Cards, Attacks
from pokepocketsim.match import Match
from pokepocketsim.my_mcc_sampler import mcc_sampler, mcc_debug_snapshot, reset_mcc_debug

# 追加: 唯一のアクションエンコーダ
from pokepocketsim.policy.action_encoding import (
    build_encoder_from_files, compute_layout
)

# ポリシー群（集約 import は消して、実体モジュールから）
from pokepocketsim.policy.random_policy import RandomPolicy
from pokepocketsim.policy.state_encoder import StateEncoder
from pokepocketsim.policy.az_mcts_policy import AlphaZeroMCTSPolicy

# =======================================================
# ====== 設定項目（このブロックだけ編集すればOK）==========

# ログ制御フラグ
LOG_DEBUG_DETAIL = False  # True にすると [DEBUG] 系の詳細ログを出す

## --- デッキ指定 --- ##
DECK1_KEY = "deck01" # 対戦に使うデッキ名を変更
DECK2_KEY = "deck02"

## --- 実行回数など --- ##
TARGET_EPISODES = 800 # 目標エピソード数
NUM_MATCHES     = 1  # 1回の実行で回すバッチ数
MATCH_SLEEP_SEC = 0.2    # 1戦ごとに待つ秒数（0.0なら待機なし、0.5なら0.5秒待機）

"""
【設定メモ】
SELFPLAY_ALPHAZERO_MODE = True の場合でも、次は「併用」できるようにしている（移行完了）。
- PhaseD-Q（online_mix）: 行動選択（MCTS(pi) と CQL Q の混合）
- MCC（USE_MCC=True）    : 部分観測の補完（学習ログ生成・解析の補助）
- PBRS（USE_REWARD_SHAPING=True）: shaped reward / 進捗特徴量を付与（主に学習ログ用）

※ 注意:
  - PBRS は value_net を使う設定にしている場合、事前に ValueNet の学習成果物が必要。
  - ただし PBRS 自体は「報酬の付与」であり、online_mix の行動選択ロジックとは独立。
"""

# === 新フラグ: AlphaZero型自己対戦モードに切り替えるかどうか ===
SELFPLAY_ALPHAZERO_MODE = True
USE_MCTS_POLICY = True
AZ_MCTS_NUM_SIMULATIONS = 64  # AlphaZero MCTS のシミュレーション回数（チューニング用）

# どの方策を使うか
#   "az_mcts"    : AlphaZero 方式。MCTS内部でモデルを使い、最終手はMCTSの訪問回数から選ぶ
#   "model_only" : MCTSを使わず、モデルのポリシー出力だけで手を選ぶ
#   "random"     : ポリシーを使わず、合法手から一様ランダムに選ぶ
#   "online_mix" : 1手ごとにモデルとMCTSのポリシーを混合して手を選ぶ（OnlineMixedPolicy）
P1_POLICY = "online_mix"
P2_POLICY = "online_mix"

# --- POLICY TRACE（ai vs ai 起動時に自動アクティブ） ---
os.environ.setdefault("POLICY_TRACE", "1")

_POLICY_BOOT_EMITTED = False
def _emit_policy_boot_logs_once():
    global _POLICY_BOOT_EMITTED
    if _POLICY_BOOT_EMITTED:
        return
    _POLICY_BOOT_EMITTED = True

    try:
        import policy_trace_monkeypatch  # noqa
        if os.getenv("POLICY_TRACE") == "1":
            print("[POLICY_TRACE] enabled (policy_trace_monkeypatch loaded)")
    except Exception as _e:
        print("[POLICY_TRACE] import failed:", _e)

    print(f"[POLICY_SPEC] P1_POLICY={P1_POLICY} P2_POLICY={P2_POLICY} SELFPLAY_ALPHAZERO_MODE={SELFPLAY_ALPHAZERO_MODE} USE_MCTS_POLICY={USE_MCTS_POLICY}")

# --- AlphaZero policy/value モデルの配置（世代ごとにここだけ変える） ---
AZ_MODEL_DIR = r"D:\alpha_zero_models\gen000"
AZ_MODEL_FILENAME = "selfplay_supervised_pv_gen000.pt"
AZ_MODEL_PATH = os.path.join(AZ_MODEL_DIR, AZ_MODEL_FILENAME)
os.environ.setdefault("AZ_MODEL_DIR", AZ_MODEL_DIR)
os.environ.setdefault("AZ_PV_MODEL_PATH", AZ_MODEL_PATH)

# --- 勝敗ラベル z の強さ設定（end_reason ごと） ---
#   - PRIZE_OUT  : サイド取り切りによる決着
#   - BASICS_OUT : タネ切れによる決着
#   - DECK_OUT   : デッキアウトによる決着
#   - UNKNOWN    : 終了理由が不明な場合
#
# --- z/value の基準値テーブル（終局理由ごとのスケール） ---
# ここで設定するのは「勝利したときのプラス側スケール」のみ。
#   ・自分(me)が勝者の場合:  end_reason に応じて +Z_TABLE[reason] を採用
#   ・自分(me)が敗者の場合:  end_reason に関わらず一律で LOSS_Z（-1.0）
#   ・引き分け / UNKNOWN:    DRAW_Z（0.0）をベース（あとから途中盤面シェーピングを加算）
#
# 片側視点での「終局時の価値」の概念上の順番（勝利時の絶対値の序列）:
#   敗北(LOSS_Z=-1.0) < デッキアウト勝ち(+0.3) < 引き分け(DRAW_Z=0.0) < サイド/BASICS勝ち(+1.0)
#
# ※ 実際の z は、上記の終局 z に「途中盤面シェーピング（サイド差・ターン数ペナルティ）」を
#    加えたうえで [-1.0, +1.0] にクリップしてから各ステップに付与されます。
#    このときシェーピングの絶対値は Z_PROGRESS_MAX で抑制し、
#    「負け < デッキアウト勝ち < 引き分け < サイド/BASICS勝ち」の序列が崩れないようにします。
Z_TABLE = {
    "PRIZE_OUT":   1.0,  # ちゃんとサイドを取り切って勝利（最大）
    "BASICS_OUT":  1.0,  # タネ切れ勝ち（ほぼサイド勝ちに近い）
    "DECK_OUT":   -0.2,   # デッキアウト勝ちは「実質損」扱い（ただし負け(-1.0)よりは上）
    "TIMEOUT":     0.2,  # 時間切れなど
    "SUDDEN_DEATH": 0.7, # サドンデス（中〜やや強め）
    "CONCEDE":     0.5,  # 投了による勝ち
    "UNKNOWN":     0.2,  # 理由不明時
}

# 敗北・引き分けの基準値（end_reason に依らず一定）
LOSS_Z = -1.0   # 負けは一律で -1.0
DRAW_Z = 0.0    # 引き分けは 0.0

# --- 途中盤面由来 z シェーピング用の係数 ---
# ・サイド差（相手よりサイドが少ないほどプラス）
# ・ターン数ペナルティ（長期戦になりすぎるとマイナス）
INTERMEDIATE_PRIZE_WEIGHT = 0.4        # サイド差が最大 6 枚開いたとき ±0.5 程度（raw 値）
INTERMEDIATE_TURN_PENALTY_START = 40   # このターン数以降は少しずつマイナス
INTERMEDIATE_TURN_PENALTY_PER_TURN = 0.01
INTERMEDIATE_MAX_TURN_PENALTY = 0.4    # ターン数ペナルティの下限（最大で −0.5）
# ・デッキ残枚数差（自分の山札が多いほどプラス）
INTERMEDIATE_DECK_COUNT_WEIGHT = 0.6   # deck_count 差が最大級に開いたとき ±0.3 程度（raw 値）

# 途中盤面シェーピングの最終上限（絶対値）。
# raw なシェーピング値を一度計算したあと、[-Z_PROGRESS_MAX, +Z_PROGRESS_MAX] にクリップする。
# これにより終局 tier（勝ち方の強さ）の序列を壊さない。
Z_PROGRESS_MAX = 0.2

# --- Phase A: 公開IDと完全情報IDを同時に生成（教師/生徒データを同時出力） ---
ALWAYS_OUTPUT_BOTH_IDS = True

# --- 完全情報ベクトルのスキーマ版（Phase A-④ 用） ---
FULL_VEC_VERSION = "v1"

# === MCC（Monte-Carlo Completion）設定 ===
LOG_FULL_INFO = True              # ← 神視点ログ（全ての非公開情報）を出すか
USE_MCC       = os.getenv("USE_MCC", "1") == "1"  # ← MCCを使って非公開情報を補完して期待値を計算するか
MCC_SAMPLES   = 16                # ← 1評価あたりのサンプル数K（例: 16/32/64）
MCC_TOP_K     = 0                 # ← ログ/補完で保持する山札トップN（head=上からN枚）
SKIP_DECKOUT_LOGGING: bool = False  # True=山札切れ試合は記録しない（Falseにすると記録する）

# === PBRS（Potential-Based Reward Shaping）設定 ===
#  - shaped_r = r + γΦ(s') - Φ(s)
#  - 既定では ValueNet を Φ(s) として使う（事前に学習済みモデルが必要）
USE_REWARD_SHAPING: bool = True
REWARD_SHAPING_GAMMA: float = 0.99

# ValueNet（Φ(s)）の成果物パス（例: v3 の学習結果）
VALUE_NET_MODEL_DIR = r"D:\date\value_net_run_v3"
VALUE_NET_MODEL_FILENAME = "value_net.pt"
VALUE_NET_MODEL_PATH = os.path.join(VALUE_NET_MODEL_DIR, VALUE_NET_MODEL_FILENAME)
os.environ.setdefault("VALUE_NET_MODEL_PATH", VALUE_NET_MODEL_PATH)

# --- 試合ごとのファイル保持/削除（圧迫対策） ---
KEEP_MATCH_LOG_FILE = False   # Trueなら ai_vs_ai_match_*.log を保持、Falseなら集約後すぐ削除
KEEP_MATCH_ML_FILE  = False   # Trueなら ai_vs_ai_match_*.ml.jsonl を保持、Falseなら集約後すぐ削除

# ★ 追加: 集約（PRIVATE_IDS）への反映が完了したら per-match の .ml.jsonl を削除する
DELETE_PER_MATCH_ML_AFTER_PRIVATE_IDS = True
# ★ 追加: 連番ログの出力先（None ならシステムの一時ディレクトリ）
PER_MATCH_LOG_DIR = None

# --- 追加: writer 負荷対策（環境変数なしで既定オン） ---
JSONL_ROTATE_LINES   = 0   # これを超えたら _00001.jsonl, _00002.jsonl... にローテーション（行数基準）
IO_BUFFERING_BYTES   = 1024 * 1024  # 1MiB バッファで書き込み
BATCH_FLUSH_INTERVAL = 100      # 100バッチごとに明示 flush（0 で無効）

# 並列数（用途別）: MCC用/ランダム用
CPU_WORKERS_MCC    = 6   # MCC有効時の既定並列数
MCC_EVERY       = int(os.getenv("MCC_EVERY", "1"))
CPU_WORKERS_RANDOM = 7   # ランダム用の既定並列数（0=自動にしたければここを変える）
TORCH_THREADS_PER_PROC = 1



# === learnable を強制/優先するためのフラグ ===
PREFER_LEARNABLE = True
REQUIRE_LEARNABLE = os.getenv("REQUIRE_LEARNABLE", "0") == "1"

# 入力記録リスト（必要なら）
recorded_inputs = []

# === コールドスタート（学習成果物が無いとき自動生成） ===
COLD_START = os.getenv("COLD_START", "1") == "1"  # 最初はONにしておく

# ============================================================
# Phase D CQL Q(s,a) ローダ＆評価ヘルパ
# ============================================================

# Phase D CQL の成果物パス
PHASED_Q_LEARNABLE_PATH = r"D:\date\phaseD_cql_run_p1\learnable_phaseD_cql.d3"
PHASED_Q_META_PATH      = r"D:\date\phaseD_cql_run_p1\train_phaseD_cql_meta.json"

# Phase D Q を ai vs ai で使うかどうか
USE_PHASED_Q: bool = True

# 内部キャッシュ
_PHASED_Q_ALGO = None
_PHASED_Q_OBS_DIM: int | None = None
_PHASED_Q_ACTION_DIM: int | None = None
_PHASED_Q_LAST_FULL_OBS_VEC: list[float] | None = None

# ★ 追加: Phase D Q をオンラインで π にミックスするかどうか
PHASED_Q_MIX_ENABLED: bool = bool(int(os.getenv("PHASED_Q_MIX_ENABLED", "1")))
# ★ 追加: π と Q のブレンド率（0.0=純粋π, 1.0=純粋Q）
PHASED_Q_MIX_LAMBDA: float = float(os.getenv("PHASED_Q_MIX_LAMBDA", "0.30"))
# ★ 追加: Q 側の softmax 温度（大きいほど分布がなだらかになる）
PHASED_Q_MIX_TEMPERATURE: float = float(os.getenv("PHASED_Q_MIX_TEMPERATURE", "1.0"))

if LOG_DEBUG_DETAIL:
    print(f"[CFG] d3rlpy.__version__={d3rlpy.__version__}")

# --- console log tee (stdout/stderr -> file) ---
_CONSOLE_TEE_FP = None
_CONSOLE_TEE_STDOUT0 = None
_CONSOLE_TEE_STDERR0 = None
_CONSOLE_TEE_STDERR_CONSOLE = None

class _ConsoleTeeStream:
    def __init__(self, base, fp):
        self._base = base
        self._fp = fp
        try:
            self._console_tee_active = True
        except Exception:
            pass
        try:
            self._console_tee_path = getattr(fp, "name", "") or ""
        except Exception:
            self._console_tee_path = ""

    def write(self, s):
        try:
            self._base.write(s)
        except Exception:
            pass
        try:
            self._fp.write(s)
        except Exception:
            pass
        try:
            return len(s)
        except Exception:
            return 0

    def flush(self):
        try:
            self._base.flush()
        except Exception:
            pass
        try:
            self._fp.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._base.isatty())
        except Exception:
            return False

    @property
    def encoding(self):
        try:
            return getattr(self._base, "encoding", "utf-8")
        except Exception:
            return "utf-8"

def _setup_console_tee(log_dir, prefix="ai_vs_ai_console", enable=True, fixed_path=None):
    global _CONSOLE_TEE_FP, _CONSOLE_TEE_STDOUT0, _CONSOLE_TEE_STDERR0

    if not enable:
        return
    if _CONSOLE_TEE_FP is not None:
        return

    try:
        if fixed_path:
            path = fixed_path
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
        else:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"{prefix}_{os.getpid()}.log")
    except Exception:
        return

    try:
        fp = open(path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return

    try:
        _CONSOLE_TEE_STDOUT0 = sys.stdout
        _CONSOLE_TEE_STDERR0 = sys.stderr

        # NOTE: az_mcts_policy.py は os.write(2, ...) で書く場合がある。
        #       sys.stderr 差し替えだけでは拾えないため、OSレベル fd=2 もログへ向ける。
        try:
            import os as _os
            global _CONSOLE_TEE_STDERR_CONSOLE
            _fd2 = _os.dup(2)
            _os.dup2(fp.fileno(), 2)
            _CONSOLE_TEE_STDERR_CONSOLE = _os.fdopen(_fd2, "w", encoding="utf-8", buffering=1)
        except Exception:
            _CONSOLE_TEE_STDERR_CONSOLE = None

        sys.stdout = _ConsoleTeeStream(_CONSOLE_TEE_STDOUT0, fp)
        _stderr0 = _CONSOLE_TEE_STDERR_CONSOLE if _CONSOLE_TEE_STDERR_CONSOLE is not None else _CONSOLE_TEE_STDERR0
        sys.stderr = _ConsoleTeeStream(_stderr0, fp)
        _CONSOLE_TEE_FP = fp

        def _close():
            global _CONSOLE_TEE_FP, _CONSOLE_TEE_STDOUT0, _CONSOLE_TEE_STDERR0, _CONSOLE_TEE_STDERR_CONSOLE
            try:
                if _CONSOLE_TEE_STDOUT0 is not None:
                    sys.stdout = _CONSOLE_TEE_STDOUT0
                if _CONSOLE_TEE_STDERR0 is not None:
                    sys.stderr = _CONSOLE_TEE_STDERR0
            except Exception:
                pass

            # restore OS-level stderr (fd=2) BEFORE closing fp
            try:
                import os as _os
                if _CONSOLE_TEE_STDERR_CONSOLE is not None:
                    try:
                        _os.dup2(_CONSOLE_TEE_STDERR_CONSOLE.fileno(), 2)
                    except Exception:
                        pass
                    try:
                        _CONSOLE_TEE_STDERR_CONSOLE.close()
                    except Exception:
                        pass
                    _CONSOLE_TEE_STDERR_CONSOLE = None
            except Exception:
                pass

            try:
                if _CONSOLE_TEE_FP is not None:
                    _CONSOLE_TEE_FP.flush()
                    _CONSOLE_TEE_FP.close()
            except Exception:
                pass
            _CONSOLE_TEE_FP = None
            _CONSOLE_TEE_STDOUT0 = None
            _CONSOLE_TEE_STDERR0 = None

        atexit.register(_close)
    except Exception:
        try:
            fp.close()
        except Exception:
            pass
        return

def _setup_console_tee_to_file(log_path: str, enable: bool = True) -> None:
    try:
        d = os.path.dirname(log_path)
        if not d:
            d = "."
    except Exception:
        d = "."

    # 既に console_tee が同一パスへ tee 済みなら二重に開始しない
    try:
        target = os.path.normcase(os.path.normpath(os.path.abspath(str(log_path))))
        obj = getattr(sys, "stdout", None)
        for _ in range(16):
            if obj is None:
                break
            if getattr(obj, "_console_tee_active", False):
                cur = getattr(obj, "_console_tee_path", "") or ""
                if not cur:
                    fp0 = getattr(obj, "_fp", None)
                    cur = getattr(fp0, "name", "") if fp0 is not None else ""
                cur = os.path.normcase(os.path.normpath(os.path.abspath(str(cur))))
                if cur == target:
                    return
            obj = getattr(obj, "_base", None)
    except Exception:
        pass

    _setup_console_tee(d, prefix="ai_vs_ai_console", enable=enable, fixed_path=log_path)

# --- 出力関係の保存場所 ---
LOG_FILE = "ai_vs_ai_match.log"  # AI対戦ログ（任意）
RAW_JSONL_PATH  = r"D:\date\ai_vs_ai_match_all.jsonl"       # 生JSONL
IDS_JSONL_PATH  = r"D:\date\ai_vs_ai_match_all_ids.jsonl"   # ID変換済みJSONL
PRIVATE_IDS_JSON_PATH = r"D:\date\ai_vs_ai_match_all_private_ids.jsonl"
GAMELOG_DIR = os.path.dirname(RAW_JSONL_PATH)

# --- single gamelog: save ALL console prints into one game_id.log (shared by main/worker) ---
_RUN_GAME_ID = os.getenv("AI_VS_AI_RUN_GAME_ID", "").strip()
if not _RUN_GAME_ID:
    _RUN_GAME_ID = str(uuid.uuid4())
os.environ["AI_VS_AI_RUN_GAME_ID"] = _RUN_GAME_ID

_GAMELOG_PATH = os.path.join(GAMELOG_DIR, f"{_RUN_GAME_ID}.log")
os.environ["AI_VS_AI_GAMELOG_PATH"] = _GAMELOG_PATH
os.environ["AI_VS_AI_GAMELOG_ACTIVE"] = "0"

def _gamelog_append_line(msg: str) -> None:
    try:
        d = os.path.dirname(os.path.abspath(_GAMELOG_PATH))
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    try:
        with open(_GAMELOG_PATH, "a", encoding="utf-8", buffering=1) as _fp:
            _fp.write(str(msg).rstrip("\n") + "\n")
    except Exception:
        pass

ENABLE_CONSOLE_TEE = bool(int(os.getenv("AI_VS_AI_ENABLE_CONSOLE_TEE", "1")))

def _console_tee_is_active_for_path(path: str) -> bool:
    try:
        target = os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
        obj = getattr(sys, "stdout", None)
        for _ in range(16):
            if obj is None:
                break

            cur = getattr(obj, "_console_tee_path", "") or ""
            if not cur:
                fp0 = getattr(obj, "_fp", None)
                cur = getattr(fp0, "name", "") if fp0 is not None else ""

            if cur:
                cur = os.path.normcase(os.path.normpath(os.path.abspath(str(cur))))
                if cur == target:
                    return True

            obj = getattr(obj, "_base", None)
    except Exception:
        pass
    return False

_tee_ok = False
_tee_err = None
try:
    try:
        from console_tee import _tee_console_start as _tee_console_start_unified
        _tee_console_start_unified(_GAMELOG_PATH, enable=ENABLE_CONSOLE_TEE)
    except Exception:
        from console_tee import _setup_console_tee_to_file as _setup_console_tee_to_file_unified
        _setup_console_tee_to_file_unified(_GAMELOG_PATH, enable=ENABLE_CONSOLE_TEE)

    if ENABLE_CONSOLE_TEE and (not _console_tee_is_active_for_path(_GAMELOG_PATH)):
        _setup_console_tee_to_file(_GAMELOG_PATH, enable=ENABLE_CONSOLE_TEE)

    _tee_ok = (ENABLE_CONSOLE_TEE and _console_tee_is_active_for_path(_GAMELOG_PATH))
except Exception as _e:
    _tee_err = _e
    try:
        _setup_console_tee_to_file(_GAMELOG_PATH, enable=ENABLE_CONSOLE_TEE)
        _tee_ok = (ENABLE_CONSOLE_TEE and _console_tee_is_active_for_path(_GAMELOG_PATH))
    except Exception as _e2:
        _tee_err = _e2

os.environ["AI_VS_AI_GAMELOG_ACTIVE"] = ("1" if _tee_ok else "0")

if os.getenv("AI_VS_AI_GAMELOG_PRINTED", "") != "1":
    try:
        _self = os.path.normcase(os.path.normpath(os.path.abspath(__file__)))
    except Exception:
        _self = str(__file__)
    try:
        _argv0 = os.path.normcase(os.path.normpath(os.path.abspath(sys.argv[0])))
    except Exception:
        _argv0 = str(sys.argv[0])

def _stdout_is_teed_to_path(_path: str) -> bool:
    try:
        target = os.path.normcase(os.path.normpath(os.path.abspath(str(_path))))
        obj = getattr(sys, "stdout", None)
        for _ in range(16):
            if obj is None:
                break

            cur = getattr(obj, "_console_tee_path", "") or ""
            if not cur:
                fp0 = getattr(obj, "_fp", None)
                cur = getattr(fp0, "name", "") if fp0 is not None else ""

            if cur:
                cur = os.path.normcase(os.path.normpath(os.path.abspath(str(cur))))
                if cur == target:
                    return True

            obj = getattr(obj, "_base", None)
    except Exception:
        pass
    return False

if os.getenv("AI_VS_AI_GAMELOG_PRINTED", "") != "1":
    os.environ["AI_VS_AI_GAMELOG_PRINTED"] = "1"

    _line = f"[GAMELOG] game_id={_RUN_GAME_ID} path={_GAMELOG_PATH} self={_self} argv0={_argv0} tee_ok={_tee_ok}"
    print(_line)
    if (not _stdout_is_teed_to_path(_GAMELOG_PATH)):
        _gamelog_append_line(_line)

    if (not _tee_ok) and (_tee_err is not None):
        _eline = f"[GAMELOG][TEE_ERR] {_tee_err!r}"
        print(_eline)
        if (not _stdout_is_teed_to_path(_GAMELOG_PATH)):
            _gamelog_append_line(_eline)

MODEL_DIR_P1 = AZ_MODEL_DIR
MODEL_DIR_P2 = AZ_MODEL_DIR

# 出力先ディレクトリが無ければ作成
for p in (RAW_JSONL_PATH, IDS_JSONL_PATH, PRIVATE_IDS_JSON_PATH):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

_GLOBAL_MATCH = None
_GLOBAL_ENCODER = None

def register_match_encoder(match, encoder=None):
    global _GLOBAL_MATCH, _GLOBAL_ENCODER
    _GLOBAL_MATCH = match
    if encoder is None and match is not None:
        try:
            encoder = getattr(match, "encoder", None)
        except Exception:
            encoder = None
    _GLOBAL_ENCODER = encoder

def phaseD_q_load_if_needed() -> None:
    """
    Phase D CQL の learnable .d3 を lazy にロードし、グローバルにキャッシュする。
    ロードに失敗した場合は _PHASED_Q_ALGO は None のまま。
    """
    global _PHASED_Q_ALGO, _PHASED_Q_OBS_DIM, _PHASED_Q_ACTION_DIM

    if _PHASED_Q_ALGO is not None:
        return
    if not USE_PHASED_Q:
        return

    if not os.path.exists(PHASED_Q_LEARNABLE_PATH):
        print(f"[PhaseD Q] learnable not found: {PHASED_Q_LEARNABLE_PATH}", flush=True)
        return

    try:
        # d3rlpy v2 系では save_model(...) と対になるのは load_learnable(...)
        learnable = d3rlpy.load_learnable(PHASED_Q_LEARNABLE_PATH)

        # Learnable から CQL アルゴリズムを復元する
        try:
            from d3rlpy.algos import DiscreteCQL  # あなたが使っている CQL クラスに合わせて変更
            algo = DiscreteCQL.from_learnable(learnable)
        except Exception:
            # うまく復元できなかった場合はいったん learnable をそのまま保持
            algo = learnable
    except Exception as e:
        print(f"[PhaseD Q] failed to load .d3: {PHASED_Q_LEARNABLE_PATH} err={e!r}", flush=True)
        return

    _PHASED_Q_ALGO = algo

    # メタから obs_dim / action_dim を拾っておく（チェック用）
    try:
        with open(PHASED_Q_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        _PHASED_Q_OBS_DIM = int(meta.get("obs_dim", 0))
        _PHASED_Q_ACTION_DIM = int(meta.get("action_dim", 0))
    except Exception:
        _PHASED_Q_OBS_DIM = None
        _PHASED_Q_ACTION_DIM = None

    print(
        f"[PhaseD Q] loaded CQL learnable from {PHASED_Q_LEARNABLE_PATH} "
        f"(obs_dim={_PHASED_Q_OBS_DIM} action_dim={_PHASED_Q_ACTION_DIM})",
        flush=True,
    )

def phaseD_q_evaluate(
    obs_vec: list[float] | np.ndarray,
    action_candidates_vec: list[list[float]] | np.ndarray,
) -> np.ndarray | None:
    """
    1 状態 obs_vec と、その状態での action_candidates_vec (K x action_dim) を受け取り、
      Q(s, a_k) の配列 (K,) を返す。

    失敗時や USE_PHASED_Q=False のときは None を返す。
    """
    if not USE_PHASED_Q:
        return None

    phaseD_q_load_if_needed()
    if _PHASED_Q_ALGO is None:
        return None

    obs = np.asarray(obs_vec, dtype=np.float32)
    cands = np.asarray(action_candidates_vec, dtype=np.float32)

    if obs.ndim == 1:
        # shape: (obs_dim,) -> (1, obs_dim)
        obs = obs[None, :]

    if obs.ndim != 2:
        print(f"[PhaseD Q] invalid obs shape: {obs.shape}")
        return None

    if cands.ndim != 2:
        print(f"[PhaseD Q] invalid cands shape: {cands.shape}")
        return None

    # 事前に記録しておいた obs_dim / action_dim と食い違う場合はあえて使わない
    if _PHASED_Q_OBS_DIM is not None and obs.shape[1] != _PHASED_Q_OBS_DIM:
        print(
            f"[PhaseD Q] obs_dim mismatch: got={obs.shape[1]} "
            f"expected={_PHASED_Q_OBS_DIM}"
        )

        allow_pad = bool(int(os.getenv("PHASED_Q_ALLOW_OBS_PAD", "0")))
        if not allow_pad:
            return None

        # ★暫定: mismatch をパディング/切り詰めで通す（デバッグ用）
        try:
            exp = int(_PHASED_Q_OBS_DIM)
            got = int(obs.shape[1])
            if got < exp:
                pad = np.zeros((obs.shape[0], exp - got), dtype=obs.dtype)
                obs = np.concatenate([obs, pad], axis=1)
            else:
                obs = obs[:, :exp]
            print(f"[PhaseD Q] ⚠️ obs adjusted for debug: {got} -> {obs.shape[1]}")
        except Exception:
            return None

    if _PHASED_Q_ACTION_DIM is not None and cands.shape[1] != _PHASED_Q_ACTION_DIM:
        print(
            f"[PhaseD Q] action_dim mismatch: got={cands.shape[1]} "
            f"expected={_PHASED_Q_ACTION_DIM}"
        )
        return None

    # obs を候補数 K にブロードキャストして (K, obs_dim) にする
    num_actions = cands.shape[0]
    obs_batch = np.repeat(obs, num_actions, axis=0)

    # d3rlpy の CQL は predict_value(x, action) で Q(s,a) を返す
    try:
        values = _PHASED_Q_ALGO.predict_value(obs_batch, cands)
    except Exception as e:
        print(f"[PhaseD Q] predict_value failed: {e!r}")
        return None

    # values は shape (K,) を想定
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] != num_actions:
        print(
            f"[PhaseD Q] unexpected values shape: {values.shape}, "
            f"num_actions={num_actions}"
        )
        return None

    return values

def phaseD_mix_pi_with_q(pi, q_values):
    """
    Phase D の Q 値と元の π をブレンドして新しい方策分布を返す。

    - pi, q_values は同じ長さの 1 次元配列を想定
    - 何かおかしければ安全側として元の pi をそのまま返す
    """
    try:
        # numpy 配列に変換
        pi_arr = np.asarray(pi, dtype=np.float64)
        q_arr = np.asarray(q_values, dtype=np.float64)

        # 1 次元配列 & 次元一致チェック
        if pi_arr.ndim != 1 or q_arr.ndim != 1:
            return pi
        if pi_arr.shape[0] != q_arr.shape[0]:
            return pi

        # Q から softmax 分布を作る（温度付き）
        tau = max(PHASED_Q_MIX_TEMPERATURE, 1e-6)
        # 数値安定化のため最大値を引いてから softmax
        q_shift = q_arr - np.max(q_arr)
        q_soft = np.exp(q_shift / tau)
        s = float(q_soft.sum())
        if not np.isfinite(s) or s <= 0.0:
            return pi
        q_soft = q_soft / s

        # λ で π と Q-softmax を線形補間
        lam = min(max(PHASED_Q_MIX_LAMBDA, 0.0), 1.0)
        mixed = (1.0 - lam) * pi_arr + lam * q_soft
        try:
            _k = int(min(3, mixed.size))
            if _k > 0:
                _mix_idx = list(np.argsort(mixed)[- _k:][::-1])
                _pi_idx  = list(np.argsort(pi_arr)[- _k:][::-1])
                _q_idx   = list(np.argsort(q_soft)[- _k:][::-1])
                _mix_top = ",".join([f"{int(i)}:{float(mixed[int(i)]):.6f}" for i in _mix_idx])
                _pi_top  = ",".join([f"{int(i)}:{float(pi_arr[int(i)]):.6f}" for i in _pi_idx])
                _q_top   = ",".join([f"{int(i)}:{float(q_soft[int(i)]):.6f}" for i in _q_idx])
                _chosen = int(np.argmax(mixed))
                _chosen_p = float(mixed[_chosen]) if 0 <= _chosen < int(mixed.size) else float("nan")
                print(f"[PhaseD-Q][MIX_CORE] lam={float(lam):.6f} tau={float(tau):.6f} n_cand={int(mixed.size)} "
                      f"pi_top={_pi_top} q_top={_q_top} mix_top={_mix_top} chosen={_chosen}:{_chosen_p:.6f}",
                      flush=True)
        except Exception:
            pass


        # 念のためもう一度正規化
        msum = float(mixed.sum())
        if not np.isfinite(msum) or msum <= 0.0:
            return pi
        mixed = mixed / msum

        return mixed.astype(np.float32).tolist()
    except Exception:
        # 例外が出た場合は元の π をそのまま返して安全側に倒す
        return pi

def _dump_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return path

def _build_default_card_vocab():
    # Cards Enum から安定（ID昇順）で {card_id:int -> idx:int} を作る
    ids = []
    for c in Cards:
        v = c.value[0]
        try:
            cid = int(v if not isinstance(v, (list, tuple)) else v[0])
            ids.append(cid)
        except Exception:
            continue
    ids = sorted(set(ids))
    return {int(cid): i for i, cid in enumerate(ids)}

def _build_default_action_types():
    # ActionType Enum を ID昇順で並べた名前リスト（例: ["end_turn", "play_bench", ...]）
    pairs = []
    for t in ActionType:
        try:
            pairs.append((int(t.value), str(t.name).lower()))
        except Exception:
            pass
    pairs.sort(key=lambda x: x[0])
    return [name for _, name in pairs]

def bootstrap_artifacts_if_missing(out_dir: str):
    """
    学習成果物が無くても走れる最低限:
      - card_id2idx.json
      - action_types.json
      - scaler.npz（恒等）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) card_id2idx.json
    vocab_path = os.path.join(out_dir, "card_id2idx.json")
    if not os.path.exists(vocab_path):
        vocab = _build_default_card_vocab()
        _dump_json(vocab_path, vocab)
        print(f"[BOOT] wrote default card_id2idx.json → {vocab_path} (size={len(vocab)})")

    # 2) action_types.json
    atypes_path = os.path.join(out_dir, "action_types.json")
    if not os.path.exists(atypes_path):
        types = _build_default_action_types()
        _dump_json(atypes_path, types)
        print(f"[BOOT] wrote default action_types.json → {atypes_path} (K={len(types)})")

    # 3) scaler.npz（恒等）… 外部ヒントで次元を決めて作成（未指定は1）
    scaler_path = os.path.join(out_dir, "scaler.npz")
    obs_dim = int(os.getenv("OBS_DIM_HINT", "1"))
    _ensure_scaler_mask(scaler_path, obs_dim, out_dir)

    return vocab_path, atypes_path, scaler_path

# =======================================================
# === 推論側アクション埋め込み（学習と同一定義）を追加  ===
# =======================================================
import numpy as np

TYPE_SCHEMAS = { 7: ["stack_index"] }  # 進化(EVOLVE)
MAX_ARGS = max((len(v) for v in ACTION_SCHEMAS.values()), default=3)
# ランタイム既定値（apply_train_schema のロールバック用）
_RUNTIME_DEFAULTS = {
    "ACTION_SCHEMAS": dict(ACTION_SCHEMAS),
    "TYPE_SCHEMAS": dict(TYPE_SCHEMAS),
    "MAX_ARGS": int(MAX_ARGS),
}

# ▼追加: 学習アーティファクトの指紋と完全一致チェック（train_meta.json を検証）
import hashlib
import datetime as _dt

STRICT_SCHEMA_CHECK = os.getenv("STRICT_SCHEMA_CHECK", "0") == "1"

# ==== 行動候補を 32 次元ベクトルとして出力する設定（action_candidates_vec / action_vec_dim） ====
EMIT_CANDIDATE_FEATURES = True          # True なら action_candidates_vec / action_vec_dim を各レコードに出力
EMIT_OBS_VEC_FOR_CANDIDATES = True      # True なら obs_vec も各レコードに出力（学習を 1 ファイルで完結させたい場合に便利）

# 追加: 空の legal_actions はキーごと出さない
DROP_EMPTY_LEGAL_ACTIONS = True

def _safe_encode_obs_for_candidates(sb_src, encoder, la_ids=None):
    """obs_vec を『必ず』生成し、0次元だった場合はその場で警告を出す。"""
    try:
        me  = sb_src.get("me",  {}) if isinstance(sb_src, dict) else {}
        opp = sb_src.get("opp", {}) if isinstance(sb_src, dict) else {}
        feat = {"me": me, "opp": opp}
        if isinstance(la_ids, list) and la_ids:
            feat["legal_actions"] = la_ids

        out = encoder.encode_state(feat)
        try:
            import numpy as _np
            arr = _np.asarray(out, dtype=_np.float32).reshape(-1)
            if arr.size == 0:
                print("[OBS] ⚠️ encoder から 0次元ベクトルが返りました（scaler不在/不一致の可能性）")
            return arr.tolist()
        except Exception:
            if isinstance(out, list) and len(out) == 0:
                print("[OBS] ⚠️ obs_vec が [] です（encoder を確認してください）")
            return out if isinstance(out, list) else []
    except Exception as e:
        print(f"[OBS] encode_state failed (with legal_actions): {e}")
        # legal_actions 無しで再トライ
        try:
            out = encoder.encode_state({"me": me, "opp": opp})
            try:
                import numpy as _np
                arr = _np.asarray(out, dtype=_np.float32).reshape(-1)
                if arr.size == 0:
                    print("[OBS] ⚠️ encoder から 0次元ベクトルが返りました（fallback, no legal_actions）")
                return arr.tolist()
            except Exception:
                return out if isinstance(out, list) else []
        except Exception:
            return []

def _embed_legal_actions_32d(la_ids):  # pyright: ignore[reportUnusedFunction]
    outs = []
    try:
        import numpy as np
    except Exception:
        return []

    TARGET_DIM = 32

    def _zeros():
        return [0.0] * TARGET_DIM

    def _to_id_vec(a):
        if isinstance(a, list) and len(a) > 0:
            return a
        if isinstance(a, tuple) and len(a) > 0:
            return list(a)
        if isinstance(a, int):
            return [a]
        if isinstance(a, dict):
            v = a.get("id")
            if isinstance(v, int):
                return [v]
            return None
        return None

    src = (la_ids or [])
    for a in src:
        a_vec = _to_id_vec(a)
        if not isinstance(a_vec, list) or not a_vec:
            outs.append(_zeros())
            continue

        try:
            v = encode_action_from_vec_32d(a_vec)
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception:
            outs.append(_zeros())
            continue

        if arr.size < TARGET_DIM:
            pad = np.zeros(TARGET_DIM - arr.size, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.size > TARGET_DIM:
            arr = arr[:TARGET_DIM]

        # “全 -1” は無効候補として 0 に落とす（全滅はさせない）
        try:
            if arr.size == TARGET_DIM and bool(np.all(np.isfinite(arr))) and bool(np.all(np.abs(arr + 1.0) <= 1e-9)):
                outs.append(_zeros())
                continue
        except Exception:
            pass

        # NaN/Inf も 0 に落とす
        try:
            if not bool(np.all(np.isfinite(arr))):
                outs.append(_zeros())
                continue
        except Exception:
            outs.append(_zeros())
            continue

        outs.append(arr.astype(np.float32).tolist())

    return outs

def _pad_action_vecs_to_dim(vecs, target_dim):
    """Policy/model 入力用に action vec を target_dim へ 0 パディング/切詰めする。"""
    if not vecs:
        return vecs
    outs = []
    for v in vecs:
        if isinstance(v, np.ndarray):
            vv = v.reshape(-1).tolist()
        elif isinstance(v, (list, tuple)):
            vv = list(v)
        else:
            outs.append(v)
            continue
        if len(vv) < target_dim:
            vv = vv + [0] * (target_dim - len(vv))
        elif len(vv) > target_dim:
            vv = vv[:target_dim]
        outs.append(vv)
    return outs

def _fingerprint_action_schema():
    """
    学習側と完全一致の材料・直列化：
      spec = {"TYPE_SCHEMAS": <list>, "MAX_ARGS": <int>}
    を json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    """
    try:
        with open(_action_types_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            types = obj
        elif isinstance(obj, dict):
            try:
                items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
            except Exception:
                items = sorted(obj.items(), key=lambda x: str(x[0]))
                items = [(k, v) for k, v in items]
            types = [v for _, v in items]
        else:
            types = []

        spec = {"TYPE_SCHEMAS": types, "MAX_ARGS": int(MAX_ARGS)}
        blob = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest(), spec
    except Exception as e:
        print(f"[CHECK] schema 指紋生成エラー: {e}")
        return None, None

# ★ 追加: 起動時に一度だけスキーマ指紋を計算してログに出す（未使用警告の抑止も兼ねる）
_atp = globals().get("_action_types_path", None)
if _atp:
    _schema_sha, _schema_spec = _fingerprint_action_schema()
else:
    _schema_sha, _schema_spec = (None, {"error": "action_types_path is not set yet"})
if _schema_sha:
    print(f"[SYNC] action_schema_sha={_schema_sha}")

# どこか共通ユーティリティ付近（_assert_scaler_mask の近く）に追加
def load_or_make_scaler(path: str, obs_dim: int):
    import os, numpy as np
    # 1) 既存のスケーラファイルがあれば、それを優先して使用する
    if os.path.exists(path):
        try:
            data = np.load(path)
            mean, std = data["mean"], data["std"]
            d_file = int(mean.shape[0])
            if obs_dim and obs_dim != d_file:
                print(f"[SCALER] dim mismatch: file={d_file} obs={obs_dim} → identity scaler を作り直します")
                raise ValueError("scaler dim mismatch")
            return mean.astype(np.float32), std.astype(np.float32)
        except Exception as e:
            print(f"[SCALER] load failed ({e}) → identity scaler を新規作成します")

    # 2) ファイルが存在しないか、読み込みに失敗した場合のみ identity scaler を作成
    if obs_dim <= 0:
        raise RuntimeError(
            f"[SCALER] scaler.npz が存在せず obs_dim={obs_dim} です。"
            " 特徴次元を推定できないため identity scaler を自動生成できません。"
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mean = np.zeros(obs_dim, np.float32)
    std  = np.ones (obs_dim, np.float32)
    np.savez(path, mean=mean, std=std)
    print(f"[SCALER] wrote identity scaler: {path} (dim={obs_dim})")
    return mean, std

# 旧 _assert_scaler_mask を差し替え
def _ensure_scaler_mask(scaler_path: str, state_dim: int, model_dir: str):
    import os, numpy as np
    # スケーラは必ず存在させるが、既存ファイルがあればその次元を優先
    mean, std = load_or_make_scaler(scaler_path, state_dim)
    scaler_dim = int(mean.shape[0])
    if state_dim and state_dim != scaler_dim:
        print(f"[SCALER] warning: state_dim={state_dim} と scaler_dim={scaler_dim} が不一致です → scaler_dim を採用します")
    state_dim = scaler_dim

    # 任意の obs_mask.npy があれば形だけ合わせる（無ければ identity を作る）
    mask_path = os.path.join(model_dir, "obs_mask.npy")
    if os.path.exists(mask_path):
        try:
            mask = np.load(mask_path).astype(np.bool_)
            if mask.ndim != 1 or mask.shape[0] != state_dim:
                print(f"[SCALER] obs_mask 形状不一致: {mask.shape} vs {state_dim} → identity で再生成")
                mask = np.ones(state_dim, dtype=np.bool_)
                np.save(mask_path, mask)
                print(f"[SCALER] wrote obs_mask.npy: {mask_path} (dim={state_dim})")
        except Exception as e:
            print(f"[SCALER] obs_mask load failed ({e}) → identity を作成")
            mask = np.ones(state_dim, dtype=np.bool_)
            np.save(mask_path, mask)
            print(f"[SCALER] wrote obs_mask.npy: {mask_path} (dim={state_dim})")
    else:
        if state_dim > 0:
            mask = np.ones(state_dim, dtype=np.bool_)
            np.save(mask_path, mask)
            print(f"[SCALER] wrote obs_mask.npy: {mask_path} (dim={state_dim})")

    return mean, std

def _fingerprint_vocab_from_json_file(path: str):
    """
    学習時と同じ正規化（sort_keys=True, separators=(',', ':'), ensure_ascii=True）
    で JSON を直列化したバイト列に対する SHA256 を返す。
    """
    import json, hashlib, os
    if not os.path.exists(path):
        return None, 0
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)  # 型はいじらない
    blob = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), len(obj)

def _fingerprint_scaler_npz(path: str):
    try:
        import numpy as _np
        if not os.path.exists(path):
            return None, {}
        data = _np.load(path, allow_pickle=False)
        keys = sorted(list(data.files))
        h = hashlib.sha256()
        shapes = {}
        for k in keys:
            arr = data[k]
            h.update(k.encode())
            h.update(arr.tobytes())
            shapes[k] = tuple(arr.shape)
        return h.hexdigest(), shapes
    except Exception:
        # numpy が無い / 破損ファイル / 読み込み失敗などは None, {} を返す
        return None, {}

def _load_train_meta(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[CHECK] train_meta.json not found: {path}")
        return None
    except Exception as e:
        print(f"[CHECK] train_meta.json read error: {e}")
        return None

def _strip_privates_recursive(x):
    """
    dict/list を再帰走査して、キー名が *_private で終わるもの
    および 'private' 単独キーを完全に除去する。
    """
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(k, str) and (k.endswith("_private") or k == "private"):
                continue  # ← ここで完全にドロップ
            out[k] = _strip_privates_recursive(v)
        return out
    if isinstance(x, list):
        return [_strip_privates_recursive(v) for v in x]
    return x

# 学習で保存した card_id2idx.json をロード（one-hot と末尾スカラーの両方で使用）
import json as _json, os

def _resolve_card_idx_path_or_bootstrap():
    # まず既存の探索候補
    env_p = os.getenv("CARD_ID2IDX_PATH")
    cands = []
    if env_p:
        cands.append(env_p)

    # online_mix / az_mcts でも MODEL_DIR_* 配下の vocab が本命になることがあるので常に候補に入れる
    try:
        if MODEL_DIR_P1:
            cands.append(os.path.join(MODEL_DIR_P1, "card_id2idx.json"))
    except Exception:
        pass
    try:
        if MODEL_DIR_P2:
            cands.append(os.path.join(MODEL_DIR_P2, "card_id2idx.json"))
    except Exception:
        pass

    # 互換: 明示的に model のときも（重複しても後で dedup する）
    if str(P1_POLICY).lower() == "model":
        cands.append(os.path.join(MODEL_DIR_P1, "card_id2idx.json"))
    if str(P2_POLICY).lower() == "model":
        cands.append(os.path.join(MODEL_DIR_P2, "card_id2idx.json"))

    cands += [
        "d3rlpy_logs/run_p1/card_id2idx.json",
        "d3rlpy_logs/run_p2/card_id2idx.json",
    ]

    # 期待 obs_dim は「診断用」に推定する（vocabファイル選択の正解件数とは切り離す）
    def _infer_expected_obs_dim():
        for _k in ("PHASED_Q_OBS_DIM", "PHASED_Q_EXPECTED_OBS_DIM", "PHASED_Q_MODEL_OBS_DIM"):
            try:
                _v = globals().get(_k, None)
            except Exception:
                _v = None
            try:
                _vi = int(_v) if _v is not None else 0
            except Exception:
                _vi = 0
            if _vi > 0:
                return _vi
        try:
            _env = os.getenv("PHASED_Q_EXPECTED_OBS_DIM")
            _vi = int(_env) if _env else 0
            if _vi > 0:
                return _vi
        except Exception:
            pass
        return 2448

    def _infer_slot_vocab_from_obs_dim(_obs_dim):
        try:
            od = int(_obs_dim)
        except Exception:
            return None, None
        if od > 0 and (od - 8) % 4 == 0:
            return int((od - 8) // 4), "4*V+8"
        if od > 0 and (od - 7) % 4 == 0:
            return int((od - 7) // 4), "4*V+7"
        return None, None

    _exp_obs_dim = _infer_expected_obs_dim()
    _exp_slot_vocab, _exp_formula = _infer_slot_vocab_from_obs_dim(_exp_obs_dim)

    # 後段のログ用に残す（選択ロジックの正解件数には使わない）
    try:
        globals()["_PHASED_Q_EXP_OBS_DIM"] = int(_exp_obs_dim)
    except Exception:
        globals()["_PHASED_Q_EXP_OBS_DIM"] = _exp_obs_dim
    globals()["_PHASED_Q_EXP_SLOT_VOCAB"] = _exp_slot_vocab
    globals()["_PHASED_Q_EXP_SLOT_FORMULA"] = _exp_formula

    # 重複除去（順序維持）
    _seen = set()
    _cands2 = []
    for p in cands:
        if not p:
            continue
        try:
            key = os.path.abspath(p)
        except Exception:
            key = str(p)
        if key in _seen:
            continue
        _seen.add(key)
        _cands2.append(p)
    cands = _cands2

    # ここは「優先順位どおりに存在するものを採用」に戻す（obs_dim→vocab件数の混線を避ける）
    for p in cands:
        if p and os.path.exists(p):
            if LOG_DEBUG_DETAIL:
                try:
                    with open(p, "r", encoding="utf-8") as _f:
                        _d = _json.load(_f)
                    _vsz = (len(_d) if isinstance(_d, dict) else -1)
                except Exception:
                    _vsz = -1
                try:
                    print(
                        f"[SYNC] vocab pick: path={p} vocab_size={_vsz}"
                        f" exp_obs_dim={_exp_obs_dim} exp_slot_vocab={_exp_slot_vocab} ({_exp_formula})"
                    )
                except Exception:
                    pass
            return p

    # ここに来たら存在しない → コールドスタートで生成
    if COLD_START:
        out_dir = MODEL_DIR_P1  # ひとまず run_p1 側に揃える
        vocab_path, _, _ = bootstrap_artifacts_if_missing(out_dir)
        return vocab_path
    raise FileNotFoundError("card_id2idx.json not found and COLD_START=0")

_card_idx_path = _resolve_card_idx_path_or_bootstrap()
with open(_card_idx_path, "r", encoding="utf-8") as _f:
    _CARD_ID2IDX = {int(k): int(v) for k, v in _json.load(_f).items()}
if LOG_DEBUG_DETAIL:
    try:
        _vsz = (len(_CARD_ID2IDX) if isinstance(_CARD_ID2IDX, dict) else -1)
        _max_idx = (max(_CARD_ID2IDX.values()) if isinstance(_CARD_ID2IDX, dict) and _CARD_ID2IDX else -1)
        _exp_obs_dim = globals().get("_PHASED_Q_EXP_OBS_DIM", None)
        _exp_slot_vocab = globals().get("_PHASED_Q_EXP_SLOT_VOCAB", None)
        _exp_formula = globals().get("_PHASED_Q_EXP_SLOT_FORMULA", None)

        print("[SYNC] vocab from:", _card_idx_path, "size=", _vsz, "max_idx=", _max_idx)

        # 重要: ここは「カードプールの実数」ではなく「観測ベクトルのスロット設計」側の診断
        if _exp_obs_dim is not None:
            print("[SYNC] expected obs_dim (diagnostic):", _exp_obs_dim, "slot_vocab=", _exp_slot_vocab, "formula=", _exp_formula)
    except Exception:
        print("[SYNC] vocab from:", _card_idx_path)

# ===== 共通32次元エンコーダの構築 =====
# 学習の成果物と同じファイルを使う（仕様2）
_action_types_path_candidates = []
if str(P1_POLICY).lower() == "model":
    _action_types_path_candidates.append(os.path.join(MODEL_DIR_P1, "action_types.json"))
if str(P2_POLICY).lower() == "model":
    _action_types_path_candidates.append(os.path.join(MODEL_DIR_P2, "action_types.json"))
_action_types_path_candidates += [
    "d3rlpy_logs/run_p1/action_types.json",
    "d3rlpy_logs/run_p2/action_types.json",
]

_action_types_path = next((p for p in _action_types_path_candidates if os.path.exists(p)), None)
if not _action_types_path:
    if COLD_START:
        # card_id2idx と同じ場所（run_p1）で作る
        out_dir = os.path.dirname(_card_idx_path) or MODEL_DIR_P1
        _, atypes_path, _ = bootstrap_artifacts_if_missing(out_dir)
        _action_types_path = atypes_path
    else:
        raise FileNotFoundError("action_types.json が見つかりません（COLD_START=0）")

# TYPE_SCHEMAS / MAX_ARGS は既存定義を使用（なければフォールバック）
try:
    from pokepocketsim.action import TYPE_SCHEMAS as _TYPE_SCHEMAS_ORIG
except Exception:
    _TYPE_SCHEMAS_ORIG = {}
TYPE_SCHEMAS.update(_TYPE_SCHEMAS_ORIG)
try:
    from pokepocketsim.action import MAX_ARGS as _MAX_ARGS_ORIG
except Exception:
    _MAX_ARGS_ORIG = 3
MAX_ARGS = max(MAX_ARGS, _MAX_ARGS_ORIG, 3)

# ★ 追加: モデル/MCTS デバッグ用フラグ（デフォルトOFF）
DEBUG_MODEL_MCTS = os.getenv("DEBUG_MODEL_MCTS", "1") == "1"
DEBUG_MODEL_MCTS_EVERY = int(os.getenv("DEBUG_MODEL_MCTS_EVERY", "1"))

def _trace_policy_step(tag, pol, obs_vec, la_ids, chosen_vec=None, pi=None):
    """
    MCTS/モデルの出力が本当に出ているかを最小コストで可視化するトレース。
    DEBUG_MODEL_MCTS=1 のときのみ出力。
    """
    if not DEBUG_MODEL_MCTS:
        return
    try:
        import math
        n_cands = len(la_ids) if isinstance(la_ids, list) else 0

        pi_stats = ""
        if isinstance(pi, list) and pi:
            s = float(sum(pi))
            mx = float(max(pi))
            arg = int(max(range(len(pi)), key=lambda i: pi[i]))
            ent = 0.0
            for p in pi:
                pp = float(p)
                if pp > 0:
                    ent -= pp * math.log(pp + 1e-12)
            pi_stats = f" sum={s:.3f} max={mx:.3f} argmax={arg} H={ent:.3f}"

        cv = chosen_vec if chosen_vec is not None else ""
        print(f"[MCTS_TRACE] {tag} pol={type(pol).__name__} n_cands={n_cands}{pi_stats} chosen={cv}")
    except Exception as e:
        print(f"[MCTS_TRACE] {tag} trace failed: {e}")

def _wrap_select_action_for_trace(pol, tag):
    """
    対戦中の select_action 呼び出しそのものをトレースするラッパ。
    （MCTSが本当に使われているかの決定打）
    """
    if not DEBUG_MODEL_MCTS:
        return
    try:
        if not hasattr(pol, "select_action"):
            return
        if getattr(pol, "_select_action_wrapped", False):
            return

        orig = pol.select_action
        def wrapped(obs_vec, la_ids, *args, **kwargs):
            a_vec, pi = orig(obs_vec, la_ids, *args, **kwargs)
            _trace_policy_step(f"{tag}/select_action", pol, obs_vec, la_ids, a_vec, pi)
            return a_vec, pi

        pol.select_action = wrapped
        pol._select_action_wrapped = True
    except Exception:
        pass

# ★ 最小版: PhaseD Q をオンラインで π にミックスするラッパ（混ぜるだけ）
def _wrap_select_action_with_phased_q(pol, tag):
    try:
        if not USE_PHASED_Q:
            return
        if not PHASED_Q_MIX_ENABLED:
            return
    except NameError:
        return

    try:
        if getattr(pol, "_phased_q_wrapped", False):
            try:
                pol._phased_q_tag = tag
                pol.phased_q_tag = tag
            except Exception:
                pass
            return
    except Exception:
        pass

    _main = getattr(pol, "main", None) or getattr(pol, "main_policy", None)
    try:
        if not (getattr(pol, "use_mcts", False) or getattr(_main, "use_mcts", False)):
            return
    except Exception:
        return

    _entrypoints = ("select_action_index_online", "select_action_index", "select_action", "act", "__call__", "get_action", "choose_action")
    _callable_eps = [n for n in _entrypoints if callable(getattr(pol, n, None))]
    if not _callable_eps:
        return

    _callable_eps = _callable_eps[:1]

    def _as_list(v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        try:
            tl = getattr(v, "tolist", None)
        except Exception:
            tl = None
        if callable(tl):
            try:
                vv = tl()
            except Exception:
                vv = None
            if isinstance(vv, list):
                return vv
            if isinstance(vv, tuple):
                return list(vv)
        return None

    def _is_numeric_vec(v):
        v = _as_list(v)
        if not isinstance(v, (list, tuple)) or len(v) <= 0:
            return False
        for x in v:
            try:
                float(x)
            except Exception:
                return False
        return True

    def _topk_pairs(vals, k=3):
        try:
            pairs = []
            for i, v in enumerate(vals):
                try:
                    pairs.append((int(i), float(v)))
                except Exception:
                    pairs.append((int(i), 0.0))
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:max(1, int(k))]
            return ",".join([f"{i}:{v:.3f}" for i, v in pairs])
        except Exception:
            return ""

    def _normalize_base_pi(pi, n):
        if isinstance(pi, dict) and "pi" in pi:
            pi = pi["pi"]

        if isinstance(pi, (list, tuple)):
            out = []
            for x in list(pi)[:n]:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(0.0)
            if len(out) < n:
                out += [0.0] * (n - len(out))
            return out

        if isinstance(pi, dict):
            out = [0.0] * n
            for k, v in pi.items():
                if isinstance(k, int) and 0 <= k < n:
                    try:
                        out[k] = float(v)
                    except Exception:
                        pass
            return out

        return None

    def _normalize_cand_vecs_32d(cand, n_expect):
        if cand is None:
            return None
        if not isinstance(cand, (list, tuple)) or not cand:
            return None
        try:
            if len(cand) != n_expect:
                return None
        except Exception:
            return None

        out = []
        for row in cand:
            if isinstance(row, (list, tuple)):
                if len(row) == 32:
                    out.append([float(x) for x in row])
                    continue
                if len(row) == 17:
                    rr = [float(x) for x in row] + [0.0] * (32 - 17)
                    out.append(rr)
                    continue
                return None
            return None
        return out

    def _make_cand_vecs_32d(la_list, kwargs):
        # 0) 第2引数 la_list 自体が 32D/17D 候補ベクトルなら、それを最優先で採用
        cand0 = _normalize_cand_vecs_32d(la_list, len(la_list))
        if cand0 is not None:
            return cand0

        # 1) 呼び出し元が 32D を渡す（最優先）
        cand = (
            kwargs.get("action_candidates_vec", None)
            or kwargs.get("action_candidates_vecs", None)
            or kwargs.get("cand_vecs", None)
            or kwargs.get("candidates_vec", None)
        )
        cand = _normalize_cand_vecs_32d(cand, len(la_list))
        if cand is not None:
            return cand

        # 2) converter から作る（5→32 or 5→17→32 をここに寄せる）
        conv = kwargs.get("converter", None) or kwargs.get("action_converter", None)
        if conv is None:
            try:
                conv = getattr(pol, "converter", None) or getattr(pol, "action_converter", None)
            except Exception:
                conv = None

        if conv is not None:
            fn32 = getattr(conv, "convert_legal_actions_32d", None)
            if callable(fn32):
                try:
                    cand2 = fn32(la_list)
                except Exception:
                    cand2 = None
                cand2 = _normalize_cand_vecs_32d(cand2, len(la_list))
                if cand2 is not None:
                    return cand2

            fn = getattr(conv, "convert_legal_actions", None)
            if callable(fn):
                try:
                    cand3 = fn(la_list)
                except Exception:
                    cand3 = None
                cand3 = _normalize_cand_vecs_32d(cand3, len(la_list))
                if cand3 is not None:
                    return cand3

        return None

    def _wrap_one(ep_name):
        orig = getattr(pol, ep_name)

        def _phased_q_emit_summary(reason="game_over", reset=True):
            if bool(getattr(pol, "_phased_q_summary_emitted", False)):
                return
            setattr(pol, "_phased_q_summary_emitted", True)

            st = getattr(pol, "_phased_q_stats", None)
            if not isinstance(st, dict):
                return
            calls_total = int(st.get("calls_total", 0))
            if calls_total <= 0:
                return

            calls_q_used = int(st.get("calls_q_used", 0))
            calls_q_eval_none = int(st.get("calls_q_eval_none", 0))

            sk_obs = int(st.get("skip_obs_not_numeric", 0))
            sk_la_missing = int(st.get("skip_la_list_missing", 0))
            sk_la_empty = int(st.get("skip_la_list_empty", 0))
            sk_cand = int(st.get("skip_cand_vecs_missing", 0))
            sk_ep = int(st.get("skip_ep_select_action_index_online", 0))

            la_n = int(st.get("la_len_n", 0))
            la_sum = float(st.get("la_len_sum", 0.0))
            la_min = st.get("la_len_min", None)
            la_max = st.get("la_len_max", None)
            la_avg = (la_sum / float(la_n)) if la_n > 0 else 0.0

            mix_changed = int(st.get("mix_changed", 0))
            mix_same = int(st.get("mix_same", 0))
            mix_mcts_idx_none = int(st.get("mix_mcts_idx_none", 0))
            mix_change_rate = (float(mix_changed) / float(calls_q_used)) if calls_q_used > 0 else 0.0

            pi_changed = int(st.get("pi_changed", 0))
            pi_l1_n = int(st.get("pi_l1_n", 0))
            pi_l1_sum = float(st.get("pi_l1_sum", 0.0))
            pi_l1_avg = (pi_l1_sum / float(pi_l1_n)) if pi_l1_n > 0 else 0.0
            pi_change_rate = (float(pi_changed) / float(pi_l1_n)) if pi_l1_n > 0 else 0.0

            print(
                f"[PhaseD-Q][SUMMARY] tag={tag} reason={reason}"
                f" calls_total={calls_total}"
                f" q_used={calls_q_used} q_eval_none={calls_q_eval_none}"
                f" mix_changed={mix_changed} mix_same={mix_same} mix_mcts_idx_none={mix_mcts_idx_none}"
                f" mix_change_rate={mix_change_rate:.3f}"
                f" pi_changed={pi_changed} pi_change_rate={pi_change_rate:.3f} pi_l1_avg={pi_l1_avg:.6f}"
                f" skip_obs_not_numeric={sk_obs} skip_la_missing={sk_la_missing}"
                f" skip_la_empty={sk_la_empty} skip_cand_missing={sk_cand}"
                f" skip_ep_select_action_index_online={sk_ep}"
                f" la_len_avg={la_avg:.2f} la_len_min={la_min} la_len_max={la_max}"
                ,
                flush=True,
            )

            if reset:
                st.clear()

        if not hasattr(pol, "phaseD_q_emit_summary"):
            setattr(pol, "phaseD_q_emit_summary", _phased_q_emit_summary)

        if not getattr(pol, "_phased_q_stats_atexit_registered", False):
            try:
                import atexit as _atexit
                _atexit.register(lambda: _phased_q_emit_summary(reason="atexit", reset=False))
                setattr(pol, "_phased_q_stats_atexit_registered", True)
            except Exception:
                setattr(pol, "_phased_q_stats_atexit_registered", True)

        def wrapped(*args, **kwargs):
            _LOG_DETAIL = bool(globals().get("LOG_DEBUG_DETAIL", False))

            st = getattr(pol, "_phased_q_stats", None)
            if not isinstance(st, dict):
                st = {}
                setattr(pol, "_phased_q_stats", st)

            st["calls_total"] = int(st.get("calls_total", 0)) + 1

            ret = orig(*args, **kwargs)

            if isinstance(ret, tuple) and len(ret) == 2:
                base_out, pi = ret
            else:
                base_out, pi = ret, None

            try:
                _log_decide = bool(globals().get("LOG_DECIDE_ALWAYS", True))
            except Exception:
                _log_decide = True

            _t = None
            _pl = None
            try:
                _t = kwargs.get("t", None) or kwargs.get("turn", None) or kwargs.get("turn_i", None) or kwargs.get("step", None) or kwargs.get("ply", None)
            except Exception:
                _t = None
            try:
                _pl = kwargs.get("player_name", None) or kwargs.get("player", None)
            except Exception:
                _pl = None
            if _pl is None:
                try:
                    _pl = getattr(pol, "player_name", None) or getattr(pol, "_player_name", None)
                except Exception:
                    _pl = None

            _decide_pre_line = None
            _decide_post_line = None

            if _log_decide:
                try:
                    _pi0 = pi
                    if _pi0 is None and ep_name == "select_action_index_online":
                        _sd0 = None
                        try:
                            if len(args) >= 1 and isinstance(args[0], dict):
                                _sd0 = args[0]
                        except Exception:
                            _sd0 = None
                        if _sd0 is None:
                            _sd0 = kwargs.get("state_dict", None) if isinstance(kwargs.get("state_dict", None), dict) else None
                        if isinstance(_sd0, dict):
                            _pi0 = _sd0.get("mcts_pi", None) or _sd0.get("pi", None)

                    _pi_len = len(_pi0) if isinstance(_pi0, list) else (len(_pi0.get("pi")) if isinstance(_pi0, dict) and isinstance(_pi0.get("pi"), list) else "NA")
                except Exception:
                    _pi_len = "NA"
                try:
                    _bo_t = type(base_out).__name__
                except Exception:
                    _bo_t = "<?>"
                _decide_pre_line = f"[DECIDE_PRE] tag={tag} ep={ep_name} t={_t if _t is not None else 'NA'} player={_pl if _pl is not None else 'NA'} base_out_type={_bo_t} pi_len={_pi_len}"
                try:
                    setattr(pol, "_last_decide_pre_line", _decide_pre_line)
                except Exception:
                    pass

            # entrypoint ごとに obs_vec / la_list を取り出す（推測を増やさない）
            obs_vec = None
            la_list = None

            if ep_name == "select_action_index_online":
                # ここは state_dict から拾う（スキップしない）
                state_dict = None
                if len(args) >= 1 and isinstance(args[0], dict):
                    state_dict = args[0]
                if state_dict is None:
                    state_dict = kwargs.get("state_dict", None) if isinstance(kwargs.get("state_dict", None), dict) else None

                kw2 = dict(kwargs)
                if isinstance(state_dict, dict):
                    for _k, _v in state_dict.items():
                        if _k not in kw2:
                            kw2[_k] = _v

                # ★追加: OnlineMixedPolicy 側が埋めた mcts_pi / pi を優先して拾う（base_pi の一様落ちを防ぐ）
                try:
                    if pi is None:
                        pi = kw2.get("mcts_pi", None) or kw2.get("pi", None)
                except Exception:
                    pass

                # player / converter を可能な限り回収（legal_actions の serialize と obs 補完に使う）
                _player = kw2.get("player", None)
                try:
                    if _player is None:
                        _player = kw2.get("pl", None) or kw2.get("player_obj", None)
                except Exception:
                    pass

                _conv = kw2.get("converter", None)
                try:
                    if _conv is None:
                        _conv = kw2.get("action_converter", None) or kw2.get("converter_obj", None)
                except Exception:
                    pass
                try:
                    if _conv is None:
                        _conv = getattr(pol, "converter", None) or getattr(pol, "action_converter", None)
                except Exception:
                    pass
                try:
                    if _conv is not None and "converter" not in kw2:
                        kw2["converter"] = _conv
                except Exception:
                    pass

                try:
                    if _t is None:
                        _t = state_dict.get("t", None) or state_dict.get("turn", None) or state_dict.get("turn_i", None) or state_dict.get("step", None) or state_dict.get("ply", None)
                except Exception:
                    pass
                try:
                    if _pl is None:
                        _pl = state_dict.get("player_name", None) or state_dict.get("player", None)
                except Exception:
                    pass

                # obs_vec は state_dict 優先、次に kwargs、最後に converter で補完
                obs_vec = _as_list(
                    (state_dict.get("obs_vec", None) if isinstance(state_dict, dict) else None)
                    or (state_dict.get("obs", None) if isinstance(state_dict, dict) else None)
                    or (state_dict.get("public_obs_vec", None) if isinstance(state_dict, dict) else None)
                    or (state_dict.get("full_obs_vec", None) if isinstance(state_dict, dict) else None)
                    or kw2.get("obs_vec", None)
                    or kw2.get("obs", None)
                    or kw2.get("public_obs_vec", None)
                    or kw2.get("full_obs_vec", None)
                )

                # legal_actions を拾う（args[1] -> state_dict -> serialize(player) で 5-int へ）
                if len(args) >= 2 and la_list is None:
                    la_list = args[1]

                if la_list is None and isinstance(state_dict, dict):
                    la_list = (
                        state_dict.get("legal_actions", None)
                        or state_dict.get("legal_actions_list", None)
                        or state_dict.get("legal_actions_vec", None)
                        or state_dict.get("legal_actions_vecs", None)
                        or state_dict.get("legal_actions_19d", None)
                        or state_dict.get("la_list", None)
                    )

                try:
                    if isinstance(la_list, (list, tuple)) and la_list:
                        _x0 = la_list[0]
                        if hasattr(_x0, "serialize") and _player is not None:
                            _tmp = []
                            for _a in list(la_list):
                                try:
                                    _tmp.append(_a.serialize(_player))
                                except Exception:
                                    _tmp = []
                                    break
                            if _tmp:
                                la_list = _tmp
                except Exception:
                    pass

                # 最終フォールバック：player 側の直近キャッシュ
                try:
                    if (la_list is None or not la_list) and _player is not None:
                        la_list = getattr(_player, "_last_legal_actions_before", None)
                except Exception:
                    pass

                # 下流の _make_cand_vecs_32d に届くよう kw2 に合流
                try:
                    if la_list is not None and "legal_actions_19d" not in kw2:
                        kw2["legal_actions_19d"] = la_list
                except Exception:
                    pass
                try:
                    if la_list is not None and "la_list" not in kw2:
                        kw2["la_list"] = la_list
                except Exception:
                    pass

                # converter で obs を補完（state_dict が UI 用で obs_vec が無いケースを救う）
                try:
                    if not _is_numeric_vec(obs_vec) and _conv is not None and isinstance(state_dict, dict):
                        _feat = dict(state_dict)
                        if la_list is not None and "legal_actions_19d" not in _feat:
                            _feat["legal_actions_19d"] = la_list
                        if la_list is not None and "legal_actions" not in _feat:
                            _feat["legal_actions"] = la_list

                        _vec = None
                        try:
                            _fn = getattr(_conv, "encode_state", None)
                            _vec = _fn(_feat) if callable(_fn) else None
                        except Exception:
                            _vec = None

                        if _vec is None:
                            try:
                                _fn = getattr(_conv, "convert_state", None)
                                _vec = _fn(_feat) if callable(_fn) else None
                            except Exception:
                                _vec = None

                        if _vec is None:
                            try:
                                _fn = getattr(_conv, "build_obs", None)
                                _vec = _fn(_feat) if callable(_fn) else None
                            except Exception:
                                _vec = None

                        obs_vec = _as_list(_vec)
                except Exception:
                    pass

                kwargs = kw2

                if la_list is None:
                    st["skip_la_list_missing"] = int(st.get("skip_la_list_missing", 0)) + 1
                    if _log_decide:
                        try:
                            _sd_keys = ",".join(list(state_dict.keys())[:15]) if isinstance(state_dict, dict) else "NA"
                        except Exception:
                            _sd_keys = "NA"
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=la_list_missing state_keys={_sd_keys}", flush=True)
                    return ret

                if not la_list:
                    st["skip_la_list_empty"] = int(st.get("skip_la_list_empty", 0)) + 1
                    if _log_decide:
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=la_list_empty", flush=True)
                    return ret

                if not _is_numeric_vec(obs_vec):
                    st["skip_obs_not_numeric"] = int(st.get("skip_obs_not_numeric", 0)) + 1
                    if _log_decide:
                        try:
                            _sd_keys = ",".join(list(state_dict.keys())[:15]) if isinstance(state_dict, dict) else "NA"
                        except Exception:
                            _sd_keys = "NA"
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=obs_not_numeric state_keys={_sd_keys}", flush=True)
                    return ret

                if la_list is None:
                    st["skip_la_list_missing"] = int(st.get("skip_la_list_missing", 0)) + 1
                    if _log_decide:
                        try:
                            _sd_keys = ",".join(list(state_dict.keys())[:15]) if isinstance(state_dict, dict) else "NA"
                        except Exception:
                            _sd_keys = "NA"
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=la_list_missing state_keys={_sd_keys}", flush=True)
                    return ret

                la_list = la_list if isinstance(la_list, list) else list(la_list)
                if not la_list:
                    st["skip_la_list_empty"] = int(st.get("skip_la_list_empty", 0)) + 1
                    if _log_decide:
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=la_list_empty", flush=True)
                    return ret

            else:
                if len(args) >= 1:
                    obs_vec = _as_list(args[0])
                if len(args) >= 2:
                    la_list = args[1]

                if not _is_numeric_vec(obs_vec):
                    obs_vec = _as_list(
                        kwargs.get("obs_vec", None)
                        or kwargs.get("obs", None)
                        or kwargs.get("public_obs_vec", None)
                        or kwargs.get("full_obs_vec", None)
                    )

                if la_list is None:
                    la_list = (
                        kwargs.get("legal_actions", None)
                        or kwargs.get("legal_actions_list", None)
                        or kwargs.get("legal_actions_vec", None)
                        or kwargs.get("legal_actions_vecs", None)
                        or kwargs.get("legal_actions_19d", None)
                        or kwargs.get("la_list", None)
                    )

                if not _is_numeric_vec(obs_vec):
                    st["skip_obs_not_numeric"] = int(st.get("skip_obs_not_numeric", 0)) + 1
                    _c = int(getattr(pol, "_phased_q_skip_count", 0))
                    if _LOG_DETAIL and _c < 20:
                        setattr(pol, "_phased_q_skip_count", _c + 1)
                        try:
                            _a0t = type(args[0]).__name__ if len(args) >= 1 else "None"
                        except Exception:
                            _a0t = "<?>"
                        try:
                            _a1t = type(args[1]).__name__ if len(args) >= 2 else "None"
                        except Exception:
                            _a1t = "<?>"
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=obs_not_numeric a0={_a0t} a1={_a1t} kw_obs={'obs_vec' in kwargs or 'obs' in kwargs}")
                    return ret

                if la_list is None:
                    st["skip_la_list_missing"] = int(st.get("skip_la_list_missing", 0)) + 1
                    _c = int(getattr(pol, "_phased_q_skip_count", 0))
                    if _LOG_DETAIL and _c < 20:
                        setattr(pol, "_phased_q_skip_count", _c + 1)
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=la_list_missing kw_keys_sample={','.join(list(kwargs.keys())[:10])}")
                    return ret

                la_list = la_list if isinstance(la_list, list) else list(la_list)
                if not la_list:
                    st["skip_la_list_empty"] = int(st.get("skip_la_list_empty", 0)) + 1
                    _c = int(getattr(pol, "_phased_q_skip_count", 0))
                    if _LOG_DETAIL and _c < 20:
                        setattr(pol, "_phased_q_skip_count", _c + 1)
                        print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=la_list_empty")
                    return ret

            _la_len = int(len(la_list))
            st["la_len_n"] = int(st.get("la_len_n", 0)) + 1
            st["la_len_sum"] = float(st.get("la_len_sum", 0.0)) + float(_la_len)
            if st.get("la_len_min", None) is None or _la_len < int(st.get("la_len_min", _la_len)):
                st["la_len_min"] = int(_la_len)
            if st.get("la_len_max", None) is None or _la_len > int(st.get("la_len_max", _la_len)):
                st["la_len_max"] = int(_la_len)

            cand_vecs_f = _make_cand_vecs_32d(la_list, kwargs)
            if cand_vecs_f is None:
                st["skip_cand_vecs_missing"] = int(st.get("skip_cand_vecs_missing", 0)) + 1
                _c = int(getattr(pol, "_phased_q_skip_count", 0))
                if _LOG_DETAIL and _c < 20:
                    setattr(pol, "_phased_q_skip_count", _c + 1)
                    print(f"[PhaseD-Q][SKIP] tag={tag} ep={ep_name} reason=cand_vecs_missing la_len={len(la_list)} kw_has_cand={('cand_vecs' in kwargs) or ('action_candidates_vec' in kwargs) or ('action_candidates_vecs' in kwargs)}")
                return ret

            try:
                obs_vec_f = [float(x) for x in obs_vec]
            except Exception:
                return ret

            q_vals = phaseD_q_evaluate(obs_vec_f, cand_vecs_f)
            if q_vals is None:
                st["calls_q_eval_none"] = int(st.get("calls_q_eval_none", 0)) + 1
                return ret

            base_pi = _normalize_base_pi(pi, len(la_list))
            if not isinstance(base_pi, list) or len(base_pi) != len(la_list):
                n = len(la_list)
                base_pi = [1.0 / float(n)] * n

            mixed_pi = phaseD_mix_pi_with_q(base_pi, q_vals)

            # 安全化
            try:
                if not isinstance(mixed_pi, list) or len(mixed_pi) != len(la_list):
                    raise ValueError("mixed_pi_bad_shape")
                w = []
                s = 0.0
                for p in mixed_pi:
                    try:
                        pf = float(p)
                    except Exception:
                        pf = 0.0
                    if not (pf > 0.0):
                        pf = 0.0
                    w.append(pf)
                    s += pf
                if not (s > 0.0):
                    raise ValueError("mixed_pi_sum0")
                mixed_pi = [x / s for x in w]
            except Exception:
                mixed_pi = base_pi

            try:
                _l1 = 0.0
                for _a, _b in zip(mixed_pi, base_pi):
                    _l1 += abs(float(_a) - float(_b))
            except Exception:
                _l1 = 0.0
            st["pi_l1_n"] = int(st.get("pi_l1_n", 0)) + 1
            st["pi_l1_sum"] = float(st.get("pi_l1_sum", 0.0)) + float(_l1)
            if float(_l1) > 1e-12:
                st["pi_changed"] = int(st.get("pi_changed", 0)) + 1

            import random as _random
            new_idx = _random.choices(range(len(la_list)), weights=mixed_pi, k=1)[0]
            a_vec_new = la_list[new_idx]

            if _log_decide:
                try:
                    _p_sel = float(mixed_pi[int(new_idx)]) if isinstance(mixed_pi, list) and 0 <= int(new_idx) < len(mixed_pi) else float("nan")
                except Exception:
                    _p_sel = float("nan")
                try:
                    _la_len_dbg = int(len(la_list))
                except Exception:
                    _la_len_dbg = -1
                _decide_post_line = f"[DECIDE_POST] tag={tag} ep={ep_name} t={_t if _t is not None else 'NA'} player={_pl if _pl is not None else 'NA'} la_len={_la_len_dbg} selected_idx={int(new_idx)} selected_p={_p_sel:.6f}"
                try:
                    setattr(pol, "_last_decide_post_line", _decide_post_line)
                except Exception:
                    pass

            st["calls_q_used"] = int(st.get("calls_q_used", 0)) + 1

            _lam = float(PHASED_Q_MIX_LAMBDA)
            _tau = float(PHASED_Q_MIX_TEMPERATURE)
            _mcts_top = _topk_pairs(base_pi, k=3)
            _q_top = _topk_pairs(q_vals, k=3)
            _mix_top = _topk_pairs(mixed_pi, k=3)

            _mcts_idx = None
            try:
                try:
                    import numpy as np
                    _INT_TYPES = (int, np.integer)
                except Exception:
                    _INT_TYPES = (int,)

                if isinstance(base_out, _INT_TYPES):
                    _mcts_idx = int(base_out)
                else:
                    if a_vec_new is base_out:
                        _mcts_idx = int(new_idx)
                    else:
                        _mcts_idx = int(la_list.index(base_out))
            except Exception:
                _mcts_idx = None

            _decide_diff_line = None

            if _mcts_idx is None:
                st["mix_mcts_idx_none"] = int(st.get("mix_mcts_idx_none", 0)) + 1
                if _log_decide:
                    _decide_diff_line = f"[DECIDE_DIFF] tag={tag} ep={ep_name} t={_t if _t is not None else 'NA'} player={_pl if _pl is not None else 'NA'} mcts_idx=None mix_idx={int(new_idx)} changed=NA"
            else:
                if int(new_idx) != int(_mcts_idx):
                    st["mix_changed"] = int(st.get("mix_changed", 0)) + 1
                    if _log_decide:
                        _decide_diff_line = f"[DECIDE_DIFF] tag={tag} ep={ep_name} t={_t if _t is not None else 'NA'} player={_pl if _pl is not None else 'NA'} mcts_idx={int(_mcts_idx)} mix_idx={int(new_idx)} changed=1"
                else:
                    st["mix_same"] = int(st.get("mix_same", 0)) + 1
                    if _log_decide:
                        _decide_diff_line = f"[DECIDE_DIFF] tag={tag} ep={ep_name} t={_t if _t is not None else 'NA'} player={_pl if _pl is not None else 'NA'} mcts_idx={int(_mcts_idx)} mix_idx={int(new_idx)} changed=0"

            try:
                if _log_decide:
                    if _decide_pre_line is not None:
                        setattr(pol, "_last_decide_pre_line", _decide_pre_line)
                    if _decide_post_line is not None:
                        setattr(pol, "_last_decide_post_line", _decide_post_line)
            except Exception:
                pass

            try:
                if _decide_diff_line is not None:
                    setattr(pol, "_last_decide_diff_line", _decide_diff_line)
            except Exception:
                pass

            _must = False
            try:
                _must = (_mcts_idx is not None and int(new_idx) != int(_mcts_idx))
            except Exception:
                _must = False

            if _LOG_DETAIL or _must:
                print(
                    f"[PhaseD-Q][MIX] tag={tag} ep={ep_name} use_q=True"
                    f" lam={_lam:.3f} tau={_tau:.3f}"
                    f" mcts_idx={_mcts_idx} mix_idx={int(new_idx)}"
                    f" mcts_top3={_mcts_top} q_top3={_q_top} mix_top3={_mix_top}",
                    flush=True,
                )

            _out_action = a_vec_new
            try:
                if ep_name in ("select_action_index_online", "select_action_index"):
                    _out_action = int(new_idx)
            except Exception:
                _out_action = a_vec_new

            if isinstance(ret, tuple) and len(ret) == 2:
                if isinstance(pi, dict):
                    _pi = dict(pi)
                    _pi["pi"] = mixed_pi
                    _pi["pi_base_mcts"] = base_pi
                    _pi["phaseD_q_values"] = [float(x) for x in q_vals]
                    _pi["phaseD_q_used"] = True
                    _pi["phaseD_mix"] = {
                        "lambda": float(PHASED_Q_MIX_LAMBDA),
                        "temperature": float(PHASED_Q_MIX_TEMPERATURE),
                    }
                    return _out_action, _pi
                return _out_action, mixed_pi

            return _out_action

        return wrapped


    for _ep in _callable_eps:
        setattr(pol, _ep, _wrap_one(_ep))

    pol._phased_q_wrapped = True
    pol.phased_q_wrapped = True
    pol._phased_q_tag = tag
    pol.phased_q_tag = tag

    print(f"[PhaseD-Q][WRAP] tag={tag} pol_id={id(pol)} class={type(pol).__name__} methods={','.join(_callable_eps)}")

def _wrap_select_action_for_mcts_counter(pol, tag):
    """
    USE_MCTS_POLICY=True かつ pol.use_mcts=True のとき、
    select_action 呼び出し回数をカウントしてコンソールに簡単なログを出すラッパ。
    （1プロセスあたり最初の数回だけを表示）
    """
    try:
        if not USE_MCTS_POLICY:
            return pol
    except NameError:
        # USE_MCTS_POLICY がこのモジュールに無い場合は何もしない
        return pol
    try:
        if getattr(pol, "_mcts_counter_wrapped", False):
            return pol

        _entrypoints = ("select_action", "act", "get_action", "choose_action")
        _callable_eps = [n for n in _entrypoints if callable(getattr(pol, n, None))]
        if not _callable_eps:
            return pol

        def _call_compat(fn, args, kwargs):
            try:
                import inspect as _inspect
            except Exception:
                return fn(*args, **kwargs)

            try:
                sig = _inspect.signature(fn)
            except Exception:
                return fn(*args, **kwargs)

            try:
                for p in sig.parameters.values():
                    if p.kind == p.VAR_KEYWORD:
                        return fn(*args, **kwargs)
            except Exception:
                return fn(*args, **kwargs)

            try:
                kw2 = {}
                for k, v in (kwargs or {}).items():
                    if k in sig.parameters:
                        kw2[k] = v
                return fn(*args, **kw2)
            except Exception:
                return fn(*args)

        def _wrap_one(ep_name):
            orig = getattr(pol, ep_name)

            def wrapped(*args, **kwargs):
                ret = _call_compat(orig, args, kwargs)

                try:
                    if getattr(pol, "use_mcts", False):
                        cnt = int(getattr(pol, "_mcts_call_count", 0)) + 1
                        setattr(pol, "_mcts_call_count", cnt)
                        if cnt <= 10:
                            sims = int(getattr(pol, "num_simulations", 0) or 0)
                            print(f"[MCTS] calls={cnt} sims={sims} ep={ep_name} tag={tag}")
                except Exception:
                    pass

                return ret

            return wrapped

        for _ep in _callable_eps:
            setattr(pol, _ep, _wrap_one(_ep))

        # _run_mcts が存在する場合はここもラップ（MCTS が「本当に動いたか」を確定できる）
        try:
            _run = getattr(pol, "_run_mcts", None)
        except Exception:
            _run = None

        if callable(_run) and not getattr(pol, "_run_mcts_wrapped", False):
            orig_run = _run

            def _wrapped_run(*args, **kwargs):
                try:
                    cnt = int(getattr(pol, "_run_mcts_call_count", 0)) + 1
                    setattr(pol, "_run_mcts_call_count", cnt)

                    sims = None
                    try:
                        sims = kwargs.get("num_simulations", None)
                    except Exception:
                        sims = None
                    if sims is None:
                        try:
                            if len(args) >= 3:
                                sims = args[2]
                        except Exception:
                            sims = None
                    if sims is None:
                        try:
                            sims = int(getattr(pol, "num_simulations", 0) or 0)
                        except Exception:
                            sims = None

                    if cnt <= 10:
                        print(f"[MCTS][_run_mcts] calls={cnt} sims={sims} tag={tag}", flush=True)
                except Exception:
                    pass

                return _call_compat(orig_run, args, kwargs)

            try:
                setattr(pol, "_orig__run_mcts", orig_run)
                setattr(pol, "_run_mcts_wrapped", True)
                setattr(pol, "_run_mcts_call_count", 0)
                setattr(pol, "_run_mcts_call_count_max", 10)
                setattr(pol, "_run_mcts_tag", tag)
                pol._run_mcts = _wrapped_run
            except Exception:
                pass

        pol._mcts_counter_wrapped = True
        return pol
    except Exception:
        return pol

def encode_action_from_vec_32d(five_ints):
    try:
        v = _encode_action_raw(five_ints)

        # _encode_action_raw の生出力（例: 17d）を Policy 期待次元（例: 32d）へ揃える
        if isinstance(v, np.ndarray):
            vv = v.reshape(-1).tolist()
        elif isinstance(v, (list, tuple)):
            vv = list(v)
        else:
            vv = [v]

        try:
            target_dim = int(globals().get("ACTION_VEC_DIM", 0) or 0)
        except Exception:
            target_dim = 0

        if target_dim > 0:
            if len(vv) < target_dim:
                vv = vv + [0] * (target_dim - len(vv))
            elif len(vv) > target_dim:
                vv = vv[:target_dim]

        return vv
    except NameError:
        raise RuntimeError(
            "Action encoder is not initialized yet. "
            "Call build_encoder_from_files(...) before using encode_action_from_vec_32d."
        )

def _attach_action_encoder_if_supported(pol):
    """
    モデル方策や GPU クライアント側にアクション埋め込み関数を渡せるフックが
    用意されている場合に差し込む（後方互換のためのベストエフォート）。
    """
    try:
        enc32 = globals().get("encode_action_from_vec_32d", None)
        enc19 = globals().get("encode_action_from_vec_19d", None)
        enc = enc32 if callable(enc32) else (enc19 if callable(enc19) else None)
        if enc is None:
            return

        if hasattr(pol, "set_action_encoder") and callable(getattr(pol, "set_action_encoder")):
            pol.set_action_encoder(enc)
        elif hasattr(pol, "action_encoder_fn"):
            pol.action_encoder_fn = enc
    except Exception:
        pass

def _dump_policy_entrypoint_where(pol, label="pol"):
    try:
        import inspect
    except Exception:
        inspect = None

    print(f"[WHERE] {label} class={type(pol).__name__} id={id(pol)}")

    fn = getattr(pol, "select_action_index_online", None)
    print(f"[WHERE] {label}.select_action_index_online callable={callable(fn)}")
    if callable(fn):
        try:
            print(f"[WHERE] {label}.select_action_index_online file={fn.__code__.co_filename}")
            print(f"[WHERE] {label}.select_action_index_online line={fn.__code__.co_firstlineno}")
        except Exception as e:
            print(f"[WHERE] {label}.select_action_index_online codeinfo error={e!r}")

    if inspect is not None:
        try:
            print(f"[WHERE] {label} class file={inspect.getsourcefile(type(pol))}")
        except Exception as e:
            print(f"[WHERE] {label} classfile error={e!r}")

def _wrap_az_select_action(pol, tag="az"):
    """
    AlphaZeroMCTSPolicy の entrypoint を薄くラップして、
    「呼ばれたか」「env/cand_vecs が渡っているか」「例外で落ちていないか」をログに出す。
    *args/**kwargs をそのまま通すので env= 等のキーワード引数を壊さない。
    """
    try:
        if pol is None:
            return pol
    except Exception:
        return pol

    try:
        if getattr(pol, "_az_select_action_wrapped", False):
            try:
                setattr(pol, "_az_select_action_wrap_tag", tag)
            except Exception:
                pass
            return pol
    except Exception:
        pass

    _entrypoints = ("select_action", "select_action_index_online", "select_action_index")
    _callable_eps = [n for n in _entrypoints if callable(getattr(pol, n, None))]
    if not _callable_eps:
        return pol

    def _wrap_one(ep_name):
        orig = getattr(pol, ep_name)

        try:
            import inspect
            _sig = inspect.signature(orig) if callable(orig) else None
        except Exception:
            _sig = None

        def wrapped(*args, **kwargs):
            try:
                use_mcts = bool(getattr(pol, "use_mcts", False))
            except Exception:
                use_mcts = False
            try:
                sims = int(getattr(pol, "num_simulations", 0) or 0)
            except Exception:
                sims = 0

            # env / cand_vecs を kwargs or positional から復元して表示
            _env = None
            _cand = None

            try:
                _env = kwargs.get("env", None)
            except Exception:
                _env = None
            try:
                _cand = kwargs.get("cand_vecs", None)
            except Exception:
                _cand = None

            if (_env is None or _cand is None) and (_sig is not None):
                try:
                    _b = _sig.bind_partial(*args, **kwargs)
                    if _env is None:
                        _env = _b.arguments.get("env", None)
                    if _cand is None:
                        _cand = _b.arguments.get("cand_vecs", None)
                except Exception:
                    pass

            if _cand is None:
                try:
                    if len(args) >= 2:
                        _cand = args[1]
                except Exception:
                    pass

            try:
                env_name = type(_env).__name__ if _env is not None else None
            except Exception:
                env_name = None
            try:
                cand_len = len(_cand) if isinstance(_cand, (list, tuple)) else None
            except Exception:
                cand_len = None

            print(
                f"[AZ][WRAP][CALL] tag={tag} ep={ep_name} use_mcts={int(use_mcts)} sims={int(sims)} env={env_name} cand_len={cand_len}",
                flush=True,
            )

            try:
                ret = orig(*args, **kwargs)
            except TypeError as e:
                print(f"[AZ][WRAP][TYPEERROR] tag={tag} ep={ep_name} err={e!r} kwargs_keys={list(kwargs.keys())[:20]}", flush=True)
                raise
            except Exception as e:
                print(f"[AZ][WRAP][EXC] tag={tag} ep={ep_name} err={e!r}", flush=True)
                raise

            print(
                f"[AZ][WRAP][RET] tag={tag} ep={ep_name} ret_type={type(ret).__name__}",
                flush=True,
            )
            return ret

        return wrapped

    for _ep in _callable_eps:
        try:
            setattr(pol, _ep, _wrap_one(_ep))
        except Exception:
            pass

    try:
        pol._az_select_action_wrapped = True
        pol._az_select_action_wrap_tag = tag
    except Exception:
        pass

    print(f"[AZ][WRAP][OK] tag={tag} pol_id={id(pol)} class={type(pol).__name__} methods={','.join(_callable_eps)}", flush=True)
    return pol


def build_policy(which: str, model_dir: str):
    """
    ポリシー種別とフラグに応じて方策を生成する。
      - SELFPLAY_ALPHAZERO_MODE かつ USE_MCTS_POLICY=1 のとき:
          → AlphaZeroMCTSPolicy（MCTSあり）
      - SELFPLAY_ALPHAZERO_MODE かつ USE_MCTS_POLICY=0 のとき:
          → which="model_only" などで「モデルのみモード」を選択（MCTSなし）
      - それ以外:
          - which が "az_mcts" / "az-mcts" / "mcts" / "model_only" / "az_model" → AlphaZeroMCTSPolicy
          - which が "online_mix"                                 → OnlineMixedPolicy（モデル＋MCTS＋PhaseD-Q のオンライン混合）
          - which が "random" / 空 / None                        → RandomPolicy
          - その他                                             → 警告して RandomPolicy
    """
    name = (str(which).strip().lower() if which is not None else "random")

    # AlphaZeroMCTSPolicy の詳細ログ（[AZ][DECISION] など）を確実に出す
    # （未指定なら有効化。明示的に 0/1 を設定している場合は尊重）
    try:
        if os.getenv("AZ_DECISION_LOG", "") == "" and name in ("az_model", "az_mcts", "online_mix", "model_only"):
            os.environ["AZ_DECISION_LOG"] = "1"
    except Exception:
        pass

    def _ensure_az_select_action_env_compatible(_pol, _tag=""):
        try:
            import inspect
            _fn = getattr(_pol, "select_action", None)
            _sig = inspect.signature(_fn) if callable(_fn) else None
            _has_env = False
            _has_varkw = False
            if _sig is not None:
                _has_env = ("env" in _sig.parameters)
                _has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values())
            if not (_has_env or _has_varkw):
                _pol.select_action = _pol.__class__.select_action.__get__(_pol, _pol.__class__)
                if os.getenv("AZ_DECISION_LOG", "0") == "1":
                    print(f"[AZ][WRAP_FIX]{_tag} rebound select_action to class method (env kwarg compatible)", flush=True)
        except Exception:
            pass

    is_model_only = (name in ("model_only", "az_model", "az-model", "azmodel"))
    is_az_name = name in ("az_mcts", "az-mcts", "mcts", "model_only", "az_model", "az-model", "azmodel")

    # 1) フラグ優先: AlphaZero 自己対戦モードで USE_MCTS_POLICY=1 のときは、
    #    az_mcts 系指定のときだけ MCTS にする（random 側まで巻き込まない）
    if SELFPLAY_ALPHAZERO_MODE and USE_MCTS_POLICY and name in ("az_mcts", "az-mcts", "mcts"):
        try:
            from pokepocketsim.policy.az_mcts_policy import AlphaZeroMCTSPolicy

            pol = AlphaZeroMCTSPolicy(model_dir=model_dir)
            # MCTS 有効モード: policy 側のフラグを ON（シミュレーション回数は先頭設定から取得）
            setattr(pol, "use_mcts", True)
            try:
                sims = int(os.getenv("AZ_MCTS_NUM_SIMULATIONS", str(AZ_MCTS_NUM_SIMULATIONS)))
            except Exception:
                sims = 64
            setattr(pol, "num_simulations", sims)
            return pol
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] AlphaZeroMCTSPolicy load failed in SELFPLAY_ALPHAZERO_MODE "
                f"(policy='{which}', model_dir='{model_dir}'): {e}"
            )

    # 2) AlphaZero モデルのみモード / 通常時の az_mcts 系
    if is_az_name:
        try:
            from pokepocketsim.policy.az_mcts_policy import AlphaZeroMCTSPolicy
            pol = AlphaZeroMCTSPolicy(model_dir=model_dir)
            if is_model_only:
                # ★モデルのみモード（MCTSを無効化するためのフラグをポリシー側に渡す）
                setattr(pol, "disable_mcts", True)
                # MCTS を明示的に無効化
                setattr(pol, "use_mcts", False)
                setattr(pol, "num_simulations", 0)
                print("[POLICY] using AlphaZeroMCTSPolicy in model_only mode (MCTS disabled if supported)")
            else:
                # ★az_mcts/mcts 指定なら Selfplayフラグ無しでも MCTS を有効化する
                setattr(pol, "use_mcts", True)
                try:
                    sims = int(AZ_MCTS_NUM_SIMULATIONS)
                except Exception:
                    sims = 64
                setattr(pol, "num_simulations", sims)
                print(f"[POLICY] using AlphaZeroMCTSPolicy in MCTS mode (sims={sims})")
            return pol
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] AlphaZeroMCTSPolicy load failed for policy='{which}' model_dir='{model_dir}': {e}"
            )

    # 3) OnlineMixedPolicy（モデル＋MCTS＋PhaseD-Q のオンライン混合）
    if name == "online_mix":
        try:
            from pokepocketsim.policy.online_mixed_policy import OnlineMixedPolicy
            from pokepocketsim.policy.az_mcts_policy import AlphaZeroMCTSPolicy

            # メイン: MCTS 付き AlphaZero 方策
            main_pol = AlphaZeroMCTSPolicy(model_dir=model_dir)
            setattr(main_pol, "use_mcts", True)
            try:
                sims = int(AZ_MCTS_NUM_SIMULATIONS)
            except Exception:
                sims = 64
            setattr(main_pol, "num_simulations", sims)

            # --- 重要: select_action が env= を受け取れないラッパに差し替わっている場合に備えて差し戻す ---
            try:
                import inspect
                _fn = getattr(main_pol, "select_action", None)
                _sig = inspect.signature(_fn) if callable(_fn) else None
                _has_env = False
                _has_varkw = False
                if _sig is not None:
                    _has_env = ("env" in _sig.parameters)
                    _has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values())
                if not (_has_env or _has_varkw):
                    main_pol.select_action = main_pol.__class__.select_action.__get__(main_pol, main_pol.__class__)
                    if os.getenv("AZ_DECISION_LOG", "0") == "1":
                        print("[AZ][WRAP_FIX] rebound select_action to class method (env kwarg compatible)", flush=True)
            except Exception:
                pass

            _dump_policy_entrypoint_where(main_pol, label="online_mix.main(before_wrap)")

            # AZ: main 側の entrypoint を観測したい場合のみラップ（デフォルトは無効）
            try:
                if os.getenv("AZ_WRAP_AZ_LOG", "0") == "1":
                    main_pol = _wrap_az_select_action(main_pol, tag="online_mix.main")
            except Exception as _e:
                print(f"[AZ][WRAP] online_mix.main failed: {_e!r}")

            # PhaseD-Q: main 側にはラップしない（outer 側に統一して重複ログを防ぐ）

            _dump_policy_entrypoint_where(main_pol, label="online_mix.main(after_wrap)")

            # フォールバック: RandomPolicy
            fallback_pol = RandomPolicy()

            pol = OnlineMixedPolicy(
                main_policy=main_pol,
                fallback_policy=fallback_pol,
                mix_prob=1.0,
                model_dir=model_dir,
            )

            _dump_policy_entrypoint_where(pol, label="online_mix.outer(before_wrap)")

            # PhaseD-Q: OnlineMixedPolicy 側の entrypoint を捕まえる（実際の呼び出し元がこちらの可能性が高い）
            try:
                _wrap_select_action_with_phased_q(pol, tag="online_mix.outer")
            except Exception as _e:
                print(f"[PhaseD-Q][WRAP] online_mix.outer failed: {_e!r}")

            _dump_policy_entrypoint_where(pol, label="online_mix.outer(after_wrap)")

            # 外側にもフラグを伝播（MATCH_POLICY ログで見えるようにする）
            try:
                wrapped = bool(
                    getattr(pol, "phased_q_wrapped", False) or getattr(pol, "_phased_q_wrapped", False)
                    or getattr(main_pol, "phased_q_wrapped", False) or getattr(main_pol, "_phased_q_wrapped", False)
                )
                tag = (
                    getattr(pol, "phased_q_tag", None) or getattr(pol, "_phased_q_tag", None)
                    or getattr(main_pol, "phased_q_tag", None) or getattr(main_pol, "_phased_q_tag", None)
                )
                setattr(pol, "_phased_q_wrapped", wrapped)
                setattr(pol, "phased_q_wrapped", wrapped)
                setattr(pol, "_phased_q_tag", tag)
                setattr(pol, "phased_q_tag", tag)
            except Exception as _e:
                print(f"[PhaseD-Q][PROPAGATE] to OnlineMixedPolicy skipped: {_e!r}")

            if globals().get("LOG_DEBUG_DETAIL", False):
                print(
                    "[POLICY][online_mix] "
                    f"PHASED_Q_MIX_ENABLED={bool(globals().get('PHASED_Q_MIX_ENABLED', False))} "
                    f"USE_PHASED_Q={bool(globals().get('USE_PHASED_Q', False))} "
                    f"wrapped={bool(getattr(pol, 'phased_q_wrapped', False))} "
                    f"tag={getattr(pol, 'phased_q_tag', None)}"
                )

            print(
                "[POLICY] using OnlineMixedPolicy (online_mix: main=MCTS+model + PhaseD-Q, fallback=random) "
                f"(sims={sims}, phased_q_wrapped={getattr(pol, 'phased_q_wrapped', False)}, phased_q_tag={getattr(pol, 'phased_q_tag', None)})"
            )
            return pol
        except Exception as e:
            print(f"[ERROR] OnlineMixedPolicy load failed ({e}) (P*_POLICY=online_mix, model_dir={model_dir})")
            raise

    if name in ("random", "",):
        return RandomPolicy()

    if globals().get("LOG_DEBUG_DETAIL", False):
        print(f"[POLICY] unknown policy '{which}' → fallback to RandomPolicy()")
    return RandomPolicy()

# 共有エンコーダ関数を生成（唯一の正解）
_encode_action_raw, _CARD_ID2IDX, _ACTION_TYPES, (K, V, ACTION_VEC_DIM_RAW) = build_encoder_from_files(
    _card_idx_path, _action_types_path, ACTION_SCHEMAS, TYPE_SCHEMAS, MAX_ARGS
)

# ログ用の候補ベクトル次元は 32 に固定（実エンコーダ出力は 32 次元にパディング／切り詰め）
ACTION_VEC_DIM = 32

if LOG_DEBUG_DETAIL:
    print("\n[SYNC] ===== Policy boot spec =====")
    print(f"[SYNC] vocab_path   : {_card_idx_path}")
    print(f"[SYNC] action_types : {_action_types_path} (K={len(_ACTION_TYPES)})")
    print(f"[SYNC] ACTION_VEC_DIM_RAW= {ACTION_VEC_DIM_RAW}")
    print(f"[SYNC] ACTION_VEC_DIM_LOG     = {ACTION_VEC_DIM}")
    print("[SYNC] OBS_BENCH_MAX= 8")

# 追加: 学習時 32 想定ならズレを即警告（これは重要なので DEBUG とは別扱い）
EXPECTED_DIM = int(os.getenv("EXPECTED_ACTION_VEC_DIM", "32"))
if ACTION_VEC_DIM != EXPECTED_DIM:
    print(
        f"[WARN] ACTION_VEC_DIM(log)={ACTION_VEC_DIM} (expected {EXPECTED_DIM}). "
        "→ 学習時の action_types.json / TYPE_SCHEMAS / MAX_ARGS と一致しているか確認してください。"
    )

# ★ 追加: 行動スキーマ長が 0 なら実行中止（Evals=0 の温床を即発見）
ASSERT_ACTIONS_POSITIVE = os.getenv("ASSERT_ACTIONS_POSITIVE", "1") == "1"
if ASSERT_ACTIONS_POSITIVE and len(ACTION_SCHEMAS) == 0 and len(TYPE_SCHEMAS) == 0:
    raise RuntimeError("[FATAL] ACTION_SCHEMAS と TYPE_SCHEMAS の両方が空です。評価要求を生成できません。")
elif len(ACTION_SCHEMAS) == 0 and len(TYPE_SCHEMAS) > 0:
    if LOG_DEBUG_DETAIL:
        print("[SYNC] ACTION_SCHEMAS は空ですが TYPE_SCHEMAS があるため継続します（K+MAX_ARGS+1 仕様）。")

_active_model_dir = (
    MODEL_DIR_P1 if str(P1_POLICY).lower() == "model"
    else (MODEL_DIR_P2 if str(P2_POLICY).lower() == "model" else None)
)


# ★追加: obs_vec を必ず出すための強制エンコーダ装着
def _find_encoder_dir_candidates():
    """scaler.npz を持つ“もっとも妥当な”ディレクトリ候補を順に返す"""
    cands = []
    if _active_model_dir:
        cands.append(_active_model_dir)                # 1) アクティブ（あれば）
    cands.extend([MODEL_DIR_P1, MODEL_DIR_P2])         # 2) 明示のモデルディレクトリ
    try:
        vocab_dir = os.path.dirname(_card_idx_path)    # 3) 語彙ファイルのある場所
        if vocab_dir:
            cands.append(vocab_dir)
    except Exception:
        pass
    # 重複除去
    seen, out = set(), []
    for d in cands:
        if d and d not in seen:
            out.append(d); seen.add(d)
    return out

def _ensure_match_encoder(match, pol1, pol2):
    """
    可能なら方策の encoder を流用。無ければ候補ディレクトリから scaler.npz を探して StateEncoder を作る。
    （スケーラが無い場合は obs_vec を安全に生成できないので明示警告）
    """
    def _infer_expected_obs_dim():
        for _k in ("PHASED_Q_OBS_DIM", "PHASED_Q_EXPECTED_OBS_DIM", "PHASED_Q_MODEL_OBS_DIM"):
            try:
                _v = globals().get(_k, None)
            except Exception:
                _v = None
            try:
                _vi = int(_v) if _v is not None else 0
            except Exception:
                _vi = 0
            if _vi > 0:
                return _vi
        try:
            _env = os.getenv("PHASED_Q_EXPECTED_OBS_DIM")
            _vi = int(_env) if _env else 0
            if _vi > 0:
                return _vi
        except Exception:
            pass
        return 2448

    _EXP_OBS_DIM = int(_infer_expected_obs_dim())

    def _to_len(x):
        try:
            if x is None:
                return -1
            if hasattr(x, "reshape"):
                try:
                    x = x.reshape(-1)
                except Exception:
                    pass
            if hasattr(x, "shape") and getattr(x, "shape", None) is not None:
                try:
                    return int(x.shape[0])
                except Exception:
                    pass
            if isinstance(x, (list, tuple)):
                return len(x)
        except Exception:
            pass
        return -1

    def _probe_encoder_dim(_enc):
        try:
            ps = getattr(match, "public_state", None)
            if ps is None or not isinstance(ps, dict):
                ps = {}
            try:
                out = _enc(ps, None)
                return _to_len(out)
            except TypeError:
                pass
            try:
                out = _enc(public_state=ps, legal_actions=None)
                return _to_len(out)
            except TypeError:
                pass
            try:
                out = _enc(player=ps, legal_actions=None)
                return _to_len(out)
            except TypeError:
                pass
        except Exception:
            pass
        return -1

    enc = getattr(pol1, "encoder", None) or getattr(pol2, "encoder", None)

    # ★ 重要: 既存 encoder が期待次元を返せない場合は捨てる（135次元をここで排除）
    try:
        _enc_dim = _probe_encoder_dim(enc) if enc is not None else -1
        if _enc_dim > 0 and _enc_dim != _EXP_OBS_DIM:
            print(f"[ENCODER] existing policy encoder dim mismatch: got={_enc_dim} exp={_EXP_OBS_DIM} -> rebuild")
            enc = None
    except Exception:
        pass

    if enc is None:
        spicked = None
        for d in _find_encoder_dir_candidates():
            sp = os.path.join(d, "scaler.npz")
            if os.path.exists(sp):
                spicked = sp
                break

        try:
            if spicked:
                # 期待次元は scaler.npz（mean/std）から推定できるなら最優先。まず盤面設計ベースの build_obs_partial_vec を試し、ダメなら build_obs_full_vec にフォールバック。
                class _FullObsVecEncoder:
                    def __init__(self, scaler_path=None, expected_dim=2448):
                        self.scaler_path = scaler_path
                        self.expected_dim = int(expected_dim) if expected_dim else 2448
                        self.mean = None
                        self.std = None
                        if scaler_path:
                            try:
                                import numpy as _np
                                _d = _np.load(scaler_path)
                                if "mean" in _d and "std" in _d:
                                    self.mean = _d["mean"]
                                    self.std = _d["std"]
                                    try:
                                        # ★ 重要: object配列で「要素1個（中身が2448配列）」のケースを剥がす
                                        try:
                                            if isinstance(self.mean, _np.ndarray) and getattr(self.mean, "dtype", None) == object and int(getattr(self.mean, "size", 0)) == 1:
                                                self.mean = self.mean.item()
                                        except Exception:
                                            pass
                                        try:
                                            if isinstance(self.std, _np.ndarray) and getattr(self.std, "dtype", None) == object and int(getattr(self.std, "size", 0)) == 1:
                                                self.std = self.std.item()
                                        except Exception:
                                            pass

                                        # ★ 重要: 1D 化して expected_dim を確定（ただし 1 なら上書きしない）
                                        self.mean = _np.asarray(self.mean, dtype=_np.float32).reshape(-1)
                                        self.std = _np.asarray(self.std, dtype=_np.float32).reshape(-1)

                                        _dim = int(getattr(self.mean, "shape", [0])[0])
                                        if _dim > 1:
                                            self.expected_dim = _dim
                                    except Exception:
                                        pass

                            except Exception:
                                self.mean = None
                                self.std = None

                    def encode_state(self, feat):
                        try:
                            import numpy as _np

                            sbp = feat if isinstance(feat, dict) else {}

                            # feat が {"me","opp","legal_actions"} だけでも build_obs_* が動くよう補完
                            if "current_player_name" not in sbp:
                                _me = sbp.get("me", {}) if isinstance(sbp, dict) else {}
                                _nm = None
                                if isinstance(_me, dict):
                                    _nm = _me.get("name", None)
                                if _nm is None:
                                    _nm = "p1"
                                sbp = dict(sbp)
                                sbp["current_player_name"] = _nm

                            if "turn" not in sbp:
                                sbp = dict(sbp)
                                sbp["turn"] = 0

                            vec = None

                            # まず「盤面→特徴量（設計ベース）」を優先
                            try:
                                if "build_obs_partial_vec" in globals() and callable(globals().get("build_obs_partial_vec", None)):
                                    vec = build_obs_partial_vec(sbp)
                            except Exception:
                                vec = None

                            # ダメなら従来の full_vec へフォールバック
                            if not isinstance(vec, (list, tuple)) or not vec:
                                vec = build_obs_full_vec(sbp)

                            # ★ 重要: vec が「外側1要素」で包まれて返る経路がある（[vec] / [[vec]] / object配列 / dict等）ので多段で剥がす
                            try:
                                _v = vec
                                for _ in range(8):
                                    if isinstance(_v, dict):
                                        for _k in ("obs_vec", "obs", "vec", "x", "full_obs_vec"):
                                            if _k in _v:
                                                _v = _v[_k]
                                                break
                                        else:
                                            break
                                        continue

                                    if isinstance(_v, (list, tuple)) and len(_v) == 1:
                                        _v = _v[0]
                                        continue

                                    if isinstance(_v, _np.ndarray) and getattr(_v, "dtype", None) == object and getattr(_v, "ndim", 0) == 1 and int(_v.shape[0]) == 1:
                                        _v = _v[0]
                                        continue

                                    break
                                vec = _v
                            except Exception:
                                pass

                            x = _np.asarray(vec, dtype=_np.float32)

                            # ★ 重要: (1, expected_dim) のような“バッチ次元付き”を潰して 1 次元に正規化する
                            try:
                                if getattr(x, "ndim", 1) == 2 and int(getattr(x, "shape", (0, 0))[0]) == 1:
                                    x = x[0]

                                if getattr(x, "ndim", 1) != 1:
                                    x = x.reshape(-1)

                                if getattr(x, "shape", None) != (self.expected_dim,):
                                    _y = _np.zeros((self.expected_dim,), dtype=_np.float32)
                                    _n = int(min(int(x.shape[0]), int(self.expected_dim)))
                                    if _n > 0:
                                        _y[:_n] = x[:_n]
                                    x = _y
                            except Exception:
                                try:
                                    x = _np.zeros((self.expected_dim,), dtype=_np.float32)
                                except Exception:
                                    return [0.0] * int(self.expected_dim)

                            # scaler が expected_dim と一致する場合のみ適用（不一致なら“安全に”未適用）
                            if self.mean is not None and self.std is not None:
                                if getattr(self.mean, "shape", None) == x.shape and getattr(self.std, "shape", None) == x.shape:
                                    _std = _np.where(self.std == 0, 1.0, self.std)
                                    x = (x - self.mean) / _std

                            return x
                        except Exception:
                            try:
                                import numpy as _np
                                return _np.zeros((self.expected_dim,), dtype=_np.float32)
                            except Exception:
                                return [0.0] * int(self.expected_dim)

                    def __call__(self, public_state=None, legal_actions=None, *a, **k):
                        # ★ PhaseD-Q 側の呼び出しゆらぎに対応（kwargs から public_state を拾う）
                        if public_state is None:
                            for _kk in ("public_state", "player", "state", "feat"):
                                try:
                                    if _kk in k:
                                        public_state = k.get(_kk, None)
                                        break
                                except Exception:
                                    pass
                        if public_state is None and a:
                            try:
                                public_state = a[0]
                            except Exception:
                                pass
                        if legal_actions is None:
                            try:
                                if "legal_actions" in k:
                                    legal_actions = k.get("legal_actions", None)
                            except Exception:
                                pass

                        out = self.encode_state(public_state)
                        try:
                            import numpy as _np

                            if hasattr(out, "reshape"):
                                vec = out.reshape(-1).tolist()
                            elif hasattr(out, "tolist"):
                                vec = out.tolist()
                            else:
                                vec = list(out)

                            # ★ 重要: vec を必ず「expected_dim の 1次元 list[float]」に正規化してキャッシュ
                            _v = _np.asarray(vec, dtype=_np.float32).reshape(-1)
                            if int(getattr(_v, "shape", (0,))[0]) != int(self.expected_dim):
                                _y = _np.zeros((int(self.expected_dim),), dtype=_np.float32)
                                _n = int(min(int(_v.shape[0]), int(self.expected_dim)))
                                if _n > 0:
                                    _y[:_n] = _v[:_n]
                                _v = _y
                            vec = _v.tolist()
                            self._last_full_obs_vec = vec

                            # ★ 重要: state(dict) 側にも載せて PhaseD-Q の ensure_obs_vec が拾えるようにする
                            try:
                                if isinstance(public_state, dict):
                                    public_state["obs_vec"] = vec
                                    public_state["full_obs_vec"] = vec
                            except Exception:
                                pass
                        except Exception:
                            pass
                        return out

                    def encode_full_obs_vec(self, public_state=None, legal_actions=None, *a, **k):
                        return self.__call__(public_state, legal_actions, *a, **k)

                    def encode_obs_vec_full(self, public_state=None, legal_actions=None, *a, **k):
                        return self.__call__(public_state, legal_actions, *a, **k)

                    def get_full_obs_vec(self, public_state=None, legal_actions=None, *a, **k):
                        return self.__call__(public_state, legal_actions, *a, **k)

                    def make_full_obs_vec(self, public_state=None, legal_actions=None, *a, **k):
                        return self.__call__(public_state, legal_actions, *a, **k)

                    def encode_full_obs(self, public_state=None, legal_actions=None, *a, **k):
                        return self.__call__(public_state, legal_actions, *a, **k)

                    def get_full_obs(self, public_state=None, legal_actions=None, *a, **k):
                        return self.__call__(public_state, legal_actions, *a, **k)

                    def make_full_obs(self, public_state=None, legal_actions=None, *a, **k):
                        return self.__call__(public_state, legal_actions, *a, **k)

                    @property
                    def full_obs_vec(self):
                        try:
                            return getattr(self, "_last_full_obs_vec", None)
                        except Exception:
                            return None

                try:
                    enc = _FullObsVecEncoder(scaler_path=spicked, expected_dim=_EXP_OBS_DIM)
                    print(f"[ENCODER] using scaler: {spicked} (full_obs_vec) exp_dim={_EXP_OBS_DIM}")
                except Exception:
                    enc = StateEncoder(scaler_path=spicked)
                    print(f"[ENCODER] using scaler: {spicked} (legacy_state_encoder)")
            else:
                print("[ENCODER] ❌ scaler.npz が見つからないため obs_vec を安全に生成できません。")
                enc = None
        except Exception as e:
            print(f"[ENCODER] encoder init failed: {e}")
            enc = None


    if enc is not None:
        match.encoder = enc

        # ★ 重要: 既存 encoder が 135 などで載っていたケースでも上書きして統一する
        try:
            try: pol1.encoder = enc
            except Exception: pass
            try: pol2.encoder = enc
            except Exception: pass
        except Exception:
            pass

        print("[ENCODER] match.encoder をセットしました（obs_vec を出力）")

        # ★ PhaseD-Q 用: encoder が生成した「フル次元 obs_vec」をキャッシュ
        try:
            import types as _types
            if not getattr(enc, "_phased_q_cache_wrapped", False):
                enc._phased_q_cache_wrapped = True
                _orig_call = enc.__call__

                def _call(self, public_state=None, legal_actions=None, *a, **k):
                    out = _orig_call(public_state, legal_actions, *a, **k)
                    try:
                        if hasattr(out, "reshape"):
                            vec = out.reshape(-1).tolist()
                        elif hasattr(out, "tolist"):
                            vec = out.tolist()
                        else:
                            vec = list(out)

                        if vec and isinstance(vec[0], list):
                            vec = vec[0]

                        globals()["_PHASED_Q_LAST_FULL_OBS_VEC"] = vec
                    except Exception:
                        pass
                    return out

                enc.__call__ = _types.MethodType(_call, enc)
        except Exception:
            pass

        try:
            def _ensure_public_players(_m):
                try:
                    class _AttrDict(dict):
                        def __getattr__(self, name):
                            try:
                                return self[name]
                            except KeyError:
                                raise AttributeError(name)
                        def __setattr__(self, name, value):
                            self[name] = value

                    ps = getattr(_m, "public_state", None)
                    if ps is None or not isinstance(ps, dict):
                        ps = _AttrDict()
                        _m.public_state = ps
                    elif not isinstance(ps, _AttrDict):
                        ps = _AttrDict(ps)
                        _m.public_state = ps

                    # すでに players が「有効に」入っていれば何もしない（空/不正形は未設定扱いで埋める）
                    try:
                        _pv = ps.get("players")
                        if isinstance(_pv, (list, tuple)) and len(_pv) > 0:
                            ps["turn"] = int(getattr(_m, "turn", 0))
                            return
                    except Exception:
                        pass

                    def _to_public_player_obj(x):
                        if x is None:
                            return None
                        if isinstance(x, dict):
                            return x
                        d = {}
                        try:
                            n = getattr(x, "name", None)
                            if n is not None:
                                d["name"] = n
                        except Exception:
                            pass
                        for k in ("player_id", "pid", "side", "index"):
                            try:
                                v = getattr(x, k, None)
                                if v is not None:
                                    d[k] = v
                            except Exception:
                                pass
                        return d if d else {"name": str(x)}

                    # 取り得る候補を広めに拾う（dict / list / その他を許容）
                    cand = None
                    for attr in ("players", "players_by_name", "player_by_name", "player_map", "_players", "_player_by_name"):
                        v = getattr(_m, attr, None)
                        if v is not None:
                            cand = v
                            break

                    if cand is None:
                        _p1 = getattr(_m, "starting_player", None) or getattr(_m, "p1", None) or getattr(_m, "player1", None) or getattr(_m, "player_1", None)
                        _p2 = getattr(_m, "second_player", None) or getattr(_m, "p2", None) or getattr(_m, "player2", None) or getattr(_m, "player_2", None)
                        if _p1 is not None or _p2 is not None:
                            cand = {"p1": _p1, "p2": _p2}

                    players_list = None

                    if isinstance(cand, dict):
                        if ("p1" in cand) or ("p2" in cand):
                            p1 = _to_public_player_obj(cand.get("p1"))
                            p2 = _to_public_player_obj(cand.get("p2"))
                            players_list = [x for x in (p1, p2) if x is not None]
                        else:
                            players_list = [_to_public_player_obj(v) for v in list(cand.values())]
                            players_list = [x for x in players_list if x is not None]
                    elif isinstance(cand, (list, tuple)):
                        players_list = [_to_public_player_obj(v) for v in cand]
                        players_list = [x for x in players_list if x is not None]
                    elif cand is not None:
                        players_list = [_to_public_player_obj(cand)]
                        players_list = [x for x in players_list if x is not None]

                    if not players_list:
                        players_list = [{"name": "p1"}, {"name": "p2"}]

                    ps["players"] = list(players_list)
                    ps["turn"] = int(getattr(_m, "turn", 0))

                except Exception:
                    try:
                        ps = getattr(_m, "public_state", None)
                        if ps is None or not isinstance(ps, dict):
                            ps = {}
                            _m.public_state = ps
                        ps["players"] = [{"name": "p1"}, {"name": "p2"}]
                        ps["turn"] = int(getattr(_m, "turn", 0))
                    except Exception:
                        pass

            def _dbg_public_players(match, tag=""):
                ps = getattr(match, "public_state", None)
                players = None
                if isinstance(ps, dict):
                    players = ps.get("players", None)
                t = type(players).__name__
                ln = (len(players) if isinstance(players, (list, tuple)) else -1)
                head = None
                if isinstance(players, (list, tuple)) and len(players) > 0:
                    head = players[0]
                print(f"[OBSDBG]{tag} public_state.players type={t} len={ln} head_type={type(head).__name__ if head is not None else 'None'} ps_keys={(list(ps.keys()) if isinstance(ps, dict) else 'N/A')}")

            _ensure_public_players(match)
            _dbg_public_players(match, tag=" after_ensure")

        except Exception:
            pass

            _ensure_public_players(match)

            # match の初期化が後で進む場合に備えて、存在するメソッドだけ後追いフック
            for _mname in (
                "setup_battle_and_bench",
                "setup_battle_and_bench_for_player",
                "setup_players",
                "setup_game",
                "start_game",
                "start",
                "run",
            ):
                try:
                    _orig = getattr(match, _mname, None)
                    if _orig is None or not callable(_orig):
                        continue
                    if getattr(_orig, "__ptm_wrapped__", False):
                        continue

                    def _wrap(fn):
                        def _inner(*a, **kw):
                            r = fn(*a, **kw)
                            _ensure_public_players(match)
                            return r
                        _inner.__ptm_wrapped__ = True
                        return _inner

                    setattr(match, _mname, _wrap(_orig))
                except Exception:
                    pass

            import policy_trace_monkeypatch as ptm

            # 関数名ゆらぎに対応して “現在の match/encoder” をできるだけ登録
            try:
                if hasattr(ptm, "register_match"):
                    ptm.register_match(match)
            except Exception:
                pass
            try:
                if hasattr(ptm, "set_current_match"):
                    ptm.set_current_match(match)
            except Exception:
                pass
            try:
                if hasattr(ptm, "register_match_encoder"):
                    ptm.register_match_encoder(match, getattr(match, "encoder", None))
            except Exception:
                pass
            try:
                if hasattr(ptm, "set_current_encoder"):
                    ptm.set_current_encoder(getattr(match, "encoder", None))
            except Exception:
                pass
        except Exception:
            pass
    else:
        print("[ENCODER] ⚠️ encoder を用意できません（obs_vec は空[]になります）")


deck1_recipe = ALL_DECK_RECIPES[DECK1_KEY]
deck2_recipe = ALL_DECK_RECIPES[DECK2_KEY]
deck1_type = DECK_TYPE_NAMES[DECK1_KEY]
deck2_type = DECK_TYPE_NAMES[DECK2_KEY]


def _next_rotated_path(base_path: str, idx: int) -> str:
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:05d}{ext}"

def writer_loop(queue: "Queue", stop_event: "Event"):
    """
    集中ライター。全ワーカーから受け取ったJSONL行を
    RAW_JSONL_PATH / IDS_JSONL_PATH / PRIVATE_IDS_JSON_PATH の「同一ファイル」に安全に追記する。
    """
    import queue as _queue_mod

    raw_idx = 0
    ids_idx = 0
    priv_idx = 0
    raw_path = RAW_JSONL_PATH
    ids_path = IDS_JSONL_PATH
    priv_path = PRIVATE_IDS_JSON_PATH

    # どこに出力しているか見失わないように絶対パスも表示
    print(f"[OUT] raw={raw_path} (abs={os.path.abspath(raw_path)})")
    print(f"[OUT] ids={ids_path} (abs={os.path.abspath(ids_path)})")
    print(f"[OUT] private_ids={priv_path} (abs={os.path.abspath(priv_path)})")

    # ディレクトリが無いと open で落ちるので、念のため作る
    try:
        rd = os.path.dirname(os.path.abspath(raw_path))
        if rd:
            os.makedirs(rd, exist_ok=True)
        idd = os.path.dirname(os.path.abspath(ids_path))
        if idd:
            os.makedirs(idd, exist_ok=True)
        pd = os.path.dirname(os.path.abspath(priv_path))
        if pd:
            os.makedirs(pd, exist_ok=True)
    except Exception:
        pass

    # 小さな対戦数/途中確認でも「ファイルに見える」ように、時間ベースでも flush する
    flush_every_sec = float(os.getenv("WRITER_FLUSH_SEC", "2.0"))
    do_fsync = (str(os.getenv("WRITER_FSYNC", "0")).strip() == "1")
    last_flush_t = time.time()

    raw_f = open(raw_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
    ids_f = open(ids_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
    priv_ids_f = open(priv_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
    raw_lines_written = 0
    id_lines_written = 0
    priv_id_lines_written = 0
    batch_count = 0

    def _flush_all() -> None:
        nonlocal last_flush_t
        try:
            raw_f.flush(); ids_f.flush(); priv_ids_f.flush()
        except Exception:
            pass
        if do_fsync:
            try: os.fsync(raw_f.fileno())
            except Exception: pass
            try: os.fsync(ids_f.fileno())
            except Exception: pass
            try: os.fsync(priv_ids_f.fileno())
            except Exception: pass
        last_flush_t = time.time()

    try:
        while True:
            try:
                item = queue.get(timeout=0.5)
            except _queue_mod.Empty:
                # writer 側で stop_event を見て穏当終了 & 定期 flush
                if stop_event.is_set():
                    break
                if flush_every_sec > 0 and (time.time() - last_flush_t) >= flush_every_sec:
                    _flush_all()
                continue

            if item is None:
                # POISON PILL（終端）
                break

            try:
                # item 形式（想定）: ("batch", raw_lines: List[str], id_lines: List[str], priv_id_lines: List[str])
                if not isinstance(item, (list, tuple)) or len(item) == 0:
                    print(f"[writer] 予期しないメッセージ形式: {type(item)} → スキップします")
                    continue

                kind = item[0]
                if kind != "batch":
                    # 不明メッセージは無視（将来の拡張用）
                    continue

                # 安全に各配列を取り出す（長さ不足でも落ちないように）
                # item: ("batch", game_id, raw_lines, id_lines, priv_id_lines) を優先して解釈（旧形式も許容）
                game_id = item[1] if len(item) >= 2 and isinstance(item[1], str) else None
                _base_i = 2 if game_id is not None else 1

                raw_lines      = item[_base_i]     if len(item) > _base_i     and isinstance(item[_base_i], list) else []
                id_lines       = item[_base_i + 1] if len(item) > _base_i + 1 and isinstance(item[_base_i + 1], list) else []
                priv_id_lines  = item[_base_i + 2] if len(item) > _base_i + 2 and isinstance(item[_base_i + 2], list) else []

                # 受け取った game_id.log に parent 側の [DONE]/[WINRATE]/[writer] も追記する
                # NOTE: tee はプロセス開始時に 1 回だけ起動する。ここでの再起動は stdout チェーンの多重化を招く。
                if game_id is not None:
                    pass

                # まとめ書き（ループより高速）
                if raw_lines:
                    raw_f.write('\n'.join(raw_lines) + '\n')
                    raw_lines_written += len(raw_lines)

                if id_lines:
                    ids_f.write('\n'.join(id_lines) + '\n')
                    id_lines_written += len(id_lines)

                if priv_id_lines:
                    priv_ids_f.write('\n'.join(priv_id_lines) + '\n')
                    priv_id_lines_written += len(priv_id_lines)

                batch_count += 1
                if BATCH_FLUSH_INTERVAL > 0 and (batch_count % BATCH_FLUSH_INTERVAL) == 0:
                    _flush_all()
                elif flush_every_sec > 0 and (time.time() - last_flush_t) >= flush_every_sec:
                    _flush_all()

                if JSONL_ROTATE_LINES > 0 and raw_lines_written >= JSONL_ROTATE_LINES:
                    _flush_all()
                    raw_f.close()
                    raw_idx += 1
                    raw_path = _next_rotated_path(RAW_JSONL_PATH, raw_idx)
                    raw_f = open(raw_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
                    raw_lines_written = 0
                    print(f"[writer] rotate raw -> {raw_path}")

                if JSONL_ROTATE_LINES > 0 and id_lines_written >= JSONL_ROTATE_LINES:
                    _flush_all()
                    ids_f.close()
                    ids_idx += 1
                    ids_path = _next_rotated_path(IDS_JSONL_PATH, ids_idx)
                    ids_f = open(ids_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
                    id_lines_written = 0
                    print(f"[writer] rotate ids -> {ids_path}")

                if JSONL_ROTATE_LINES > 0 and priv_id_lines_written >= JSONL_ROTATE_LINES:
                    _flush_all()
                    priv_ids_f.close()
                    priv_idx += 1
                    priv_path = _next_rotated_path(PRIVATE_IDS_JSON_PATH, priv_idx)
                    priv_ids_f = open(priv_path, 'a', encoding='utf-8', buffering=IO_BUFFERING_BYTES)
                    priv_id_lines_written = 0
                    print(f"[writer] rotate private_ids -> {priv_path}")

            except Exception as e:
                print(f"[writer] 追記中に例外: {e}  ※このバッチはスキップします")
                continue
    finally:
        try:
            _flush_all()
        except Exception:
            pass
        try:
            raw_f.close()
        except Exception:
            pass
        try:
            ids_f.close()
        except Exception:
            pass
        try:
            priv_ids_f.close()
        except Exception:
            pass
        try:
            raw_sz = os.path.getsize(raw_path) if os.path.exists(raw_path) else -1
            ids_sz = os.path.getsize(ids_path) if os.path.exists(ids_path) else -1
            prv_sz = os.path.getsize(priv_path) if os.path.exists(priv_path) else -1
            print(f"[writer] closed: raw_lines={raw_lines_written} ids_lines={id_lines_written} priv_lines={priv_id_lines_written} batches={batch_count}")
            print(f"[writer] files: raw={raw_path} size={raw_sz}  ids={ids_path} size={ids_sz}  priv={priv_path} size={prv_sz}")
        except Exception:
            pass

def play_continuous_matches_worker(num_matches: int, queue: "Queue", mcc_agg=None, gpu_req_q=None, run_game_id=None) -> None:
    """
    既存 play_continuous_matches のワーカープロセス版。
    ファイルは開かず、各試合ぶんの raw_lines / id_lines / priv_id_lines を "batch" として queue に送る。
    """
    print(f"[TARGET] 目標対戦数={TARGET_EPISODES} / このワーカー予定={num_matches}")

    # 追加: このワーカー内のローカル進捗
    logged_count = 0
    skipped_deckout_count = 0
    pi_missing_public = 0
    pi_missing_private = 0

    # 追加: pi_source 内訳（public/private）を worker 内で集計
    pi_source_hist_public = {}
    pi_source_hist_private = {}

    # --- BLAS / Torch のスレッド数を抑制（過剰並列の防止） ---
    try:
        os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS_PER_PROC))
        os.environ.setdefault("MKL_NUM_THREADS", str(TORCH_THREADS_PER_PROC))
    except Exception:
        pass

    # --- ワーカー固有の乱数シードを設定 ---
    try:
        import numpy as np
    except Exception:
        np = None
    seed_base = int(time.time()) ^ os.getpid() ^ random.getrandbits(32)
    random.seed(seed_base)
    torch.manual_seed(seed_base & 0x7fffffff)
    if np is not None:
        np.random.seed(seed_base & 0x7fffffff)

    # ★★★ 子プロセス内で方策を遅延ロード＆フック（未定義ならスキップ）
    try:
        _ensure_policies()
    except NameError:
        pass

    # ★ 追加: このワーカー専用の方策インスタンスを用意（policy_p1 が未定義でも安全に動作させる）
    try:
        _p1 = policy_p1
        _p2 = policy_p2
    except NameError:
        _p1 = None
        _p2 = None
    if _p1 is None:
        _p1 = build_policy(P1_POLICY, MODEL_DIR_P1)
    if _p2 is None:
        _p2 = build_policy(P2_POLICY, MODEL_DIR_P2)
    local_policy_p1 = _p1
    local_policy_p2 = _p2

    # ✅ ここを追加（各プロセスでコンバータを用意）
    converter = get_converter()
    assert hasattr(converter, "convert_record")

    def _dump_policy_stats_worker(label: str, pol) -> None:
        try:
            if pol is None:
                print(f"[POLICY_STATS/worker {os.getpid()}] {label}: policy=None")
                return

            total = getattr(pol, "stats_total", None)
            model = getattr(pol, "stats_from_model", None)
            fallback = getattr(pol, "stats_from_fallback", None)
            rnd = getattr(pol, "stats_from_random", None)
            err = getattr(pol, "stats_errors", None)
            last = getattr(pol, "last_source", None)

            if total is None and model is None and fallback is None and rnd is None and err is None:
                pol_name = getattr(pol, "__class__", type(pol)).__name__
                print(f"[POLICY_STATS/worker {os.getpid()}] {label}: class={pol_name} (no stats_*)")
                return

            total = int(total or 0)
            model = int(model or 0)
            fallback = int(fallback or 0)
            rnd = int(rnd or 0)
            err = int(err or 0)

            if total > 0:
                model_r = model / total * 100.0
                fallback_r = fallback / total * 100.0
                rnd_r = rnd / total * 100.0
            else:
                model_r = fallback_r = rnd_r = 0.0

            ms = getattr(pol, "_main_success_calls", None)
            mf = getattr(pol, "_main_failed_calls", None)
            fb = getattr(pol, "_fallback_calls", None)

            print(
                f"[POLICY_STATS/worker {os.getpid()}] {label}: total={total} "
                f"model={model}({model_r:.2f}%) "
                f"fallback={fallback}({fallback_r:.2f}%) "
                f"random={rnd}({rnd_r:.2f}%) errors={err} last={last} "
                f"main_ok={ms} main_fail={mf} fallback_ok={fb}"
            )
        except Exception as _e:
            print(f"[POLICY_STATS/worker {os.getpid()}] {label}: dump failed: {_e!r}")

    for match_num in range(1, num_matches + 1):
        print(f"\n{'='*60}\n第{match_num}戦開始（worker）\n{'='*60}")

        base_dir = (PER_MATCH_LOG_DIR if PER_MATCH_LOG_DIR is not None else tempfile.gettempdir())
        match_log_file = os.path.join(base_dir, f"ai_vs_ai_match_{os.getpid()}_{match_num}.log")
        ml_log_file    = os.path.join(base_dir, f"ai_vs_ai_match_{os.getpid()}_{match_num}.ml.jsonl")

        # ★ 追加: 残骸誤読を防ぐため、mlログを必ず空にする
        open(ml_log_file, "w", encoding="utf-8").close()

        # --- プレイヤー・マッチ作成（毎戦リセット） ---
        fresh_deck1 = Deck(cards=make_deck_from_recipe(deck1_recipe))
        fresh_deck2 = Deck(cards=make_deck_from_recipe(deck2_recipe))
        p1, p2 = Player("p1", fresh_deck1, is_bot=True), Player("p2", fresh_deck2, is_bot=True)
        first, second = decide_first_player(p1, p2)

        def _recipe_to_enums(recipe):
            cards = make_deck_from_recipe(recipe)
            out = []
            for c in cards:
                cid = 0
                enum_ = getattr(c, "card_enum", None)
                if isinstance(enum_, int):
                    cid = enum_
                elif hasattr(enum_, "value"):
                    v = enum_.value
                    if isinstance(v, (tuple, list)) and v:
                        cid = int(v[0])
                    elif isinstance(v, int):
                        cid = v
                if cid == 0 and hasattr(c, "id"):
                    v = getattr(c, "id")
                    if isinstance(v, int):
                        cid = v
                    elif hasattr(v, "value"):
                        vv = v.value
                        if isinstance(vv, (tuple, list)) and vv:
                            cid = int(vv[0])
                        elif isinstance(vv, int):
                            cid = vv
                out.append(int(cid) if isinstance(cid, int) else 0)
            return out

        p1.initial_deck_enums = _recipe_to_enums(deck1_recipe)
        p2.initial_deck_enums = _recipe_to_enums(deck2_recipe)
        p1.initial_deck_locked = True
        p2.initial_deck_locked = True
        _run_gid = (str(run_game_id).strip() if run_game_id else "")
        if not _run_gid:
            _run_gid = os.getenv("AI_VS_AI_RUN_GAME_ID", "").strip()
        if not _run_gid:
            _run_gid = _RUN_GAME_ID

        run_id = _run_gid
        game_id = f"{run_id}_{os.getpid()}_{match_num}"

        try:
            _emit_policy_boot_logs_once()
        except NameError:
            pass

        reward_shaping_obj = None
        if USE_MCC:
            try:
                try:
                    from pokepocketsim import reward_shaping as reward_shaping
                except Exception:
                    import reward_shaping

                try:
                    from pokepocketsim.my_mcc_sampler import mcc_sampler as _mcc_sampler
                except Exception:
                    from my_mcc_sampler import mcc_sampler as _mcc_sampler

                # MCC を「回す」ために value_net が必要（ここは仮の定数モデル）
                # ※本命はあなたの ValueNet（学習済み）をここに渡してください
                class _ConstantValueNet(torch.nn.Module):
                    def forward(self, x):
                        try:
                            b = int(x.shape[0])
                        except Exception:
                            b = 1
                        return torch.zeros((b, 1), dtype=torch.float32, device=x.device)

                value_net = _ConstantValueNet()

                # MCC だけ回したいので scale=0.0（報酬は変更しない）
                reward_shaping_obj = reward_shaping.create_reward_shaping(
                    value_net=value_net,
                    mcc_sampler=_mcc_sampler,
                    scale=0.0,
                )
            except Exception as e:
                reward_shaping_obj = None
                try:
                    print(f"[MCC][WARN] reward_shaping init failed: {e}")
                except Exception:
                    pass
                try:
                    print(f"[MCC][WARN] reward_shaping init failed: {e}")
                except Exception:
                    pass

        match = Match(
            first, second,
            log_file=match_log_file,
            log_mode=True,
            game_id=game_id,
            mcc_top_k=MCC_TOP_K,
            use_reward_shaping=bool(USE_MCC and (reward_shaping_obj is not None)),
            reward_shaping=reward_shaping_obj,
            skip_deckout_logging=SKIP_DECKOUT_LOGGING,
        )
        match.policy_p1 = local_policy_p1
        match.policy_p2 = local_policy_p2

        # ★変更: 方策が encoder を持たなくても obs_vec を出せるように強制装着
        _ensure_match_encoder(match, local_policy_p1, local_policy_p2)
        # ★追加: PhaseD-Q / OnlineMixedPolicy 用に converter を match / policy に接続
        try:
            conv = globals().get("converter", None)
            if conv is None:
                conv = get_converter()
                globals()["converter"] = conv

            match.converter = conv
            match.action_converter = conv
            for _pol in (local_policy_p1, local_policy_p2):
                # unwrap: OnlineMixedPolicy / wrappers → 内側にも converter を伝播
                _pols = [_pol]
                try:
                    _mp = getattr(_pol, "main_policy", None) or getattr(_pol, "main", None)
                    if _mp is not None:
                        _pols.append(_mp)
                except Exception:
                    pass
                try:
                    _pp = getattr(_pol, "policy", None)
                    if _pp is not None:
                        _pols.append(_pp)
                except Exception:
                    pass

                for _p in _pols:
                    if _p is None:
                        continue
                    if getattr(_p, "converter", None) is None:
                        try:
                            _p.converter = conv
                        except Exception:
                            pass
                    if getattr(_p, "action_converter", None) is None:
                        try:
                            _p.action_converter = conv
                        except Exception:
                            pass
        except Exception:
            pass

        # ★追加: obs_vec を確実に出力
        if EMIT_OBS_VEC_FOR_CANDIDATES and getattr(match, "encoder", None) is not None:
            try:
                def _get_public_state(p):
                    if p is None:
                        return None
                    ps = getattr(p, "public_state", None)
                    if callable(ps):
                        try:
                            return ps()
                        except Exception:
                            return None
                    if isinstance(ps, dict):
                        return ps
                    for nm in ("to_public_state", "get_public_state"):
                        fn = getattr(p, nm, None)
                        if callable(fn):
                            try:
                                out = fn()
                                return out if isinstance(out, dict) else None
                            except Exception:
                                pass
                    return None

                def _get_match_public_players(_m):
                    ps = getattr(_m, "public_state", None)
                    if callable(ps):
                        try:
                            ps = ps()
                        except Exception:
                            ps = None

                    players = None
                    try:
                        players = getattr(ps, "players", None)
                    except Exception:
                        players = None
                    if players is None and isinstance(ps, dict):
                        try:
                            players = ps.get("players", None)
                        except Exception:
                            players = None

                    return players if isinstance(players, list) else None

                me_player  = getattr(match, "starting_player", None) or getattr(match, "player1", None) or getattr(match, "p1", None) or getattr(match, "first_player", None)
                opp_player = getattr(match, "second_player", None) or getattr(match, "player2", None) or getattr(match, "p2", None) or getattr(match, "second_player", None)

                me_ps  = _get_public_state(me_player)
                opp_ps = _get_public_state(opp_player)

                # ★重要: Player 側の public_state がまだ無い場合があるため、
                #        match.public_state["players"] から拾い直す（me/opp が取れていても実施）
                if not isinstance(me_ps, dict) or not isinstance(opp_ps, dict):
                    players = _get_match_public_players(match)
                    if isinstance(players, list) and len(players) >= 2:
                        _p0 = players[0]
                        _p1 = players[1]

                        if not isinstance(me_ps, dict):
                            me_ps = _p0 if isinstance(_p0, dict) else _get_public_state(_p0)
                        if not isinstance(opp_ps, dict):
                            opp_ps = _p1 if isinstance(_p1, dict) else _get_public_state(_p1)

                if not isinstance(me_ps, dict) or not isinstance(opp_ps, dict):
                    print("[OBS] ⚠️ cannot find public_state players on match; skip initial obs_vec.")
                else:
                    def _get_legal_actions_any(_m, _player=None):
                        for nm in (
                            "legal_actions",
                            "get_legal_actions",
                            "list_legal_actions",
                            "enumerate_legal_actions",
                            "legal_actions_for_player",
                            "legal_actions_for",
                        ):
                            fn = getattr(_m, nm, None)
                            if callable(fn):
                                try:
                                    return fn()
                                except TypeError:
                                    try:
                                        return fn(_player) if _player is not None else []
                                    except Exception:
                                        pass
                                except Exception:
                                    pass

                        for attr in ("legal_actions_list", "_legal_actions", "_legal_actions_cache"):
                            v = getattr(_m, attr, None)
                            if isinstance(v, (list, tuple)):
                                return list(v)

                        return []

                    sb = {"me": me_ps, "opp": opp_ps}

                    turn_player = (
                        getattr(match, "turn_player", None)
                        or getattr(match, "player_in_turn", None)
                        or getattr(match, "current_player", None)
                        or me_player
                    )

                    la_src = _get_legal_actions_any(match, turn_player)

                    conv = (
                        getattr(match, "converter", None)
                        or getattr(match, "action_converter", None)
                        or globals().get("converter", None)
                    )

                    la_ids = None
                    if conv is not None and hasattr(conv, "convert_legal_actions_32d"):
                        try:
                            la_ids = conv.convert_legal_actions_32d(la_src)
                        except Exception:
                            la_ids = None

                    if not isinstance(la_ids, list):
                        la_ids = []
                        for a in (la_src or []):
                            try:
                                v = a.to_id_vec()
                                if isinstance(v, (list, tuple)) and len(v) > 0:
                                    la_ids.append(list(v))
                            except Exception:
                                pass

                    if not la_ids:
                        print("[OBS] ⚠️ legal actions empty; skip initial obs_vec.")
                    else:
                        obs_vec = _safe_encode_obs_for_candidates(sb, match.encoder, la_ids)
                        if isinstance(obs_vec, list) and len(obs_vec) > 0:
                            match.obs_vec_example = obs_vec
                            print(f"[OBS] ✅ obs_vec generated (len={len(obs_vec)})")
                        else:
                            print("[OBS] ⚠️ obs_vec is empty ([]); check scaler or encoder.")
            except Exception as e:
                print(f"[OBS] ⚠️ obs_vec generation failed: {e}")

        match.log_full_info  = LOG_FULL_INFO
        match.use_mcc        = USE_MCC
        match.mcc_samples    = MCC_SAMPLES

        match.cpu_workers_mcc = CPU_WORKERS_MCC
        match.mcc_every       = MCC_EVERY

        match.mcc_top_k      = MCC_TOP_K
        match.ml_log_file    = ml_log_file
        match.skip_deckout_logging = SKIP_DECKOUT_LOGGING

        match.opp_archetypes = [
            {"enums": _recipe_to_enums(ALL_DECK_RECIPES["deck01"]), "weight": 0.4},
            {"enums": _recipe_to_enums(ALL_DECK_RECIPES["deck02"]), "weight": 0.6},
        ]

        with open(match_log_file, "w", encoding="utf-8") as f:
            f.write(f"AI 同士対戦ログ  match(worker)={match_num}\n")
        # --- 対戦実行（試合後の集計ログまで game_id.log に統一） ---
        try:
            from contextlib import nullcontext as _nullcontext
        except Exception:
            _nullcontext = None

        _use_gamelog_ctx = True
        try:
            _active = (os.getenv("AI_VS_AI_GAMELOG_ACTIVE", "0") == "1")
            _path0  = os.getenv("AI_VS_AI_GAMELOG_PATH", "")

            def _norm(p: str) -> str:
                try:
                    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))
                except Exception:
                    return str(p)

            _want = os.path.join(GAMELOG_DIR, str(game_id) + ".log")

            # env が無い/壊れている場合でも、stdout チェーンから現在の tee 先を拾う
            if (not _active) or (not _path0):
                try:
                    obj = getattr(sys, "stdout", None)
                    for _ in range(16):
                        if obj is None:
                            break
                        if getattr(obj, "_console_tee_active", False):
                            cur = getattr(obj, "_console_tee_path", "") or ""
                            if not cur:
                                fp0 = getattr(obj, "_fp", None)
                                cur = getattr(fp0, "name", "") if fp0 is not None else ""
                            _path0 = cur
                            _active = True
                            break
                        obj = getattr(obj, "_base", None)
                except Exception:
                    pass

            if _active and _path0:
                _use_gamelog_ctx = False
        except Exception:
            _use_gamelog_ctx = True

        _ctx = (_nullcontext() if (not _use_gamelog_ctx and _nullcontext is not None) else
                GameLogContext(game_id, log_dir=GAMELOG_DIR, to_console=True))

        mcc_calls_this_game = None
        _mcc_calls_before = None
        try:
            _mcc_calls_before = int(mcc_debug_snapshot().get("total_calls", 0))
        except Exception:
            _mcc_calls_before = None

        with _ctx:
            match.play_one_match()

            try:
                _mcc_calls_after = int(mcc_debug_snapshot().get("total_calls", 0))
                if _mcc_calls_before is not None:
                    mcc_calls_this_game = int(_mcc_calls_after - _mcc_calls_before)
                    print(f"[MCC_CALLS] game_id={game_id} calls={mcc_calls_this_game} (samples={MCC_SAMPLES})")
            except Exception:
                pass

            # ★ 追加: Match から終了理由を拾っておく（後段の JSONL 出力で使う）
            end_reason = getattr(match, "_end_reason", None)
            if not end_reason:
                try:
                    gr = getattr(match, "game_result", None)
                    if isinstance(gr, dict):
                        end_reason = gr.get("reason") or gr.get("end_reason")
                except Exception:
                    pass
            end_reason = (str(end_reason).upper() if end_reason else "UNKNOWN")

            print(f"第{match_num}戦終了（worker）  →  {match_log_file}")

            # --- 解析用のソースは、可能なら常に .ml.jsonl（構造化）を最優先 ---
            src_path = ml_log_file
            entries_all = parse_log_file(src_path)

            # ✅ 勝者推定用に“未加工の全行”を先に確保
            _entries_for_winner = entries_all[:]

            # ★ 先に“未フィルタ全行”でデッキアウト判定
            _is_deckout_game = _is_deckout_game_from_all(entries_all)

            if SKIP_DECKOUT_LOGGING and _is_deckout_game:
                print(f"[FILTER] deck-out detected → skip game_id={game_id}")
                # 集計を加算（“スキップ”と“試行”）
                try:
                    if mcc_agg is not None:
                        mcc_agg["skipped_deckout"]   = int(mcc_agg.get("skipped_deckout", 0)) + 1
                        mcc_agg["attempted_matches"] = int(mcc_agg.get("attempted_matches", 0)) + 1
                    skipped_deckout_count += 1
                    # 進捗を表示（このワーカーのローカル + 全体の合計）
                    total_logged_all = int(mcc_agg.get("logged_matches", 0)) if mcc_agg is not None else logged_count
                    print(f"[COUNT/worker {os.getpid()}] logged={logged_count} skipped_deckout={skipped_deckout_count} total_tried={match_num} | total_logged_all={total_logged_all}")
                except Exception:
                    pass
                # per-match ファイル後始末（任意）
                try:
                    if os.path.exists(match_log_file):
                        os.remove(match_log_file)
                    if os.path.exists(ml_log_file):
                        os.remove(ml_log_file)
                except Exception:
                    pass
                if MATCH_SLEEP_SEC > 0.0:
                    time.sleep(MATCH_SLEEP_SEC)
                continue

        # ★ デッキアウトでない場合のみ、意思決定行フィルタを適用
        entries_all = [e for e in entries_all if _is_decision_entry(e)]

        # ★ 追加: この試合の game_id の行だけを採用（保険）
        def _gid_of(e):
            gid = e.get("game_id")
            if not gid:
                sb = e.get("state_before") or {}
                sa = e.get("state_after") or {}
                gid = sb.get("game_id") or sa.get("game_id")
            return gid

        entries = [e for e in entries_all if _gid_of(e) == game_id]
        if not entries and src_path != match_log_file:
            # ml が古かった可能性 → .log を再読込して同様にフィルタ
            entries_all = parse_log_file(match_log_file)

            # ✅ フォールバック時も勝者推定用の未加工コピーを差し替え
            _entries_for_winner = entries_all[:]

            # フォールバックでもデッキアウト判定→スキップを先に
            _is_deckout_game = _is_deckout_game_from_all(entries_all)
            if SKIP_DECKOUT_LOGGING and _is_deckout_game:
                print(f"[FILTER] deck-out detected (fallback) → skip game_id={game_id}")
                try:
                    if mcc_agg is not None:
                        mcc_agg["skipped_deckout"]   = int(mcc_agg.get("skipped_deckout", 0)) + 1
                        mcc_agg["attempted_matches"] = int(mcc_agg.get("attempted_matches", 0)) + 1
                    skipped_deckout_count += 1
                    total_logged_all = int(mcc_agg.get("logged_matches", 0)) if mcc_agg is not None else logged_count
                    print(f"[COUNT/worker {os.getpid()}] logged={logged_count} skipped_deckout={skipped_deckout_count} total_tried={match_num} | total_logged_all={total_logged_all}")
                except Exception:
                    pass
                try:
                    if os.path.exists(match_log_file):
                        os.remove(match_log_file)
                    if os.path.exists(ml_log_file):
                        os.remove(ml_log_file)
                except Exception:
                    pass
                if MATCH_SLEEP_SEC > 0.0:
                    time.sleep(MATCH_SLEEP_SEC)
                continue

            # 意思決定行フィルタを適用してから game_id で絞る
            entries_all = [e for e in entries_all if _is_decision_entry(e)]
            entries = [e for e in entries_all if _gid_of(e) == game_id]

        try:
            def _allow_meta(e):
                return True
            entries = [e for e in entries if _allow_meta(e)]
        except Exception:
            pass

        # --- ここで「完全情報」と「公開版（非公開情報除去）」を用意 ---
        entries_full = entries                                  # 無加工＝完全情報
        entries_pub  = [_strip_privates_recursive(e) for e in entries]
        entries_pub  = trim_entries_after_canonical_terminal(entries_pub)

        # ✅ この試合(game_id)の“未加工ログ”から勝者を先に一度だけ推定
        entries_for_winner = [e for e in _entries_for_winner if _gid_of(e) == game_id]
        try:
            winner = (
                infer_winner_from_entries(entries_for_winner, fallback_p1="p1", fallback_p2="p2")
                if entries_for_winner else "unknown"
            )
        except Exception:
            winner = "unknown"

        raw_lines = []
        id_lines  = []
        priv_id_lines = []
        did_log_this_match = False  # ← 先に安全に初期化
        action_records = 0          # ← a_idx を持つ行動ステップ数
        actions_with_obs = 0        # ← そのうち obs_vec が非空のステップ数

        # --- RAW 側: LOG_FULL_INFO=True なら完全情報、False なら公開版を書き出す ---
        for entry in (entries_full if LOG_FULL_INFO else entries_pub):
            try:
                if isinstance(entry, dict) and "end_reason" not in entry:
                    entry["end_reason"] = end_reason
            except Exception:
                pass
            raw_lines.append(json.dumps(entry, ensure_ascii=False))

        # ✅ 勝者推定は“未加工 + 同一game_id”で実施済み（entries_for_winner 参照）

        # --- IDS 側: LOG_FULL_INFO=False のときのみ出力（★ デッキアウト試合は除外可） ---
        # ★ Phase A: ALWAYS_OUTPUT_BOTH_IDS=True の場合、LOG_FULL_INFO に関わらず公開IDも出力する
        if (not LOG_FULL_INFO) or ALWAYS_OUTPUT_BOTH_IDS:

            # 追加：バッファの初期化
            id_lines = []

            for t, entry in enumerate(entries_pub):
                if SKIP_DECKOUT_LOGGING and _is_deckout_game:
                    continue
                id_entry = converter.convert_record(entry)

                # ★ 追加: ゲーム終了理由を付与（Match から取得した end_reason）
                try:
                    if "end_reason" not in id_entry:
                        id_entry["end_reason"] = end_reason
                except Exception:
                    pass

                # 既存キーを消してから t を確実に書く（後段での上書き対策）
                id_entry.pop("turn_index", None)
                id_entry.pop("ply_index", None)
                id_entry["turn_index"] = t
                id_entry["ply_index"]  = t

                if has_minus_one(id_entry):
                    print("⚠️ 変換に -1 が残っています ->", id_entry)

                # z 付与
                try:
                    cur = None
                    sb_src = entry.get("state_before") or {}
                    if isinstance(sb_src, dict):
                        cur = sb_src.get("current_player")

                    # current_player を p1/p2 視点に正規化
                    pov = None
                    if cur in ("p1", "me"):
                        pov = "p1"
                    elif cur in ("p2", "opp"):
                        pov = "p2"

                    _z = 0
                    if isinstance(winner, str):
                        if winner == "draw":
                            _z = 0
                        elif winner in ("p1", "p2") and pov is not None:
                            # POV プレイヤーが勝者なら +1 / 敗者なら -1
                            _z = 1 if winner == pov else -1
                    id_entry["z"] = _z
                except Exception:
                    id_entry["z"] = 0

                # pi 計算（候補は変換前 → ID化）
                try:
                    la_src = _pick_legal_actions(entry)
                    la_ids = converter.convert_legal_actions(la_src) if hasattr(converter, "convert_legal_actions") else None
                    if la_ids is None:
                        la_ids = []
                        if isinstance(la_src, list) and la_src:
                            if all(isinstance(a, int) for a in la_src):
                                la_ids = [[int(a)] for a in la_src]
                            elif all(isinstance(a, dict) and ("id" in a) for a in la_src):
                                la_ids = [[int(a.get("id"))] for a in la_src if isinstance(a.get("id"), (int, str)) and str(a.get("id")).isdigit()]
                            elif all(isinstance(a, str) and a.isdigit() for a in la_src):
                                la_ids = [[int(a)] for a in la_src]

                    # まず、元ログに MCTS 由来の π があればそれを優先的に利用（entry/state_before/state_after を探索）
                    pi_src = None
                    pi_from = None
                    if isinstance(entry, dict):
                        for _k in ("pi", "mcts_pi", "policy_pi", "root_pi", "_mcts_pi"):
                            try:
                                _v = entry.get(_k)
                            except Exception:
                                _v = None
                            if _v is not None:
                                pi_src = _v
                                pi_from = f"entry:{_k}"
                                break

                        if pi_src is None:
                            sb0 = entry.get("state_before") or {}
                            if isinstance(sb0, dict):
                                for _k in ("pi", "mcts_pi", "policy_pi", "root_pi", "_mcts_pi"):
                                    try:
                                        _v = sb0.get(_k)
                                    except Exception:
                                        _v = None
                                    if _v is not None:
                                        pi_src = _v
                                        pi_from = f"state_before:{_k}"
                                        break

                        if pi_src is None:
                            sa0 = entry.get("state_after") or {}
                            if isinstance(sa0, dict):
                                for _k in ("pi", "mcts_pi", "policy_pi", "root_pi", "_mcts_pi"):
                                    try:
                                        _v = sa0.get(_k)
                                    except Exception:
                                        _v = None
                                    if _v is not None:
                                        pi_src = _v
                                        pi_from = f"state_after:{_k}"
                                        break

                    if isinstance(pi_src, dict) and isinstance(la_ids, list):
                        try:
                            _tmp = [0.0] * len(la_ids)
                            for _k, _v in pi_src.items():
                                try:
                                    ii = int(_k)
                                    if 0 <= ii < len(_tmp):
                                        _tmp[ii] = float(_v)
                                except Exception:
                                    continue
                            pi_src = _tmp
                        except Exception:
                            pi_src = None

                    if isinstance(pi_src, list) and isinstance(la_ids, list) and len(pi_src) == len(la_ids):
                        try:
                            id_entry["pi"] = [float(x) for x in pi_src]
                            id_entry["pi_source"] = ("raw" if not pi_from else f"raw:{pi_from}")
                        except Exception:
                            id_entry.pop("pi", None)

                    act_vec = None
                    ar_src = entry.get("action_result") or {}
                    act_src = ar_src.get("action") or ar_src.get("macro") if isinstance(ar_src, dict) else None
                    if act_src is not None:
                        if isinstance(act_src, list) and all(isinstance(x, int) for x in act_src):
                            act_vec = act_src
                        else:
                            act_vec = converter.action_to_id(act_src)

                    if act_vec is None:
                        ar_conv = id_entry.get("action_result") or {}
                        _tmp = ar_conv.get("action") or ar_conv.get("macro")
                        if isinstance(_tmp, list) and all(isinstance(x, int) for x in _tmp):
                            act_vec = _tmp
                        elif isinstance(_tmp, int):
                            act_vec = [_tmp]

                    _pi = None
                    if ("pi" not in id_entry) and isinstance(la_ids, list) and la_ids and (act_vec is not None):
                        idx = -1
                        for i, a in enumerate(la_ids):
                            if isinstance(a, list) and a == act_vec:
                                idx = i
                                break
                            if isinstance(a, int) and isinstance(act_vec, list) and len(act_vec) == 1 and a == act_vec[0]:
                                idx = i
                                break
                        if idx >= 0:
                            _pi = [0] * len(la_ids)
                            _pi[idx] = 1
                    if _pi is not None and ("pi" not in id_entry):
                        id_entry["pi"] = _pi
                        id_entry["pi_source"] = "onehot_from_legal_actions"

                    # ★ 追加: legal_actions を “空なら” 同期（上書きはしない）
                    if (not id_entry.get("legal_actions")) and isinstance(la_ids, list) and la_ids:
                        id_entry["legal_actions"] = la_ids

                except Exception:
                    pass

                # a_idx フォールバック
                try:
                    # まず action_result / act_vec から a_idx を決める
                    ar2 = entry.get("action_result") or {}
                    act2 = ar2.get("action") or ar2.get("macro") if isinstance(ar2, dict) else None
                    if hasattr(converter, "action_to_index") and act2 is not None:
                        _idx = converter.action_to_index(act2)
                        if isinstance(_idx, int) and _idx >= 0:
                            id_entry["a_idx"] = int(_idx)
                    else:
                        _act_vec = act_vec if isinstance(act_vec, list) else None
                        if _act_vec and len(_act_vec) > 0 and isinstance(_act_vec[0], int) and _act_vec[0] >= 0:
                            id_entry["a_idx"] = int(_act_vec[0])

                    # まだ a_idx が決まっておらず、pi と la_ids があれば pi から復元する
                    if ("a_idx" not in id_entry) and ("pi" in id_entry) and isinstance(la_ids, list) and la_ids:
                        _pi = id_entry.get("pi")
                        if isinstance(_pi, list) and len(_pi) == len(la_ids) and _pi:
                            max_i = max(range(len(_pi)), key=lambda i: _pi[i])
                            chosen = la_ids[max_i]
                            if isinstance(chosen, int):
                                id_entry["a_idx"] = int(chosen)
                            elif isinstance(chosen, list) and chosen and isinstance(chosen[0], int):
                                id_entry["a_idx"] = int(chosen[0])
                except Exception:
                    pass

# 候補打点モデル向けフィールド
                if EMIT_CANDIDATE_FEATURES:
                    try:
                        la_src2 = _pick_legal_actions(entry)  # pyright: ignore[reportUnusedVariable]
                        la_ids2 = converter.convert_legal_actions(la_src2) if hasattr(converter, "convert_legal_actions") else []  # pyright: ignore[reportUnusedVariable]
                        # ★ 修正: legal_actions は 5ints のまま保持（32d化は encode_action_from_vec_32d 側で実施）
                        # 新: 正式フィールド（32d 統一）
                        id_entry["action_candidates_vec"] = _embed_legal_actions_32d(la_ids2)
                        id_entry["action_vec_dim"] = ACTION_VEC_DIM
                        id_entry["legal_actions_vec_dim"] = id_entry["action_vec_dim"]

                        # --- DEBUG: cand_vec の健全性を確認（スパム防止のため間引き） ---
                        try:
                            _dbg_cand = bool(globals().get("DEBUG_PHASEDQ_CAND", True))
                            _dbg_every = int(globals().get("DEBUG_PHASEDQ_CAND_EVERY", 50))
                            if _dbg_cand and (t < 5 or (t % _dbg_every) == 0):
                                _cands = id_entry.get("action_candidates_vec")
                                _n_la = len(la_ids2) if isinstance(la_ids2, list) else -1
                                _n_cv = len(_cands) if isinstance(_cands, list) else -1
                                _bad_dim = 0
                                _all_zero = 0
                                _all_m1 = 0
                                if isinstance(_cands, list):
                                    for _v in _cands:
                                        if not (isinstance(_v, list) and len(_v) == ACTION_VEC_DIM):
                                            _bad_dim += 1
                                            continue
                                        try:
                                            if all(float(x) == 0.0 for x in _v):
                                                _all_zero += 1
                                            if all(float(x) == -1.0 for x in _v):
                                                _all_m1 += 1
                                        except Exception:
                                            pass
                                _head = None
                                try:
                                    if isinstance(la_ids2, list) and la_ids2:
                                        _head = la_ids2[0]
                                except Exception:
                                    _head = None
                                print(
                                    f"[CANDDBG] t={t} la_src2_len={len(la_src2) if isinstance(la_src2, list) else 'NA'} "
                                    f"la_ids2_len={_n_la} cand_vec_len={_n_cv} bad_dim={_bad_dim} all_zero={_all_zero} all_m1={_all_m1} "
                                    f"la0_type={type(_head).__name__ if _head is not None else 'None'} la0_len={(len(_head) if isinstance(_head, list) else 'NA')}",
                                    flush=True,
                                )
                        except Exception:
                            pass

                        if EMIT_OBS_VEC_FOR_CANDIDATES:
                            sb_src_conv = id_entry.get("state_before") or {}  # pyright: ignore[reportUnusedVariable]
                            id_entry["obs_vec"] = build_obs_partial_vec(sb_src_conv)

                            # ★ 追加: AlphaZero MCTS ポリシーから π を再計算して付与（自己対戦モードのみ）
                            if SELFPLAY_ALPHAZERO_MODE and USE_MCTS_POLICY and (not str(id_entry.get("pi_source", "")).startswith("raw")):
                                try:
                                    _dbg_pi = bool(globals().get("DEBUG_RECOMPUTE_PI", True))
                                    cur_player = None
                                    try:
                                        _sb = entry.get("state_before") or {}
                                        if isinstance(_sb, dict):
                                            cur_player = _sb.get("current_player") or _sb.get("player") or _sb.get("turn_player")
                                    except Exception:
                                        cur_player = None

                                    mcts_policy = None
                                    if cur_player in ("p1", "me"):
                                        mcts_policy = local_policy_p1
                                    elif cur_player in ("p2", "opp"):
                                        mcts_policy = local_policy_p2
                                    else:
                                        mcts_policy = local_policy_p1 or local_policy_p2

                                    obs_vec_for_mcts = id_entry.get("obs_vec")

                                    # unwrap: OnlineMixedPolicy / wrappers → 内側のポリシーへ
                                    if mcts_policy is not None and hasattr(mcts_policy, "main_policy"):
                                        try:
                                            mcts_policy = getattr(mcts_policy, "main_policy")
                                        except Exception:
                                            pass
                                    if mcts_policy is not None and hasattr(mcts_policy, "policy"):
                                        try:
                                            mcts_policy = getattr(mcts_policy, "policy")
                                        except Exception:
                                            pass

                                    # MCTS に渡す候補は 32d を優先（無ければ従来通り la_ids2）
                                    la_for_mcts = None
                                    try:
                                        la_for_mcts = id_entry.get("action_candidates_vec")
                                    except Exception:
                                        la_for_mcts = None
                                    if not (
                                        isinstance(la_for_mcts, list)
                                        and la_for_mcts
                                        and all(isinstance(_v, list) and len(_v) == ACTION_VEC_DIM for _v in la_for_mcts)
                                    ):
                                        la_for_mcts = la_ids2

                                    if _dbg_pi and (t < 5):
                                        print(
                                            f"[PIDBG] t={t} pol_class={(getattr(mcts_policy, '__class__', type(mcts_policy)).__name__ if mcts_policy is not None else 'None')} "
                                            f"has_select_action={hasattr(mcts_policy, 'select_action')} obs_len={(len(obs_vec_for_mcts) if isinstance(obs_vec_for_mcts, list) else 'NA')} "
                                            f"la_ids2_len={(len(la_ids2) if isinstance(la_ids2, list) else 'NA')}",
                                            flush=True,
                                        )

                                    if (
                                        mcts_policy is not None
                                        and isinstance(obs_vec_for_mcts, list)
                                        and isinstance(la_for_mcts, list)
                                        and la_for_mcts
                                        and hasattr(mcts_policy, "select_action")
                                    ):
                                        a_vec_mcts, pi_mcts = mcts_policy.select_action(obs_vec_for_mcts, la_for_mcts)

                                        if _dbg_pi and (t < 5):
                                            print(
                                                f"[PIDBG] t={t} select_action_ret a_vec_type={type(a_vec_mcts).__name__} "
                                                f"pi_type={type(pi_mcts).__name__} pi_len={(len(pi_mcts) if isinstance(pi_mcts, list) else 'NA')} "
                                                f"expect_len={len(la_for_mcts)}",
                                                flush=True,
                                            )

                                        if DEBUG_MODEL_MCTS and (t % DEBUG_MODEL_MCTS_EVERY) == 1:
                                            _trace_policy_step(
                                                tag=f"recompute_pi t={t}",
                                                pol=mcts_policy,
                                                obs_vec=obs_vec_for_mcts,
                                                la_ids=la_for_mcts,
                                                chosen_vec=a_vec_mcts,
                                                pi=pi_mcts,
                                            )
                                        if isinstance(pi_mcts, list) and len(pi_mcts) == len(la_for_mcts):
                                            id_entry["pi"] = [float(x) for x in pi_mcts]
                                            id_entry["pi_source"] = "recomputed_mcts"
                                        elif _dbg_pi and (t < 5):
                                            print(f"[PIDBG] t={t} pi_mcts length mismatch -> skip attach", flush=True)
                                    elif _dbg_pi and (t < 5):
                                        print(f"[PIDBG] t={t} recompute_pi skipped (missing method or bad inputs)", flush=True)
                                except Exception as _e:
                                    try:
                                        if bool(globals().get("DEBUG_RECOMPUTE_PI", True)) and (t < 5):
                                            print(f"[PIDBG] t={t} recompute_pi failed: {_e!r}", flush=True)

                                    except Exception:
                                        pass

                        # ★ 追加: pi（候補上の one-hot 分布）を出力
                        if "pi" not in id_entry:
                            aidx = id_entry.get("a_idx")  # pyright: ignore[reportUnusedVariable]
                            if isinstance(aidx, int) and isinstance(la_ids2, list) and la_ids2:
                                try:
                                    n = len(la_ids2)  # pyright: ignore[reportUnusedVariable]
                                    pi = [0.0] * n  # pyright: ignore[reportUnusedVariable]
                                    j = None  # pyright: ignore[reportUnusedVariable]
                                    for i, _la in enumerate(la_ids2):
                                        if isinstance(_la, int) and _la == aidx:
                                            j = i
                                            break
                                        if isinstance(_la, list) and _la and isinstance(_la[0], int) and _la[0] == aidx:
                                            j = i
                                            break
                                    if j is not None:
                                        pi[j] = 1.0
                                        id_entry["pi"] = pi
                                        id_entry["pi_source"] = "onehot_from_a_idx"
                                except Exception:
                                    # a_idx が候補リストに見つからない場合は pi は付けない
                                    pass
                    except Exception:
                        pass

                # ★ 追加：空 legal_actions を削除（任意フラグ）
                if DROP_EMPTY_LEGAL_ACTIONS and isinstance(id_entry.get("legal_actions"), list) and not id_entry["legal_actions"]:
                    id_entry.pop("legal_actions", None)

                # ★ 追加: 行動ステップ数と obs_vec 付きステップ数を数える
                if "a_idx" in id_entry:
                    action_records += 1
                    v = id_entry.get("obs_vec")
                    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                        actions_with_obs += 1

                # ★ カウンタ: “最終的に pi が無かった行”だけ数える
                if "pi" not in id_entry:
                    try:
                        pi_missing_public += 1
                    except Exception:
                        pass

                try:
                    k = id_entry.get("pi_source")
                    if not k:
                        k = "(none)"
                    pi_source_hist_public[k] = int(pi_source_hist_public.get(k, 0)) + 1
                except Exception:
                    pass

                id_lines.append(json.dumps(id_entry, ensure_ascii=False))

        # --- PRIVATE_IDS 側: LOG_FULL_INFO=True のときのみ出力（完全情報ベースの全ID化） ---
        if LOG_FULL_INFO or ALWAYS_OUTPUT_BOTH_IDS:
            for t, entry in enumerate(entries_full):
                if SKIP_DECKOUT_LOGGING and _is_deckout_game:
                    continue
                id_entry_priv = converter.convert_record(entry, keep_private=True)

                # ★ 追加: ゲーム終了理由を付与（Match から取得した end_reason）
                try:
                    if "end_reason" not in id_entry_priv:
                        id_entry_priv["end_reason"] = end_reason
                except Exception:
                    pass

                # index は必ず t（既存キーはクリアしてから）
                id_entry_priv.pop("turn_index", None)
                id_entry_priv.pop("ply_index", None)
                id_entry_priv["turn_index"] = t
                id_entry_priv["ply_index"]  = t

                if has_minus_one(id_entry_priv):
                    print("⚠️ PRIVATE_IDS 変換に -1 が残っています ->", id_entry_priv)

                # z 付与
                try:
                    cur = None
                    sb_src = entry.get("state_before") or {}
                    if isinstance(sb_src, dict):
                        cur = sb_src.get("current_player")

                    # current_player を p1/p2 視点に正規化
                    pov = None
                    if cur in ("p1", "me"):
                        pov = "p1"
                    elif cur in ("p2", "opp"):
                        pov = "p2"

                    _z = 0
                    if isinstance(winner, str):
                        if winner == "draw":
                            _z = 0
                        elif winner in ("p1", "p2") and pov is not None:
                            # POV プレイヤーが勝者なら +1 / 敗者なら -1
                            _z = 1 if winner == pov else -1
                    id_entry_priv["z"] = _z
                except Exception:
                    id_entry_priv["z"] = 0

                # pi 計算＋ legal_actions 同期
                try:
                    la_src = _pick_legal_actions(entry)
                    la_ids = converter.convert_legal_actions(la_src) if hasattr(converter, "convert_legal_actions") else None
                    if la_ids is None:
                        la_ids = []
                        if isinstance(la_src, list) and la_src:
                            if all(isinstance(a, int) for a in la_src):
                                la_ids = [[int(a)] for a in la_src]
                            elif all(isinstance(a, dict) and ("id" in a) for a in la_src):
                                la_ids = [[int(a.get("id"))] for a in la_src if isinstance(a.get("id"), (int, str)) and str(a.get("id")).isdigit()]
                            elif all(isinstance(a, str) and a.isdigit() for a in la_src):
                                la_ids = [[int(a)] for a in la_src]

                    # まず、元ログに MCTS 由来の π があればそれを優先的に利用（entry/state_before/state_after を探索）
                    pi_src = None
                    pi_from = None
                    if isinstance(entry, dict):
                        for _k in ("pi", "mcts_pi", "policy_pi", "root_pi", "_mcts_pi"):
                            try:
                                _v = entry.get(_k)
                            except Exception:
                                _v = None
                            if _v is not None:
                                pi_src = _v
                                pi_from = f"entry:{_k}"
                                break

                        if pi_src is None:
                            sb0 = entry.get("state_before") or {}
                            if isinstance(sb0, dict):
                                for _k in ("pi", "mcts_pi", "policy_pi", "root_pi", "_mcts_pi"):
                                    try:
                                        _v = sb0.get(_k)
                                    except Exception:
                                        _v = None
                                    if _v is not None:
                                        pi_src = _v
                                        pi_from = f"state_before:{_k}"
                                        break

                        if pi_src is None:
                            sa0 = entry.get("state_after") or {}
                            if isinstance(sa0, dict):
                                for _k in ("pi", "mcts_pi", "policy_pi", "root_pi", "_mcts_pi"):
                                    try:
                                        _v = sa0.get(_k)
                                    except Exception:
                                        _v = None
                                    if _v is not None:
                                        pi_src = _v
                                        pi_from = f"state_after:{_k}"
                                        break

                    if isinstance(pi_src, dict) and isinstance(la_ids, list):
                        try:
                            _tmp = [0.0] * len(la_ids)
                            for _k, _v in pi_src.items():
                                try:
                                    ii = int(_k)
                                    if 0 <= ii < len(_tmp):
                                        _tmp[ii] = float(_v)
                                except Exception:
                                    continue
                            pi_src = _tmp
                        except Exception:
                            pi_src = None

                    if isinstance(pi_src, list) and isinstance(la_ids, list) and len(pi_src) == len(la_ids):
                        try:
                            id_entry_priv["pi"] = [float(x) for x in pi_src]
                            id_entry_priv["pi_source"] = ("raw" if not pi_from else f"raw:{pi_from}")
                        except Exception:
                            id_entry_priv.pop("pi", None)

                    act_vec = None
                    ar_src = entry.get("action_result") or {}
                    act_src = ar_src.get("action") or ar_src.get("macro") if isinstance(ar_src, dict) else None
                    if act_src is not None:
                        if isinstance(act_src, list) and all(isinstance(x, int) for x in act_src):
                            act_vec = act_src
                        else:
                            act_vec = converter.action_to_id(act_src)
                    if act_vec is None:
                        ar_conv = id_entry_priv.get("action_result") or {}
                        _tmp = ar_conv.get("action") or ar_conv.get("macro")
                        if isinstance(_tmp, list) and all(isinstance(x, int) for x in _tmp):
                            act_vec = _tmp
                        elif isinstance(_tmp, int):
                            act_vec = [_tmp]

                    _pi = None
                    if ("pi" not in id_entry_priv) and isinstance(la_ids, list) and la_ids and (act_vec is not None):
                        idx = -1
                        for i, a in enumerate(la_ids):
                            if isinstance(a, list) and a == act_vec:
                                idx = i
                                break
                            if isinstance(a, int) and isinstance(act_vec, list) and len(act_vec) == 1 and a == act_vec[0]:
                                idx = i
                                break
                        if idx >= 0:
                            _pi = [0] * len(la_ids)
                            _pi[idx] = 1
                    if _pi is not None and ("pi" not in id_entry_priv):
                        id_entry_priv["pi"] = _pi
                        id_entry_priv["pi_source"] = "onehot_from_legal_actions"

                    # ★ 追加: legal_actions を “空なら” 同期
                    if (not id_entry_priv.get("legal_actions")) and isinstance(la_ids, list) and la_ids:
                        id_entry_priv["legal_actions"] = la_ids

                except Exception:
                    pass

                # a_idx フォールバック
                try:
                    # まず action_result / act_vec から a_idx を決める
                    if ("a_idx" not in id_entry_priv):
                        ar2 = entry.get("action_result") or {}
                        act2 = ar2.get("action") or ar2.get("macro") if isinstance(ar2, dict) else None
                        if hasattr(converter, "action_to_index") and act2 is not None:
                            _idx = converter.action_to_index(act2)
                            if isinstance(_idx, int) and _idx >= 0:
                                id_entry_priv["a_idx"] = int(_idx)
                        else:
                            _act_vec = act_vec if isinstance(act_vec, list) else None
                            if (_act_vec and len(_act_vec) > 0 and isinstance(_act_vec[0], int) and _act_vec[0] >= 0):
                                id_entry_priv["a_idx"] = int(_act_vec[0])

                    # まだ a_idx が決まっておらず、pi と la_ids があれば pi から復元する
                    if ("a_idx" not in id_entry_priv) and ("pi" in id_entry_priv) and isinstance(la_ids, list) and la_ids:
                        _pi = id_entry_priv.get("pi")
                        if isinstance(_pi, list) and len(_pi) == len(la_ids) and _pi:
                            max_i = max(range(len(_pi)), key=lambda i: _pi[i])
                            chosen = la_ids[max_i]
                            if isinstance(chosen, int):
                                id_entry_priv["a_idx"] = int(chosen)
                            elif isinstance(chosen, list) and chosen and isinstance(chosen[0], int):
                                id_entry_priv["a_idx"] = int(chosen[0])
                except Exception:
                    pass

# 候補打点モデル向けフィールド
                if EMIT_CANDIDATE_FEATURES:
                    try:
                        la_src2 = _pick_legal_actions(entry)  # pyright: ignore[reportUnusedVariable]
                        la_ids2 = converter.convert_legal_actions(la_src2) if hasattr(converter, "convert_legal_actions") else []  # pyright: ignore[reportUnusedVariable]
                        # ★ 修正: legal_actions は 5ints のまま保持（32d化は encode_action_from_vec_32d 側で実施）
                        # 新: 正式フィールド（32d 統一）
                        id_entry_priv["action_candidates_vec"] = _embed_legal_actions_32d(la_ids2)
                        id_entry_priv["action_vec_dim"] = ACTION_VEC_DIM
                        id_entry_priv["legal_actions_vec_dim"] = id_entry_priv["action_vec_dim"]

                        # --- DEBUG: cand_vec の健全性を確認（スパム防止のため間引き） ---
                        try:
                            _dbg_cand = bool(globals().get("DEBUG_PHASEDQ_CAND", True))
                            _dbg_every = int(globals().get("DEBUG_PHASEDQ_CAND_EVERY", 50))
                            if _dbg_cand and (t < 5 or (t % _dbg_every) == 0):
                                _cands = id_entry_priv.get("action_candidates_vec")
                                _n_la = len(la_ids2) if isinstance(la_ids2, list) else -1
                                _n_cv = len(_cands) if isinstance(_cands, list) else -1
                                _bad_dim = 0
                                _all_zero = 0
                                _all_m1 = 0
                                if isinstance(_cands, list):
                                    for _v in _cands:
                                        if not (isinstance(_v, list) and len(_v) == ACTION_VEC_DIM):
                                            _bad_dim += 1
                                            continue
                                        try:
                                            if all(float(x) == 0.0 for x in _v):
                                                _all_zero += 1
                                            if all(float(x) == -1.0 for x in _v):
                                                _all_m1 += 1
                                        except Exception:
                                            pass
                                _head = None
                                try:
                                    if isinstance(la_ids2, list) and la_ids2:
                                        _head = la_ids2[0]
                                except Exception:
                                    _head = None
                                print(
                                    f"[CANDDBG/priv] t={t} la_src2_len={len(la_src2) if isinstance(la_src2, list) else 'NA'} "
                                    f"la_ids2_len={_n_la} cand_vec_len={_n_cv} bad_dim={_bad_dim} all_zero={_all_zero} all_m1={_all_m1} "
                                    f"la0_type={type(_head).__name__ if _head is not None else 'None'} la0_len={(len(_head) if isinstance(_head, list) else 'NA')}"
                                )
                        except Exception:
                            pass

                        if EMIT_OBS_VEC_FOR_CANDIDATES:
                            sb_src_conv = id_entry_priv.get("state_before") or {}  # pyright: ignore[reportUnusedVariable]
                            id_entry_priv["obs_vec"] = build_obs_partial_vec(sb_src_conv)

                            # ★ 追加: AlphaZero MCTS ポリシーから π を再計算して付与（自己対戦モードのみ）
                            if SELFPLAY_ALPHAZERO_MODE and USE_MCTS_POLICY and (not str(id_entry_priv.get("pi_source", "")).startswith("raw")):
                                try:
                                    _dbg_pi = bool(globals().get("DEBUG_RECOMPUTE_PI", True))
                                    mcts_policy = local_policy_p1 or local_policy_p2
                                    obs_vec_for_mcts = id_entry_priv.get("obs_vec")

                                    # unwrap: OnlineMixedPolicy / wrappers → 内側のポリシーへ
                                    if mcts_policy is not None and hasattr(mcts_policy, "main_policy"):
                                        try:
                                            mcts_policy = getattr(mcts_policy, "main_policy")
                                        except Exception:
                                            pass
                                    if mcts_policy is not None and hasattr(mcts_policy, "policy"):
                                        try:
                                            mcts_policy = getattr(mcts_policy, "policy")
                                        except Exception:
                                            pass

                                    # MCTS に渡す候補は 32d を優先（無ければ従来通り la_ids2）
                                    la_for_mcts = None
                                    try:
                                        la_for_mcts = id_entry_priv.get("action_candidates_vec")
                                    except Exception:
                                        la_for_mcts = None
                                    if not (
                                        isinstance(la_for_mcts, list)
                                        and la_for_mcts
                                        and all(isinstance(_v, list) and len(_v) == ACTION_VEC_DIM for _v in la_for_mcts)
                                    ):
                                        la_for_mcts = la_ids2

                                    if _dbg_pi and (t < 5):
                                        print(
                                            f"[PIDBG/priv] t={t} pol_class={(getattr(mcts_policy, '__class__', type(mcts_policy)).__name__ if mcts_policy is not None else 'None')} "
                                            f"has_select_action={hasattr(mcts_policy, 'select_action')} obs_len={(len(obs_vec_for_mcts) if isinstance(obs_vec_for_mcts, list) else 'NA')} "
                                            f"la_ids2_len={(len(la_ids2) if isinstance(la_ids2, list) else 'NA')}"
                                        )

                                    if (
                                        mcts_policy is not None
                                        and isinstance(obs_vec_for_mcts, list)
                                        and isinstance(la_for_mcts, list)
                                        and la_for_mcts
                                        and hasattr(mcts_policy, "select_action")
                                    ):
                                        a_vec_mcts, pi_mcts = mcts_policy.select_action(obs_vec_for_mcts, la_for_mcts)

                                        if _dbg_pi and (t < 5):
                                            print(
                                                f"[PIDBG/priv] t={t} select_action_ret a_vec_type={type(a_vec_mcts).__name__} "
                                                f"pi_type={type(pi_mcts).__name__} pi_len={(len(pi_mcts) if isinstance(pi_mcts, list) else 'NA')} "
                                                f"expect_len={len(la_for_mcts)}"
                                            )

                                        if DEBUG_MODEL_MCTS and (t % DEBUG_MODEL_MCTS_EVERY) == 1:
                                            _trace_policy_step(
                                                tag=f"recompute_pi_priv t={t}",
                                                pol=mcts_policy,
                                                obs_vec=obs_vec_for_mcts,
                                                la_ids=la_for_mcts,
                                                chosen_vec=a_vec_mcts,
                                                pi=pi_mcts,
                                            )
                                        if isinstance(pi_mcts, list) and len(pi_mcts) == len(la_for_mcts):
                                            id_entry_priv["pi"] = [float(x) for x in pi_mcts]
                                            id_entry_priv["pi_source"] = "recomputed_mcts"
                                        elif _dbg_pi and (t < 5):
                                            print(f"[PIDBG/priv] t={t} pi_mcts length mismatch -> skip attach")
                                    elif _dbg_pi and (t < 5):
                                        print(f"[PIDBG/priv] t={t} recompute_pi skipped (missing method or bad inputs)")
                                except Exception as _e:
                                    try:
                                        if bool(globals().get("DEBUG_RECOMPUTE_PI", True)) and (t < 5):
                                            print(f"[PIDBG/priv] t={t} recompute_pi failed: {_e!r}")
                                    except Exception:
                                        pass

                        # ★ 追加: pi（候補上の one-hot 分布）を出力
                        if "pi" not in id_entry_priv:
                            aidx = id_entry_priv.get("a_idx")  # pyright: ignore[reportUnusedVariable]
                            if isinstance(aidx, int) and isinstance(la_ids2, list) and la_ids2:
                                try:
                                    n = len(la_ids2)  # pyright: ignore[reportUnusedVariable]
                                    pi = [0.0] * n  # pyright: ignore[reportUnusedVariable]
                                    j = None  # pyright: ignore[reportUnusedVariable]
                                    for i, _la in enumerate(la_ids2):
                                        if isinstance(_la, int) and _la == aidx:
                                            j = i
                                            break
                                        if isinstance(_la, list) and _la and isinstance(_la[0], int) and _la[0] == aidx:
                                            j = i
                                            break
                                    if j is not None:
                                        pi[j] = 1.0
                                        id_entry_priv["pi"] = pi
                                        id_entry_priv["pi_source"] = "onehot_from_a_idx"
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # ★ 空 legal_actions は落とす（任意）
                if DROP_EMPTY_LEGAL_ACTIONS and isinstance(id_entry_priv.get("legal_actions"), list) and not id_entry_priv["legal_actions"]:
                    id_entry_priv.pop("legal_actions", None)

                # ★★★★★ ここが重要：完全情報ベクトルを必ず付与する
                try:
                    attach_fullvec_fields(id_entry_priv)  # obs_full_vec / obs_full_dim / obs_full_version
                    if "obs_full_vec" not in id_entry_priv:
                        print("[FULLVEC] warn: obs_full_vec not set (game_id=", id_entry_priv.get("state_before",{}).get("game_id"), ")")
                except Exception as e:
                    print("[FULLVEC] skip due to error:", e)

                # ★ カウンタ: “最終的に pi が無かった行”だけ数える
                if "pi" not in id_entry_priv:
                    try:
                        pi_missing_private += 1
                    except Exception:
                        pass

                # ← 集中ライタへ渡す private 用バッファに積む
                try:
                    k = id_entry_priv.get("pi_source")
                    if not k:
                        k = "(none)"
                    pi_source_hist_private[k] = int(pi_source_hist_private.get(k, 0)) + 1
                except Exception:
                    pass

                priv_id_lines.append(json.dumps(id_entry_priv, ensure_ascii=False))

        # ★ 追加: 「obs_vec のない試合」はログごと破棄
        #   - 条件1: 行動ステップが1つ以上ある
        #   - 条件2: obs_vec が非空のステップ数 < 行動ステップ数
        #            （= 1つでも obs_vec の付いていない行動がある / 全く付いていない）
        if (action_records > 0) and (actions_with_obs < action_records):
            def _has_good_obs_vec(_obj: dict) -> bool:
                if "a_idx" not in _obj:
                    return True
                _v = _obj.get("obs_vec")
                return isinstance(_v, list) and _v and all(isinstance(x, (int, float)) for x in _v)

            _new_id_lines = []
            for _line in id_lines:
                try:
                    _obj = json.loads(_line)
                except Exception:
                    _new_id_lines.append(_line)
                    continue
                if _has_good_obs_vec(_obj):
                    _new_id_lines.append(json.dumps(_obj, ensure_ascii=False))
            id_lines = _new_id_lines

            _new_priv_lines = []
            for _line in priv_id_lines:
                try:
                    _obj = json.loads(_line)
                except Exception:
                    _new_priv_lines.append(_line)
                    continue
                if _has_good_obs_vec(_obj):
                    _new_priv_lines.append(json.dumps(_obj, ensure_ascii=False))
            priv_id_lines = _new_priv_lines

        # ここで“まとめて”判定（writer へ送るのは winner 決定後に変更）
        did_log_this_match = (len(raw_lines) > 0) or (len(id_lines) > 0) or (len(priv_id_lines) > 0)

        # --- 集約済みなので、保持フラグに応じて per-match ファイルを削除 ---
        try:
            if (not KEEP_MATCH_ML_FILE) and os.path.exists(ml_log_file):
                os.remove(ml_log_file)
            if (not KEEP_MATCH_LOG_FILE) and os.path.exists(match_log_file):
                os.remove(match_log_file)
        except Exception:
            pass
        # ★ 追加: 勝者推定してログ＆親へ集計
        # セーフガード: もしこの時点までに winner が定義されていない経路があれば unknown で初期化
        if 'winner' not in locals():
            winner = "unknown"

        try:
            print(f"[RESULT] match={match_num} game_id={game_id} winner={winner}")
            if mcc_agg is not None:
                if winner == "p1":
                    mcc_agg["wins_p1"] = int(mcc_agg.get("wins_p1", 0)) + 1
                elif winner == "p2":
                    mcc_agg["wins_p2"] = int(mcc_agg.get("wins_p2", 0)) + 1
                elif winner == "draw":
                    mcc_agg["wins_draw"] = int(mcc_agg.get("wins_draw", 0)) + 1
                else:
                    mcc_agg["wins_unknown"] = int(mcc_agg.get("wins_unknown", 0)) + 1
        except Exception as _e:
            print(f"[worker] winner print failed: {_e}")

        # ★ 重要: 「実際にログを出力した試合のみ」num_matches を加算（デッキアウトは earlier-continue 済み）
        if mcc_agg is not None and did_log_this_match:
            mcc_agg["num_matches"] = int(mcc_agg.get("num_matches", 0)) + 1

        # ★ 追加: game_summary レコード（学習データには使わない簡約版の勝敗ログ）
        #   - ai_vs_ai_match_all.jsonl にのみ 1ゲーム1行で追加される
        #   - record_type="game_summary" を付けておくことで後段の解析側で簡単にフィルタ可能
        try:
            if did_log_this_match:
                end_reason_for_game = None

                # raw_lines からこの game_id の end_reason を走査して最後のものを採用
                for line in raw_lines:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if str(obj.get("game_id")) != str(game_id):
                        continue
                    r = obj.get("end_reason") or obj.get("_end_reason")
                    if r:
                        end_reason_for_game = str(r)

                if not end_reason_for_game:
                    end_reason_for_game = "UNKNOWN"

                summary_entry = {
                    "record_type": "game_summary",
                    "game_id": game_id,
                    "match": match_num,
                    "winner": winner,
                    "end_reason": end_reason_for_game,
                    "is_deckout_game": bool(_is_deckout_game),
                    "action_records": int(action_records),
                    "actions_with_obs": int(actions_with_obs),
                    "pi_missing_public": int(pi_missing_public),
                    "pi_missing_private": int(pi_missing_private),
                    "mcc_calls": (int(mcc_calls_this_game) if isinstance(mcc_calls_this_game, int) else 0),
                }
                # ★ 注意: summary は raw_lines にだけ追加する（ids / private_ids には追加しない）
                raw_lines.append(json.dumps(summary_entry, ensure_ascii=False))
        except Exception as _e:
            print(f"[worker] game_summary build failed: {_e}")

        # ★ 追加: 勝敗ラベル z を各行に付与する（me 視点の value）
        try:
            # 終局理由に応じた基準スケール（勝利時の tier）
            end_reason_str = str(end_reason or "UNKNOWN").upper()
            base_z = Z_TABLE.get(end_reason_str, 0.0)

            def _calc_z_for_entry(obj):
                # --- 状態情報の取り出し ---
                sb = obj.get("state_before") or {}
                me = sb.get("me") or {}
                opp = sb.get("opp") or {}

                me_player = me.get("player")  # "p1" / "p2" を想定

                # --- 終局結果に応じた終局 z ---
                if winner in ("p1", "p2") and me_player in ("p1", "p2"):
                    if winner == me_player:
                        # 自分が勝者 → 勝ち方（end_reason）に応じたプラス値
                        terminal_z = base_z
                    else:
                        # 自分が敗者 → 終局理由に関係なく一律 LOSS_Z
                        terminal_z = LOSS_Z
                elif winner == "draw":
                    terminal_z = DRAW_Z
                else:
                    terminal_z = DRAW_Z

                # --- 途中盤面からのシェーピング（raw 値） ---
                z_progress_raw = 0.0

                # サイド差: 相手よりサイド残りが少ないほどプラス
                my_prize = me.get("prize_count")
                opp_prize = opp.get("prize_count")
                if isinstance(my_prize, (int, float)) and isinstance(opp_prize, (int, float)):
                    # prize_count は「残り枚数」なので、
                    # (相手の残り − 自分の残り) がプラスだと有利とみなす
                    prize_diff = (opp_prize - my_prize) / 6.0  # おおよそ [-1, +1] の範囲に収まる想定
                    z_progress_raw += INTERMEDIATE_PRIZE_WEIGHT * prize_diff

                # デッキ残枚数差: 自分の山札が多いほどプラス、少ないほどマイナス
                my_deck_count = me.get("deck_count")
                opp_deck_count = opp.get("deck_count")
                if isinstance(my_deck_count, (int, float)) and isinstance(opp_deck_count, (int, float)):
                    deck_diff = (float(my_deck_count) - float(opp_deck_count)) / 60.0
                    if deck_diff > 1.0:
                        deck_diff = 1.0
                    elif deck_diff < -1.0:
                        deck_diff = -1.0
                    z_progress_raw += INTERMEDIATE_DECK_COUNT_WEIGHT * deck_diff

                # ターン数ペナルティ: 長期戦になりすぎると少しずつマイナス
                turn_index = obj.get("turn_index")
                if turn_index is None:
                    turn_index = sb.get("turn")
                if isinstance(turn_index, int) and turn_index >= INTERMEDIATE_TURN_PENALTY_START:
                    over = turn_index - INTERMEDIATE_TURN_PENALTY_START + 1
                    z_progress_raw -= min(
                        INTERMEDIATE_TURN_PENALTY_PER_TURN * over,
                        INTERMEDIATE_MAX_TURN_PENALTY,
                    )

                # --- シェーピング幅のクリップ ---
                if z_progress_raw > Z_PROGRESS_MAX:
                    z_progress = Z_PROGRESS_MAX
                elif z_progress_raw < -Z_PROGRESS_MAX:
                    z_progress = -Z_PROGRESS_MAX
                else:
                    z_progress = z_progress_raw

                # --- 合成＆クリップ ---
                z_total = terminal_z + z_progress
                if z_total > 1.0:
                    z_total = 1.0
                elif z_total < -1.0:
                    z_total = -1.0

                return float(z_total)

            if did_log_this_match:
                new_id_lines = []
                for line in id_lines:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        new_id_lines.append(line)
                        continue
                    obj["z"] = _calc_z_for_entry(obj)
                    new_id_lines.append(json.dumps(obj, ensure_ascii=False))
                id_lines = new_id_lines

                new_priv_lines = []
                for line in priv_id_lines:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        new_priv_lines.append(line)
                        continue
                    obj["z"] = _calc_z_for_entry(obj)
                    new_priv_lines.append(json.dumps(obj, ensure_ascii=False))
                priv_id_lines = new_priv_lines

                if end_reason_str == "DECK_OUT":
                    _tail_n = int(globals().get("DECKOUT_TAIL_STEPS", 30))

                    def _mark_deckout_tail(_lines):
                        _objs = []
                        for _line in _lines:
                            try:
                                _o = json.loads(_line)
                            except Exception:
                                _objs.append((_line, None))
                                continue
                            _objs.append((_line, _o))

                        _idxs = []
                        for _i, (_ln, _o) in enumerate(_objs):
                            if isinstance(_o, dict) and ("a_idx" in _o):
                                _idxs.append(_i)

                        _tail = _idxs[-_tail_n:] if _tail_n > 0 else []

                        for _i, (_ln, _o) in enumerate(_objs):
                            if not isinstance(_o, dict):
                                continue
                            if "a_idx" in _o:
                                _o["is_deckout_game"] = True
                            if _i in _tail:
                                _k = _tail.index(_i)
                                _o["near_deckout"] = True
                                _o["near_deckout_k"] = int(len(_tail) - 1 - _k)

                        _out = []
                        for _ln, _o in _objs:
                            if isinstance(_o, dict):
                                _out.append(json.dumps(_o, ensure_ascii=False))
                            else:
                                _out.append(_ln)
                        return _out

                    id_lines = _mark_deckout_tail(id_lines)
                    priv_id_lines = _mark_deckout_tail(priv_id_lines)
        except Exception as _e:
            print(f"[worker] z label attach failed: {_e}")

        # ここで“まとめて” writer へ送る（z 付与後）
        if did_log_this_match:
            queue.put(("batch", game_id, raw_lines, id_lines, priv_id_lines))
        else:
            queue.put(("batch", game_id, [], [], []))

        # 追加: “ログに残した試合（=出力済み）”と“試行した試合”の加算＋進捗表示
        try:
            if mcc_agg is not None:
                mcc_agg["attempted_matches"] = int(mcc_agg.get("attempted_matches", 0)) + 1
                if did_log_this_match:
                    mcc_agg["logged_matches"] = int(mcc_agg.get("logged_matches", 0)) + 1
            if did_log_this_match:
                logged_count += 1
            total_logged_all = int(mcc_agg.get("logged_matches", 0)) if mcc_agg is not None else logged_count
            print(f"[COUNT/worker {os.getpid()}] logged={logged_count} skipped_deckout={skipped_deckout_count} total_tried={match_num} | total_logged_all={total_logged_all}")
            _dump_policy_stats_worker("P1", local_policy_p1)
            _dump_policy_stats_worker("P2", local_policy_p2)
        except Exception:
            pass

        # ワーカー側の一時ログは即削除（容量対策）
        # --- Windowsのロック対策：PermissionErrorをリトライして安全に削除 ---
        def _safe_remove(path, retries=5, delay=0.1):
            import gc
            for _ in range(retries):
                try:
                    os.remove(path)
                    return True
                except PermissionError:
                    gc.collect()
                    time.sleep(delay)
                except FileNotFoundError:
                    return True
            return False

        # ★ 修正: 常に per-match ファイルを削除（最後の1戦も残さない）
        if os.path.exists(match_log_file):
            _safe_remove(match_log_file)
        if os.path.exists(ml_log_file):
            _safe_remove(ml_log_file)

        if MATCH_SLEEP_SEC > 0.0:
            time.sleep(MATCH_SLEEP_SEC)

    # 追加: ワーカー終了時に自分の MCC 統計を親に合算（試合数はここでは合算しない）
    try:
        dbg = mcc_debug_snapshot()
        if mcc_agg is not None:
            mcc_agg["total_calls"] = mcc_agg.get("total_calls", 0) + int(dbg.get("total_calls", 0))
            # mcc_agg["num_matches"] は勝者集計側で「ログを書いた試合のみ」加算する
            mcc_agg["pi_missing_public"] = mcc_agg.get("pi_missing_public", 0) + int(pi_missing_public)
            mcc_agg["pi_missing_private"] = mcc_agg.get("pi_missing_private", 0) + int(pi_missing_private)
    except Exception:

        pass
    finally:
        reset_mcc_debug()

def run_random_matches_multiprocess(to_run: int):
    """
    ランダム対戦のときのみ採用する並列実行エントリ。
    同一の RAW_JSONL_PATH / IDS_JSONL_PATH に集中ライターで追記する。
    """
    from multiprocessing import Manager

    n_auto = max(1, min((cpu_count() or 2) - 1, to_run))
    if USE_MCC:
        # MCCあり（CPUオンリーMCC想定）→ 既定は4ワーカー
        n_proc = min(to_run, CPU_WORKERS_MCC) if CPU_WORKERS_MCC > 0 else n_auto
    else:
        # MCCなしのランダム回し → 既定は自動（≒9ワーカー）
        n_proc = min(to_run, CPU_WORKERS_RANDOM) if CPU_WORKERS_RANDOM > 0 else n_auto

    base = to_run // n_proc
    rem  = to_run % n_proc
    chunks = [base + (1 if i < rem else 0) for i in range(n_proc)]
    chunks = [c for c in chunks if c > 0]

    print(f"[multiprocess] processes={len(chunks)}  plan={chunks}")

    q = Queue(maxsize=len(chunks) * 16)
    stop = Event()

    # 追加: 子→親で MCC 統計を集約する共有辞書
    mgr = Manager()
    mcc_agg = mgr.dict({
        "total_calls": 0,
        "num_matches": 0,          # ※ これはMCCの内部統計としてそのまま残す
        "wins_p1": 0,
        "wins_p2": 0,
        "wins_draw": 0,
        "wins_unknown": 0,
        # 追加: 実際に“ログとして出力した”試合数と、スキップ件数
        "logged_matches": 0,       # ← デッキアウトを除いた試合数（あなたが知りたい数）
        "skipped_deckout": 0,      # ← デッキアウトで除外した試合数
        "attempted_matches": 0,    # ← 試行した総試合数（= logged + skipped）
        # 追加: π が元ログに無かった/長さ不一致だった決定ステップ数
        "pi_missing_public": 0,
        "pi_missing_private": 0,
    })

    writer = Process(target=writer_loop, args=(q, stop), daemon=True)
    writer.start()

    workers = []
    for c in chunks:
        # 追加: mcc_agg をワーカーに渡す
        p = Process(target=play_continuous_matches_worker, args=(c, q, mcc_agg, None, _run_game_id), daemon=False)
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # ライターに終端を通知して終了待ち
    q.put(None)
    stop.set()
    writer.join()

    # 追加: 親が合算結果を返す
    res = dict(mcc_agg)
    print("[DONE]", res)
    return res

def run_model_matches_multiprocess(to_run: int):
    """
    モデル対戦を複数プロセスで回す。
    - gpu_q_server が無い/起動失敗した場合は CPU フォールバック（gpu_req_q=None のまま）で継続
    - GPU サーバが起動できた場合のみ、ワーカーに gpu_req_q を渡して GPU 経由で evaluate_q
    """
    from multiprocessing import Process, Queue, Event, Manager, cpu_count
    import os

    if to_run <= 0:
        print("[multiprocess-model] to_run=0 → 何もしません")
        return {}

    # --- 並列数の計画（過剰並列を避けるためクリップ） ---
    n_auto = max(1, min((cpu_count() or 2) - 1, to_run))
    n_proc = min(to_run, CPU_WORKERS_MCC) if CPU_WORKERS_MCC > 0 else n_auto
    n_proc = max(1, min(n_proc, n_auto, to_run))

    base = to_run // n_proc
    rem  = to_run % n_proc
    chunks = [base + (1 if i < rem else 0) for i in range(n_proc)]
    chunks = [c for c in chunks if c > 0]
    print(f"[multiprocess-model] processes={len(chunks)}  plan={chunks}")

    # --- 共有ライタ & 集約用 dict ---
    q = Queue(maxsize=len(chunks) * 16)
    stop = Event()
    mgr = Manager()
    mcc_agg = mgr.dict({
        "total_calls": 0,
        "num_matches": 0,          # ※ これはMCCの内部統計としてそのまま残す
        "wins_p1": 0,
        "wins_p2": 0,
        "wins_draw": 0,
        "wins_unknown": 0,
        # 追加: 実際に“ログとして出力した”試合数と、スキップ件数
        "logged_matches": 0,       # ← デッキアウトを除いた試合数（あなたが知りたい数）
        "skipped_deckout": 0,      # ← デッキアウトで除外した試合数
        "attempted_matches": 0,    # ← 試行した総試合数（= logged + skipped）
        # 追加: π が元ログに無かった/長さ不一致だった決定ステップ数
        "pi_missing_public": 0,
        "pi_missing_private": 0,
    })

    # --- GPU サーバ可否判定（インポートが通らなければ即フォールバック） ---
    try:
        from gpu_q_server import GPUQServer  # サーバは別プロセス
        gpu_available = True
    except Exception as e:
        print("[WARN] gpu_q_server が見つかりません / 読み込み失敗:", e)
        print("→ GPU サーバ無しで CPU 実行にフォールバックします。")
        GPUQServer = None  # 型だけ残す
        gpu_available = False

    use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[GPUQServer] device candidate = {use_device}")

    # --- GPU サーバ起動の準備（使えるなら） ---
    gpu_req_q = None
    gpu_stop  = None
    gpu_server = None

    # どちらの方策をモデル読み込みに使うか（P1優先）
    model_dir = MODEL_DIR_P1 if P1_POLICY.lower() == "model" else MODEL_DIR_P2

    # learnable を最優先で渡す（無ければ従来の weights）
    learnable_path = os.path.join(model_dir, "learnable.d3")
    weights_path   = os.path.join(model_dir, "model_final.d3")
    model_path_for_gpu = learnable_path if os.path.exists(learnable_path) else weights_path

    if gpu_available:
        if not os.path.exists(model_path_for_gpu):
            print(f"[WARN] モデルファイルが見つかりません: {model_path_for_gpu}")
            print("→ GPU サーバは起動せず、CPU 経路で継続します。")
            gpu_available = False

    # --- GPU サーバのブリッジ設定 & 起動（可能な場合のみ） ---
    if gpu_available:
        try:
            import gpu_q_server as _gqs_mod
            # 候補→32次元への唯一エンコーダを GPU サーバにも渡す
            setattr(_gqs_mod, "encode_action_from_vec_32d", encode_action_from_vec_32d)
            enc19 = globals().get("encode_action_from_vec_19d")
            if enc19 is not None:
                setattr(_gqs_mod, "encode_action_from_vec_19d", enc19)

            setattr(_gqs_mod, "ACTION_SCHEMAS_INFER", ACTION_SCHEMAS)
            setattr(_gqs_mod, "TYPE_SCHEMAS_INFER", TYPE_SCHEMAS)
            setattr(_gqs_mod, "ACTION_TYPES_LIST", _ACTION_TYPES)  # K の源泉
            # 仕様識別子（32d 固定）
            setattr(_gqs_mod, "ENCODER_SPEC", f"32d-fixed")
            print(f"[GPUQServer] encoder={getattr(_gqs_mod,'ENCODER_SPEC','(unspecified)')} "
                  f"K={len(_ACTION_TYPES)} MAX_ARGS={MAX_ARGS}")
        except Exception as e:
            print("[WARN] GPU ブリッジ設定に失敗しました（CPU フォールバックします）:", e)
            gpu_available = False

    if gpu_available:
        try:
            print(f"[GPUQServer] starting. device={use_device}  model={model_path_for_gpu}")
            gpu_req_q = Queue(maxsize=4096)
            gpu_stop  = Event()
            gpu_server = GPUQServer(
                req_q=gpu_req_q,
                stop_ev=gpu_stop,
                model_path=model_path_for_gpu,
                device=use_device,
                max_batch=256,
                max_wait_ms=2
            )
            gpu_server.start()
        except Exception as e:
            print("[WARN] GPUQServer 起動に失敗（CPU フォールバックします）:", e)
            gpu_req_q = None
            gpu_server = None

    # --- scaler/obs_mask の形状チェック（学習成果物との齟齬を早期検知） ---
    try:
        expected_d = int(globals().get("state_dim", 0) or 0)
    except Exception:
        expected_d = 0
    if expected_d <= 0:
        expected_d = int(os.getenv("OBS_DIM_HINT", "1"))
    _ensure_scaler_mask(os.path.join(model_dir, "scaler.npz"), expected_d, model_dir)

    # --- 共有ライタ起動 ---
    writer = Process(target=writer_loop, args=(q, stop), daemon=True)
    writer.start()

    # --- ワーカー起動（gpu_req_q が None なら CPU 経路で evaluate_q される） ---
    workers = []
    for c in chunks:
        p = Process(
            target=play_continuous_matches_worker,
            args=(c, q, mcc_agg, gpu_req_q),
            daemon=False
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # --- writer 終了 ---
    q.put(None)
    stop.set()
    writer.join()

    # --- GPU サーバ停止（起動していれば） ---
    if gpu_server is not None and gpu_stop is not None:
        gpu_stop.set()
        gpu_server.join()
        print("[GPUQServer] stopped.")

    res = dict(mcc_agg)
    print("[DONE]", res)
    return res

def has_minus_one(obj, parent_key=None, path=()):
    # dict は子を走査
    if isinstance(obj, dict):
        for k, v in obj.items():
            if has_minus_one(v, k, path + (k,)):
                return True
        return False

    # list の扱い
    elif isinstance(obj, list):
        # A) 単一アクションベクトル: 末尾サフィックスの -1 を許容
        if parent_key in ("action", "action_vec", "macro"):
            # 最初に -1 が現れた位置 j 以降が全て -1 なら OK（途中に非 -1 があれば NG）
            j = next((i for i, x in enumerate(obj) if x == -1), None)
            if j is not None and any((x != -1) for x in obj[j:]):
                return True
            # ネストがあれば通常通り再帰
            for i, x in enumerate(obj):
                if isinstance(x, (list, dict)) and has_minus_one(x, parent_key, path + (i,)):
                    return True
            return False

        # B) 候補群: legal_actions は各要素配列で A の規則を適用
        if parent_key == "legal_actions":
            for i, a in enumerate(obj):
                if isinstance(a, list):
                    j = next((k for k, y in enumerate(a) if y == -1), None)
                    if j is not None and any((y != -1) for y in a[j:]):
                        return True
                elif has_minus_one(a, "action", path + (i,)):
                    return True
            return False

        # 通常の再帰
        for i, x in enumerate(obj):
            if has_minus_one(x, parent_key, path + (i,)):
                return True
        return False

    # スカラ値
    else:
        if obj == -1:
            # reward / meta 配下は -1 を許容
            if len(path) >= 1 and path[0] in ("reward", "meta"):
                return False
            return True
        return False

def assert_no_minus_one_in_entries(entries, context: str = ""):
    """
    entries に許容外の -1 が含まれていないかを検査する。
    has_minus_one の許容パス定義に従ってチェックし、検出時は index とスニペットを含む
    エラーメッセージを出力するために例外を送出する。
    """
    for idx, e in enumerate(entries):
        if has_minus_one(e):
            try:
                snippet = json.dumps(e, ensure_ascii=False)[:400]
            except Exception:
                snippet = str(e)[:400]
            raise RuntimeError(f"[FATAL] JSON出力を中止: エントリ[{idx}]に -1 を検出 context={context} snippet={snippet}")

def _is_decision_entry(e: dict) -> bool:
    if not isinstance(e, dict):
        return False
    ar = e.get("action_result") or {}
    act = ar.get("action") if isinstance(ar, dict) else None
    if act is None and isinstance(ar, dict):
        act = ar.get("macro")
    return (
        act is not None
        and isinstance(e.get("state_before"), dict)
        and isinstance(e.get("state_after"), dict)
    )


# ▼ これをヘルパー関数群の近く（CardNameToIdConverter の上/下どちらでもOK）に追加
def _pick_legal_actions(entry: dict):
    """
    候補手を取り出す優先順:
      1) top-level entry['legal_actions']
      2) action_result['legal_actions']
      3) action_result.substeps[*].legal_actions（後ろから）
      4) state_before / state_after の top-level 'legal_actions'
      5) state_before/after の me / opp の 'legal_actions'
      見つからなければ []
    """
    if not isinstance(entry, dict):
        return []

    # 1) top-level
    la = entry.get("legal_actions")
    if isinstance(la, list) and la:
        return la

    # 2) action_result 直下
    ar = entry.get("action_result") or {}
    if isinstance(ar, dict):
        la2 = ar.get("legal_actions")
        if isinstance(la2, list) and la2:
            return la2

        # 3) substeps を後ろから
        subs = ar.get("substeps")
        if isinstance(subs, list) and subs:
            for st in reversed(subs):
                if isinstance(st, dict):
                    la3 = st.get("legal_actions")
                    if isinstance(la3, list) and la3:
                        return la3

    # 4) state_before / state_after 直下
    for k in ("state_before", "state_after"):
        st = entry.get(k) or {}
        if isinstance(st, dict):
            la4 = st.get("legal_actions")
            if isinstance(la4, list) and la4:
                return la4

    # 5) state_* の me / opp 内
    for k in ("state_before", "state_after"):
        st = entry.get(k) or {}
        if isinstance(st, dict):
            for side in ("me", "opp"):
                s = st.get(side) or {}
                if isinstance(s, dict):
                    la5 = s.get("legal_actions")
                    if isinstance(la5, list) and la5:
                        return la5

    return []

class CardNameToIdConverter:
    # ----------------------------------------------
    def _card_token_to_id(self, token):
        """
        token が str ならそのまま、[name, n] なら token[0] を ID 化
        """
        if isinstance(token, list):
            token = token[0]
        return self.convert_card_name_to_id(token)
    # ----------------------------------------------

    def __init__(self):
        # カード名からIDへのマッピングを作成
        self.name_to_id = {}
        for card in Cards:
            # カードIDと日本語名を取得
            card_id = card.value[0]
            jp_name = card.value[1]
            
            # 日本語名を追加
            self.name_to_id[jp_name] = card_id
            
            # 英語名がある場合は英語名も追加
            if len(card.value) > 2 and card.value[2]:
                en_name = card.value[2]
                self.name_to_id[en_name] = card_id
        
        # ワザ名からIDへのマッピングを作成
        self.attack_name_to_id = {}
        for attack in Attacks:
            # ワザIDと日本語名を取得
            attack_id = attack.value[0]
            jp_name = attack.value[1]
            
            # 日本語名を追加
            self.attack_name_to_id[jp_name] = attack_id
            
            # 英語名がある場合は英語名も追加
            if len(attack.value) > 2 and attack.value[2]:
                en_name = attack.value[2]
                self.attack_name_to_id[en_name] = attack_id
            
            # Enum 名も許容
            self.attack_name_to_id[attack.name] = attack_id
            self.attack_name_to_id[attack.name.lower()] = attack_id


        # 特性名からIDへのマッピングを作成
        from pokepocketsim.cards_enum import Ability
        self.ability_name_to_id = {}
        for ability in Ability:
            ability_id = ability.value[0]
            jp_name = ability.value[1]
            self.ability_name_to_id[jp_name] = ability_id
            if len(ability.value) > 2 and ability.value[2]:
                en_name = ability.value[2]
                self.ability_name_to_id[en_name] = ability_id
        
        # アクションタイプID
        self.ACTION_TYPE = {
            "end_turn":       ActionType.END_TURN.value,
            "play_bench":     ActionType.PLAY_BENCH.value,
            "attach_energy":  ActionType.ATTACH_ENERGY.value,
            "use_item":       ActionType.USE_ITEM.value,
            "use_supporter":  ActionType.USE_SUPPORTER.value,
            "retreat":        ActionType.RETREAT.value,
            "attack":         ActionType.ATTACK.value,
            "evolve":         ActionType.EVOLVE.value,
            "ability":        ActionType.USE_ABILITY.value,
            "stadium":        ActionType.PLAY_STADIUM.value,
            "stadium_effect": ActionType.STADIUM_EFFECT.value,
            "attach_tool":    ActionType.ATTACH_TOOL.value,
        }
        
        # 場所のID
        self.PLACE_ID = {
            "バトル場": 0,
            "ベンチ1": 1,
            "ベンチ2": 2,
            "ベンチ3": 3,
            "ベンチ4": 4,
            "ベンチ5": 5,
        }
        
        # タイプのID
        self.TYPE_ID = {
            "grass": 0,      # くさ
            "fire": 1,       # ほのお
            "water": 2,      # みず
            "lightning": 3,  # でんき
            "psychic": 4,    # エスパー
            "fighting": 5,   # かくとう
            "dark": 6,       # あく
            "metal": 7,      # はがね
            "colorless": 8,  # 無色
            "dragon": 9,     # ドラゴン
        }

        # 日本語/別表記も許容（例: でんき, 炎, 水 など）
        self.TYPE_ID.update({
            "くさ": 0, "草": 0,
            "ほのお": 1, "炎": 1,
            "みず": 2, "水": 2,
            "でんき": 3, "電気": 3,
            "エスパー": 4,
            "かくとう": 5, "格闘": 5,
            "あく": 6, "悪": 6,
            "はがね": 7, "鋼": 7,
            "無色": 8,
            "ドラゴン": 9,
        })

    def convert_card_name_to_id(self, card_name) -> int:
        """カード名／ID／数字文字列を安全にIDへ"""
        if isinstance(card_name, int):
            return card_name
        if isinstance(card_name, list):
            card_name = card_name[0]
        # ★ 数値文字列をそのままIDとして受け入れ
        if isinstance(card_name, str) and card_name.isdigit():
            return int(card_name)
        if isinstance(card_name, str) and card_name in self.name_to_id:
            return self.name_to_id[card_name]
        print(f"警告: カード名 '{card_name}' が見つかりません")
        return -1

    def convert_attack_name_to_id(self, attack_name) -> int:
        """ワザ名／ID／数字文字列を安全にIDへ"""
        if isinstance(attack_name, int):
            return attack_name
        if isinstance(attack_name, list):
            attack_name = attack_name[0]
        # ★ 数値文字列をIDとして受け入れ
        if isinstance(attack_name, str) and attack_name.isdigit():
            return int(attack_name)
        if not isinstance(attack_name, str):
            print(f"警告: ワザ名が文字列ではありません: {attack_name}")
            return -1

        key = attack_name.strip()
        # 直接一致（日本語・英語・Enum名登録済み）
        if key in self.attack_name_to_id:
            return self.attack_name_to_id[key]

        # 小文字で再トライ
        key_l = key.lower()
        if key_l in self.attack_name_to_id:
            return self.attack_name_to_id[key_l]

        # 最後の保険：Enum名を直接参照
        try:
            return Attacks[key.upper()].value[0]
        except Exception:
            pass

        print(f"警告: ワザ名 '{attack_name}' が見つかりません")
        return -1

    def convert_type_to_id(self, type_name) -> int:
        """
        タイプ名をIDに変換。既に int の場合はそのまま返す。
        数字文字列 "3" のようなケースは int に変換して返す。
        英語名/日本語名は従来通りマップで解決。
        不明な場合のみ -1。
        """
        # すでに数値なら素通し
        if isinstance(type_name, int):
            return type_name
        # "3" のような数字文字列なら int 化して返す
        if isinstance(type_name, str) and type_name.isdigit():
            return int(type_name)
        # 文字列でマップにある場合
        if isinstance(type_name, str) and type_name in self.TYPE_ID:
            return self.TYPE_ID[type_name]
        print(f"警告: タイプ '{type_name}' をIDに変換できません")
        return -1
    
    def convert_list_of_cards(self, card_list):
        """
        カード名またはIDのリストを「全て展開」したIDリストに変換
        例: ["雷エネルギー", ["雷エネルギー", 2], 70005] → [4, 4, 4, 70005]
        """
        converted = []
        for item in card_list:
            if isinstance(item, str):
                converted.append(self._card_token_to_id(item))
            elif isinstance(item, int):
                converted.append(item)
            elif (isinstance(item, list) and len(item) == 2 and isinstance(item[1], int)):
                # [カード名 or ID, 枚数] → 展開
                card_token, n = item
                card_id = self._card_token_to_id(card_token) if isinstance(card_token, str) else card_token
                converted.extend([card_id] * n)
            else:
                # その他（想定外）はそのまま
                converted.append(item)
        return converted
    
    def action_to_id(self, action):
        """アクションをID形式に変換（新形式[5ints]は素通し）"""
        if not action or len(action) == 0:
            return [-1]  # unknown

        # --- 新形式をそのまま通す ---
        if isinstance(action[0], int):
            # 既に [t, main, p3, p4, p5] などの整数配列
            return action

        kind = action[0]
        if kind not in self.ACTION_TYPE:
            return [-1]  # unknown
        
        action_id = self.ACTION_TYPE[kind]
        
        if kind == "attach_energy":
            # [action_id, エネルギーID, 場所ID, ポケモンID]
            if len(action) >= 4:
                energy_id = self.convert_card_name_to_id(action[1])
                place_id = self.PLACE_ID.get(action[2], -1)
                pokemon_id = self.convert_card_name_to_id(action[3])
                return [action_id, energy_id, place_id, pokemon_id]
            else:
                return [-1]  # パラメータ不足
        elif kind == "use_item":
            # [action_id, カードID]
            if len(action) >= 2:
                card_id = self._card_token_to_id(action[1])
                return [action_id, card_id]
            else:
                return [-1]
        elif kind == "use_supporter":
            # action = ["use_supporter", <supporter name or id>, ...]
            supp_id = self._card_token_to_id(action[1])
            # Bossの指令だけ追加引数を取る
            if supp_id == 70002:  # 必要なら定数化してもOK
                slot_idx = action[2] if len(action) > 2 else -1
                poke_id  = self.convert_card_name_to_id(action[3]) if len(action) > 3 else -1
                return [self.ACTION_TYPE["use_supporter"], supp_id, slot_idx, poke_id]
            else:
                # それ以外のサポーター
                return [self.ACTION_TYPE["use_supporter"], supp_id]
        elif kind == "retreat":
            # [action_id]
            return [action_id]
        elif kind == "attack":
            # [action_id, ポケモンID, 技ID]
            if len(action) >= 3:
                pokemon_id = self.convert_card_name_to_id(action[1])
                attack_id = self.convert_attack_name_to_id(action[2])
                return [action_id, pokemon_id, attack_id]
            else:
                return [-1]
        elif kind == "play_bench":
            # [action_id, ポケモンID]
            if len(action) >= 2:
                pokemon_id = self.convert_card_name_to_id(action[1])
                return [action_id, pokemon_id]
            else:
                return [-1]
        elif kind == "end_turn":
            # [action_id]
            return [action_id]
        elif kind == "evolve":
            # [action_id, 元ポケモンID, 進化先ポケモンID]
            if len(action) >= 3:
                from_pokemon_id = self.convert_card_name_to_id(action[1])
                to_pokemon_id = self.convert_card_name_to_id(action[2])
                return [action_id, from_pokemon_id, to_pokemon_id]
            else:
                return [-1]
        elif kind == "ability":
            # [action_id, 特性ID, 予備(整数プレースホルダ)]
            if len(action) >= 2:
                ability_id = self.ability_name_to_id.get(action[1], -1)
                return [action_id, ability_id, -1]
            else:
                return [-1]
        elif kind == "stadium":
            # [action_id, スタジアムID]
            if len(action) >= 2:
                stadium_id = self._card_token_to_id(action[1])
                return [action_id, stadium_id]
            else:
                return [-1]
        elif kind == "stadium_effect":
            # [action_id, スタジアムID]
            if len(action) >= 2:
                stadium_id = self._card_token_to_id(action[1])
                return [action_id, stadium_id]
            else:
                return [-1]
        elif kind == "attach_tool":
            # [action_id, ツールID, 場所ID, ポケモンID]
            if len(action) >= 4:
                tool_id = self._card_token_to_id(action[1])
                place_id = self.PLACE_ID.get(action[2], -1)
                pokemon_id = self.convert_card_name_to_id(action[3])
                return [action_id, tool_id, place_id, pokemon_id]
            else:
                return [-1]
        else:
            return [-1]  # unknown
    
    def convert_legal_actions(self, legal_actions):
        """legal_actionsのカード名をIDに変換（新形式[5ints]は素通し）"""
        converted_actions = []
        for action in legal_actions:
            if isinstance(action, list) and len(action) > 0:
                # 既に整数配列ならそのまま
                if all(isinstance(x, int) for x in action):
                    converted_actions.append(action)
                else:
                    converted_actions.append(self.action_to_id(action))
            else:
                # 非リスト（None / tuple / str / 単一int 等）は必ず数値配列へ正規化
                if action is None:
                    continue
                if isinstance(action, tuple):
                    action = list(action)
                    if len(action) > 0 and all(isinstance(x, int) for x in action):
                        converted_actions.append(action)
                    else:
                        converted_actions.append(self.action_to_id(action))
                elif isinstance(action, int):
                    converted_actions.append([int(action)])
                else:
                    converted_actions.append(self.action_to_id([action]))
        return converted_actions

    def convert_legal_actions_32d(self, legal_actions, player=None):
        """
        legal_actions を 32次元の候補ベクトル（list[list[int]]）へ変換する。
        既存の convert_legal_actions()（=ID配列化）を前段に使い、
        各アクションID配列を 32 要素に pad/truncate する。
        """
        try:
            la_ids = self.convert_legal_actions(legal_actions if legal_actions is not None else [])
        except Exception:
            la_ids = []

        out = []
        for v in la_ids or []:
            if v is None:
                continue

            if not isinstance(v, list):
                try:
                    v = list(v)
                except Exception:
                    continue

            vv = []
            for x in v:
                try:
                    vv.append(int(x))
                except Exception:
                    vv.append(-1)

            if len(vv) < 32:
                vv = vv + [-1] * (32 - len(vv))
            elif len(vv) > 32:
                vv = vv[:32]

            out.append(vv)

        return out

    def convert_legal_actions_vec32(self, legal_actions, player=None):
        return self.convert_legal_actions_32d(legal_actions, player=player)

    def legal_actions_to_vec32(self, legal_actions, player=None):
        return self.convert_legal_actions_32d(legal_actions, player=player)

    def convert_action_result(self, action_result, keep_private=False):
        """action_resultのカード名をIDに変換（substepsを含めて再帰的に処理）"""
        if action_result is None:
            return None

        import copy
        converted_result = copy.deepcopy(action_result)

        # action（新形式[ints]は素通し、旧形式はID化）
        if 'action' in converted_result:
            a = converted_result['action']
            if isinstance(a, list) and len(a) > 0:
                if not all(isinstance(x, int) for x in a):
                    converted_result['action'] = self.action_to_id(a)

        # ★ 追加: macro も action と同様に正規化（action が無い時は action に寄せる）
        if 'macro' in converted_result and 'action' not in converted_result:
            m = converted_result['macro']
            if isinstance(m, list) and len(m) > 0:
                converted_result['action'] = m if all(isinstance(x, int) for x in m) else self.action_to_id(m)
            # 冗長化を避けるため macro は削除
            converted_result.pop('macro', None)

        # supporter フィールド（actionで use_supporter がID化済みなら削除）
        if 'supporter' in converted_result:
            if ('action' in converted_result and
                isinstance(converted_result['action'], list) and
                len(converted_result['action']) > 0 and
                converted_result['action'][0] == self.ACTION_TYPE.get('use_supporter', -1)):
                del converted_result['supporter']
            else:
                converted_result['supporter'] = self._card_token_to_id(converted_result['supporter'])

        # item フィールド（actionで use_item がID化済みなら削除）
        if 'item' in converted_result:
            if ('action' in converted_result and
                isinstance(converted_result['action'], list) and
                len(converted_result['action']) > 0 and
                converted_result['action'][0] == self.ACTION_TYPE.get('use_item', -1)):
                del converted_result['item']
            else:
                converted_result['item'] = self._card_token_to_id(converted_result['item'])

        # selected_pokemon（既にIDなら素通し／dict{name}にも対応）
        if "selected_pokemon" in converted_result:
            sp = converted_result["selected_pokemon"]
            if isinstance(sp, int):
                pass  # そのまま
            elif isinstance(sp, dict) and "name" in sp:
                converted_result["selected_pokemon"] = self.convert_card_name_to_id(sp["name"])
            else:
                converted_result["selected_pokemon"] = self.convert_card_name_to_id(sp)

        # ---- substeps を再帰的にID化 ----
        if 'substeps' in converted_result and isinstance(converted_result['substeps'], list):
            new_steps = []
            for step in converted_result['substeps']:
                if not isinstance(step, dict):
                    new_steps.append(step)
                    continue

                step_conv = {}

                # phase はそのまま
                if 'phase' in step:
                    step_conv['phase'] = step['phase']

                # スナップショットを再帰変換
                if 'state_before' in step:
                    step_conv['state_before'] = self.convert_state(step['state_before'], keep_private=keep_private)
                if 'state_after' in step:
                    step_conv['state_after'] = self.convert_state(step['state_after'], keep_private=keep_private)

                # 合法手は5整数配列に正規化
                if 'legal_actions' in step:
                    step_conv['legal_actions'] = self.convert_legal_actions(step['legal_actions'])

                # 選択ベクトル（新旧対応）
                if 'action_vec' in step:
                    av = step['action_vec']
                    if isinstance(av, list) and len(av) > 0 and not all(isinstance(x, int) for x in av):
                        step_conv['action_vec'] = self.action_to_id(av)
                    else:
                        step_conv['action_vec'] = av

                # インデックスはそのまま
                if 'action_index' in step:
                    step_conv['action_index'] = step['action_index']

                new_steps.append(step_conv)

            converted_result['substeps'] = new_steps

        return converted_result

    def convert_state(self, state, keep_private=False):
        """stateオブジェクトのカード名をIDに変換"""
        if state is None:
            return None
        converted_state = state.copy()
        # ▼非公開フィールドの扱い: IDS（公開）では削除、PRIVATE_IDS（完全情報）では保持
        if not keep_private:
            for k in ("me_private","opp_private","private","opp_hand","opp_deck","me_deck"):
                converted_state.pop(k, None)

        # スキーマガード: discard は使用しない（discard_pile のみ許可）
        if 'discard' in converted_state:
            raise ValueError("state schema violation: use 'discard_pile' only (found 'discard')")

        # me / opp が入っている場合は再帰的に変換
        for key in ('me', 'opp'):
            if key in converted_state and isinstance(converted_state[key], dict):
                converted_state[key] = self._convert_player_state(
                converted_state[key]
            )

        # handフィールドの変換
        if 'hand' in converted_state:
            converted_state['hand'] = self.convert_list_of_cards(converted_state['hand'])

        # 追加: private系で現れうる prize_enum のID化
        if 'prize_enum' in converted_state:
            converted_state['prize_enum'] = self.convert_list_of_cards(converted_state['prize_enum'])

        # 追加: private系で現れうる revealed のID化
        if 'revealed' in converted_state:
            converted_state['revealed'] = self.convert_list_of_cards(converted_state['revealed'])

        # legal_actions（トップレベル）もID化する
        if 'legal_actions' in converted_state:
            converted_state['legal_actions'] = self.convert_legal_actions(converted_state['legal_actions'])

        # discard_pileフィールドの変換
        if 'discard_pile' in converted_state:
            converted_state['discard_pile'] = self.convert_list_of_cards(converted_state['discard_pile'])

        # active_pokemonの変換
        if 'active_pokemon' in converted_state and converted_state['active_pokemon']:
            active_pokemon = converted_state['active_pokemon'].copy()
            if 'name' in active_pokemon and active_pokemon['name']:
                active_pokemon['name'] = self.convert_card_name_to_id(active_pokemon['name'])
            if 'energies' in active_pokemon:
                active_pokemon['energies'] = self.convert_list_of_cards(active_pokemon['energies'])
            if 'tools' in active_pokemon:
                active_pokemon['tools'] = self.convert_list_of_cards(active_pokemon['tools'])
            # ★ 追加: 型/弱点をID化（bench と揃える）
            if 'type' in active_pokemon and isinstance(active_pokemon['type'], str):
                active_pokemon['type'] = self.convert_type_to_id(active_pokemon['type'])
            if 'weakness' in active_pokemon and isinstance(active_pokemon['weakness'], str):
                active_pokemon['weakness'] = self.convert_type_to_id(active_pokemon['weakness'])
            converted_state['active_pokemon'] = active_pokemon

        # bench_pokemonの変換
        if 'bench_pokemon' in converted_state and isinstance(converted_state['bench_pokemon'], list):
            converted_bench = []
            for pokemon in converted_state['bench_pokemon']:
                if isinstance(pokemon, dict):
                    converted_pokemon = pokemon.copy()
                    if 'name' in converted_pokemon and converted_pokemon['name']:
                        converted_pokemon['name'] = self.convert_card_name_to_id(converted_pokemon['name'])
                    if 'energies' in converted_pokemon:
                        converted_pokemon['energies'] = self.convert_list_of_cards(converted_pokemon['energies'])
                    if 'tools' in converted_pokemon:
                        converted_pokemon['tools'] = self.convert_list_of_cards(converted_pokemon['tools'])
                    if 'type' in converted_pokemon and isinstance(converted_pokemon['type'], str):
                        converted_pokemon['type'] = self.convert_type_to_id(converted_pokemon['type'])
                    if 'weakness' in converted_pokemon and isinstance(converted_pokemon['weakness'], str):
                        converted_pokemon['weakness'] = self.convert_type_to_id(converted_pokemon['weakness'])
                    converted_bench.append(converted_pokemon)
                else:
                    converted_bench.append(pokemon)
            converted_state['bench_pokemon'] = converted_bench

        # active_stadiumの変換
        if 'active_stadium' in converted_state and converted_state['active_stadium']:
            if isinstance(converted_state['active_stadium'], str):
                converted_state['active_stadium'] = self.convert_card_name_to_id(converted_state['active_stadium'])

        # --- keep_private=True の場合は *_private をID化して保持 ---
        if keep_private:
            for _pk in ("me_private","opp_private"):
                if _pk in state and isinstance(state[_pk], dict):
                    # privateブロック自体も既存の公開変換ロジックでID化
                    _pv = self.convert_state(state[_pk], keep_private=True)
                    try:
                        _db = _pv.get("deck_bag_counts")
                        if isinstance(_db, dict):
                            def _to_id(x):
                                if isinstance(x, int): return x
                                xs = str(x)
                                if xs.isdigit(): return int(xs)
                                cid = self.convert_card_name_to_id(xs)
                                return cid if isinstance(cid, int) else -1
                            _pv["deck_bag_counts"] = { _to_id(k): (int(v) if not isinstance(v, bool) else int(v)) for k, v in _db.items() }
                            _pv["deck_bag_counts"] = { kk: (vv if vv >= 0 else 0) for kk, vv in _pv["deck_bag_counts"].items() }
                    except Exception:
                        pass
                    converted_state[_pk] = _pv

        return converted_state

    def _convert_player_state(self, player_dict):
        # ※ convert_state は呼ばない（無限再帰回避）
        converted = player_dict.copy()

        # hand / discard_pile
        if 'hand' in converted:
            converted['hand'] = self.convert_list_of_cards(converted['hand'])
        if 'discard_pile' in converted:
            converted['discard_pile'] = self.convert_list_of_cards(converted['discard_pile'])

        # active_pokemon
        if 'active_pokemon' in converted and converted['active_pokemon']:
            ap = converted['active_pokemon'].copy()
            if 'name' in ap and ap['name']:
                ap['name'] = self.convert_card_name_to_id(ap['name'])
            if 'energies' in ap:
                ap['energies'] = self.convert_list_of_cards(ap['energies'])
            if 'tools' in ap:
                ap['tools'] = self.convert_list_of_cards(ap['tools'])
            if 'type' in ap:
                ap['type'] = self.convert_type_to_id(ap['type'])
            if 'weakness' in ap:
                ap['weakness'] = self.convert_type_to_id(ap['weakness'])
            converted['active_pokemon'] = ap

        # bench_pokemon
        if 'bench_pokemon' in converted and isinstance(converted['bench_pokemon'], list):
            new_bench = []
            for pk in converted['bench_pokemon']:
                if isinstance(pk, dict):
                    pk2 = pk.copy()
                    if 'name' in pk2 and pk2['name']:
                        pk2['name'] = self.convert_card_name_to_id(pk2['name'])
                    if 'energies' in pk2:
                        pk2['energies'] = self.convert_list_of_cards(pk2['energies'])
                    if 'tools' in pk2:
                        pk2['tools'] = self.convert_list_of_cards(pk2['tools'])
                    if 'type' in pk2:
                        pk2['type'] = self.convert_type_to_id(pk2['type'])
                    if 'weakness' in pk2:
                        pk2['weakness'] = self.convert_type_to_id(pk2['weakness'])
                    new_bench.append(pk2)
                else:
                    new_bench.append(pk)
            converted['bench_pokemon'] = new_bench

        # active_stadium
        if 'active_stadium' in converted and converted['active_stadium']:
            if isinstance(converted['active_stadium'], str):
                converted['active_stadium'] = self.convert_card_name_to_id(converted['active_stadium'])

        # legal_actions / action
        if 'legal_actions' in converted:
            converted['legal_actions'] = self.convert_legal_actions(converted['legal_actions'])
        if 'action' in converted and isinstance(converted['action'], list):
            converted['action'] = self.action_to_id(converted['action'])

        return converted

    def convert_record(self, record, keep_private=False):
        """単一のレコードを変換"""
        converted_record = record.copy()
        
        # state_beforeの変換
        if 'state_before' in converted_record:
            converted_record['state_before'] = self.convert_state(converted_record['state_before'], keep_private=keep_private)
        
        # state_afterの変換
        if 'state_after' in converted_record:
            converted_record['state_after'] = self.convert_state(converted_record['state_after'], keep_private=keep_private)
        
        # state_opponentの変換
        if 'state_opponent' in converted_record:
            converted_record['state_opponent'] = self.convert_state(converted_record['state_opponent'])
        
        # state_opponent_afterの変換
        if 'state_opponent_after' in converted_record:
            converted_record['state_opponent_after'] = self.convert_state(converted_record['state_opponent_after'])
        
        # legal_actionsの変換
        if 'legal_actions' in converted_record:
            converted_record['legal_actions'] = self.convert_legal_actions(converted_record['legal_actions'])
        
        # action_resultの変換
        if 'action_result' in converted_record:
            converted_record['action_result'] = self.convert_action_result(converted_record['action_result'], keep_private=keep_private)
        
        # ※ done は入力のまま保持（上書きしない）
        # もし 'done' が無いログを古いフォーマットで扱う必要がある場合だけ、
        # 安全な推定（例えば state_after の 'done' を使うなど）を別途実装する。
        
        # 追加: 入力 record に *_private が含まれる場合、convert_state を用いて ID 化した上で復元する
        if keep_private:
            try:
                src = record if isinstance(record, dict) else {}
                for state_key in ("state_before", "state_after"):
                    src_state = src.get(state_key, {})
                    if isinstance(src_state, dict) and src_state:
                        dst_state = converted_record.setdefault(state_key, {})
                        if isinstance(dst_state, dict):
                            for k in ("me_private", "opp_private"):
                                if k in src_state and k not in dst_state:
                                    try:
                                        _pv = self.convert_state(src_state[k], keep_private=True)  # ★ 伝播
                                    except Exception:
                                        _pv = src_state[k]
                                    # deck_bag_counts のキーがカード名の場合は ID に置換
                                    try:
                                        _db = _pv.get("deck_bag_counts")
                                        if isinstance(_db, dict):
                                            def _to_card_id_strict_local(x):
                                                if isinstance(x, int):
                                                    return x
                                                xs = str(x)
                                                if xs.isdigit():
                                                    return int(xs)
                                                cid = self.convert_card_name_to_id(xs)
                                                return cid if isinstance(cid, int) else -1
                                            _pv["deck_bag_counts"] = {
                                                _to_card_id_strict_local(key): (int(val) if isinstance(val, (int,)) else int(str(val)))
                                                for key, val in _db.items()
                                            }
                                            _pv["deck_bag_counts"] = {
                                                k2: (v2 if v2 >= 0 else 0)
                                                for k2, v2 in _pv["deck_bag_counts"].items()
                                            }
                                    except Exception:
                                        pass
                                    dst_state[k] = _pv
            except Exception:
                pass
        
        return converted_record



import types

def get_converter():
    """
    CardNameToIdConverter の実体を返す。
    互換目的で convert_record が無い版だった場合は、既存の
    convert_state / convert_action_result / convert_legal_actions を使って
    同等の convert_record を注入してから返す。
    """
    conv = CardNameToIdConverter()

    if not hasattr(conv, "convert_record"):
        def _convert_record(self, record, keep_private: bool = False):
            if record is None:
                return None
            out = dict(record)

            if "state_before" in out:
                out["state_before"] = self.convert_state(out["state_before"], keep_private=keep_private)
            if "state_after" in out:
                out["state_after"] = self.convert_state(out["state_after"], keep_private=keep_private)

            if "state_opponent" in out:
                out["state_opponent"] = self.convert_state(out["state_opponent"], keep_private=False)
            if "state_opponent_after" in out:
                out["state_opponent_after"] = self.convert_state(out["state_opponent_after"], keep_private=False)

            if "legal_actions" in out:
                out["legal_actions"] = self.convert_legal_actions(out["legal_actions"])

            if "action_result" in out:
                out["action_result"] = self.convert_action_result(out["action_result"], keep_private=keep_private)

            return out

        conv.convert_record = types.MethodType(_convert_record, conv)
        print("[PATCH] convert_record を注入しました")

    return conv

# 追加ここから（parse_log_file の上あたり）-----------------------------------
try:
    import numpy as np
    _NP_INTEGER = (np.integer,)
except Exception:
    np = None
    _NP_INTEGER = ()

def _normalize_player_name(v):
    """'p1'/'p2' と推定できる文字/数値を正規化。分からなければ None"""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("p1", "player1", "1"): return "p1"
        if s in ("p2", "player2", "2"): return "p2"
        # "p1(turn)" のような文字列にも一応対応
        if "p1" in s: return "p1"
        if "p2" in s: return "p2"
    elif isinstance(v, (int,) + _NP_INTEGER):
        # 0/1 を p1/p2 とみなす（実装依存。違えば下で unknown になるだけ）
        return "p1" if int(v) == 0 else "p2" if int(v) == 1 else None
    return None

def infer_winner_from_entries(entries, fallback_p1="p1", fallback_p2="p2"):
    """
    各試合の entries から勝者を 'p1' / 'p2' / 'draw' / 'unknown' で返す。
    改良点:
      - 「最後の done==1」ではなく、まず「非ダミーの done==1 で reward!=0」を後ろから優先
      - それが無ければ、全ターミナルのうち reward!=0 を後ろから探索
      - まだ決まらなければ、盤面情報（prize_count / bench + active / deck_count）で推定
      - どれも無ければ draw（ダミー行だけ）／entries 無なら unknown
    """
    # 終局候補を抽出（順序維持）
    terminals = []
    for e in entries:
        if e.get("done") == 1 or (isinstance(e.get("state_after"), dict) and e["state_after"].get("done") == 1):
            terminals.append(e)
    if not terminals:
        return "unknown"

    def _winner_from_entry(e):
        sa = e.get("state_after") or {}
        meta = e.get("meta") or {}

        # winner の明示フィールド（sa / meta 両方＋ゆらぎ）
        for src in (sa, meta):
            for key in ("winner", "winner_name", "winner_player", "win_side", "winner_side"):
                if key in src:
                    who = _normalize_player_name(src[key])
                    if who: return who
            # result が dict の場合
            res = src.get("result") if isinstance(src.get("result"), dict) else None
            if res:
                who = _normalize_player_name(res.get("winner") or res.get("win_side"))
                if who: return who

        # 手番候補（state_after → state_before の順で探索）
        actor = None
        for src in (sa, e.get("state_before") or {}):
            for k in ("current_player_name", "current_player", "turn_player", "actor", "player"):
                if k in src:
                    actor = _normalize_player_name(src[k])
                    if actor:
                        break
            if actor:
                break

        r = e.get("reward")
        if r is not None:
            try:
                rr = float(r)
            except Exception:
                rr = None
            if rr is not None and rr != 0.0 and actor in ("p1", "p2"):
                return actor if rr > 0 else ("p2" if actor == "p1" else "p1")

        return None  # ここではまだ決めない

    # 1) 非ダミー（meta.terminal でない）かつ reward!=0 を後ろから優先
    for e in reversed(terminals):
        meta = e.get("meta") or {}
        if not bool(meta.get("terminal", False)):
            w = _winner_from_entry(e)
            if w in ("p1", "p2"):
                return w

    # 2) それでも決まらないなら、全ターミナルのうち reward!=0 を後ろから探索
    for e in reversed(terminals):
        w = _winner_from_entry(e)
        if w in ("p1", "p2"):
            return w

    # 3) 盤面から推定（最後のレコードを利用）
    last = terminals[-1]
    sa = last.get("state_after") or {}
    me = sa.get("me") or {}
    opp = sa.get("opp") or {}

    # 3-1) サイド切れ（優先）
    pm = me.get("prize_count")
    po = opp.get("prize_count")
    if isinstance(pm, int) and pm == 0 and isinstance(me.get("player", None), str):
        return _normalize_player_name(me["player"]) or "unknown"
    if isinstance(po, int) and po == 0 and isinstance(opp.get("player", None), str):
        return _normalize_player_name(opp["player"]) or "unknown"

    # 3-2) ベンチ切れ（active が倒れていて bench_count==0）
    def _no_board(side: dict) -> bool:
        active = side.get("active_pokemon") or {}
        hp = active.get("hp", None)
        bench = side.get("bench_count", None)
        return (isinstance(bench, int) and bench == 0) and (isinstance(hp, (int, float)) and hp <= 0)

    if _no_board(me) and isinstance(opp.get("player", None), str):
        return _normalize_player_name(opp["player"]) or "unknown"
    if _no_board(opp) and isinstance(me.get("player", None), str):
        return _normalize_player_name(me["player"]) or "unknown"

    # 3-3) デッキ切れ（deck_count==0 → その側が敗北）
    dm = me.get("deck_count")
    do = opp.get("deck_count")
    if isinstance(dm, int) and dm == 0 and isinstance(opp.get("player", None), str):
        return _normalize_player_name(opp["player"]) or "unknown"
    if isinstance(do, int) and do == 0 and isinstance(me.get("player", None), str):
        return _normalize_player_name(me["player"]) or "unknown"

    # 4) ここまで来たら、ダミー行のみとみなし draw
    return "draw"

# === 末尾の「終局以降」を出さないためのフィルタ =========================
TRIM_POST_TERMINAL = True  # 必要なら無効化可

def _is_terminal_entry(e: dict) -> bool:
    if not isinstance(e, dict):
        return False
    if e.get("done") == 1:
        return True
    sa = e.get("state_after") or {}
    if isinstance(sa, dict) and sa.get("done") == 1:
        return True
    meta = e.get("meta") or {}
    return bool(meta.get("terminal", False))

def _is_deckout_game_from_all(entries_all: list) -> bool:
    """未フィルタの全行からデッキアウト試合かどうかを判定する"""
    if not isinstance(entries_all, list):
        return False
    try:
        # 明示のメタ情報があれば最優先（＝それ以外はデッキアウト扱いしない）
        for e in entries_all:
            meta = e.get("meta") or {}
            if not isinstance(meta, dict):
                continue
            reason = str(meta.get("reason", "")).upper()
            term_reason = str(meta.get("terminal_reason", "")).upper()
            # "DECK_OUT", "DECKOUT", "DECK_OUT_P1" などをまとめて拾う
            if "DECK" in reason or "DECK" in term_reason:
                return True
    except Exception:
        pass
    return False

def _is_non_dummy_terminal(e: dict) -> bool:
    """「実アクション由来 or 明示的な終局」を優先（ダミーFINALを除外）"""
    if not _is_terminal_entry(e):
        return False
    meta = e.get("meta") or {}
    if meta.get("terminal", False):
        return True
    ar = e.get("action_result") or {}
    if isinstance(ar, dict) and (ar.get("action") is not None or ar.get("macro") is not None):
        return True
    r = e.get("reward", None)
    try:
        return (r is not None) and (float(r) != 0.0)
    except Exception:
        return False

def _trim_one_game(gentries: list) -> list:
    """
    1ゲーム分の entries を、
      1) 最初に現れる「非ダミー終局」をカット位置とし（無ければ最初の終局）、
      2) その“前の”終局行は捨て（ダミーFINAL等）、
      3) カット位置より後はすべて捨てる
    形に整える。
    """
    terminals = [i for i, e in enumerate(gentries) if _is_terminal_entry(e)]
    if not terminals:
        return gentries

    non_dummy = [i for i in terminals if _is_non_dummy_terminal(gentries[i])]
    cutoff = non_dummy[0] if non_dummy else terminals[0]

    trimmed = []
    for i, e in enumerate(gentries):
        if i > cutoff:
            break  # 終局以降を落とす
        if i < cutoff and _is_terminal_entry(e):
            continue  # 手前に出たダミー終局行は捨てる
        trimmed.append(e)
    return trimmed

def trim_entries_after_canonical_terminal(entries: list) -> list:
    """入力 entries を game_id ごとに _trim_one_game で整形"""
    if not TRIM_POST_TERMINAL:
        return entries
    out = []
    cur_gid = object()
    group = []
    def _flush(g):
        out.extend(_trim_one_game(g))
    for e in entries + [None]:  # センチネルで最後をflush
        gid = None
        if isinstance(e, dict):
            gid = e.get("game_id")
            if gid is None:
                sb = e.get("state_before") or {}
                sa = e.get("state_after") or {}
                gid = sb.get("game_id") or sa.get("game_id")
        if gid != cur_gid:
            if group:
                _flush(group)
            group = []
            cur_gid = gid
        if e is not None:
            group.append(e)
    return out
# ======================================================================

# --- 完全情報ベクトル（obs_full_vec）生成ユーティリティ --------------------
def _vec_from_bagcounts(bag: dict, V: int):
    import numpy as _np
    v = _np.zeros(V, dtype=_np.float32)
    if isinstance(bag, dict):
        for k, c in bag.items():
            try:
                cid = int(k)
                idx = _CARD_ID2IDX.get(cid, None)
                if idx is None:
                    continue
                v[idx] += float(c if not isinstance(c, bool) else int(c))
            except Exception:
                continue
    return v

def _vec_from_id_list(lst, V: int):
    import numpy as _np
    v = _np.zeros(V, dtype=_np.float32)
    if isinstance(lst, list):
        for x in lst:
            try:
                cid = int(x if not isinstance(x, (list, tuple)) else x[0])
                idx = _CARD_ID2IDX.get(cid, None)
                if idx is None:
                    continue
                v[idx] += 1.0
            except Exception:
                continue
    return v

def build_obs_full_vec(sb_priv: dict):
    """
    sb_priv: id化済みの state_before（keep_private=True で変換済み）を想定
    返り値: list[float] （固定次元 = 4*V + 7）
    """
    import numpy as _np
    Vloc = len(_CARD_ID2IDX)

    me_priv  = (sb_priv.get("me_private")  or {})
    opp_priv = (sb_priv.get("opp_private") or {})
    me_pub   = (sb_priv.get("me")          or {})
    opp_pub  = (sb_priv.get("opp")         or {})

    me_bag   = _vec_from_bagcounts(me_priv.get("deck_bag_counts"), Vloc)
    opp_bag  = _vec_from_bagcounts(opp_priv.get("deck_bag_counts"), Vloc)

    me_prize = _vec_from_id_list(me_priv.get("prize_enum", []), Vloc)
    opp_prize= _vec_from_id_list(opp_priv.get("prize_enum", []), Vloc)

    turn = int(sb_priv.get("turn") or 0)
    cur  = sb_priv.get("current_player")

    me_player  = me_pub.get("player")
    opp_player = opp_pub.get("player")

    cur_onehot = _np.array(
        [
            1.0 if cur == me_player else 0.0,
            1.0 if cur == opp_player else 0.0,
        ],
        dtype=_np.float32,
    )

    me_prize_cnt  = int(me_pub.get("prize_count")  or len(me_priv.get("prize_enum", [])) or 0)
    opp_prize_cnt = int(opp_pub.get("prize_count") or len(opp_priv.get("prize_enum", [])) or 0)
    me_deck_cnt   = int(me_pub.get("deck_count")   or 0)
    opp_deck_cnt  = int(opp_pub.get("deck_count")  or 0)

    scalars = _np.array(
        [turn, cur_onehot[0], cur_onehot[1], me_prize_cnt, opp_prize_cnt, me_deck_cnt, opp_deck_cnt],
        dtype=_np.float32,
    )

    vec = _np.concatenate([me_bag, opp_bag, me_prize, opp_prize, scalars], axis=0)
    return vec.astype(_np.float32).tolist()

def build_obs_partial_vec(sb_pub: dict):
    """
    sb_pub: state_before の公開版（me / opp のみ）を想定
    返り値: list[float]

    構成イメージ:
      - me_hand_slots_vec        : 30スロット × Vloc（位置付き手札 one-hot）
      - me_board_vec             : 自分アクティブ＋ベンチのポケモンID one-hot
      - me_discard_vec           : 自分トラッシュ内カードID one-hot
      - opp_board_vec            : 相手アクティブ＋ベンチのポケモンID one-hot
      - opp_discard_vec          : 相手トラッシュ内カードID one-hot
      - me_energy_vec            : 自分の場（アクティブ＋ベンチ）に付いているエネルギーカードID one-hot
      - opp_energy_vec           : 相手の場に付いているエネルギーカードID one-hot
      - me_active_energy_type_vec: 自分アクティブに付いているエネルギーの「種類」 one-hot（card_id ベース）
      - opp_active_energy_type_vec: 相手アクティブに付いているエネルギーの種類 one-hot
      - me_bench_energy_slots_vec: ベンチごとのエネルギー種類分布（max 8 スロット × Vloc）
      - opp_bench_energy_slots_vec: 相手ベンチごとのエネルギー種類分布
      - me_bench_board_slots_vec : ベンチごとのポケモンID分布（max 8 スロット × Vloc）
      - opp_bench_board_slots_vec: 相手ベンチごとのポケモンID分布

      - scalars                  : 11スカラー
                                   (turn, cur_is_me, cur_is_opp,
                                    me/opp prize, me/opp deck,
                                    me/opp hand_count,
                                    me/opp discard_pile_count)
      - me_status_vec            : 自分アクティブの特殊状態 one-hot（どく/やけど/マヒ/ねむり/こんらん）
      - opp_status_vec           : 相手アクティブの特殊状態 one-hot
      - extra_scalars            : 18スカラー
                                   (me/opp HP, me/opp エネ枚数, me/opp ベンチ数,
                                    me/opp damage_counters,
                                    me/opp has_weakness, me/opp has_resistance,
                                    me/opp is_basic, me/opp is_ex,
                                    me/opp retreat_cost)
      - type_vecs                : 40 次元
                                   (me_active_type(10), opp_active_type(10),
                                    me_weak_type(10),  opp_weak_type(10),
                                    me_resist_type(10),opp_resist_type(10))  ※ type -1 は all-zero
      - ability_flags            : 4スカラー
                                   (me ability_present/used, opp ability_present/used)
      - tools_vec                : 2 × Vloc
                                   (me_active_tools_vec, opp_active_tools_vec)
      - stadium_vec              : スタジアムカードID one-hot
                                   （top-level stadium と各 side の active_stadium を統合）
      - turn_flags               : 8スカラー
                                   (me: energy_attached, supporter_used, stadium_played, stadium_effect_used,
                                    opp: 同様 4 フラグ)
      - prev_ko_flag             : 1スカラー（前のターンで自分ポケモンが気絶したか）

    ※ エネルギーやタイプは card_id/type_id ベースで区別（基本エネ／特殊エネの違いは card_id か type_id に反映されている前提）。
    """
    import numpy as _np
    Vloc = len(_CARD_ID2IDX)
    TYPE_DIM = 10  # タイプID 0〜9 を one-hot

    if not isinstance(sb_pub, dict):
        # 形が想定外なら空ベクトルを返す（上流でフィルタされる）
        return []

    me_pub  = (sb_pub.get("me")  or {})
    opp_pub = (sb_pub.get("opp") or {})

    # --- 手札 30 スロット one-hot ---
    def _vec_from_hand_slots(hand, V: int, max_slots: int = 30):
        v = _np.zeros(max_slots * V, dtype=_np.float32)
        if isinstance(hand, list):
            for i, x in enumerate(hand):
                if i >= max_slots:
                    break
                try:
                    # [card_id, n] 形式も許容
                    cid = int(x[0] if isinstance(x, (list, tuple)) else x)
                    idx = _CARD_ID2IDX.get(cid, None)
                    if idx is None:
                        continue
                    base = i * V
                    v[base + idx] = 1.0
                except Exception:
                    continue
        return v

    # --- 盤面ポケモンID（アクティブ＋ベンチ） ---
    def _collect_pokemon_board_ids(pub: dict):
        ids = []

        active = pub.get("active_pokemon")
        if isinstance(active, dict):
            try:
                cid = int(active.get("name"))
                ids.append(cid)
            except Exception:
                pass

        bench = pub.get("bench_pokemon")
        if isinstance(bench, list):
            for obj in bench:
                if isinstance(obj, dict):
                    try:
                        cid = int(obj.get("name"))
                        ids.append(cid)
                    except Exception:
                        continue

        return _vec_from_id_list(ids, Vloc)

    # --- ベンチごとのポケモンID分布（max_bench_slots × Vloc） ---
    def _build_bench_board_slots_vec(pub: dict, V: int, max_bench_slots: int = 8):
        v = _np.zeros(max_bench_slots * V, dtype=_np.float32)
        bench = pub.get("bench_pokemon")
        if not isinstance(bench, list):
            return v
        for i, obj in enumerate(bench):
            if i >= max_bench_slots:
                break
            if not isinstance(obj, dict):
                continue
            try:
                cid = int(obj.get("name"))
                idx = _CARD_ID2IDX.get(cid, None)
                if idx is None:
                    continue
                base = i * V
                v[base + idx] = 1.0
            except Exception:
                continue
        return v

    # --- 単一ポケモンからエネルギーカード ID を集める ---
    def _collect_energy_ids_from_poke(poke: dict):
        ids = []
        if not isinstance(poke, dict):
            return ids
        # 可能性のあるキーを全部見る
        for key in ("energies", "energy", "attached_energies", "attached_energy", "energy_cards"):
            lst = poke.get(key)
            if isinstance(lst, list):
                for x in lst:
                    try:
                        cid = int(x[0] if isinstance(x, (list, tuple)) else x)
                        ids.append(cid)
                    except Exception:
                        continue
        return ids

    # --- 場に付いているエネルギーカードID（アクティブ＋ベンチ合算） ---
    def _collect_attached_energy_ids(pub: dict):
        ids = []

        active = pub.get("active_pokemon")
        if isinstance(active, dict):
            ids.extend(_collect_energy_ids_from_poke(active))

        bench = pub.get("bench_pokemon")
        if isinstance(bench, list):
            for obj in bench:
                if isinstance(obj, dict):
                    ids.extend(_collect_energy_ids_from_poke(obj))

        return _vec_from_id_list(ids, Vloc)

    # --- アクティブポケモンの HP / 特殊状態 / エネ枚数 ---
    def _extract_active_features(pub: dict):
        """
        戻り値:
          hp: float
          status_vec: 長さ5（poison/burn/paralysis/sleep/confusion）
          energy_count: float
        """
        hp = 0.0
        status_vec = _np.zeros(5, dtype=_np.float32)
        energy_count = 0.0

        active = pub.get("active_pokemon")
        if isinstance(active, dict):
            # HP（remaining_hp / hp / current_hp のどれかがあれば使う）
            for key in ("remaining_hp", "hp", "current_hp"):
                if key in active:
                    try:
                        hp = float(active.get(key) or 0.0)
                        break
                    except Exception:
                        pass

            # 特殊状態
            raw_cond = active.get("special_conditions") or active.get("conditions") or []
            cond_list = []
            if isinstance(raw_cond, list):
                cond_list = raw_cond
            elif isinstance(raw_cond, str):
                cond_list = [raw_cond]

            def _norm(s):
                if not isinstance(s, str):
                    return ""
                return s.strip().lower()

            for c in cond_list:
                k = _norm(c)
                if k in ("poison", "poisoned", "どく"):
                    status_vec[0] = 1.0
                elif k in ("burn", "burned", "やけど"):
                    status_vec[1] = 1.0
                elif k in ("paralyze", "paralyzed", "paralysis", "マヒ", "まひ"):
                    status_vec[2] = 1.0
                elif k in ("sleep", "asleep", "ねむり"):
                    status_vec[3] = 1.0
                elif k in ("confuse", "confused", "こんらん"):
                    status_vec[4] = 1.0

            # 付いているエネ枚数
            for key in ("energies", "energy", "attached_energies", "attached_energy", "energy_cards"):
                lst = active.get(key)
                if isinstance(lst, list):
                    energy_count += float(len(lst))

        return hp, status_vec, energy_count

    # --- アクティブの追加メタ情報 ---
    def _extract_active_meta(pub: dict):
        """
        戻り値 dict:
          damage_counters: float
          type_id: int
          has_weakness: float
          weakness_type: int
          has_resistance: float
          resistance_type: int
          is_basic: float
          is_ex: float
          retreat_cost: float
          ability_present: float
          ability_used: float
          tools_ids: List[int]
        """
        active = pub.get("active_pokemon")
        info = {
            "damage_counters": 0.0,
            "type_id": -1,
            "has_weakness": 0.0,
            "weakness_type": -1,
            "has_resistance": 0.0,
            "resistance_type": -1,
            "is_basic": 0.0,
            "is_ex": 0.0,
            "retreat_cost": 0.0,
            "ability_present": 0.0,
            "ability_used": 0.0,
            "tools_ids": [],
        }
        if not isinstance(active, dict):
            return info

        try:
            info["damage_counters"] = float(active.get("damage_counters") or 0.0)
        except Exception:
            pass

        try:
            t = active.get("type")
            if t is not None:
                info["type_id"] = int(t)
        except Exception:
            pass

        def _as_bool(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return 1.0
                if v in ("0", "false", "no", "n", "off", ""):
                    return 0.0
            return 0.0

        info["has_weakness"] = _as_bool(active.get("has_weakness"))
        info["has_resistance"] = _as_bool(active.get("has_resistance"))
        info["is_basic"] = _as_bool(active.get("is_basic"))
        info["is_ex"] = _as_bool(active.get("is_ex"))

        try:
            wt = active.get("weakness_type")
            if wt is not None:
                info["weakness_type"] = int(wt)
        except Exception:
            pass
        try:
            rt = active.get("resistance_type")
            if rt is not None:
                info["resistance_type"] = int(rt)
        except Exception:
            pass

        try:
            rc = active.get("retreat_cost")
            if rc is not None:
                info["retreat_cost"] = float(rc)
        except Exception:
            pass

        info["ability_present"] = _as_bool(
            active.get("ability_present") if "ability_present" in active else active.get("has_ability")
        )
        info["ability_used"] = _as_bool(
            active.get("ability_used") if "ability_used" in active else active.get("ability_used_this_turn")
        )

        tools_ids = []
        for key in ("tools", "tool", "attached_tools", "pokemon_tools"):
            lst = active.get(key)
            if isinstance(lst, list):
                for x in lst:
                    try:
                        cid = int(x[0] if isinstance(x, (list, tuple)) else x)
                        tools_ids.append(cid)
                    except Exception:
                        continue
            elif lst is not None:
                try:
                    cid = int(lst)
                    tools_ids.append(cid)
                except Exception:
                    pass
        info["tools_ids"] = tools_ids

        return info

    # --- タイプIDを one-hot に変換 ---
    def _type_id_to_onehot(type_id: int, dim: int = TYPE_DIM):
        v = _np.zeros(dim, dtype=_np.float32)
        try:
            t = int(type_id)
            if 0 <= t < dim:
                v[t] = 1.0
        except Exception:
            pass
        return v

    # --- アクティブに付いているエネルギーの「種類」 one-hot ---
    def _collect_active_energy_type_vec(pub: dict, V: int):
        active = pub.get("active_pokemon")
        ids = _collect_energy_ids_from_poke(active) if isinstance(active, dict) else []
        return _vec_from_id_list(ids, V)

    # --- ベンチごとのエネルギー種類分布（max_bench_slots × Vloc） ---
    def _build_bench_energy_slots_vec(pub: dict, V: int, max_bench_slots: int = 8):
        v = _np.zeros(max_bench_slots * V, dtype=_np.float32)
        bench = pub.get("bench_pokemon")
        if not isinstance(bench, list):
            return v
        for i, obj in enumerate(bench):
            if i >= max_bench_slots:
                break
            if not isinstance(obj, dict):
                continue
            ids = _collect_energy_ids_from_poke(obj)
            if not ids:
                continue
            slot_vec = _vec_from_id_list(ids, V)
            base = i * V
            v[base:base + V] = slot_vec
        return v

    # --- 「ターン内 1回」系のフラグを抽出 ---
    def _extract_turn_flags_for_side(pub_side: dict, sb: dict, side_key: str):
        """
        返り値: np.array(4,)
          [energy_attached, supporter_used, stadium_played, stadium_effect_used]
        それぞれ 0.0/1.0。キーが存在しない場合は 0.0。
        """
        flags = {
            "energy_attached": 0.0,
            "supporter_used": 0.0,
            "stadium_played": 0.0,
            "stadium_effect_used": 0.0,
        }

        # サイド側にぶら下がっているフラグ候補
        cand_maps = [
            pub_side,
            sb.get("turn_flags") or {},
            sb.get("meta") or {},
        ]

        def _as_bool(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return 1.0
                if v in ("0", "false", "no", "n", "off", ""):
                    return 0.0
            return 0.0

        # キーの候補を広めにとる（実際に存在するものだけ効く）
        key_candidates = {
            "energy_attached": [
                "energy_attached_this_turn",
                "energy_attach_this_turn",
                "energy_attached",
                "has_attached_energy_this_turn",
            ],
            "supporter_used": [
                "supporter_used_this_turn",
                "supporter_played_this_turn",
                "supporter_used",
            ],
            "stadium_played": [
                "stadium_played_this_turn",
                "stadium_used_this_turn",
                "stadium_played",
            ],
            "stadium_effect_used": [
                "stadium_effect_used_this_turn",
                "stadium_effect_used",
                "stadium_effect_this_turn",
            ],
        }

        for name, keys in key_candidates.items():
            for d in cand_maps:
                if not isinstance(d, dict):
                    continue
                for k in keys:
                    if k in d:
                        flags[name] = max(flags[name], _as_bool(d.get(k)))
                # side_key 付きのネームスペースも一応見る
                side_map = d.get(side_key) if isinstance(d.get(side_key), dict) else None
                if isinstance(side_map, dict):
                    for k in keys:
                        if k in side_map:
                            flags[name] = max(flags[name], _as_bool(side_map.get(k)))

        return _np.array(
            [
                flags["energy_attached"],
                flags["supporter_used"],
                flags["stadium_played"],
                flags["stadium_effect_used"],
            ],
            dtype=_np.float32,
        )

    # --- 前のターンに自分ポケモンが気絶したかフラグ ---
    def _extract_prev_ko_flag(sb: dict, pub_side: dict, side_key: str):
        """
        返り値: float（0.0 or 1.0）
        候補キー:
          - sb["prev_knockout"][side_key]
          - sb["last_turn_knockout"][side_key]
          - pub_side["ko_last_turn"], pub_side["knocked_out_last_turn"] など
        """
        def _as_bool(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return 1.0
                if v in ("0", "false", "no", "n", "off", ""):
                    return 0.0
            return 0.0

        # state_before 直下に side ごとの情報があるケース
        for key in ("prev_knockout", "last_turn_knockout", "previous_turn_knockout"):
            d = sb.get(key)
            if isinstance(d, dict) and side_key in d:
                return _as_bool(d.get(side_key))

        # サイド側に直接ぶら下がっているケース
        for key in ("ko_last_turn", "knocked_out_last_turn", "pokemon_fainted_last_turn"):
            if key in pub_side:
                return _as_bool(pub_side.get(key))

        return 0.0

    # --- 基本ベクトル ---
    me_hand_slots_vec = _vec_from_hand_slots(me_pub.get("hand", []), Vloc)
    me_board_vec      = _collect_pokemon_board_ids(me_pub)
    me_discard_vec    = _vec_from_id_list(me_pub.get("discard_pile", []), Vloc)

    opp_board_vec     = _collect_pokemon_board_ids(opp_pub)
    opp_discard_vec   = _vec_from_id_list(opp_pub.get("discard_pile", []), Vloc)

    me_energy_vec     = _collect_attached_energy_ids(me_pub)
    opp_energy_vec    = _collect_attached_energy_ids(opp_pub)

    # --- アクティブのエネルギー「種類」ベクトル ---
    me_active_energy_type_vec  = _collect_active_energy_type_vec(me_pub, Vloc)
    opp_active_energy_type_vec = _collect_active_energy_type_vec(opp_pub, Vloc)

    # --- ベンチごとのエネルギー種類分布 ---
    me_bench_energy_slots_vec  = _build_bench_energy_slots_vec(me_pub, Vloc)
    opp_bench_energy_slots_vec = _build_bench_energy_slots_vec(opp_pub, Vloc)

    # --- ベンチごとのポケモンID分布 ---
    me_bench_board_slots_vec  = _build_bench_board_slots_vec(me_pub, Vloc)
    opp_bench_board_slots_vec = _build_bench_board_slots_vec(opp_pub, Vloc)

    # --- ターン情報・サイド・山札残数＋手札枚数＋トラッシュ枚数 ---
    turn = int(sb_pub.get("turn") or 0)
    cur  = sb_pub.get("current_player")

    me_player  = me_pub.get("player")
    opp_player = opp_pub.get("player")

    cur_onehot = _np.array(
        [
            1.0 if cur == me_player else 0.0,
            1.0 if cur == opp_player else 0.0,
        ],
        dtype=_np.float32,
    )

    me_prize_cnt  = int(me_pub.get("prize_count") or 0)
    opp_prize_cnt = int(opp_pub.get("prize_count") or 0)
    me_deck_cnt   = int(me_pub.get("deck_count")  or 0)
    opp_deck_cnt  = int(opp_pub.get("deck_count") or 0)

    me_hand_cnt   = int(me_pub.get("hand_count") or (len(me_pub.get("hand")) if isinstance(me_pub.get("hand"), list) else 0))
    opp_hand_cnt  = int(opp_pub.get("hand_count") or (len(opp_pub.get("hand")) if isinstance(opp_pub.get("hand"), list) else 0))

    me_discard_cnt  = int(me_pub.get("discard_pile_count") or (len(me_pub.get("discard_pile")) if isinstance(me_pub.get("discard_pile"), list) else 0))
    opp_discard_cnt = int(opp_pub.get("discard_pile_count") or (len(opp_pub.get("discard_pile")) if isinstance(opp_pub.get("discard_pile"), list) else 0))

    scalars = _np.array(
        [
            turn,
            cur_onehot[0],
            cur_onehot[1],
            me_prize_cnt,
            opp_prize_cnt,
            me_deck_cnt,
            opp_deck_cnt,
            me_hand_cnt,
            opp_hand_cnt,
            me_discard_cnt,
            opp_discard_cnt,
        ],
        dtype=_np.float32,
    )

    # --- アクティブ状態／エネ枚数／ベンチ数 ---
    me_hp,  me_status_vec,  me_energy_cnt  = _extract_active_features(me_pub)
    opp_hp, opp_status_vec, opp_energy_cnt = _extract_active_features(opp_pub)

    me_bench = me_pub.get("bench_pokemon")
    opp_bench = opp_pub.get("bench_pokemon")

    def _count_real_bench(lst):
        """
        bench_pokemon は [dict or 0 or None, ...] のようにスロット数で埋められることがあるため、
        実際にポケモンが存在するスロット（dict）のみを数える。
        スタジアム効果により 3〜8 スロットになる変動もこのカウントで吸収する。
        """
        if not isinstance(lst, list):
            return 0.0
        c = 0
        for obj in lst:
            if isinstance(obj, dict):
                c += 1
        return float(c)

    me_bench_cnt  = _count_real_bench(me_bench)
    opp_bench_cnt = _count_real_bench(opp_bench)

    # --- アクティブの追加メタ情報 ---
    me_meta  = _extract_active_meta(me_pub)
    opp_meta = _extract_active_meta(opp_pub)

    extra_scalars = _np.array(
        [
            me_hp,
            opp_hp,
            me_energy_cnt,
            opp_energy_cnt,
            me_bench_cnt,
            opp_bench_cnt,
            me_meta["damage_counters"],
            opp_meta["damage_counters"],
            me_meta["has_weakness"],
            opp_meta["has_weakness"],
            me_meta["has_resistance"],
            opp_meta["has_resistance"],
            me_meta["is_basic"],
            opp_meta["is_basic"],
            me_meta["is_ex"],
            opp_meta["is_ex"],
            me_meta["retreat_cost"],
            opp_meta["retreat_cost"],
        ],
        dtype=_np.float32,
    )

    # --- タイプ関連 one-hot ---
    me_type_vec        = _type_id_to_onehot(me_meta["type_id"], TYPE_DIM)
    opp_type_vec       = _type_id_to_onehot(opp_meta["type_id"], TYPE_DIM)
    me_weak_type_vec   = _type_id_to_onehot(me_meta["weakness_type"], TYPE_DIM)
    opp_weak_type_vec  = _type_id_to_onehot(opp_meta["weakness_type"], TYPE_DIM)
    me_resist_type_vec = _type_id_to_onehot(me_meta["resistance_type"], TYPE_DIM)
    opp_resist_type_vec= _type_id_to_onehot(opp_meta["resistance_type"], TYPE_DIM)

    type_vecs = _np.concatenate(
        [
            me_type_vec,
            opp_type_vec,
            me_weak_type_vec,
            opp_weak_type_vec,
            me_resist_type_vec,
            opp_resist_type_vec,
        ],
        axis=0,
    )

    # --- アビリティフラグ ---
    ability_flags = _np.array(
        [
            me_meta["ability_present"],
            me_meta["ability_used"],
            opp_meta["ability_present"],
            opp_meta["ability_used"],
        ],
        dtype=_np.float32,
    )

    # --- アクティブポケモンに付いているどうぐ ---
    me_tools_vec  = _vec_from_id_list(me_meta["tools_ids"], Vloc)
    opp_tools_vec = _vec_from_id_list(opp_meta["tools_ids"], Vloc)

    # --- スタジアムカード ---
    stadium_ids = []
    stadium = sb_pub.get("stadium")
    if isinstance(stadium, dict):
        # {"name": id} / {"id": id} の両方を許容
        if "name" in stadium:
            try:
                stadium_ids.append(int(stadium.get("name")))
            except Exception:
                pass
        if "id" in stadium:
            try:
                stadium_ids.append(int(stadium.get("id")))
            except Exception:
                pass
    elif stadium is not None:
        try:
            stadium_ids.append(int(stadium))
        except Exception:
            pass

    # サイド別 active_stadium も拾う
    for side_pub in (me_pub, opp_pub):
        st = side_pub.get("active_stadium")
        if isinstance(st, dict):
            if "name" in st:
                try:
                    stadium_ids.append(int(st.get("name")))
                except Exception:
                    pass
            if "id" in st:
                try:
                    stadium_ids.append(int(st.get("id")))
                except Exception:
                    pass
        elif st is not None:
            try:
                stadium_ids.append(int(st))
            except Exception:
                pass

    stadium_vec = _vec_from_id_list(stadium_ids, Vloc)

    # --- ターン内 1回系フラグ（自分・相手） ---
    me_turn_flags  = _extract_turn_flags_for_side(me_pub, sb_pub, "me")
    opp_turn_flags = _extract_turn_flags_for_side(opp_pub, sb_pub, "opp")
    turn_flags = _np.concatenate([me_turn_flags, opp_turn_flags], axis=0)

    # --- 前のターンで自分ポケモンが気絶したか ---
    prev_ko_flag = _np.array(
        [_extract_prev_ko_flag(sb_pub, me_pub, "me")],
        dtype=_np.float32,
    )

    vec = _np.concatenate(
        [
            me_hand_slots_vec,
            me_board_vec,
            me_discard_vec,
            opp_board_vec,
            opp_discard_vec,
            me_energy_vec,
            opp_energy_vec,
            me_active_energy_type_vec,
            opp_active_energy_type_vec,
            me_bench_energy_slots_vec,
            opp_bench_energy_slots_vec,
            me_bench_board_slots_vec,
            opp_bench_board_slots_vec,
            scalars,
            me_status_vec,
            opp_status_vec,
            extra_scalars,
            type_vecs,
            ability_flags,
            me_tools_vec,
            opp_tools_vec,
            stadium_vec,
            turn_flags,
            prev_ko_flag,
        ],
        axis=0,
    )
    return vec.astype(_np.float32).tolist()

def attach_fullvec_fields(id_entry_priv: dict):
    """
    PRIVATE_IDS の単一エントリに obs_full_vec / obs_full_dim / obs_full_version を付与
    """
    try:
        sbp = id_entry_priv.get("state_before") or {}
        fv = build_obs_full_vec(sbp)
        id_entry_priv["obs_full_vec"] = fv
        id_entry_priv["obs_full_dim"] = len(fv)
        id_entry_priv["obs_full_version"] = FULL_VEC_VERSION
    except Exception:
        # 失敗しても他のフィールドは書き出す
        pass

def parse_log_file(log_file_path: str):
    """
    ログファイルを解析して、各ターンの状態とアクションを抽出する。

    取りこぼし防止のポイント（修正版）:
    1) [STATE_BEFORE] ブロックの「内側探索」でも、素の1行JSON
        （行頭 '{' で、かつ "state_before" と "state_after" を含む）を検出して
        その場で entries に追加する（ターンエンドや終局の素JSONも取りこぼさない）。
    2) そのブロックで素JSONを1件以上拾えた場合は、フォールバック（sb/ar/sa の合成）を出力しない。
    3) 重複統合キーに action_result.action（なければ macro）を含め、
        同一ターン内の複数アクション（例: どうぐ付与 → ターンエンド）を別レコードとして保持。
        action_result が無い終局行は ("FINAL",) を付けて区別。
    4) [LEGAL_ACTIONS] 行があればフォールバック生成時に付与（素JSON側には元々含まれる）。
    """
    import json

    def _pop_reward_done(dct):
        """dict から reward/done を取り出し（存在すれば削除して返す）"""
        if not isinstance(dct, dict):
            return None, None
        r = dct.pop('reward', None) if 'reward' in dct else None
        d = dct.pop('done', None) if 'done' in dct else None
        return r, d

    def _extract_game_id(*objs):
        """state_before/state_after から最初に見つかった game_id を返す"""
        for o in objs:
            if isinstance(o, dict) and 'game_id' in o:
                return o['game_id']
        return None

    def _best_entry(prev, new):
        """
        重複時の優先順位: done=1 > state_afterあり > action_resultあり > 新しい方
        （※ キーに action を含めるため、ここに来る重複は稀だが保険）
        """
        prev_done = prev.get('done') == 1
        new_done = new.get('done') == 1
        if prev_done != new_done:
            return new if new_done else prev

        prev_has_after = prev.get('state_after') is not None
        new_has_after = new.get('state_after') is not None
        if prev_has_after != new_has_after:
            return new if new_has_after else prev

        prev_has_action = prev.get('action_result') is not None
        new_has_action = new.get('action_result') is not None
        if prev_has_action != new_has_action:
            return new if new_has_action else prev

        return new  # デフォルト: 新しい方

    def _mk_key(entry):
        """重複統合キー: (game_id, turn, current_player, action_key)"""
        state = entry.get('state_after') or entry.get('state_before') or {}
        gid = entry.get('game_id') or _extract_game_id(entry.get('state_after'), entry.get('state_before'))
        turn = state.get('turn')
        current = state.get('current_player')

        ar = entry.get('action_result') or {}
        act = None
        if isinstance(ar, dict):
            act = ar.get('action') or ar.get('macro')

        # --- 変更: FINAL 行（ダミー終局）は current_player に依存させない ---
        if act is None and entry.get('done') == 1:
            act_key = ('FINAL',)
            return (gid, turn, None, act_key)

        # それ以外は従来通り
        if act is None:
            act_key = ('NOACT',)
        else:
            # list などをタプル化してハッシュ可能に
            act_key = tuple(act) if isinstance(act, (list, tuple)) else (str(act),)

        return (gid, turn, current, act_key)

    entries = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()

        # ---- (A) 素の1行JSON（top-level に state_before/state_after を持つ） ----
        if line.startswith('{') and '"state_before"' in line and '"state_after"' in line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and 'state_before' in obj and 'state_after' in obj:
                    sb = obj.get('state_before')
                    sa = obj.get('state_after')
                    ar = obj.get('action_result')
                    la = obj.get('legal_actions')
                    meta = obj.get('meta')

                    # reward/done は top-level 優先。なければ state_after→state_before の順で拾って昇格
                    reward = obj.get('reward')
                    done = obj.get('done')
                    if reward is None or done is None:
                        if isinstance(sa, dict):
                            r, d = _pop_reward_done(sa)
                            if reward is None:
                                reward = r
                            if done is None:
                                done = d
                        if (reward is None or done is None) and isinstance(sb, dict):
                            r, d = _pop_reward_done(sb)
                            if reward is None:
                                reward = r
                            if done is None:
                                done = d

                    entry = {
                        'game_id': _extract_game_id(sa, sb),
                        'state_before': sb,
                        'action_result': ar,
                        'state_after': sa,
                        'legal_actions': la,
                    }
                    if reward is not None:
                        entry['reward'] = reward
                    if done is not None:
                        entry['done'] = done
                    if meta is not None:
                        entry['meta'] = meta

                    entries.append(entry)
                    i += 1
                    continue
            except json.JSONDecodeError:
                pass  # 下の分岐で扱う

        # ---- (B) [STATE_BEFORE] ブロック ----
        if line.startswith('[STATE_BEFORE]'):
            state_before = line.replace('[STATE_BEFORE]', '').strip()

            action_result = None
            state_after = None
            legal_actions = None
            inline_records = []  # ブロック内で見つけた素JSONを蓄積

            # 次の [STATE_BEFORE] が出るまで探索（EOF まで）
            j = i + 1
            while j < n:
                nxt_raw = lines[j]
                nxt = nxt_raw.strip()

                if nxt.startswith('[STATE_BEFORE]'):
                    # 次ブロック開始 → 内側探索終了
                    break

                # 追加: ブロック内でも [LEGAL_ACTIONS] を拾う（フォールバック用）
                if nxt.startswith('[LEGAL_ACTIONS]') and legal_actions is None:
                    try:
                        legal_actions = json.loads(nxt.replace('[LEGAL_ACTIONS]', '').strip())
                    except Exception:
                        pass
                    j += 1
                    continue

                # 追加: ブロック内でも素の1行JSONを検出して entries に即追加
                if nxt.startswith('{') and '"state_before"' in nxt and '"state_after"' in nxt:
                    try:
                        obj = json.loads(nxt)
                        if isinstance(obj, dict) and 'state_before' in obj and 'state_after' in obj:
                            sb = obj.get('state_before')
                            sa = obj.get('state_after')
                            ar = obj.get('action_result')
                            la = obj.get('legal_actions')
                            meta = obj.get('meta')

                            reward = obj.get('reward')
                            done = obj.get('done')
                            if reward is None or done is None:
                                if isinstance(sa, dict):
                                    r, d = _pop_reward_done(sa)
                                    if reward is None:
                                        reward = r
                                    if done is None:
                                        done = d
                                if (reward is None or done is None) and isinstance(sb, dict):
                                    r, d = _pop_reward_done(sb)
                                    if reward is None:
                                        reward = r
                                    if done is None:
                                        done = d

                            entry = {
                                'game_id': _extract_game_id(sa, sb),
                                'state_before': sb,
                                'action_result': ar,
                                'state_after': sa,
                                'legal_actions': la,
                            }
                            if reward is not None:
                                entry['reward'] = reward
                            if done is not None:
                                entry['done'] = done
                            if meta is not None:
                                entry['meta'] = meta

                            inline_records.append(entry)
                    except Exception:
                        pass
                    j += 1
                    continue

                # 既存: [ACTION_RESULT] / [STATE_OBJ_AFTER]
                if nxt.startswith('[ACTION_RESULT]') and action_result is None:
                    action_result = nxt.replace('[ACTION_RESULT]', '').strip()
                    j += 1
                    continue

                if nxt.startswith('[STATE_OBJ_AFTER]') and state_after is None:
                    state_after = nxt.replace('[STATE_OBJ_AFTER]', '').strip()
                    j += 1
                    continue

                j += 1

            # --- 出力 ---
            if inline_records:
                # ブロック内で素JSONを拾えた → それを優先してそのまま追加
                entries.extend(inline_records)
            else:
                # フォールバック: [STATE_BEFORE] と後続の [ACTION_RESULT]/[STATE_OBJ_AFTER] から1件作る
                try:
                    sb = json.loads(state_before) if state_before else None
                    ar = json.loads(action_result) if action_result else None
                    sa = json.loads(state_after) if state_after else None

                    reward = None
                    done = None
                    if isinstance(sa, dict):
                        r, d = _pop_reward_done(sa)
                        reward = r if r is not None else reward
                        done = d if d is not None else done
                    if isinstance(sb, dict) and (reward is None or done is None):
                        r, d = _pop_reward_done(sb)
                        reward = r if r is not None else reward
                        done = d if d is not None else done

                    entry = {
                        'game_id': _extract_game_id(sa, sb),
                        'state_before': sb,
                        'action_result': ar,
                        'state_after': sa
                    }
                    # ブロック内で見つけた LEGAL_ACTIONS を付与（sa/sb には無いケース向け）
                    if legal_actions is not None:
                        entry['legal_actions'] = legal_actions

                    if reward is not None:
                        entry['reward'] = reward
                    if done is not None:
                        entry['done'] = done

                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"JSON解析エラー: {e}\nproblem line: {line}")

            # 消費した範囲をスキップ
            i = j
            continue

        # ---- (C) 単独の [STATE_OBJ_AFTER] 行 ----
        if line.startswith('[STATE_OBJ_AFTER]'):
            state_after = line.replace('[STATE_OBJ_AFTER]', '').strip()
            try:
                sa = json.loads(state_after)
                reward, done = _pop_reward_done(sa)

                entry = {
                    'game_id': _extract_game_id(sa),
                    'state_before': None,
                    'action_result': None,
                    'state_after': sa
                }
                if reward is not None:
                    entry['reward'] = reward
                if done is not None:
                    entry['done'] = done

                entries.append(entry)

            except Exception as e:
                print(f"[STATE_OBJ_AFTER] 単独行のJSON変換でエラー: {e}")

            i += 1
            continue

        # ---- その他の行はスキップ ----
        i += 1

    # ---- (D) 重複統合（同一 game_id / turn / current_player / action_key）----
    dedup = {}
    order = []  # 出現順を維持
    for e in entries:
        key = _mk_key(e)
        if key not in dedup:
            dedup[key] = e
            order.append(key)
        else:
            dedup[key] = _best_entry(dedup[key], e)

    result = [dedup[k] for k in order]
    return result


def save_to_single_line_json(entries, output_file_path: str):
    """
    エントリを1行のJSONファイルとして保存する
    """
    # ★ 追加: 保存前に変換ミス（-1）混入を検査
    try:
        assert_no_minus_one_in_entries(entries, context=output_file_path)
    except Exception as e:
        print(str(e))
        return

    # 全エントリを1つのオブジェクトとして保存（1行）
    data = {
        "entries": entries,
        "total_count": len(entries)
    }
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    
    print(f"{len(entries)}個のエントリを1行のJSONとして{output_file_path}に保存しました。")

def convert_entries_to_ids(entries, converter):
    """
    エントリのカード名をIDに変換し、-1が含まれていたら警告を表示し、その時点までで変換を中断
    """
    converted_entries = []
    for idx, entry in enumerate(entries):
        converted_entry = converter.convert_record(entry)
        converted_entries.append(converted_entry)  # ここで必ず追加

        if has_minus_one(converted_entry):
            print(f"\n⚠️ エントリ[{idx}]に変換ミス（-1）が含まれています。内容を確認してください。")
            print(json.dumps(converted_entry, ensure_ascii=False, indent=2))
            print(f"★ 変換処理を中断します。{idx}件までを書き込みます。")
            break   # ここでループを抜ける
    return converted_entries

def save_id_converted_json(entries, output_file_path: str):
    """
    ID変換されたエントリを1行のJSONファイルとして保存する
    """
    # ★ 追加: 保存前に変換ミス（-1）混入を検査
    try:
        assert_no_minus_one_in_entries(entries, context=output_file_path)
    except Exception as e:
        print(str(e))
        return

    # 全エントリを1つのオブジェクトとして保存（1行）
    data = {
        "entries": entries,
        "total_count": len(entries)
    }
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    
    print(f"{len(entries)}個のID変換済みエントリを1行のJSONとして{output_file_path}に保存しました。")

def _binom_wilson(k, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    margin = (z * ((phat*(1-phat)/n + (z*z)/(4*n*n)) ** 0.5)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def decide_first_player(player1, player2):
    # 人間プレイヤーがいる場合、どちらかに選ばせる
    if not player1.is_bot:
        print(f"{player1.name}、コイントス！『表:0』か『裏:1』を選んでください。")
        user_choice = input("0/1: ")
        coin = random.choice(["0", "1"])
        print(f"コイントスの結果: {coin}")
        if user_choice == coin:
            print(f"当たり！{player1.name}は先攻／後攻を選べます。")
            order = input("先攻なら 1、後攻なら 2 を入力: ")
            if order == "1":
                return player1, player2
            else:
                return player2, player1
        else:
            print(f"ハズレ！{player2.name}が先攻／後攻を選びます。")
            if player2.is_bot:
                first = random.choice([player2, player1])
                print(f"{player2.name}が{'先攻' if first == player2 else '後攻'}を選びました。")
                return (first, player1) if first == player2 else (player1, player2)
            else:
                order = input(f"{player2.name}、先攻なら 1、後攻なら 2 を入力: ")
                if order == "1":
                    return player2, player1
                else:
                    return player1, player2
    else:
        # 両方AIならランダム
        first = random.choice([player1, player2])
        print(f"コイントスで{first.name}が先攻です。")
        second = player2 if first == player1 else player1
        return first, second


def count_existing_episodes(jsonl_base_path, max_check=None):
    """
    ローテーション（_00001.jsonl など）を含めてユニーク game_id を数える
    """
    import glob, os, json

    root, ext = os.path.splitext(jsonl_base_path)
    # ベース本体 + ローテーション済みをすべて対象
    paths = [jsonl_base_path]
    if JSONL_ROTATE_LINES > 0:
        paths += sorted(glob.glob(f"{root}_*{ext}"))

    seen = set()
    count = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_check and i >= max_check:
                    break
                try:
                    entry = json.loads(line)
                    gid = None
                    if entry.get("state_before") and isinstance(entry["state_before"], dict):
                        gid = entry["state_before"].get("game_id")
                    elif entry.get("state_after") and isinstance(entry["state_after"], dict):
                        gid = entry["state_after"].get("game_id")
                    if gid and gid not in seen:
                        seen.add(gid)
                        count += 1
                except Exception:
                    continue
    return count

def audit_entries_basic(entries, converter: 'CardNameToIdConverter'):
    """
    各エントリの
      - legal_actions 空/欠損
      - 変換後 -1 混入
      - 選択行動が legal_actions に含まれているか（粗判定）
      - 行動タイプの頻度
    をざっくりチェックして dict で返す
    """
    bad_empty_legal = 0
    bad_minus_one   = 0
    bad_illegal_pick= 0
    act_type_hist   = Counter()

    for e in entries:
        la = e.get("legal_actions") or []
        ar = e.get("action_result") or {}
        act = ar.get("action")

        if not la:
            bad_empty_legal += 1

        # -1 監査（ID変換後で厳しめに見る）
        id_e = converter.convert_record(e)
        if has_minus_one(id_e):
            bad_minus_one += 1

        # 行動タイプ集計（5ints or 旧形式両対応）
        a_type = None
        if isinstance(act, list) and act:
            if isinstance(act[0], int):
                a_type = act[0]  # 5ints: [type, action_id, ...]
            else:
                a_type = converter.ACTION_TYPE.get(act[0], -1)
            act_type_hist[a_type] += 1

        # 「選択が合法か」の粗判定
        if la and act:
            in_legal = False
            if all(isinstance(x, list) and x and isinstance(x[0], int) for x in la) and isinstance(act[0], int):
                in_legal = act in la
            else:
                try:
                    act_head = act[0] if isinstance(act[0], str) else None
                    la_heads = [aa[0] if isinstance(aa, list) and aa else None for aa in la]
                    in_legal = act_head in la_heads
                except Exception:
                    pass
            if not in_legal:
                bad_illegal_pick += 1

    return {
        "empty_legal_blocks": bad_empty_legal,
        "minus_one_records": bad_minus_one,
        "illegal_picks": bad_illegal_pick,
        "act_type_hist": dict(act_type_hist),
    }


def print_turn_stats_from_raw_jsonl(path: str) -> None:
    """
    RAW_JSONL_PATH から 1ゲームあたりの決定ステップ数を集計してコンソールに出力する。
    [TURN_STATS] タグで [WINRATE]/[PROGRESS] と並ぶようにする。
    """
    from collections import Counter
    import os

    if not path:
        print("[TURN_STATS] RAW_JSONL_PATH is empty.")
        return
    if not os.path.exists(path):
        print(f"[TURN_STATS] file not found: {path}")
        return

    counts = Counter()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                gid = e.get("game_id")
                if gid is not None:
                    counts[gid] += 1
    except Exception as e:
        print(f"[TURN_STATS] failed to read {path}: {e}")
        return

    games = list(counts.values())
    games.sort()

    print(f"[TURN_STATS] games={len(games)}")
    if not games:
        return

    avg = sum(games) / len(games)
    min_turns = games[0]
    max_turns = games[-1]
    sample = games[:20]

    print(f"[TURN_STATS] min={min_turns} max={max_turns} avg={avg:.2f}")
    print(f"[TURN_STATS] sample_first20={sample}")


def print_pi_stats_from_raw_jsonl(path: str) -> None:
    """
    RAW_JSONL_PATH から、方策分布 π の「鋭さ」をざっくり集計して出力する。
    - 対象: pi と legal_actions を両方持っているレコード
    - 計算: max_prob, entropy(= -∑ p log p)
    """
    from collections import Counter
    import math
    import os

    if not path:
        print("[PI_STATS] RAW_JSONL_PATH is empty.")
        return
    if not os.path.exists(path):
        print(f"[PI_STATS] file not found: {path}")
        return

    n_samples = 0
    n_actions_list = []
    max_probs = []
    entropies = []
    n_actions_hist = Counter()

    def _get_pi(rec):
        if isinstance(rec, dict):
            if isinstance(rec.get("pi"), list):
                return rec["pi"]
            if isinstance(rec.get("pi_raw"), list):
                return rec["pi_raw"]
            if isinstance(rec.get("policy_pi"), list):
                return rec["policy_pi"]
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                la = rec.get("legal_actions")
                pi = _get_pi(rec)

                if not isinstance(la, list) or not la:
                    continue
                if not isinstance(pi, list) or len(pi) != len(la):
                    continue

                try:
                    probs = [float(x) for x in pi]
                except Exception:
                    continue
                s = sum(probs)
                if not (s > 0.0):
                    continue
                probs = [p / s for p in probs]

                max_p = max(probs)
                ent = 0.0
                for p in probs:
                    if p > 0.0:
                        ent -= p * math.log(p)
                k = len(probs)

                n_samples += 1
                n_actions_list.append(k)
                max_probs.append(max_p)
                entropies.append(ent)
                n_actions_hist[k] += 1
    except Exception as e:
        print(f"[PI_STATS] failed to read {path}: {e}")
        return

    print(f"[PI_STATS] samples={n_samples}")
    if not n_samples:
        return

    avg_k = sum(n_actions_list) / n_samples
    avg_max_p = sum(max_probs) / n_samples
    avg_ent = sum(entropies) / n_samples

    if avg_k > 0:
        ent_uniform = math.log(avg_k)
    else:
        ent_uniform = float("nan")

    print(f"[PI_STATS] avg_actions={avg_k:.3f}")
    print(f"[PI_STATS] avg_max_prob={avg_max_p:.4f}")
    print(f"[PI_STATS] avg_entropy={avg_ent:.4f} (uniform≈{ent_uniform:.4f})")
    common_k = n_actions_hist.most_common(10)
    print(f"[PI_STATS] n_actions_hist_top10={common_k}")

def print_end_reason_stats_from_ids_jsonl(path: str) -> None:
    """
    JSONL からゲーム終了理由ごとの件数を集計して出力する。
    end_reason / _end_reason / game_result.reason / meta 経由を優先的に見る。
    """
    import os
    from collections import Counter

    if not path:
        print("[END_REASON_STATS] path is empty.")
        return
    if not os.path.exists(path):
        print(f"[END_REASON_STATS] file not found: {path}")
        return

    reason_by_gid = {}
    winner_by_gid = {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                gid = rec.get("game_id")
                if gid is None:
                    continue
                gid = str(gid)

                # 1) トップレベル
                reason = rec.get("end_reason") or rec.get("_end_reason")

                # 2) game_result / result 直下
                if reason is None:
                    gr = rec.get("game_result") or rec.get("result") or {}
                    if isinstance(gr, dict):
                        reason = gr.get("reason") or gr.get("end_reason")

                # 3) meta 配下 (meta.end_reason / meta.game_result.reason 等)
                if reason is None:
                    meta = rec.get("meta")
                    if isinstance(meta, dict):
                        reason = meta.get("end_reason")
                        if reason is None:
                            gr2 = meta.get("game_result") or meta.get("result") or {}
                            if isinstance(gr2, dict):
                                reason = gr2.get("reason") or gr2.get("end_reason")

                if not reason:
                    reason = "UNKNOWN"

                reason_by_gid[gid] = str(reason).upper()

                # 勝者も同様に取得（winner / _winner / game_result.winner / meta 経由）
                winner = rec.get("winner") or rec.get("_winner")

                if winner is None:
                    grw = rec.get("game_result") or rec.get("result") or {}
                    if isinstance(grw, dict):
                        winner = grw.get("winner")

                if winner is None:
                    meta = rec.get("meta")
                    if isinstance(meta, dict):
                        if winner is None:
                            winner = meta.get("winner")
                        if winner is None:
                            grw2 = meta.get("game_result") or meta.get("result") or {}
                            if isinstance(grw2, dict):
                                winner = grw2.get("winner")

                if not winner:
                    winner = "UNKNOWN"

                # 小文字・大文字揺れを吸収するため lower に寄せておく
                winner_by_gid[gid] = str(winner).lower()
    except Exception as e:
        print(f"[END_REASON_STATS] failed to read {path}: {e}")
        return

    hist = Counter(reason_by_gid.values())
    total = sum(hist.values())
    print(f"[END_REASON_STATS] games={total}")
    for k in sorted(hist.keys()):
        print(f"[END_REASON_STATS] {k}={hist[k]}")

    # 終了理由ごとの勝敗集計
    winrate_by_reason = {}
    for gid, reason in reason_by_gid.items():
        w = winner_by_gid.get(gid, "unknown").lower()
        bucket = winrate_by_reason.setdefault(
            reason,
            {"games": 0, "p1": 0, "p2": 0, "draw": 0, "unknown": 0},
        )
        bucket["games"] += 1
        if w == "p1":
            bucket["p1"] += 1
        elif w == "p2":
            bucket["p2"] += 1
        elif w == "draw":
            bucket["draw"] += 1
        else:
            bucket["unknown"] += 1

    total_games_winrate = sum(b["games"] for b in winrate_by_reason.values())
    print(f"[END_REASON_WINRATE] games={total_games_winrate}")
    for reason in sorted(winrate_by_reason.keys()):
        b = winrate_by_reason[reason]
        print(
            "[END_REASON_WINRATE] {reason}: games={games} p1={p1} p2={p2} draw={draw} unknown={unknown}".format(
                reason=reason,
                games=b.get("games", 0),
                p1=b.get("p1", 0),
                p2=b.get("p2", 0),
                draw=b.get("draw", 0),
                unknown=b.get("unknown", 0),
            )
        )

    # BASICS_OUT + PRIZE_OUT 限定の P1 勝率を計算して出力
    non_deck_reasons = ["BASICS_OUT", "PRIZE_OUT"]

    total_non_deck_games = 0
    total_non_deck_p1 = 0
    total_non_deck_p2 = 0
    total_non_deck_draw = 0
    total_non_deck_unknown = 0

    for r in non_deck_reasons:
        b = winrate_by_reason.get(r, {})
        total_non_deck_games += b.get("games", 0)
        total_non_deck_p1 += b.get("p1", 0)
        total_non_deck_p2 += b.get("p2", 0)
        total_non_deck_draw += b.get("draw", 0)
        total_non_deck_unknown += b.get("unknown", 0)

    print("[END_REASON_NONDECK_WINRATE] reasons=BASICS_OUT+PRIZE_OUT")
    print(
        "[END_REASON_NONDECK_WINRATE] games={games} p1={p1} p2={p2} draw={draw} unknown={unknown}".format(
            games=total_non_deck_games,
            p1=total_non_deck_p1,
            p2=total_non_deck_p2,
            draw=total_non_deck_draw,
            unknown=total_non_deck_unknown,
        )
    )

    if total_non_deck_games > 0:
        p1_rate = 100.0 * total_non_deck_p1 / total_non_deck_games
        p2_rate = 100.0 * total_non_deck_p2 / total_non_deck_games
    else:
        p1_rate = 0.0
        p2_rate = 0.0

    print(
        "[END_REASON_NONDECK_WINRATE] p1_winrate={p1:.2f}% p2_winrate={p2:.2f}% (non-deck games only)".format(
            p1=p1_rate,
            p2=p2_rate,
        )
    )

    # === OnlineMixedPolicy の選択ソース統計（この関数の最後のアンカー） ===
    try:
        from pokepocketsim.policy.online_mixed_policy import OnlineMixedPolicy

        def _dump_policy_stats(label: str, pol) -> None:
            if pol is None:
                return
            if not isinstance(pol, OnlineMixedPolicy):
                return
            total = getattr(pol, "stats_total", 0) or 0
            model = getattr(pol, "stats_from_model", 0) or 0
            fallback = getattr(pol, "stats_from_fallback", 0) or 0
            rnd = getattr(pol, "stats_from_random", 0) or 0
            err = getattr(pol, "stats_errors", 0) or 0

            if total > 0:
                model_r = model / total * 100.0
                fallback_r = fallback / total * 100.0
                rnd_r = rnd / total * 100.0
            else:
                model_r = fallback_r = rnd_r = 0.0

            print(
                f"[POLICY_STATS] {label}: total={total} "
                f"model={model}({model_r:.2f}%) "
                f"fallback={fallback}({fallback_r:.2f}%) "
                f"random={rnd}({rnd_r:.2f}%) errors={err}"
            )

        # POLICY_STATS は複数回呼ばれても 1 回だけ出力する
        if getattr(print_end_reason_stats_from_ids_jsonl, "_policy_stats_dumped", False):
            return
        setattr(print_end_reason_stats_from_ids_jsonl, "_policy_stats_dumped", True)

        # グローバル名前空間から policy_p1 / policy_p2 をゆるく取得
        g = globals()
        pol_p1 = g.get("policy_p1") or g.get("p1_policy") or g.get("pol_p1")
        pol_p2 = g.get("policy_p2") or g.get("p2_policy") or g.get("pol_p2")

        _dump_policy_stats("P1", pol_p1)
        _dump_policy_stats("P2", pol_p2)

    except Exception as _e:
        # 統計出力で落ちないようにしておく
        print(f"[POLICY_STATS] dump skipped due to error: {_e!r}")


def _analyze_end_reason_and_winner(path: str):
    """
    JSONL（通常は RAW_JSONL_PATH）を走査して、
    game_id ごとに最終的な end_reason / winner を集計し、
    「理由別 × winner 別」の件数を dict で返す。
    """
    import os
    from collections import defaultdict

    try:
        if not path or not os.path.exists(path):
            return {}

        try:
            import orjson as _json
            _loads = _json.loads
            binary = True
        except Exception:
            import json as _json
            _loads = _json.loads
            binary = False

        games = {}

        if binary:
            f_open_args = {"mode": "rb"}
        else:
            f_open_args = {"mode": "r", "encoding": "utf-8"}

        with open(path, **f_open_args) as f:
            for line in f:
                if binary:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _loads(line)
                    except Exception:
                        continue
                else:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _loads(line)
                    except Exception:
                        continue

                if not isinstance(rec, dict):
                    continue

                gid = rec.get("game_id") or rec.get("match_id") or rec.get("game_uuid")
                if gid is None:
                    meta = rec.get("meta")
                    if isinstance(meta, dict):
                        gid = meta.get("game_id") or meta.get("match_id")
                if gid is None:
                    sb = rec.get("state_before")
                    if isinstance(sb, dict):
                        gid = sb.get("game_id") or sb.get("match_id")
                if gid is None:
                    continue
                gid = str(gid)

                winner = rec.get("winner")
                if winner is None:
                    meta = rec.get("meta")
                    if isinstance(meta, dict):
                        winner = meta.get("winner")
                if winner is None:
                    gr = rec.get("game_result")
                    if isinstance(gr, dict):
                        winner = gr.get("winner")
                if winner is None:
                    winner = "unknown"

                reason = rec.get("end_reason") or rec.get("_end_reason")
                if reason is None:
                    meta = rec.get("meta")
                    if isinstance(meta, dict):
                        reason = meta.get("end_reason") or meta.get("result_reason")
                        if reason is None:
                            gr2 = meta.get("game_result") or meta.get("result") or {}
                            if isinstance(gr2, dict):
                                reason = gr2.get("reason") or gr2.get("end_reason")
                if reason is None:
                    gr = rec.get("game_result")
                    if isinstance(gr, dict):
                        reason = gr.get("end_reason") or gr.get("reason") or gr.get("result_reason")
                if reason is None:
                    reason = "UNKNOWN"

                games[gid] = {
                    "reason": str(reason).upper(),
                    "winner": str(winner).lower(),
                }

        stats = {}
        for info in games.values():
            reason = info.get("reason") or "UNKNOWN"
            winner = info.get("winner") or "unknown"

            st = stats.setdefault(reason, {
                "games": 0,
                "p1": 0,
                "p2": 0,
                "draw": 0,
                "unknown": 0,
            })
            st["games"] += 1

            if winner == "p1":
                st["p1"] += 1
            elif winner == "p2":
                st["p2"] += 1
            elif winner == "draw":
                st["draw"] += 1
            else:
                st["unknown"] += 1

        return stats
    except Exception:
        return {}

if __name__ == "__main__":
    # --- unified gamelog header/footer (stdout/stderr tee is handled by console_tee only) ---
    _run_game_id = os.getenv("AI_VS_AI_RUN_GAME_ID", "").strip()
    if not _run_game_id:
        _run_game_id = _RUN_GAME_ID
        os.environ["AI_VS_AI_RUN_GAME_ID"] = _run_game_id

    try:
        _gamelog_path = os.getenv("AI_VS_AI_GAMELOG_PATH", "").strip()
        if _gamelog_path:
            os.environ["AI_VS_AI_GAMELOG_ACTIVE"] = "1"

        try:
            _self = os.path.normcase(os.path.normpath(os.path.abspath(__file__)))
        except Exception:
            _self = str(__file__)
        try:
            _argv0 = os.path.normcase(os.path.normpath(os.path.abspath(sys.argv[0])))
        except Exception:
            _argv0 = str(sys.argv[0])
        print(f"[SELF] file={_self} argv0={_argv0} pid={os.getpid()}")

        from datetime import datetime
        _ts0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[LOG] ===== game log begin ===== ts={_ts0} game_id={_run_game_id}")
        if _gamelog_path:
            print(f"[LOG] path={os.path.abspath(_gamelog_path)}")
        print(f"[LOG] pid={os.getpid()} python={sys.version.split()[0]}")
        print(f"[LOG] cwd={os.getcwd()}")
        try:
            print(f"[LOG] argv={sys.argv}")
        except Exception:
            pass
        print(f"[LOG] ===========================")

        def _gamelog_end():
            try:
                _ts1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[LOG] ===== game log end ===== ts={_ts1} status=OK")
            except Exception:
                pass

        atexit.register(_gamelog_end)
    except Exception:
        pass

    # ここだけが実行エントリになる（import時に動かない）

    # ★ AlphaZeroMCTSPolicy の読み込み確認（AZ_MCTS_DEBUG_IMPORT=1 のときのみ）
    if os.getenv("AZ_MCTS_DEBUG_IMPORT") == "1":
        try:
            from pokepocketsim.policy.az_mcts_policy import AlphaZeroMCTSPolicy
            debug_dir = MODEL_DIR_P1 or "."
            print(f"[AZ_MCTS_DEBUG] model_dir(P1) = {debug_dir}")
            pol_dbg = AlphaZeroMCTSPolicy(model_dir=debug_dir)
            print(
                f"[AZ_MCTS_DEBUG] type={type(pol_dbg)} "
                f"obs_dim={getattr(pol_dbg, 'obs_dim', None)} "
                f"cand_dim={getattr(pol_dbg, 'cand_dim', None)} "
                f"has_model={getattr(pol_dbg, 'model', None) is not None}"
            )
        except Exception as e:
            print(f"[AZ_MCTS_DEBUG][WARN] AlphaZeroMCTSPolicy import/load failed: {e}")

    to_run = NUM_MATCHES

    # モデル対戦が含まれるかで自動選択
    if str(P1_POLICY).lower() == "model" or str(P2_POLICY).lower() == "model":
        stats = run_model_matches_multiprocess(to_run)
    else:
        stats = run_random_matches_multiprocess(to_run)

    print("[DONE]", stats)

    # --- 追加: 試合数ベースの勝率表示 ---
    try:
        if isinstance(stats, dict):
            w1 = int(stats.get("wins_p1", 0))
            w2 = int(stats.get("wins_p2", 0))
            wd = int(stats.get("wins_draw", 0))
            wu = int(stats.get("wins_unknown", 0))

            # deck-out 除外後に「ログとして残った試合数」
            denom = int(stats.get("logged_matches", stats.get("num_matches", 0)))
            if denom <= 0:
                denom = w1 + w2 + wd + wu

            if denom > 0:
                wr1 = (w1 / denom) * 100.0
                wr2 = (w2 / denom) * 100.0
                wrd = (wd / denom) * 100.0
                print(f"[WINRATE] p1={wr1:.2f}% p2={wr2:.2f}% draw={wrd:.2f}% (matches={denom})")
            else:
                print("[WINRATE] no matches logged")
    except Exception:
        pass

    try:
        if isinstance(stats, dict):
            pm_pub  = int(stats.get("pi_missing_public", 0))
            pm_priv = int(stats.get("pi_missing_private", 0))
            print(f"[PI_RAW_ABSENT] public={pm_pub} private={pm_priv}")  # 元ログに raw π が無かった／使えなかった件数
    except Exception:
        pass

    # --- 追加: 累積「試合数」で進捗を表示（追記方式向け） ---
    def _count_unique_game_ids(_path):
        """JSONL を走査してユニークな game_id 件数（=試合数）を数える"""
        try:
            if not _path or not os.path.exists(_path):
                return 0

            # orjson があれば使う（高速）。無ければ json。
            try:
                import orjson as _json
                _loads = _json.loads
            except Exception:
                import json as _json
                _loads = _json.loads

            gids = set()
            with open(_path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = _loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue

                    # top-level / meta / state_before などから game_id を拾う
                    gid = rec.get("game_id") or rec.get("match_id") or rec.get("game_uuid")
                    if gid is None:
                        meta = rec.get("meta")
                        if isinstance(meta, dict):
                            gid = meta.get("game_id") or meta.get("match_id")
                    if gid is None:
                        sb = rec.get("state_before")
                        if isinstance(sb, dict):
                            gid = sb.get("game_id") or sb.get("match_id")

                    if gid is None:
                        continue

                    # int/str 両対応で正規化
                    gids.add(str(gid))

            return len(gids)
        except Exception:
            return 0

    cur_raw  = _count_unique_game_ids(RAW_JSONL_PATH)
    cur_ids  = _count_unique_game_ids(IDS_JSONL_PATH)
    cur_priv = _count_unique_game_ids(PRIVATE_IDS_JSON_PATH)
    goal     = int(TARGET_EPISODES)

    print(f"[PROGRESS] cumulative games raw={cur_raw} ids={cur_ids} private_ids={cur_priv} / target={goal}")

    # ids と private_ids の両方に揃っている試合数を有効件数として判定
    cur_effective = min(cur_ids, cur_priv) if (cur_ids > 0 and cur_priv > 0) else max(cur_ids, cur_priv)

    if cur_effective >= goal:
        print(f"[PROGRESS] target reached: effective_games={cur_effective} >= {goal}")
    else:
        remain = goal - cur_effective
        print(f"[PROGRESS] remaining games: {remain} (effective_games={cur_effective} / target={goal})")

    print_turn_stats_from_raw_jsonl(RAW_JSONL_PATH)
    print_pi_stats_from_raw_jsonl(IDS_JSONL_PATH)
    print_end_reason_stats_from_ids_jsonl(RAW_JSONL_PATH)

    try:
        erw = _analyze_end_reason_and_winner(RAW_JSONL_PATH)
        if isinstance(erw, dict) and erw:
            total_games = sum(st.get("games", 0) for st in erw.values())
            print(f"[END_REASON_WINRATE] games={total_games}")
            for reason, st in sorted(erw.items()):
                r = reason or "UNKNOWN"
                g = int(st.get("games", 0))
                w1 = int(st.get("p1", 0))
                w2 = int(st.get("p2", 0))
                wd = int(st.get("draw", 0))
                wu = int(st.get("unknown", 0))
                print(
                    f"[END_REASON_WINRATE] {r}: games={g} "
                    f"p1={w1} p2={w2} draw={wd} unknown={wu}"
                )
        else:
            print("[END_REASON_WINRATE] no data")
    except Exception as e:
        print(f"[END_REASON_WINRATE][WARN] failed to analyze ({e})")
