# -*- coding: utf-8 -*-
"""
config.py : ai vs ai.py の設定項目を集約
--------------------------------------------------------------------------------
このファイルは ai vs ai.py から「編集すべき設定値」を分離したもの。

- ai vs ai.py は config.py から設定値を import して実行する
- worker.py は __main__（ai vs ai.py / __mp_main__）のグローバルを参照するため、
  ai vs ai.py 側では config の設定値をグローバル名として import している
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

__all__ = [
    "ALWAYS_OUTPUT_BOTH_IDS",
    "AZ_ACTION_ENCODER_FN32",
    "AZ_ACTION_ENCODER_FN32_SPEC",
    "AZ_MCTS_NUM_SIMULATIONS",
    "BATCH_FLUSH_INTERVAL",
    "CPU_WORKERS_MCC",
    "CPU_WORKERS_RANDOM",
    "DECK1_KEY",
    "DECK2_KEY",
    "DELETE_PER_MATCH_ML_AFTER_PRIVATE_IDS",
    "FULL_VEC_VERSION",
    "GAMELOG_DIR",
    "IDS_JSONL_PATH",
    "INTERMEDIATE_DECK_COUNT_WEIGHT",
    "INTERMEDIATE_MAX_TURN_PENALTY",
    "INTERMEDIATE_PRIZE_WEIGHT",
    "INTERMEDIATE_TURN_PENALTY_PER_TURN",
    "INTERMEDIATE_TURN_PENALTY_START",
    "IO_BUFFERING_BYTES",
    "JSONL_ROTATE_LINES",
    "KEEP_MATCH_LOG_FILE",
    "KEEP_MATCH_ML_FILE",
    "LOG_DEBUG_DETAIL",
    "LOG_FILE",
    "LOG_FULL_INFO",
    "MATCH_SLEEP_SEC",
    "MCC_EVERY",
    "MCC_SAMPLES",
    "MCC_TOP_K",
    "NUM_MATCHES",
    "P1_POLICY",
    "P2_POLICY",
    "PER_MATCH_LOG_DIR",
    "PREFER_LEARNABLE",
    "PRIVATE_IDS_JSON_PATH",
    "RAW_JSONL_PATH",
    "REQUIRE_LEARNABLE",
    "REWARD_SHAPING_GAMMA",
    "SELFPLAY_ALPHAZERO_MODE",
    "SKIP_DECKOUT_LOGGING",
    "TARGET_EPISODES",
    "TORCH_THREADS_PER_PROC",
    "USE_MCC",
    "USE_MCTS_POLICY",
    "USE_REWARD_SHAPING",
    "VALUE_NET_MODEL_DIR",
    "VALUE_NET_MODEL_FILENAME",
    "VALUE_NET_MODEL_PATH",
    "WRITER_FLUSH_SEC",
    "WRITER_FSYNC",
    "Z_PROGRESS_MAX",
]

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

# === 必須: cand_dim!=5 のモデルで使う action encoder（fn32）をコードで固定 ===
# policy_factory.py はここ（config.py）だけを唯一の正として参照し、取れなければ即エラー停止します。
# ダミー注入・自動探索は禁止なので、未実装のまま実行すると必ず停止します（意図通り）。
AZ_ACTION_ENCODER_PROBE_ACTION_ID = 0

def encode_action_32d(action_id: int) -> list[float]:
    """
    action_id -> 32次元ベクトル（list[float], len=32）を返す関数をここに実装してください。
    未実装のまま動かすと policy_factory の注入直後 probe で必ず止まります（ダミーで進みません）。
    """
    raise NotImplementedError(
        "config.py: encode_action_32d(action_id) を実装してください（list[float] len=32 を必ず返す）"
    )

# policy_factory 側が最優先で読む “callable 本体”
AZ_ACTION_ENCODER_FN32 = encode_action_32d

# （任意）SPEC を使わないので空でOK。設定が残っていても callable 優先にします。
AZ_ACTION_ENCODER_FN32_SPEC = ""

# どの方策を使うか
#   "az_mcts"    : AlphaZero 方式。MCTS内部でモデルを使い、最終手はMCTSの訪問回数から選ぶ
#   "model_only" : MCTSを使わず、モデルのポリシー出力だけで手を選ぶ
#   "random"     : ポリシーを使わず、合法手から一様ランダムに選ぶ
#   "online_mix" : 1手ごとにモデルとMCTSのポリシーを混合して手を選ぶ（OnlineMixedPolicy）
P1_POLICY = "online_mix"
P2_POLICY = "online_mix"

# --- POLICY TRACE（ai vs ai 起動時に自動アクティブ） ---
os.environ.setdefault("POLICY_TRACE", "1")


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
WRITER_FSYNC         = bool(int(os.getenv("WRITER_FSYNC", "0")))    # 1=flush毎にfsync（重いが安全）
WRITER_FLUSH_SEC     = float(os.getenv("WRITER_FLUSH_SEC", "0.0"))  # 0=無効（秒）/ >0=定期flush

# 並列数（用途別）: MCC用/ランダム用
CPU_WORKERS_MCC    = 6   # MCC有効時の既定並列数
MCC_EVERY       = int(os.getenv("MCC_EVERY", "1"))
CPU_WORKERS_RANDOM = 7   # ランダム用の既定並列数（0=自動にしたければここを変える）
TORCH_THREADS_PER_PROC = 1



# === learnable を強制/優先するためのフラグ ===
PREFER_LEARNABLE = True
REQUIRE_LEARNABLE = os.getenv("REQUIRE_LEARNABLE", "0") == "1"

# 入力記録リスト（必要なら）

LOG_FILE = "ai_vs_ai_match.log"  # AI対戦ログ（任意）
RAW_JSONL_PATH  = r"D:\date\ai_vs_ai_match_all.jsonl"       # 生JSONL
IDS_JSONL_PATH  = r"D:\date\ai_vs_ai_match_all_ids.jsonl"   # ID変換済みJSONL
PRIVATE_IDS_JSON_PATH = r"D:\date\ai_vs_ai_match_all_private_ids.jsonl"
GAMELOG_DIR = os.path.dirname(RAW_JSONL_PATH)


