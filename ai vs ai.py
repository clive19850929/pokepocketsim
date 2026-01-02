# jsonファイルに変換した時は[-1]など変換にミスした選択肢が無いか確認する

"""
================================================================================
自己対戦（Self-Play）実行系 + 学習ログ生成：全体まとめ（関係整理版）
--------------------------------------------------------------------------------
このプロジェクトは大きく「自己対戦を回してログを作る実行系」と、
「ログから学習してモデルを更新する学習系」の循環で構成される。

  (実行: Self-Play) → (ログ生成: raw / ids / private_ids) → (学習: Phase B/C/D)
  → (生成モデル/learnable更新) → (次のSelf-Playへ反映)

================================================================================
1. 実行系（Self-Play）の中心：ai vs ai.py の責務
--------------------------------------------------------------------------------
ai vs ai.py は “統合（オーケストレーター）” であり、主に以下を担う。

  1) policy 構築（per-process / worker内で遅延ロード）
     - マルチプロセス前提のため、各 worker 内で build_policy() を呼び
       worker専用 policy を生成する（プロセス間共有を避ける）。
     - online_mix の概念:
         main_policy  : AlphaZeroMCTSPolicy（モデル推論 + MCTSで π を生成）
         Q側補助      : PhaseD-Q（d3rlpy CQL learnable による Q(s,a) 評価）
       混合（λ/温度/topk/モデルパス等）は online_mix 実装側で管理する。

     - 重要: MCTS の探索中など「重い online_mix を回すと破綻し得る局面」では、
       Player.select_action のガードにより軽量な経路へ逃がす分岐があり得る
       （探索内部で過剰に推論/混合を回さないための安全弁）。

  2) ログ統合（1ゲーム=1単位 / 構造化ログ優先）
     - stdout/stderr を {GAMELOG_DIR}/{game_id}.log に集約（GameLogContext）。
     - ML用の構造化ログは .ml.jsonl を優先し、解析は
         .ml.jsonl → 取得不能時のみ .log
       の順でフォールバックする。
     - “残骸誤読”防止のため .ml.jsonl は毎試合必ず空ファイルに初期化して書き始める。

  3) 行動決定ログの「試合中」可視化（DECIDE pending → 行動直前flush）
     - Player.select_action 側では [DECIDE_PRE]/[DECIDE_POST]/[DECIDE_DIFF] を
       その場で出力せず pending に積む。
     - Player.act_and_regather_actions 冒頭で pending を必ず flush してから
       “行動行” を出す。
       → 「試合中にモデル/混合判断が呼ばれている」をログ上で定量確認できる。

  4) 終局 SUMMARY を “1ゲーム1回だけ” に制御
     - 終局時に policy 側の log_summary / on_game_end 等を探索して呼ぶが、
       match._policy_summary_logged により 1ゲーム1回に固定する。

  5) 学習データ(JSONL)生成（worker → writer のバッチ集約）
     - worker は raw / ids / private_ids を batch として Queue に送る。
     - writer が統合して 1本の JSONL を出力（flush/fsync など重い I/O を集中）。

================================================================================
2. ログ（学習データ）の種類と用途
--------------------------------------------------------------------------------
Self-Play が出力する学習用ログは “役割別に3系統” に分かれる。

  A) raw
     - 人間可読寄りの構造化ログ。
     - LOG_FULL_INFO=True なら完全情報、False なら公開版（privates 除去）。
     - end_reason（PRIZE_OUT / BASICS_OUT / DECK_OUT ...）を併記。
     - 1ゲーム1行の game_summary（record_type="game_summary"）を raw に追記可能。

  B) ids（公開情報ベースの整数ID化）
     - 公開情報（部分観測）を主軸に学習へ載せる主戦場。
     - pi は “元ログに raw π があれば優先”、無ければ onehot 等で補完。
     - action_candidates_vec（例: 32d）を legal_actions から付与可能。
     - obs_vec（公開状態特徴量）を付与可能（obs_vector などを利用）。

  C) private_ids（完全情報ベースの整数ID化）
     - keep_private=True の完全情報ID化。
     - obs_full_vec 等、後段学習（Phase D）に必要な “完全情報寄り特徴量” を付与可能。

  例外: DECK_OUT の扱い（容量/品質対策）
     - SKIP_DECKOUT_LOGGING=True の場合、デッキアウト判定を “未加工全行” で先に行い、
       該当試合は early-continue で writer に流さない（学習品質と容量の両面対策）。

================================================================================
3. 主要ファイルの役割（依存方向が分かる整理）
--------------------------------------------------------------------------------
[ 実行・統合（オーケストレーション） ]
  - ai vs ai.py       : 実行入口。プロセス起動/統合/集計/デバッグ制御/最終出力方針。
  - worker.py         : 1プロセスで試合を回す本体。Match/Player を作り play_match() を実行。
  - writer.py         : 集中 writer。複数 worker の batch を統合して JSONL を出力。
  - config.py         : 実行設定（フラグ/パス/パラメータ）の唯一の集約点。

[ 方策（policy） ]
  - policy_factory.py : build_policy() の集約。online_mix / az_mcts / random 等の組み立て。
  - az_mcts_policy.py : AlphaZeroMCTSPolicy 本体（モデル推論→MCTS→π生成）。
  - phaseD_q.py       : PhaseD-Q（CQL learnable）の lazy load / Q評価 / π混合。
  - phased_q_mixer.py : 旧/分離案の置き場候補（現行未使用の可能性あり）。

[ 特徴量（状態表現） ]
  - obs_vector.py     : 公開状態→obs_vec（partial）生成。
                        set_card_id2idx(card_id2idx) で語彙注入後に利用。
  - legal_actions.py  : legal_actions の抽出・整形、候補ベクトル化（action_candidates_vec 等）。
                        ai vs ai.py から import して __main__ に再公開（worker が参照する設計のため）。
  - policy/state_encoder.py / policy/action_encoding.py :
                        policy内部の別経路エンコード、候補ベクトル化など。

[ シミュレータ（ゲーム進行） ]
  - match.py / player.py / action.py : 進行/合法手/行動実行/勝敗/ログ出力の核。
  - battle_logger.py                : 通常ログ + 構造化ログ（.ml.jsonl）の土台。
  - action_space.py                 : 行動空間定義、ID化、候補ベクトル化の基盤。

[ MCC / リワードシェーピング（Phase C の核） ]
  - my_mcc_sampler.py / mcc_ctx.py : MCC（隠れ情報補完サンプリング）実装・統計。
  - reward_shaping.py              : PBRS（Potential Φ(s)）設計、shaped_r 付与の集約候補。

================================================================================
4. MCC（Monte-Carlo Completion）の重要注意点（運用上の落とし穴）
--------------------------------------------------------------------------------
  - USE_MCC=True でも calls=0 になる場合がある。
    典型原因は MCC 呼び出しフックが use_reward_shaping 判定に縛られており、
    use_mcc=True だけでは reward_shaping 側が実行されないケース。

  - Self-Play で MCC を実際に回すには、Match/Player 側のフック条件を
        use_reward_shaping OR use_mcc
    に統一している必要がある。

  - MCC 呼び出し回数は mcc_debug_snapshot().total_calls の差分で計測し、
    [MCC_CALLS] game_id=... calls=...
    のように game_id 単位で確認する。

================================================================================
5. Phase B / C / D の位置づけ（ログ→学習→モデル更新）
--------------------------------------------------------------------------------
  Phase B（教師あり: PV）
    - 入力: ids ログ（公開情報ベース）
    - 出力: PVモデル（例: selfplay_supervised_pv_gen000.pt）
    - 目的: 盤面→(policy,value) を安定化し、MCTS の基盤にする。

  Phase D（CQL: Q）
    - 入力: private_ids ログから構築した RL dataset（npz）
    - 出力: CQL learnable（例: learnable_phaseD_cql.d3）
    - 目的: PhaseD-Q が online_mix の “Q側” として参照する中核。

  Phase C（橋渡し: PBRS + MCC）
    - 目的1: DECK_OUT に流れやすい自己対戦を抑制し「勝ちに行く短期目的」を薄く混ぜる。
    - 目的2: 部分観測でも安定して学べる表現・ラベルを作り、B と D のギャップを埋める。
    - 目的3: 隠れ情報を覗かず MCC で “それっぽい完全情報候補” を K 個サンプルし学習を安定化。

================================================================================
6. 動作確認の最低ライン（ログでチェック）
--------------------------------------------------------------------------------
  - 起動直後:
      [POLICY_SPEC] に online_mix / USE_MCTS_POLICY 等が出る
      → 実行設定と policy 構築が合っている証拠

  - 試合中（行動直前）:
      [DECIDE_PRE]/[DECIDE_POST]/[DECIDE_DIFF] が行動行の直前に出る
      → モデル/混合判断が “実際の行動” と同期している証拠

  - 終局:
      [PhaseD-Q][SUMMARY] が 1ゲーム1回だけ出る
      → once 制御が効いている証拠

  - MCC:
      [MCC_CALLS] ... calls>0 が理想
      calls=0 の場合は “フック条件（use_reward_shaping縛り等）” を疑う

================================================================================
7. フロー図（Self-Play → ログ → 学習 → 次のSelf-Play）
--------------------------------------------------------------------------------

  ┌───────────────────────────┐
  │         ai vs ai.py       │
  │ (orchestrator / launcher) │
  └───────────────┬───────────┘
                  │ spawn
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌───────▼────────┐
│    worker.py   │  │    writer.py   │
│ (play matches) │  │ (merge JSONL)  │
└───────┬────────┘  └───────┬────────┘
        │                    │
        │ per match logs     │ merged outputs
        │ (.ml.jsonl/.log)   │ (all games)
        │                    │
        │  batches (raw/ids/private_ids) via Queue
        └──────────────┬───────────────────┘
                       │
                       ▼
         ┌───────────────────────────┐
         │        Output Logs        │
         │  raw / ids / private_ids  │
         └──────────────┬────────────┘
                        │
                        ▼
     ┌───────────────────────────┐
     │        Phase B (PV)       │
     │ ids → supervised dataset  │
     │ → PV model (.pt)          │
     └──────────────┬────────────┘
                    │ supplies PV for MCTS
                    ▼
     ┌───────────────────────────┐
     │  az_mcts_policy.py (MCTS) │
     │  PV inference → π (MCTS)  │
     └──────────────┬────────────┘
                    │ mixes with Q
                    ▼
     ┌───────────────────────────┐
     │       Phase D (CQL)       │
     │ private_ids → RL dataset  │
     │ → learnable (.d3)         │
     └──────────────┬────────────┘
                    │ used by PhaseD-Q
                    ▼
     ┌───────────────────────────┐
     │        phaseD_q.py        │
     │   Q(s,a) eval + π mixing  │
     └──────────────┬────────────┘
                    │ (online_mix)
                    └───────→ back to Self-Play

  [Phase C: PBRS + MCC] は Self-Play を壊さず補強する “橋渡し”。
  - MCC: my_mcc_sampler.py / mcc_ctx.py
  - PBRS: reward_shaping.py
  - 注意: MCC フック条件が use_reward_shaping に縛られると calls=0 になり得るため、
          Match/Player 側で (use_reward_shaping OR use_mcc) の統一が前提。

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
# ====== 設定項目（config.py を編集：このファイルでは直書きしない）==========

# NOTE:
# - Windows spawn / worker.py が __main__（=本ファイル）のグローバルを参照するため、
#   config.py の設定値はこのモジューバル名として import しておく。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR and _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import (
    ALWAYS_OUTPUT_BOTH_IDS,
    AZ_MCTS_NUM_SIMULATIONS,
    BATCH_FLUSH_INTERVAL,
    CPU_WORKERS_MCC,
    CPU_WORKERS_RANDOM,
    DECK1_KEY,
    DECK2_KEY,
    DELETE_PER_MATCH_ML_AFTER_PRIVATE_IDS,
    FULL_VEC_VERSION,
    GAMELOG_DIR,
    IDS_JSONL_PATH,
    INTERMEDIATE_DECK_COUNT_WEIGHT,
    INTERMEDIATE_MAX_TURN_PENALTY,
    INTERMEDIATE_PRIZE_WEIGHT,
    INTERMEDIATE_TURN_PENALTY_PER_TURN,
    INTERMEDIATE_TURN_PENALTY_START,
    IO_BUFFERING_BYTES,
    JSONL_ROTATE_LINES,
    KEEP_MATCH_LOG_FILE,
    KEEP_MATCH_ML_FILE,
    LOG_DEBUG_DETAIL,
    LOG_FILE,
    LOG_FULL_INFO,
    MATCH_SLEEP_SEC,
    MCC_EVERY,
    MCC_SAMPLES,
    MCC_TOP_K,
    NUM_MATCHES,
    P1_POLICY,
    P2_POLICY,
    PER_MATCH_LOG_DIR,
    PREFER_LEARNABLE,
    PRIVATE_IDS_JSON_PATH,
    RAW_JSONL_PATH,
    REQUIRE_LEARNABLE,
    REWARD_SHAPING_GAMMA,
    SELFPLAY_ALPHAZERO_MODE,
    SKIP_DECKOUT_LOGGING,
    TARGET_EPISODES,
    TORCH_THREADS_PER_PROC,
    USE_MCC,
    USE_MCTS_POLICY,
    USE_REWARD_SHAPING,
    VALUE_NET_MODEL_DIR,
    VALUE_NET_MODEL_FILENAME,
    VALUE_NET_MODEL_PATH,
    WRITER_FLUSH_SEC,
    WRITER_FSYNC,
    Z_PROGRESS_MAX,
)
from policy_factory import build_policy
from obs_vector import build_obs_partial_vec, set_card_id2idx
from legal_actions import (
    set_action_encoder as _set_action_encoder,
    _safe_encode_obs_for_candidates,
    _embed_legal_actions_32d,
    encode_action_from_vec_32d,
    _attach_action_encoder_if_supported,
    _pick_legal_actions,
)

# -------------------------------------------------------
# 以降の設定メモ・補助はこのファイルに残す（値そのものは config.py）
# -------------------------------------------------------

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

    print(f"[POLICY_SPEC] P1_POLICY={P1_POLICY} P2_POLICY=...DE={SELFPLAY_ALPHAZERO_MODE} USE_MCTS_POLICY={USE_MCTS_POLICY}")

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
#    PRIZE_OUT/BASICS_OUT（強い勝ち） > CONCEDE（中） > SUDDEN_DEATH（中） > TIMEOUT/UNKNOWN（弱）
#    そして “DECK_OUT 勝ち” はあえて弱く/マイナスにして、
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
# これにより終局 tier（勝ち方の強さ）の序列を壊さない。
# （係数/上限は config.py を参照）

# --- Phase A: 公開IDと完全情報IDを同時に生成（教師/生徒データを同時出力） ---

# --- 完全情報ベクトルのスキーマ版（Phase A-④ 用） ---

# === MCC（Monte-Carlo Completion）設定 ===

# === PBRS（Potential-Based Reward Shaping）設定 ===
#  - shaped_r = r + γΦ(s') - Φ(s)
#  - 既定では ValueNet を Φ(s) として使う（事前に学習済みモデルが必要）

# ValueNet（Φ(s)）の成果物パス（例: v3 の学習結果）
# --- 試合ごとのファイル保持/削除（圧迫対策） ---
# ★ 追加: 集約（PRIVATE_IDS）への反映が完了したら per-match の .ml.jsonl を削除する
# ★ 追加: 連番ログの出力先（None ならシステムの一時ディレクトリ）
# --- 追加: writer 負荷対策（環境変数なしで既定オン） ---
# 並列数（用途別）: MCC用/ランダム用


# === learnable を強制/優先するためのフラグ ===

# （上の各フラグ/パス/並列数/I/O は config.py を参照）

# 入力記録リスト（必要なら）
recorded_inputs = []

# === コールドスタート（学習成果物が無いとき自動生成） ===
COLD_START = os.getenv("COLD_START", "1") == "1"  # 最初はONにしておく

# ============================================================
# Phase D CQL Q(s,a) ローダ＆評価ヘルパ（phaseD_q.py へ分離）
# ============================================================

from phaseD_q import (
    PHASED_Q_LEARNABLE_PATH,
    PHASED_Q_META_PATH,
    USE_PHASED_Q,
    PHASED_Q_MIX_ENABLED,
    PHASED_Q_MIX_LAMBDA,
    PHASED_Q_MIX_TEMPERATURE,
    phaseD_q_load_if_needed,
    phaseD_q_evaluate,
    phaseD_mix_pi_with_q,
)

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

# --- 出力関係の保存場所（config.py へ分離） ---
# LOG_FILE / RAW_JSONL_PATH / IDS_JSONL_PATH / PRIVATE_IDS_JSON_PATH / GAMELOG_DIR

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




# 共有エンコーダ関数を生成（唯一の正解）
_encode_action_raw, _CARD_ID2IDX, _ACTION_TYPES, (K, V, ACTION_VEC_DIM_RAW) = build_encoder_from_files(
    _card_idx_path, _action_types_path, ACTION_SCHEMAS, TYPE_SCHEMAS, MAX_ARGS
)
set_card_id2idx(_CARD_ID2IDX)

# ログ用の候補ベクトル次元は 32 に固定（実エンコーダ出力は 32 次元にパディング／切り詰め）
ACTION_VEC_DIM = 32

# legal_actions 側へアクションエンコーダを注入（候補ベクトル生成で使用）
_set_action_encoder(_encode_action_raw, ACTION_VEC_DIM)

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

    import writer
    writer = Process(
        target=writer.writer_loop,
        args=(
            q,
            stop,
            RAW_JSONL_PATH,
            IDS_JSONL_PATH,
            PRIVATE_IDS_JSON_PATH,
            IO_BUFFERING_BYTES,
            JSONL_ROTATE_LINES,
            WRITER_FSYNC,
            WRITER_FLUSH_SEC,
            BATCH_FLUSH_INTERVAL,
        ),
        daemon=True,
    )
    writer.start()

    workers = []
    import worker
    for c in chunks:
        # 追加: mcc_agg をワーカーに渡す
        p = Process(target=worker.play_continuous_matches_worker, args=(c, q, mcc_agg, None, _run_game_id), daemon=False)
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
    import writer
    writer = Process(
        target=writer.writer_loop,
        args=(
            q,
            stop,
            RAW_JSONL_PATH,
            IDS_JSONL_PATH,
            PRIVATE_IDS_JSON_PATH,
            IO_BUFFERING_BYTES,
            JSONL_ROTATE_LINES,
            WRITER_FSYNC,
            WRITER_FLUSH_SEC,
            BATCH_FLUSH_INTERVAL,
        ),
        daemon=True,
    )
    writer.start()

    # --- ワーカー起動（gpu_req_q が None なら CPU 経路で evaluate_q される） ---
    workers = []
    import worker
    for c in chunks:
        p = Process(
            target=worker.play_continuous_matches_worker,
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
