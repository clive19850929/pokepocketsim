# ai vs ai.py 起点の呼び出し関係（入口→worker→match/player→policy→mcts_env）

## 入口 → worker / writer / config の接続点
- **入口（`ai vs ai.py` の `__main__`）→ `run_model_matches_multiprocess` / `run_random_matches_multiprocess`**
  呼び出し元: `ai vs ai.py:4638-4642` → 呼び出し先: `run_model_matches_multiprocess` / `run_random_matches_multiprocess`【F:ai vs ai.py†L4638-L4642】

- **`ai vs ai.py` → `config` モジュール import（設定の取り込み）**
  呼び出し元: `ai vs ai.py:279-329` → 呼び出し先: `config`（モジュール import）【F:ai vs ai.py†L279-L329】

- **`run_random_matches_multiprocess` → `writer.writer_loop`**
  呼び出し元: `ai vs ai.py:2078-2094` → 呼び出し先: `writer.writer_loop`【F:ai vs ai.py†L2078-L2094】
  呼び出し先定義: `writer.py:28`【F:writer.py†L28】

- **`run_random_matches_multiprocess` → `_play_matches_worker_entrypoint`**
  呼び出し元: `ai vs ai.py:2099-2103` → 呼び出し先: `_play_matches_worker_entrypoint`【F:ai vs ai.py†L2099-L2103】

- **`run_model_matches_multiprocess` → `writer.writer_loop`**
  呼び出し元: `ai vs ai.py:2247-2263` → 呼び出し先: `writer.writer_loop`【F:ai vs ai.py†L2247-L2263】
  呼び出し先定義: `writer.py:28`【F:writer.py†L28】

- **`run_model_matches_multiprocess` → `_play_matches_worker_entrypoint`**
  呼び出し元: `ai vs ai.py:2266-2272` → 呼び出し先: `_play_matches_worker_entrypoint`【F:ai vs ai.py†L2266-L2272】

- **`_play_matches_worker_entrypoint` → `worker.play_continuous_matches_worker`**
  呼び出し元: `ai vs ai.py:4526-4560` → 呼び出し先: `worker.play_continuous_matches_worker`【F:ai vs ai.py†L4526-L4560】

## worker → match/player → policy の接続点
- **`worker.play_continuous_matches_worker` → `policy_factory.build_policy`**
  呼び出し元: `worker.py:85-87` → 呼び出し先: `policy_factory.build_policy`【F:worker.py†L85-L87】
  呼び出し先定義: `policy_factory.py:1378`【F:policy_factory.py†L1378】

- **`worker.play_continuous_matches_worker` → `Player(...)`**
  呼び出し元: `worker.py:190-194` → 呼び出し先: `pokepocketsim.player.Player.__init__`【F:worker.py†L190-L194】
  呼び出し先定義: `pokepocketsim/player.py:37-45`【F:pokepocketsim/player.py†L37-L45】

- **`worker.play_continuous_matches_worker` → `Match(...)`**
  呼び出し元: `worker.py:283-292` → 呼び出し先: `pokepocketsim.match.Match.__init__`【F:worker.py†L283-L292】
  呼び出し先定義: `pokepocketsim/match.py:12-26`【F:pokepocketsim/match.py†L12-L26】

- **`worker.play_continuous_matches_worker` → `Match.play_one_match()`**
  呼び出し元: `worker.py:551-552` → 呼び出し先: `pokepocketsim.match.Match.play_one_match`【F:worker.py†L551-L552】
  呼び出し先定義: `pokepocketsim/match.py:853-854`【F:pokepocketsim/match.py†L853-L854】

## policy → az_mcts_policy → mcts_env の接続点
- **`policy_factory.build_policy` → `AlphaZeroMCTSPolicy(...)`**
  呼び出し元: `policy_factory.py:1497-1502` / `1525-1529`（分岐によりいずれか） → 呼び出し先: `pokepocketsim.policy.az_mcts_policy.AlphaZeroMCTSPolicy`【F:policy_factory.py†L1497-L1502】【F:policy_factory.py†L1525-L1529】
  呼び出し先定義: `pokepocketsim/policy/az_mcts_policy.py:83-100`【F:pokepocketsim/policy/az_mcts_policy.py†L83-L100】

- **`AlphaZeroMCTSPolicy.select_action_index` → `MatchPlayerSimEnv(...)`（`mcts_env` への接続点）**
  呼び出し元: `pokepocketsim/policy/az_mcts_policy.py:705-716` → 呼び出し先: `pokepocketsim.policy.mcts_env.MatchPlayerSimEnv`【F:pokepocketsim/policy/az_mcts_policy.py†L705-L716】
  呼び出し先定義: `pokepocketsim/policy/mcts_env.py:10-36`【F:pokepocketsim/policy/mcts_env.py†L10-L36】

- **`AlphaZeroMCTSPolicy.select_action` → `_run_mcts(...)`**
  呼び出し元: `pokepocketsim/policy/az_mcts_policy.py:533-538` → 呼び出し先: `AlphaZeroMCTSPolicy._run_mcts`【F:pokepocketsim/policy/az_mcts_policy.py†L533-L538】
