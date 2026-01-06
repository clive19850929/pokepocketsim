# placeholder for AlphaZeroMCTSPolicy (to be implemented)

import os
import random
import time
import traceback
from typing import List, Tuple, Optional, Any, Protocol

import torch


class MCTSSimEnvProtocol(Protocol):
    """
    AlphaZero 型 MCTS が利用する環境インターフェース。

    必須:
    - clone(): 現在の状態から独立なコピーを返す
    - legal_actions(): 現状態の合法手 ID のリスト（このプロジェクトでは 5-int）を返す
    - step(action): 指定 action(5-int) を 1 手適用して状態を進める
    - is_terminal(): 終局かどうかを返す
    - result(): ルート視点の value（勝ち=1, 引き分け=0, 負け=-1）を返す（非終局では例外推奨）

    MCTS で prior/value を使うために要求:
    - get_obs_vec(): 現在手番視点の obs_vec（list[float]）を返す
    - value_to_root(value_current_player): モデル value（現在手番視点）を root 視点に変換して返す
    """

    def clone(self) -> "MCTSSimEnvProtocol":
        ...

    def legal_actions(self) -> List[Any]:
        ...

    def step(self, action: Any) -> None:
        ...

    def is_terminal(self) -> bool:
        ...

    def result(self) -> float:
        ...

    def get_obs_vec(self) -> List[float]:
        ...

    def value_to_root(self, value_current_player: float) -> float:
        ...


class MCTSNode:
    """
    AlphaZero 型 MCTS 用のノード構造。

    - parent: 親ノード（ルートの場合は None）
    - children: action_key -> MCTSNode の辞書（action_key は hashable 化したもの）
    - N: 訪問回数
    - W: 累積価値（root 視点の value の合計）
    - Q: 平均価値（W / N）
    - P: 事前確率（policy ネットからの prior、親→このノードの edge prior）
    - action_from_parent: 親からこのノードへ遷移する action_id（このプロジェクトでは 5-int）
    """

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
        state_key: Optional[Any] = None,
        action_from_parent: Optional[Any] = None,
    ) -> None:
        self.parent: Optional["MCTSNode"] = parent
        self.children: dict = {}
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = float(prior)
        self.state_key: Optional[Any] = state_key
        self.action_from_parent: Optional[Any] = action_from_parent

    def is_leaf(self) -> bool:
        return not self.children


class AlphaZeroMCTSPolicy:
    """
    PolicyValueNet の prior/value を leaf で使い、PUCT で探索する AlphaZero 型 MCTS。

    重要: フォールバック禁止
    - モデル未ロード
    - obs_vec 生成不能 / 次元不一致
    - cand_vecs 生成不能（cand_dim!=5 で action_encoder_fn 不在/失敗）
    - env の step/turn/合法手整合性不一致
    などは即 RuntimeError。
    """

    def __init__(
        self,
        model_dir: str,
        model_filename: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = model_dir or "."
        self.model_filename = model_filename or "selfplay_supervised_pv_gen000.pt"
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.obs_dim: Optional[int] = None
        self.cand_dim: Optional[int] = None
        self.hidden_dim: int = 256

        self.action_encoder_fn = None

        self.temperature: float = 1.0
        self.greedy: bool = False

        self.num_simulations: int = 50
        self.c_puct: float = 1.5
        self.dirichlet_alpha: float = 0.3
        self.dirichlet_eps: float = 0.25

        self.mcts_pi_temperature: float = 1.0

        self.use_mcts: bool = False
        try:
            v = os.getenv("USE_MCTS_POLICY", None)
            if v is None:
                v = os.getenv("AZ_USE_MCTS", None)
            if v is not None:
                self.use_mcts = str(v).strip().lower() in ("1", "true", "yes", "y", "on")
        except Exception:
            self.use_mcts = False

        try:
            v = os.getenv("AZ_MCTS_NUM_SIMULATIONS", None)
            if v is not None and str(v).strip() != "":
                self.num_simulations = int(float(v))
        except Exception:
            pass

        try:
            v = os.getenv("AZ_MCTS_PI_TEMPERATURE", None)
            if v is not None and str(v).strip() != "":
                self.mcts_pi_temperature = float(v)
        except Exception:
            self.mcts_pi_temperature = 1.0

        if os.getenv("AZ_DECISION_LOG", "0") == "1":
            try:
                print(f"[AZ][INIT] use_mcts={int(self.use_mcts)} sims={int(self.num_simulations)}", flush=True)
            except Exception:
                pass

        self._load_model()

    def _load_model(self) -> None:
        """
        model_dir 配下から PolicyValueNet のチェックポイントを探してロードする。
        失敗はフォールバックせず RuntimeError。
        """
        path = self.model_filename
        if not os.path.isabs(path):
            path = os.path.join(self.model_dir, path)

        if not os.path.exists(path):
            cand_path = None
            try:
                for name in sorted(os.listdir(self.model_dir)):
                    if name.lower().endswith(".pt"):
                        cand_path = os.path.join(self.model_dir, name)
                        break
            except Exception:
                cand_path = None

            if cand_path is None or not os.path.exists(cand_path):
                raise RuntimeError(f"[AlphaZeroMCTSPolicy] model file not found in {self.model_dir}")
            path = cand_path

        data = torch.load(path, map_location=self.device)
        if isinstance(data, dict) and "model_state_dict" in data:
            state_dict = data["model_state_dict"]
            self.obs_dim = int(data.get("obs_dim", 0) or 0)
            self.cand_dim = int(data.get("cand_dim", 0) or 0)
            self.hidden_dim = int(data.get("hidden_dim", 256) or 256)
        else:
            state_dict = data
            self.obs_dim = None
            self.cand_dim = None
            self.hidden_dim = 256

        from train_selfplay_supervised import PolicyValueNet

        if self.obs_dim is None or self.cand_dim is None or int(self.obs_dim) <= 0 or int(self.cand_dim) <= 0:
            raise RuntimeError("[AlphaZeroMCTSPolicy] obs_dim / cand_dim missing or invalid in checkpoint.")

        model = PolicyValueNet(self.obs_dim, self.cand_dim, hidden_dim=self.hidden_dim)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

        print(
            f"[AlphaZeroMCTSPolicy] loaded model from {path} "
            f"(obs_dim={self.obs_dim}, cand_dim={self.cand_dim}, hidden_dim={self.hidden_dim})"
        )

    def set_action_encoder(self, fn) -> None:
        self.action_encoder_fn = fn

    def _as_list_vec(self, v):
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

    def _coerce_5int_row(self, row):
        try:
            if row is None:
                return None
            if isinstance(row, tuple):
                row = list(row)
            elif isinstance(row, list):
                row = row[:]
            else:
                return None

            if len(row) < 5:
                row = row + [0] * (5 - len(row))
            elif len(row) > 5:
                row = row[:5]

            out = []
            for x in row:
                out.append(int(x))
            return out
        except Exception:
            return None

    def _coerce_5int_rows(self, rows):
        out = []
        if rows is None:
            return out
        if isinstance(rows, tuple):
            rows = list(rows)
        if not isinstance(rows, list):
            return out
        for r in rows:
            rr = self._coerce_5int_row(r)
            if rr is not None:
                out.append(rr)
        return out

    def _extract_legal_actions_5(self, state_dict: Any, actions: List[Any], player: Any = None) -> List[List[int]]:
        la_5 = []

        if isinstance(state_dict, dict):
            for k in ("legal_actions_5", "legal_actions_vec", "legal_actions"):
                try:
                    v = state_dict.get(k, None)
                except Exception:
                    v = None
                la_5 = self._coerce_5int_rows(v)
                if la_5:
                    return la_5

        m = getattr(player, "match", None) if player is not None else None
        conv = None
        if m is not None:
            try:
                conv = getattr(m, "converter", None)
                if conv is None:
                    conv = getattr(m, "action_converter", None)
            except Exception:
                conv = None

        if conv is not None:
            fn_la = getattr(conv, "convert_legal_actions", None)
            if callable(fn_la):
                la_5 = self._coerce_5int_rows(fn_la(actions or []))
                if la_5:
                    return la_5

        tmp = []
        for a in actions or []:
            fn = getattr(a, "to_id_vec", None)
            if callable(fn):
                tmp.append(fn())
                continue
            if isinstance(a, (list, tuple)):
                tmp.append(list(a) if isinstance(a, tuple) else a)

        la_5 = self._coerce_5int_rows(tmp)
        return la_5

    def _extract_cand_vecs_for_model(self, state_dict: Any, la_5: List[List[int]]) -> Optional[Any]:
        cdim = int(self.cand_dim or 0)
        if cdim <= 0:
            return None

        if cdim == 5:
            return la_5 if la_5 else None

        if isinstance(state_dict, dict):
            for k in ("action_candidates_vec", "action_candidates_vecs", "cand_vecs", "cand_vecs_32d"):
                try:
                    v = state_dict.get(k, None)
                except Exception:
                    v = None
                if isinstance(v, list) and v:
                    try:
                        v0 = v[0]
                        if isinstance(v0, (list, tuple)) and len(v0) == cdim:
                            return v
                    except Exception:
                        pass

        return None

    def get_obs_vec(self, state_dict: Any = None, actions: Optional[List[Any]] = None, player: Any = None):
        target_dim = int(self.obs_dim or 0)

        def _dim_ok(v):
            if target_dim <= 0:
                return True
            if not isinstance(v, (list, tuple)):
                return False
            return len(v) == target_dim

        if isinstance(state_dict, dict):
            for k in ("obs_vec", "obs_vec_az", "az_obs_vec", "observation_vec", "obs", "x"):
                try:
                    vv = state_dict.get(k, None)
                except Exception:
                    vv = None
                vv = self._as_list_vec(vv)
                if vv is not None and _dim_ok(vv):
                    return vv

        m = getattr(player, "match", None) if player is not None else None
        if m is not None:
            try:
                vv = getattr(m, "obs_vec_example", None)
            except Exception:
                vv = None
            vv = self._as_list_vec(vv)
            if vv is not None and _dim_ok(vv):
                return vv

        enc = getattr(m, "encoder", None) if m is not None else None
        if enc is not None:
            for meth in ("encode_obs_vec", "encode_obs", "encode_for_player", "encode", "build_obs_vec", "make_obs_vec", "get_obs_vec"):
                fn = getattr(enc, meth, None)
                if not callable(fn):
                    continue
                for args in ((m, player, actions), (m, player), (player, actions), (player,), (m, actions), (m,), ()):
                    try:
                        out = fn(*args)
                    except TypeError:
                        continue
                    vv = self._as_list_vec(out)
                    if vv is not None and _dim_ok(vv):
                        return vv

        return None

    def encode_obs_vec(self, state_dict: Any = None, actions: Optional[List[Any]] = None, player: Any = None):
        return self.get_obs_vec(state_dict=state_dict, actions=actions, player=player)

    def _build_obs_tensor(self, obs_vec):
        if self.model is None or self.obs_dim is None:
            raise RuntimeError("[AZ] model or obs_dim is missing (no fallback).")

        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"[AZ] numpy is required for obs tensor build: {e}")

        obs_arr = np.asarray(obs_vec, dtype="float32").reshape(-1)
        if int(obs_arr.shape[0]) != int(self.obs_dim):
            raise RuntimeError(f"[AZ] obs_dim mismatch: expected={int(self.obs_dim)} got={int(obs_arr.shape[0])}")
        return torch.from_numpy(obs_arr).view(1, -1).to(self.device)

    def _build_cands_tensor_from_action_ids(self, legal_action_ids: List[Any], cand_vecs: Optional[Any] = None):
        if self.model is None or self.cand_dim is None:
            raise RuntimeError("[AZ] model or cand_dim is missing (no fallback).")

        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"[AZ] numpy is required for cand tensor build: {e}")

        cdim = int(self.cand_dim or 0)
        if cdim <= 0:
            raise RuntimeError("[AZ] cand_dim must be positive (no fallback).")

        cand_list = []

        if cand_vecs is not None:
            try:
                for v in cand_vecs:
                    enc_arr = np.asarray(v, dtype="float32").reshape(-1)
                    if enc_arr.shape[-1] < cdim:
                        pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                        enc_arr = np.concatenate([enc_arr.reshape(-1), pad], axis=0)
                    elif enc_arr.shape[-1] > cdim:
                        enc_arr = enc_arr.reshape(-1)[:cdim]
                    cand_list.append(enc_arr)
            except Exception:
                cand_list = []

        if not cand_list:
            if cdim == 5:
                for aid in legal_action_ids:
                    enc_arr = np.asarray(aid, dtype="float32").reshape(-1)
                    if enc_arr.shape[-1] < cdim:
                        pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                        enc_arr = np.concatenate([enc_arr.reshape(-1), pad], axis=0)
                    elif enc_arr.shape[-1] > cdim:
                        enc_arr = enc_arr.reshape(-1)[:cdim]
                    cand_list.append(enc_arr)
            else:
                if self.action_encoder_fn is None:
                    raise RuntimeError(f"[AZ] action_encoder_fn is required for cand_dim={int(cdim)} (no fallback).")
                for aid in legal_action_ids:
                    enc = self.action_encoder_fn(aid)
                    enc_arr = np.asarray(enc, dtype="float32").reshape(-1)
                    if enc_arr.shape[-1] < cdim:
                        pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                        enc_arr = np.concatenate([enc_arr, pad], axis=0)
                    elif enc_arr.shape[-1] > cdim:
                        enc_arr = enc_arr[:cdim]
                    cand_list.append(enc_arr)

        if not cand_list:
            raise RuntimeError("[AZ] failed to build cand_list (no fallback).")
        if len(cand_list) != int(len(legal_action_ids)):
            raise RuntimeError("[AZ] cand_list length mismatch (no fallback).")

        cands_arr = np.stack(cand_list, axis=0)
        return torch.from_numpy(cands_arr).to(self.device)

    def _policy_value(self, obs_vec, legal_action_ids: List[Any], cand_vecs: Optional[Any] = None) -> Tuple[List[float], float]:
        if self.model is None:
            raise RuntimeError("[AZ] model is None (no fallback).")
        if self.obs_dim is None or self.cand_dim is None:
            raise RuntimeError("[AZ] obs_dim/cand_dim missing (no fallback).")

        obs_tensor = self._build_obs_tensor(obs_vec)
        cands_tensor = self._build_cands_tensor_from_action_ids(legal_action_ids, cand_vecs=cand_vecs)

        self.model.eval()
        with torch.no_grad():
            logits_list, values = self.model(obs_tensor, [cands_tensor])

        if not logits_list:
            raise RuntimeError("[AZ] model returned empty logits_list (no fallback).")

        logits = logits_list[0].detach().cpu()
        if logits.ndim != 1 or int(logits.shape[0]) != int(len(legal_action_ids)):
            raise RuntimeError(
                f"[AZ] logits shape mismatch: shape={tuple(logits.shape)} n_actions={int(len(legal_action_ids))} (no fallback)."
            )

        temp = float(getattr(self, "temperature", 1.0) or 1.0)
        if temp <= 0.0:
            temp = 1.0
        scaled_logits = logits / temp
        probs_t = torch.softmax(scaled_logits, dim=-1)
        probs = probs_t.numpy().astype("float64")
        s = float(probs.sum())
        if not (s > 0.0):
            raise RuntimeError("[AZ] probs_sum<=0 (no fallback).")
        priors = (probs / s).tolist()

        v = values
        try:
            v0 = float(v.detach().view(-1)[0].cpu().item())
        except Exception:
            try:
                v0 = float(v[0])
            except Exception:
                raise RuntimeError("[AZ] value extraction failed (no fallback).")

        return priors, v0

    def select_action(
        self,
        obs_vec,
        legal_action_ids: List[Any],
        env: Optional["MCTSSimEnvProtocol"] = None,
        cand_vecs: Optional[Any] = None,
    ) -> Tuple[Optional[Any], List[float]]:
        if not isinstance(legal_action_ids, list) or not legal_action_ids:
            raise RuntimeError("[AZ] legal_action_ids must be a non-empty list (no fallback).")

        if self.model is None or self.obs_dim is None or self.cand_dim is None:
            raise RuntimeError("[AZ] model or dims missing (no fallback).")

        use_mcts = bool(getattr(self, "use_mcts", False))
        num_sims = int(getattr(self, "num_simulations", 0) or 0)

        if os.getenv("AZ_DECISION_LOG", "0") == "1":
            try:
                env_name_always = type(env).__name__ if env is not None else None
            except Exception:
                env_name_always = None
            print(
                f"[AZ][MCTS][PRECHECK_ALWAYS] model_ok=1 use_mcts={int(use_mcts)} env={env_name_always} sims={int(num_sims)} n_actions={int(len(legal_action_ids))}",
                flush=True,
            )

        priors_model, _v_model = self._policy_value(obs_vec, legal_action_ids, cand_vecs=cand_vecs)
        probs = priors_model
        decision_src = "model"

        if use_mcts and int(num_sims) > 0:
            if env is None:
                raise RuntimeError("[AZ] use_mcts=1 but env is None (no fallback).")

            try:
                mcts_pi = self._run_mcts(env, legal_action_ids, num_simulations=num_sims)
            except Exception as e:
                import time
                import traceback
                from ..debug_dump import write_debug_dump

                m = getattr(env, "_match", None)
                p = getattr(env, "_player", None)
                forced_active = None
                try:
                    fa = getattr(m, "forced_actions", None) if m is not None else None
                    if isinstance(fa, (list, tuple)):
                        forced_active = len(fa) > 0
                except Exception:
                    forced_active = None

                legal_actions_serialized = []
                try:
                    for i, vec in enumerate(legal_action_ids):
                        if isinstance(vec, tuple):
                            vec = list(vec)
                        if not isinstance(vec, list):
                            vec = [0, 0, 0, 0, 0]
                        if len(vec) < 5:
                            vec = list(vec) + [0] * (5 - len(vec))
                        if len(vec) > 5:
                            vec = list(vec)[:5]
                        legal_actions_serialized.append(
                            {"i": i, "action_type": None, "name": None, "vec": vec}
                        )
                except Exception:
                    legal_actions_serialized = []

                payload = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "run_context": {
                        "game_id": getattr(m, "game_id", None) if m is not None else None,
                        "turn": getattr(m, "turn", None) if m is not None else None,
                        "player": getattr(p, "name", None) if p is not None else None,
                        "forced_actions_active": forced_active,
                    },
                    "action_context": {
                        "selected_vec": None,
                        "selected_source": "mcts",
                        "legal_actions_serialized": legal_actions_serialized,
                    },
                    "mcts_context": {
                        "num_simulations": int(num_sims),
                        "n_actions": int(len(legal_action_ids)),
                    },
                    "traceback": traceback.format_exception(type(e), e, e.__traceback__),
                }
                dump_path = write_debug_dump(payload)
                print(f"[DEBUG_DUMP] wrote: {dump_path}", flush=True)
                raise
            if not (isinstance(mcts_pi, list) and int(len(mcts_pi)) == int(len(legal_action_ids))):
                raise RuntimeError("[AZ] _run_mcts returned invalid pi (no fallback).")
            probs = mcts_pi
            decision_src = "mcts"

        if bool(getattr(self, "greedy", False)):
            idx = int(max(range(len(probs)), key=lambda i: float(probs[i])))
            pick = "argmax"
        else:
            r = random.random()
            acc = 0.0
            idx = len(legal_action_ids) - 1
            for i, p in enumerate(probs):
                acc += float(p)
                if r <= acc:
                    idx = i
                    break
            pick = "sample"

        chosen_action = legal_action_ids[idx]

        try:
            self.last_decision_src = str(decision_src)
            self.last_pick = str(pick)
        except Exception:
            pass

        if os.getenv("AZ_DECISION_LOG", "0") == "1":
            try:
                a_repr = repr(chosen_action)
                if len(a_repr) > 160:
                    a_repr = a_repr[:160] + "..."
            except Exception:
                a_repr = "<unrepr>"
            print(
                f"[AZ][DECISION] src={decision_src} pick={pick} idx={int(idx)} n_actions={int(len(legal_action_ids))} action={a_repr}",
                flush=True,
            )

        return chosen_action, probs

    def select_action_index_online(self, state_dict: Any, actions: List[Any], player: Any = None, return_pi: bool = False):
        if os.getenv("AZ_DECISION_LOG", "0") == "1":
            try:
                m = getattr(player, "match", None) if player is not None else None
                gid = getattr(m, "game_id", None) if m is not None else None
                turn = getattr(m, "turn", None) if m is not None else None
                n_act = int(len(actions)) if isinstance(actions, list) else -1
                print(
                    f"[AZ][DECISION][CALL] entry=select_action_index_online game_id={gid} turn={turn} n_actions={n_act} return_pi={int(bool(return_pi))}",
                    flush=True,
                )
            except Exception:
                pass

        return self.select_action_index(state_dict, actions, player=player, return_pi=return_pi)

    def select_action_index(self, state_dict: Any, actions: List[Any], player: Any = None, return_pi: bool = False) -> Any:
        _az_log = (os.getenv("AZ_DECISION_LOG", "0") == "1")
        _gid = None
        _turn = None
        try:
            m = getattr(player, "match", None) if player is not None else None
            _gid = getattr(m, "game_id", None) if m is not None else None
            _turn = getattr(m, "turn", None) if m is not None else None
        except Exception:
            _gid = None
            _turn = None

        if _az_log:
            try:
                n_act = int(len(actions)) if isinstance(actions, list) else -1
                print(
                    f"[AZ][DECISION][CALL] entry=select_action_index game_id={_gid} turn={_turn} n_actions={n_act} return_pi={int(bool(return_pi))}",
                    flush=True,
                )
            except Exception:
                pass

        if not isinstance(actions, list) or not actions:
            raise RuntimeError("[AZ] actions is empty (no fallback).")

        obs_vec = None
        if isinstance(state_dict, dict):
            obs_vec = state_dict.get("obs_vec", None)
        if obs_vec is None:
            obs_vec = self.get_obs_vec(state_dict=state_dict, actions=actions, player=player)
        if obs_vec is None:
            raise RuntimeError("[AZ] obs_vec is missing (no fallback).")

        la_5 = self._extract_legal_actions_5(state_dict, actions, player=player)
        if not la_5:
            raise RuntimeError("[AZ] failed to extract legal_actions_5 (no fallback).")

        if isinstance(state_dict, dict):
            state_dict["legal_actions_5"] = la_5
            state_dict["legal_actions_vec"] = la_5
            state_dict["legal_actions"] = la_5
            state_dict["az_la5_n"] = int(len(la_5))

        legal_action_ids: List[Any] = list(la_5)

        env = None
        use_mcts = bool(getattr(self, "use_mcts", False))
        num_sims = int(getattr(self, "num_simulations", 0) or 0)

        if use_mcts and num_sims > 0:
            if player is None or getattr(player, "match", None) is None:
                raise RuntimeError("[AZ] use_mcts=1 but player or player.match is missing (no fallback).")

            MatchPlayerSimEnv = None
            try:
                from .mcts_env import MatchPlayerSimEnv as _MatchPlayerSimEnv
                MatchPlayerSimEnv = _MatchPlayerSimEnv
            except Exception as e_rel:
                try:
                    from pokepocketsim.policy.mcts_env import MatchPlayerSimEnv as _MatchPlayerSimEnv
                    MatchPlayerSimEnv = _MatchPlayerSimEnv
                except Exception as e_abs:
                    raise RuntimeError(f"[AZ] MatchPlayerSimEnv import failed: rel={repr(e_rel)} abs={repr(e_abs)}")

            env = MatchPlayerSimEnv(player.match, player)

        cand_vecs = self._extract_cand_vecs_for_model(state_dict, la_5)

        chosen_action, pi = self.select_action(obs_vec, legal_action_ids, env=env, cand_vecs=cand_vecs)

        if chosen_action is None:
            raise RuntimeError("[AZ] chosen_action is None (no fallback).")

        try:
            idx = legal_action_ids.index(chosen_action)
        except ValueError:
            raise RuntimeError("[AZ] chosen_action not found in legal_action_ids (no fallback).")

        if not (isinstance(pi, list) and len(pi) == len(actions)):
            raise RuntimeError("[AZ] pi missing or length mismatch (no fallback).")

        if isinstance(state_dict, dict):
            state_dict["mcts_pi"] = [float(x) for x in pi]
            state_dict["pi"] = [float(x) for x in pi]
            state_dict["mcts_idx"] = int(idx)
            state_dict["mcts_pi_present"] = 1
            state_dict["mcts_pi_len"] = int(len(pi))
            state_dict["mcts_pi_type"] = type(pi).__name__
            state_dict["mcts_pi_from"] = "az_pi"
            try:
                state_dict["az_decision_src"] = str(getattr(self, "last_decision_src", "unknown"))
                state_dict["az_decision_pick"] = str(getattr(self, "last_pick", "unknown"))
            except Exception:
                pass

        try:
            self.last_pi = [float(x) for x in pi]
            self.mcts_pi = [float(x) for x in pi]
            self.last_mcts_pi = [float(x) for x in pi]
        except Exception:
            pass

        if _az_log:
            try:
                print(
                    f"[AZ][DECISION] src={str(getattr(self, 'last_decision_src', 'unknown'))} pick={str(getattr(self, 'last_pick', 'unknown'))} idx={int(idx)} n_actions={int(len(actions))} pi_from=az_pi",
                    flush=True,
                )
            except Exception:
                pass

        return (int(idx), [float(x) for x in pi]) if bool(return_pi) else int(idx)

    def _run_mcts(
        self,
        env: "MCTSSimEnvProtocol",
        legal_action_ids: List[Any],
        num_simulations: Optional[int] = None,
    ) -> List[float]:
        try:
            _n0 = len(legal_action_ids) if isinstance(legal_action_ids, list) else -1
        except Exception:
            _n0 = -1
        try:
            _env_name = type(env).__name__ if env is not None else None
        except Exception:
            _env_name = None
        print(
            f"[AZ][MCTS][_run_mcts][ENTER_ALWAYS] env={_env_name} n_actions={int(_n0)} num_simulations={num_simulations}",
            flush=True,
        )

        try:
            self._mcts_enter_counter = int(getattr(self, "_mcts_enter_counter", 0) or 0) + 1
        except Exception:
            self._mcts_enter_counter = 1

        if env is None:
            raise RuntimeError("[AZ][MCTS] env is None (no fallback).")
        if not isinstance(legal_action_ids, list) or not legal_action_ids:
            raise RuntimeError("[AZ][MCTS] legal_action_ids is empty (no fallback).")

        sims = num_simulations if num_simulations is not None else getattr(self, "num_simulations", 0)
        if int(sims) <= 0:
            raise RuntimeError(f"[AZ][MCTS] sims<=0 (sims={int(sims)}) (no fallback).")

        try:
            self._mcts_t0_perf = time.perf_counter()
        except Exception:
            self._mcts_t0_perf = None

        def _akey(a):
            try:
                hash(a)
                return a
            except Exception:
                pass
            if isinstance(a, list):
                return tuple(_akey(x) for x in a)
            if isinstance(a, tuple):
                return tuple(_akey(x) for x in a)
            if isinstance(a, dict):
                try:
                    items = sorted(a.items(), key=lambda kv: str(kv[0]))
                except Exception:
                    items = list(a.items())
                return tuple((_akey(k), _akey(v)) for k, v in items)
            try:
                return ("repr", repr(a))
            except Exception:
                return ("id", id(a))

        root = MCTSNode(parent=None, prior=1.0, state_key=None, action_from_parent=None)

        root_actions_env = env.legal_actions()
        if not isinstance(root_actions_env, list) or not root_actions_env:
            raise RuntimeError("[AZ][MCTS] env.legal_actions() returned empty at root (no fallback).")

        if int(len(root_actions_env)) != int(len(legal_action_ids)):
            raise RuntimeError(
                f"[AZ][MCTS] root legal_actions length mismatch: env={int(len(root_actions_env))} given={int(len(legal_action_ids))} (no fallback)."
            )

        try:
            for i in range(len(legal_action_ids)):
                if root_actions_env[i] != legal_action_ids[i]:
                    raise RuntimeError("[AZ][MCTS] root legal_actions content mismatch (no fallback).")
        except Exception as e:
            raise RuntimeError(f"[AZ][MCTS] root legal_actions mismatch detail: {e}")

        obs_root = env.get_obs_vec()
        priors_root, v_cur_root = self._policy_value(obs_root, legal_action_ids, cand_vecs=None)

        try:
            v_root = float(env.value_to_root(float(v_cur_root)))
        except Exception:
            raise RuntimeError("[AZ][MCTS] env.value_to_root is missing or failed (no fallback).")

        priors = [float(x) for x in priors_root]

        use_dir = os.getenv("AZ_MCTS_DIRICHLET", "0") == "1"
        if use_dir:
            try:
                import numpy as np
                alpha = float(self.dirichlet_alpha)
                eps = float(self.dirichlet_eps)
                if alpha > 0.0 and eps > 0.0:
                    noise = np.random.dirichlet([alpha] * len(priors)).astype("float64").tolist()
                    priors = [float((1.0 - eps) * float(p) + eps * float(n)) for p, n in zip(priors, noise)]
                    s = float(sum(priors))
                    if not (s > 0.0):
                        raise RuntimeError("[AZ][MCTS] dirichlet produced non-positive sum (no fallback).")
                    priors = [float(p) / s for p in priors]
            except Exception as e:
                raise RuntimeError(f"[AZ][MCTS] dirichlet failed: {e} (no fallback).")

        root_keys = []
        for aid, p in zip(legal_action_ids, priors):
            k = _akey(aid)
            root_keys.append(k)
            root.children[k] = MCTSNode(parent=root, prior=float(p), state_key=None, action_from_parent=aid)

        sim_ok = 0
        sim_err = 0

        mute = os.getenv("AZ_MCTS_MUTE", "1") == "1"
        echo = os.getenv("AZ_MCTS_ECHO", ("1" if mute else "0")) == "1"
        if mute:
            import contextlib
            import io

            @contextlib.contextmanager
            def _mute_stdio():
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    yield buf
        else:
            import contextlib

            @contextlib.contextmanager
            def _mute_stdio():
                yield None

        def _echo_from_buf(_buf_text: str):
            if not echo:
                return
            try:
                k = "_mcts_echo_budget"
                b = getattr(self, k, None)
                if b is None:
                    b = int(os.getenv("AZ_MCTS_ECHO_BUDGET", "40") or "40")
                b = int(b)
                if b <= 0:
                    return

                la5_hits = []
                act_hits = []
                for ln in str(_buf_text).splitlines():
                    if "[MCTS_ENV][LA5]" in ln:
                        la5_hits.append(ln)
                    elif "[ACTION_SERIALIZE][MCTS]" in ln:
                        act_hits.append(ln)

                hits = (la5_hits + act_hits)[:60]

                if not hits:
                    return

                for ln in hits:
                    if b <= 0:
                        break
                    print(ln, flush=True)
                    b -= 1

                setattr(self, k, b)
            except Exception:
                pass

        for sim_i in range(int(sims)):
            buf_text = None
            with _mute_stdio() as buf:
                try:
                    sim_env = env.clone()
                    node = root
                    path = [root]

                    last_action = None

                    while (not sim_env.is_terminal()) and (not node.is_leaf()):
                        total_N = sum(c.N for c in node.children.values()) or 1
                        best_child = None
                        best_score = None

                        for child in node.children.values():
                            u = self.c_puct * float(child.P) * (float(total_N) ** 0.5) / (1.0 + float(child.N))
                            score = float(child.Q) + float(u)
                            if best_score is None or score > best_score:
                                best_score = score
                                best_child = child

                        if best_child is None:
                            raise RuntimeError("[AZ][MCTS] selection failed: best_child is None (no fallback).")

                        if best_child.action_from_parent is None:
                            raise RuntimeError("[AZ][MCTS] selection failed: action_from_parent is None (no fallback).")

                        last_action = best_child.action_from_parent
                        sim_env.step(last_action)

                        node = best_child
                        path.append(node)

                    if sim_env.is_terminal():
                        leaf_value = float(sim_env.result())
                    else:
                        next_actions = sim_env.legal_actions()
                        if not isinstance(next_actions, list) or not next_actions:
                            raise RuntimeError("[AZ][MCTS] non-terminal but legal_actions empty (no fallback).")

                        obs_leaf = sim_env.get_obs_vec()
                        priors_leaf, v_cur_leaf = self._policy_value(obs_leaf, next_actions, cand_vecs=None)

                        try:
                            v_root_leaf = float(sim_env.value_to_root(float(v_cur_leaf)))
                        except Exception:
                            raise RuntimeError("[AZ][MCTS] env.value_to_root failed at leaf (no fallback).")

                        leaf_value = v_root_leaf

                        p_sum = float(sum(float(x) for x in priors_leaf))
                        if not (p_sum > 0.0):
                            raise RuntimeError("[AZ][MCTS] priors_leaf sum<=0 (no fallback).")
                        priors_leaf_norm = [float(x) / p_sum for x in priors_leaf]

                        for aid, p in zip(next_actions, priors_leaf_norm):
                            k = _akey(aid)
                            if k in node.children:
                                continue
                            node.children[k] = MCTSNode(parent=node, prior=float(p), state_key=None, action_from_parent=aid)

                    v = float(leaf_value)
                    for nd in path:
                        nd.N += 1
                        nd.W += v
                        nd.Q = nd.W / float(max(1, nd.N))

                    sim_ok += 1
                except Exception as e:
                    sim_err += 1
                    try:
                        extra = ""
                        if mute and buf is not None:
                            try:
                                extra = str(buf.getvalue())
                                try:
                                    la5_hits = []
                                    act_hits = []
                                    for ln in str(extra).splitlines():
                                        if "[MCTS_ENV][LA5]" in ln:
                                            la5_hits.append(ln)
                                        elif "[ACTION_SERIALIZE][MCTS]" in ln:
                                            act_hits.append(ln)
                                    if la5_hits or act_hits:
                                        extra = "\n".join((la5_hits + act_hits)[:60])
                                except Exception:
                                    pass
                                if len(extra) > 2000:
                                    extra = extra[:2000] + "\n. (truncated)"
                            except Exception:
                                extra = ""
                        raise RuntimeError(
                            f"[AZ][MCTS][SIM_FAIL] sim={int(sim_i)} ok={int(sim_ok)} "
                            f"etype={type(e).__name__} err={repr(e)}"
                            + (("\n[MCTS_MUTE_BUFFER]\n" + extra) if extra else "")
                        )
                    except Exception:
                        raise
                finally:
                    if mute and buf is not None:
                        try:
                            buf_text = str(buf.getvalue())
                        except Exception:
                            buf_text = None

            if mute and buf_text:
                _echo_from_buf(buf_text)

        self._last_mcts_stats = {"sims": int(sims), "ok": int(sim_ok), "err": 0}

        visit_counts: List[float] = []
        for k in root_keys:
            child = root.children.get(k)
            n = float(child.N) if child is not None else 0.0
            visit_counts.append(n)

        total_visits = float(sum(visit_counts))
        if not (total_visits > 0.0):
            ctx_game_id = None
            ctx_turn = None
            ctx_player = None
            ctx_forced = None
            ctx_forced_len = None
            try:
                m = getattr(env, "_match", None)
                ctx_game_id = getattr(m, "game_id", None) if m is not None else None
                ctx_turn = getattr(m, "turn", None) if m is not None else None
            except Exception:
                ctx_game_id = None
                ctx_turn = None
            try:
                p = getattr(env, "_player", None)
                ctx_player = getattr(p, "name", None) if p is not None else None
            except Exception:
                ctx_player = None
            try:
                fn = getattr(env, "_forced_is_active", None)
                if callable(fn):
                    ctx_forced = bool(fn())
            except Exception:
                ctx_forced = None
            try:
                fn = getattr(env, "_get_forced_actions_raw", None)
                if callable(fn):
                    fr = fn()
                    if isinstance(fr, (list, tuple)):
                        ctx_forced_len = int(len(fr))
            except Exception:
                ctx_forced_len = None
            try:
                print(
                    "[AZ][MCTS][TOTAL_VISITS_ZERO]"
                    f" game_id={ctx_game_id}"
                    f" turn={ctx_turn}"
                    f" player={ctx_player}"
                    f" forced_active={ctx_forced}"
                    f" forced_len={ctx_forced_len}"
                    f" n_actions={int(len(legal_action_ids))}"
                    f" num_simulations={int(sims)}"
                    f" sim_ok={int(sim_ok)}"
                    f" sim_err={int(sim_err)}"
                    f" root_children={int(len(root.children))}"
                    f" root_N={getattr(root, 'N', None)}",
                    flush=True,
                )
            except Exception:
                pass
            try:
                import time
                from ..debug_dump import write_debug_dump

                def _coerce_vec(v):
                    if isinstance(v, tuple):
                        v = list(v)
                    if not isinstance(v, list):
                        v = [0, 0, 0, 0, 0]
                    if len(v) < 5:
                        v = list(v) + [0] * (5 - len(v))
                    if len(v) > 5:
                        v = list(v)[:5]
                    return v

                payload = {
                    "error_type": "total_visits<=0",
                    "error_message": "[AZ][MCTS] total_visits<=0 (no fallback).",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "run_context": {
                        "game_id": ctx_game_id,
                        "turn": ctx_turn,
                        "player": ctx_player,
                        "forced_actions_active": ctx_forced,
                    },
                    "action_context": {
                        "selected_vec": None,
                        "selected_source": "mcts",
                        "legal_actions_serialized": [
                            {"i": i, "action_type": None, "name": None, "vec": _coerce_vec(aid)}
                            for i, aid in enumerate(legal_action_ids)
                        ],
                    },
                    "mcts_context": {
                        "num_simulations": int(sims),
                        "total_visits": float(total_visits),
                        "n_actions": int(len(legal_action_ids)),
                        "any_exception_counters": {
                            "sim_ok": int(sim_ok),
                            "sim_err": int(sim_err),
                        },
                    },
                }
                dump_path = write_debug_dump(payload)
                print(f"[DEBUG_DUMP] wrote: {dump_path}", flush=True)
            except Exception:
                pass
            raise RuntimeError("[AZ][MCTS] total_visits<=0 (no fallback).")

        pi_temp = float(getattr(self, "mcts_pi_temperature", 1.0) or 1.0)
        if pi_temp <= 0.0:
            best_i = int(max(range(len(visit_counts)), key=lambda i: float(visit_counts[i])))
            pi = [0.0] * int(len(visit_counts))
            pi[best_i] = 1.0
        else:
            power = 1.0 / float(pi_temp)
            ww = [float(n) ** float(power) for n in visit_counts]
            s = float(sum(ww))
            if not (s > 0.0):
                raise RuntimeError("[AZ][MCTS] pi normalization failed (no fallback).")
            pi = [float(x) / s for x in ww]

        try:
            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                elapsed_ms = None
                try:
                    t0 = getattr(self, "_mcts_t0_perf", None)
                    if t0 is not None:
                        elapsed_ms = int((time.perf_counter() - float(t0)) * 1000.0)
                except Exception:
                    elapsed_ms = None

                topk = 3 if len(pi) >= 3 else len(pi)
                order = sorted(range(len(pi)), key=lambda i: float(visit_counts[i]), reverse=True)[:topk]
                top_parts = []
                for i in order:
                    try:
                        a = legal_action_ids[i]
                        a_repr = repr(a)
                        if len(a_repr) > 120:
                            a_repr = a_repr[:120] + "..."
                    except Exception:
                        a_repr = "<unrepr>"
                    top_parts.append(f"#{i}:N={visit_counts[i]:.0f},p={pi[i]:.3f},a={a_repr}")

                print(
                    f"[AZ][MCTS][CONFIRM] sims={int(sims)} ok={int(sim_ok)} err=0 "
                    f"elapsed_ms={elapsed_ms} root_total_visits={float(total_visits):.1f} "
                    f"top3={' | '.join(top_parts)} stats={getattr(self, '_last_mcts_stats', None)}",
                    flush=True,
                )
        except Exception:
            pass

        return [float(x) for x in pi]

    # ======================================================================
    #  以下、AlphaZero 型 MCTS 用の補助メソッド群（未使用の器。必要なら後で整理）
    # ======================================================================

    def _create_root_node(
        self,
        state_key: Optional[Any],
        legal_action_ids: List[Any],
        priors: List[float],
    ) -> MCTSNode:
        root = MCTSNode(parent=None, prior=1.0, state_key=state_key, action_from_parent=None)

        if not legal_action_ids:
            return root

        n = min(len(legal_action_ids), len(priors))
        for i in range(n):
            aid = legal_action_ids[i]
            p = float(priors[i])
            if p < 0.0:
                p = 0.0
            child = MCTSNode(parent=root, prior=p, state_key=None, action_from_parent=aid)
            root.children[aid] = child

        total_p = sum(child.P for child in root.children.values())
        if total_p > 0.0:
            inv_total = 1.0 / total_p
            for child in root.children.values():
                child.P *= inv_total

        return root

    def _select_child(self, node: MCTSNode) -> Tuple[Any, MCTSNode]:
        if not node.children:
            raise ValueError("_select_child called on a leaf node without children.")

        total_N = sum(child.N for child in node.children.values())
        if total_N <= 0:
            total_N = 1

        best_action = None
        best_child = None
        best_score = None

        for action_id, child in node.children.items():
            Q = child.Q
            U = self.c_puct * child.P * ((total_N ** 0.5) / (1.0 + child.N))
            score = Q + U

            if best_score is None or score > best_score:
                best_score = score
                best_action = action_id
                best_child = child

        return best_action, best_child

    def _expand_node(self, node: MCTSNode, action_priors: List[Tuple[Any, float]]) -> None:
        for action_id, prior in action_priors:
            p = float(prior)
            if p < 0.0:
                p = 0.0
            if action_id in node.children:
                node.children[action_id].P = p
            else:
                node.children[action_id] = MCTSNode(
                    parent=node,
                    prior=p,
                    state_key=None,
                    action_from_parent=action_id,
                )

        total_p = sum(child.P for child in node.children.values())
        if total_p > 0.0:
            inv_total = 1.0 / total_p
            for child in node.children.values():
                child.P *= inv_total

    def _backup(self, path: List[MCTSNode], leaf_value: float) -> None:
        v = float(leaf_value)
        for node in path:
            node.N += 1
            node.W += v
            node.Q = node.W / max(1, node.N)

    def _apply_dirichlet_noise_to_root(self, root: MCTSNode) -> None:
        if not root.children:
            return

        import numpy as np

        n = len(root.children)
        if n == 0:
            return

        alpha = float(self.dirichlet_alpha)
        eps = float(self.dirichlet_eps)

        if alpha <= 0.0 or eps <= 0.0:
            return

        noise = np.random.dirichlet([alpha] * n).astype("float64")
        total_p = sum(child.P for child in root.children.values())
        if total_p <= 0.0:
            total_p = 1.0
        base_ps = [child.P / total_p for child in root.children.values()]

        for (action_id, child), base_p, eta in zip(root.children.items(), base_ps, noise):
            child.P = (1.0 - eps) * base_p + eps * float(eta)
