# placeholder for AlphaZeroMCTSPolicy (to be implemented)

import os
import random
import time
from typing import List, Tuple, Optional, Any, Protocol

import torch


class MCTSSimEnvProtocol(Protocol):
    """
    AlphaZero 型 MCTS が利用する環境インターフェース（フェーズ2, 設計のみ）。

    実装クラスは、以下の操作を提供する必要がある:
    - clone(): 現在の状態から独立なコピーを返す
    - legal_actions(): 現状態の合法手 ID のリストを返す
    - step(action): 指定 action を 1 手適用して状態を進める
    - is_terminal(): 終局かどうかを返す
    - result(): ルート視点の value（例: 勝ち=1, 引き分け=0, 負け=-1）を返す
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


class MCTSNode:
    """
    AlphaZero 型 MCTS 用のノード構造（フェーズ1）。

    ここでは以下のみを保持するシンプルな器として定義する:
    - state_key: 状態識別用のキー（将来、盤面のハッシュなどを入れる想定）
    - parent: 親ノード（ルートの場合は None）
    - children: action_id -> MCTSNode の辞書
    - N: 訪問回数
    - W: 累積価値（root 視点の value の合計）
    - Q: 平均価値（W / N）
    - P: 事前確率（policy ネットからの prior）
    - action_from_parent: 親からこのノードに到達する際に打った行動 ID
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
        """
        子ノードを一つも持たない場合に leaf とみなす。
        """
        return not self.children

class AlphaZeroMCTSPolicy:
    """
    AlphaZero 風の policy+value ネットワークを用いた方策クラス（簡易版）。

    現時点では、legal_action_ids からエンコードした candidate ベクトルと obs_vec から
    PolicyValueNet を用いて logits を計算し、softmax により π（行動分布）を求めて
    1手をサンプリングする「モデルのみ方策」として動作する。
    本格的な MCTS（木探索）はまだ実装しておらず、将来的に PolicyValueNet + MCTS に
    差し替えることを想定している。
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

        # encode_action_from_vec_32d 等の「候補 → 埋め込み」エンコーダを受け取るためのフック
        self.action_encoder_fn = None

        # サンプリング制御用パラメータ
        self.temperature: float = 1.0   # 1.0 で従来どおり、<1.0 で鋭く、>1.0 でフラットに
        self.greedy: bool = False       # True のときは argmax で選択（サンプリングしない）

        # --- MCTS 関連ハイパーパラメータ（フェーズ1: まだ select_action からは未使用） ---
        self.num_simulations: int = 50
        self.c_puct: float = 1.5
        self.dirichlet_alpha: float = 0.3
        self.dirichlet_eps: float = 0.25

        # --- MCTS ゲート（環境変数から上書き可能） ---
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

        if os.getenv("AZ_DECISION_LOG", "0") == "1":
            try:
                print(f"[AZ][INIT] use_mcts={int(self.use_mcts)} sims={int(self.num_simulations)}", flush=True)
                print(f"[AZ][FILECHECK] __file__={__file__} has_CALL_RUN_MCTS_ALWAYS={('CALL_RUN_MCTS_ALWAYS' in open(__file__, encoding='utf-8', errors='ignore').read())} co_consts_has_CALL_RUN_MCTS_ALWAYS={('CALL_RUN_MCTS_ALWAYS' in (getattr(getattr(getattr(getattr(self, 'select_action', None), '__func__', getattr(self, 'select_action', None)), '__code__', None), 'co_consts', ()) or ())) }", flush=True)
            except Exception:
                pass

        self._load_model()

    def _load_model(self) -> None:
        """
        model_dir 配下から PolicyValueNet のチェックポイントを探してロードする。
        現時点では、train_selfplay_supervised.py が保存した .pt を想定している。
        """
        try:
            path = self.model_filename
            if not os.path.isabs(path):
                path = os.path.join(self.model_dir, path)

            if not os.path.exists(path):
                # デフォルト名が無ければ、model_dir 内の最初の .pt を探す
                cand_path = None
                try:
                    for name in sorted(os.listdir(self.model_dir)):
                        if name.lower().endswith(".pt"):
                            cand_path = os.path.join(self.model_dir, name)
                            break
                except Exception:
                    cand_path = None

                if cand_path is None or not os.path.exists(cand_path):
                    print(f"[AlphaZeroMCTSPolicy][WARN] model file not found in {self.model_dir}")
                    return
                path = cand_path

            data = torch.load(path, map_location=self.device)
            if isinstance(data, dict) and "model_state_dict" in data:
                state_dict = data["model_state_dict"]
                self.obs_dim = int(data.get("obs_dim", 0) or 0)
                self.cand_dim = int(data.get("cand_dim", 0) or 0)
                self.hidden_dim = int(data.get("hidden_dim", 256) or 256)
            else:
                # 古い形式など、state_dict だけが保存されているケース
                state_dict = data
                self.obs_dim = None
                self.cand_dim = None
                self.hidden_dim = 256

            try:
                # PolicyValueNet は train_selfplay_selfsupervised.py 内で定義されている前提
                from train_selfplay_supervised import PolicyValueNet

                if self.obs_dim is None or self.cand_dim is None or self.obs_dim <= 0 or self.cand_dim <= 0:
                    print(
                        "[AlphaZeroMCTSPolicy][WARN] obs_dim / cand_dim がチェックポイントに含まれていません。"
                        " モデルはロードしますが、select_action ではランダム方策を用います。"
                    )
                    self.model = None
                else:
                    model = PolicyValueNet(self.obs_dim, self.cand_dim, hidden_dim=self.hidden_dim)
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.model = model
                    print(
                        f"[AlphaZeroMCTSPolicy] loaded model from {path} "
                        f"(obs_dim={self.obs_dim}, cand_dim={self.cand_dim}, hidden_dim={self.hidden_dim})"
                    )
            except Exception as e:
                print(f"[AlphaZeroMCTSPolicy][WARN] failed to construct PolicyValueNet from checkpoint: {e}")
                self.model = None
        except Exception as e:
            print(f"[AlphaZeroMCTSPolicy][WARN] model load failed: {e}")
            self.model = None

    def set_action_encoder(self, fn) -> None:
        """
        ai vs ai.py 側の _attach_action_encoder_if_supported から渡される、
        「候補 → 埋め込みベクトル」エンコーダを受け取るためのフック。
        cand_dim=5 のモデルでは不要（5-int をそのまま候補ベクトルとして使用）。
        cand_dim=32 等のモデルでは必要になる。
        """
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
        """
        row を 5-int の list に正規化する（不足は0埋め、過剰は切り詰め）。
        """
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
                try:
                    out.append(int(x))
                except Exception:
                    out.append(0)
            return out
        except Exception:
            return None

    def _coerce_5int_rows(self, rows):
        """
        rows を list[list[int]]（各行5-int）に正規化する。
        """
        out = []
        try:
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
        except Exception:
            return []
        return out

    def _extract_legal_actions_5(self, state_dict: Any, actions: List[Any], player: Any = None) -> List[List[int]]:
        """
        5-int（5コード）を「必ず」復元する。

        優先順位:
          1) state_dict["legal_actions_5"] / ["legal_actions_vec"] / ["legal_actions"]
          2) match.converter.convert_legal_actions(actions)
          3) Action.to_id_vec()
          4) 最後の保険: actions が list/tuple の場合だけ回収
        """
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
                try:
                    la_5 = self._coerce_5int_rows(fn_la(actions or []))
                except Exception:
                    la_5 = []
                if la_5:
                    return la_5

        tmp = []
        for a in actions or []:
            fn = getattr(a, "to_id_vec", None)
            if callable(fn):
                try:
                    tmp.append(fn())
                    continue
                except Exception:
                    pass
            if isinstance(a, (list, tuple)):
                try:
                    tmp.append(list(a) if isinstance(a, tuple) else a)
                    continue
                except Exception:
                    pass

        la_5 = self._coerce_5int_rows(tmp)

        # ★追加: actions と同じ長さに揃える（1手=1行を保証）
        try:
            n_act = int(len(actions)) if actions is not None else -1
        except Exception:
            n_act = -1

        if n_act >= 0 and len(la_5) != n_act:
            aligned = []
            for a in actions or []:
                row = None
                try:
                    fn = getattr(a, "to_id_vec", None)
                except Exception:
                    fn = None
                if callable(fn):
                    try:
                        row = fn()
                    except Exception:
                        row = None

                if row is None and isinstance(a, (list, tuple)):
                    try:
                        row = list(a) if isinstance(a, tuple) else a
                    except Exception:
                        row = None

                rr = self._coerce_5int_row(row)
                if rr is None:
                    rr = [0, 0, 0, 0, 0]
                aligned.append(rr)

            la_5 = aligned

        return la_5

    def _extract_cand_vecs_for_model(self, state_dict: Any, la_5: List[List[int]]) -> Optional[Any]:
        """
        モデルに渡す cand_vecs を確定する。
        cand_dim=5 のときは必ず 5-int（=la_5）を採用して main が受け取る。
        """
        cdim = int(self.cand_dim or 0)
        if cdim <= 0:
            return None

        if cdim == 5:
            return la_5 if la_5 else None

        # cand_dim != 5 の場合のみ、state_dict の候補（埋め込み済み）を使う余地がある
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

    def _uniform_pi(self, legal_action_ids: List[Any]) -> Tuple[Optional[Any], List[float]]:
        """
        legal_action_ids に対して一様分布の π を返す簡易版。
        """
        n = len(legal_action_ids)
        if n == 0:
            return None, []
        pi = [1.0 / n] * n
        idx = random.randrange(n)
        return legal_action_ids[idx], pi

    def get_obs_vec(self, state_dict: Any = None, actions: Optional[List[Any]] = None, player: Any = None):
        """
        PhaseD-Q / OnlineMixedPolicy 側から呼べる「obs_vec getter」。

        - 目的: self.obs_dim（例: 2448）と一致する obs_vec を list[float] で返す
        - 返せない場合は None
        """
        target_dim = int(self.obs_dim or 0)

        def _dim_ok(v):
            if target_dim <= 0:
                return True
            if not isinstance(v, (list, tuple)):
                return False
            try:
                return len(v) == target_dim
            except Exception:
                return False

        # 1) state_dict 内の既存候補を最優先で拾う
        if isinstance(state_dict, dict):
            for k in ("obs_vec", "obs_vec_az", "az_obs_vec", "observation_vec", "obs", "x"):
                try:
                    vv = state_dict.get(k, None)
                except Exception:
                    vv = None
                vv = self._as_list_vec(vv)
                if vv is not None and _dim_ok(vv):
                    return vv

        # 2) player.match 側に例示がぶら下がっていれば拾う（ai vs ai.py の match.obs_vec_example 等）
        m = getattr(player, "match", None) if player is not None else None
        if m is not None:
            try:
                vv = getattr(m, "obs_vec_example", None)
            except Exception:
                vv = None
            vv = self._as_list_vec(vv)
            if vv is not None and _dim_ok(vv):
                return vv

        # 3) match.encoder を叩いて「2448 が返る呼び方」を探索
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
                    except Exception:
                        out = None
                    vv = self._as_list_vec(out)
                    if vv is not None and _dim_ok(vv):
                        return vv

        # 4) encode_state フォールバック（feat={"me":..., "opp":..., "legal_actions":...} の流儀）
        fn = getattr(enc, "encode_state", None) if enc is not None else None
        if callable(fn) and m is not None and player is not None:
            try:
                me_fn = getattr(player, "public_state", None)
                me = me_fn() if callable(me_fn) else None

                opp = None
                ps = getattr(m, "public_state", None)
                players = ps.get("players") if isinstance(ps, dict) else None
                if not isinstance(players, list) or not players:
                    sp = getattr(m, "starting_player", None)
                    sp2 = getattr(m, "second_player", None)
                    players = [x for x in (sp, sp2) if x is not None]
                for p0 in players or []:
                    if p0 is not None and p0 is not player:
                        opp_fn = getattr(p0, "public_state", None)
                        if callable(opp_fn):
                            opp = opp_fn()
                        break

                la_ids = []
                conv2 = getattr(m, "converter", None)
                if conv2 is None:
                    conv2 = getattr(m, "action_converter", None)
                if conv2 is not None:
                    fn_la = getattr(conv2, "convert_legal_actions", None)
                    if callable(fn_la):
                        try:
                            la_ids = fn_la(actions or [])
                        except Exception:
                            la_ids = []

                if not la_ids:
                    for a in actions or []:
                        try:
                            if hasattr(a, "to_id_vec"):
                                la_ids.append(a.to_id_vec())
                            else:
                                la_ids.append(a if isinstance(a, list) else [int(a)])
                        except Exception:
                            continue

                feat = {"me": me if isinstance(me, dict) else {}, "opp": opp if isinstance(opp, dict) else {}}
                if isinstance(la_ids, list) and la_ids:
                    feat["legal_actions"] = la_ids

                out = fn(feat)
                try:
                    import numpy as np
                    arr = np.asarray(out, dtype=np.float32).reshape(-1)
                    vv = arr.tolist()
                except Exception:
                    vv = self._as_list_vec(out)

                if vv is not None and _dim_ok(vv):
                    return vv
            except Exception:
                pass

        return None

    def encode_obs_vec(self, state_dict: Any = None, actions: Optional[List[Any]] = None, player: Any = None):
        # OnlineMixedPolicy 側から呼びやすい別名
        return self.get_obs_vec(state_dict=state_dict, actions=actions, player=player)

    def select_action(
        self,
        obs_vec,
        legal_action_ids: List[Any],
        env: Optional["MCTSSimEnvProtocol"] = None,
        cand_vecs: Optional[Any] = None,
    ) -> Tuple[Optional[Any], List[float]]:
        """
        AlphaZero 風の行動選択インターフェース。

        基本動作は policy-only:
        obs_vec / legal_action_ids から candidate ベクトルを構築し、
        PolicyValueNet による logits から π を計算して 1手をサンプリングする。

        ただし、use_mcts=True かつ env が与えられ、num_simulations > 0 のときは
        _run_mcts(...) を実行し、visit count ベースの π で上書きする。

        モデルが利用できない場合は、一様ランダム方策にフォールバックする。

        注意:
        - cand_dim=5 のモデルでは、legal_action_ids は 5ints の list を前提として
          そのまま候補ベクトルとして使用する（action_encoder_fn は不要）。
        - cand_dim!=5 のモデルでは、cand_vecs（埋め込み済み）があればそれを使い、
          無ければ action_encoder_fn で埋め込みを作る。
        """
        if not isinstance(legal_action_ids, list) or not legal_action_ids:
            return None, []

        def _set_last(src, pick):
            try:
                self.last_decision_src = str(src)
                self.last_pick = str(pick)
            except Exception:
                pass

        # --- MCTS ゲートの前提を「早期 return 前」に確定ログとして出す ---
        use_mcts = bool(getattr(self, "use_mcts", False))
        num_sims = int(getattr(self, "num_simulations", 0) or 0)
        try:
            env_name_always = type(env).__name__ if env is not None else None
        except Exception:
            env_name_always = None
        try:
            model_ok = int(self.model is not None and self.obs_dim is not None and self.cand_dim is not None and int(self.obs_dim) > 0 and int(self.cand_dim) > 0)
        except Exception:
            model_ok = 0
        if os.getenv("AZ_DECISION_LOG", "0") == "1":
            print(
                f"[AZ][MCTS][PRECHECK_ALWAYS] model_ok={int(model_ok)} use_mcts={int(use_mcts)} env={env_name_always} sims={int(num_sims)} n_actions={int(len(legal_action_ids))}",
                flush=True,
            )

        # モデル / 次元情報が揃っていない場合は一様ランダム
        if self.model is None or self.obs_dim is None or self.cand_dim is None:
            _set_last("uniform", "model_or_dims_missing")
            if os.getenv("AZ_DECISION_LOG", "0") == "1":
                try:
                    print("[AZ][DECISION] src=uniform reason=model_or_dims_missing", flush=True)
                except Exception:
                    pass
            return self._uniform_pi(legal_action_ids)
        if self.obs_dim <= 0 or self.cand_dim <= 0:
            _set_last("uniform", "obs_or_cand_dim_nonpos")
            if os.getenv("AZ_DECISION_LOG", "0") == "1":
                try:
                    print("[AZ][DECISION] src=uniform reason=obs_or_cand_dim_nonpos", flush=True)
                except Exception:
                    pass
            return self._uniform_pi(legal_action_ids)

        try:
            import numpy as np

            # --- obs_vec を (1, obs_dim) のテンソルに変換 ---
            if isinstance(obs_vec, (list, tuple)):
                obs_arr = np.asarray(obs_vec, dtype="float32")
            else:
                obs_arr = np.asarray(obs_vec, dtype="float32")
            if obs_arr.ndim == 0:
                obs_arr = obs_arr.reshape(1)
            if obs_arr.ndim != 1:
                obs_arr = obs_arr.reshape(-1)
            if obs_arr.shape[0] != self.obs_dim:
                print(
                    f"[AlphaZeroMCTSPolicy][WARN] obs_dim mismatch: "
                    f"expected={self.obs_dim}, got={obs_arr.shape[0]}"
                )
                return self._uniform_pi(legal_action_ids)
            obs_tensor = torch.from_numpy(obs_arr).view(1, -1).to(self.device)

            # --- candidate ベクトルを構築 ---
            cand_list = []
            cdim = int(self.cand_dim or 0)

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
                # cand_dim=5 は「5-int をそのまま候補」として使う（ここが main 受領の確定点）
                if cdim == 5:
                    try:
                        for aid in legal_action_ids:
                            enc_arr = np.asarray(aid, dtype="float32").reshape(-1)
                            if enc_arr.shape[-1] < cdim:
                                pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                                enc_arr = np.concatenate([enc_arr.reshape(-1), pad], axis=0)
                            elif enc_arr.shape[-1] > cdim:
                                enc_arr = enc_arr.reshape(-1)[:cdim]
                            cand_list.append(enc_arr)
                    except Exception:
                        cand_list = []
                else:
                    # cand_dim!=5 の場合のみ action_encoder_fn を要求
                    if self.action_encoder_fn is None:
                        print(f"[AZ][MCTS][UNIFORM_EARLY_ALWAYS] file={__file__} reason=no_action_encoder cand_dim={int(cdim)}", flush=True)
                        return self._uniform_pi(legal_action_ids)
                    for aid in legal_action_ids:
                        try:
                            enc = self.action_encoder_fn(aid)
                        except Exception:
                            print(
                                f"[AlphaZeroMCTSPolicy][WARN] action_encoder_fn failed for action={aid}."
                            )
                            return self._uniform_pi(legal_action_ids)

                        enc_arr = np.asarray(enc, dtype="float32").reshape(-1)
                        if enc_arr.shape[-1] < cdim:
                            pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                            enc_arr = np.concatenate([enc_arr, pad], axis=0)
                        elif enc_arr.shape[-1] > cdim:
                            enc_arr = enc_arr[:cdim]
                        cand_list.append(enc_arr)

            if not cand_list:
                return None, []

            cands_arr = np.stack(cand_list, axis=0)
            cands_tensor = torch.from_numpy(cands_arr).to(self.device)

            # --- PolicyValueNet で logits / value を計算 ---
            self.model.eval()
            with torch.no_grad():
                logits_list, values = self.model(obs_tensor, [cands_tensor])

            if not logits_list:
                return self._uniform_pi(legal_action_ids)

            logits = logits_list[0].detach().cpu()
            if logits.ndim != 1 or logits.shape[0] != len(legal_action_ids):
                print(
                    f"[AlphaZeroMCTSPolicy][WARN] logits shape mismatch: "
                    f"shape={tuple(logits.shape)}, n_actions={len(legal_action_ids)}"
                )
                return self._uniform_pi(legal_action_ids)

            # --- softmax で π を計算（温度付き） ---
            temp = float(getattr(self, "temperature", 1.0) or 1.0)
            if temp <= 0.0:
                temp = 1.0
            scaled_logits = logits / temp
            probs_t = torch.softmax(scaled_logits, dim=-1)
            probs = probs_t.numpy().astype("float64")
            s = float(probs.sum())
            if not (s > 0.0):
                if os.getenv("AZ_DECISION_LOG", "0") == "1":
                    try:
                        os.write(2, b"[AZ][DECISION] src=uniform reason=probs_sum<=0\n")
                    except Exception:
                        pass
                return self._uniform_pi(legal_action_ids)
            probs = (probs / s).tolist()

            decision_src = "model"

            try:
                if os.getenv("AZ_DECISION_LOG", "0") == "1":
                    print(
                        f"[AZ][MCTS][BEFORE_GATE_ALWAYS] use_mcts={int(use_mcts)} env={env_name_always} sims={int(num_sims)} n_actions={int(len(legal_action_ids))}",
                        flush=True,
                    )
            except Exception:
                pass

            # --- MCTS が有効な場合は、visit count ベースの π で上書き ---
            if os.getenv("AZ_DECISION_LOG", "0") == "1":
                print(
                    f"[AZ][MCTS][GATE_ALWAYS] use_mcts={int(use_mcts)} env={env_name_always} sims={int(num_sims)} n_actions={int(len(legal_action_ids))}",
                    flush=True,
                )

            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                try:
                    env_name = type(env).__name__ if env is not None else None
                except Exception:
                    env_name = None
                print(
                    f"[AZ][MCTS][GATE] use_mcts={int(use_mcts)} env={env_name} sims={int(num_sims)} n_actions={int(len(legal_action_ids))}",
                    flush=True,
                )

            if not (use_mcts and env is not None and num_sims > 0):
                if os.getenv("AZ_DECISION_LOG", "0") == "1":
                    print("[AZ][MCTS][SKIP_RUN_MCTS_ALWAYS] skip _run_mcts", flush=True)

                if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                    reasons = []
                    try:
                        if not use_mcts:
                            reasons.append("use_mcts=0")
                    except Exception:
                        reasons.append("use_mcts=?")
                    try:
                        if env is None:
                            reasons.append("env=None")
                    except Exception:
                        reasons.append("env=?")
                    try:
                        if int(num_sims) <= 0:
                            reasons.append("sims<=0")
                    except Exception:
                        reasons.append("sims=?")
                    try:
                        if int(len(legal_action_ids)) <= 0:
                            reasons.append("n_actions<=0")
                    except Exception:
                        reasons.append("n_actions=?")

                    rs = ",".join(reasons) if reasons else "unknown"
                    print(f"[AZ][MCTS][SKIP] reasons={rs}", flush=True)

            if use_mcts and env is not None and num_sims > 0:
                print("[AZ][MCTS][CALL_RUN_MCTS_ALWAYS] calling _run_mcts", flush=True)

                if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                    print("[AZ][MCTS][CALL] calling _run_mcts", flush=True)

                _mcc_before_mcts = None
                try:
                    from my_mcc_sampler import mcc_debug_snapshot
                    _mcc_before_mcts = int(mcc_debug_snapshot().get("total_calls", 0))
                except Exception:
                    _mcc_before_mcts = None

                try:
                    mcts_pi = self._run_mcts(env, legal_action_ids, num_simulations=num_sims)
                    if isinstance(mcts_pi, list) and len(mcts_pi) == len(legal_action_ids):
                        probs = mcts_pi
                        decision_src = "mcts"

                    if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                        try:
                            s_pi = float(sum(mcts_pi)) if isinstance(mcts_pi, list) else -1.0
                        except Exception:
                            s_pi = -1.0
                        st = getattr(self, "_last_mcts_stats", None)
                        print(f"[AZ][MCTS][RET] stats={st} pi_sum={s_pi}", flush=True)

                except Exception as e:
                    print(
                        f"[AlphaZeroMCTSPolicy][WARN] _run_mcts failed ({e}); "
                        "fallback to policy-only π."
                    )
                finally:
                    try:
                        from my_mcc_sampler import mcc_debug_snapshot
                        _mcc_after_mcts = int(mcc_debug_snapshot().get("total_calls", 0))
                        if _mcc_before_mcts is not None:
                            d = int(_mcc_after_mcts - _mcc_before_mcts)
                            if d != 0 or (os.getenv("AZ_MCC_LOG_ZERO", "0") == "1"):
                                print(f"[MCC][AZ][MCTS] delta_calls={d} sims={num_sims}", flush=True)
                    except Exception:
                        pass

            # --- 行動選択: greedy or サンプリング ---
            if bool(getattr(self, "greedy", False)):
                idx = int(max(range(len(probs)), key=lambda i: probs[i]))
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

            _set_last(decision_src, pick)

            if os.getenv("AZ_DECISION_LOG", "0") == "1":
                try:
                    a_repr = repr(chosen_action)
                    if len(a_repr) > 160:
                        a_repr = a_repr[:160] + "..."
                except Exception:
                    a_repr = "<unrepr>"
                try:
                    print(
                        f"[AZ][DECISION] src={decision_src} pick={pick} idx={int(idx)} "
                        f"n_actions={int(len(legal_action_ids))} action={a_repr}",
                        flush=True,
                    )
                except Exception:
                    pass

            return chosen_action, probs
        except Exception as e:
            print(f"[AlphaZeroMCTSPolicy][WARN] select_action failed ({e}); fallback to uniform.")
            return self._uniform_pi(legal_action_ids)

    def select_action_index_online(self, state_dict: Any, actions: List[Any], player: Any = None, return_pi: bool = False):
        """
        OnlineMixedPolicy が優先的に呼ぶ入口。
        """
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
        """
        Player.select_action から呼ばれるラッパーメソッド。

        - state_dict / actions から 5-int（legal_actions_5）を必ず復元
        - cand_dim=5 のモデルなら、その 5-int を main が直接受け取る
        - select_action(...) を呼んで chosen_action を得る
        - chosen_action に対応するインデックスを返す
        """
        _mcc_before = None
        try:
            from my_mcc_sampler import mcc_debug_snapshot
            _mcc_before = int(mcc_debug_snapshot().get("total_calls", 0))
        except Exception:
            _mcc_before = None

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

        _decision_src = "unknown"
        _decision_pick = "unknown"
        _mcts_expected = 0
        _mcts_entered = 0
        _mcts_stats_changed = 0
        _env_status = None
        _env_name = None

        def _ret(i, pi):
            try:
                ii = int(i)
            except Exception:
                ii = 0

            if _az_log:
                try:
                    print(
                        f"[AZ][DECISION][RET] game_id={_gid} turn={_turn} idx={int(ii)} src={_decision_src} pick={_decision_pick} "
                        f"mcts_expected={int(_mcts_expected)} mcts_entered={int(_mcts_entered)} mcts_stats_changed={int(_mcts_stats_changed)} "
                        f"env_status={_env_status} env={_env_name} return_pi={int(bool(return_pi))}",
                        flush=True,
                    )
                except Exception:
                    pass

            return (ii, pi) if bool(return_pi) else ii

        def _set_onehot_pi(idx0, pi_from):
            nonlocal _decision_src, _decision_pick
            _decision_src = "uniform"
            _decision_pick = "onehot"
            pi0 = None
            try:
                n = int(len(actions))
                ii = int(idx0)
                pi0 = [0.0] * n
                if 0 <= ii < n:
                    pi0[ii] = 1.0

                if isinstance(state_dict, dict):
                    state_dict["mcts_pi"] = pi0
                    state_dict["pi"] = pi0
                    state_dict["mcts_idx"] = int(ii)
                    state_dict["mcts_pi_present"] = 1
                    state_dict["mcts_pi_len"] = int(len(pi0))
                    state_dict["mcts_pi_type"] = type(pi0).__name__
                    state_dict["mcts_pi_from"] = str(pi_from)

                    state_dict["az_decision_src"] = "uniform"
                    state_dict["az_decision_pick"] = "onehot"
                    state_dict["az_decision_note"] = str(pi_from)
            except Exception:
                pi0 = None

            try:
                self.last_pi = pi0
                self.mcts_pi = pi0
                self.last_mcts_pi = pi0
            except Exception:
                pass

            if _az_log:
                try:
                    print(
                        f"[AZ][DECISION] src=uniform pick=onehot idx={int(idx0)} n_actions={int(len(actions))} note={str(pi_from)}",
                        flush=True,
                    )
                except Exception:
                    pass

            return pi0 if pi0 is not None else []

        try:
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
                _decision_src = "uniform"
                _decision_pick = "fallback_no_actions"
                return _ret(0, [])

            # obs_vec は state_dict["obs_vec"] を想定（無ければ自前で生成も試す）
            obs_vec = None
            if isinstance(state_dict, dict):
                obs_vec = state_dict.get("obs_vec", None)
            if obs_vec is None:
                obs_vec = self.get_obs_vec(state_dict=state_dict, actions=actions, player=player)
            if obs_vec is None:
                idx0 = random.randint(0, len(actions) - 1)

                # ★追加: π を必ず残す（one-hot）
                pi0 = _set_onehot_pi(idx0, "onehot_obs_missing")

                return _ret(idx0, pi0 if pi0 is not None else [])

            # 5-int（5コード）を必ず復元して state_dict に固定（main が読む入口）
            la_5 = self._extract_legal_actions_5(state_dict, actions, player=player)
            if isinstance(state_dict, dict):
                try:
                    state_dict["legal_actions_5"] = la_5
                    state_dict["legal_actions_vec"] = la_5
                    state_dict["legal_actions"] = la_5
                    state_dict["az_la5_n"] = int(len(la_5))
                except Exception:
                    pass

            if not la_5:
                idx0 = random.randint(0, len(actions) - 1)

                # ★追加: π を必ず残す（one-hot）
                pi0 = _set_onehot_pi(idx0, "onehot_la5_missing")

                return _ret(idx0, pi0 if pi0 is not None else [])

            # モデルが受け取る legal_action_ids は「cand_dim=5 なら 5-int をそのまま」
            legal_action_ids: List[Any] = list(la_5)

            # env を必要に応じて組み立て（player.match があれば MatchPlayerSimEnv を利用）
            env = None
            if player is not None and getattr(player, "match", None) is not None:
                try:
                    MatchPlayerSimEnv = None
                    try:
                        from .mcts_env import MatchPlayerSimEnv as _MatchPlayerSimEnv
                        MatchPlayerSimEnv = _MatchPlayerSimEnv
                    except Exception as e_rel:
                        try:
                            from pokepocketsim.policy.mcts_env import MatchPlayerSimEnv as _MatchPlayerSimEnv
                            MatchPlayerSimEnv = _MatchPlayerSimEnv
                        except Exception as e_abs:
                            MatchPlayerSimEnv = None
                            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                                try:
                                    print(f"[AZ][MCTS][ENV_IMPORT_FAIL] rel={repr(e_rel)} abs={repr(e_abs)}", flush=True)
                                except Exception:
                                    pass
                            if isinstance(state_dict, dict):
                                try:
                                    state_dict["az_env_status"] = "import_fail"
                                except Exception:
                                    pass

                    if MatchPlayerSimEnv is not None:
                        env = MatchPlayerSimEnv(player.match, player)
                        if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                            try:
                                print(f"[AZ][MCTS][ENV_OK] env={type(env).__name__}", flush=True)
                            except Exception:
                                pass
                        if isinstance(state_dict, dict):
                            try:
                                state_dict["az_env_status"] = "ok"
                                state_dict["az_env_name"] = type(env).__name__
                            except Exception:
                                pass
                    else:
                        env = None
                        if isinstance(state_dict, dict):
                            try:
                                if state_dict.get("az_env_status", None) is None:
                                    state_dict["az_env_status"] = "none"
                            except Exception:
                                pass
                except Exception as e:
                    env = None
                    if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                        try:
                            print(f"[AZ][MCTS][ENV_BUILD_FAIL] err={repr(e)}", flush=True)
                        except Exception:
                            pass
                    if isinstance(state_dict, dict):
                        try:
                            state_dict["az_env_status"] = "build_fail"
                            state_dict["az_env_err"] = repr(e)
                        except Exception:
                            pass
            else:
                try:
                    why = []
                    if player is None:
                        why.append("player=None")
                    else:
                        try:
                            if getattr(player, "match", None) is None:
                                why.append("player.match=None")
                        except Exception:
                            why.append("player.match=?")
                    rs = ",".join(why) if why else "unknown"
                    if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                        print(f"[AZ][MCTS][ENV_SKIP] reasons={rs}", flush=True)
                    if isinstance(state_dict, dict):
                        try:
                            state_dict["az_env_status"] = "skip"
                            state_dict["az_env_skip_reasons"] = rs
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                if isinstance(state_dict, dict):
                    _env_status = state_dict.get("az_env_status", None)
                    _env_name = state_dict.get("az_env_name", None)
                if _env_name is None and env is not None:
                    _env_name = type(env).__name__
                if _env_status is None:
                    _env_status = "none" if env is None else "ok"
            except Exception:
                pass

            if _az_log:
                try:
                    print(f"[AZ][MCTS][ENV] game_id={_gid} turn={_turn} status={_env_status} env={_env_name}", flush=True)
                except Exception:
                    pass

            # cand_vecs を確定（cand_dim=5 のモデルなら la_5 を必ず採用）
            cand_vecs = self._extract_cand_vecs_for_model(state_dict, la_5)

            use_mcts = bool(getattr(self, "use_mcts", False))
            num_sims = int(getattr(self, "num_simulations", 0) or 0)
            _mcts_expected = 1 if (use_mcts and env is not None and num_sims > 0) else 0

            _t0_before = getattr(self, "_mcts_t0_perf", None)
            _st_before = getattr(self, "_last_mcts_stats", None)
            _st_before_id = id(_st_before)

            _cnt_before = 0
            try:
                _cnt_before = int(getattr(self, "_mcts_enter_counter", 0) or 0)
            except Exception:
                _cnt_before = 0

            if _az_log:
                try:
                    print(
                        f"[AZ][DECISION][CALL_SELECT] game_id={_gid} turn={_turn} use_mcts={int(use_mcts)} sims={int(num_sims)} env={_env_name}",
                        flush=True,
                    )
                except Exception:
                    pass

            if _az_log:
                try:
                    _fn_inst = getattr(self, "select_action", None)
                    _fn_cls = getattr(getattr(self, "__class__", None), "select_action", None)

                    _shadow = False
                    try:
                        _shadow = ("select_action" in getattr(self, "__dict__", {}))
                    except Exception:
                        _shadow = False

                    def _get_code(_fn):
                        _code = None
                        try:
                            _code = getattr(_fn, "__code__", None)
                        except Exception:
                            _code = None
                        if _code is None:
                            try:
                                _code = getattr(getattr(_fn, "__func__", None), "__code__", None)
                            except Exception:
                                _code = None
                        return _code

                    _code_inst = _get_code(_fn_inst)
                    _code_cls = _get_code(_fn_cls)

                    _inst_file = _code_inst.co_filename if _code_inst is not None else "?"
                    _inst_line = int(_code_inst.co_firstlineno) if _code_inst is not None else -1
                    _inst_name = getattr(_fn_inst, "__qualname__", getattr(_fn_inst, "__name__", "select_action"))

                    _cls_file = _code_cls.co_filename if _code_cls is not None else "?"
                    _cls_line = int(_code_cls.co_firstlineno) if _code_cls is not None else -1
                    _cls_name = getattr(_fn_cls, "__qualname__", getattr(_fn_cls, "__name__", "select_action"))

                    print(
                        f"[AZ][DEBUG][SELECT_ACTION_BINDING] game_id={_gid} turn={_turn} "
                        f"shadow={int(_shadow)} "
                        f"inst=file={_inst_file} line={_inst_line} name={_inst_name} "
                        f"cls=file={_cls_file} line={_cls_line} name={_cls_name}",
                        flush=True,
                    )
                except Exception:
                    pass

            chosen_action, pi = self.select_action(obs_vec, legal_action_ids, env=env, cand_vecs=cand_vecs)

            _t0_after = getattr(self, "_mcts_t0_perf", None)
            _st_after = getattr(self, "_last_mcts_stats", None)

            _cnt_after = 0
            try:
                _cnt_after = int(getattr(self, "_mcts_enter_counter", 0) or 0)
            except Exception:
                _cnt_after = 0

            _mcts_entered = 1 if ((_cnt_after != _cnt_before) or (_t0_after is not None and _t0_after != _t0_before)) else 0
            _mcts_stats_changed = 1 if id(_st_after) != _st_before_id else 0

            if _az_log:
                try:
                    print(
                        f"[AZ][MCTS][OBS] game_id={_gid} turn={_turn} expected={int(_mcts_expected)} entered={int(_mcts_entered)} "
                        f"stats_changed={int(_mcts_stats_changed)} cnt_before={int(_cnt_before)} cnt_after={int(_cnt_after)} stats={_st_after}",
                        flush=True,
                    )
                except Exception:
                    pass

            try:
                src0 = str(getattr(self, "last_decision_src", "unknown"))
            except Exception:
                src0 = "unknown"
            try:
                pick0 = str(getattr(self, "last_pick", "unknown"))
            except Exception:
                pick0 = "unknown"

            if src0 in ("mcts", "model", "uniform"):
                _decision_src = src0
            else:
                try:
                    model_ok = int(self.model is not None and self.obs_dim is not None and self.cand_dim is not None and int(self.obs_dim) > 0 and int(self.cand_dim) > 0)
                except Exception:
                    model_ok = 0
                if int(model_ok) == 0:
                    _decision_src = "uniform"
                else:
                    _decision_src = "mcts" if int(_mcts_entered) == 1 else "model"

            if pick0 in ("argmax", "sample"):
                _decision_pick = pick0
            else:
                try:
                    _decision_pick = "argmax" if bool(getattr(self, "greedy", False)) else "sample"
                except Exception:
                    _decision_pick = "unknown"

            try:
                self.last_decision_src = str(_decision_src)
                self.last_pick = str(_decision_pick)
            except Exception:
                pass

            if isinstance(state_dict, dict):
                try:
                    state_dict["az_decision_src"] = str(_decision_src)
                    state_dict["az_decision_pick"] = str(_decision_pick)
                except Exception:
                    pass

            if chosen_action is None:
                idx0 = random.randint(0, len(actions) - 1)
                pi0 = _set_onehot_pi(idx0, "onehot_chosen_action_none")
                return _ret(idx0, pi0)

            try:
                idx = legal_action_ids.index(chosen_action)
            except ValueError:
                idx = random.randint(0, len(actions) - 1)

            # ★追加: π を必ず state_dict / 属性に残す（長さ不一致時は one-hot に退避）
            pi_norm = None
            pi_from = "az_pi"
            try:
                if isinstance(pi, list) and pi:
                    if len(pi) == len(actions):
                        pi_norm = [float(x) for x in pi]
                    else:
                        pi_norm = [0.0] * int(len(actions))
                        pi_norm[int(idx)] = 1.0
                        pi_from = "onehot_pi_len_mismatch"
                else:
                    pi_norm = [0.0] * int(len(actions))
                    pi_norm[int(idx)] = 1.0
                    pi_from = "onehot_pi_missing"
            except Exception:
                pi_norm = None

            try:
                if pi_norm is not None:
                    if isinstance(state_dict, dict):
                        state_dict["mcts_pi"] = pi_norm
                        state_dict["pi"] = pi_norm
                        state_dict["mcts_idx"] = int(idx)
                        state_dict["mcts_pi_present"] = 1
                        state_dict["mcts_pi_len"] = int(len(pi_norm))
                        state_dict["mcts_pi_type"] = type(pi_norm).__name__
                        state_dict["mcts_pi_from"] = str(pi_from)

                        state_dict["az_decision_src"] = str(_decision_src)
                        state_dict["az_decision_pick"] = str(_decision_pick)

                    self.last_pi = pi_norm
                    self.mcts_pi = pi_norm
                    self.last_mcts_pi = pi_norm
            except Exception:
                pass

            if _az_log:
                try:
                    print(
                        f"[AZ][DECISION] src={str(_decision_src)} pick={str(_decision_pick)} idx={int(idx)} n_actions={int(len(actions))} pi_from={str(pi_from)}",
                        flush=True,
                    )
                except Exception:
                    pass

            return _ret(idx, pi_norm if pi_norm is not None else [])
        except Exception as e:
            if _az_log:
                try:
                    print(
                        f"[AZ][DECISION][EXC] entry=select_action_index game_id={_gid} turn={_turn} "
                        f"etype={type(e).__name__} err={repr(e)}",
                        flush=True,
                    )
                    if os.getenv("AZ_DECISION_TRACEBACK", "0") == "1":
                        import traceback
                        print(traceback.format_exc(), flush=True)
                except Exception:
                    pass

            idx0 = 0
            try:
                if isinstance(actions, list) and actions:
                    idx0 = random.randint(0, len(actions) - 1)
            except Exception:
                idx0 = 0

            pi0 = _set_onehot_pi(idx0, "onehot_exception")

            return (idx0, pi0) if bool(return_pi) else idx0
        finally:
            try:
                from my_mcc_sampler import mcc_debug_snapshot
                _mcc_after = int(mcc_debug_snapshot().get("total_calls", 0))
                if _mcc_before is not None:
                    d = int(_mcc_after - _mcc_before)

                    m = getattr(player, "match", None) if player is not None else None
                    gid = getattr(m, "game_id", None) if m is not None else None
                    turn = getattr(m, "turn", None) if m is not None else None

                    if d != 0 or (os.getenv("AZ_MCC_LOG_ZERO", "0") == "1"):
                        print(f"[MCC][AZ][DECISION] game_id={gid} turn={turn} delta_calls={d}", flush=True)
            except Exception:
                pass

    def _run_mcts(
        self,
        env: "MCTSSimEnvProtocol",
        legal_action_ids: List[Any],
        num_simulations: Optional[int] = None,
    ) -> List[float]:
        """
        内部用: MCTS (PUCT) を実行し、ルートにおける visit count に基づく π を返す。

        現時点では以下の制限を持つプレースホルダ実装であり、select_action から
        条件付き（use_mcts=True かつ env があり sims>0）で呼び出される:
        - PolicyValueNet による評価は行わず、すべての prior を一様分布とする
        - leaf ノードの value は 0.0 とし、結果として Q は 0 付近に保たれる

        将来的には env から obs_vec を構築し、PolicyValueNet により
        (prior, value) を得る形に差し替える。
        """
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

        import contextlib
        import io

        @contextlib.contextmanager
        def _mute_stdio():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                yield buf

        n_actions = len(legal_action_ids)
        if n_actions == 0:
            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                try:
                    env_name = type(env).__name__
                except Exception:
                    env_name = None
                print(f"[AZ][MCTS][_run_mcts][SKIP] env={env_name} reason=n_actions==0", flush=True)
            return []

        sims = num_simulations if num_simulations is not None else getattr(self, "num_simulations", 0)

        try:
            self._mcts_t0_perf = time.perf_counter()
        except Exception:
            self._mcts_t0_perf = None

        if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
            try:
                env_name = type(env).__name__
            except Exception:
                env_name = None
            print(f"[AZ][MCTS][_run_mcts][ENTER] env={env_name} sims={int(sims)} n_actions={int(n_actions)}", flush=True)

        if sims <= 0:
            # シミュレーション回数が指定されていない場合は一様分布を返す
            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                try:
                    env_name = type(env).__name__
                except Exception:
                    env_name = None
                print(f"[AZ][MCTS][_run_mcts][SKIP] env={env_name} reason=sims<=0 sims={int(sims)}", flush=True)
            return [1.0 / float(n_actions)] * n_actions

        # ルートノードを構築し、子ノードを一様 prior で初期化
        root = MCTSNode(parent=None, prior=1.0, state_key=None, action_from_parent=None)

        def _akey(a):
            """dict/set のキーに使えるように action_id を hashable 化する。"""
            try:
                hash(a)
                return a
            except Exception:
                pass

            # 典型: 5-int の list（あるいはネスト list）
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

            # 最後の手段: repr に落として同一呼び出し内で安定化
            try:
                return ("repr", repr(a))
            except Exception:
                return ("id", id(a))

        sim_ok = 0
        sim_err = 0

        root_keys = []
        uniform_p = 1.0 / float(n_actions)
        for aid in legal_action_ids:
            child = MCTSNode(parent=root, prior=uniform_p, state_key=None, action_from_parent=aid)
            k = _akey(aid)
            root_keys.append(k)
            root.children[k] = child

        for _ in range(sims):
            try:
                with _mute_stdio():
                    # 環境をクローンして、1 回分のシミュレーションを行う
                    sim_env = env.clone()
                    node = root

                    # --- Selection: 葉ノード or 終局まで PUCT で子を辿る ---
                    while (not node.is_leaf()) and (not sim_env.is_terminal()):
                        total_N = sum(c.N for c in node.children.values()) or 1
                        best_child = None
                        best_score = None
                        for child in node.children.values():
                            u = self.c_puct * child.P * (total_N ** 0.5) / (1.0 + child.N)
                            score = child.Q + u
                            if best_score is None or score > best_score:
                                best_score = score
                                best_child = child
                        if best_child is None:
                            break
                        if best_child.action_from_parent is not None:
                            sim_env.step(best_child.action_from_parent)
                        node = best_child

                    # --- Expansion & Evaluation ---
                    if sim_env.is_terminal():
                        # 終局状態なら env.result() をそのまま leaf value とする
                        leaf_value = float(sim_env.result())
                    else:
                        # まだ終局でなければ、合法手を列挙して一様 prior で子ノードを追加。
                        # 現時点では PolicyValueNet による評価は行わず、leaf value=0.0 とする。
                        next_actions = sim_env.legal_actions()
                        if next_actions:
                            p_child = 1.0 / float(len(next_actions))
                            for aid in next_actions:
                                k = _akey(aid)
                                if k in node.children:
                                    continue
                                child = MCTSNode(parent=node, prior=p_child, state_key=None, action_from_parent=aid)
                                node.children[k] = child
                        leaf_value = 0.0

                    # --- Backup: ルートまで value を逆伝播 ---
                    v = leaf_value
                    cur = node
                    while cur is not None:
                        cur.N += 1
                        cur.W += v
                        cur.Q = cur.W / float(max(1, cur.N))
                        cur = cur.parent

                sim_ok += 1
            except Exception:
                sim_err += 1
                continue

        try:
            self._last_mcts_stats = {"sims": int(sims), "ok": int(sim_ok), "err": int(sim_err)}
        except Exception:
            pass

        # ルートの子ノードの visit count から π を計算
        visit_counts: List[float] = []
        for k in root_keys:
            child = root.children.get(k)
            n = float(child.N) if child is not None else 0.0
            visit_counts.append(n)

        total_visits = sum(visit_counts)
        if total_visits <= 0.0:
            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                st = getattr(self, "_last_mcts_stats", None)
                elapsed_ms = None
                try:
                    t0 = getattr(self, "_mcts_t0_perf", None)
                    if t0 is not None:
                        elapsed_ms = int((time.perf_counter() - float(t0)) * 1000.0)
                except Exception:
                    elapsed_ms = None
                print(
                    f"[AZ][MCTS][_run_mcts][NO_VISITS] sims={int(sims)} ok={int(sim_ok)} err={int(sim_err)} "
                    f"elapsed_ms={elapsed_ms} stats={st}",
                    flush=True,
                )
            return [1.0 / float(n_actions)] * n_actions

        # ★確定ログ: root_total_visits / top3 / elapsed_ms（return直前）
        try:
            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                st = getattr(self, "_last_mcts_stats", None)

                elapsed_ms = None
                try:
                    t0 = getattr(self, "_mcts_t0_perf", None)
                    if t0 is not None:
                        elapsed_ms = int((time.perf_counter() - float(t0)) * 1000.0)
                except Exception:
                    elapsed_ms = None

                topk = 3 if n_actions >= 3 else n_actions
                order = sorted(range(n_actions), key=lambda i: float(visit_counts[i]), reverse=True)[:topk]

                top3_parts = []
                for i in order:
                    try:
                        a = legal_action_ids[i]
                    except Exception:
                        a = None
                    try:
                        a_repr = repr(a)
                        if len(a_repr) > 120:
                            a_repr = a_repr[:120] + "..."
                    except Exception:
                        a_repr = "<unrepr>"
                    try:
                        n_i = float(visit_counts[i])
                    except Exception:
                        n_i = 0.0
                    p_i = n_i / float(total_visits) if float(total_visits) > 0.0 else 0.0
                    top3_parts.append(f"#{i}:N={n_i:.0f},p={p_i:.3f},a={a_repr}")

                print(
                    f"[AZ][MCTS][CONFIRM] sims={int(sims)} ok={int(sim_ok)} err={int(sim_err)} "
                    f"elapsed_ms={elapsed_ms} root_total_visits={float(total_visits):.1f} "
                    f"top3={' | '.join(top3_parts)} stats={st}",
                    flush=True,
                )
        except Exception:
            pass

        return [n / total_visits for n in visit_counts]

    # ======================================================================
    #  以下、AlphaZero 型 MCTS 用の補助メソッド群（フェーズ1: まだ未使用）
    # ======================================================================

    def _create_root_node(
        self,
        state_key: Optional[Any],
        legal_action_ids: List[Any],
        priors: List[float],
    ) -> MCTSNode:
        """
        ルートノードを生成し、legal_action_ids と priors から子ノード群を初期化する。
        ここでは「prior を持つ子ノードだけを生やした空の木」を作るだけで、
        実際のシミュレーション（step/clone）はまだ行わない。
        """
        root = MCTSNode(parent=None, prior=1.0, state_key=state_key, action_from_parent=None)

        if not legal_action_ids:
            return root

        # priors の長さが legal_action_ids と異なる場合は安全側に合わせる
        n = min(len(legal_action_ids), len(priors))
        for i in range(n):
            aid = legal_action_ids[i]
            p = float(priors[i])
            if p < 0.0:
                p = 0.0
            child = MCTSNode(parent=root, prior=p, state_key=None, action_from_parent=aid)
            root.children[aid] = child

        # 正規化は必須ではないが、後段の PUCT の安定のために 1 に揃えておく
        total_p = sum(child.P for child in root.children.values())
        if total_p > 0.0:
            inv_total = 1.0 / total_p
            for child in root.children.values():
                child.P *= inv_total

        return root

    def _select_child(self, node: MCTSNode) -> Tuple[Any, MCTSNode]:
        """
        PUCT ルールにしたがって、与えられた node の子の中から 1 つを選択する。
        ここでは Q + U（U は prior に基づくボーナス）でスコアを計算する。
        """
        if not node.children:
            raise ValueError("_select_child called on a leaf node without children.")

        # 合計訪問回数（親ノードの N）を子ノードの UCT 計算に用いる
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
        """
        与えられたノードに対して、(action_id, prior) のリストから子ノードを生やす。
        すでに存在する子がある場合は prior を上書きするだけで、N/W/Q は変更しない。
        """
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
        """
        ルートから leaf までのノード列（path）に対して value を逆伝播する。
        フェーズ1では「root 視点の value をそのまま全ノードに加算するだけ」の簡易版とし、
        プレイヤー交代による符号反転などはまだ行わない。
        """
        v = float(leaf_value)
        for node in path:
            node.N += 1
            node.W += v
            node.Q = node.W / max(1, node.N)

    def _apply_dirichlet_noise_to_root(self, root: MCTSNode) -> None:
        """
        ルートノードの子ノードの prior P に、Dirichlet ノイズを混ぜる。
        フェーズ1ではまだ呼び出さないが、AlphaZero 標準の「探索の多様性確保」用の器。
        """
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
