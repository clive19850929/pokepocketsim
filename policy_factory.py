"""
policy_factory.py

このファイルは ai vs ai.py（commit: 0b758a8aa4caf5425e7f07f8dd4f5632e08fc1e7）から
「policy 構築系（PhaseD-Q 混合ラッパ + AlphaZero/MCTS/online_mix の build_policy）」を分離したものです。

- ai vs ai.py から `from policy_factory import build_policy` で読み込み、worker.py 側が __main__ 参照しても動くようにします。
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR and _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import (
    AZ_MCTS_NUM_SIMULATIONS,
    LOG_DEBUG_DETAIL,
    SELFPLAY_ALPHAZERO_MODE,
    USE_MCTS_POLICY,
)

try:
    from phaseD_q import (
        USE_PHASED_Q,
        PHASED_Q_MIX_ENABLED,
        PHASED_Q_MIX_LAMBDA,
        PHASED_Q_MIX_TEMPERATURE,
        phaseD_q_load_if_needed,
        phaseD_q_evaluate,
        phaseD_mix_pi_with_q,
    )
except Exception as _e:
    USE_PHASED_Q = False
    PHASED_Q_MIX_ENABLED = False
    PHASED_Q_MIX_LAMBDA = 0.0
    PHASED_Q_MIX_TEMPERATURE = 1.0

    def phaseD_q_load_if_needed(*args, **kwargs):
        return None

    def phaseD_q_evaluate(*args, **kwargs):
        return None

    def phaseD_mix_pi_with_q(pi, *args, **kwargs):
        return pi

    try:
        if LOG_DEBUG_DETAIL:
            print(f"[policy_factory] phaseD_q import failed: {_e}")
    except Exception:
        pass

def _resolve_action_encoder_fn32_from_env(tag: str = ""):
    """
    AZ_ACTION_ENCODER_FN="module:function" を唯一の正として解決する。
    自動探索はしない。解決できなければ理由を出して例外停止する（呼び出し側で）。
    """
    spec = ""
    try:
        spec = str(os.getenv("AZ_ACTION_ENCODER_FN", "") or "").strip()
    except Exception:
        spec = ""

    if not spec:
        return None, "AZ_ACTION_ENCODER_FN is empty"

    if ":" not in spec:
        return None, "AZ_ACTION_ENCODER_FN must be 'module:function'"

    mod_name, fn_name = spec.split(":", 1)
    mod_name = str(mod_name or "").strip()
    fn_name = str(fn_name or "").strip()

    if not mod_name or not fn_name:
        return None, "AZ_ACTION_ENCODER_FN must be 'module:function' (both non-empty)"

    try:
        import importlib
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None, f"import_module failed: {type(e).__name__} {e!r}"

    fn = None
    try:
        fn = getattr(mod, fn_name, None)
    except Exception:
        fn = None

    if not callable(fn):
        return None, f"resolved attr is not callable: module={mod_name} attr={fn_name}"

    # ここでは「呼べる」ことまで（戻りlen=32検証は action_id が要るので select_action 側で落ちる）
    return fn, "ok"

def _resolve_action_encoder_fn32_from_config(tag: str = ""):
    """
    config.py から fn32 を解決する（唯一の正）。
    - AZ_ACTION_ENCODER_FN32: callable を直接渡す
    - AZ_ACTION_ENCODER_FN32_SPEC: "module:function" で指定する
    どちらも無ければエラー（自動探索・ダミー注入は禁止）。
    """
    _fn = None
    _spec = ""
    _fn_err = ""
    _spec_err = ""

    try:
        from config import AZ_ACTION_ENCODER_FN32 as _C_FN
        _fn = _C_FN
    except Exception as e:
        _fn = None
        _fn_err = f"{type(e).__name__} {e!r}"

    if callable(_fn):
        return _fn, "ok(callable)"

    try:
        from config import AZ_ACTION_ENCODER_FN32_SPEC as _C_SPEC
        _spec = str(_C_SPEC or "").strip()
    except Exception as e:
        _spec = ""
        _spec_err = f"{type(e).__name__} {e!r}"

    if not _spec:
        return None, (
            "Missing AZ_ACTION_ENCODER_FN32 (callable) and AZ_ACTION_ENCODER_FN32_SPEC('module:function') "
            f"in config.py (callable_err={_fn_err!r}, spec_err={_spec_err!r})"
        )

    # SPEC から import
    if ":" not in _spec:
        return None, f"Invalid AZ_ACTION_ENCODER_FN32_SPEC format (expected 'module:function', got={_spec!r})"

    mod_name, fn_name = _spec.split(":", 1)
    mod_name = mod_name.strip()
    fn_name = fn_name.strip()

    # placeholder 防止（昔のテンプレが残っていた場合）
    if mod_name == "module" or mod_name.startswith("module."):
        return None, (
            "AZ_ACTION_ENCODER_FN32_SPEC uses placeholder module name 'module'. "
            "Set a real importable module path (e.g., 'your_pkg.your_mod:your_fn'). "
            f"(got={_spec!r})"
        )

    try:
        import importlib
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None, f"import_module failed: {type(e).__name__} {e!r} (spec={_spec!r})"

    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        return None, f"resolved attr is not callable: module={mod_name} attr={fn_name} (spec={_spec!r})"

    return fn, "ok(spec)"

def _attach_action_encoder_fn32_required(pol, tag: str = ""):
    """
    cand_dim!=5 の AlphaZeroMCTSPolicy に、fn(action_id)->list[float](len=cand_dim) を必須注入する。
    受け取れない場合は原因ログを出して即死する（自動探索・ダミー注入・自動次元合わせは禁止）。
    """
    cdim = 0
    try:
        cdim = int(getattr(pol, "cand_dim", 0) or 0)
    except Exception:
        cdim = 0

    if int(cdim) == 5:
        try:
            print(f"[AZ][ENCODER][SKIP] tag={tag} cand_dim=5 (fn32 not required)", flush=True)
        except Exception:
            pass
        return

    if int(cdim) <= 0:
        try:
            print(f"[AZ][ENCODER][FATAL] tag={tag} cand_dim<=0 cand_dim={cdim} (model metadata missing)", flush=True)
        except Exception:
            pass
        raise RuntimeError(f"[AZ][ENCODER][FATAL] cand_dim<=0 cand_dim={cdim}")

    fn, reason = _resolve_action_encoder_fn32_from_config(tag=tag)

    if fn is None:
        try:
            cwd = os.getcwd()
        except Exception:
            cwd = "<?>"
        try:
            exe = sys.executable
        except Exception:
            exe = "<?>"
        try:
            import config as _cfg
            cfg_file = getattr(_cfg, "__file__", None)
        except Exception:
            cfg_file = None

        try:
            print(
                f"[AZ][ENCODER][FATAL] tag={tag} cand_dim={int(cdim)} reason={reason} "
                f"cwd={cwd!r} exe={exe!r} policy_factory={__file__!r} config_file={cfg_file!r}",
                flush=True,
            )
        except Exception:
            pass

        try:
            head = sys.path[:12]
            print(f"[AZ][ENCODER][FATAL] sys.path.head={head!r}", flush=True)
        except Exception:
            pass

        raise RuntimeError(f"[AZ][ENCODER][FATAL] fn32 unresolved: {reason}")

    setter = getattr(pol, "set_action_encoder", None)
    if not callable(setter):
        try:
            print(
                f"[AZ][ENCODER][FATAL] tag={tag} cand_dim={int(cdim)} "
                f"policy_has_set_action_encoder=0 class={type(pol).__name__} file={__file__!r}",
                flush=True,
            )
        except Exception:
            pass
        raise RuntimeError("[AZ][ENCODER][FATAL] policy has no set_action_encoder(fn)")

    def _wrapped(action_id):
        v = fn(action_id)

        if isinstance(v, tuple):
            v = list(v)

        if not isinstance(v, list):
            try:
                tl = getattr(v, "tolist", None)
            except Exception:
                tl = None
            if callable(tl):
                try:
                    v = tl()
                except Exception:
                    v = None
                if isinstance(v, tuple):
                    v = list(v)

        if not isinstance(v, list):
            try:
                v = list(v)
            except Exception:
                v = None

        if (not isinstance(v, list)) or (len(v) != int(cdim)):
            raise RuntimeError(
                f"[AZ][ENCODER][FATAL] tag={tag} encoder returned invalid vector "
                f"(need len={int(cdim)}) got={type(v)} len={(len(v) if isinstance(v, list) else 'NA')}"
            )

        out = []
        for x in v:
            out.append(float(x))
        return out

    try:
        _wrapped(0)
    except Exception as e:
        try:
            print(
                f"[AZ][ENCODER][FATAL] tag={tag} cand_dim={int(cdim)} probe_failed err={type(e).__name__} {e!r}",
                flush=True,
            )
        except Exception:
            pass
        raise

    setter(_wrapped)

    # --- 必須: 注入した fn32 が「本当に呼べて」「len=32 を返す」ことをここで確定させる ---
    _probe_aid = 0
    try:
        from config import AZ_ACTION_ENCODER_PROBE_ACTION_ID as _PROBE
        _probe_aid = int(_PROBE)
    except Exception:
        _probe_aid = 0

    try:
        _probe_v = fn(int(_probe_aid))
    except Exception as e:
        try:
            print(
                f"[AZ][ENCODER][FATAL] tag={tag} cand_dim={int(cdim)} "
                f"probe_aid={int(_probe_aid)} probe_exc={e!r}",
                flush=True,
            )
        except Exception:
            pass
        raise RuntimeError(f"[AZ][ENCODER][FATAL] fn32 probe failed: {e!r}")

    if isinstance(_probe_v, tuple):
        _probe_v = list(_probe_v)

    if not isinstance(_probe_v, list) or len(_probe_v) != 32:
        try:
            _t = type(_probe_v).__name__
            _l = (len(_probe_v) if isinstance(_probe_v, list) else None)
        except Exception:
            _t = "<?>"
            _l = None
        try:
            print(
                f"[AZ][ENCODER][FATAL] tag={tag} cand_dim={int(cdim)} "
                f"probe_aid={int(_probe_aid)} probe_type={_t} probe_len={_l} (expected_len=32)",
                flush=True,
            )
        except Exception:
            pass
        raise RuntimeError("[AZ][ENCODER][FATAL] fn32 must return list[float] len=32")

    try:
        for _x in _probe_v:
            float(_x)
    except Exception as e:
        try:
            print(
                f"[AZ][ENCODER][FATAL] tag={tag} cand_dim={int(cdim)} "
                f"probe_aid={int(_probe_aid)} non_numeric={e!r}",
                flush=True,
            )
        except Exception:
            pass
        raise RuntimeError("[AZ][ENCODER][FATAL] fn32 must return numeric list[float] len=32")

    try:
        setattr(pol, "_policy_factory_fn32_injected", True)
    except Exception:
        pass

    try:
        import inspect
        mod = inspect.getmodule(fn)
        mod_file = getattr(mod, "__file__", None) if mod is not None else None
    except Exception:
        mod_file = None

    try:
        code = getattr(fn, "__code__", None)
        fn_file = code.co_filename if code is not None else "<?>"
        fn_line = int(code.co_firstlineno) if code is not None else -1
    except Exception:
        fn_file = "<?>"
        fn_line = -1

    try:
        print(
            f"[AZ][ENCODER][OK] tag={tag} cand_dim={int(cdim)} "
            f"fn={getattr(fn, '__qualname__', getattr(fn, '__name__', 'fn'))} "
            f"fn_file={fn_file!r} fn_line={int(fn_line)} module_file={mod_file!r}",
            flush=True,
        )
    except Exception:
        pass

from pokepocketsim.policy.random_policy import RandomPolicy

def _coerce_vec_dim(v, dim):
    """
    v を list[float] に正規化し、長さ dim に揃える（不足は 0.0 埋め、過剰は切り詰め）。
    v が None / 変換不能なら None を返す。
    """
    if v is None:
        return None

    try:
        d = int(dim)
    except Exception:
        d = 0
    if d <= 0:
        return None

    # list / tuple
    if isinstance(v, tuple):
        v = list(v)

    # numpy / torch / array-like (tolist)
    if not isinstance(v, list):
        try:
            tl = getattr(v, "tolist", None)
        except Exception:
            tl = None
        if callable(tl):
            try:
                vv = tl()
            except Exception:
                vv = None
            if isinstance(vv, tuple):
                vv = list(vv)
            v = vv

    if not isinstance(v, list):
        return None

    out = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)

    if len(out) < d:
        out += [0.0] * (d - len(out))
    elif len(out) > d:
        out = out[:d]

    return out

def _resolve_action_encoder_callable_from_env():
    """
    AZ_ACTION_ENCODER_FN="module:function" を最優先で解決する。
    """
    spec = None
    try:
        spec = os.getenv("AZ_ACTION_ENCODER_FN", None)
        if spec is not None:
            spec = str(spec).strip()
    except Exception:
        spec = None

    if not spec:
        return None, None

    if ":" not in spec:
        return None, f"bad_spec(no_colon):{spec}"

    mod_name, fn_name = spec.split(":", 1)
    mod_name = (mod_name or "").strip()
    fn_name = (fn_name or "").strip()
    if not mod_name or not fn_name:
        return None, f"bad_spec(empty):{spec}"

    try:
        import importlib
        m = importlib.import_module(mod_name)
        fn = getattr(m, fn_name, None)
        if callable(fn):
            return fn, f"env:{spec}"
        return None, f"not_callable:{spec}"
    except Exception as e:
        return None, f"import_fail:{spec} err={e!r}"

def _attach_action_encoder_fn32_if_needed(pol, model_dir, tag=""):
    """
    互換用エントリポイント。
    現行方針では env 参照や次元自動補正（ゼロ埋め/切り詰め）を禁止し、
    config.py の指定を唯一の正として必須注入する。
    """
    _attach_action_encoder_fn32_required(pol, tag=tag)
    return

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
        # Prefer instance-level binding to the core implementation so later class-level monkeypatch
        # cannot break env= / cand_vecs= keyword arguments.
        try:
            import types as _types
            if _pol is not None and hasattr(_pol, "_select_action_core"):
                if not bool(getattr(_pol, "_select_action_shadowed", False)):
                    try:
                        _pol.select_action = _types.MethodType(_pol.__class__._select_action_core, _pol)
                        _pol._select_action_shadowed = True
                        if str(_tag):
                            _pol._select_action_shadow_tag = str(_tag)
                    except Exception:
                        pass
        except Exception:
            pass

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
        except Exception:
            pass

        try:
            _fn_inst = getattr(_pol, "select_action", None)
            _fn_cls = getattr(getattr(_pol, "__class__", None), "select_action", None)

            _shadow = False
            try:
                _shadow = ("select_action" in getattr(_pol, "__dict__", {}))
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
                f"[POLICY_FACTORY][DEBUG][SELECT_ACTION_BINDING] tag={_tag} "
                f"shadow={int(_shadow)} "
                f"inst=file={_inst_file} line={_inst_line} name={_inst_name} "
                f"cls=file={_cls_file} line={_cls_line} name={_cls_name}",
                flush=True,
            )
        except Exception:
            pass

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

            # ★必須: cand_dim!=5 のとき fn32 を環境変数から解決して注入（失敗ならここで即死）
            _attach_action_encoder_fn32_required(pol, tag="az_mcts(selfplay)")

            # MCTS 有効モード: policy 側のフラグを ON（シミュレーション回数は先頭設定から取得）
            setattr(pol, "use_mcts", True)
            try:
                sims = int(os.getenv("AZ_MCTS_NUM_SIMULATIONS", str(AZ_MCTS_NUM_SIMULATIONS)))
            except Exception:
                sims = 64
            setattr(pol, "num_simulations", sims)

            _attach_action_encoder_fn32_if_needed(pol, model_dir=model_dir, tag="az_mcts(selfplay)")
            _ensure_az_select_action_env_compatible(pol, _tag="az_mcts(selfplay)")

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

            # ★必須: cand_dim!=5 のとき fn32 を環境変数から解決して注入（失敗ならここで即死）
            _attach_action_encoder_fn32_required(pol, tag=f"az({'model_only' if is_model_only else 'mcts'})")

            if is_model_only:
                # ★モデルのみモード（MCTSを無効化するためのフラグをポリシー側に渡す）
                setattr(pol, "disable_mcts", True)
                # MCTS を明示的に無効化
                setattr(pol, "use_mcts", False)
                setattr(pol, "num_simulations", 0)
                print("[POLICY] using AlphaZeroMCTSPolicy in model_only mode (MCTS disabled if supported)")
                _attach_action_encoder_fn32_if_needed(pol, model_dir=model_dir, tag="az_model_only")
                _ensure_az_select_action_env_compatible(pol, _tag="az_model_only")
            else:
                # ★az_mcts/mcts 指定なら Selfplayフラグ無しでも MCTS を有効化する
                setattr(pol, "use_mcts", True)
                try:
                    sims = int(AZ_MCTS_NUM_SIMULATIONS)
                except Exception:
                    sims = 64
                setattr(pol, "num_simulations", sims)
                print(f"[POLICY] using AlphaZeroMCTSPolicy in MCTS mode (sims={sims})")
                _attach_action_encoder_fn32_if_needed(pol, model_dir=model_dir, tag="az_mcts(normal)")
                _ensure_az_select_action_env_compatible(pol, _tag="az_mcts(normal)")
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

            # ★必須: cand_dim!=5 のとき fn32 を環境変数から解決して注入（失敗ならここで即死）
            _attach_action_encoder_fn32_required(main_pol, tag="online_mix.main")

            setattr(main_pol, "use_mcts", True)
            try:
                sims = int(AZ_MCTS_NUM_SIMULATIONS)
            except Exception:
                sims = 64
            setattr(main_pol, "num_simulations", sims)

            _attach_action_encoder_fn32_if_needed(main_pol, model_dir=model_dir, tag="online_mix.main")

            # --- 重要: select_action の env= / cand_vecs= 互換を保証（instance に shadow binding） ---
            try:
                _ensure_az_select_action_env_compatible(main_pol, _tag="online_mix.main")
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
            try:
                _ensure_az_select_action_env_compatible(main_pol, _tag="online_mix.main(after_wrap)")
            except Exception:
                pass


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
