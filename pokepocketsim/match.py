import random
from .player import Player
from .attack import Attack
from .action import ActionType
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING
import copy



if TYPE_CHECKING:
    from .action import Action
class Match:
    def __init__(
        self,
        starting_player: Player,
        second_player: Player,
        log_file: Optional[str] = None,
        log_mode: bool = False,
        show_opponent_bench: bool = True,
        game_id: Optional[str] = None,
        ml_log_file: Optional[str] = None,
        use_reward_shaping: bool = True,
        reward_shaping: Optional[Any] = None,
        mcc_top_k: int = 0,
        skip_deckout_logging: bool = False,
    ) -> None:
        starting_player.set_opponent(second_player)
        second_player.set_opponent(starting_player)
        starting_player.match = self
        second_player.match = self
        self.starting_player: Player = starting_player
        self.second_player: Player = second_player

        self.log_file: Optional[str] = log_file
        self.log_mode: bool = log_mode
        self.show_opponent_bench: bool = show_opponent_bench
        self.game_id: Optional[str] = game_id
        self.ml_log_file: Optional[str] = ml_log_file

        self.use_reward_shaping = use_reward_shaping
        self.reward_shaping = reward_shaping
        self.mcc_top_k       = int(mcc_top_k)
        self.skip_deckout_logging: bool = bool(skip_deckout_logging)

        self.turn: int = 0
        self.game_over: bool = False


        def _enum_from_card(obj):
            try:
                enum_ = getattr(obj, "card_enum", None)
                if isinstance(enum_, int):
                    return enum_
                if hasattr(enum_, "value"):
                    v = enum_.value
                    if isinstance(v, (tuple, list)) and v:
                        return int(v[0])
                    if isinstance(v, int):
                        return v
                return int(getattr(obj, "id", 0)) or 0
            except Exception:
                return 0

        for p in (self.starting_player, self.second_player):
            try:
                if not hasattr(p, "initial_deck_enums") or not p.initial_deck_enums:
                    lst = []
                    for c in getattr(p.deck, "cards", []) or []:
                        obj = c[-1] if isinstance(c, list) and c else c
                        lst.append(_enum_from_card(obj))
                    p.initial_deck_enums = lst[:]  # ここでは山札のみ。必要に応じて配布直後に上書き
            except Exception:
                try:
                    p.initial_deck_enums = []
                except Exception:
                    pass

    def _infer_end_reason(self) -> str:
        p1 = self.starting_player
        p2 = self.second_player

        # 1) サイド取り切り（最優先で通常決着）
        try:
            if p1.prize_left() == 0 or p2.prize_left() == 0:
                return "PRIZE_OUT"
        except Exception:
            pass

        # 2) 場にポケモンがいない（ベンチ切れ）
        def _benchless(p):
            return (not p.active_card) and (not getattr(p, "bench", []) or len(p.bench) == 0)
        try:
            if _benchless(p1) or _benchless(p2):
                return "BASICS_OUT"
        except Exception:
            pass

        # 3) 山札切れ
        try:
            d1 = len(getattr(getattr(p1, "deck", None), "cards", []) or [])
            d2 = len(getattr(getattr(p2, "deck", None), "cards", []) or [])
            if d1 == 0 or d2 == 0:
                return "DECK_OUT"
        except Exception:
            pass

        # 4) ターン上限
        if getattr(self, "turn", 0) > 100:
            return "TURN_LIMIT"

        # フォールバック
        return "MATCH_END"

    def log_print(self, *args, **kwargs):
        import os
        import sys
        import re

        def _stdout_is_same_as_logfile() -> bool:
            try:
                if not self.log_file:
                    return False

                def _norm(p: str) -> str:
                    try:
                        return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))
                    except Exception:
                        try:
                            return os.path.normcase(os.path.normpath(str(p)))
                        except Exception:
                            return str(p)

                target = _norm(self.log_file)

                try:
                    so = getattr(sys, "stdout", None)
                except Exception:
                    so = None

                seen = set()
                while so is not None and id(so) not in seen:
                    seen.add(id(so))

                    try:
                        fp = getattr(so, "_fp", None)
                        fp_name = getattr(fp, "name", None) if fp is not None else None
                        if fp_name and _norm(fp_name) == target:
                            return True
                    except Exception:
                        pass

                    try:
                        name = getattr(so, "name", None)
                        if name and _norm(name) == target:
                            return True
                    except Exception:
                        pass

                    try:
                        so = getattr(so, "_base", None)
                    except Exception:
                        so = None

                return False
            except Exception:
                return False

        def _emit_line(line: str):
            print(line, flush=True)
            if self.log_file and (not _stdout_is_same_as_logfile()):
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    print(line, file=f, flush=True)

        try:
            # args 全体から “実際に出る1行” を組み立てて判定（複数引数printにも耐える）
            sep = kwargs.get("sep", " ")
            try:
                s_line = sep.join([str(a) for a in args])
            except Exception:
                s_line = str(args[0]) if args else ""

            # 行動行: "36 p1:" / "1 p2:" のような先頭パターンで判定
            is_action_line = False
            try:
                is_action_line = bool(re.match(r"^\s*\d+\s+p[12]\s*[:：]", s_line))
            except Exception:
                is_action_line = False

            if is_action_line:
                try:
                    ap = getattr(self, "_active_player_for_log", None)
                    pol0 = getattr(ap, "policy", None) if ap is not None else None
                except Exception:
                    pol0 = None

                def _iter_policies(root):
                    seen = set()
                    stack = [(root, 0)]
                    while stack:
                        obj, depth = stack.pop()
                        if obj is None:
                            continue
                        oid = id(obj)
                        if oid in seen:
                            continue
                        seen.add(oid)
                        yield obj, depth
                        try:
                            mp = getattr(obj, "main_policy", None)
                        except Exception:
                            mp = None
                        try:
                            fb = getattr(obj, "fallback_policy", None)
                        except Exception:
                            fb = None
                        if fb is not None:
                            stack.append((fb, depth + 1))
                        if mp is not None:
                            stack.append((mp, depth + 1))

                def _score_decide_lines(obj, depth, pre, post, diff):
                    s = 0
                    txt = ""
                    try:
                        txt = (str(pre or "") + "\n" + str(post or "") + "\n" + str(diff or ""))
                    except Exception:
                        txt = ""
                    try:
                        if "online_mix.main" in txt:
                            s += 200
                        if "source=main" in txt:
                            s += 180
                        if "tag=online_mix.outer" in txt or "source=outer" in txt:
                            s -= 60

                        if ("mcts_sel=" in txt) or ("mcts_ref=" in txt):
                            s += 260
                        if ("lam=" in txt) and ("tau=" in txt) and (("q_span=" in txt) or ("q_top3=" in txt)):
                            s += 120

                        if ("mcts_top3" in txt) or ("q_top3" in txt) or ("mix_top3" in txt):
                            s += 40
                        if diff and ("mcts_idx=" in str(diff)) and ("None" not in str(diff)):
                            s += 80

                        if "changed=NA" in txt:
                            s -= 200
                        if "mcts_idx=None" in txt:
                            s -= 400
                        if "pi_len=NA" in txt:
                            s -= 200
                        if ("base_out_type=int" in txt) and ("pi_len=NA" in txt):
                            s -= 200
                        if "base_out_type=int" in txt:
                            s -= 40
                    except Exception:
                        pass

                    try:
                        _tag = getattr(obj, "_phased_q_tag", None)
                        if _tag is None:
                            _tag = getattr(obj, "phased_q_tag", None)
                        if isinstance(_tag, str):
                            if "main" in _tag:
                                s += 40
                            if "outer" in _tag:
                                s -= 20
                    except Exception:
                        pass

                    s += int(depth)
                    return s

                picked = None
                _pre = _post = _diff = None

                best = None  # (score, pol, pre, post, diff)

                if pol0 is not None:
                    for pol, depth in _iter_policies(pol0):
                        try:
                            pre  = getattr(pol, "_last_decide_pre_line", None)
                            post = getattr(pol, "_last_decide_post_line", None)
                            diff = getattr(pol, "_last_decide_diff_line", None)
                        except Exception:
                            pre = post = diff = None

                        if pre or post or diff:
                            sc = _score_decide_lines(pol, depth, pre, post, diff)
                            if (best is None) or (sc > best[0]):
                                best = (sc, pol, pre, post, diff)

                if best is not None:
                    picked = best[1]
                    _pre, _post, _diff = best[2], best[3], best[4]

                if picked is not None:
                    try:
                        if _pre:
                            _emit_line(_pre)
                        if _post:
                            _emit_line(_post)
                        if _diff:
                            _emit_line(_diff)
                    except Exception:
                        pass

                    try:
                        if _pre is not None:
                            setattr(picked, "_last_decide_pre_line", None)
                        if _post is not None:
                            setattr(picked, "_last_decide_post_line", None)
                        if _diff is not None:
                            setattr(picked, "_last_decide_diff_line", None)
                    except Exception:
                        pass
        except Exception:
            pass

        # 本文の出力（stdout が log_file に流れているなら二重書き込みしない）
        print(*args, **kwargs, flush=True)
        if self.log_file and (not _stdout_is_same_as_logfile()):
            with open(self.log_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f, flush=True)

    def _emit_phased_q_summary_once(self, reason: str = "game_over"):
        # === PhaseD-Q サマリを「集計して1回だけ」出す（use_q=True の回数 = calls_q_used を保証） ===
        try:
            if getattr(self, "_phased_q_summary_emitted", False):
                return
            setattr(self, "_phased_q_summary_emitted", True)

            _reason = str(reason) if reason is not None else "game_over"
            _gid = getattr(self, "game_id", None)
            _gid_s = str(_gid) if _gid is not None else "NA"

            def _iter_policies(root):
                seen = set()
                stack = [root]
                while stack:
                    obj = stack.pop()
                    if obj is None:
                        continue
                    oid = id(obj)
                    if oid in seen:
                        continue
                    seen.add(oid)
                    yield obj

                    for attr in (
                        "main_policy", "fallback_policy",
                        "main", "fallback",
                        "policy", "inner", "wrapped",
                        "main_pol", "fallback_pol",
                    ):
                        try:
                            child = getattr(obj, attr, None)
                        except Exception:
                            child = None
                        if child is not None:
                            stack.append(child)

            agg = {
                "calls_total": 0,
                "calls_q_used": 0,
                "calls_q_eval_none": 0,

                "mix_changed": 0,
                "mix_same": 0,
                "mix_mcts_idx_none": 0,

                "skip_obs_not_numeric": 0,
                "skip_la_list_missing": 0,
                "skip_la_list_empty": 0,
                "skip_cand_vecs_missing": 0,
                "skip_ep_select_action_index_online": 0,

                "pi_changed": 0,
                "pi_l1_n": 0,
                "pi_l1_sum": 0.0,

                "la_len_n": 0,
                "la_len_sum": 0.0,
                "la_len_min": None,
                "la_len_max": None,
            }

            visited = []

            for _pl in (getattr(self, "starting_player", None), getattr(self, "second_player", None)):
                if _pl is None:
                    continue
                _pol = None
                try:
                    _pol = getattr(_pl, "policy", None)
                except Exception:
                    _pol = None
                if _pol is None:
                    try:
                        _pol = getattr(_pl, "ai_policy", None)
                    except Exception:
                        _pol = None
                if _pol is None:
                    try:
                        _pol = getattr(_pl, "pol", None)
                    except Exception:
                        _pol = None

                for _obj in _iter_policies(_pol):
                    visited.append(_obj)

                    st = getattr(_obj, "_phased_q_stats", None)
                    if isinstance(st, dict):
                        try:
                            agg["calls_total"] += int(st.get("calls_total", 0) or 0)
                        except Exception:
                            pass
                        try:
                            _q_used = st.get("calls_q_used", None)
                            if _q_used is None:
                                _q_used = st.get("calls_mix_applied", 0)
                            agg["calls_q_used"] += int(_q_used or 0)
                        except Exception:
                            pass
                        try:
                            agg["calls_q_eval_none"] += int(st.get("calls_q_eval_none", 0) or 0)
                        except Exception:
                            pass

                        try:
                            agg["mix_changed"] += int(st.get("mix_changed", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["mix_same"] += int(st.get("mix_same", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["mix_mcts_idx_none"] += int(st.get("mix_mcts_idx_none", 0) or 0)
                        except Exception:
                            pass

                        try:
                            agg["skip_obs_not_numeric"] += int(st.get("skip_obs_not_numeric", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["skip_la_list_missing"] += int(st.get("skip_la_list_missing", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["skip_la_list_empty"] += int(st.get("skip_la_list_empty", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["skip_cand_vecs_missing"] += int(st.get("skip_cand_vecs_missing", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["skip_ep_select_action_index_online"] += int(st.get("skip_ep_select_action_index_online", 0) or 0)
                        except Exception:
                            pass

                        try:
                            agg["pi_changed"] += int(st.get("pi_changed", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["pi_l1_n"] += int(st.get("pi_l1_n", 0) or 0)
                        except Exception:
                            pass
                        try:
                            agg["pi_l1_sum"] += float(st.get("pi_l1_sum", 0.0) or 0.0)
                        except Exception:
                            pass

                        try:
                            ln = int(st.get("la_len_n", 0) or 0)
                            ls = float(st.get("la_len_sum", 0.0) or 0.0)
                            if ln > 0:
                                agg["la_len_n"] += ln
                                agg["la_len_sum"] += ls
                                _mn = st.get("la_len_min", None)
                                _mx = st.get("la_len_max", None)
                                if _mn is not None:
                                    if agg["la_len_min"] is None or int(_mn) < int(agg["la_len_min"]):
                                        agg["la_len_min"] = int(_mn)
                                if _mx is not None:
                                    if agg["la_len_max"] is None or int(_mx) > int(agg["la_len_max"]):
                                        agg["la_len_max"] = int(_mx)
                        except Exception:
                            pass

            calls_total = int(agg.get("calls_total", 0))
            if calls_total <= 0:
                # 何も呼ばれていないなら出さない（スパム回避）
                return

            q_used = int(agg.get("calls_q_used", 0))
            q_eval_none = int(agg.get("calls_q_eval_none", 0))

            mix_changed = int(agg.get("mix_changed", 0))
            mix_same = int(agg.get("mix_same", 0))
            mix_mcts_idx_none = int(agg.get("mix_mcts_idx_none", 0))
            mix_change_rate = (float(mix_changed) / float(q_used)) if q_used > 0 else 0.0

            pi_changed = int(agg.get("pi_changed", 0))
            pi_l1_n = int(agg.get("pi_l1_n", 0))
            pi_l1_sum = float(agg.get("pi_l1_sum", 0.0))
            pi_l1_avg = (pi_l1_sum / float(pi_l1_n)) if pi_l1_n > 0 else 0.0
            pi_change_rate = (float(pi_changed) / float(pi_l1_n)) if pi_l1_n > 0 else 0.0

            sk_obs = int(agg.get("skip_obs_not_numeric", 0))
            sk_la_missing = int(agg.get("skip_la_list_missing", 0))
            sk_la_empty = int(agg.get("skip_la_list_empty", 0))
            sk_cand = int(agg.get("skip_cand_vecs_missing", 0))
            sk_ep = int(agg.get("skip_ep_select_action_index_online", 0))

            la_n = int(agg.get("la_len_n", 0))
            la_sum = float(agg.get("la_len_sum", 0.0))
            la_min = agg.get("la_len_min", None)
            la_max = agg.get("la_len_max", None)
            la_avg = (la_sum / float(la_n)) if la_n > 0 else 0.0

            self.log_print(
                f"[PhaseD-Q][SUMMARY] game_id={_gid_s} reason={_reason}"
                f" calls_total={calls_total}"
                f" q_used={q_used} q_eval_none={q_eval_none}"
                f" mix_changed={mix_changed} mix_same={mix_same} mix_mcts_idx_none={mix_mcts_idx_none}"
                f" mix_change_rate={mix_change_rate:.3f}"
                f" pi_changed={pi_changed} pi_change_rate={pi_change_rate:.3f} pi_l1_avg={pi_l1_avg:.6f}"
                f" skip_obs_not_numeric={sk_obs} skip_la_missing={sk_la_missing}"
                f" skip_la_empty={sk_la_empty} skip_cand_missing={sk_cand}"
                f" skip_ep_select_action_index_online={sk_ep}"
                f" la_len_avg={la_avg:.2f} la_len_min={la_min} la_len_max={la_max}"
            )

            # atexit の per-policy SUMMARY を抑止し、次ゲームに持ち越さない
            for _obj in visited:
                try:
                    setattr(_obj, "_phased_q_summary_emitted", True)
                except Exception:
                    pass
                try:
                    st = getattr(_obj, "_phased_q_stats", None)
                    if isinstance(st, dict):
                        st.clear()
                except Exception:
                    pass
        except Exception:
            pass

    def log_online_mix_summary_once(self, reason: str = "game_over", winner: str = None) -> None:
        try:
            if getattr(self, "_online_mix_summary_printed", False):
                return
            setattr(self, "_online_mix_summary_printed", True)

            _reason = str(reason) if reason is not None else "game_over"
            _gid = getattr(self, "game_id", None)
            _gid_s = str(_gid) if _gid is not None else "NA"
            _winner = winner if winner is not None else getattr(self, "winner", None)
            _winner_s = str(_winner) if _winner is not None else "NA"

            def _iter_policies(root):
                seen = set()
                stack = [root]
                while stack:
                    obj = stack.pop()
                    if obj is None:
                        continue
                    oid = id(obj)
                    if oid in seen:
                        continue
                    seen.add(oid)
                    yield obj

                    for attr in (
                        "main_policy", "fallback_policy",
                        "main", "fallback",
                        "policy", "inner", "wrapped",
                        "main_pol", "fallback_pol",
                        "base", "_base",
                    ):
                        try:
                            child = getattr(obj, attr, None)
                        except Exception:
                            child = None
                        if child is not None:
                            stack.append(child)

            def _get_policy(_p):
                if _p is None:
                    return None
                for nm in ("policy", "ai_policy", "pol", "agent", "ai"):
                    try:
                        v = getattr(_p, nm, None)
                    except Exception:
                        v = None
                    if v is not None:
                        return v
                return None

            def _pull_counts(_obj):
                total = None
                model = None
                fallback = None
                random_ = None
                errors = None
                main_ok = None
                main_fail = None
                fb_calls = None

                try:
                    total = getattr(_obj, "_sum_total", None)
                except Exception:
                    total = None
                if total is None:
                    try:
                        total = getattr(_obj, "stats_total", None)
                    except Exception:
                        total = None
                if total is None:
                    total = 0

                try:
                    model = getattr(_obj, "_sum_used_main", None)
                except Exception:
                    model = None
                if model is None:
                    try:
                        model = getattr(_obj, "stats_from_model", None)
                    except Exception:
                        model = None
                if model is None:
                    model = 0

                try:
                    fallback = getattr(_obj, "_sum_used_fallback", None)
                except Exception:
                    fallback = None
                if fallback is None:
                    try:
                        fallback = getattr(_obj, "stats_from_fallback", None)
                    except Exception:
                        fallback = None
                if fallback is None:
                    fallback = 0

                try:
                    random_ = getattr(_obj, "_sum_used_random", None)
                except Exception:
                    random_ = None
                if random_ is None:
                    try:
                        random_ = getattr(_obj, "stats_from_random", None)
                    except Exception:
                        random_ = None
                if random_ is None:
                    random_ = 0

                try:
                    errors = getattr(_obj, "stats_errors", None)
                except Exception:
                    errors = None
                if errors is None:
                    errors = 0

                try:
                    main_ok = getattr(_obj, "_main_success_calls", None)
                except Exception:
                    main_ok = None
                if main_ok is None:
                    main_ok = 0

                try:
                    main_fail = getattr(_obj, "_main_failed_calls", None)
                except Exception:
                    main_fail = None
                if main_fail is None:
                    main_fail = 0

                try:
                    fb_calls = getattr(_obj, "_fallback_calls", None)
                except Exception:
                    fb_calls = None
                if fb_calls is None:
                    fb_calls = 0

                try:
                    total = int(total or 0)
                    model = int(model or 0)
                    fallback = int(fallback or 0)
                    random_ = int(random_ or 0)
                    errors = int(errors or 0)
                    main_ok = int(main_ok or 0)
                    main_fail = int(main_fail or 0)
                    fb_calls = int(fb_calls or 0)
                except Exception:
                    return None

                # OnlineMixedPolicy 由来っぽいものだけ拾う（誤検知を避ける最低限の条件）
                if (total + model + fallback + random_ + errors + main_ok + main_fail + fb_calls) <= 0:
                    return None
                return {
                    "total": total,
                    "model": model,
                    "fallback": fallback,
                    "random": random_,
                    "errors": errors,
                    "main_ok": main_ok,
                    "main_fail": main_fail,
                    "fb_calls": fb_calls,
                }

            def _sum_for_player(_pl):
                root = _get_policy(_pl)
                out = {
                    "total": 0,
                    "model": 0,
                    "fallback": 0,
                    "random": 0,
                    "errors": 0,
                    "main_ok": 0,
                    "main_fail": 0,
                    "fb_calls": 0,
                }
                for obj in _iter_policies(root):
                    d = _pull_counts(obj)
                    if d is None:
                        continue
                    for k in out.keys():
                        try:
                            out[k] += int(d.get(k, 0))
                        except Exception:
                            pass
                return out

            p1 = getattr(self, "starting_player", None)
            p2 = getattr(self, "second_player", None)

            s1 = _sum_for_player(p1)
            s2 = _sum_for_player(p2)

            self.log_print(
                f"[ONLINE_MIX][SUMMARY] game_id={_gid_s} reason={_reason} winner={_winner_s}"
                f" p1_total={s1['total']} p1_model={s1['model']} p1_fallback={s1['fallback']} p1_random={s1['random']} p1_errors={s1['errors']}"
                f" p2_total={s2['total']} p2_model={s2['model']} p2_fallback={s2['fallback']} p2_random={s2['random']} p2_errors={s2['errors']}"
                f" main_ok={s1['main_ok'] + s2['main_ok']} main_fail={s1['main_fail'] + s2['main_fail']} fb_calls={s1['fb_calls'] + s2['fb_calls']}"
            )
        except Exception:
            pass

    def calculate_turn_reward(self, active_player: Player, non_active_player: Player) -> Dict[str, float]:
        active_reward = 0.0
        non_active_reward = 0.0

        # 勝利判定
        if self.game_over:
            if active_player.prize_left() == 0:
                active_reward = 1.0
                non_active_reward = 0.0
                self.log_print(f"{active_player.name} が全てのサイドカードを獲得しました。{active_player.name} の勝利です！")
            elif non_active_player.prize_left() == 0:
                active_reward = 0.0
                non_active_reward = 1.0
                self.log_print(f"{non_active_player.name} が全てのサイドカードを獲得しました。{non_active_player.name} の勝利です！")
            elif hasattr(self, 'winner') and self.winner:
                if self.winner == active_player.name:
                    active_reward = 1.0
                    non_active_reward = 0.0
                else:
                    active_reward = 0.0
                    non_active_reward = 1.0


        return {
            active_player.name: active_reward,
            non_active_player.name: non_active_reward
        }

    def setup_phase(self):
        if getattr(self, 'log_mode', False):
            pass
        else:
            if self.starting_player.is_bot and self.second_player.is_bot:
                for player in [self.starting_player, self.second_player]:
                    self.log_print(f'【{player.name}のデッキリスト（60枚）】')
                    all_cards = []
                    all_cards.extend(player.hand)
                    all_cards.extend(player.prize_cards)
                    all_cards.extend(player.deck.cards)
                    for i, card in enumerate(all_cards):
                        self.log_print(f'  {i+1:02d}: {getattr(card, "name", str(card))}')
                    self.log_print('')
        # 人間プレイヤーの特定
        if not self.starting_player.is_bot:
            viewing_player = self.starting_player
        elif not self.second_player.is_bot:
            viewing_player = self.second_player
        else:
            viewing_player = self.starting_player

        self.log_print('---------------------------------------------------')
        self.log_print("【先攻 0ターン】バトル場とベンチをセットします")
        self.starting_player.setup_battle_and_bench(self.second_player, viewing_player=viewing_player)

        self.log_print('---------------------------------------------------')
        self.log_print("【後攻 0ターン】バトル場とベンチをセットします")
        self.second_player.setup_battle_and_bench(self.starting_player, viewing_player=viewing_player)

        # ←← ここから追加：どちらが何回マリガンしたかを明示する
        for p in (self.starting_player, self.second_player):
            if getattr(p, "mulligan_count", 0) > 0:
                self.log_print(f"{p.name} は {p.mulligan_count} 回マリガンしました。")

        # ------★ ここでサイドを作成 ★------
        for player in sorted(
                [self.starting_player, self.second_player],
                key=lambda p: p.mulligan_count):
            if not player.prize_cards:
                player.prize_cards = [player.deck.draw_card() for _ in range(6)]
                if player.print_actions:
                    self.log_print(f"{player.name} はサイドカードを6枚セットしました。")

    def show_setup_status(self):
        if getattr(self, 'log_mode', False):
            return
        if self.starting_player.is_bot and self.second_player.is_bot:
            return
        self.log_print('--------------------------------')
        for player in [self.starting_player, self.second_player]:
            self.log_print(f"{player.name} バトル場: {player.active_card[-1] if player.active_card else 'なし'}")
            if player.bench:
                self.log_print(f"{player.name} ベンチ: {[stack[-1] for stack in player.bench]}")
            else:
                self.log_print(f"{player.name} ベンチ: なし")
        self.log_print('--------------------------------')
        self.log_print('バトル開始！！')

    def _select_viewing_player(self, active_player):
        if not self.starting_player.is_bot:
            return self.starting_player
        elif not self.second_player.is_bot:
            return self.second_player
        else:
            return active_player

    def play_one_match(self, viewing_player=None) -> None:
        try:
            # 追加: 報酬シェイピング（PBRS）を開始
            if (self.use_reward_shaping or getattr(self, "use_mcc", False)) and self.reward_shaping:
                try:
                    self.reward_shaping.reset(self.starting_player, self.second_player, self)
                except TypeError:
                    try:
                        self.reward_shaping.reset(self.starting_player, self)
                    except TypeError:
                        self.reward_shaping.reset()

            # === 追加: 山札切れフィルタの事前設定とバッファ開始 ===
            for _p in (self.starting_player, self.second_player):
                if getattr(_p, "logger", None) is not None:
                    if hasattr(_p.logger, "set_deckout_filter"):
                        _p.logger.set_deckout_filter(self.skip_deckout_logging)
                    if hasattr(_p.logger, "begin_buffering"):
                        _p.logger.begin_buffering()

            # 修正後（match.py / Match.play_one_match 冒頭付近：setup_phase より前へ）

            # === ここからマリガン周りの表示強化 & CPUの選択肢抑止 ===

            # ★追加: 人間だけ候補を出す / CPU は出さない
            self.starting_player.print_actions = (not self.starting_player.is_bot)
            self.second_player.print_actions   = (not self.second_player.is_bot)

            p1 = self.starting_player
            p2 = self.second_player

            # ★追加: encoder / PhaseD-Q 用に match 参照と public_state(players) を用意
            try:
                p1.match = self
                p2.match = self

                if not hasattr(self, "public_state") or not isinstance(getattr(self, "public_state", None), dict):
                    self.public_state = {}
                self.public_state["players"] = [p1, p2]
                self.public_state["turn"] = int(getattr(self, "turn", 0))
            except Exception:
                pass

            self.setup_phase()
            # 終局重複抑止フラグを試合開始時にクリア
            for k in ("_terminal_emitted", "_terminal_counted", "_transition_done", "_phased_q_summary_emitted", "_online_mix_summary_printed"):
                try:
                    if hasattr(self, k):
                        delattr(self, k)
                except Exception:
                    pass
            self.log_print('--------------------------------')

            # ★追加: マリガン回数の表示
            if p1.mulligan_count > 0:
                self.log_print(f"{p1.name} はマリガン {p1.mulligan_count} 回。")
            if p2.mulligan_count > 0:
                self.log_print(f"{p2.name} はマリガン {p2.mulligan_count} 回。")

            if p1.mulligan_count != p2.mulligan_count:
                if p1.mulligan_count < p2.mulligan_count:
                    draw_player = p1
                    diff = p2.mulligan_count - p1.mulligan_count
                    mulliganer = p2
                else:
                    draw_player = p2
                    diff = p1.mulligan_count - p2.mulligan_count
                    mulliganer = p1

                self.log_print(f"{mulliganer.name} のマリガンにより、{draw_player.name} は差分 {diff} 枚を任意でドローできます。")

                for i in range(diff):
                    if not draw_player.is_bot:
                        ans = input(f"{i+1}枚目を引きますか？ (y/n): ")
                        if ans.lower() != 'y':
                            continue
                    card = draw_player.deck.draw_card()
                    if card:
                        draw_player.hand.append(card)
                        if viewing_player is not None and draw_player == viewing_player:
                            self.log_print(f"{draw_player.name} はカードを1枚引きました: {getattr(card, 'name', str(card))}")
                        else:
                            self.log_print(f"{draw_player.name} はカードを1枚引きました。")
                        from pokepocketsim.card import Card
                        if isinstance(card, Card) and getattr(card, 'is_basic', False):
                            if not draw_player.is_bot:
                                ans = input(f"引いた {card.name} をベンチに出しますか？ (y/n): ")
                                if ans.lower() == 'y':
                                    if len(draw_player.bench) < 5:
                                        draw_player.bench.append([card])
                                        draw_player.hand.remove(card)
                                        self.log_print(f"{card.name} をベンチに出しました")
                                    else:
                                        self.log_print("ベンチが満員です")
                            else:
                                if len(draw_player.bench) < 5:
                                    draw_player.bench.append([card])
                                    draw_player.hand.remove(card)
                    else:
                        self.log_print("山札が切れています")

            self.show_setup_status()
            while not self.game_over:
                active_player = self.starting_player if self.turn % 2 == 0 else self.second_player
                try:
                    self._active_player_for_log = active_player
                except Exception:
                    pass
                viewing_player = self._select_viewing_player(active_player)

                # ★追加: encoder / PhaseD-Q 用に public_state をターンごとに更新
                try:
                    if not hasattr(self, "public_state") or not isinstance(getattr(self, "public_state", None), dict):
                        self.public_state = {}
                    if "players" not in self.public_state:
                        self.public_state["players"] = [self.starting_player, self.second_player]
                    self.public_state["turn"] = int(getattr(self, "turn", 0))
                except Exception:
                    pass

                # ★追加: ターンごとにも明示
                active_player.print_actions = (not active_player.is_bot)

                # ポリシー割当（AI 使用時）
                if hasattr(self, "policy_p1") and hasattr(self, "policy_p2"):
                    branch = ""
                    name = getattr(active_player, "name", None)

                    if name == "p1":
                        active_player.policy = self.policy_p1
                        branch = "by_name_p1"
                    elif name == "p2":
                        active_player.policy = self.policy_p2
                        branch = "by_name_p2"
                    else:
                        active_player.policy = self.policy_p1 if active_player is self.starting_player else self.policy_p2
                        branch = "by_starting_fallback"

                    # PhaseD-Q WRAP が match を辿れるように保持
                    try:
                        pol = getattr(active_player, "policy", None)
                        setattr(pol, "_match", self)
                    except Exception:
                        pass
                    try:
                        # OnlineMixedPolicy の内側も必要なら渡す
                        if hasattr(pol, "main_policy"):
                            setattr(pol.main_policy, "_match", self)
                        if hasattr(pol, "fallback_policy"):
                            setattr(pol.fallback_policy, "_match", self)
                    except Exception:
                        pass

                    try:
                        dbg_cnt = int(getattr(self, "_policy_assign_debug_count", 0)) + 1
                        setattr(self, "_policy_assign_debug_count", dbg_cnt)
                        if dbg_cnt <= 10:
                            pol = getattr(active_player, "policy", None)
                            print(
                                "[MATCH_POLICY] assign"
                                f" player_name={name}"
                                f" branch={branch}"
                                f" pol_id={id(pol) if pol is not None else None}"
                                f" class={type(pol).__name__ if pol is not None else None}"
                                f" phased_q_wrapped={getattr(pol, '_phased_q_wrapped', False) if pol is not None else None}"
                                f" phased_q_tag={getattr(pol, '_phased_q_tag', None) if pol is not None else None}"
                            )
                    except Exception:
                        pass


                active_player.setup_turn(self, viewing_player=viewing_player)

                ret = active_player.start_turn(self, viewing_player=viewing_player)

                self.game_over = bool(self.game_over or getattr(self, "game_over", False) or ret)
                if not self.game_over:
                    self.starting_player.pokemon_check()
                    self.second_player.pokemon_check()
                    for p in [self.starting_player, self.second_player]:
                        if (p.active_card is None or not p.active_card) and (not p.bench or len(p.bench) == 0):
                            if p.opponent:
                                self.log_print(f"{p.name} の場から全てのポケモンがいなくなりました。{p.opponent.name} の勝利です！")
                                self.game_over = True
                                self.winner = p.opponent.name
                                self._end_reason = "BASICS_OUT"
                            break
                    if active_player.prize_left() == 0:
                        self.game_over = True
                        self.winner = active_player.name
                        self._end_reason = "PRIZE_OUT"
                        self.log_print(f"{active_player.name} が全てのサイドカードを獲得しました。{active_player.name} の勝利です！")
                    elif active_player.opponent and active_player.opponent.prize_left() == 0:
                        self.game_over = True
                        self.winner = active_player.opponent.name
                        self._end_reason = "PRIZE_OUT"
                        self.log_print(f"{active_player.opponent.name} が全てのサイドカードを獲得しました。{active_player.opponent.name} の勝利です！")
                non_active_player = self.second_player if active_player == self.starting_player else self.starting_player
                self.calculate_turn_reward(active_player, non_active_player)
                if self.game_over:
                    # --- 理由を必ず確定（未設定なら推定） ---
                    if not getattr(self, "_end_reason", None):
                        self._end_reason = self._infer_end_reason()
                    r = (getattr(self, "_end_reason", "") or "").upper()


                    # === 終局ログは「ちょうど1回だけ」 ===
                    try:
                        already_logged = any(
                            getattr(pl.logger, "_terminal_logged", False)
                            for pl in (self.starting_player, self.second_player)
                            if hasattr(pl, "logger") and pl.logger
                        )
                        if not already_logged and getattr(active_player, "logger", None):
                            if r in ("DECK_OUT", "TURN_LIMIT"):
                                # --- デッキアウト/ターン上限は meta（ダミー）1本だけ ---
                                active_player.logger.log_terminal_step(reason=r)
                                setattr(self, "_terminal_emitted", True)
                            else:
                                # --- 通常決着（PRIZE_OUT/BASICS_OUT）は 非ダミー done:1 を1本だけ ---
                                #     （最後の行動が拾えていない場合に備えて合成して書く。metaは付けない）
                                try:
                                    pre  = active_player.logger.build_state_snapshot()
                                    post = active_player.logger.build_state_snapshot()
                                    last_actions  = getattr(active_player.logger, "last_legal_actions_before", []) or []
                                    action_result = active_player.logger.build_action_result(
                                        getattr(active_player, "last_action", None)
                                    )
                                    reward = 1.0 if getattr(self, "winner", None) == active_player.name else 0.0
                                    active_player.logger.log_step(
                                        pre_state=pre,
                                        legal_actions=last_actions,
                                        action_result=action_result,
                                        post_state=post,
                                        force_reward=reward,
                                        force_done=1,   # ← 非ダミー終局
                                        meta=None       # ← meta は出さない
                                    )
                                except Exception:
                                    # 合成に失敗しても落とさない
                                    pass
                    except Exception as e:
                        self.log_print(f"[final-log-error] {e}")

                    self.log_print()
                    self.log_print("------ GAME OVER -------")
                    if hasattr(self, 'winner') and self.winner:
                        self.log_print(f"{self.winner} の勝利！")
                    else:
                        self.log_print(f"{active_player.name} won {active_player.points}-{non_active_player.points}")
                    self.log_print(f"after {self.turn} turns")

                    # === OnlineMixedPolicy サマリは「ちょうど1回だけ」 ===
                    try:
                        self.log_online_mix_summary_once(reason=str(r), winner=getattr(self, "winner", None))
                    except Exception:
                        pass

                    # === PhaseD-Q サマリは「ちょうど1回だけ」 ===
                    try:
                        self._emit_phased_q_summary_once(reason=str(r))
                    except Exception:
                        pass

                    break

                if self.turn > 100:
                    self.log_print()
                    self.log_print("Game terminated at turn 100 due to infinite loop")
                    self.game_over = True
                    self._end_reason = "TURN_LIMIT"      # ← ここを追加（上位でも理由参照できるように）
                    # === ターン上限終了時も「1回だけ」フォールバック ===
                    try:
                        already_logged = any(
                            getattr(pl.logger, "_terminal_logged", False)
                            for pl in (self.starting_player, self.second_player)
                            if hasattr(pl, "logger") and pl.logger
                        )
                        if not already_logged:
                            active_player.logger.log_terminal_step(reason="TURN_LIMIT")  # ← 大文字に統一（任意だが推奨）
                            setattr(self, "_terminal_emitted", True)
                    except Exception:
                        pass

                    # === OnlineMixedPolicy サマリは「ちょうど1回だけ」 ===
                    try:
                        self.log_online_mix_summary_once(reason="TURN_LIMIT", winner=getattr(self, "winner", None))
                    except Exception:
                        pass

                    # === PhaseD-Q サマリは「ちょうど1回だけ」 ===
                    try:
                        self._emit_phased_q_summary_once(reason="TURN_LIMIT")
                    except Exception:
                        pass

                self.turn += 1

        finally:
            try:
                reason = str(getattr(self, "_end_reason", "") or "").upper()
                # ★ 通常決着は一切ダミーを出さない
                emit_dummy = (reason in ("DECK_OUT", "TURN_LIMIT"))
                for pl in (self.starting_player, self.second_player):
                    if pl is not None and getattr(pl, "logger", None):
                        if emit_dummy and not getattr(self, "_terminal_emitted", False) and not getattr(pl.logger, "_terminal_logged", False):
                            pl.logger.log_terminal_step(reason=reason)
                            setattr(self, "_terminal_emitted", True)
                        pl.logger.end_buffering(reason or None)
            except Exception as e:
                self.log_print(f"[final-log-error] {e}")

            # === 追加: 例外終了でもサマリが必ず1回出る最終フォールバック ===
            try:
                if not getattr(self, "_online_mix_summary_printed", False):
                    r0 = str(getattr(self, "_end_reason", None) or ("game_over" if getattr(self, "game_over", False) else "finally"))
                    self.log_online_mix_summary_once(reason=r0, winner=getattr(self, "winner", None))
            except Exception:
                pass
            try:
                if not getattr(self, "_phased_q_summary_emitted", False):
                    r0 = str(getattr(self, "_end_reason", None) or ("game_over" if getattr(self, "game_over", False) else "finally"))
                    self._emit_phased_q_summary_once(reason=r0)
            except Exception:
                pass

            # --- (A) 既存のファイルハンドルを閉じる ---
            for attr in ("_log_fp",):
                fp = getattr(self, attr, None)
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass
                    finally:
                        setattr(self, attr, None)

            # --- (B) 各プレイヤの BattleLogger を明示クローズ ---
            for p in (getattr(self, "starting_player", None),
                    getattr(self, "second_player", None)):
                if p is not None and getattr(p, "logger", None) is not None:
                    if hasattr(p.logger, "close"):
                        try:
                            p.logger.close()
                        except Exception:
                            pass

            # 終局フラグのお掃除（次の試合へ持ち越さない）
            for k in ("_terminal_emitted", "_terminal_counted", "_transition_done", "_online_mix_summary_printed"):
                try:
                    if hasattr(self, k):
                        delattr(self, k)
                except Exception:
                    pass


    def serialize(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "player1": self.starting_player.serialize(),
            "player2": self.second_player.serialize(),
        }

    def simulate_turn_actions(self, player: "Player") -> List[Tuple[int, List["Action"]]]:
        match_copy = copy.deepcopy(self)
        match_copy.turn += 1
        player_copy = copy.deepcopy(player)
        player_copy.print_actions = False
        player_copy.evaluate_actions = False
        player_copy.has_added_energy = False
        player_copy.has_used_trainer = False

        if player_copy.active_card:
            player_copy.active_card[-1].update_conditions()
            if match_copy.turn > 2:
                player_copy.active_card[-1].can_evolve = True
        elif match_copy.turn > 2:
            if len(player_copy.bench) == 0:
                raise Exception("Player lost this turn")
            else:
                player_copy.set_active_card_from_bench(random.choice(player_copy.bench))
        drawn_card = player_copy.deck.draw_card()
        if drawn_card is not None:
            player_copy.hand.append(drawn_card)

        all_sequences: List[Tuple[int, List["Action"]]] = []
        self._simulate_recursive(
            match_copy,
            player_copy,
            current_sequence=[],
            all_sequences=all_sequences,
            depth=0,
        )

        unique_sequences = []
        seen_sequences = set()
        for sequence in all_sequences:
            sequence_tuple = tuple(action.name for action in sequence[1])
            if sequence_tuple not in seen_sequences:
                seen_sequences.add(sequence_tuple)
                unique_sequences.append(sequence)
        return unique_sequences

    @staticmethod
    def _simulate_recursive(
        match: "Match",
        player: "Player",
        current_sequence: List["Action"],
        all_sequences: List[Tuple[int, List["Action"]]],
        depth: int,
    ) -> None:
        if depth > 10:
            return
        actions = player.gather_actions(match)
        for action in actions:
            player_copy = copy.deepcopy(player)
            match_copy = copy.deepcopy(match)
            new_actions = player_copy.act_and_regather_actions(match_copy, action)
            new_sequence = current_sequence + [action]
            if new_actions and player_copy.can_continue:
                Match._simulate_recursive(
                    match_copy,
                    player_copy,
                    new_sequence,
                    all_sequences,
                    depth=depth + 1,
                )
            else:
                evaluation = Player.evaluate_player(player_copy)
                all_sequences.append((evaluation, new_sequence))

    def __repr__(self) -> str:
        return f"Match(starting_player={self.starting_player.name}, second_player={self.second_player.name}, turn={self.turn})"

    # === 追加: UI向けの公開状態を構築して返す（選択UI・スナップショット用） ===
    def build_public_state_for_ui(self, viewer: "Player") -> Dict[str, Any]:
        # いまの手番プレイヤー名
        current_player = self.starting_player if (self.turn % 2 == 0) else self.second_player
        current_name = getattr(current_player, "name", None)

        def _active_dict(p: "Player"):
            if p.active_card:
                top = p.active_card[-1]
                return _card_public_dict(top)
            return None

        def _bench_list(p: "Player"):
            out = []
            for stk in getattr(p, "bench", []) or []:
                top = stk[-1] if isinstance(stk, list) and stk else None
                out.append(_card_public_dict(top) if top else 0)
            return out

        # 自分（viewer）の公開情報
        me = {
            "active_pokemon": _active_dict(viewer),
            "bench_pokemon": _bench_list(viewer),
            # ★ここが肝：手札/トラッシュは種類情報付きの dict 配列で渡す
            "hand": [_ui_obj(c) for c in getattr(viewer, "hand", [])],
            "discard_pile": [_ui_obj(c) for c in getattr(viewer, "discard_pile", [])],
            "deck_count": len(getattr(getattr(viewer, "deck", None), "cards", []) or []),
        }

        # 相手（公開情報）
        opp = getattr(viewer, "opponent", None)
        if opp:
            opp_state = {
                "active_pokemon": _active_dict(opp),
                "bench_pokemon": _bench_list(opp),
                "hand_count": len(getattr(opp, "hand", []) or []),
                # 相手のトラッシュも並べたいなら _ui_obj にしてOK（任意）
                "discard_pile": [_ui_obj(c) for c in getattr(opp, "discard_pile", [])],
                "deck_count": len(getattr(getattr(opp, "deck", None), "cards", []) or []),
            }
        else:
            opp_state = None

        # スタジアム（どちらから見ても同じ）
        stadium = None
        try:
            stadium = viewer.active_stadium.name if getattr(viewer, "active_stadium", None) else None
        except Exception:
            stadium = None

        # === 追加: PhaseD-Q 用の obs_vec / full_obs_vec を可能なら付与 ===
        obs_vec = None
        full_obs_vec = None

        def _as_list(v):
            try:
                if v is None:
                    return None
                if hasattr(v, "tolist"):
                    v = v.tolist()
                return list(v) if isinstance(v, (list, tuple)) else v
            except Exception:
                return v

        def _is_numeric_vec(v):
            try:
                if not isinstance(v, (list, tuple)) or not v:
                    return False
                for x in list(v)[:16]:
                    float(x)
                return True
            except Exception:
                return False

        try:
            # (1) logger のスナップショットに obs_vec が入っていればそれを優先
            snap = None
            lg = getattr(viewer, "logger", None)
            if lg is not None and hasattr(lg, "build_state_snapshot"):
                try:
                    snap = lg.build_state_snapshot()
                except Exception:
                    snap = None

            if isinstance(snap, dict):
                try:
                    obs_vec = snap.get("obs_vec", None) or snap.get("obs", None) or snap.get("public_obs_vec", None)
                    full_obs_vec = snap.get("full_obs_vec", None) or snap.get("full_obs", None)
                except Exception:
                    obs_vec = None
                    full_obs_vec = None

            obs_vec = _as_list(obs_vec)
            full_obs_vec = _as_list(full_obs_vec)

            # (2) 無ければ converter で生成を試みる（UI辞書を入力として渡す）
            if not _is_numeric_vec(obs_vec):
                conv = None
                try:
                    pol = getattr(viewer, "policy", None)

                    # ★main 優先で辿る（outer を見ない）
                    for _pp in (getattr(pol, "main_policy", None), getattr(pol, "fallback_policy", None), pol):
                        if _pp is None:
                            continue
                        try:
                            conv = getattr(_pp, "converter", None)
                            if conv is None:
                                conv = getattr(_pp, "action_converter", None)
                        except Exception:
                            conv = None
                        if conv is not None:
                            break
                except Exception:
                    conv = None
                if conv is not None:
                    sd_tmp = {
                        "turn": self.turn,
                        "current_player_name": current_name,
                        "me": me,
                        "opp": opp_state,
                        "active_stadium": stadium,
                    }

                    vec = None
                    try:
                        fn = getattr(conv, "encode_state", None)
                        vec = fn(sd_tmp) if callable(fn) else None
                    except Exception:
                        vec = None
                    if vec is None:
                        try:
                            fn = getattr(conv, "convert_state", None)
                            vec = fn(sd_tmp) if callable(fn) else None
                        except Exception:
                            vec = None
                    if vec is None:
                        try:
                            fn = getattr(conv, "build_obs", None)
                            vec = fn(sd_tmp) if callable(fn) else None
                        except Exception:
                            vec = None

                    vec = _as_list(vec)

                    if isinstance(vec, dict):
                        for _k in ("obs_vec", "obs", "public_obs_vec", "full_obs_vec", "x", "vec"):
                            if _k in vec:
                                vec = _as_list(vec.get(_k, None))
                                break

                    if _is_numeric_vec(vec):
                        obs_vec = list(vec)

            if not _is_numeric_vec(obs_vec):
                obs_vec = None
            if full_obs_vec is not None and not _is_numeric_vec(full_obs_vec):
                full_obs_vec = None
        except Exception:
            obs_vec = None
            full_obs_vec = None

        return {
            "turn": self.turn,
            "t": self.turn,
            "game_id": getattr(self, "game_id", None),
            "current_player_name": current_name,
            "player_name": getattr(viewer, "name", None),
            "me": me,
            "opp": opp_state,
            "active_stadium": stadium,
            "obs_vec": obs_vec,
            "full_obs_vec": full_obs_vec,
        }

# === UI表示用ヘルパー（0:ポケモン,1:グッズ,2:サポ,3:どうぐ,4:スタジアム,5:その他,6:エネルギー） ===
def _ui_cat(card):
    try:
        if getattr(card, "hp", None) is not None or getattr(card, "is_pokemon", False): return 0
        if getattr(card, "is_item", False):       return 1
        if getattr(card, "is_supporter", False):  return 2
        if getattr(card, "is_tool", False):       return 3
        if getattr(card, "is_stadium", False):    return 4
        name = getattr(card, "name", "")
        if getattr(card, "is_energy", False) or ("エネルギー" in name): return 6
        return 5
    except Exception:
        return 5

def _ui_obj(card):
    nm = getattr(card, "name", str(card))
    cat = _ui_cat(card)
    d = {"name": nm, "cat": cat}
    # 互換性のため is_* も付与（プレゼン側のヒューリスティクスに効く）
    if cat == 0: d["is_pokemon"]   = True
    if cat == 1: d["is_item"]      = True
    if cat == 2: d["is_supporter"] = True
    if cat == 3: d["is_tool"]      = True
    if cat == 4: d["is_stadium"]   = True
    if cat == 6: d["is_energy"]    = True
    return d

def _card_public_dict(card_top):
    # バトル場/ベンチの1枚（進化スタックのトップ）をUI向けに最小限で辞書化
    if card_top is None:
        return None
    try:
        tools = [t.name for t in getattr(card_top, "tools", [])] if getattr(card_top, "tools", None) else []
        energies = []
        for e in getattr(card_top, "attached_energies", []) or []:
            nm = getattr(e, "name", None)
            if nm:
                energies.append(nm)
        return {
            "name": getattr(card_top, "name", str(card_top)),
            "hp": getattr(card_top, "hp", None),
            "energies": energies,
            "tools": tools,
        }
    except Exception:
        return {"name": getattr(card_top, "name", str(card_top))}
