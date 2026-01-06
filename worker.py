from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.queues import Queue


# ============================================================
# worker.py（ai vs ai.py から分離）
#
# Windows の multiprocessing(spawn) で安全に動かすため、
# 子プロセス側では __main__（= __mp_main__）にある ai vs ai.py の
# グローバル定義（設定値・ユーティリティ関数・import 済みモジュール等）を
# 取り込み、元の worker 実装をそのまま実行します。
# ============================================================

def _bind_main_globals() -> None:
    """
    __main__（= ai vs ai.py / spawn 時は __mp_main__）のグローバルを
    この worker モジュールへコピーして、元の worker 実装が参照する
    変数・関数・モジュール名を解決できるようにする。
    """
    import __main__ as _m
    g = globals()
    for k, v in _m.__dict__.items():
        if k.startswith("__"):
            continue
        if k in ("_bind_main_globals", "play_continuous_matches_worker"):
            continue
        g[k] = v


def play_continuous_matches_worker(num_matches: int, queue: "Queue", mcc_agg=None, gpu_req_q=None, run_game_id=None) -> None:
    """
    既存 play_continuous_matches のワーカープロセス版。
    ファイルは開かず、各試合ぶんの raw_lines / id_lines / priv_id_lines を "batch" として queue に送る。
    """
    _bind_main_globals()
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

    def _wiring_where(obj) -> str:
        try:
            import inspect
            if obj is None:
                return "None"
            if callable(obj):
                fn = obj
            else:
                fn = getattr(obj, "__class__", type(obj))
            mod = inspect.getmodule(fn)
            file = inspect.getsourcefile(fn) or ""
            line = 0
            try:
                line = int(inspect.getsourcelines(fn)[1])
            except Exception:
                line = 0
            name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
            modn = getattr(mod, "__name__", None)
            return f"name={name} file={file} line={line} mod={modn}"
        except Exception:
            try:
                return repr(obj)
            except Exception:
                return "<unrepr>"

    if os.getenv("AZ_DUMP_WIRING", "0") == "1":
        try:
            print(f"[WIRING][worker] get_converter: {_wiring_where(get_converter)}", flush=True)
            print(f"[WIRING][worker] converter: {_wiring_where(converter)}", flush=True)
            print(f"[WIRING][worker] converter.convert_record: {_wiring_where(getattr(converter, 'convert_record', None))}", flush=True)
            print(f"[WIRING][worker] converter.convert_legal_actions: {_wiring_where(getattr(converter, 'convert_legal_actions', None))}", flush=True)
            print(f"[WIRING][worker] local_policy_p1: {_wiring_where(local_policy_p1)}", flush=True)
            print(f"[WIRING][worker] local_policy_p2: {_wiring_where(local_policy_p2)}", flush=True)
            enc1 = getattr(local_policy_p1, "encoder", None)
            enc2 = getattr(local_policy_p2, "encoder", None)
            print(f"[WIRING][worker] p1.encoder: {_wiring_where(enc1)}", flush=True)
            print(f"[WIRING][worker] p2.encoder: {_wiring_where(enc2)}", flush=True)
        except Exception:
            pass

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
                    reward_shaping = importlib.import_module("pokepocketsim.reward_shaping")
                except Exception:
                    reward_shaping = importlib.import_module("reward_shaping")

                try:
                    _mcc_sampler = importlib.import_module("pokepocketsim.my_mcc_sampler").mcc_sampler
                except Exception:
                    _mcc_sampler = importlib.import_module("my_mcc_sampler").mcc_sampler

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

