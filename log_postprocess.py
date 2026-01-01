#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import random
import math
from collections import Counter, defaultdict
import orjson

# Defaults are configured from the runner (ai vs ai*.py).
RAW_JSONL_PATH = None
IDS_JSONL_PATH = None
PRIVATE_IDS_JSON_PATH = None
JSONL_ROTATE_LINES = None
FULL_VEC_VERSION = None

def configure_defaults(**kwargs):
    """Set module-level defaults (safe no-op for None values)."""
    for k, v in kwargs.items():
        if v is not None:
            globals()[k] = v

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

