import copy, random
from collections import Counter

_MCC_DEBUG = {
    "total_calls": 0,        # mcc_sampler が呼ばれた回数（=サンプル回数）
    "match_calls": {},       # {game_id: calls}
    "top_k_hist": {},        # {top_k_value: calls}
}

def _mcc_debug_bump(game_id, top_k):
    try:
        _MCC_DEBUG["total_calls"] += 1
        if game_id is not None:
            _MCC_DEBUG["match_calls"][game_id] = _MCC_DEBUG["match_calls"].get(game_id, 0) + 1
        _MCC_DEBUG["top_k_hist"][int(top_k)] = _MCC_DEBUG["top_k_hist"].get(int(top_k), 0) + 1
    except Exception:
        pass

def mcc_debug_snapshot():
    """現在までの MCC 利用状況のスナップショットを返す（読み取り専用想定）"""
    return {
        "total_calls": _MCC_DEBUG["total_calls"],
        "num_matches": len(_MCC_DEBUG["match_calls"]),
        "per_match_calls": dict(_MCC_DEBUG["match_calls"]),
        "top_k_hist": dict(_MCC_DEBUG["top_k_hist"]),
    }

def reset_mcc_debug():
    """カウンタをゼロクリア"""
    _MCC_DEBUG["total_calls"] = 0
    _MCC_DEBUG["match_calls"].clear()
    _MCC_DEBUG["top_k_hist"].clear()

def _cards_to_enums(cards):
    # Card.card_enum が int 前提（別の形なら合わせて変えてください）
    out = []
    for c in cards:
        obj = c[-1] if isinstance(c, list) and c else c

        v = getattr(obj, "card_enum", None)
        if isinstance(v, int):
            out.append(int(v))
            continue
        if hasattr(v, "value"):
            vv = v.value
            if isinstance(vv, (tuple, list)) and vv:
                try:
                    out.append(int(vv[0]))
                    continue
                except Exception:
                    pass
            if isinstance(vv, int):
                out.append(int(vv))
                continue

        vid = getattr(obj, "id", 0)
        try:
            out.append(int(vid or 0))
        except Exception:
            out.append(0)

    return out

def _bag_from_enums(enum_list):
    """
    enum の多重集合を [[cid, cnt], ...] に変換（順不同）
    """
    cnt = Counter([cid for cid in enum_list if isinstance(cid, int) and cid != 0])
    return [[cid, int(n)] for cid, n in cnt.items()]

# --- 追加ヘルパ（既存のインデント・空行・コメント位置は維持） ---
def _bag_to_flat(bag):
    arr = []
    for cid, n in bag:
        arr.extend([cid] * int(n))
    return arr

def _bag_from_counts_dict(d):
    return [[int(cid), int(n)] for cid, n in d.items() if isinstance(cid, int) and n > 0]

def _bag_subtract(bag, sub_counts):
    if not sub_counts:
        return bag[:]
    left = Counter({cid: n for cid, n in bag})
    for cid, n in sub_counts.items():
        if isinstance(cid, int):
            left[cid] -= int(n)
    return [[cid, int(n)] for cid, n in left.items() if n > 0]

def _sample_without_replacement_from_bag(bag, k):
    flat = _bag_to_flat(bag)
    if k <= 0 or not flat:
        return [], bag[:]
    k = min(k, len(flat))
    picks = random.sample(flat, k)
    sub = Counter(picks)
    new_bag = _bag_subtract(bag, sub)
    return picks, new_bag

def _normalize_weights(ws):
    tot = sum(max(0.0, float(w)) for w in ws) or 1.0
    return [max(0.0, float(w)) / tot for w in ws]

def _sample_deck_from_pool(pool_counts_dict, size=60):
    bag = _bag_from_counts_dict(pool_counts_dict or {})
    flat = _bag_to_flat(bag)
    if not flat:
        return []
    if len(flat) <= size:
        # プールが小さすぎる場合は全採用（足りない分は重複を許さず size 未満で妥協）
        return flat[:size]
    return random.sample(flat, size)

def mcc_sampler(base_state: dict, ctx: dict) -> dict:
    s = dict(base_state)
    s["me_private"]  = dict(base_state.get("me_private")  or {})
    s["opp_private"] = dict(base_state.get("opp_private") or {})
    match  = ctx.get("match")
    me_p   = ctx.get("player")
    opp_p  = getattr(me_p, "opponent", None) if me_p is not None else None

    # TopN の長さ（未設定なら 0）
    top_k = int(ctx.get("top_k", getattr(match, "mcc_top_k", 0)) or 0)
    if top_k < 0:
        top_k = 0

    # --- MCC デバッグ: 呼ばれたことを記録（_phi_mcc から K 回呼ばれる想定） ---
    _mcc_debug_bump(getattr(match, "game_id", None), top_k)

    # --- 自分側（完全情報をそのまま入れる） ---
    if me_p is not None:
        allow_peek_me = bool(ctx.get("allow_peek_me", False))
        if allow_peek_me:
            my_deck_ids  = _cards_to_enums(getattr(me_p.deck, "cards", []))         # 残り山札
            my_prize_ids = _cards_to_enums(getattr(me_p, "prize_cards", []))        # サイド6枚
            if top_k > 0:
                s["me_private"]["deck_topN_head_enum"] = my_deck_ids[:top_k]
            else:
                s["me_private"].setdefault("deck_topN_head_enum", [])
            s["me_private"]["deck_bag_counts"] = _bag_from_enums(my_deck_ids)
            s["me_private"]["prize_enum"]      = my_prize_ids


        # --- 追加：覗き禁止時は既知デッキから条件付きサンプルに切替 ---
        my_known_list = ctx.get("my_known_decklist_enums")  # 60枚の enum 配列
        if not allow_peek_me and isinstance(my_known_list, list) and len(my_known_list) >= 60:
            known_counts = Counter(int(x) for x in my_known_list if isinstance(x, int))
            # 公開済み（自分の手札/場/トラッシュ等）の枚数を差し引く
            my_public_counts = Counter(ctx.get("my_public_enum_counts", {}))
            bag = _bag_subtract(_bag_from_counts_dict(known_counts), my_public_counts)
            # プライズ6枚をサンプル（既知の公開プライズがあれば除外済を期待）
            sampled_prize, bag_after_prize = _sample_without_replacement_from_bag(bag, 6)
            s["me_private"]["prize_enum"] = sampled_prize
            # TopN（山札の先頭）をサンプル（ここでは表示のみ、bag は消費しない）
            if top_k > 0:
                head, _ = _sample_without_replacement_from_bag(bag_after_prize, top_k)
                s["me_private"]["deck_topN_head_enum"] = head
            else:
                s["me_private"].setdefault("deck_topN_head_enum", [])
            # 残り山札のバッグ（プライズ差引後）
            s["me_private"]["deck_bag_counts"] = bag_after_prize
        elif not allow_peek_me:
            # 必要情報不足
            s["me_private"].setdefault("deck_topN_head_enum", [])
            s["me_private"].setdefault("deck_bag_counts", [])
            s["me_private"].setdefault("prize_enum", [])

    # --- 相手側（MCCで補完する） ---
    if opp_p is not None:
        allow_peek_opp = bool(ctx.get("allow_peek_opp", False))
        if allow_peek_opp:
            opp_deck_ids = _cards_to_enums(getattr(opp_p.deck, "cards", []))  # 残り山札（順序は未知）


            # 既に base_state に制約がある場合は尊重（なければサンプル）
            if not s["opp_private"].get("deck_topN_head_enum"):
                pool = opp_deck_ids[:]
                random.shuffle(pool)  # 重複対応のためシャッフル→スライス
                s["opp_private"]["deck_topN_head_enum"] = pool[:top_k] if top_k > 0 else []

            # 多重集合（bag）は残り山札のカウント
            s["opp_private"]["deck_bag_counts"] = _bag_from_enums(opp_deck_ids)

            # 手札 / サイドは未知 → 既存が無ければ空（必要なら別途ジェネレータで推定）
            s["opp_private"].setdefault("hand_enum", [])
            s["opp_private"].setdefault("prize_enum", [])

        # --- 追加：アーキタイプ候補と観測に基づく条件付きサンプル ---
        if not allow_peek_opp:
            opp_arch = ctx.get("opp_archetypes")  # [{"enums":[…60], "weight":0.6}, ...]
            if isinstance(opp_arch, list) and opp_arch:
                weights = _normalize_weights([a.get("weight", 1.0) for a in opp_arch])
                r = random.random()
                cum = 0.0
                pick = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        pick = i
                        break
                enums60 = opp_arch[pick].get("enums") or []
                if isinstance(enums60, list) and len(enums60) >= 60:
                    base_counts = Counter(int(x) for x in enums60 if isinstance(x, int))
                    # 観測で公開された枚数を差し引く
                    obs = Counter(ctx.get("opp_observed_enum_counts", {}))
                    bag = _bag_subtract(_bag_from_counts_dict(base_counts), obs)
                    # 既知プライズ（公開済み）があればさらに差し引く
                    known_prize = ctx.get("opp_prize_known_enums", []) or []
                    if known_prize:
                        bag = _bag_subtract(bag, Counter(known_prize))
                    # サンプル対象サイズ
                    hand_size   = int(ctx.get("opp_hand_size", 0) or 0)
                    prize_unk   = int(ctx.get("opp_prize_unknown", 0) or 0)
                    # プライズ（不明分のみ）サンプル
                    sampled_prize, bag = _sample_without_replacement_from_bag(bag, prize_unk)
                    # 手札サンプル（サイズが与えられていれば）
                    sampled_hand, bag2 = _sample_without_replacement_from_bag(bag, hand_size) if hand_size > 0 else ([], bag)
                    # TopN（山札の先頭）サンプル
                    if top_k > 0:
                        head, _ = _sample_without_replacement_from_bag(bag2, top_k)
                        s["opp_private"]["deck_topN_head_enum"] = head
                    else:
                        s["opp_private"].setdefault("deck_topN_head_enum", [])
                    # 結果を反映
                    s["opp_private"]["deck_bag_counts"] = bag2
                    if sampled_hand:
                        s["opp_private"]["hand_enum"] = sampled_hand
                    if sampled_prize or known_prize:
                        s["opp_private"]["prize_enum"] = (known_prize + sampled_prize)

            else:
                # --- 追加：アーキタイプ不明 → カードプールから60枚デッキをサンプル ---
                pool_counts = ctx.get("opp_card_pool_counts")  # {card_id: max_count}
                if isinstance(pool_counts, dict) and pool_counts:
                    sampled_deck_enums = _sample_deck_from_pool(pool_counts, 60)
                    base_counts = Counter(int(x) for x in sampled_deck_enums if isinstance(x, int))
                    # 観測で公開された枚数を差し引く
                    obs = Counter(ctx.get("opp_observed_enum_counts", {}))
                    bag = _bag_subtract(_bag_from_counts_dict(base_counts), obs)
                    # 既知プライズ（公開済み）があればさらに差し引く
                    known_prize = ctx.get("opp_prize_known_enums", []) or []
                    if known_prize:
                        bag = _bag_subtract(bag, Counter(known_prize))
                    # サンプル対象サイズ
                    hand_size   = int(ctx.get("opp_hand_size", 0) or 0)
                    prize_unk   = int(ctx.get("opp_prize_unknown", 6) or 0)
                    # プライズ（不明分のみ）サンプル
                    sampled_prize, bag = _sample_without_replacement_from_bag(bag, prize_unk)
                    # 手札サンプル（サイズが与えられていれば）
                    sampled_hand, bag2 = _sample_without_replacement_from_bag(bag, hand_size) if hand_size > 0 else ([], bag)
                    # TopN（山札の先頭）サンプル
                    if top_k > 0:
                        head, _ = _sample_without_replacement_from_bag(bag2, top_k)
                        s["opp_private"]["deck_topN_head_enum"] = head
                    else:
                        s["opp_private"].setdefault("deck_topN_head_enum", [])
                    # 結果を反映
                    s["opp_private"]["deck_bag_counts"] = bag2
                    if sampled_hand:
                        s["opp_private"]["hand_enum"] = sampled_hand
                    if sampled_prize or known_prize:
                        s["opp_private"]["prize_enum"] = (known_prize + sampled_prize)
                else:
                    # 必要情報不足
                    s["opp_private"].setdefault("deck_topN_head_enum", [])
                    s["opp_private"].setdefault("deck_bag_counts", [])
                    s["opp_private"].setdefault("hand_enum", [])
                    s["opp_private"].setdefault("prize_enum", [])

    return s
