"""
obs_vector.py

ai vs ai.py（commit: 74fc3a670356ca80e56a42839a0eb43166e98e98）から
公開情報 state_before を obs_vec（list[float]）へ変換する処理を分離したモジュール。

- ai vs ai.py から `from obs_vector import build_obs_partial_vec, set_card_id2idx` で利用する。
- set_card_id2idx() でカード語彙（card_id -> index）を注入してから build_obs_partial_vec() を呼ぶ。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_CARD_ID2IDX: Dict[int, int] = {}

def set_card_id2idx(card_id2idx: Optional[Dict[int, int]]):
    global _CARD_ID2IDX
    _CARD_ID2IDX = dict(card_id2idx or {})

def build_obs_partial_vec(sb_pub: dict):
    """
    sb_pub: state_before の公開版（me / opp のみ）を想定
    返り値: list[float]

    構成イメージ:
      - me_hand_slots_vec        : 30スロット × Vloc（位置付き手札 one-hot）
      - me_board_vec             : 自分アクティブ＋ベンチのポケモンID one-hot
      - me_discard_vec           : 自分トラッシュ内カードID one-hot
      - opp_board_vec            : 相手アクティブ＋ベンチのポケモンID one-hot
      - opp_discard_vec          : 相手トラッシュ内カードID one-hot
      - me_energy_vec            : 自分の場（アクティブ＋ベンチ）に付いているエネルギーカードID one-hot
      - opp_energy_vec           : 相手の場に付いているエネルギーカードID one-hot
      - me_active_energy_type_vec: 自分アクティブに付いているエネルギーの「種類」 one-hot（card_id ベース）
      - opp_active_energy_type_vec: 相手アクティブに付いているエネルギーの種類 one-hot
      - me_bench_energy_slots_vec: ベンチごとのエネルギー種類分布（max 8 スロット × Vloc）
      - opp_bench_energy_slots_vec: 相手ベンチごとのエネルギー種類分布
      - me_bench_board_slots_vec : ベンチごとのポケモンID分布（max 8 スロット × Vloc）
      - opp_bench_board_slots_vec: 相手ベンチごとのポケモンID分布

      - scalars                  : 11スカラー
                                   (turn, cur_is_me, cur_is_opp,
                                    me/opp prize, me/opp deck,
                                    me/opp hand_count,
                                    me/opp discard_pile_count)
      - me_status_vec            : 自分アクティブの特殊状態 one-hot（どく/やけど/マヒ/ねむり/こんらん）
      - opp_status_vec           : 相手アクティブの特殊状態 one-hot
      - extra_scalars            : 18スカラー
                                   (me/opp HP, me/opp エネ枚数, me/opp ベンチ数,
                                    me/opp damage_counters,
                                    me/opp has_weakness, me/opp has_resistance,
                                    me/opp is_basic, me/opp is_ex,
                                    me/opp retreat_cost)
      - type_vecs                : 40 次元
                                   (me_active_type(10), opp_active_type(10),
                                    me_weak_type(10),  opp_weak_type(10),
                                    me_resist_type(10),opp_resist_type(10))  ※ type -1 は all-zero
      - ability_flags            : 4スカラー
                                   (me ability_present/used, opp ability_present/used)
      - tools_vec                : 2 × Vloc
                                   (me_active_tools_vec, opp_active_tools_vec)
      - stadium_vec              : スタジアムカードID one-hot
                                   （top-level stadium と各 side の active_stadium を統合）
      - turn_flags               : 8スカラー
                                   (me: energy_attached, supporter_used, stadium_played, stadium_effect_used,
                                    opp: 同様 4 フラグ)
      - prev_ko_flag             : 1スカラー（前のターンで自分ポケモンが気絶したか）

    ※ エネルギーやタイプは card_id/type_id ベースで区別（基本エネ／特殊エネの違いは card_id か type_id に反映されている前提）。
    """
    import numpy as _np
    Vloc = len(_CARD_ID2IDX)
    TYPE_DIM = 10  # タイプID 0〜9 を one-hot

    if not isinstance(sb_pub, dict):
        # 形が想定外なら空ベクトルを返す（上流でフィルタされる）
        return []

    me_pub  = (sb_pub.get("me")  or {})
    opp_pub = (sb_pub.get("opp") or {})

    # --- 手札 30 スロット one-hot ---
    def _vec_from_hand_slots(hand, V: int, max_slots: int = 30):
        v = _np.zeros(max_slots * V, dtype=_np.float32)
        if isinstance(hand, list):
            for i, x in enumerate(hand):
                if i >= max_slots:
                    break
                try:
                    # [card_id, n] 形式も許容
                    cid = int(x[0] if isinstance(x, (list, tuple)) else x)
                    idx = _CARD_ID2IDX.get(cid, None)
                    if idx is None:
                        continue
                    base = i * V
                    v[base + idx] = 1.0
                except Exception:
                    continue
        return v

    # --- 盤面ポケモンID（アクティブ＋ベンチ） ---
    def _collect_pokemon_board_ids(pub: dict):
        ids = []

        active = pub.get("active_pokemon")
        if isinstance(active, dict):
            try:
                cid = int(active.get("name"))
                ids.append(cid)
            except Exception:
                pass

        bench = pub.get("bench_pokemon")
        if isinstance(bench, list):
            for obj in bench:
                if isinstance(obj, dict):
                    try:
                        cid = int(obj.get("name"))
                        ids.append(cid)
                    except Exception:
                        continue

        return _vec_from_id_list(ids, Vloc)

    # --- ベンチごとのポケモンID分布（max_bench_slots × Vloc） ---
    def _build_bench_board_slots_vec(pub: dict, V: int, max_bench_slots: int = 8):
        v = _np.zeros(max_bench_slots * V, dtype=_np.float32)
        bench = pub.get("bench_pokemon")
        if not isinstance(bench, list):
            return v
        for i, obj in enumerate(bench):
            if i >= max_bench_slots:
                break
            if not isinstance(obj, dict):
                continue
            try:
                cid = int(obj.get("name"))
                idx = _CARD_ID2IDX.get(cid, None)
                if idx is None:
                    continue
                base = i * V
                v[base + idx] = 1.0
            except Exception:
                continue
        return v

    # --- 単一ポケモンからエネルギーカード ID を集める ---
    def _collect_energy_ids_from_poke(poke: dict):
        ids = []
        if not isinstance(poke, dict):
            return ids
        # 可能性のあるキーを全部見る
        for key in ("energies", "energy", "attached_energies", "attached_energy", "energy_cards"):
            lst = poke.get(key)
            if isinstance(lst, list):
                for x in lst:
                    try:
                        cid = int(x[0] if isinstance(x, (list, tuple)) else x)
                        ids.append(cid)
                    except Exception:
                        continue
        return ids

    # --- 場に付いているエネルギーカードID（アクティブ＋ベンチ合算） ---
    def _collect_attached_energy_ids(pub: dict):
        ids = []

        active = pub.get("active_pokemon")
        if isinstance(active, dict):
            ids.extend(_collect_energy_ids_from_poke(active))

        bench = pub.get("bench_pokemon")
        if isinstance(bench, list):
            for obj in bench:
                if isinstance(obj, dict):
                    ids.extend(_collect_energy_ids_from_poke(obj))

        return _vec_from_id_list(ids, Vloc)

    # --- アクティブポケモンの HP / 特殊状態 / エネ枚数 ---
    def _extract_active_features(pub: dict):
        """
        戻り値:
          hp: float
          status_vec: 長さ5（poison/burn/paralysis/sleep/confusion）
          energy_count: float
        """
        hp = 0.0
        status_vec = _np.zeros(5, dtype=_np.float32)
        energy_count = 0.0

        active = pub.get("active_pokemon")
        if isinstance(active, dict):
            # HP（remaining_hp / hp / current_hp のどれかがあれば使う）
            for key in ("remaining_hp", "hp", "current_hp"):
                if key in active:
                    try:
                        hp = float(active.get(key) or 0.0)
                        break
                    except Exception:
                        pass

            # 特殊状態
            raw_cond = active.get("special_conditions") or active.get("conditions") or []
            cond_list = []
            if isinstance(raw_cond, list):
                cond_list = raw_cond
            elif isinstance(raw_cond, str):
                cond_list = [raw_cond]

            def _norm(s):
                if not isinstance(s, str):
                    return ""
                return s.strip().lower()

            for c in cond_list:
                k = _norm(c)
                if k in ("poison", "poisoned", "どく"):
                    status_vec[0] = 1.0
                elif k in ("burn", "burned", "やけど"):
                    status_vec[1] = 1.0
                elif k in ("paralyze", "paralyzed", "paralysis", "マヒ", "まひ"):
                    status_vec[2] = 1.0
                elif k in ("sleep", "asleep", "ねむり"):
                    status_vec[3] = 1.0
                elif k in ("confuse", "confused", "こんらん"):
                    status_vec[4] = 1.0

            # 付いているエネ枚数
            for key in ("energies", "energy", "attached_energies", "attached_energy", "energy_cards"):
                lst = active.get(key)
                if isinstance(lst, list):
                    energy_count += float(len(lst))

        return hp, status_vec, energy_count

    # --- アクティブの追加メタ情報 ---
    def _extract_active_meta(pub: dict):
        """
        戻り値 dict:
          damage_counters: float
          type_id: int
          has_weakness: float
          weakness_type: int
          has_resistance: float
          resistance_type: int
          is_basic: float
          is_ex: float
          retreat_cost: float
          ability_present: float
          ability_used: float
          tools_ids: List[int]
        """
        active = pub.get("active_pokemon")
        info = {
            "damage_counters": 0.0,
            "type_id": -1,
            "has_weakness": 0.0,
            "weakness_type": -1,
            "has_resistance": 0.0,
            "resistance_type": -1,
            "is_basic": 0.0,
            "is_ex": 0.0,
            "retreat_cost": 0.0,
            "ability_present": 0.0,
            "ability_used": 0.0,
            "tools_ids": [],
        }
        if not isinstance(active, dict):
            return info

        try:
            info["damage_counters"] = float(active.get("damage_counters") or 0.0)
        except Exception:
            pass

        try:
            t = active.get("type")
            if t is not None:
                info["type_id"] = int(t)
        except Exception:
            pass

        def _as_bool(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return 1.0
                if v in ("0", "false", "no", "n", "off", ""):
                    return 0.0
            return 0.0

        info["has_weakness"] = _as_bool(active.get("has_weakness"))
        info["has_resistance"] = _as_bool(active.get("has_resistance"))
        info["is_basic"] = _as_bool(active.get("is_basic"))
        info["is_ex"] = _as_bool(active.get("is_ex"))

        try:
            wt = active.get("weakness_type")
            if wt is not None:
                info["weakness_type"] = int(wt)
        except Exception:
            pass
        try:
            rt = active.get("resistance_type")
            if rt is not None:
                info["resistance_type"] = int(rt)
        except Exception:
            pass

        try:
            rc = active.get("retreat_cost")
            if rc is not None:
                info["retreat_cost"] = float(rc)
        except Exception:
            pass

        info["ability_present"] = _as_bool(
            active.get("ability_present") if "ability_present" in active else active.get("has_ability")
        )
        info["ability_used"] = _as_bool(
            active.get("ability_used") if "ability_used" in active else active.get("ability_used_this_turn")
        )

        tools_ids = []
        for key in ("tools", "tool", "attached_tools", "pokemon_tools"):
            lst = active.get(key)
            if isinstance(lst, list):
                for x in lst:
                    try:
                        cid = int(x[0] if isinstance(x, (list, tuple)) else x)
                        tools_ids.append(cid)
                    except Exception:
                        continue
            elif lst is not None:
                try:
                    cid = int(lst)
                    tools_ids.append(cid)
                except Exception:
                    pass
        info["tools_ids"] = tools_ids

        return info

    # --- タイプIDを one-hot に変換 ---
    def _type_id_to_onehot(type_id: int, dim: int = TYPE_DIM):
        v = _np.zeros(dim, dtype=_np.float32)
        try:
            t = int(type_id)
            if 0 <= t < dim:
                v[t] = 1.0
        except Exception:
            pass
        return v

    # --- アクティブに付いているエネルギーの「種類」 one-hot ---
    def _collect_active_energy_type_vec(pub: dict, V: int):
        active = pub.get("active_pokemon")
        ids = _collect_energy_ids_from_poke(active) if isinstance(active, dict) else []
        return _vec_from_id_list(ids, V)

    # --- ベンチごとのエネルギー種類分布（max_bench_slots × Vloc） ---
    def _build_bench_energy_slots_vec(pub: dict, V: int, max_bench_slots: int = 8):
        v = _np.zeros(max_bench_slots * V, dtype=_np.float32)
        bench = pub.get("bench_pokemon")
        if not isinstance(bench, list):
            return v
        for i, obj in enumerate(bench):
            if i >= max_bench_slots:
                break
            if not isinstance(obj, dict):
                continue
            ids = _collect_energy_ids_from_poke(obj)
            if not ids:
                continue
            slot_vec = _vec_from_id_list(ids, V)
            base = i * V
            v[base:base + V] = slot_vec
        return v

    # --- 「ターン内 1回」系のフラグを抽出 ---
    def _extract_turn_flags_for_side(pub_side: dict, sb: dict, side_key: str):
        """
        返り値: np.array(4,)
          [energy_attached, supporter_used, stadium_played, stadium_effect_used]
        それぞれ 0.0/1.0。キーが存在しない場合は 0.0。
        """
        flags = {
            "energy_attached": 0.0,
            "supporter_used": 0.0,
            "stadium_played": 0.0,
            "stadium_effect_used": 0.0,
        }

        # サイド側にぶら下がっているフラグ候補
        cand_maps = [
            pub_side,
            sb.get("turn_flags") or {},
            sb.get("meta") or {},
        ]

        def _as_bool(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return 1.0
                if v in ("0", "false", "no", "n", "off", ""):
                    return 0.0
            return 0.0

        # キーの候補を広めにとる（実際に存在するものだけ効く）
        key_candidates = {
            "energy_attached": [
                "energy_attached_this_turn",
                "energy_attach_this_turn",
                "energy_attached",
                "has_attached_energy_this_turn",
            ],
            "supporter_used": [
                "supporter_used_this_turn",
                "supporter_played_this_turn",
                "supporter_used",
            ],
            "stadium_played": [
                "stadium_played_this_turn",
                "stadium_used_this_turn",
                "stadium_played",
            ],
            "stadium_effect_used": [
                "stadium_effect_used_this_turn",
                "stadium_effect_used",
                "stadium_effect_this_turn",
            ],
        }

        for name, keys in key_candidates.items():
            for d in cand_maps:
                if not isinstance(d, dict):
                    continue
                for k in keys:
                    if k in d:
                        flags[name] = max(flags[name], _as_bool(d.get(k)))
                # side_key 付きのネームスペースも一応見る
                side_map = d.get(side_key) if isinstance(d.get(side_key), dict) else None
                if isinstance(side_map, dict):
                    for k in keys:
                        if k in side_map:
                            flags[name] = max(flags[name], _as_bool(side_map.get(k)))

        return _np.array(
            [
                flags["energy_attached"],
                flags["supporter_used"],
                flags["stadium_played"],
                flags["stadium_effect_used"],
            ],
            dtype=_np.float32,
        )

    # --- 前のターンに自分ポケモンが気絶したかフラグ ---
    def _extract_prev_ko_flag(sb: dict, pub_side: dict, side_key: str):
        """
        返り値: float（0.0 or 1.0）
        候補キー:
          - sb["prev_knockout"][side_key]
          - sb["last_turn_knockout"][side_key]
          - pub_side["ko_last_turn"], pub_side["knocked_out_last_turn"] など
        """
        def _as_bool(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return 1.0
                if v in ("0", "false", "no", "n", "off", ""):
                    return 0.0
            return 0.0

        # state_before 直下に side ごとの情報があるケース
        for key in ("prev_knockout", "last_turn_knockout", "previous_turn_knockout"):
            d = sb.get(key)
            if isinstance(d, dict) and side_key in d:
                return _as_bool(d.get(side_key))

        # サイド側に直接ぶら下がっているケース
        for key in ("ko_last_turn", "knocked_out_last_turn", "pokemon_fainted_last_turn"):
            if key in pub_side:
                return _as_bool(pub_side.get(key))

        return 0.0

    # --- 基本ベクトル ---
    me_hand_slots_vec = _vec_from_hand_slots(me_pub.get("hand", []), Vloc)
    me_board_vec      = _collect_pokemon_board_ids(me_pub)
    me_discard_vec    = _vec_from_id_list(me_pub.get("discard_pile", []), Vloc)

    opp_board_vec     = _collect_pokemon_board_ids(opp_pub)
    opp_discard_vec   = _vec_from_id_list(opp_pub.get("discard_pile", []), Vloc)

    me_energy_vec     = _collect_attached_energy_ids(me_pub)
    opp_energy_vec    = _collect_attached_energy_ids(opp_pub)

    # --- アクティブのエネルギー「種類」ベクトル ---
    me_active_energy_type_vec  = _collect_active_energy_type_vec(me_pub, Vloc)
    opp_active_energy_type_vec = _collect_active_energy_type_vec(opp_pub, Vloc)

    # --- ベンチごとのエネルギー種類分布 ---
    me_bench_energy_slots_vec  = _build_bench_energy_slots_vec(me_pub, Vloc)
    opp_bench_energy_slots_vec = _build_bench_energy_slots_vec(opp_pub, Vloc)

    # --- ベンチごとのポケモンID分布 ---
    me_bench_board_slots_vec  = _build_bench_board_slots_vec(me_pub, Vloc)
    opp_bench_board_slots_vec = _build_bench_board_slots_vec(opp_pub, Vloc)

    # --- ターン情報・サイド・山札残数＋手札枚数＋トラッシュ枚数 ---
    turn = int(sb_pub.get("turn") or 0)
    cur  = sb_pub.get("current_player")

    me_player  = me_pub.get("player")
    opp_player = opp_pub.get("player")

    cur_onehot = _np.array(
        [
            1.0 if cur == me_player else 0.0,
            1.0 if cur == opp_player else 0.0,
        ],
        dtype=_np.float32,
    )

    me_prize_cnt  = int(me_pub.get("prize_count") or 0)
    opp_prize_cnt = int(opp_pub.get("prize_count") or 0)
    me_deck_cnt   = int(me_pub.get("deck_count")  or 0)
    opp_deck_cnt  = int(opp_pub.get("deck_count") or 0)

    me_hand_cnt   = int(me_pub.get("hand_count") or (len(me_pub.get("hand")) if isinstance(me_pub.get("hand"), list) else 0))
    opp_hand_cnt  = int(opp_pub.get("hand_count") or (len(opp_pub.get("hand")) if isinstance(opp_pub.get("hand"), list) else 0))

    me_discard_cnt  = int(me_pub.get("discard_pile_count") or (len(me_pub.get("discard_pile")) if isinstance(me_pub.get("discard_pile"), list) else 0))
    opp_discard_cnt = int(opp_pub.get("discard_pile_count") or (len(opp_pub.get("discard_pile")) if isinstance(opp_pub.get("discard_pile"), list) else 0))

    scalars = _np.array(
        [
            turn,
            cur_onehot[0],
            cur_onehot[1],
            me_prize_cnt,
            opp_prize_cnt,
            me_deck_cnt,
            opp_deck_cnt,
            me_hand_cnt,
            opp_hand_cnt,
            me_discard_cnt,
            opp_discard_cnt,
        ],
        dtype=_np.float32,
    )

    # --- アクティブ状態／エネ枚数／ベンチ数 ---
    me_hp,  me_status_vec,  me_energy_cnt  = _extract_active_features(me_pub)
    opp_hp, opp_status_vec, opp_energy_cnt = _extract_active_features(opp_pub)

    me_bench = me_pub.get("bench_pokemon")
    opp_bench = opp_pub.get("bench_pokemon")

    def _count_real_bench(lst):
        """
        bench_pokemon は [dict or 0 or None, ...] のようにスロット数で埋められることがあるため、
        実際にポケモンが存在するスロット（dict）のみを数える。
        スタジアム効果により 3〜8 スロットになる変動もこのカウントで吸収する。
        """
        if not isinstance(lst, list):
            return 0.0
        c = 0
        for obj in lst:
            if isinstance(obj, dict):
                c += 1
        return float(c)

    me_bench_cnt  = _count_real_bench(me_bench)
    opp_bench_cnt = _count_real_bench(opp_bench)

    # --- アクティブの追加メタ情報 ---
    me_meta  = _extract_active_meta(me_pub)
    opp_meta = _extract_active_meta(opp_pub)

    extra_scalars = _np.array(
        [
            me_hp,
            opp_hp,
            me_energy_cnt,
            opp_energy_cnt,
            me_bench_cnt,
            opp_bench_cnt,
            me_meta["damage_counters"],
            opp_meta["damage_counters"],
            me_meta["has_weakness"],
            opp_meta["has_weakness"],
            me_meta["has_resistance"],
            opp_meta["has_resistance"],
            me_meta["is_basic"],
            opp_meta["is_basic"],
            me_meta["is_ex"],
            opp_meta["is_ex"],
            me_meta["retreat_cost"],
            opp_meta["retreat_cost"],
        ],
        dtype=_np.float32,
    )

    # --- タイプ関連 one-hot ---
    me_type_vec        = _type_id_to_onehot(me_meta["type_id"], TYPE_DIM)
    opp_type_vec       = _type_id_to_onehot(opp_meta["type_id"], TYPE_DIM)
    me_weak_type_vec   = _type_id_to_onehot(me_meta["weakness_type"], TYPE_DIM)
    opp_weak_type_vec  = _type_id_to_onehot(opp_meta["weakness_type"], TYPE_DIM)
    me_resist_type_vec = _type_id_to_onehot(me_meta["resistance_type"], TYPE_DIM)
    opp_resist_type_vec= _type_id_to_onehot(opp_meta["resistance_type"], TYPE_DIM)

    type_vecs = _np.concatenate(
        [
            me_type_vec,
            opp_type_vec,
            me_weak_type_vec,
            opp_weak_type_vec,
            me_resist_type_vec,
            opp_resist_type_vec,
        ],
        axis=0,
    )

    # --- アビリティフラグ ---
    ability_flags = _np.array(
        [
            me_meta["ability_present"],
            me_meta["ability_used"],
            opp_meta["ability_present"],
            opp_meta["ability_used"],
        ],
        dtype=_np.float32,
    )

    # --- アクティブポケモンに付いているどうぐ ---
    me_tools_vec  = _vec_from_id_list(me_meta["tools_ids"], Vloc)
    opp_tools_vec = _vec_from_id_list(opp_meta["tools_ids"], Vloc)

    # --- スタジアムカード ---
    stadium_ids = []
    stadium = sb_pub.get("stadium")
    if isinstance(stadium, dict):
        # {"name": id} / {"id": id} の両方を許容
        if "name" in stadium:
            try:
                stadium_ids.append(int(stadium.get("name")))
            except Exception:
                pass
        if "id" in stadium:
            try:
                stadium_ids.append(int(stadium.get("id")))
            except Exception:
                pass
    elif stadium is not None:
        try:
            stadium_ids.append(int(stadium))
        except Exception:
            pass

    # サイド別 active_stadium も拾う
    for side_pub in (me_pub, opp_pub):
        st = side_pub.get("active_stadium")
        if isinstance(st, dict):
            if "name" in st:
                try:
                    stadium_ids.append(int(st.get("name")))
                except Exception:
                    pass
            if "id" in st:
                try:
                    stadium_ids.append(int(st.get("id")))
                except Exception:
                    pass
        elif st is not None:
            try:
                stadium_ids.append(int(st))
            except Exception:
                pass

    stadium_vec = _vec_from_id_list(stadium_ids, Vloc)

    # --- ターン内 1回系フラグ（自分・相手） ---
    me_turn_flags  = _extract_turn_flags_for_side(me_pub, sb_pub, "me")
    opp_turn_flags = _extract_turn_flags_for_side(opp_pub, sb_pub, "opp")
    turn_flags = _np.concatenate([me_turn_flags, opp_turn_flags], axis=0)

    # --- 前のターンで自分ポケモンが気絶したか ---
    prev_ko_flag = _np.array(
        [_extract_prev_ko_flag(sb_pub, me_pub, "me")],
        dtype=_np.float32,
    )

    vec = _np.concatenate(
        [
            me_hand_slots_vec,
            me_board_vec,
            me_discard_vec,
            opp_board_vec,
            opp_discard_vec,
            me_energy_vec,
            opp_energy_vec,
            me_active_energy_type_vec,
            opp_active_energy_type_vec,
            me_bench_energy_slots_vec,
            opp_bench_energy_slots_vec,
            me_bench_board_slots_vec,
            opp_bench_board_slots_vec,
            scalars,
            me_status_vec,
            opp_status_vec,
            extra_scalars,
            type_vecs,
            ability_flags,
            me_tools_vec,
            opp_tools_vec,
            stadium_vec,
            turn_flags,
            prev_ko_flag,
        ],
        axis=0,
    )
    return vec.astype(_np.float32).tolist()
