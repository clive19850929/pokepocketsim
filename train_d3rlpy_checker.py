#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_d3rlpy_checker.py (enhanced)
- 自動検出はそのまま（--npz 省略可）
- 追加の“中身”健全性チェックを実装:
  * エピソード再構成（terminals が末端で閉じているか、不自然な連続 True/False）
  * 終端報酬の分布・クラス不均衡
  * 観測のゼロ/極小分散、NaN/Inf、外れ値率
  * 行動の one-hot 部分の飽和（0/1 への張り付き率）
  * エピソード長の分布（p50/p90/max）
  * 同一観測の重複率（サンプリング近似）
"""

import argparse, os, sys, json
import numpy as np
from typing import Optional, Tuple, Dict, Any

FILENAME_NPZ = "d3rlpy_dataset_all.npz"
FILENAME_MASK = "obs_mask.npy"
FILENAME_SCALER = "scaler.npz"
FILENAME_ACTION_TYPES = "action_types.json"
MAX_ARGS = 3
ACTION_MIN_DEFAULT = 0.0
ACTION_MAX_DEFAULT = 1.0

def info(x): print(f"[INFO] {x}")
def warn(x): print(f"[WARN] {x}")
def error(x): print(f"[ERROR] {x}")

def find_nearby(names, max_depth=4) -> Optional[str]:
    if isinstance(names, str): names=[names]
    roots = []
    try: roots.append(os.getcwd())
    except: pass
    try: roots.append(os.path.dirname(os.path.abspath(__file__)))
    except: pass
    for r in roots:
        for n in names:
            p = os.path.join(r, n)
            if os.path.exists(p): return p
    # shallow walk
    cands=[]
    def _walk(root, d):
        if d>max_depth: return
        try:
            with os.scandir(root) as it:
                for e in it:
                    try:
                        if e.is_file() and e.name in names: cands.append(e.path)
                        elif e.is_dir(): _walk(e.path, d+1)
                    except: continue
        except: pass
    for r in roots: _walk(r,0)
    return cands[0] if cands else None

def autodetect_npz(explicit: Optional[str]) -> Optional[str]:
    if explicit: return explicit
    env=os.getenv("D3RLPY_DATASET")
    if env and os.path.exists(env):
        info(f"D3RLPY_DATASET から検出: {env}")
        return env
    near=find_nearby([FILENAME_NPZ])
    if near:
        info(f"近傍から検出: {near}")
        return near
    for p in [rf"D:\date\{FILENAME_NPZ}", rf"C:\data\{FILENAME_NPZ}"]:
        if os.path.exists(p):
            info(f"既定候補から検出: {p}")
            return p
    return None

def autodetect_sidefile(base_dir, explicit, filename):
    if explicit and os.path.exists(explicit): return explicit
    p=os.path.join(base_dir, filename)
    if os.path.exists(p): return p
    near=find_nearby([filename])
    return near

def load_npz(path):
    d=np.load(path)
    req=["observations","actions","rewards","terminals"]
    for k in req:
        if k not in d.files: raise RuntimeError(f"npz に {k} がありません: {path}")
    obs=d["observations"].astype(np.float32,copy=False)
    act=d["actions"].astype(np.float32,copy=False)
    rew=d["rewards"].astype(np.float32,copy=False)
    ter=d["terminals"].astype(np.bool_,copy=False)
    return obs,act,rew,ter

def basic_check(obs,act,rew,ter)->Dict[str,Any]:
    rep={}
    ok=True
    rep["obs_shape"]=tuple(obs.shape)
    rep["act_shape"]=tuple(act.shape)
    rep["rew_shape"]=tuple(rew.shape)
    rep["ter_shape"]=tuple(ter.shape)
    n=len(rew)
    if obs.ndim!=2 or act.ndim!=2 or rew.ndim!=1 or ter.ndim!=1: ok=False
    if not (len(obs)==len(act)==len(ter)==n):
        ok=False; warn("サンプル数が一致しません（obs/act/rew/ter）")

    if act.size>0:
        a_min=float(np.nanmin(act)); a_max=float(np.nanmax(act))
        rep["action_min"]=a_min; rep["action_max"]=a_max
        if a_min < ACTION_MIN_DEFAULT-1e-5 or a_max > ACTION_MAX_DEFAULT+1e-5:
            warn(f"actions レンジ逸脱: min={a_min:.4f}, max={a_max:.4f}")
        else:
            info("actions は [0,1] レンジ内です。")
        act_dim=act.shape[1]; inferred_K=act_dim-(MAX_ARGS+1)
        rep["action_dim"]=act_dim; rep["inferred_K"]=inferred_K
        if inferred_K<=0:
            ok=False; warn(f"inferred_K={inferred_K}（K+{MAX_ARGS}+1 前提を満たしません）")
    else:
        ok=False; warn("actions が空です。")

    rep["basic_ok"]=ok
    print(f"  shapes: obs={rep['obs_shape']} act={rep['act_shape']} rew={rep['rew_shape']} ter={rep['ter_shape']}")
    return rep

def episode_stats(ter: np.ndarray)->Dict[str,Any]:
    rep={}
    ends=np.where(ter)[0].tolist()
    if not ends or ends[-1]!=len(ter)-1: ends.append(len(ter)-1)
    starts=[0]+[i+1 for i in ends[:-1]]
    lens=[e-s+1 for s,e in zip(starts,ends)]
    rep["num_episodes"]=len(lens)
    rep["len_p50"]=float(np.percentile(lens,50)) if lens else 0
    rep["len_p90"]=float(np.percentile(lens,90)) if lens else 0
    rep["len_max"]=int(max(lens)) if lens else 0
    # 途中 True の塊/連続異常
    consec_true = int(np.max(np.diff(np.where(np.concatenate(([ter[0]], ter[:-1] != ter[1:], [True])))[0])[::2])) if ter.size>0 else 0
    rep["max_consecutive_terminals"]=consec_true
    print(f"  episodes: {rep['num_episodes']} | len p50/p90/max = {rep['len_p50']:.0f}/{rep['len_p90']:.0f}/{rep['len_max']}")
    if consec_true>1:
        warn(f"terminals=True の連続が {consec_true} 観測続く区間があります（エピソード境界の異常の可能性）")
    return rep

def reward_stats(rew: np.ndarray)->Dict[str,Any]:
    rep={}
    rep["mean"]=float(np.mean(rew)); rep["std"]=float(np.std(rew))
    rep["p1"]=float(np.percentile(rew,1)); rep["p50"]=float(np.percentile(rew,50))
    rep["p99"]=float(np.percentile(rew,99))
    pos=int((rew>0).sum()); neg=int((rew<0).sum()); zero=int((rew==0).sum()); n=rew.size
    rep["n_pos"]=pos; rep["n_neg"]=neg; rep["n_zero"]=zero; rep["n"]=n
    if n>0:
        print(f"  reward: mean={rep['mean']:.4f} std={rep['std']:.4f} | >0:{pos} ==0:{zero} <0:{neg}")
        if pos==0 or neg==0:
            warn("終端報酬の符号が片寄っています（勝ち/負け片寄りの可能性）")
    return rep

def obs_quality(obs: np.ndarray)->Dict[str,Any]:
    rep={}
    n,d=obs.shape
    # NaN/Inf
    nan_ct=int(np.isnan(obs).sum()); inf_ct=int(np.isinf(obs).sum())
    rep["nan_count"]=nan_ct; rep["inf_count"]=inf_ct
    # 分散
    with np.errstate(invalid="ignore"):
        var=np.nanvar(obs, axis=0)
    zero_var=int(np.sum(var<=1e-12))
    small_var=int(np.sum(var<=1e-8))
    rep["zero_var_dims"]=int(zero_var); rep["small_var_dims"]=int(small_var)
    # 外れ値（|z|>6）
    m=np.nanmean(obs,axis=0); s=np.nanstd(obs,axis=0)+1e-6
    z=np.abs((obs-m)/s)
    outlier_ratio=float(np.mean(z>6.0))
    rep["outlier_ratio"]=outlier_ratio
    print(f"  obs: NaN={nan_ct}, Inf={inf_ct}, zero-var={zero_var}, small-var(≤1e-8)={small_var}, outliers(|z|>6)={outlier_ratio:.4f}")
    if nan_ct>0 or inf_ct>0: warn("観測に NaN/Inf を含みます。前処理を確認してください。")
    if small_var>0: warn("ほぼ定数の観測次元があります。マスクの効き残しを確認してください。")
    return rep

def action_saturation(act: np.ndarray, inferred_K: int)->Dict[str,Any]:
    rep={}
    if inferred_K<=0: return {"ok":False}
    onehot=act[:, :inferred_K]
    rest=act[:, inferred_K:]
    # 0/1 近傍へ張り付き率
    eps=1e-4
    near0=float(np.mean(onehot<eps))
    near1=float(np.mean(onehot>1-eps))
    rep["onehot_near0_ratio"]=near0; rep["onehot_near1_ratio"]=near1
    # マスク列（末尾 1 次元が action-valid マスクの想定）
    rep["rest_min"]=float(np.min(rest)) if rest.size else 0.0
    rep["rest_max"]=float(np.max(rest)) if rest.size else 0.0
    print(f"  action onehot saturation: near0={near0:.3f}, near1={near1:.3f} | rest[min,max]=[{rep['rest_min']:.3f},{rep['rest_max']:.3f}]")
    if near1>0.99:
        warn("one-hot がほぼ常に 1 へ張り付いています（データ多様性が乏しい可能性）。")
    return rep

def duplicate_ratio(obs: np.ndarray, sample=5000)->Dict[str,Any]:
    n=min(sample, obs.shape[0])
    if n<2: return {"approx_dup_ratio":0.0}
    idx=np.random.RandomState(0).choice(obs.shape[0], size=n, replace=False)
    X=np.round(obs[idx], decimals=6)
    keys=[X[i].tobytes() for i in range(n)]
    uniq=len(set(keys))
    dup=1.0-(uniq/n)
    print(f"  approx duplicate rate (rounded 1e-6): {dup:.4f}")
    return {"approx_dup_ratio":float(dup)}

def check_obs_mask_shape(obs, mask_path)->Dict[str,Any]:
    rep={"obs_mask":mask_path,"obs_mask_ok":True}
    if not mask_path:
        warn("obs_mask.npy 見つからず（任意）"); rep["obs_mask_ok"]=False; return rep
    try:
        m=np.load(mask_path,allow_pickle=False)
        rep["mask_len"]=int(m.shape[0]); rep["obs_dim"]=int(obs.shape[1])
        if m.ndim!=1: warn(f"obs_mask 次元!=1: {m.shape}"); rep["obs_mask_ok"]=False
        if m.shape[0]!=obs.shape[1]:
            warn(f"obs_dim({obs.shape[1]}) と obs_mask_len({m.shape[0]}) が不一致（前処理手順を再確認）")
            rep["obs_mask_ok"]=False
    except Exception as e:
        warn(f"obs_mask 読込失敗: {e}"); rep["obs_mask_ok"]=False
    return rep

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--obs-mask", type=str, default=None)
    parser.add_argument("--scaler", type=str, default=None)
    parser.add_argument("--action-types", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args=parser.parse_args()

    npz_path=autodetect_npz(args.npz)
    if not npz_path:
        error(f"{FILENAME_NPZ} を自動検出できませんでした。D3RLPY_DATASET 環境変数 or 近傍配置を確認してください。")
        sys.exit(2)
    info(f"npz: {npz_path}")
    base_dir=os.path.dirname(os.path.abspath(npz_path))

    mask_path=autodetect_sidefile(base_dir, args.obs_mask, FILENAME_MASK)
    scaler_path=autodetect_sidefile(base_dir, args.scaler, FILENAME_SCALER)
    action_types_path=autodetect_sidefile(base_dir, args.action_types, FILENAME_ACTION_TYPES)
    info(f"obs_mask: {mask_path if mask_path else '(none)'}")
    info(f"scaler  : {scaler_path if scaler_path else '(none)'}")
    info(f"atypes  : {action_types_path if action_types_path else '(none)'}")

    overall_ok=True
    try:
        obs,act,rew,ter=load_npz(npz_path)

        print("\n=== BASIC ===")
        rep_basic=basic_check(obs,act,rew,ter); overall_ok&=rep_basic.get("basic_ok",False)

        print("\n=== EPISODES ===")
        rep_epi=episode_stats(ter)

        print("\n=== REWARD ===")
        rep_rew=reward_stats(rew)

        print("\n=== OBS QUALITY ===")
        rep_obs=obs_quality(obs)

        print("\n=== ACTION SATURATION ===")
        inferred_K=int(rep_basic.get("inferred_K",-1))
        rep_act=action_saturation(act,inferred_K)

        print("\n=== DUPLICATES (approx) ===")
        rep_dup=duplicate_ratio(obs)

        print("\n=== OBS MASK SHAPE ===")
        rep_mask2=check_obs_mask_shape(obs, mask_path)

        # ざっくり NG 判定を補強
        if rep_obs.get("nan_count",0)>0 or rep_obs.get("inf_count",0)>0: overall_ok=False
        if rep_obs.get("zero_var_dims",0)>0 and rep_mask2.get("obs_mask_ok",True)==False: overall_ok=False
        if rep_dup.get("approx_dup_ratio",0.0)>0.5:
            warn("観測の重複率が高いです（データ多様性不足の可能性）。"); overall_ok=False

        # 保存（任意）
        if args.out_dir:
            try:
                os.makedirs(args.out_dir, exist_ok=True)
                report={
                    "npz":npz_path,
                    "obs_mask":mask_path,
                    "scaler":scaler_path,
                    "action_types":action_types_path,
                    "basic":rep_basic,
                    "episodes":rep_epi,
                    "reward":rep_rew,
                    "obs_quality":rep_obs,
                    "action_sat":rep_act,
                    "duplicates":rep_dup,
                    "mask_shape":rep_mask2,
                    "overall_ok":bool(overall_ok),
                }
                out_json=os.path.join(args.out_dir,"dataset_report_detailed.json")
                with open(out_json,"w",encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
                info(f"wrote: {out_json}")
            except Exception as e:
                warn(f"レポート書き出し失敗: {e}")

    except Exception as e:
        overall_ok=False
        error(f"検査中に例外: {type(e).__name__}: {e}")

    print("\n=== RESULT ===")
    print(f"overall_ok: {overall_ok}")
    sys.exit(0 if overall_ok else 2)

if __name__=="__main__":
    main()
