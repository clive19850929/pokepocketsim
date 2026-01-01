import os
import glob
import json
import random
import pickle
from datetime import datetime

# === 設定ここから ===

# 学習データにしたい自己対戦ログの場所
# 例: D:\date\ に ai_vs_ai_match_*.ml.jsonl がある想定
LOG_DIR_GLOB = r"D:\date\*.ml.jsonl"

# 出力先（同じDドライブ配下に保存する）
OUTPUT_PKL_PATH = r"D:\date\bc_dataset.pkl"

# train/val の比率（8:2なら0.8）
TRAIN_RATIO = 0.8

# === 設定ここまで ===


def iter_jsonl_lines(path):
    """1つの .jsonl ファイルを1行ずつ辞書でyieldする"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 壊れた行があったらスキップ
                continue


def collect_samples_from_logs(paths):
    """
    ログファイル群を読み、(obs_raw, action_type_id) のペアをすべて集める。
    obs_raw は state_before のそのままの辞書。
    action_type_id は action_result["action"][0] の整数。
    """
    obs_list = []
    acttype_list = []

    for p in paths:
        for step in iter_jsonl_lines(p):
            # 必要なキーが揃ってなければス躍る
            if "state_before" not in step:
                continue
            if "action_result" not in step:
                continue
            if "action" not in step["action_result"]:
                continue

            obs_raw = step["state_before"]

            # 例: [action_type_id, arg1, arg2, arg3, arg4]
            act_full = step["action_result"]["action"]
            if (not isinstance(act_full, list)) or (len(act_full) == 0):
                continue

            action_type_id = act_full[0]

            # ここで「学習に使えるかどうか」のフィルタをかけたい場合は追加できる
            # 例: ターンが無効なもの / debug 専用ステップ を除外したい等があればここに書く

            obs_list.append(obs_raw)
            acttype_list.append(action_type_id)

    return obs_list, acttype_list


def train_val_split(obs_list, acttype_list, train_ratio):
    """
    同じ順番で対応している obs_list と acttype_list を
    train/val に分ける。
    """
    assert len(obs_list) == len(acttype_list)
    idxs = list(range(len(obs_list)))
    random.shuffle(idxs)

    cut = int(len(idxs) * train_ratio)

    train_idx = idxs[:cut]
    val_idx   = idxs[cut:]

    train_obs = [obs_list[i] for i in train_idx]
    train_act = [acttype_list[i] for i in train_idx]

    val_obs   = [obs_list[i] for i in val_idx]
    val_act   = [acttype_list[i] for i in val_idx]

    return train_obs, train_act, val_obs, val_act


def main():
    # 1. ログファイルを列挙
    paths = sorted(glob.glob(LOG_DIR_GLOB))
    if not paths:
        print("[WARN] 対象となる .ml.jsonl が見つかりませんでした:", LOG_DIR_GLOB)

    print("[INFO] load files:")
    for p in paths:
        print("  ", p)

    # 2. ログを読み込み、(状態, 行動type) を収集
    obs_list, acttype_list = collect_samples_from_logs(paths)
    print(f"[INFO] collected samples: {len(obs_list)} steps")

    if len(obs_list) == 0:
        print("[WARN] サンプルが0件です。ログのパスやフォーマットを確認してください。")

    # 3. train/val split
    train_obs, train_act, val_obs, val_act = train_val_split(
        obs_list,
        acttype_list,
        TRAIN_RATIO,
    )

    print(f"[INFO] train samples: {len(train_obs)}")
    print(f"[INFO] val samples  : {len(val_obs)}")

    # 4. メタ情報
    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_total_steps": len(obs_list),
        "num_train_steps": len(train_obs),
        "num_val_steps": len(val_obs),
        "src_files": paths,
        "train_ratio": TRAIN_RATIO,
        "note": "obs is raw state_before dict (partial info); action_type is action_result['action'][0]",
    }

    # 5. pickleで保存
    bundle = {
        "train_obs": train_obs,
        "train_action_type": train_act,
        "val_obs": val_obs,
        "val_action_type": val_act,
        "info": meta,
    }

    # 出力先ディレクトリを作成
    out_dir = os.path.dirname(OUTPUT_PKL_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[INFO] wrote dataset pickle -> {OUTPUT_PKL_PATH}")


if __name__ == "__main__":
    # 乱数シャッフルの再現性をある程度持たせたい場合はseed固定してもいい
    random.seed(1234)
    main()
