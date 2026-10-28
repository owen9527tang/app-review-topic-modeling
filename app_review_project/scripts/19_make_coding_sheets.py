"""P4 加固：生成两份**互盲**的人工编码表（各 100 条）。

为什么拆成两份而不是一份带 A/B 两列：
  同一张表里 B 会看到 A 已填的编码，一致性会被「抄」污染。拆成两个文件、且 B 的行序打乱，
  才能保证两人独立。

输入：data/sample_200.csv（06_make_sample.py 产出的 200 条分层抽样）
输出：data/coding_A.csv、data/coding_B.csv（列：review_id, app_name, content, spec, primary, note）

星级列**不写入编码表**，避免编码者按星级反推语气（手册要求只看文字）。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLE = DATA / "sample_200.csv"
A_SHEET = DATA / "coding_A.csv"
B_SHEET = DATA / "coding_B.csv"

N_PER_CODER = 100
SEED_A = 42
SEED_B = 7


def pick(sample: pd.DataFrame) -> pd.DataFrame:
    """按星级等比例取 100 条，保证 1 星 / 5 星 / 中间星级都有代表。"""
    parts = []
    for rating, grp in sample.groupby("rating"):
        k = int(round(len(grp) * N_PER_CODER / len(sample)))
        k = max(1, min(k, len(grp)))
        parts.append(grp.sample(n=k, random_state=SEED_A))
    out = pd.concat(parts, ignore_index=True)

    # 补齐/裁剪到正好 100 条：差额从最多的一档里随机删或补
    if len(out) > N_PER_CODER:
        drop = out.sample(n=len(out) - N_PER_CODER, random_state=SEED_A).index
        out = out.drop(drop)
    elif len(out) < N_PER_CODER:
        pool = sample[~sample["review_id"].isin(out["review_id"])]
        extra = pool.sample(n=N_PER_CODER - len(out), random_state=SEED_A)
        out = pd.concat([out, extra], ignore_index=True)
    return out.reset_index(drop=True)


def main():
    if not SAMPLE.exists():
        raise SystemExit(f"缺少 {SAMPLE}，先运行 scripts/06_make_sample.py")

    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype={"review_id": str})
    base = pick(sample)

    cols = ["review_id", "app_name", "content"]
    a = base[cols].copy()
    b = base[cols].sample(frac=1, random_state=SEED_B).reset_index(drop=True)  # 行序打乱

    for df, path in [(a, A_SHEET), (b, B_SHEET)]:
        df["spec"] = ""
        df["primary"] = ""
        df["note"] = ""
        df.to_csv(path, index=False, encoding="utf-8-sig")

    dist = base["rating"].value_counts().sort_index()
    print(f"A：{A_SHEET}（{len(a)} 行）")
    print(f"B：{B_SHEET}（{len(b)} 行，行序已打乱）")
    print("星级构成：", {int(k): int(v) for k, v in dist.items()})
    print("两份表的 review_id 集合是否一致：", set(a["review_id"]) == set(b["review_id"]))


if __name__ == "__main__":
    main()
