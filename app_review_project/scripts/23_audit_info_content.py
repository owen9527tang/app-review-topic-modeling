"""抽查 `has_num` / `has_object` 的规则判定是否准确。

`22_info_content.py` 的两个指标是规则生成的，是全仓库唯一未经人工校验的变量。
本脚本抽 30 条（分三组，每组 10 条、跨五个星级），只要求人判「机器的标记对不对」，
用来给出一个可上报的准确率，而不是重做编码实验。

用法：
  python scripts/23_audit_info_content.py            # 生成抽查表 data/audit_info_content.csv
  python scripts/23_audit_info_content.py --check    # 读回填好的表，打印准确率

抽查表含评论正文，只在本地保留，不进仓库。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVIEWS = DATA / "reviews_analysis.csv"
SHEET = DATA / "audit_info_content.csv"
SCRIPT22 = ROOT / "scripts" / "22_info_content.py"

SEED = 20261029
PER_RATING = 2  # 每组每个星级取 2 条 → 5 档 × 2 = 10 条/组


def load_rules():
    """复用 22 的规则，避免两套口径漂移。"""
    spec = importlib.util.spec_from_file_location("info_content", SCRIPT22)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NUM_RE, mod.has_object, mod.MED_OBJECTS


def num_hits(text: str, num_re) -> str:
    t = str(text or "")
    seen = []
    for m in num_re.finditer(t):
        w = m.group(0)
        if w not in seen:
            seen.append(w)
    return "、".join(seen[:6])


def obj_hits(text: str, words) -> str:
    t = str(text or "")
    seen = [w for w in words if w in t]
    return "、".join(seen[:6])


def pick(pool: pd.DataFrame, rng, n_per=PER_RATING, total=10):
    """每个星级取 n_per 条；某档不足时用其余星级的行补足。"""
    picks = []
    for r in [1, 2, 3, 4, 5]:
        sub = pool[pool["rating"] == r]
        k = min(n_per, len(sub), total - sum(len(p) for p in picks))
        if k:
            picks.append(sub.sample(k, random_state=int(rng.integers(1 << 31))))
    picked_ids = pd.concat(picks)["review_id"] if picks else pd.Series(dtype=str)
    left = total - len(picked_ids)
    if left > 0:
        rest = pool[~pool["review_id"].isin(picked_ids)]
        picks.append(rest.sample(min(left, len(rest)), random_state=int(rng.integers(1 << 31))))
    return pd.concat(picks).head(total)


def make():
    num_re, obj_fn, obj_words = load_rules()
    df = pd.read_csv(REVIEWS, encoding="utf-8-sig", dtype={"review_id": str})
    df["content"] = df["content"].fillna("").astype(str)
    df["has_num"] = df["content"].str.contains(num_re).astype(int)
    df["has_object"] = df["content"].map(obj_fn).astype(int)
    df["cn_len"] = df["content"].str.count(r"[一-龥]")

    rng = np.random.default_rng(SEED)
    groups = {
        "A 有数量词": df[df["has_num"] == 1],
        "B 有对象无数量词": df[(df["has_object"] == 1) & (df["has_num"] == 0)],
        "C 两者都没有": df[(df["has_object"] == 0) & (df["has_num"] == 0)],
    }
    rows = []
    for gname, pool in groups.items():
        part = pick(pool, rng).copy()
        part["group"] = gname
        rows.append(part)
    sheet = pd.concat(rows, ignore_index=True)

    sheet["num_hit"] = sheet["content"].map(lambda t: num_hits(t, num_re))
    sheet["obj_hit"] = sheet["content"].map(lambda t: obj_hits(t, obj_words))
    sheet = sheet.rename(columns={"has_num": "机器_数量词", "has_object": "机器_对象"})
    sheet.insert(0, "序号", [
        f"{g}-{i:02d}" for g, i in zip(sheet["group"].str[0],
                                       sheet.groupby("group").cumcount() + 1)
    ])
    sheet["人工_数量词对"] = ""
    sheet["人工_对象对"] = ""
    sheet["备注"] = ""
    cols = ["序号", "group", "review_id", "app_name", "rating", "cn_len",
            "机器_数量词", "num_hit", "机器_对象", "obj_hit",
            "人工_数量词对", "人工_对象对", "备注", "content"]
    sheet[cols].to_csv(SHEET, index=False, encoding="utf-8-sig")

    print(f"已生成 {SHEET.relative_to(ROOT)}（{len(sheet)} 条，跨五个星级）")
    print("填「人工_数量词对」「人工_对象对」：机器的标记与文本一致填 1，不一致填 0；")
    print("拿不准的行填 ? 并写备注，这些行不计入准确率。填完运行：")
    print("  python scripts/23_audit_info_content.py --check")


def check():
    if not SHEET.exists():
        raise SystemExit("先运行不带参数的命令生成抽查表")
    d = pd.read_csv(SHEET, encoding="utf-8-sig", dtype={"review_id": str})
    for who, flag, human in [("数量词", "机器_数量词", "人工_数量词对"),
                             ("对象", "机器_对象", "人工_对象对")]:
        h = pd.to_numeric(d[human], errors="coerce")
        m = pd.to_numeric(d[flag], errors="coerce")
        ok = h.notna()
        m1, m0 = ok & (m == 1), ok & (m == 0)
        print(f"\n【{who}】已判 {int(ok.sum())} 条")
        if m1.sum():
            print(f"  机器判 1 的 {int(m1.sum())} 条中，人工同意 {int((h[m1] == 1).sum())} 条"
                  f"（precision = {(h[m1] == 1).mean():.0%}）")
        if m0.sum():
            print(f"  机器判 0 的 {int(m0.sum())} 条中，人工同意 {int((h[m0] == 0).sum())} 条"
                  f"（specificity = {(h[m0] == 0).mean():.0%}）")
        acc = (h[ok] == m[ok]).mean() if ok.sum() else float("nan")
        print(f"  合计一致率 = {acc:.0%}（n={int(ok.sum())}）")

    miss = d[pd.to_numeric(d["人工_数量词对"], errors="coerce").isna()
             | pd.to_numeric(d["人工_对象对"], errors="coerce").isna()]
    if len(miss):
        print("\n未判的行：")
        for _, r in miss.iterrows():
            print(f"  {r['序号']} {r['review_id']}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else make()
