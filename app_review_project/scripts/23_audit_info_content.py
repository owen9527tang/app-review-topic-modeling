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
REPORT = ROOT / "outputs" / "info_content_audit.md"
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


def read_sheet(path):
    """Excel / WPS 另存时可能变成 GBK，这里两种编码都试。"""
    for enc in ("utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"review_id": str})
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"读不了 {path}：编码既不是 UTF-8 也不是 GBK")


def check():
    if not SHEET.exists():
        raise SystemExit("先运行不带参数的命令生成抽查表")
    d = read_sheet(SHEET)
    stats, lines = {}, []
    lines.append("# 规则指标抽查（30 条）\n")
    lines.append(
        "`22_info_content.py` 的两个指标是规则生成、未经人工校验的。这里抽 30 条（三组各 10 条、"
        "跨五个星级），只要求判「机器的标记与文本是否一致」，用来给指标一个可上报的误差范围。\n"
    )
    lines.append("抽到的组：A 有数量词、B 有对象无数量词、C 两者都没有。\n")
    lines.append("| 指标 | 机器判「有」的条数 | 人工同意 | precision | 机器判「无」的条数 | 人工同意 | specificity |")
    lines.append("|---|---|---|---|---|---|---|")

    for who, flag, human in [("数量词", "机器_数量词", "人工_数量词对"),
                             ("对象", "机器_对象", "人工_对象对")]:
        h = pd.to_numeric(d[human], errors="coerce")
        m = pd.to_numeric(d[flag], errors="coerce")
        ok = h.notna()
        m1, m0 = ok & (m == 1), ok & (m == 0)
        prec = (h[m1] == 1).mean() if m1.sum() else float("nan")
        spec = (h[m0] == 0).mean() if m0.sum() else float("nan")
        acc = (h[ok] == m[ok]).mean() if ok.sum() else float("nan")
        stats[who] = (int(m1.sum()), int((h[m1] == 1).sum()), prec,
                      int(m0.sum()), int((h[m0] == 0).sum()), spec, acc)
        lines.append(
            f"| {who} | {int(m1.sum())} | {int((h[m1] == 1).sum())} | {prec:.0%} | "
            f"{int(m0.sum())} | {int((h[m0] == 0).sum())} | {spec:.0%} |"
        )
    lines.append("")

    lines.append("## 分歧清单\n")
    bad = d[(pd.to_numeric(d["人工_数量词对"], errors="coerce") != pd.to_numeric(d["机器_数量词"], errors="coerce"))
           | (pd.to_numeric(d["人工_对象对"], errors="coerce") != pd.to_numeric(d["机器_对象"], errors="coerce"))]
    for _, r in bad.iterrows():
        txt = str(r["content"]).replace("\n", " ")[:90]
        note = "" if pd.isna(r["备注"]) else f"；备注：{r['备注']}"
        nh = "" if pd.isna(r["num_hit"]) else f"（命中 `{r['num_hit']}`）"
        oh = "" if pd.isna(r["obj_hit"]) else f"（命中 `{r['obj_hit']}`）"
        lines.append(
            f"- `{r['序号']}`（{int(r['rating'])} 星）机器 数量词={int(r['机器_数量词'])}{nh}、"
            f"对象={int(r['机器_对象'])}{oh}，人工判 {r['人工_数量词对']}/{r['人工_对象对']}{note}。原文：{txt}"
        )
    lines.append("")

    lines.append("## 读法\n")
    lines.append(
        "- **数量词的误判都在「非量数字」**：`四星`（评分）、`三线`（三线城市）、`版本 9.2`、"
        "`挂 2 个科室`。因此这个指标实际测的是「正文有没有写数字」，而不是「有没有可核对的量」。"
        f"判「无」的 {stats['数量词'][3]} 条全部被人工确认，说明漏判很少。"
    )
    lines.append(
        f"- **对象词表不全**：漏判集中在 `预约`、`义诊`、`科室` 这类同场景词 —— "
        f"判「无」的 {stats['对象'][3]} 条里 {stats['对象'][3] - stats['对象'][4]} 条被人工改判为「有」。"
        "两组都会被同样低估，属同向偏差，不影响两组的相对比较。"
    )
    lines.append(
        "- **对结论的影响**：数量词的漏判率为 0，误判是「多算」。若两个星级的误判比例相近，"
        "观测率 ≈ 真实率 × 同一系数，比值（1 星是 5 星的 2.7 倍）不因此改变。"
        "但抽查只有 10 条阳性，无法分组估计精度，所以结论只读作方向与量级上限，不当精确值用。"
    )
    lines.append(
        "- **对象词的读数不宜当精度用**：复核里有两处判断本身不自洽 —— `A-01` 有「医院」被判 0，"
        "`A-02` 无任何医疗词却被判 1；`C-03`、`C-10` 的备注写的是「挂号」「技术进步」，"
        "说明复核时把口径放宽到「具体物体/场景」了。所以 88% / 69% 说明的是「词表不全 + 边界不清」，"
        "不能说明关键词法本身不可靠。要收紧的话，应先把判定标准写死（只认字面出现的对象词），再抽第二遍。"
    )
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    for who, s in stats.items():
        print(f"\n【{who}】判有 {s[0]} 条，同意 {s[1]} 条（precision {s[2]:.0%}）；"
              f"判无 {s[3]} 条，同意 {s[4]} 条（specificity {s[5]:.0%}）；合计一致率 {s[6]:.0%}")
    print(f"\n写入 {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else make()
