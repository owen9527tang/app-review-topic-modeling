"""P4 加固：产品类型异质性 + 更细的固定效应（App×年、App×月）。

回答两个追问：
  1. 「分段结构是不是某一类产品的产物？」——按 four 类产品分组重看方向与形状。
  2. 「换成 App 内部、同一时间段内的比较，结论还在吗？」——加 App×年、App×月 固定效应。

输入：data/topic_doc_weights.csv（07_topic_model.py 产出，k=10）
输出：outputs/heterogeneity.md、data/heterogeneity.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
WEIGHTS = DATA / "topic_doc_weights.csv"
CSV_OUT = DATA / "heterogeneity.csv"
REPORT = OUT / "heterogeneity.md"

K = 10

TOPIC_MAP = {
    0: "医生解答与沟通",
    1: "医院挂号与就医流程",
    2: "活动打卡与 AI 识别",
    3: "专家资源与医术口碑",
    4: "付费问诊次数与退款",
    5: "免费咨询与买药",
    6: "App 功能与挂号操作",
    7: "客服售后与配送体检",
    8: "技术故障与更新",
    9: "患者获益与异地就医",
}

# 按产品类型分层（proposal 第 4 节的四类）
APP_TYPE = {
    "平安好医生": "平台型",
    "京东健康": "平台型",
    "微医": "互联网医院",
    "好大夫在线": "互联网医院",
    "微脉": "互联网医院",
    "春雨医生": "在线问诊",
    "丁香医生": "在线问诊",
    "蚂蚁阿福": "AI 健康助手",
}

# 汇报用的六个关键议题（其余在 CSV 里）
KEY = [2, 8, 6, 7, 4, 0, 3]

TYPE_ORDER = ["平台型", "互联网医院", "在线问诊", "AI 健康助手"]


def md_table(df, cols, headers, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating, np.integer)):
                cells.append(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(int(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load():
    df = pd.read_csv(WEIGHTS, encoding="utf-8-sig", dtype={"review_id": str})
    df["rating"] = df["rating"].astype(int)
    df["year"] = df["year"].fillna("").astype(str)
    df["doctor_merged"] = df["t00"] + df["t03"]
    df["app_type"] = df["app_name"].map(APP_TYPE)
    df["app_year"] = df["app_name"] + "_" + df["year"]
    df["app_month"] = df["app_name"] + "_" + df["month"]
    df["is1"] = (df["rating"] == 1).astype(int)
    return df


def type_overview(df):
    rows = []
    for t in TYPE_ORDER:
        d = df[df["app_type"] == t]
        rows.append({
            "产品类型": t,
            "App": "、".join(sorted(d["app_name"].unique())),
            "n": len(d),
            "1星": int((d["rating"] == 1).sum()),
            "5星": int((d["rating"] == 5).sum()),
            "中段": int(d["rating"].isin([2, 3, 4]).sum()),
        })
    return pd.DataFrame(rows)


def raw_by_type(df):
    """每个产品类型内 1 星 − 5 星的原始差值；同时与总体方向比对。"""
    sub = df[df["rating"].isin([1, 5])]
    items = [("doctor_merged", "医生与专家口碑（合计）")] + [(f"t{i:02d}", TOPIC_MAP[i]) for i in KEY]
    overall = {c: sub[sub.is1 == 1][c].mean() - sub[sub.is1 == 0][c].mean() for c, _ in items}
    rows = []
    for c, name in items:
        row = {"议题": name, "总体": round(overall[c], 3)}
        same = total = 0
        for t in TYPE_ORDER:
            d = sub[sub["app_type"] == t]
            a, b = d[d.is1 == 1][c], d[d.is1 == 0][c]
            diff = a.mean() - b.mean()
            _, p = stats.ttest_ind(a, b, equal_var=False)
            row[t] = round(float(diff), 3)
            row[f"p_{t}"] = p
            if len(a) >= 20 and len(b) >= 20:
                total += 1
                same += int(np.sign(diff) == np.sign(overall[c]))
        row["同号"] = f"{same}/{total}"
        rows.append(row)
    return pd.DataFrame(rows)


def fit(sub, c, fe_terms, cluster="app_name"):
    model = smf.ols(f"{c} ~ is1 + {fe_terms}", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub[cluster]}
    )
    return float(model.params["is1"]), float(model.bse["is1"]), float(model.pvalues["is1"]), int(model.nobs), int(model.df_model)


def fe_ladder(df):
    """三层固定效应：App+年（既有）→ App×年 → App×月（仅有效识别单元）。"""
    sub = df[df["rating"].isin([1, 5])].copy()

    # 有效识别单元：同一 App、同一月份里同时有 1 星与 5 星，系数才由这个单元贡献
    cell = sub.groupby("app_month").agg(n=("is1", "size"), n1=("is1", "sum"))
    cell["n5"] = cell["n"] - cell["n1"]
    eff = cell[(cell["n1"] > 0) & (cell["n5"] > 0)].index
    sub_eff = sub[sub["app_month"].isin(eff)]

    rows = []
    cols = [f"t{i:02d}" for i in range(K)] + ["doctor_merged"]
    for c in cols:
        i = "merge" if c == "doctor_merged" else int(c[1:])
        name = "医生与专家口碑（合计）" if c == "doctor_merged" else TOPIC_MAP[i]
        b0, _, p0, n0, _ = fit(sub, c, "C(app_name) + C(year)")
        b1, _, p1, n1, _ = fit(sub, c, "C(app_year)")
        b2, _, p2, n2, _ = fit(sub_eff, c, "C(app_month)")
        # 同子样本对照：只用有效单元重估 A 与 B，才能与 C 直接比较
        b0s, _, p0s, _, _ = fit(sub_eff, c, "C(app_name) + C(year)")
        b1s, _, p1s, _, _ = fit(sub_eff, c, "C(app_year)")
        rows.append({
            "topic": c, "name": name,
            "b_app_year": round(b0, 4), "p0": p0,
            "b_appxyear": round(b1, 4), "p1": p1,
            "b_appxmonth": round(b2, 4), "p2": p2,
            "b_app_year_same": round(b0s, 4), "b_appxyear_same": round(b1s, 4),
            "n_app_year": n0, "n_appxyear": n1, "n_eff": n2,
        })
    tab = pd.DataFrame(rows)
    for k in ["p0", "p1", "p2"]:
        tab[f"{k}_bh"] = multipletests(tab[k], method="fdr_bh")[1]
    diag = {
        "cells_all": int(sub["app_month"].nunique()),
        "cells_eff": int(len(eff)),
        "n_all": int(len(sub)),
        "n_eff": int(len(sub_eff)),
    }
    return tab, diag, sub, sub_eff


def shape_by_type(df):
    """能做的类型里，用同一套规则判形状（中段样本太薄的类型只报原始剖面）。"""
    rows = []
    for t in TYPE_ORDER:
        d = df[df["app_type"] == t]
        mid_n = int(d["rating"].isin([2, 3, 4]).sum())
        if mid_n < 40:
            continue  # 中段太薄，2/3/4 星会有空单元格，估出来的系数没有意义
        for c, name in [("doctor_merged", "医生与专家口碑（合计）")] + [(f"t{i:02d}", TOPIC_MAP[i]) for i in KEY if i != 0]:
            formula = f"{c} ~ C(rating) + C(app_name) + C(year)"
            if d["app_name"].nunique() < 2:
                formula = f"{c} ~ C(rating) + C(year)"
            try:
                m = smf.ols(formula, data=d).fit()
                coef = {r: float(m.params.get(f"C(rating)[T.{r}]", np.nan)) for r in [2, 3, 4, 5]}
                sig = {r: (not np.isnan(coef[r]) and m.pvalues.get(f"C(rating)[T.{r}]", 1.0) < 0.05) for r in [2, 3, 4, 5]}
                shape = classify(coef, sig)
            except Exception:
                coef, shape = {r: np.nan for r in [2, 3, 4, 5]}, "无法估计"
            rows.append({
                "产品类型": t, "中段条数": mid_n, "议题": name,
                "d2": coef[2], "d3": coef[3], "d4": coef[4], "d5": coef[5],
                "形状": shape,
            })
    return pd.DataFrame(rows)


def classify(coef, sig):
    mids = [2, 3, 4]
    if any(np.isnan(coef[r]) for r in mids + [5]):
        return "无法估计"
    mid_max = max(coef[r] for r in mids)
    if any(coef[r] > 0 and sig[r] for r in mids) and mid_max > coef[5]:
        return "中间星级峰值"
    if coef[5] > 0 and sig[5]:
        return "高端专属"
    if sum(1 for r in mids if coef[r] < 0 and sig[r]) >= 2 and coef[5] <= 0:
        return "低端专属"
    return "无稳定形状"


def main():
    df = load()
    ov = type_overview(df)
    rt = raw_by_type(df)
    fe, diag, sub, sub_eff = fe_ladder(df)
    sh = shape_by_type(df)
    pd.concat([rt.assign(表="raw_by_type")], ignore_index=True).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    fe.to_csv(DATA / "heterogeneity_fe.csv", index=False, encoding="utf-8-sig")
    sh.to_csv(DATA / "heterogeneity_shape.csv", index=False, encoding="utf-8-sig")
    write_report(df, ov, rt, fe, diag, sh)
    print(f"写入 {CSV_OUT}、{DATA / 'heterogeneity_fe.csv'}、{DATA / 'heterogeneity_shape.csv'} 与 {REPORT}")
    print(ov.to_string(index=False))
    print(diag)


def write_report(df, ov, rt, fe, diag, sh):
    fei = fe.set_index("name")
    top_gap = fei.loc[["技术故障与更新", "App 功能与挂号操作", "活动打卡与 AI 识别", "医生与专家口碑（合计）"]]

    rt_show = rt.copy()
    rt_show["p_min"] = rt_show[[f"p_{t}" for t in TYPE_ORDER]].min(axis=1)
    rt_show = rt_show[["议题", "总体", "同号"] + TYPE_ORDER]

    lines = [
        "# 产品类型异质性与更细的固定效应",
        "",
        "> 目的：回答两个追问——① 分段结构是不是某一类产品的产物；② 换成「同一个 App、同一时间段内」的比较，结论还在不在。",
        "> 数据与主报告一致（`data/topic_doc_weights.csv`，k=10，1,681 条）。产品类型划分见 proposal 第 4 节。",
        "",
        "## 1. 分组样本",
        "",
        md_table(ov, ["产品类型", "App", "n", "1星", "5星", "中段"],
                 ["产品类型", "包含 App", "n", "1 星", "5 星", "2–4 星"]),
        "",
        f"中段（2–4 星）合计只有 {int(ov['中段'].sum())} 条，分散到四类后每类都不足百条，"
        "所以**分组只判方向，不判形状**；形状仍以全样本、控制 App 与年份的系数为准。",
        "",
        "## 2. 每个产品类型内的原始差值（1 星 − 5 星）",
        "",
        md_table(rt_show, ["议题", "总体", "平台型", "互联网医院", "在线问诊", "AI 健康助手", "同号"],
                 ["议题", "总体", "平台型", "互联网医院", "在线问诊", "AI 助手", "同号类型数"]),
        "",
        "「同号类型数」＝该议题在几个产品类型中与总体差值同号（只计 1 星、5 星各 ≥20 条的组）。",
        "",
    ]

    # 读数一：按产品类型的方向一致性
    same_counts = rt.set_index("议题")["同号"]
    full = [k for k, v in same_counts.items() if v.startswith("4/4")]
    partial = [(k, v) for k, v in same_counts.items() if not v.startswith("4/4")]
    raw_idx = rt.set_index("议题")
    act = raw_idx.loc["活动打卡与 AI 识别"]

    lines += [
        f"- **四类产品全部同号的议题（{len(full)}/8）**：{'、'.join(full)}。",
        f"- **方向不稳的议题**：{'；'.join(f'{k}（{v}）' for k, v in partial) if partial else '无'}——"
        f"它在平台型是 {raw_idx.loc['客服售后与配送体检', '平台型']:+.3f}（反向）、"
        f"在 AI 助手是 {raw_idx.loc['客服售后与配送体检', 'AI 健康助手']:+.3f}（约等于 0），"
        "只在互联网医院与在线问诊里为正。因此它不能作为「1 星议题」的独立证据。",
        f"- **活动议题的原始差距几乎全部来自平台型**：总体 {act['总体']:+.3f}，"
        f"平台型 {act['平台型']:+.3f}，而互联网医院 {act['互联网医院']:+.3f}、"
        f"在线问诊 {act['在线问诊']:+.3f}、AI 助手 {act['AI 健康助手']:+.3f}。"
        "这与 `robustness.md` 的日期集中度一致（京东健康 214 条几乎全为 1 星、只覆盖两个月）。",
        "",
        "## 3. 更细的固定效应：App×年 与 App×月",
        "",
        "同一批 1 星与 5 星评论，逐步收紧参照组：",
        "",
        "- **口径 A（既有）**：`share ~ is1 + App + 年`，n = "
        f"{diag['n_all']}；参照组是「其他 App、其他年份」。",
        "- **口径 B**：`share ~ is1 + App×年`，n = "
        f"{diag['n_all']}；参照组收紧到「同一个 App、同一年」。",
        "- **口径 C**：`share ~ is1 + App×月`，只用「同一 App、同一月份里同时有 1 星与 5 星」的单元，"
        f"n = {diag['n_eff']}（{diag['cells_eff']} / {diag['cells_all']} 个月份单元有效）。"
        "参照组收紧到「同一个 App、同一个月」。",
        "",
        "A 与 B 用了全部 1,497 条，C 只用了 1,031 条，样本不同不能直接比系数，"
        "所以表中同时给出「A、B 在 C 的同一子样本上重估」的两列（标 `*`）。",
        "",
        md_table(
            top_gap.reset_index(),
            ["name", "b_app_year", "b_app_year_same", "b_appxyear_same", "b_appxmonth", "p2_bh"],
            ["议题", "A：App+年（全）", "A*（同子样本）", "B*（同子样本）", "C：App×月", "C 的 p（BH）"],
        ),
        "",
    ]

    # 读数二：同一子样本内收紧参照组，系数动了多少
    max_delta = float((fe["b_appxmonth"] - fe["b_app_year_same"]).abs().max())
    same_sign = int((np.sign(fe["b_app_year"]) == np.sign(fe["b_appxyear"])).sum())
    same_sign_c = int((np.sign(fe["b_app_year_same"]) == np.sign(fe["b_appxmonth"])).sum())
    pair = fei.loc[["技术故障与更新", "App 功能与挂号操作", "医生与专家口碑（合计）"], ["b_app_year_same", "b_appxmonth"]]
    pair_txt = "；".join(
        f"`{n}` {r['b_app_year_same']:+.3f} → {r['b_appxmonth']:+.3f}" for n, r in pair.iterrows()
    )
    lines += [
        f"- 同一子样本内，把参照组从「App+年」收紧到「App+月」，11 个议题的系数最大只动了 "
        f"{max_delta:.3f}（{pair_txt}）。",
        f"- 符号一致性：A 与 B 同号 {same_sign}/11，A* 与 C 同号 {same_sign_c}/11；C 的显著性"
        f"（BH 校正后）在 11 个议题中保留了 "
        f"{int((fe['p2_bh'] < 0.05).sum())} 个。",
        "- **含义**：原始差值（活动 +0.220）到口径 A（+0.077）的落差确实来自 App 构成与年份构成；"
        "但在同一 App、同一月份内部再比较，系数不再继续收缩——剩下这部分是 App 内部的真实差异，"
        "不是「哪个 App 样本多、哪个月份评论多」造成的。",
        "",
        "## 4. 分组形状（仅供参考）",
        "",
        md_table(sh, ["产品类型", "中段条数", "议题", "d2", "d3", "d4", "d5", "形状"],
                 ["产品类型", "中段条数", "议题", "2 星", "3 星", "4 星", "5 星", "形状"]),
        "",
        "平台型的中段只有 11 条（4 星几乎没有），估不出系数，已从表中剔除。"
        "其余三类每类也只有几十条，系数标准误很大，**这张表只用于看方向是否离谱**，"
        "不作为形状判定依据；形状判定仍以主报告第 4.2 节（全样本、控制 App 与年份）为准。",
        "",
        "## 5. 结论与边界",
        "",
        f"- 分段结构的方向不依赖某一类产品：8 个关键议题里 7 个四类同号；唯一不稳的是 `客服售后与配送体检`"
        "（平台型反向、AI 助手约为 0），应降级为弱证据，不与活动、付费并列。",
        "- 收紧到 App×月、在同一子样本内比较，中段议题（技术故障、App 功能）与高端议题（医生合并）的系数几乎不动，"
        "**说明分段结构不是 App×时间构成造成的**，而是 App 内部的真实差异。",
        f"- 活动议题的原始差距集中在平台型（{act['平台型']:+.3f}），其他三类只有 "
        f"{act['互联网医院']:+.3f}–{act['AI 健康助手']:+.3f}。这从另一条路径支持主报告"
        "「只报固定效应量级 +0.07、不外推为普遍现象」的口径。",
        "- **仍不能说的话**：中段只有 184 条，分组后无法判形状；App×月 只保留 "
        f"{diag['n_eff']} 条，接近可识别下限，系数不能当点估计引用，只能当方向证据。",
        "- 8 个 App 聚类太少，所有 p 值都应按「偏宽松」读；真正的稳健性仍依赖 `stability.md` 的多种子与 Bootstrap。",
        "",
        "*生成脚本：`scripts/17_heterogeneity.py`。*",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
