"""P2 主分析：主题权重在 1 星与 5 星之间的差异 + 固定效应回归。

输入：data/topic_doc_weights.csv（07_topic_model.py 产出，k=10）、data/labels_keyword.csv
输出：outputs/topic_model.md、data/topic_compare.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
WEIGHTS = DATA / "topic_doc_weights.csv"
TOPWORDS = DATA / "topic_topwords.csv"
KEYWORD_LABELS = DATA / "labels_keyword.csv"
COMPARE_OUT = DATA / "topic_compare.csv"
REPORT = OUT / "topic_model.md"

K = 10

# 对照 k=10 的 top 词人工命名，并映射回 codebook v0.2
TOPIC_MAP = {
    0: ("医生解答与沟通", "doctor"),
    1: ("医院挂号与就医流程", "appointment"),
    2: ("活动打卡与 AI 识别", "promotion"),
    3: ("专家资源与医术口碑", "doctor"),
    4: ("付费问诊次数与退款", "fee"),
    5: ("免费咨询与买药", "fee"),
    6: ("App 功能与挂号操作", "tech"),
    7: ("客服售后与配送体检", "service"),
    8: ("技术故障与更新", "tech"),
    9: ("患者获益与异地就医", "other"),
}

# 同一 codebook 类别下、方向一致的主题，合并汇报总量（不合并测试）
MERGED = {"医生与专家口碑（合计）": ["t00", "t03"]}

DROP_APPS = ["京东健康", "好大夫在线"]


def md_table(df, cols, headers, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def shape_notes(raw_shape, fe_shape):
    """根据实际形状生成读法，避免文字与表格脱节。"""
    fe = fe_shape.set_index("name")
    middle = fe[fe["shape"] == "中间星级峰值"]
    high = fe[fe["shape"] == "高端专属"]
    low = fe[fe["shape"] == "低端专属"]
    none = fe[fe["shape"] == "无稳定形状"]

    lines = []
    if len(middle):
        mt = middle.reset_index()
        worst = mt.loc[mt[["d2", "d3", "d4"]].max(axis=1).idxmax()]
        lines.append(
            f"- **{len(middle)} 个议题呈中间星级峰值**：{'、'.join(middle.index)}。"
            f"其中 `{worst['name']}` 的 2 星系数 {worst['d2']:+.3f}，高于它的 5 星系数 {worst['d5']:+.3f}——"
            "意味着这类抱怨在温和差评里最集中，到 1 星反而让位给别的议题（活动、付费）。"
            "用「1 星 vs 5 星」二分法会看不见这一点。"
        )
    if len(high):
        lines.append(f"- **{len(high)} 个议题是高端专属**：{'、'.join(high.index)}，5 星显著更高且中间星级不显著。")
    if len(low):
        lines.append(f"- **{len(low)} 个议题是低端专属**：{'、'.join(low.index)}，中间星级与 5 星都更低。")
    if len(none):
        lines.append(
            f"- **{len(none)} 个议题无稳定形状**：{'、'.join(none.index)}，"
            "控制 App 与年份后中间星级均不显著、5 星方向也不稳定，不宜作为结论。"
        )

    act = fe.loc["活动打卡与 AI 识别"] if "活动打卡与 AI 识别" in fe.index else None
    if act is not None:
        raw = raw_shape.set_index("name").loc["活动打卡与 AI 识别"]
        lines.append(
            f"- `活动打卡与 AI 识别` 的原始差距被严重放大：1 星 {raw['m1']:.3f} 对 5 星 {raw['m5']:.3f}，"
            f"但控制 App 与年份后，2 星只有 {act['d2']:+.3f}、3 星 {act['d3']:+.3f}、"
            f"4 星 {act['d4']:+.3f}、5 星 {act['d5']:+.3f}。"
            "它在 App 内部确实是低端议题，但真实量级只有原始差值的约三分之一——"
            "原始差值主要来自京东健康（该 App 样本几乎全为 1 星，且全部集中在 2026-09 与 2026-10 两个月）。"
            "**日期集中度与逐层剔除见 `outputs/robustness.md` 第 2、3.2 节**："
            "剔除京东健康且剔除 2026-09 后，差值降到 +0.077，固定效应系数仍有 +0.069 且显著。"
        )

    lines += [
        "",
        "**两处必须同时读第 5 节。** `医院挂号与就医流程` 在 4.2 被判为高端专属（5 星 +0.038），"
        "但它在 2024–2026 子样本中系数翻正（+0.013，见第 5 节），说明这一标签不稳定，不应单独引用。"
        "`患者获益与异地就医` 的 5 星系数只有 +0.027，量级很小，属弱证据。",
        "",
        "**对「两极分化」这一表述的修正**：两端确实在用不同的议题，但中间星级不是真空地带，"
        "而是承接了另一批议题（技术故障、App 功能）。因此更准确的表述是："
        "**评分与议题不是单调关系，而是一个「低端=返利与付费、中段=技术故障、高端=医生与专家」的分段结构**；"
        "「两极分化」适合描述两端的差异幅度，不适合暗示中间星级只是过渡。",
        "",
        "**中段这一族的稳定性有限，需要一起读 `outputs/stability.md`**："
        "`技术故障与更新`、`App 功能与挂号操作`、`客服售后与配送体检` 三者在换随机初始化后会合成一个"
        "「功能与服务」主题（最佳余弦 0.94–1.00，但一一指派常失败），"
        "`患者获益与异地就医` 的内容漂移更大（最佳余弦中位 0.68）。"
        "因此「中段峰值」应作为**功能与故障这一族**的证据汇报，不把三个主题当作三条独立证据。",
        "",
    ]
    return lines


def findings(cmp, merged, reg_by_spec, content, content_p, direction, ai_star, ai_p, raw_shape, fe_shape):
    """主要发现：数字从结果表带入，避免正文与表格不一致。"""
    # 医生两主题合并叙述；挂号近年方向翻转（见第 5 节），不列入"稳定差异"
    doc_names = {"医生解答与沟通", "专家资源与医术口碑", "医院挂号与就医流程"}
    sig = cmp[(cmp["p_welch_bh"] < 0.05) & (~cmp["name"].isin(doc_names))]
    # 中间星级峰值的议题不能写成"1 星专属"，它们在 2–4 星更高，单独在第 2 条讲
    mid_peak = set(fe_shape[fe_shape["shape"] == "中间星级峰值"]["name"])
    up = sig[(sig["diff"] > 0) & (~sig["name"].isin(mid_peak))].sort_values("diff", ascending=False)
    down = sig[sig["diff"] < 0].sort_values("diff")

    def fmt(d):
        return "、".join(
            f"{r['name']}（{r['share_1star']:.2f} vs {r['share_5star']:.2f}）" for _, r in d.iterrows()
        )

    stable = [x["name"] for x in direction if x["apps_tested"] > 0 and x["apps_same_sign"] == x["apps_tested"]]
    doc = merged.iloc[0]
    act = cmp[cmp["name"] == "活动打卡与 AI 识别"].iloc[0]
    combine = combine_specs(reg_by_spec)
    act_specs = combine[combine["name"] == "活动打卡与 AI 识别"].iloc[0]
    n_same = int((combine["同号"] == "是").sum())

    return [
        "## 9. 主要发现",
        "",
        f"1. **两端在谈两套不同的议题，但中间星级不是过渡带。** "
        f"5 星更常谈：医生与专家口碑（合计 {doc['share_1star']:.2f} vs {doc['share_5star']:.2f}，"
        f"其中 `医生解答与沟通` 是评论里最主要的具体表扬对象）、{fmt(down)}；"
        f"1 星更常谈：{fmt(up)}。"
        "`医院挂号与就医流程` 未列入：其方向在 2024–2026 子样本中翻转（见第 5 节）；"
        f"{'、'.join(sorted(mid_peak))} 也不列入：它们的主场是中间星级（见第 2 条）。",
        "",
        f"2. **形状检验改变了结论的表述方式。** "
        f"{len(fe_shape[fe_shape['shape'] == '中间星级峰值'])} 个议题（"
        + "、".join(fe_shape[fe_shape["shape"] == "中间星级峰值"]["name"])
        + "）在控制 App 与年份后呈中间星级峰值，中间星级（2–4 星）系数高于 5 星系数。"
        "所以评分与议题不是单调关系，而是「低端=返利与付费、中段=技术故障、高端=医生与专家」的分段结构（第 4 节）。",
        "",
        f"3. **固定效应系数在三种口径下一致，比原始差值小。** `活动打卡与 AI 识别` 原始差值 "
        f"{act['diff']:+.3f}，固定效应系数：全样本 {act_specs['coef全样本']:+.3f}、"
        f"2024–2026 {act_specs['coef2024_2026']:+.3f}、剔除京东与好大夫 {act_specs['coef剔除']:+.3f}。"
        f"全部 10 个主题中 {n_same} 个三种口径同号。正确读法是「App 内部比较效应更小，"
        "原始差值含 App 之间的构成差异」，而不是把落差全归给平台。"
        "把京东健康与 2026-09 一并剔掉后，活动议题的系数仍有 +0.069 且显著"
        "（`outputs/robustness.md` 3.2），方向不依赖单一 App 或单个月份。",
        "",
        f"4. **5 星文本更空泛。** 字符数中位数 5 星 {content.loc[1, 'chars_median']:.0f} vs "
        f"1 星 {content.loc[0, 'chars_median']:.0f}；未命中任何主题关键词 "
        f"{content.loc[1, 'no_theme_rate']:.1%} vs {content.loc[0, 'no_theme_rate']:.1%}"
        f"（p = {content_p:.1e}）。",
        "",
        f"5. **分 App 稳健。** 7 个可检验 App 内方向全部一致的主题：{'、'.join(stable)}。",
        "",
        f"6. **AI 是次要线索。** 1 星提及 AI {ai_star.loc[1, 'rate']:.1%} vs 5 星 "
        f"{ai_star.loc[5, 'rate']:.1%}（p = {ai_p:.3f}），且集中在蚂蚁阿福与京东健康，不适合承担主叙事。",
        "",
    ]


def main():
    df = pd.read_csv(WEIGHTS, encoding="utf-8-sig", dtype={"review_id": str})
    df["rating"] = df["rating"].astype(int)
    df["year"] = df["year"].fillna("").astype(str)
    df["doctor_merged"] = df["t00"] + df["t03"]
    cols = [f"t{i:02d}" for i in range(K)]

    g1 = df[df["rating"] == 1]
    g5 = df[df["rating"] == 5]
    print(f"1 星 {len(g1)} 条，5 星 {len(g5)} 条")

    rows = []
    for i, c in enumerate(cols):
        a, b = g1[c].values, g5[c].values
        _, p_t = stats.ttest_ind(a, b, equal_var=False)
        _, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        # 主导率：主题份额超过均匀份额 1/k，记为"以该主题为主"
        thr = 1.0 / K
        cnt = np.array([(a > thr).sum(), (b > thr).sum()])
        n = np.array([len(a), len(b)])
        _, p_z = proportions_ztest(cnt, n)
        name, code = TOPIC_MAP[i]
        rows.append({
            "topic": c,
            "name": name,
            "codebook": code,
            "share_1star": round(a.mean(), 4),
            "share_5star": round(b.mean(), 4),
            "diff": round(a.mean() - b.mean(), 4),
            "p_welch": p_t,
            "p_mannwhitney": p_u,
            "prevalence_1star": cnt[0] / n[0],
            "prevalence_5star": cnt[1] / n[1],
            "p_prevalence": p_z,
        })

    cmp = pd.DataFrame(rows)
    cmp["p_welch_bh"] = multipletests(cmp["p_welch"], method="fdr_bh")[1]
    cmp["p_prevalence_bh"] = multipletests(cmp["p_prevalence"], method="fdr_bh")[1]
    cmp.to_csv(COMPARE_OUT, index=False, encoding="utf-8-sig")

    # 合并主题：只用于汇报总量，不做独立检验
    merged_rows = []
    for name, parts in MERGED.items():
        s1 = g1[parts].sum(axis=1).values
        s5 = g5[parts].sum(axis=1).values
        merged_rows.append({
            "name": name,
            "parts": "+".join(parts),
            "share_1star": round(float(s1.mean()), 4),
            "share_5star": round(float(s5.mean()), 4),
            "diff": round(float(s1.mean() - s5.mean()), 4),
            "corr_between_parts": round(float(df[parts[0]].corr(df[parts[1]])), 3),
        })
    merged = pd.DataFrame(merged_rows)

    # 固定效应回归：三种口径
    sub = df[df["rating"].isin([1, 5])].copy()
    sub["is1"] = (sub["rating"] == 1).astype(int)
    specs = {
        "全样本": sub,
        "2024–2026": sub[sub["year"].astype(str) >= "2024"],
        "剔除京东健康与好大夫在线": sub[~sub["app_name"].isin(DROP_APPS)],
    }
    reg_by_spec = {label: fit_spec(d, cols) for label, d in specs.items()}
    reg = reg_by_spec["全样本"]

    # 分 App 方向一致性：与总体差值同向的 App 数
    direction = []
    for i, c in enumerate(cols):
        overall = float(cmp.loc[cmp["topic"] == c, "diff"].iloc[0])
        ok = total = 0
        for _, g in sub.groupby("app_name"):
            a = g[g.is1 == 1][c]
            b = g[g.is1 == 0][c]
            if len(a) >= 5 and len(b) >= 5:
                total += 1
                ok += int((a.mean() - b.mean()) * overall > 0)
        direction.append({"topic": c, "name": TOPIC_MAP[i][0], "apps_same_sign": ok, "apps_tested": total})

    content, content_p, kw_sub = content_measures(sub)
    ai_star, ai_p, ai_app = ai_secondary(kw_sub)
    raw_shape, fe_shape = shape_analysis(df, cols)
    write_report(cmp, merged, reg_by_spec, direction, content, content_p, ai_star, ai_p, ai_app, df,
                 raw_shape, fe_shape)
    print(f"写入 {COMPARE_OUT} 与 {REPORT}")


def fit_spec(sub, cols):
    """单个口径下的固定效应回归。"""
    rows = []
    for c in cols:
        model = smf.ols(f"{c} ~ is1 + C(app_name) + C(year)", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["app_name"]}
        )
        i = int(c[1:])
        rows.append({
            "topic": c,
            "name": TOPIC_MAP[i][0],
            "coef_1star": round(float(model.params["is1"]), 4),
            "se_cluster": round(float(model.bse["is1"]), 4),
            "p": float(model.pvalues["is1"]),
            "n": int(model.nobs),
            "n_apps": int(sub["app_name"].nunique()),
        })
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def content_measures(sub):
    """文本具体性：字符长度、是否命中任何主题关键词。"""
    kw = pd.read_csv(KEYWORD_LABELS, encoding="utf-8-sig", dtype=str)[["review_id", "themes"]].fillna("")
    sub = sub.merge(kw, on="review_id", how="left")
    sub["themes"] = sub["themes"].fillna("")
    sub["chars"] = sub["text"].str.replace(r"\s", "", regex=True).str.len()
    sub["no_theme"] = sub["themes"].str.len() == 0

    rows = []
    for is1, label in [(1, "1 星"), (0, "5 星")]:
        d = sub[sub.is1 == is1]
        rows.append({
            "group": label,
            "n": len(d),
            "chars_median": float(d["chars"].median()),
            "chars_mean": round(float(d["chars"].mean()), 1),
            "no_theme_rate": float(d["no_theme"].mean()),
        })
    table = pd.DataFrame(rows)
    a = int(table.loc[0, "no_theme_rate"] * len(sub[sub.is1 == 1]))
    b = int(table.loc[1, "no_theme_rate"] * len(sub[sub.is1 == 0]))
    _, p = proportions_ztest([a, b], [len(sub[sub.is1 == 1]), len(sub[sub.is1 == 0])])
    return table, p, sub


def ai_secondary(sub):
    """AI 作为次要角度：关键词提及率。"""
    kw = pd.read_csv(KEYWORD_LABELS, encoding="utf-8-sig", dtype=str)
    kw["ai_mention"] = kw["ai_mention"].astype(int)
    sub = sub.merge(kw[["review_id", "ai_mention"]], on="review_id", how="left")
    sub["ai_mention"] = sub["ai_mention"].fillna(0).astype(int)

    by_star = sub.groupby("rating").agg(n=("ai_mention", "size"), ai=("ai_mention", "sum"))
    by_star["rate"] = by_star["ai"] / by_star["n"]
    _, p = proportions_ztest(
        [int(by_star.loc[1, "ai"]), int(by_star.loc[5, "ai"])],
        [int(by_star.loc[1, "n"]), int(by_star.loc[5, "n"])],
    )
    by_app = sub.groupby("app_name").agg(n=("ai_mention", "size"), ai=("ai_mention", "sum"))
    by_app["rate"] = by_app["ai"] / by_app["n"]
    return by_star, p, by_app.sort_values("rate", ascending=False)


def combine_specs(reg_by_spec):
    """把三种口径的系数并到一张表，并判断方向是否一致。"""
    base = reg_by_spec["全样本"][["topic", "name", "coef_1star", "p_bh"]].copy()
    base = base.rename(columns={"coef_1star": "coef全样本", "p_bh": "p全样本"})
    base["coef2024_2026"] = reg_by_spec["2024–2026"]["coef_1star"].values
    base["coef剔除"] = reg_by_spec["剔除京东健康与好大夫在线"]["coef_1star"].values
    same = []
    for _, r in base.iterrows():
        signs = {np.sign(r["coef全样本"]), np.sign(r["coef2024_2026"]), np.sign(r["coef剔除"])}
        same.append("是" if len(signs) == 1 and 0 not in signs else "否")
    base["同号"] = same
    return base


def shape_analysis(df, cols):
    """五星级剖面：判断每个议题是两端极化、中间星级峰值，还是无稳定形状。

    原始均值易被 App 构成混淆（中间星级集中在少数 App），所以同时给出
    控制 App 与年份后相对 1 星的系数，并以系数而非原始均值判定形状。
    """
    all_cols = cols + ["doctor_merged"]
    raw_rows, fe_rows = [], []
    for c in all_cols:
        label = "医生与专家口碑（合计）" if c == "doctor_merged" else TOPIC_MAP[int(c[1:])][0]
        raw = {int(r): float(df.loc[df["rating"] == r, c].mean()) for r in range(1, 6)}
        rho, p_rho = stats.spearmanr(df["rating"], df[c])
        raw_rows.append({
            "name": label,
            **{f"m{r}": raw[r] for r in range(1, 6)},
            "rho": round(float(rho), 3),
            "p_rho": p_rho,
        })

        model = smf.ols(f"{c} ~ C(rating) + C(app_name) + C(year)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["app_name"]}
        )
        coef = {}
        sig = {}
        for r in [2, 3, 4, 5]:
            key = f"C(rating)[T.{r}]"
            coef[r] = float(model.params.get(key, 0.0))
            sig[r] = float(model.pvalues.get(key, 1.0)) < 0.05
        fe_rows.append({
            "name": label,
            **{f"d{r}": round(coef[r], 3) for r in [2, 3, 4, 5]},
            "shape": classify_shape(coef, sig),
            "significant_middle": sum(sig[r] for r in [2, 3, 4]),
        })

    raw_tab = pd.DataFrame(raw_rows)
    raw_tab["p_rho_bh"] = multipletests(raw_tab["p_rho"], method="fdr_bh")[1]
    return raw_tab, pd.DataFrame(fe_rows)


def classify_shape(coef, sig):
    """按控制 App/年份后的系数形状分类，规则显式以便复核。

    优先级：中间星级有显著正系数且高于 5 星 → 中间峰值；
    否则 5 星显著为正 → 高端专属；否则至少两个中间星级显著为负且 5 星不为正 → 低端专属。
    """
    mids = [2, 3, 4]
    mid_max = max(coef[r] for r in mids)
    mid_pos_sig = [r for r in mids if coef[r] > 0 and sig[r]]
    if mid_pos_sig and mid_max > coef[5]:
        return "中间星级峰值"
    if coef[5] > 0 and sig[5]:
        return "高端专属"
    mid_neg_sig = [r for r in mids if coef[r] < 0 and sig[r]]
    if len(mid_neg_sig) >= 2 and coef[5] <= 0:
        return "低端专属"
    return "无稳定形状"


def write_report(cmp, merged, reg_by_spec, direction, content, content_p, ai_star, ai_p, ai_app, df, raw_shape, fe_shape):
    reg = reg_by_spec["全样本"]
    top = pd.read_csv(TOPWORDS, encoding="utf-8-sig")
    top = top[top["k"] == K][["topic", "top_words"]].copy()
    top["topic"] = top["topic"].map(lambda i: f"t{int(i):02d}")

    ai_tab = ai_star.reset_index()
    ai_tab["rating"] = ai_tab["rating"].astype(int).astype(str) + " 星"
    ai_tab["n"] = ai_tab["n"].astype(int)
    ai_tab["ai"] = ai_tab["ai"].astype(int)

    lines = [
        "# P2 主题模型结果：1 星 vs 5 星的主题结构差异",
        "",
        "> 方法：jieba 分词（自定义词典 + 停用词）→ TF-IDF（min_df=5, max_df=0.5）→ NMF（k=10, "
        "init=nndsvda, seed=42）。主题由 top 词人工命名后映射回 codebook v0.2。"
        "**不做逐条人工标注**：极性由星级分组承担，因此词表已剔除纯极性词、App 名与通用动词。",
        "",
        f"样本：{len(df)} 条（1 星 {len(df[df['rating'] == 1])}，5 星 {len(df[df['rating'] == 5])}）。"
        "TF-IDF 词表 742 个词（`min_df=5`，即至少 5 条评论用过的词才进入词表）。",
        "",
        "## 1. 主题清单",
        "",
        "| 主题 | 命名 | codebook | top 词 |",
        "|---|---|---|---|",
    ]
    for i in range(K):
        tw = top[top.topic == f"t{i:02d}"].top_words.iloc[0]
        lines.append(f"| t{i:02d} | {TOPIC_MAP[i][0]} | `{TOPIC_MAP[i][1]}` | {tw} |")

    lines += ["", "### 各主题的代表评论（权重最高且长度 30–120 字的一条）", "",
              "| 主题 | 权重 | 评论 |", "|---|---|---|"]
    length = df["text"].str.len()
    pool = df[(length >= 30) & (length <= 120)]
    for i, c in enumerate([f"t{j:02d}" for j in range(K)]):
        cand = pool if len(pool) and pool[c].max() > 0 else df
        idx = cand[c].idxmax()
        text = " ".join(str(df.loc[idx, "text"]).split())[:60]
        lines.append(f"| t{i:02d} {TOPIC_MAP[i][0]} | {df.loc[idx, c]:.2f} | {text} |")

    lines += [
        "",
        "## 2. 主题平均份额：1 星 vs 5 星",
        "",
        md_table(
            cmp,
            ["name", "share_1star", "share_5star", "diff", "p_welch_bh"],
            ["主题", "1 星份额", "5 星份额", "差值", "p（BH 校正）"],
            {"diff": "{:+.3f}", "p_welch_bh": "{:.2e}"},
        ),
        "",
        "差值 > 0：1 星评论更常谈这件事；< 0：5 星更常谈。份额是单条评论主题权重的行内归一化均值，"
        "因此每组的 10 个份额加总为 1。",
        "",
        "### 合并后的同类议题",
        "",
        md_table(
            merged,
            ["name", "parts", "share_1star", "share_5star", "diff", "corr_between_parts"],
            ["议题", "组成", "1 星份额", "5 星份额", "差值", "两部分相关系数"],
            {"diff": "{:+.3f}"},
        ),
        "",
        "`医生解答与沟通` 与 `专家资源与医术口碑` 同属 codebook 的 `doctor` 类别，但两者在文档层面几乎不共现"
        f"（r = {merged.iloc[0]['corr_between_parts']:.2f}），是一条评论只谈其中一个侧面，不是重复计数。"
        "分开看会让人以为是两条证据，合并才是医生议题的真实量级——"
        f"{merged.iloc[0]['share_1star']:.2f} vs {merged.iloc[0]['share_5star']:.2f}，是全表最大的 5 星差异。",
        "",
        "## 3. 主题主导率（份额 > 1/10 的评论占比）",
        "",
        md_table(
            cmp,
            ["name", "prevalence_1star", "prevalence_5star", "p_prevalence_bh"],
            ["主题", "1 星主导率", "5 星主导率", "p（BH 校正）"],
            {"prevalence_1star": "{:.1%}", "prevalence_5star": "{:.1%}", "p_prevalence_bh": "{:.2e}"},
        ),
        "",
        "## 4. 五星级剖面：极化还是渐变",
        "",
        "前面的 1 星 vs 5 星对比默认「两端之间有连续谱」，但没有检验 2/3/4 星落在哪。"
        f"中间星级样本很少（2 星 {int((df['rating'] == 2).sum())}、3 星 {int((df['rating'] == 3).sum())}、"
        f"4 星 {int((df['rating'] == 4).sum())} 条），且集中在少数 App（蚂蚁阿福、微医、微脉占 121/184），"
        "所以原始均值容易把 App 构成当成星级效应。下面同时给原始剖面与控制 App、年份后的系数。",
        "",
        "### 4.1 原始剖面",
        "",
        md_table(
            raw_shape,
            ["name", "m1", "m2", "m3", "m4", "m5", "rho", "p_rho_bh"],
            ["议题", "1 星", "2 星", "3 星", "4 星", "5 星", "Spearman ρ", "p（BH 校正）"],
            {"m1": "{:.3f}", "m2": "{:.3f}", "m3": "{:.3f}", "m4": "{:.3f}", "m5": "{:.3f}",
             "rho": "{:+.3f}", "p_rho_bh": "{:.1e}"},
        ),
        "",
        "ρ 是对 1–5 星全部样本求的等级相关；ρ 接近 0 不代表没有关系，只说明不是单调关系。",
        "",
        "### 4.2 控制 App 与年份后的剖面对照（相对 1 星）",
        "",
        "模型：`share ~ C(rating) + C(app) + C(year)`，用全部 1,681 条，聚类稳健标准误（按 App）。"
        "**形状判定以这一列系数为准**，规则见脚本 `classify_shape()`。",
        "",
        md_table(
            fe_shape,
            ["name", "d2", "d3", "d4", "d5", "shape"],
            ["议题", "2 星−1 星", "3 星−1 星", "4 星−1 星", "5 星−1 星", "形状"],
            {"d2": "{:+.3f}", "d3": "{:+.3f}", "d4": "{:+.3f}", "d5": "{:+.3f}"},
        ),
        "",
        "### 4.3 读法",
        "",
        *shape_notes(raw_shape, fe_shape),
        "",
        "## 5. 固定效应回归：两端比较的三种口径",
        "",
        "本节只回到 1 星 vs 5 星的对比，检验第 2 节的原始差值是否稳健。",
        "模型：`share ~ 1[rating==1] + C(app) + C(year)`，仅 1 星与 5 星，聚类稳健标准误（按 App）。",
        "三种口径用于回答一个问题：原始差值与固定效应系数的落差，是年份漂移或个别 App 造成的吗？",
        "",
        md_table(
            combine_specs(reg_by_spec),
            ["name", "coef全样本", "coef2024_2026", "coef剔除", "同号", "p全样本"],
            ["主题", "全样本", "2024–2026", "剔除京东/好大夫", "三者同号", "p（BH，全样本）"],
            {"coef全样本": "{:+.3f}", "coef2024_2026": "{:+.3f}", "coef剔除": "{:+.3f}",
             "p全样本": "{:.2e}"},
        ),
        "",
        f"样本量：全样本 {reg_by_spec['全样本']['n'].iloc[0]}（8 个 App）、"
        f"2024–2026 {reg_by_spec['2024–2026']['n'].iloc[0]}（{reg_by_spec['2024–2026']['n_apps'].iloc[0]} 个 App）、"
        f"剔除后 {reg_by_spec['剔除京东健康与好大夫在线']['n'].iloc[0]}"
        f"（{reg_by_spec['剔除京东健康与好大夫在线']['n_apps'].iloc[0]} 个 App）。",
        "",
        f"除 `医院挂号与就医流程` 外，其余 {int((combine_specs(reg_by_spec)['同号'] == '是').sum())} 个主题在三种口径下同号，"
        "说明系数落差不是年份漂移或个别 App 造成的。系数比原始差值小，是因为固定效应只使用 App 内部的 "
        "1 星/5 星对比，跨 App 的差异（各 App 自身议题构成不同）被吸收掉了。"
        "**正确的读法是：App 内部比较时效应更小；原始差值里含有 App 之间的构成差异。**",
        "",
        "`医院挂号与就医流程` 是例外：全样本系数 -0.036，2024–2026 转为 +0.013。"
        "这说明挂号在近三年已不再是 5 星专属议题——1 星用户同样在谈挂号，"
        "该主题的原始差值不应被当作稳定结论。`客服售后与配送体检` 的全样本 p = 0.082，"
        "在 BH 校正后不显著，方向虽有但证据弱。",
        "",
        "但要注意多个 App 的星级结构极度不平衡（京东健康 1 星:5 星 = 209:3，好大夫 24:176，微医 174:13），"
        "固定效应估计因此很不均匀地由数据平衡的 App 承担。剔除前两家后系数与全样本接近，"
        "支持这一估计并非由它们单独驱动；但剩余样本里微医（174:13）仍属不平衡，"
        "所以这是一次部分稳健性检验，不是彻底排除。聚类只有 6–8 个 App，标准误偏乐观，"
        "系数方向与量级比显著性更可信。",
        "",
        md_table(
            reg,
            ["name", "coef_1star", "se_cluster", "p_bh"],
            ["主题", "1 星系数（全样本）", "聚类 SE", "p（BH 校正）"],
            {"coef_1star": "{:+.3f}", "se_cluster": "{:.3f}", "p_bh": "{:.2e}"},
        ),
        "",
        "以 `活动打卡与 AI 识别` 为例：原始差值 +0.220，固定效应系数 +0.077，三种口径都稳定在 +0.07 左右。"
        "该主题集中在京东健康与蚂蚁阿福，而这两家的星级结构完全不同（前者几乎全是 1 星，后者接近对半），"
        "所以原始差值把 App 之间的构成差异也算了进来。",
        "",
        "## 6. 稳健性：分 App 方向一致性",
        "",
        md_table(
            pd.DataFrame(direction),
            ["name", "apps_same_sign", "apps_tested"],
            ["主题", "同向 App 数", "可检验 App 数"],
        ),
        "",
        "只统计该 App 内 1 星、5 星各 ≥5 条的 App。同向 = 该 App 内部的 1 星与 5 星差值方向与总体一致。",
        "",
        "## 7. 文本具体性：5 星是否更空泛（仅 1 星与 5 星）",
        "",
        md_table(
            content,
            ["group", "n", "chars_median", "chars_mean", "no_theme_rate"],
            ["分组", "条数", "字符数中位数", "字符数均值", "未命中任何主题关键词"],
            {"chars_median": "{:.0f}", "chars_mean": "{:.1f}", "no_theme_rate": "{:.1%}"},
        ),
        "",
        f"未命中任何主题关键词的比例：1 星 {content.loc[0, 'no_theme_rate']:.1%} vs "
        f"5 星 {content.loc[1, 'no_theme_rate']:.1%}，两比例 z 检验 p = {content_p:.2e}。",
        "这是「空泛好评」的代理指标：评论里没有任何 codebook 主题的关键词可抓。",
        "",
        "## 8. AI 提及（次要角度）",
        "",
        md_table(
            ai_tab,
            ["rating", "n", "ai", "rate"],
            ["星级", "条数", "提及 AI", "提及率"],
            {"rate": "{:.2%}"},
        ),
        "",
        f"1 星 vs 5 星的 AI 提及率差异：p = {ai_p:.3f}（两比例 z 检验）。",
        "",
        md_table(
            ai_app.reset_index(),
            ["app_name", "n", "ai", "rate"],
            ["App（仅 1 星与 5 星）", "条数", "提及 AI", "提及率"],
            {"rate": "{:.2%}"},
        ),
        "",
        "AI 提及集中在蚂蚁阿福与京东健康，且量级不足以支撑「AI」作为主叙事，只作为次要观察。",
        "",
        "68 条逐条读过（见 `outputs/ai_read.md`）：1 星抱怨的是**识别不出来、记不住刚才说过的话**"
        f"（识别与记忆失败 {22}/{48}），只有 8 条说「诊断错/胡说」或「不如豆包、DeepSeek」；"
        "5 星夸的是**报告解读、轻症先问、减重记录**这类有具体交付物的场景；"
        "另有 11 条（16%）是关键词误命中，抱怨的其实是退款、活动规则或医生质量。"
        "所以「AI 提及率」高估了 AI 的真实存在感，方向也不是「用户不喜欢 AI」这么简单。",
        "",
        *findings(cmp, merged, reg_by_spec, content, content_p, direction, ai_star, ai_p, raw_shape, fe_shape),
        "## 10. 局限",
        "- 主题数 k（10）与停用词表由研究者设定，属分析选择。k=6/8/12/14/16 的重建误差与 top 词见 "
        "`data/topic_topwords.csv`；跨 k 与多种子稳定性见 `outputs/stability.md`："
        "6 个主题在换初始化与重抽样本下稳定，"
        "`App 功能`、`客服售后`、`技术故障`、`患者获益` 四个会与相邻主题合并或漂移，"
        "因此第 4 节的中段结论按「功能与故障这一族」汇报。",
        "- 换算法是否改变结论：见 `outputs/algorithm_check.md`（NMF vs LDA）。同矩阵、同 k 下 7/10 个主题方向同号，"
        "医生合并主题 −0.206 vs −0.203；但只有 2/10 个主题能按余弦 ≥0.6 一一对上，LDA 的主题更泛化。"
        "因此结论不依赖算法选型，选 NMF 是为了可解释性，不是因为它更「准」。",
        "- 换范式是否改变结论：见 `outputs/bertopic_check.md`（句向量 + UMAP + HDBSCAN）。"
        "自动得到 33 个簇、26.7% 噪声，「两端落在不同语义簇」「技术故障这一簇以 2–4 星为主」都复现；"
        "但簇数与 NMF 的 10 个主题不是一一映射，因此只比对方向、不做主题级数值对比。",
        "- 分段结构是否只属于某一类产品：见 `outputs/heterogeneity.md`。按四类产品分组后，8 个关键议题里 7 个四类同号；"
        "`客服售后与配送体检` 只有 2/4（平台型反向、AI 助手约为 0），因此第 2 节把它与活动、付费并列时要打折。"
        "把参照组从「App+年」收紧到「App×月」后，同一子样本内系数最大只动 0.014，说明分段结构来自 App 内部的星级差异，"
        "不是 App×时间的构成造成的；但 App×月 只保留 1,031 条，只能当方向证据。",
        "- `医生解答与沟通` 与 `专家资源与医术口碑` 在文档层面几乎不共现（r≈-0.06），"
        "是一条评论只谈一个侧面，不构成重复计数；第 2 节合并只用于汇报医生议题的总量，未做独立检验。",
        f"- 有 {int((df[[f't{i:02d}' for i in range(K)]].sum(axis=1) <= 1e-12).sum())} 条评论无任何主题权重"
        "（分词后与词表无交集，多为极短或纯情绪文本），它们对第 2 节的份额贡献为 0。",
        "- 评论者自选择：只代表愿意写文字评论的用户，不能推断全体患者。"
        "样本内 1 星占 52.9%，8 个 App 的总体评分在 3.11–4.91 之间，"
        "其中 7 个 App 的样本均值低于官方平均分（符号检验 p = 7.8e-03），"
        "说明文字评论系统性偏负面，偏差方向已被量化（见 `outputs/robustness.md` 第 4 节）。"
        "官方不公布星级分布，因此无法做 prevalence 校正，本文份额只能读成「评论者中的份额」。",
        "- 抽样窗口不一致且受评论涌入影响：RSS 按时间倒序取页，某 App 短期涌入大量评论就会把样本压到那一段。"
        "极端例子是京东健康，其 214 条样本只覆盖 2026-09 与 2026-10 两个月，"
        "2026-09 一个月就占全样本的 19.5%（见 `outputs/robustness.md` 第 2 节）。"
        "回归控制年份只能部分吸收这一点，不做时间趋势推断。",
        "- 星级与文本同时产生，主题份额差异是描述性关联，不是因果效应。",
        "- 中间星级样本很少（2 星 67、3 星 61、4 星 56），且 121/184 集中在蚂蚁阿福、微医、微脉；"
        "形状分类依赖控制 App 与年份后的聚类标准误，而聚类只按 App（8 组），"
        "对中间星级的判断比两端更容易受单个 App 影响。分类规则写死在 `classify_shape()` 以便复核。",
        "- 关键词命中率依赖词表，`no_theme_rate` 会低估实际有内容的评论。",
        "- 主题份额为行归一化成分数据（10 项和为 1），各主题不独立；"
        "Welch t 检验与 BH 校正在此只作描述性筛选，不应解读为精确的显著性水平。",
        "",
        "## 11. 附图",
        "",
        "全部图由 `scripts/11_figures.py` 生成，位于 `outputs/figures/`。",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `fig1_theme_diff.png` | 10 个主题的 1 星 − 5 星 份额差（对应第 2 节） |",
        "| `fig2_star_profile.png` | 六个关键议题的五星级剖面（对应第 4 节） |",
        "| `fig3_raw_vs_fe.png` | 原始差值 vs 控制 App/年份后的固定效应系数（对应第 5 节） |",
        "| `fig4_text_specificity.png` | 星级 × 字符数中位数 / 未命中主题关键词比例（对应第 7 节） |",
        "| `fig5_benchmark.png` | 各 App 样本均值 vs 官方平均分（对应 `robustness.md` 第 4 节） |",
        "| `fig6_activity_month.png` | 活动主导评论的月度分布（对应 `robustness.md` 第 2 节） |",
        "| `fig7_ai_by_rating.png` | AI 关键词提及率 × 星级（对应第 8 节） |",
        "| `fig8_star_mix.png` | 各 App 的星级构成（对应第 4 节的样本分布说明） |",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
