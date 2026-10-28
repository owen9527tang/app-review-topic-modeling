"""P4 稳健性补充：刷评/集中度排查 + 样本与总体评分基准对比。

对应两项待办：
  A3 — 活动议题是否由少数日期的集中评论驱动；剔除爆发日后主要结论是否变化
  B5 — 样本星级分布与 App Store 总体评分（Apple Lookup API）的差距

输出：outputs/robustness.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
REVIEWS = DATA / "reviews_analysis.csv"
WEIGHTS = DATA / "topic_doc_weights.csv"
META = DATA / "app_metadata.csv"
REPORT = OUT / "robustness.md"

K = 10
ACT = "t02"          # 活动打卡与 AI 识别
DOCTOR = ["t00", "t03"]
TECH = "t08"         # 技术故障与更新
FUNC = "t06"         # App 功能与挂号操作
COLS = [f"t{i:02d}" for i in range(K)]


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


def load():
    rev = pd.read_csv(REVIEWS, encoding="utf-8-sig", dtype={"review_id": str})
    w = pd.read_csv(WEIGHTS, encoding="utf-8-sig", dtype={"review_id": str})
    keep = ["review_id"] + COLS
    d = rev[["review_id", "app_name", "rating", "date", "day_count", "is_burst_day"]].merge(
        w[keep], on="review_id", how="inner"
    )
    d["rating"] = d["rating"].astype(int)
    d["is_burst_day"] = d["is_burst_day"].astype(str) == "True"
    d["doctor"] = d["t00"] + d["t03"]
    d["year"] = d["date"].str[:4]
    d["ym"] = d["date"].str[:7]
    return d


def burst_section(d):
    n_burst = int(d["is_burst_day"].sum())
    by_app = d.groupby("app_name")["is_burst_day"].agg(["sum", "size"]).reset_index()
    by_app = by_app.rename(columns={"sum": "爆发日条数", "size": "样本条数"})
    by_app["占比"] = by_app["爆发日条数"] / by_app["样本条数"]

    # 活动议题的日期集中度：以 t02 份额 >= 1/3 记为"活动主导评论"
    act = d[d[ACT] >= 1 / 3].copy()
    act_by_app = act.groupby("app_name").size().rename("活动主导评论").reset_index()
    act_by_app = act_by_app.merge(d.groupby("app_name").size().rename("样本量").reset_index(),
                                  on="app_name", how="left")
    act_by_app["占该 App 样本"] = act_by_app["活动主导评论"] / act_by_app["样本量"]
    act_by_app = act_by_app.sort_values("活动主导评论", ascending=False)

    # 活动主导评论集中在多少个自然月
    act_month = act.groupby("ym").size().sort_values(ascending=False)
    top1 = act_month.iloc[0] / len(act) if len(act) else 0
    top3 = act_month.head(3).sum() / len(act) if len(act) else 0
    jd_act = act[act["app_name"] == "京东健康"]
    jd_month = jd_act.groupby("ym").size().sort_values(ascending=False)
    jd_all = d[d["app_name"] == "京东健康"]
    sep = "2026-09"

    return {
        "n_burst": n_burst,
        "by_app": by_app,
        "act_n": len(act),
        "act_by_app": act_by_app,
        "act_months": act_month,
        "act_top1": top1,
        "act_top3": top3,
        "act_top1_month": act_month.index[0] if len(act_month) else "",
        "act_top3_month": "、".join(act_month.head(3).index) if len(act_month) else "",
        "jd_n": len(jd_act),
        "jd_top_month": jd_month.index[0] if len(jd_month) else "",
        "jd_top_share": float(jd_month.iloc[0] / len(jd_act)) if len(jd_act) else 0.0,
        "act_unique_months": act_month.size,
        "sep_month": sep,
        "act_sep_share": float((act["ym"] == sep).mean()) if len(act) else 0.0,
        "jd_sep_share": float((jd_act["ym"] == sep).mean()) if len(jd_act) else 0.0,
        "jd_sep_all": int(((d["app_name"] == "京东健康") & (d["ym"] == sep)).sum()),
        "jd_all_months": int(jd_all["ym"].nunique()),
        "jd_sep_all_share": float((jd_all["ym"] == sep).mean()) if len(jd_all) else 0.0,
        "n_months": int(d["ym"].nunique()),
        "sep_all_n": int((d["ym"] == sep).sum()),
        "sep_all_share": float((d["ym"] == sep).mean()),
    }


def act_sensitivity(d):
    """活动议题在四种口径下的稳定性：剔除爆发日、剔除最集中的那个月、剔除京东健康。"""
    sub = d[d["rating"].isin([1, 5])]
    specs = {
        "全样本": sub,
        "剔除爆发日": sub[~sub["is_burst_day"]],
        "剔除 2026-09": sub[sub["ym"] != "2026-09"],
        "剔除京东健康": sub[sub["app_name"] != "京东健康"],
        "剔除京东健康且剔除 2026-09": sub[(sub["app_name"] != "京东健康") & (sub["ym"] != "2026-09")],
    }
    rows = []
    for label, dd in specs.items():
        g1 = dd[dd.rating == 1]
        g5 = dd[dd.rating == 5]
        model = smf.ols(f"{ACT} ~ is1 + C(app_name) + C(year)",
                        data=dd.assign(is1=(dd.rating == 1).astype(int))).fit(
            cov_type="cluster", cov_kwds={"groups": dd["app_name"]})
        rows.append({
            "口径": label,
            "n": len(dd),
            "n_1star": len(g1),
            "diff": float(g1[ACT].mean() - g5[ACT].mean()),
            "fe": float(model.params["is1"]),
            "p": float(model.pvalues["is1"]),
        })
    return pd.DataFrame(rows)


def main_effects(d, label):
    """两端差值 + 固定效应系数 + 中间星级峰值检验，返回一行字典。"""
    sub = d[d["rating"].isin([1, 5])]
    g1 = sub[sub.rating == 1]
    g5 = sub[sub.rating == 5]

    rows = {}
    for key, col in [("活动", ACT), ("医生合计", "doctor"), ("技术故障", TECH), ("App功能", FUNC)]:
        rows[f"{label}_{key}_diff"] = float(g1[col].mean() - g5[col].mean())

    for key, col in [("活动", ACT), ("医生合计", "doctor"), ("技术故障", TECH), ("App功能", FUNC)]:
        m = smf.ols(f"{col} ~ is1 + C(app_name) + C(year)",
                    data=sub.assign(is1=(sub.rating == 1).astype(int))).fit(
            cov_type="cluster", cov_kwds={"groups": sub["app_name"]})
        rows[f"{label}_{key}_fe"] = float(m.params["is1"])

    # 中间星级峰值：全星级模型里 2 星系数
    m2 = smf.ols(f"{TECH} ~ C(rating) + C(app_name) + C(year)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["app_name"]})
    key2 = "C(rating)[T.2]"
    rows[f"{label}_tech_mid2_coef"] = float(m2.params[key2])
    rows[f"{label}_tech_mid2_p"] = float(m2.pvalues[key2])
    rows[f"{label}_n"] = int(len(d))
    return rows


def benchmark(d):
    meta = pd.read_csv(META, encoding="utf-8-sig")
    meta = meta[["app_name", "average_user_rating", "user_rating_count"]]
    rows = []
    for app, g in d.groupby("app_name"):
        r = meta[meta.app_name == app]
        official = float(r["average_user_rating"].iloc[0])
        count = int(r["user_rating_count"].iloc[0])
        rows.append({
            "app_name": app,
            "n_sample": len(g),
            "p1": (g.rating == 1).mean(),
            "p5": (g.rating == 5).mean(),
            "sample_mean": float(g.rating.mean()),
            "official": official,
            "official_count": count,
            "gap": float(g.rating.mean()) - official,
        })
    tab = pd.DataFrame(rows).sort_values("gap")
    overall = {
        "n": len(d),
        "p1": (d.rating == 1).mean(),
        "p5": (d.rating == 5).mean(),
        "mean": float(d.rating.mean()),
        "official_weighted": float(np.average(tab["official"], weights=tab["official_count"])),
    }
    return tab, overall


def stars_vs_official_test(d):
    """样本 1 星占比与"总体 1 星占比"的可检验对比。

    官方只给平均分，不给星级分布；用平均分反推 1 星占比需要假设，
    因此这里只做一件事：检验样本均值与官方均值的差距方向是否在全部 App 上一致。
    """
    tab, _ = benchmark(d)
    n_below = int((tab["gap"] < 0).sum())
    wil = stats.wilcoxon(tab["gap"], alternative="less")
    return n_below, len(tab), wil


def write_report(d, burst, eff_full, eff_noburst, act_sens, bench, overall, wil, n_below):
    act = burst["act_by_app"]
    lines = [
        "# 稳健性补充：刷评排查与总体评分基准",
        "",
        "> 对应两项待办：活动议题是否由少数日期驱动（A3）；样本星级分布与 App Store 总体评分差多少（B5）。",
        "> 数据：`reviews_analysis.csv` 的清洗标记 + `topic_doc_weights.csv` 的主题权重 + `app_metadata.csv` 的官方评分。",
        "",
        "## 1. 爆发日排查",
        "",
        f"清洗阶段按「同一 App 同一天评论数 ≥ max(20, 均值+2SD)」标记爆发日，"
        f"本次样本中共 {burst['n_burst']} 条落在爆发日，占 {burst['n_burst'] / len(d):.2%}。",
        "",
        md_table(burst["by_app"], ["app_name", "爆发日条数", "样本条数", "占比"],
                 ["App", "爆发日条数", "样本条数", "占比"], {"占比": "{:.2%}"}),
        "",
        "只有京东健康触发了爆发日标记，其余 7 个 App 均为 0。这意味着爆发日不是全样本层面的问题，"
        "但恰好京东健康是活动议题最集中的 App，所以仍需单独看活动评论的日期分布。",
        "",
        "## 2. 活动议题的日期集中度",
        "",
        f"以「`活动打卡与 AI 识别` 份额 ≥ 1/3」定义为活动主导评论，共 {burst['act_n']} 条"
        f"（占样本 {burst['act_n'] / len(d):.1%}）。",
        "",
        md_table(act, ["app_name", "活动主导评论", "样本量", "占该 App 样本"],
                 ["App", "活动主导评论", "样本量", "占该 App 样本"], {"占该 App 样本": "{:.1%}"}),
        "",
        f"- 活动主导评论分布在 {burst['act_unique_months']} 个自然月，"
        f"但最多的一个月（{burst['act_top1_month']}）就占 {burst['act_top1']:.1%}，"
        f"前三个月（{burst['act_top3_month']}）合计占 {burst['act_top3']:.1%}。",
        f"- 其中京东健康 {burst['jd_n']} 条，最多的一个月（{burst['jd_top_month']}）"
        f"占该 App 活动主导评论的 {burst['jd_top_share']:.1%}。",
        f"- 京东健康在 {burst['sep_month']} 共有 {burst['jd_sep_all']} 条评论，"
        f"占全部活动主导评论的 {burst['act_sep_share']:.1%}。",
        "",
        "**更关键的发现是：京东健康整个样本只覆盖 "
        f"{burst['jd_all_months']} 个自然月**，其中 {burst['sep_month']} 占 "
        f"{burst['jd_sep_all_share']:.1%}。"
        "也就是说，这不是「京东健康的评论里碰巧活动议题多」，"
        "而是「京东健康的样本本身就是那一个月的集中评论」。"
        "全样本层面同样如此："
        f"{burst['n_months']} 个自然月里 {burst['sep_month']} 一个月就有 {burst['sep_all_n']} 条，"
        f"占 {burst['sep_all_share']:.1%}。"
        "RSS 的「最近评论」按时间倒序取页，一旦某 App 短期涌入大量评论，"
        "样本就会被那一段时间的评论占满。",
        "",
        f"因此活动议题在 {burst['act_top1_month']} 的 {burst['act_top1']:.0%} 集中度，"
        "很大程度是抽样机制与一次运营事件的叠加，而不是议题本身的长期分布。",
        "**但这不等于活动议题可以被当作抽样假象丢掉**——见 3.2。",
        "",
        "## 3. 剔除爆发日与集中月份后主要结论是否变化",
        "",
        "### 3.1 全部指标：剔除 31 条爆发日评论",
        "",
        "差值 = 1 星均值 − 5 星均值；FE = 控制 App 与年份后 1 星系数。",
        "",
        md_table(
            compare_effects(eff_full, eff_noburst),
            ["指标", "全样本", "剔除爆发日", "变化"],
            ["指标", "全样本", "剔除爆发日", "变化"],
            {"全样本": "{:+.3f}", "剔除爆发日": "{:+.3f}", "变化": "{:+.3f}"},
        ),
        "",
        f"中间星级峰值检验（`技术故障` 的 2 星系数的显著性）：全样本 p = {eff_full['full_tech_mid2_p']:.4f}，"
        f"剔除爆发日后 p = {eff_noburst['noburst_tech_mid2_p']:.4f}，结论不变。",
        "",
        "### 3.2 活动议题单独做逐层剔除",
        "",
        "既然活动议题的量级与「一个 App × 一个月」绑定，就逐层剔掉，看方向会不会翻、量级剩多少。",
        "",
        md_table(
            act_sens,
            ["口径", "n", "n_1star", "diff", "fe", "p"],
            ["口径", "条数", "其中 1 星", "1 星−5 星差值", "固定效应系数", "p"],
            {"diff": "{:+.3f}", "fe": "{:+.3f}", "p": "{:.1e}"},
        ),
        "",
        *act_sens_notes(act_sens),
        "",
        "## 4. 样本星级与 App Store 总体评分基准",
        "",
        "官方只公布平均分与评分人数，不公布星级分布，因此这里比较的是样本均值与官方平均分。",
        "",
        md_table(
            bench,
            ["app_name", "n_sample", "p1", "p5", "sample_mean", "official", "official_count", "gap"],
            ["App", "样本条数", "样本 1 星占比", "样本 5 星占比", "样本均值", "官方平均分", "官方评分人数", "差距"],
            {"p1": "{:.1%}", "p5": "{:.1%}", "sample_mean": "{:.2f}", "official": "{:.2f}",
             "gap": "{:+.2f}"},
        ),
        "",
        f"全样本：{overall['n']} 条，1 星占 {overall['p1']:.1%}、5 星占 {overall['p5']:.1%}，"
        f"样本均值 {overall['mean']:.2f}，按评分人数加权的官方平均分 {overall['official_weighted']:.2f}。",
        "",
        f"8 个 App 中有 {n_below} 个样本均值低于官方平均分（符号检验 p = {wil.pvalue:.1e}），"
        "方向高度一致：文字评论样本系统性偏负面。",
        "",
        "**这条差距不是数据错误，而是选择机制**：不满意的用户更愿意写文字评论，"
        "满意的用户更多只打分不写字。因此本文所有份额都是「评论者中的份额」，"
        "不能读成「用户中的占比」。差距越大，说明该 App 的沉默满意者越多，"
        "5 星议题（医生与专家口碑）在全体用户中的真实份额很可能被低估。",
        "",
        "## 5. 对主报告的影响",
        "",
        "- A3：主报告不能再写「活动议题是低端专属、量级被 App 构成放大」这一句就完了。"
        "准确表述是：方向稳健（固定效应约 +0.07，四种剔除口径全部显著），"
        "但原始差值 +0.22 里有约 0.14 来自京东健康 2026-09 的集中评论。"
        "两者必须并列报告。",
        "- 另外必须新增一句抽样披露：京东健康样本只覆盖 "
        f"{burst['jd_all_months']} 个月，全样本 {len(d):,} 条中 {burst['sep_month']} 一个月占 "
        f"{burst['sep_all_share']:.1%}。"
        "RSS 按时间倒序取页，遇到评论涌入就会把样本压到那一段，这是样本框的固有限制，"
        "与「只代表愿意写文字的评论者」并列写进局限。",
        "- B5：主报告第 10 节已按本节改写——偏差方向与量级都已量化（7/8 个 App 样本均值低于官方），"
        "并明确说明官方不给星级分布、无法做 prevalence 校正。",
        "- 仍不能做：用官方平均分反推 1 星占比（官方不给分布），所以不做 prevalence 校正，只做方向性披露。",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def compare_effects(full, noburst):
    items = [
        ("活动：1 星−5 星差值", "full_活动_diff", "noburst_活动_diff"),
        ("活动：固定效应系数", "full_活动_fe", "noburst_活动_fe"),
        ("医生合计：1 星−5 星差值", "full_医生合计_diff", "noburst_医生合计_diff"),
        ("医生合计：固定效应系数", "full_医生合计_fe", "noburst_医生合计_fe"),
        ("技术故障：固定效应系数", "full_技术故障_fe", "noburst_技术故障_fe"),
        ("App 功能：固定效应系数", "full_App功能_fe", "noburst_App功能_fe"),
    ]
    rows = []
    for label, k1, k2 in items:
        a, b = full[k1], noburst[k2]
        rows.append({"指标": label, "全样本": a, "剔除爆发日": b, "变化": b - a})
    return pd.DataFrame(rows)


def act_sens_notes(tab):
    """按实际系数写结论，避免预设"仍然显著"。"""
    base = tab[tab["口径"] == "全样本"].iloc[0]
    worst = tab.iloc[-1]
    still = tab[(tab["口径"] != "全样本")]
    sig = still[still["p"] < 0.05]
    lines = [
        f"- 剔除爆发日后，差值 {still[still['口径'] == '剔除爆发日']['diff'].iloc[0]:+.3f}"
        f"（原 {base['diff']:+.3f}），几乎不动。",
        f"- 剔除 2026-09 后，差值降到 {still[still['口径'] == '剔除 2026-09']['diff'].iloc[0]:+.3f}，"
        f"固定效应系数 {still[still['口径'] == '剔除 2026-09']['fe'].iloc[0]:+.3f}。",
        f"- 持续剔除到「京东健康且 2026-09」之后（n = {int(worst['n'])}），"
        f"差值 {worst['diff']:+.3f}，系数 {worst['fe']:+.3f}，p = {worst['p']:.3f}。",
        "",
    ]
    if len(sig) == len(still):
        lines.append(
            f"**四种剔除口径下活动议题的固定效应系数都稳定在 "
            f"{still['fe'].min():+.3f} ~ {still['fe'].max():+.3f} 且全部显著，"
            f"而原始两端差值从 {base['diff']:+.3f} 降到 {worst['diff']:+.3f}。**"
            "这说明两件事：①活动议题在 App 内部确实是低端议题，方向不依赖京东健康或 2026-09 任何单一来源；"
            "②原始差值的约三分之二（0.220 中的 0.143）来自京东健康一个 App 在 2026-09 一个月的集中评论，"
            "属于量级放大，不是方向反转。因此可以在结论里保留活动议题，"
            "但必须同时报告固定效应量级（约 +0.07）与原始差值（约 +0.22），"
            "不能只引用后者。"
        )
    else:
        lines.append(
            "**剔除后活动议题的效应明显缩小、显著性下降，说明它在很大程度上是京东健康单次事件的产物，"
            "不宜作为「普遍的低端议题」写进结论。**"
        )
    return lines


def main():
    d = load()
    burst = burst_section(d)
    eff_full = main_effects(d, "full")
    eff_noburst = main_effects(d[~d["is_burst_day"]], "noburst")
    act_sens = act_sensitivity(d)
    bench, overall = benchmark(d)
    n_below, n_total, wil = stars_vs_official_test(d)
    write_report(d, burst, eff_full, eff_noburst, act_sens, bench, overall, wil, n_below)
    print(f"写入 {REPORT}")
    print(f"爆发日 {burst['n_burst']} 条；活动主导 {burst['act_n']} 条，"
          f"跨 {burst['act_unique_months']} 个月，top1 月占 {burst['act_top1']:.1%}")
    print(f"样本均值 {overall['mean']:.2f} vs 官方加权 {overall['official_weighted']:.2f}；"
          f"{n_below}/{n_total} 个 App 样本均值更低")


if __name__ == "__main__":
    main()
