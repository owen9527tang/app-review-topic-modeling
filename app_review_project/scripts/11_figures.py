"""P4 图表：把主报告与两份稳健性报告的结论画出来。

输出到 outputs/figures/：
  fig1_theme_diff.png       10 个主题的 1 星 − 5 星 份额差
  fig2_star_profile.png     六个关键议题的五星级剖面（原始均值）
  fig3_raw_vs_fe.png        原始差值 vs 控制 App/年份后的固定效应系数
  fig4_text_specificity.png 星级 × 文本长度与「有具体对象」比例
  fig5_benchmark.png        各 App 样本均值 vs App Store 官方平均分
  fig6_activity_month.png   活动议题主导评论的月度分布
  fig7_ai_by_rating.png     AI 提及率 × 星级
  fig8_star_mix.png         各 App 的星级构成
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

K = 10
COLS = [f"t{i:02d}" for i in range(K)]
NAME = {
    0: "医生解答与沟通", 1: "医院挂号与就医流程", 2: "活动打卡与 AI 识别",
    3: "专家资源与医术口碑", 4: "付费问诊次数与退款", 5: "免费咨询与买药",
    6: "App 功能与挂号操作", 7: "客服售后与配送体检", 8: "技术故障与更新",
    9: "患者获益与异地就医",
}
RED, BLUE, GREY = "#C1524B", "#3B6FB6", "#8A8A8A"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def load():
    rev = pd.read_csv(DATA / "reviews_analysis.csv", encoding="utf-8-sig", dtype={"review_id": str})
    w = pd.read_csv(DATA / "topic_doc_weights.csv", encoding="utf-8-sig", dtype={"review_id": str})
    d = rev.merge(w[["review_id"] + COLS], on="review_id", how="inner")
    d["rating"] = d["rating"].astype(int)
    d["doctor"] = d["t00"] + d["t03"]
    d["year"] = d["date"].str[:4]
    d["ym"] = d["date"].str[:7]
    return d


def fig1_theme_diff(d):
    sub = d[d.rating.isin([1, 5])]
    rows = []
    for i in range(K):
        rows.append({"name": NAME[i],
                     "diff": sub.loc[sub.rating == 1, f"t{i:02d}"].mean()
                     - sub.loc[sub.rating == 5, f"t{i:02d}"].mean()})
    tab = pd.DataFrame(rows).sort_values("diff")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(tab["name"], tab["diff"],
            color=[RED if v > 0 else BLUE for v in tab["diff"]], height=0.62)
    ax.axvline(0, color="black", lw=0.8)
    for y, v in enumerate(tab["diff"]):
        ax.text(v + (0.006 if v > 0 else -0.006), y, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8)
    ax.set_xlabel("主题份额差（1 星 − 5 星）")
    ax.set_title("红色＝1 星更常谈，蓝色＝5 星更常谈（n=1,497）", fontsize=10)
    ax.set_xlim(tab["diff"].min() - 0.05, tab["diff"].max() + 0.05)
    fig.savefig(OUT / "fig1_theme_diff.png")
    plt.close(fig)


def fig2_star_profile(d):
    cols = [("医生与专家口碑（合计）", "doctor"), ("活动打卡与 AI 识别", "t02"),
            ("付费问诊次数与退款", "t04"), ("技术故障与更新", "t08"),
            ("App 功能与挂号操作", "t06"), ("客服售后与配送体检", "t07")]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label, c in cols:
        m = [d.loc[d.rating == r, c].mean() for r in range(1, 6)]
        ax.plot(range(1, 6), m, marker="o", lw=1.8, label=label)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"{r} 星\n(n={int((d.rating == r).sum())})" for r in range(1, 6)])
    ax.set_ylabel("平均主题份额")
    ax.set_title("五星级剖面：中段并不是两端的过渡带", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(OUT / "fig2_star_profile.png")
    plt.close(fig)


def fig3_raw_vs_fe(d):
    sub = d[d.rating.isin([1, 5])].assign(is1=lambda x: (x.rating == 1).astype(int))
    rows = []
    for i in range(K):
        c = f"t{i:02d}"
        raw = sub.loc[sub.is1 == 1, c].mean() - sub.loc[sub.is1 == 0, c].mean()
        m = smf.ols(f"{c} ~ is1 + C(app_name) + C(year)", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["app_name"]})
        rows.append({"name": NAME[i], "raw": raw, "fe": float(m.params["is1"])})
    tab = pd.DataFrame(rows).sort_values("raw")
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    lim = [-0.24, 0.26]
    ax.plot(lim, lim, color=GREY, lw=0.9, ls="--")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    for n, (_, r) in enumerate(tab.iterrows(), start=1):
        ax.scatter(r["raw"], r["fe"], s=120, color=BLUE, zorder=3)
        ax.text(r["raw"], r["fe"], str(n), color="white", fontsize=7.5,
                ha="center", va="center", zorder=4)
    ax.set_xlabel("原始差值（1 星 − 5 星）")
    ax.set_ylabel("固定效应系数（控制 App 与年份）")
    ax.set_title("原始差值与 App 内效应的落差（虚线为 1:1；点偏离越远，App 构成影响越大）", fontsize=9.5)
    legend = "\n".join(f"{n}. {r['name']}（{r['fe']:+.3f}）"
                       for n, (_, r) in enumerate(tab.iterrows(), start=1))
    ax.text(1.02, 0.5, legend, transform=ax.transAxes, fontsize=7.2, va="center")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "fig3_raw_vs_fe.png")
    plt.close(fig)


def fig4_text_specificity(d):
    """与主报告第 7 节同一口径：字符数 + 是否命中任何 codebook 主题关键词。"""
    kw = pd.read_csv(DATA / "labels_keyword.csv", encoding="utf-8-sig", dtype=str)[["review_id", "themes"]]
    kw["themes"] = kw["themes"].fillna("")
    dd = d.merge(kw, on="review_id", how="left")
    dd["themes"] = dd["themes"].fillna("")
    dd["chars"] = (dd["title"].fillna("") + " " + dd["content"].fillna(""))
    dd["chars"] = dd["chars"].str.replace(r"\s", "", regex=True).str.len()
    dd["no_theme"] = dd["themes"].str.len() == 0

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    m = dd.groupby("rating")["chars"].median()
    axes[0].bar(m.index.astype(str), m.values, color=GREY, width=0.6)
    for x, v in zip(m.index, m.values):
        axes[0].text(x - 1, v + 1.2, f"{v:.0f}", ha="center", fontsize=8)
    axes[0].set_title("字符数中位数", fontsize=9)
    axes[0].set_xlabel("星级")
    axes[0].set_ylim(0, m.max() * 1.25)

    r = dd.groupby("rating")["no_theme"].mean()
    axes[1].bar(r.index.astype(str), r.values, color=BLUE, width=0.6)
    for x, v in zip(r.index, r.values):
        axes[1].text(x - 1, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    axes[1].set_title("未命中任何主题关键词的比例", fontsize=9)
    axes[1].set_xlabel("星级")
    axes[1].set_ylim(0, max(r.max() * 1.3, 0.35))
    fig.suptitle("评分越高，评论越短；4–5 星最常无从抓取具体议题", fontsize=10)
    fig.savefig(OUT / "fig4_text_specificity.png")
    plt.close(fig)


def fig5_benchmark(d):
    meta = pd.read_csv(DATA / "app_metadata.csv", encoding="utf-8-sig")
    tab = d.groupby("app_name")["rating"].agg(["mean", "size"]).reset_index()
    tab = tab.merge(meta[["app_name", "average_user_rating"]], on="app_name", how="left")
    tab = tab.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(tab))
    ax.scatter(x, tab["average_user_rating"], s=48, color=GREY, label="App Store 官方平均分", zorder=3)
    ax.scatter(x, tab["mean"], s=52, color=RED, label="本文样本均值", zorder=3)
    for i, r in tab.iterrows():
        ax.plot([i, i], [r["average_user_rating"], r["mean"]], color=GREY, lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["app_name"], rotation=20, fontsize=8)
    ax.set_ylabel("评分")
    ax.set_ylim(0.6, 5.2)
    ax.set_title("文字评论样本系统性偏负面（7/8 个 App 低于官方）", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "fig5_benchmark.png")
    plt.close(fig)


def fig6_activity_month(d):
    act = d[d["t02"] >= 1 / 3].copy()
    act["group"] = np.where(act["app_name"] == "京东健康", "京东健康", "其余 7 个 App")
    months = sorted(d["ym"].dropna().unique())
    tab = (act.groupby(["ym", "group"]).size().unstack(fill_value=0)
           .reindex(months, fill_value=0))
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.bar(tab.index, tab.get("京东健康", 0), color=RED, label="京东健康")
    ax.bar(tab.index, tab.get("其余 7 个 App", 0), bottom=tab.get("京东健康", 0),
           color=BLUE, label="其余 7 个 App")
    ax.set_xticks(range(0, len(months), max(1, len(months) // 12)))
    ax.set_xticklabels([months[i] for i in range(0, len(months), max(1, len(months) // 12))],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("活动主导评论（条）")
    ax.set_title("活动议题集中在「京东健康 × 2026-09」一件事上", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "fig6_activity_month.png")
    plt.close(fig)


def fig7_ai_by_rating(d):
    """与主报告第 8 节同口径：labels_keyword.csv 的 ai_mention 标记。"""
    kw = pd.read_csv(DATA / "labels_keyword.csv", encoding="utf-8-sig", dtype=str)[["review_id", "ai_mention"]]
    kw["ai"] = kw["ai_mention"].astype(str).str.strip().isin(["1", "True", "true"])
    dd = d.merge(kw[["review_id", "ai"]], on="review_id", how="left")
    dd["ai"] = dd["ai"].fillna(False)
    tab = dd.groupby("rating")["ai"].agg(["mean", "sum", "size"])
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar(tab.index.astype(str), tab["mean"], color=GREY, width=0.6)
    for r in tab.index:
        ax.text(r - 1, tab.loc[r, "mean"] + 0.0015,
                f"{tab.loc[r,'mean']:.1%}\n({int(tab.loc[r,'sum'])})", ha="center", fontsize=7.5)
    ax.set_xlabel("星级")
    ax.set_ylabel("AI 关键词提及率")
    ax.set_title(f"AI 提及集中在中低星级，但总量仅 {int(tab['sum'].sum())} 条", fontsize=10)
    ax.set_ylim(0, tab["mean"].max() * 1.45)
    fig.savefig(OUT / "fig7_ai_by_rating.png")
    plt.close(fig)


def fig8_star_mix(d):
    tab = d.groupby("app_name")["rating"].value_counts().unstack(fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0)
    order = tab.sort_values(1, ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    left = np.zeros(len(order))
    for r in [1, 2, 3, 4, 5]:
        v = order[r].values if r in order.columns else np.zeros(len(order))
        ax.barh(order.index, v, left=left, label=f"{r} 星",
                color={1: "#8C2B26", 2: "#C1524B", 3: "#D9A29E", 4: "#9FB6D6", 5: "#3B6FB6"}[r])
        left += v
    ax.set_xlim(0, 1)
    ax.set_xlabel("星级构成")
    ax.set_title("各 App 的星级构成：中段占比差异明显", fontsize=10)
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.savefig(OUT / "fig8_star_mix.png")
    plt.close(fig)


def main():
    d = load()
    for fn in [fig1_theme_diff, fig2_star_profile, fig3_raw_vs_fe,
               fig4_text_specificity, fig5_benchmark, fig6_activity_month,
               fig7_ai_by_rating, fig8_star_mix]:
        fn(d)
        print(f"完成 {fn.__name__}")
    print(f"输出目录 {OUT}")


if __name__ == "__main__":
    main()
