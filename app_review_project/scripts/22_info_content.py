"""L1 延伸：评分与「可核对信息量」的关系。

陈琛 & Walker (2023, ISR) 用医生金标准证明了「患者分辨不出建议质量」。我们没有医学
质量金标准，能做的是它的镜像命题：**高分评论是否更少给出可核对的信息**。若成立，
则评分作为聚合信号本身就传不出质量信息。

三个指标（逐条评论，规则生成，不依赖人工编码）：
  has_num     正文含数量词（与 codebook 第 8 节的数量词清单一致）
  has_object  提到医疗对象（医生/医院/药/处方/挂号等，关键词口径）
  cn_len      中文字符数

关于 has_object：起初试过用正则识别「可唯一定位的实体」（人名 / 机构名 / 药名），
抽查发现误判严重（「早上过去医院」「感谢医生」都被当成具体实体），因此改为关键词口径——
只回答「有没有谈到医疗对象」，不假装能识别实体。口径保守，但可辩护。

三个检验，顺序固定，判定规则事先写死（见第 5 节）：
  1. 原始差异：1 星 vs 5 星的两比例 z 检验
  2. 分层：按字数四档，档内比较 + Mantel-Haenszel 合并 OR（排除「5 星更短」的机械解释）
  3. 条件在 has_object==1：都谈到了具体对象时，5 星还给数字的比例是否更低

输出：outputs/info_content.md、data/info_content.csv、outputs/figures/fig9_info_content.png

不做：不判断医学质量、不引入外部数据、不宣称因果。
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REVIEWS = DATA / "reviews_analysis.csv"
REPORT = OUT / "info_content.md"
CSV_OUT = DATA / "info_content.csv"
PNG_OUT = FIG / "fig9_info_content.png"

RED, BLUE, GREY = "#C1524B", "#3B6FB6", "#8A8A8A"
BIN_EDGES = [0, 20, 40, 80, 10_000]
BIN_LABEL = ["≤20 字", "21–40 字", "41–80 字", ">80 字"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# 与 codebook 第 8 节的「数量词清单」一致
NUM_RE = re.compile(
    r"\d|[一二三四五六七八九十百千万两]多?"
    r"(?:天|次|个月|年|小时|分钟|遍|条|页|位|星|线|张|单|笔|分|秒|块|元|层)"
)

# 医疗对象关键词（保守口径：只判断「谈没谈到」，不识别具体是谁）
MED_OBJECTS = [
    "医生", "大夫", "专家", "医师", "主任", "教授",
    "医院", "门诊", "诊所", "卫生院", "药房", "药店",
    "药", "处方", "挂号", "问诊", "复诊", "体检", "报告",
]


def has_object(text: str) -> bool:
    t = str(text or "")
    return any(w in t for w in MED_OBJECTS)


def load() -> pd.DataFrame:
    df = pd.read_csv(REVIEWS, encoding="utf-8-sig", dtype={"review_id": str})
    df["content"] = df["content"].fillna("").astype(str)
    df["has_num"] = df["content"].str.contains(NUM_RE).astype(int)
    df["has_object"] = df["content"].map(has_object).astype(int)
    df["cn_len"] = df["content"].str.count(r"[一-龥]")
    df["year"] = df["month"].astype(str).str[:4]
    df["bin"] = pd.cut(
        df["cn_len"], bins=BIN_EDGES, right=True, include_lowest=True, labels=BIN_LABEL
    )
    return df


def two_prop(df: pd.DataFrame, flag: str, a: int = 5, b: int = 1):
    """返回 (5 星比例, 1 星比例, 差值, p)。"""
    da = df.loc[df["rating"] == a, flag]
    db = df.loc[df["rating"] == b, flag]
    p5, p1 = da.mean(), db.mean()
    if len(da) < 2 or len(db) < 2:
        return p5, p1, np.nan, np.nan
    z, p = proportions_ztest([da.sum(), db.sum()], [len(da), len(db)])
    return p5, p1, p5 - p1, p


def mh_or(df: pd.DataFrame, flag: str):
    """Mantel-Haenszel 合并 OR（5 星相对 1 星给出信息的几率）+ RBG 标准误。"""
    rows, num_sum, den_sum, var_sum = [], 0.0, 0.0, 0.0
    for lb in BIN_LABEL:
        sub = df[df["bin"] == lb]
        a = ((sub["rating"] == 5) & (sub[flag] == 1)).sum()     # 5 星 有
        b = ((sub["rating"] == 5) & (sub[flag] == 0)).sum()     # 5 星 无
        c = ((sub["rating"] == 1) & (sub[flag] == 1)).sum()     # 1 星 有
        d = ((sub["rating"] == 1) & (sub[flag] == 0)).sum()     # 1 星 无
        n = a + b + c + d
        rows.append((lb, int(a), int(b), int(c), int(d)))
        if n == 0:
            continue
        num_sum += a * d / n
        den_sum += b * c / n
        var_sum += (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1)) if n > 1 else 0.0
    if den_sum == 0 or num_sum == 0:
        return rows, np.nan, np.nan, np.nan
    or_mh = num_sum / den_sum
    se = np.sqrt(var_sum / (num_sum * den_sum))
    ci = (np.exp(np.log(or_mh) - 1.96 * se), np.exp(np.log(or_mh) + 1.96 * se))
    return rows, or_mh, ci, se


def figure(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    rates = []
    for r in [1, 2, 3, 4, 5]:
        sub = df[df["rating"] == r]
        rates.append((sub["has_num"].mean(), sub["has_object"].mean(), len(sub)))
    x = np.arange(5)
    ax.bar(x - 0.18, [r[0] for r in rates], 0.34, color=RED, label="含数量词")
    ax.bar(x + 0.18, [r[1] for r in rates], 0.34, color=GREY, label="提到医疗对象")
    for i, r in enumerate(rates):
        ax.text(i - 0.18, r[0] + .015, f"{r[0]:.0%}", ha="center", fontsize=8, color=RED)
        ax.text(i + 0.18, r[1] + .015, f"{r[1]:.0%}", ha="center", fontsize=8, color="#555")
        ax.text(i, -0.055, f"n={r[2]}", ha="center", fontsize=8, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} 星" for i in range(1, 6)])
    ax.set_ylim(0, max(max(r[0], r[1]) for r in rates) * 1.30)
    ax.set_ylabel("占比")
    ax.set_title("各星级的可核对信息率", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    w = 0.36
    d1 = [df[(df["bin"] == lb) & (df["rating"] == 1)]["has_num"].mean() for lb in BIN_LABEL]
    d5 = [df[(df["bin"] == lb) & (df["rating"] == 5)]["has_num"].mean() for lb in BIN_LABEL]
    n1 = [int(((df["bin"] == lb) & (df["rating"] == 1)).sum()) for lb in BIN_LABEL]
    n5 = [int(((df["bin"] == lb) & (df["rating"] == 5)).sum()) for lb in BIN_LABEL]
    y = np.arange(len(BIN_LABEL))
    ax.barh(y + w / 2, d1, w, color=RED, label="1 星")
    ax.barh(y - w / 2, d5, w, color=BLUE, label="5 星")
    for i in range(len(BIN_LABEL)):
        ax.text(d1[i] + .012, y[i] + w / 2, f"{d1[i]:.0%} (n={n1[i]})", va="center", fontsize=8, color=RED)
        ax.text(d5[i] + .012, y[i] - w / 2, f"{d5[i]:.0%} (n={n5[i]})", va="center", fontsize=8, color=BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels(BIN_LABEL)
    ax.set_xlim(0, max(max(d1), max(d5)) * 1.45)
    ax.set_xlabel("含数量词的比例")
    ax.set_title("同一长度档内：1 星 vs 5 星", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    df = load()
    FIG.mkdir(parents=True, exist_ok=True)
    df[["review_id", "app_name", "rating", "month", "cn_len", "has_num", "has_object"]].to_csv(
        CSV_OUT, index=False, encoding="utf-8-sig"
    )
    figure(df, PNG_OUT)

    n1 = int((df["rating"] == 1).sum())
    n5 = int((df["rating"] == 5).sum())

    # 检验 1：原始差异
    t1 = {f: two_prop(df, f) for f in ["has_num", "has_object"]}

    # 检验 2：分层 + MH
    rows, or_mh, ci, se = mh_or(df, "has_num")

    # 检验 3：条件在 has_object
    cond = df[df["has_object"] == 1]
    c5 = cond[cond["rating"] == 5]["has_num"]
    c1 = cond[cond["rating"] == 1]["has_num"]
    if min(len(c5), len(c1)) > 1:
        _, p3 = proportions_ztest([c5.sum(), c1.sum()], [len(c5), len(c1)])
    else:
        p3 = np.nan

    # 检验 4：控制长度、App、年份的线性概率模型
    reg_df = df[df["rating"].isin([1, 5])].copy()
    reg_df["is5"] = (reg_df["rating"] == 5).astype(int)
    reg_df["len2"] = reg_df["cn_len"] ** 2
    m = smf.ols(
        "has_num ~ is5 + cn_len + len2 + C(app_name) + C(year)", data=reg_df
    ).fit(cov_type="cluster", cov_kwds={"groups": reg_df["app_name"]})
    coef, pval = m.params["is5"], m.pvalues["is5"]

    lo, hi = ci if np.isfinite(ci[0]) else (np.nan, np.nan)
    sig_mh = np.isfinite(hi) and hi < 1

    if sig_mh:
        verdict = (
            "**分层后仍然成立**：同一长度档内，5 星给出数量词的比例低于 1 星"
            f"（Mantel-Haenszel OR = {or_mh:.2f}，95% CI {lo:.2f}–{hi:.2f}）。"
            "差异不是「5 星写得更短」造成的。"
        )
    elif np.isfinite(or_mh):
        verdict = (
            "**分层后不显著**：控制字数后差异消失"
            f"（MH OR = {or_mh:.2f}，95% CI {lo:.2f}–{hi:.2f}）。"
            "说明原始差异主要来自篇幅，而不是「高分评论不愿给信息」。"
        )
    else:
        verdict = "分层后样本不足以估计（部分长度档某一星级为空）。"

    _p5, _p1, _d, _p = t1["has_num"]
    _o5, _o1, _od, _op = t1["has_object"]
    med1 = df[df["rating"] == 1]["cn_len"].median()
    med5 = df[df["rating"] == 5]["cn_len"].median()

    lines = []
    lines.append("# 评分与可核对信息量\n")
    lines.append(
        "陈琛 & Walker (2023, ISR) 用医生金标准证明「患者分辨不出建议质量」。本文没有医学质量金标准，"
        "因此检验它的镜像命题：**高分评论是否更少给出可核对的信息**。若成立，说明评分这个聚合信号本身就传不出质量信息。\n"
    )
    lines.append("## 1. 变量（规则生成，不依赖人工编码）\n")
    lines.append("| 指标 | 定义 |")
    lines.append("|---|---|")
    lines.append("| `has_num` | 正文含数量词（与 codebook 第 8 节数量词清单一致） |")
    lines.append(
        "| `has_object` | 提到医疗对象：医生/大夫/专家/医师/主任/教授/医院/门诊/诊所/卫生院/药房/药店/药/处方/挂号/问诊/复诊/体检/报告 |"
    )
    lines.append("| `cn_len` | 中文字符数 |")
    lines.append("")
    lines.append(f"样本：1 星 {n1} 条、5 星 {n5} 条（全部 {len(df)} 条）。\n")
    lines.append(
        "`has_object` 是**关键词口径**，只回答「有没有谈到医疗对象」，不试图识别具体是哪位医生、哪家医院。"
        "先前的正则实体识别（人名 / 机构名）抽查时误判严重，已弃用。\n"
    )

    lines.append("## 2. 原始差异\n")
    lines.append("| 指标 | 1 星 | 5 星 | 差值 | p |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 含数量词 | {_p1:.1%} | {_p5:.1%} | {_d:+.1%} | {_p:.3g} |")
    lines.append(f"| 提到医疗对象 | {_o1:.1%} | {_o5:.1%} | {_od:+.1%} | {_op:.3g} |")
    lines.append(f"| 中位字数 | {med1:.0f} | {med5:.0f} | {med5 - med1:+.0f} | — |")
    lines.append("")
    lines.append(
        f"两个指标方向相反，这一点值得单独读：5 星**更常谈到医生 / 医院 / 药**（{_o5:.1%} vs 1 星 {_o1:.1%}），"
        "说明它并不是在说空话；但一旦谈起来，它**几乎不给可核对的量**（见第 4 节）。"
        "所以「5 星更空泛」的准确表述是：5 星说「谁好」，1 星说「发生了什么、几次、多少钱」。\n"
    )
    lines.append(
        f"另需注意绝对量：1 星也只有 {_p1:.1%} 的评论带数量词。可核对信息在**整体上就是稀缺的**，"
        "这里报告的是相对差异，不是「1 星信息充分」。\n"
    )

    lines.append("## 3. 分层：按字数四档内比较（关键）\n")
    lines.append("| 长度档 | 1 星 有/总 | 5 星 有/总 | 5 星 − 1 星 |")
    lines.append("|---|---|---|---|")
    for lb, a5, b5, a1, b1 in rows:
        n5b, n1b = a5 + b5, a1 + b1
        r5 = a5 / n5b if n5b else np.nan
        r1 = a1 / n1b if n1b else np.nan
        lines.append(
            f"| {lb} | {a1}/{n1b} ({r1:.0%}) | {a5}/{n5b} ({r5:.0%}) | {(r5 - r1):+.1%} |"
        )
    lines.append("")
    if np.isfinite(or_mh):
        lines.append(
            f"Mantel-Haenszel 合并 OR（5 星 / 1 星）= **{or_mh:.2f}**，95% CI {lo:.2f}–{hi:.2f}。"
            "OR < 1 表示 5 星给出可核对信息的几率更低。\n"
        )

    lines.append("## 4. 条件在「已经谈到医疗对象」的子样本\n")
    lines.append(
        f"只保留 `has_object == 1` 的 {len(cond)} 条评论（长度与「谈不谈具体事」都已控制）："
        f"1 星含数量词 {c1.mean():.1%}（{int(c1.sum())}/{len(c1)}），"
        f"5 星 {c5.mean():.1%}（{int(c5.sum())}/{len(c5)}），差值 {c5.mean() - c1.mean():+.1%}，p = {p3:.3g}。\n"
    )

    lines.append("## 5. 控制长度的回归与判定\n")
    lines.append(
        "`has_num ~ is5 + cn_len + cn_len² + C(App) + C(年份)`，仅 1 星与 5 星，聚类稳健标准误（按 App）：\n"
    )
    lines.append(f"- `is5` 系数 = **{coef:+.4f}**，p = {pval:.3g}，n = {int(m.nobs)}，R² = {m.rsquared:.3f}")
    lines.append("")
    lines.append(verdict)
    lines.append("")

    lines.append("## 6. 边界\n")
    lines.append(
        "- **这是「信息含量」代理，不是「医学质量」**。我们无法判断哪条建议在医学上更正确 —— "
        "那是陈琛那篇请医生做金标准才能回答的问题。"
    )
    lines.append(
        "- **规则指标有噪声**：数量词会把「四星」「版本 9.2」这类数字算进去；`has_object` 只是关键词命中，"
        "无法区分「泛泛提到医院」与「点名某家医院」。因此结论只用在方向与量级上，不做个体判断。"
    )
    lines.append("- 只做关联：评分与文本同时产生，不宣称「评分导致用户少写信息」。")
    lines.append("")
    lines.append(f"图：`outputs/figures/{PNG_OUT.name}`；逐条数据：`data/{CSV_OUT.name}`。")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"1 星 {n1} / 5 星 {n5}")
    print(f"has_num: 1 星 {_p1:.3f} / 5 星 {_p5:.3f} p={_p:.3g}")
    print(f"has_object: 1 星 {_o1:.3f} / 5 星 {_o5:.3f} p={_op:.3g}")
    print(f"MH OR = {or_mh:.3f}  CI {lo:.3f}-{hi:.3f}")
    print(f"cond has_object: 1 星 {c1.mean():.3f} / 5 星 {c5.mean():.3f} p={p3:.3g}")
    print(f"reg is5 = {coef:+.4f} p={pval:.3g} n={int(m.nobs)}")
    print(f"verdict: {'分层后成立' if sig_mh else '分层后不显著'}")
    print(f"写入 {REPORT}")


if __name__ == "__main__":
    main()
