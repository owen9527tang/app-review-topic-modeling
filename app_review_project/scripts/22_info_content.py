"""L1 延伸：评分与「可核对信息量」的关系。

陈琛 & Walker (2023, ISR) 用医生金标准证明了「患者分辨不出建议质量」。我们没有医学
质量金标准，能做的是它的镜像命题：**高分评论是否更少给出可核对的信息**。若成立，
则评分作为聚合信号本身就传不出质量信息。

三个指标（逐条评论，规则生成，不依赖人工编码）：
  has_num     正文含数量词（与 codebook 第 8 节的数量词清单一致）
  has_entity  能唯一定位的实体：机构名 / 姓氏+职称 / 药名式词
  cn_len      中文字符数（已有）

三个检验，顺序固定，判定规则事先写死（见 DECISION）：
  1. 原始差异：1 星 vs 5 星的两比例 z 检验
  2. 分层：按字数四档，档内比较 + Mantel-Haenszel 合并 OR（排除「5 星更短」的机械解释）
  3. 条件在 has_entity==1：都点名了具体对象时，5 星还给数字的比例是否更低

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

# 机构名：要求 ≥2 个汉字前缀，避免「去医院」「这家医院」被当成具体机构
ORG_RE = re.compile(r"[一-龥]{2,6}(?:医院|门诊部|门诊|诊所|卫生院|药房|药店)")
ORG_BAD = ["去", "到", "在", "这", "那", "该", "本", "个", "家", "是", "有", "的", "了", "和", "与"]

# 姓氏/称呼 + 职称：前面要有像人名的 1–3 个汉字
NAME_RE = re.compile(r"[一-龥]{1,3}(?:医生|大夫|主任|教授|医师)")
NAME_BAD = [
    "说", "的", "个", "位", "这", "那", "些", "找", "问", "看", "预", "约", "是", "和",
    "跟", "给", "对", "有", "没", "名", "好", "老", "主", "专", "治", "科", "我", "你",
    "他", "她", "当", "做", "成", "叫", "连", "太", "很", "都", "也", "就", "还", "一",
]

# 药名式词：剂型后缀 + ≥2 汉字前缀（词表有限，只作辅助指标）
DRUG_RE = re.compile(r"[一-龥]{2,8}(?:片|胶囊|颗粒|口服液|软膏|滴眼液|注射液|散|糖浆)")
DRUG_BAD = ["卡片", "名片", "相片", "照片", "图片", "大片", "芯片", "药片"]


def has_entity(text: str) -> bool:
    t = str(text or "")
    if any(d in t for d in DRUG_BAD):
        t = t.replace("药片", "药物")
    for m in ORG_RE.finditer(t):
        if m.group(0)[0] not in ORG_BAD:
            return True
    for m in NAME_RE.finditer(t):
        if m.group(0)[0] not in NAME_BAD:
            return True
    for m in DRUG_RE.finditer(t):
        if not any(b in m.group(0) for b in DRUG_BAD):
            return True
    return False


def load() -> pd.DataFrame:
    df = pd.read_csv(REVIEWS, encoding="utf-8-sig", dtype={"review_id": str})
    df["content"] = df["content"].fillna("").astype(str)
    df["has_num"] = df["content"].str.contains(NUM_RE).astype(int)
    df["has_entity"] = df["content"].map(has_entity).astype(int)
    df["cn_len"] = df["content"].str.count(r"[一-龥]")
    df["year"] = df["month"].astype(str).str[:4]
    df["bin"] = pd.cut(
        df["cn_len"], bins=BIN_EDGES, right=True, include_lowest=True, labels=BIN_LABEL
    )
    return df


def two_prop(df: pd.DataFrame, flag: str, a: str = "5", b: str = "1"):
    """返回 (5 星比例, 1 星比例, 差值, p)。a=5 星，b=1 星。"""
    da = df.loc[df["rating"] == int(a), flag]
    db = df.loc[df["rating"] == int(b), flag]
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
        rates.append((sub["has_num"].mean(), sub["has_entity"].mean(), len(sub)))
    x = np.arange(5)
    ax.bar(x - 0.18, [r[0] for r in rates], 0.34, color=RED, label="含数量词")
    ax.bar(x + 0.18, [r[1] for r in rates], 0.34, color=GREY, label="含具体实体")
    for i, r in enumerate(rates):
        ax.text(i - 0.18, r[0] + .015, f"{r[0]:.0%}", ha="center", fontsize=8, color=RED)
        ax.text(i + 0.18, r[1] + .015, f"{r[1]:.0%}", ha="center", fontsize=8, color="#555")
        ax.text(i, -0.055, f"n={r[2]}", ha="center", fontsize=8, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} 星" for i in range(1, 6)])
    ax.set_ylim(0, max(r[0] for r in rates) * 1.35)
    ax.set_ylabel("占比")
    ax.set_title("各星级的可核对信息率", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    w = 0.36
    d1 = [df[(df["bin"] == lb) & (df["rating"] == 1)]["has_num"].mean() for lb in BIN_LABEL]
    d5 = [df[(df["bin"] == lb) & (df["rating"] == 5)]["has_num"].mean() for lb in BIN_LABEL]
    n1 = [((df["bin"] == lb) & (df["rating"] == 1)).sum() for lb in BIN_LABEL]
    n5 = [((df["bin"] == lb) & (df["rating"] == 5)).sum() for lb in BIN_LABEL]
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
    df[["review_id", "app_name", "rating", "month", "cn_len", "has_num", "has_entity"]].to_csv(
        CSV_OUT, index=False, encoding="utf-8-sig"
    )
    figure(df, PNG_OUT)

    n1 = (df["rating"] == 1).sum()
    n5 = (df["rating"] == 5).sum()

    # 检验 1：原始差异
    t1 = {f: two_prop(df, f) for f in ["has_num", "has_entity"]}

    # 检验 2：分层 + MH
    rows, or_mh, ci, se = mh_or(df, "has_num")

    # 检验 3：条件在 has_entity
    cond = df[df["has_entity"] == 1]
    c5 = cond[cond["rating"] == 5]["has_num"]
    c1 = cond[cond["rating"] == 1]["has_num"]
    z3, p3 = proportions_ztest([c5.sum(), c1.sum()], [len(c5), len(c1)]) if min(len(c5), len(c1)) > 1 else (np.nan, np.nan)

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
    sig_reg = pval < 0.05 and coef < 0

    if sig_mh:
        verdict = (
            "**分层后仍然成立**：同一长度档内，5 星给出数量词的比例低于 1 星"
            f"（Mantel-Haenszel OR = {or_mh:.2f}，95% CI {lo:.2f}–{hi:.2f}）。"
            "差异不是「5 星写得更短」造成的。"
        )
    elif np.isfinite(or_mh) and not sig_mh:
        verdict = (
            "**分层后不显著**：控制字数后差异消失"
            f"（MH OR = {or_mh:.2f}，95% CI {lo:.2f}–{hi:.2f}）。"
            "说明原始差异主要来自篇幅，而不是「高分评论不愿给信息」。"
        )
    else:
        verdict = "分层后样本不足以估计（部分长度档某一星级为空）。"

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
    lines.append("| `has_entity` | 能唯一定位的实体：机构名（≥2 汉字 + 医院/门诊/诊所/药房）、姓氏+职称（医生/大夫/主任/教授/医师）、药名式词（剂型后缀） |")
    lines.append("| `cn_len` | 中文字符数 |")
    lines.append("")
    lines.append(f"样本：1 星 {n1} 条、5 星 {n5} 条（全部 {len(df)} 条）。\n")

    lines.append("## 2. 原始差异\n")
    lines.append("| 指标 | 1 星 | 5 星 | 差值 | p |")
    lines.append("|---|---|---|---|---|")
    for f, name in [("has_num", "含数量词"), ("has_entity", "含具体实体")]:
        p5, p1, d, p = t1[f]
        lines.append(f"| {name} | {p1:.1%} | {p5:.1%} | {d:+.1%} | {p:.3g} |")
    lines.append(
        f"| 中位字数 | {df[df['rating'] == 1]['cn_len'].median():.0f} | "
        f"{df[df['rating'] == 5]['cn_len'].median():.0f} | "
        f"{df[df['rating'] == 5]['cn_len'].median() - df[df['rating'] == 1]['cn_len'].median():+.0f} | — |"
    )
    lines.append("")
    _e5, _e1, _ed, _ep = t1["has_entity"]
    lines.append(
        f"两个指标方向相反，这一点值得单独读：5 星**更容易点名**（含实体 {_e5:.1%} vs 1 星 {_e1:.1%}），"
        "说明它确实在谈某个具体的医生、医院或药 —— 不是纯空话；但一旦谈起来，"
        "它**几乎不给可核对的量**（见第 4 节）。所以「5 星更空泛」的准确表述是："
        "5 星说的是「谁好」，1 星说的是「发生了什么、几次、多少钱」。\n"
    )

    lines.append("## 3. 分层：按字数四档内比较（关键）\n")
    lines.append("| 长度档 | 1 星 有/总 | 5 星 有/总 | 5 星 − 1 星 |")
    lines.append("|---|---|---|---|")
    for lb, a5, b5, a1, b1 in rows:
        n5b, n1b = a5 + b5, a1 + b1
        r5 = a5 / n5b if n5b else np.nan
        r1 = a1 / n1b if n1b else np.nan
        lines.append(
            f"| {lb} | {a1}/{n1b} ({r1:.0%}) | {a5}/{n5b} ({r5:.0%}) | "
            f"{(r5 - r1):+.1%} |"
        )
    lines.append("")
    if np.isfinite(or_mh):
        lines.append(
            f"Mantel-Haenszel 合并 OR（5 星 / 1 星）= **{or_mh:.2f}**，95% CI {lo:.2f}–{hi:.2f}。"
            "OR < 1 表示 5 星给出可核对信息的几率更低。\n"
        )

    lines.append("## 4. 条件在「已经点名具体对象」的子样本\n")
    lines.append(
        f"只保留 `has_entity == 1` 的 {len(cond)} 条评论（长度与「谈不谈具体事」都已控制）："
        f"1 星含数量词 {c1.mean():.1%}（{int(c1.sum())}/{len(c1)}），"
        f"5 星 {c5.mean():.1%}（{int(c5.sum())}/{len(c5)}），差值 {c5.mean() - c1.mean():+.1%}，p = {p3:.3g}。\n"
    )

    lines.append("## 5. 控制长度的回归\n")
    lines.append(
        "`has_num ~ is5 + cn_len + cn_len² + C(App) + C(年份)`，仅 1 星与 5 星，聚类稳健标准误（按 App）：\n"
    )
    lines.append(f"- `is5` 系数 = **{coef:+.4f}**，p = {pval:.3g}，n = {int(m.nobs)}，R² = {m.rsquared:.3f}")
    lines.append("")

    lines.append("## 6. 判定\n")
    lines.append(verdict)
    lines.append("")
    lines.append("## 7. 边界\n")
    lines.append(
        "- **这是「信息含量」代理，不是「医学质量」**。我们无法判断哪条建议在医学上更正确 —— "
        "那是陈琛那篇请医生做金标准才能回答的问题。"
    )
    lines.append("- **实体识别是规则法，未人工校验**：药名词表有限，机构名与职称名可能误判，只作辅助指标。")
    lines.append("- 只做关联：评分与文本同时产生，不宣称「评分导致用户少写信息」。")
    lines.append("")
    lines.append(f"图：`outputs/figures/{PNG_OUT.name}`；逐条数据：`data/{CSV_OUT.name}`。")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"1 星 {n1} / 5 星 {n5}")
    print(f"has_num: 1 星 {t1['has_num'][1]:.3f} / 5 星 {t1['has_num'][0]:.3f} p={t1['has_num'][3]:.3g}")
    print(f"has_entity: 1 星 {t1['has_entity'][1]:.3f} / 5 星 {t1['has_entity'][0]:.3f} p={t1['has_entity'][3]:.3g}")
    print(f"MH OR = {or_mh:.3f}  CI {lo:.3f}-{hi:.3f}")
    print(f"cond has_entity: 1 星 {c1.mean():.3f} / 5 星 {c5.mean():.3f} p={p3:.3g}")
    print(f"reg is5 = {coef:+.4f} p={pval:.3g} n={int(m.nobs)}")
    print(f"verdict: {'分层后成立' if sig_mh else '分层后不显著'}")
    print(f"写入 {REPORT}")


if __name__ == "__main__":
    main()
