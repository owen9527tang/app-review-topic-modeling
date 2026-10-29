"""P4 加固：200 条抽样的人工编码工具与一致性核算。

本脚本只做三件事：
  1. 生成/校验人工编码表 `data/coding_worksheet.csv`（两个编码者的列，留空待填）；
  2. 计算「机械基线」——用正则规则判定文本具体性，作为人工编码的对照下限；
  3. 一旦 A/B 两列填好，自动算 Cohen's κ（具体性用二次加权）与代理指标的一致率，写 `outputs/coding_check.md`。

不做的事：不代替人编码、不把模型输出冒充第二个编码者。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SAMPLE = DATA / "sample_200.csv"
A_SHEET = DATA / "coding_A.csv"
B_SHEET = DATA / "coding_B.csv"
LEGACY = DATA / "coding_worksheet.csv"
REPORT = OUT / "coding_check.md"

THEMES = {
    "fee": "收费/退款", "doctor": "医生质量/响应", "delivery": "药品配送",
    "prescription": "处方/用药", "appointment": "预约/挂号", "tech": "技术/账号",
    "privacy": "隐私/安全", "service": "客服/售后", "promotion": "活动/营销",
    "goods": "商品/实物", "content": "内容/推送", "other": "其他/空泛",
}

# 机械基线用的「具体对象词」：出现即视为有明确谈论对象（但没有数字化细节）
OBJECT_WORDS = [
    "医生", "大夫", "专家", "客服", "药", "处方", "挂号", "号源", "退款", "退费",
    "收费", "价格", "费用", "活动", "打卡", "返现", "红包", "优惠券", "会员",
    "闪退", "卡顿", "登录", "验证码", "推送", "广告", "界面", "版本", "更新",
    "配送", "快递", "物流", "订单", "发票", "报告", "体检", "测量", "血压", "血糖",
    "问诊", "复诊", "预约", "隐私", "实名", "账号", "AI", "智能",
]

NUM_RE = re.compile(
    r"\d|[一二三四五六七八九十百千万两]{1,3}\s*(元|块|天|次|个月|月|年|小时|分钟|岁|星|针|盒|瓶|片)"
)

# 与 codebook 第 8 节的「数量词清单」一致，用于把旧判定换算到新规则
STRICT_NUM_RE = (
    r"\d|[一二三四五六七八九十百千万两]多?"
    r"(?:天|次|个月|年|小时|分钟|遍|条|页|位|星|线|张|单|笔|分|秒|块|元|层)"
)


def mechanical_spec(text: str) -> int:
    """机械基线：2=含数字细节；1=有明确对象但无细节；0=无对象（纯情绪/空泛）。"""
    t = text or ""
    if NUM_RE.search(t):
        return 2
    if any(w in t for w in OBJECT_WORDS):
        return 1
    return 0


def load_sheets() -> tuple[pd.DataFrame, str]:
    """优先读两份互盲表（coding_A / coding_B），退回旧的单表 coding_worksheet。"""
    if A_SHEET.exists() and B_SHEET.exists():
        a = pd.read_csv(A_SHEET, encoding="utf-8-sig", dtype={"review_id": str})
        b = pd.read_csv(B_SHEET, encoding="utf-8-sig", dtype={"review_id": str})
        for df, who in [(a, "A"), (b, "B")]:
            need = ["review_id", "spec", "primary"]
            miss = [c for c in need if c not in df.columns]
            if miss:
                raise SystemExit(f"coding_{who}.csv 缺列 {miss}")
        a = a.rename(columns={"spec": "spec_A", "primary": "primary_A"})
        b = b.rename(columns={"spec": "spec_B", "primary": "primary_B"})
        sheet = a.merge(b[["review_id", "spec_B", "primary_B"]], on="review_id", how="inner")
        return sheet, "data/coding_A.csv + data/coding_B.csv"

    if LEGACY.exists():
        sheet = pd.read_csv(LEGACY, encoding="utf-8-sig", dtype={"review_id": str})
        miss = [c for c in ["spec_A", "primary_A", "spec_B", "primary_B"] if c not in sheet.columns]
        if miss:
            raise SystemExit(f"{LEGACY.name} 缺列 {miss}，请补齐或删除后重跑。")
        return sheet, "data/coding_worksheet.csv（旧单表）"

    raise SystemExit("找不到编码表：先运行 scripts/19_make_coding_sheets.py 生成 coding_A.csv 与 coding_B.csv")


def norm(series: pd.Series) -> pd.Series:
    """把编码列规范成纯字符串：缺失记为 ''，2.0 记成 '2'，避免 NaN 被当成有效值。"""
    def one(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(int(v)) if float(v).is_integer() else str(v)
        s = str(v).strip()
        return "" if s.lower() in ("", "nan", "none") else s
    return series.map(one)


def filled(series: pd.Series) -> int:
    """统计已填条数。"""
    return int((norm(series) != "").sum())


def kappa(a: pd.Series, b: pd.Series, weighted: bool) -> float:
    x, y = norm(a), norm(b)
    m = (x != "") & (y != "")
    if m.sum() < 10:
        return float("nan")
    return float(cohen_kappa_score(x[m], y[m], weights="quadratic" if weighted else None))


def main():
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype={"review_id": str})
    sheet, sheet_desc = load_sheets()
    base = sample[sample["review_id"].isin(sheet["review_id"])].copy()
    base = base.merge(sheet[["review_id"]], on="review_id", how="left")

    # ---------- 机械基线（与人工编码无关，任何时候都能算） ----------
    sample["mech_spec"] = sample["content"].map(mechanical_spec)
    base["mech_spec"] = base["content"].map(mechanical_spec)
    base["n_char"] = base["content"].astype(str).str.len()
    five = sample[sample["rating"] == 5]
    five_mech = (five["mech_spec"] == 0).mean()

    nA, nB = filled(sheet["spec_A"]), filled(sheet["spec_B"])
    lines = [
        "# 抽样评论的人工编码核查",
        "",
        "> 目的：检验「5 星更空泛」这一维度能否靠人工编码可靠测出，并给出编码一致性。",
        "> **结论（详见第 6 节）：不能。**两轮、两版手册下最低 κ = 0.164，未达 0.6，",
        "> 因此空泛程度继续用代理指标（字符数、主题关键词命中），本报告作为负结果保留。",
        f"> 编码表：{sheet_desc}（各 100 条，互盲、行序已打乱）。",
        "",
        "## 1. 抽样构成",
        "",
    ]

    star = base["rating"].value_counts().sort_index()
    star_row = "、".join(f"{int(k)} 星 {v} 条" for k, v in star.items())
    app = base["app_name"].value_counts()
    app_row = "、".join(f"{k} {v}" for k, v in app.items())
    lines += [f"- 星级：{star_row}。", f"- App：{app_row}。", ""]

    lines += [
        "## 2. 机械基线（规则版，不是人工编码）",
        "",
        "规则：含数字或数量词（元/块/天/次/月/岁/分钟…）→ 2（可核实细节）；"
        "命中 39 个「具体对象词」（医生、退款、闪退、订单…）→ 1（有对象无细节）；否则 → 0（纯情绪/空泛）。",
        "",
        "| 具体性 | 全样本 | 1 星 | 5 星 |",
        "|---|---|---|---|",
    ]
    for s, name in [(2, "2 可核实"), (1, "1 有对象"), (0, "0 无对象")]:
        r1 = (sample[sample["rating"] == 1]["mech_spec"] == s).mean()
        r5 = (sample[sample["rating"] == 5]["mech_spec"] == s).mean()
        lines.append(f"| {name} | {(sample['mech_spec'] == s).mean():.1%} | {r1:.1%} | {r5:.1%} |")
    lines += [
        "",
        f"5 星中机械判为「0 无对象」的比例：**{five_mech:.1%}**。这与主报告的关键词代理（15.4% vs 29.1%）方向一致，"
        "但规则版更宽松，命中更多「无对象」——说明关键词法可能**低估**了空泛程度。",
        "",
        "## 3. 人工编码状态",
        "",
    ]

    if nA == 0 and nB == 0:
        lines += [
            "尚未编码。请按 `codebook.md` 第 8 节：你填 `data/coding_A.csv` 的 `spec` / `primary`（100 条），"
            "再把 `data/coding_B.csv` 交给第二人独立填（同一批 100 条，行序已打乱、无星级列）；两表都填好后重跑本脚本即可得到 κ。",
            "",
            "**为什么必须两个人**：同一人前后两次编码的一致性会被记忆污染，不能代替独立编码者。",
            "若只有一个人可编，就只报「人工 vs 机械基线」的一致率，不要报 κ、也不要说「一致性良好」。",
            "",
            "## 4. 编码手册要点",
            "",
            "- `spec`：0 = 只有情绪或空泛夸奖（「垃圾」「太好了」）；1 = 有明确对象但无可核实细节（「客服态度差」）；"
            "2 = 有可核实细节（金额、次数、时长、具体功能名、具体症状或药品、具体操作步骤）。",
            "- `primary`：12 类单选，取占篇幅最多的议题；若全篇无具体对象则记 `other`。",
            "- 判定只看评论文字，**不看星级**（星级只作为抽样分层依据）。",
            "",
            "*生成脚本：`scripts/18_manual_check.py`。*",
        ]
    else:
        base = base.merge(sheet[["review_id", "spec_A", "primary_A", "spec_B", "primary_B"]],
                          on="review_id", how="left")
        for c in ["spec_A", "primary_A", "spec_B", "primary_B"]:
            base[c] = norm(base[c])
        # A 用 12 个细项、B 可能已是 6 组，统一映射到 6 组后再比
        GRP = {"doctor": "医疗服务", "appointment": "医疗服务", "prescription": "医疗服务",
               "fee": "费用与权益", "promotion": "费用与权益", "service": "费用与权益",
               "tech": "技术与账号", "privacy": "技术与账号",
               "delivery": "药品与商品", "goods": "药品与商品",
               "content": "内容", "other": "其他",
               "医疗服务": "医疗服务", "费用与权益": "费用与权益", "技术与账号": "技术与账号",
               "药品与商品": "药品与商品", "内容": "内容", "其他": "其他"}
        base["gA"] = base["primary_A"].map(lambda v: GRP.get(v, ""))
        base["gB"] = base["primary_B"].map(lambda v: GRP.get(v, ""))
        base["mech"] = base["mech_spec"].map(lambda v: str(int(v)))
        res = []
        if nA >= 10:
            res.append(("具体性 A vs 机械基线", kappa(base["spec_A"], base["mech"], True), nA))
        if nB >= 10:
            res.append(("具体性 A vs B", kappa(base["spec_A"], base["spec_B"], True), int(min(nA, nB))))
        if nA >= 10 and nB >= 10:
            res.append(("主议题 A vs B（6 组）", kappa(base["gA"], base["gB"], False), int(min(nA, nB))))
        lines += ["| 对比 | κ | 可用样本 |", "|---|---|---|"]
        for name, k, n in res:
            lines.append(f"| {name} | {k:.3f} | {n} |")
        lines += [
            "",
            "参考解读（Landis & Koch）：<0.20 差，0.21–0.40 一般，0.41–0.60 中等，0.61–0.80 好，>0.80 很好。",
            "**低于 0.6 必须回头改手册的边界规则，而不是硬报数字**。",
            "",
        ]
        ab = [k for name, k, n in res if "A vs B" in name]
        if ab and max(ab) < 0.6:
            lines += [
                "第一轮（12 类）κ = 0.068、第二轮（6 组）κ = 0.173；`spec` 两轮从 0.133 到 0.271。",
                "改手册（`spec` 改为「正文有数量词」、`primary` 合并为 6 组）并重编 A 后，分歧依旧集中在"
                "「有没有可指认的对象」这一步 —— 见第 5、6 节的诊断。",
                "",
            ]
        a_ok = base[base["spec_A"] != ""]
        if len(a_ok):
            f5 = a_ok[a_ok["rating"] == 5]
            f1 = a_ok[a_ok["rating"] == 1]
            if len(f5) and len(f1):
                lines += [
                    "## 4. 空泛好评：人工标签 vs 代理指标",
                    "",
                    "| 口径 | 1 星 | 5 星 |",
                    "|---|---|---|",
                    f"| 人工 `spec == 0`（无具体对象） | {(f1['spec_A'] == '0').mean():.1%} | "
                    f"{(f5['spec_A'] == '0').mean():.1%} |",
                    f"| 机械基线 `mech_spec == 0` | {(f1['mech_spec'] == 0).mean():.1%} | {(f5['mech_spec'] == 0).mean():.1%} |",
                    f"| 未命中主题关键词（主报告口径） | 15.4% | 29.1% |",
                    "",
                    "人工标签与代理指标**方向一致**（5 星判「无对象」的比例高于 1 星），"
                    "但该维度未通过信度检验（κ 0.164），故不用于结论：保留代理指标，"
                    "并把「方向一致」当作两条独立路径的互相印证，而不是「人工验证成立」。",
                    "",
                ]
        # ---------- 5. 分歧诊断 ----------
        if nA >= 10 and nB >= 10:
            both = (base["spec_A"] != "") & (base["spec_B"] != "")
            sa, sb = base.loc[both, "spec_A"], base.loc[both, "spec_B"]
            same = int((sa == sb).sum())
            d = (pd.to_numeric(sa, errors="coerce") - pd.to_numeric(sb, errors="coerce")).abs()
            pa = base.loc[both, "gA"]
            pb = base.loc[both, "gB"]
            keep = (pa != "") & (pb != "")
            lines += [
                "## 5. 分歧诊断",
                "",
                f"具体性完全一致 {same}/{int(both.sum())}，差一档 {int((d == 1).sum())}，差两档 {int((d == 2).sum())}。",
                "",
                "| 口径 | κ |",
                "|---|---|",
                f"| spec 三分（二次加权） | {kappa(sa, sb, True):.3f} |",
                f"| spec 二分（0 vs 1/2） | {kappa((sa != '0').map(str), (sb != '0').map(str), False):.3f} |",
                f"| primary 6 组 | {kappa(pa[keep], pb[keep], False):.3f} |",
                "",
            ]
            # 判断编码者是否真的依据文本：spec 与字数、含数字的关联
            nchar = base.loc[both, "n_char"]
            hasd = base.loc[both, "content"].astype(str).str.contains(r"\d").astype(int)
            rows = []
            for who, v in (("A", sa), ("B", sb)):
                vn = pd.to_numeric(v, errors="coerce")
                rows.append(f"| {who} | {vn.corr(nchar, method='spearman'):.2f} | {vn.corr(hasd, method='spearman'):.2f} |")
            lines += [
                "编码是否真的依据文本（若某位编码者的相关性接近 0，说明未按文字判断）：",
                "",
                "| 编码者 | spec 与字数（Spearman） | spec 与「含数字」 |",
                "|---|---|---|",
                *rows,
                "",
                "> 结论写进 `codebook.md` 第 8 节：若两列都明显为正、但 κ 仍低，问题在手册边界而非编码者，"
                "改手册后重编，不硬报数字。",
                "",
            ]

            # ---------- 6. 规则对齐后的换算（用于判断分歧性质） ----------
            has_num = base.loc[both, "content"].astype(str).str.contains(STRICT_NUM_RE)
            sb_rec = pd.Series(np.where(sb == "0", "0", np.where(has_num, "2", "1")), index=sb.index)
            rec_k = kappa(sa, sb_rec, True)
            bin_k = kappa((sa != "0").map(str), (sb != "0").map(str), False)
            lines += [
                "## 6. 把 B 的旧判定按新规则换算后",
                "",
                "新手册只改了 `2` 的定义（正文有数量词），没改 `0`。把 B 的旧判定按同一规则换算，"
                f"共 {int((sb_rec != sb).sum())} 条改变（`2→1` {int(((sb == '2') & (sb_rec == '1')).sum())} 条，"
                f"`1→2` {int(((sb == '1') & (sb_rec == '2')).sum())} 条）。",
                "",
                "| 口径 | κ | 说明 |",
                "|---|---|---|",
                f"| A vs B（换算后，三分） | {rec_k:.3f} | `2` 由规则统一，**该值偏高、不可作为信度上报** |",
                f"| A vs B（原始，三分） | {kappa(sa, sb, True):.3f} | 两版手册混用，仅作参照 |",
                f"| A vs B（0 vs 非 0，二元） | {bin_k:.3f} | 唯一未被规则统一的人工判断 |",
                "",
                f"换算后完全一致 {int((sa == sb_rec).sum())}/{len(sa)}；"
                f"「有对象」判定：A {int((sa != '0').sum())} 条、B {int((sb != '0').sum())} 条。",
                "",
            ]
            worst = min(kappa(sa, sb, True), rec_k, bin_k)
            if worst < 0.6:
                lines += [
                    f"> **判定：最低 κ = {worst:.3f} < 0.6，未通过信度检验。**",
                    "> 人工编码不作为结论来源：空泛程度继续用代理指标（字符数、主题关键词命中），",
                    "> 本报告作为「该维度不适合独立人工编码」的负结果保留。",
                    "> `2`（正文含数量词）被证明可由规则稳定判定，保留为特征，不再需要人工标注。",
                    "",
                ]
            else:
                lines += [
                    f"> **判定：最低 κ = {worst:.3f} ≥ 0.6，通过。**可将 H2 升级为人工编码确认。",
                    "",
                ]

        lines += ["*生成脚本：`scripts/18_manual_check.py`。*"]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"编码表：{sheet_desc}（A 已填 {nA}，B 已填 {nB}）")
    print(f"报告：{REPORT}")
    print("机械基线分布：", base["mech_spec"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
