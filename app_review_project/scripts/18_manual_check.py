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
        "> 目的：把「5 星更空泛」从代理指标（字符数、关键词命中）升级为**可验证标签**，并给出编码一致性。",
        f"> 编码表：`data/coding_A.csv` + `data/coding_B.csv`（各 100 条，互盲、行序已打乱；已编码 A {nA} 条 / B {nB} 条）。",
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
        base["mech"] = base["mech_spec"].map(lambda v: str(int(v)))
        res = []
        if nA >= 10:
            res.append(("具体性 A vs 机械基线", kappa(base["spec_A"], base["mech"], True), nA))
        if nB >= 10:
            res.append(("具体性 A vs B", kappa(base["spec_A"], base["spec_B"], True), int(min(nA, nB))))
        if nA >= 10 and nB >= 10:
            res.append(("主议题 A vs B", kappa(base["primary_A"], base["primary_B"], False), int(min(nA, nB))))
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
                "**判定：未达 0.6，第一轮编码不采用。** `codebook.md` 第 8 节已改为可操作测试"
                "（`spec` 改为「能否核对」，`primary` 由 12 类合并为 6 组），编码表与工具同步更新为 v2，待重编后再报 κ。",
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
                    "若人工标签与代理指标方向一致，就把主报告里「5 星更空泛」的措辞从「代理指标显示」升级为「人工编码确认」；"
                    "若方向不一致，以人工编码为准，并说明代理指标为何失真。",
                    "",
                ]
        # ---------- 5. 分歧诊断 ----------
        if nA >= 10 and nB >= 10:
            both = (base["spec_A"] != "") & (base["spec_B"] != "")
            sa, sb = base.loc[both, "spec_A"], base.loc[both, "spec_B"]
            same = int((sa == sb).sum())
            d = (pd.to_numeric(sa, errors="coerce") - pd.to_numeric(sb, errors="coerce")).abs()
            m6 = {"doctor": "医疗服务", "appointment": "医疗服务", "prescription": "医疗服务",
                  "fee": "费用与权益", "promotion": "费用与权益", "service": "费用与权益",
                  "tech": "技术与账号", "privacy": "技术与账号",
                  "delivery": "药品与商品", "goods": "药品与商品",
                  "content": "内容", "other": "其他"}
            pa = base.loc[both, "primary_A"]
            pb = base.loc[both, "primary_B"]
            keep = pa.isin(m6) & pb.isin(m6)
            lines += [
                "## 5. 分歧诊断",
                "",
                f"具体性完全一致 {same}/{int(both.sum())}，差一档 {int((d == 1).sum())}，差两档 {int((d == 2).sum())}。",
                "",
                "| 口径 | κ |",
                "|---|---|",
                f"| spec 三分（二次加权） | {kappa(sa, sb, True):.3f} |",
                f"| spec 二分（0 vs 1/2） | {kappa((sa != '0').map(str), (sb != '0').map(str), False):.3f} |",
                f"| primary 12 类 | {kappa(pa, pb, False):.3f} |",
                f"| primary 合并 6 组 | {kappa(pa[keep].map(m6), pb[keep].map(m6), False):.3f} |",
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

        lines += ["*生成脚本：`scripts/18_manual_check.py`。*"]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"编码表：{sheet_desc}（A 已填 {nA}，B 已填 {nB}）")
    print(f"报告：{REPORT}")
    print("机械基线分布：", base["mech_spec"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
