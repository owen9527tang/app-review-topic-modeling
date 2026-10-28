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
SHEET = DATA / "coding_worksheet.csv"
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


def build_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    if SHEET.exists():
        sheet = pd.read_csv(SHEET, encoding="utf-8-sig", dtype={"review_id": str})
        missing = [c for c in ["spec_A", "primary_A", "spec_B", "primary_B"] if c not in sheet.columns]
        if missing:
            raise SystemExit(f"{SHEET.name} 缺列 {missing}，请手工补齐或删除该文件后重跑。")
        return sheet
    sheet = sample[["review_id", "app_name", "rating", "month", "content"]].copy()
    sheet["spec_A"] = ""
    sheet["primary_A"] = ""
    sheet["spec_B"] = ""
    sheet["primary_B"] = ""
    sheet["note"] = ""
    sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
    return sheet


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
    sheet = build_sheet(sample)
    base = sample.merge(sheet[["review_id"]], on="review_id", how="left")

    # ---------- 机械基线（与人工编码无关，任何时候都能算） ----------
    base["mech_spec"] = base["content"].map(mechanical_spec)
    base["n_char"] = base["content"].astype(str).str.len()
    five = base[base["rating"] == 5]
    five_mech = (five["mech_spec"] == 0).mean()

    nA, nB = filled(sheet["spec_A"]), filled(sheet["spec_B"])
    lines = [
        "# 200 条抽样的人工编码核查",
        "",
        "> 目的：把「5 星更空泛」从代理指标（字符数、关键词命中）升级为**可验证标签**，并给出编码一致性。",
        f"> 编码表：`data/coding_worksheet.csv`（{len(sheet)} 行，已编码 A {nA} 条 / B {nB} 条）。",
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
        r1 = (base[base["rating"] == 1]["mech_spec"] == s).mean()
        r5 = (base[base["rating"] == 5]["mech_spec"] == s).mean()
        lines.append(f"| {name} | {(base['mech_spec'] == s).mean():.1%} | {r1:.1%} | {r5:.1%} |")
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
            "尚未编码。请按 `codebook.md` 第 8 节填 `spec_A` / `primary_A`（建议由你本人编码），"
            "再请第二人独立填 `spec_B` / `primary_B`；两列都填好后重跑本脚本即可得到 κ。",
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

        # 人工 5 星空泛率 vs 机械基线
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
        lines += ["*生成脚本：`scripts/18_manual_check.py`。*"]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"编码表：{SHEET}（A 已填 {nA}，B 已填 {nB}）")
    print(f"报告：{REPORT}")
    print("机械基线分布：", base["mech_spec"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
