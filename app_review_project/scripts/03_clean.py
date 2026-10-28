# -*- coding: utf-8 -*-
"""P1 清洗与数据质量报告：去重、过滤、刷评检测、AI/主题关键词覆盖。"""
import csv
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

AI_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z])AI(?![A-Za-z])|人工智能|智能助手|智能医生|机器人|大模型|"
    r"DeepSeek|ChatGPT|GPT|豆包|文心一言|通义千问|讯飞|智能问诊|AI问诊|AI医生|AI助手|智能客服"
)

THEMES = {
    "fee": ("收费/退款", r"收费|退费|退款|扣费|乱收费|价格|费用|贵|会员|付费|药费|挂号费|钱"),
    "doctor": ("医生质量/响应", r"医生|大夫|回复|不理|敷衍|态度|专业|水平|响应|等待|排队|问诊|诊断|咨询"),
    "delivery": ("药品配送", r"配送|快递|发货|物流|送药|迟到|包装|订单|送货"),
    "prescription": ("处方/用药", r"处方|用药|剂量|副作用|药效|开药|药方|吃药"),
    "appointment": ("预约/挂号", r"挂号|预约|号源|候诊|门诊|复诊|排队"),
    "tech": ("技术/账号", r"登录|闪退|卡顿|版本|更新|账号|验证码|绑定|支付|系统|bug|网络|打不开|崩溃|适配"),
    "privacy": ("隐私/安全", r"隐私|信息泄露|泄露|安全|实名|授权|骚扰|电话"),
}
THEME_PATTERNS = {k: re.compile(v[1], re.I) for k, v in THEMES.items()}

BASE_FIELDS = ["app_name", "app_id", "source_page", "review_id", "author", "rating", "version", "title", "content", "updated"]
FLAG_FIELDS = [
    "date", "month", "content_len", "cn_len", "has_chinese", "is_empty", "is_short",
    "ai_mention", "theme_fee", "theme_doctor", "theme_delivery", "theme_prescription",
    "theme_appointment", "theme_tech", "theme_privacy", "any_theme",
    "dup_content", "day_count", "is_burst_day",
]
CLEAN_FIELDS = BASE_FIELDS + FLAG_FIELDS


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def normalize(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def pct(n, d):
    return round(n / d, 3) if d else 0


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw = load_csv(DATA / "reviews_raw.csv")
    metadata = {row["app_id"]: row for row in load_csv(DATA / "app_metadata.csv")}

    # 基础清洗与标记
    seen_content = set()
    for row in raw:
        content = normalize(row.get("content", ""))
        row["content"] = content
        row["title"] = normalize(row.get("title", ""))
        row["date"] = (row.get("updated", "") or "")[:10]
        row["month"] = (row.get("updated", "") or "")[:7]
        row["content_len"] = len(content)
        row["cn_len"] = len(re.findall(r"[\u4e00-\u9fff]", content))
        row["has_chinese"] = str(bool(re.search(r"[\u4e00-\u9fff]", content)))
        row["is_empty"] = str(not content)
        row["is_short"] = str(row["cn_len"] < 6)
        row["ai_mention"] = str(bool(AI_PATTERN.search(content)))
        for key, pattern in THEME_PATTERNS.items():
            row[f"theme_{key}"] = str(bool(pattern.search(content)))
        row["any_theme"] = str(any(row[f"theme_{k}"] == "True" for k in THEMES))
        key = (row["app_id"], content)
        if content and key in seen_content:
            row["dup_content"] = "True"
        else:
            row["dup_content"] = "False"
            if content:
                seen_content.add(key)

    # 同日评分爆发检测
    day_counts = Counter((row["app_id"], row["date"]) for row in raw)
    app_day_counts = defaultdict(list)
    for (app_id, _), count in day_counts.items():
        app_day_counts[app_id].append(count)
    thresholds = {}
    for app_id, counts in app_day_counts.items():
        mean = statistics.mean(counts) if counts else 0
        std = statistics.pstdev(counts) if len(counts) > 1 else 0
        thresholds[app_id] = max(20, mean + 2 * std)
    for row in raw:
        count = day_counts[(row["app_id"], row["date"])]
        row["day_count"] = count
        row["is_burst_day"] = str(count >= thresholds.get(row["app_id"], 20))

    # 写全量与分析样本
    write_csv(DATA / "reviews_clean.csv", raw, CLEAN_FIELDS)
    analysis = [
        row for row in raw
        if row["has_chinese"] == "True"
        and row["is_empty"] == "False"
        and row["is_short"] == "False"
        and row["dup_content"] == "False"
        and row["rating"].isdigit()
    ]
    write_csv(DATA / "reviews_analysis.csv", analysis, CLEAN_FIELDS)

    # 统计
    total_raw = len(raw)
    total_analysis = len(analysis)
    rating_dist = Counter(int(row["rating"]) for row in analysis if row["rating"].isdigit())
    one_star = [r for r in analysis if r["rating"] == "1"]
    five_star = [r for r in analysis if r["rating"] == "5"]
    ai_rows = [r for r in analysis if r["ai_mention"] == "True"]

    by_app = defaultdict(list)
    for row in analysis:
        by_app[(row["app_name"], row["app_id"])].append(row)

    lines = [
        "# P1 数据质量报告：互联网医疗 App 公开评论",
        "",
        f"- 原始唯一评论：**{total_raw}**",
        f"- 分析样本（中文、非空、≥6 个汉字、非重复正文、评分有效）：**{total_analysis}**",
        f"- 分析样本占比：{pct(total_analysis, total_raw):.0%}",
        "- 说明：以下统计只代表“愿意写文字评论的评论者”，不代表全体用户。",
        "",
        "## 1. 各 App 样本与总体评分基准",
        "",
        "| App | 原始 | 分析样本 | 日期范围 | 样本均分 | 1 星 | 5 星 | AI 提及 | 总体评分（人数） | 样本-总体 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, app_id in sorted(by_app.keys(), key=lambda x: x[0]):
        items = by_app[(name, app_id)]
        dates = sorted(r["date"] for r in items if r["date"])
        ratings = [int(r["rating"]) for r in items if r["rating"].isdigit()]
        avg = round(statistics.mean(ratings), 2) if ratings else 0
        meta = metadata.get(str(app_id), {})
        overall = meta.get("average_user_rating", "")
        overall_n = meta.get("user_rating_count", "")
        gap = ""
        if overall and ratings:
            try:
                gap = f"{avg - float(overall):+.2f}"
            except ValueError:
                gap = ""
        n_ai = sum(1 for r in items if r["ai_mention"] == "True")
        lines.append(
            f"| {name} | {sum(1 for r in raw if r['app_id'] == app_id)} | {len(items)} | "
            f"{dates[0] if dates else ''} 至 {dates[-1] if dates else ''} | {avg} | "
            f"{pct(sum(1 for r in items if r['rating'] == '1'), len(items)):.0%} | "
            f"{pct(sum(1 for r in items if r['rating'] == '5'), len(items)):.0%} | "
            f"{n_ai} ({pct(n_ai, len(items)):.0%}) | {overall}（{overall_n}） | {gap} |"
        )

    lines += [
        "",
        "## 2. 评分分布（分析样本）",
        "",
        "| 星级 | 条数 | 占比 |",
        "|---|---:|---:|",
    ]
    for star in range(1, 6):
        lines.append(f"| {star} 星 | {rating_dist.get(star, 0)} | {pct(rating_dist.get(star, 0), total_analysis):.0%} |")

    flag_counts = {
        "空正文": sum(1 for r in raw if r["is_empty"] == "True"),
        "少于 6 个汉字": sum(1 for r in raw if r["is_short"] == "True"),
        "无中文": sum(1 for r in raw if r["has_chinese"] == "False"),
        "重复正文（后出现）": sum(1 for r in raw if r["dup_content"] == "True"),
        "爆发日评论": sum(1 for r in raw if r["is_burst_day"] == "True"),
    }
    lines += ["", "## 3. 清洗标记", "", "| 标记 | 条数 |", "|---|---:|"]
    for name, count in flag_counts.items():
        lines.append(f"| {name} | {count} |")

    # AI 提及
    ai_1 = sum(1 for r in one_star if r["ai_mention"] == "True")
    ai_5 = sum(1 for r in five_star if r["ai_mention"] == "True")
    lines += [
        "",
        "## 4. AI 关键词提及（初步，非最终 LLM 编码）",
        "",
        f"- 分析样本中提及 AI：**{len(ai_rows)} 条（{pct(len(ai_rows), total_analysis):.1%}）**",
        f"- 1 星中提及 AI：{ai_1}/{len(one_star)}（{pct(ai_1, len(one_star)):.1%}）",
        f"- 5 星中提及 AI：{ai_5}/{len(five_star)}（{pct(ai_5, len(five_star)):.1%}）",
        "",
    ]
    if ai_rows:
        lines += ["示例（截断，仅用于检查关键词误判）：", ""]
        for row in ai_rows[:5]:
            snippet = row["content"][:80].replace("|", " ")
            lines.append(f"- [{row['rating']} 星] {snippet}")
        lines.append("")

    # 主题关键词覆盖
    theme_stats = []
    for key, (name, _) in THEMES.items():
        col = f"theme_{key}"
        n_all = sum(1 for r in analysis if r[col] == "True")
        n_1 = sum(1 for r in one_star if r[col] == "True")
        n_5 = sum(1 for r in five_star if r[col] == "True")
        theme_stats.append({
            "theme": name,
            "n": n_all,
            "share": pct(n_all, total_analysis),
            "share_1": pct(n_1, len(one_star)),
            "share_5": pct(n_5, len(five_star)),
            "gap": pct(n_1, len(one_star)) - pct(n_5, len(five_star)),
        })
    theme_stats.sort(key=lambda x: x["n"], reverse=True)

    lines += [
        "## 5. 主题关键词覆盖（初步，非最终 LLM 编码）",
        "",
        "| 主题 | 提及条数 | 全样本占比 | 1 星中占比 | 5 星中占比 | 1 星-5 星差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in theme_stats:
        lines.append(
            f"| {s['theme']} | {s['n']} | {s['share']:.0%} | {s['share_1']:.0%} | {s['share_5']:.0%} | {s['gap']:+.0%} |"
        )
    lines.append(f"| 至少命中一个主题 | {sum(1 for r in analysis if r['any_theme'] == 'True')} | "
                 f"{pct(sum(1 for r in analysis if r['any_theme'] == 'True'), total_analysis):.0%} | - | - | - |")

    # 结论
    ai_rate = pct(len(ai_rows), total_analysis)
    top_gap = sorted(theme_stats, key=lambda x: x["gap"], reverse=True)[:3]
    lines += [
        "",
        "## 6. 初步结论与下一步",
        "",
        f"- AI 关键词提及率：**{ai_rate:.1%}**。"
        + ("低于 5%，AI 作为核心角度偏薄，建议加 1–2 个 AI 属性强的平台，或把 AI 降为次要角度。"
           if ai_rate < 0.05 else "达到 5%，可以作为核心角度之一。"),
        "- 1 星更强主题（关键词差）：" + "；".join(f"{s['theme']}（{s['gap']:+.0%}）" for s in top_gap) + "。",
        "- 以上为主题关键词代理，不是最终 LLM 编码；P2 用 7 类 codebook + 2 个标记做正式标注。",
        "- 样本与总体评分差距说明文字评论存在选择偏差；报告只写“在评论者中”。",
        "",
        "## 7. 文件",
        "",
        "- `data/reviews_raw.csv`：去重后的原始评论",
        "- `data/app_metadata.csv`：App 总体评分与评分人数",
        "- `data/reviews_clean.csv`：全量 + 清洗标记",
        "- `data/reviews_analysis.csv`：分析样本",
        "- `outputs/data_quality.md`：本报告",
    ]

    report = OUT / "data_quality.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"原始：{total_raw}；分析样本：{total_analysis}")
    print(f"AI 提及率：{ai_rate:.1%}")
    print(f"报告：{report}")


if __name__ == "__main__":
    main()
