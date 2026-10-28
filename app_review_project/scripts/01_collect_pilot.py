# -*- coding: utf-8 -*-
"""P0 试点：采集 6 个互联网医疗 App 的 App Store 公开评论各 1 页。

数据源：Apple App Store 中国区公开评论 RSS
https://itunes.apple.com/cn/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json
"""
import csv
import json
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

APPS = [
    ("平安好医生", 1552354076),
    ("京东健康", 1498018017),
    ("微医", 595277934),
    ("好大夫在线", 919502358),
    ("春雨医生", 478834039),
    ("丁香医生", 911261462),
]

BASE = "https://itunes.apple.com/cn/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

FIELDS = ["app_name", "app_id", "review_id", "author", "rating", "version", "title", "content", "updated"]


def fetch(app_id, page=1):
    """抓取一页公开评论 JSON。"""
    url = BASE.format(page=page, app_id=app_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def as_list(value):
    """RSS 在只有一条时返回 dict，这里统一成 list。"""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def label(node, *keys):
    """安全读取嵌套的 label 字段。"""
    cur = node
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return ""
        cur = cur[key]
    return cur.get("label", "") if isinstance(cur, dict) else ""


def parse_entry(app_name, app_id, entry):
    return {
        "app_name": app_name,
        "app_id": app_id,
        "review_id": label(entry, "id"),
        "author": label(entry, "author", "name"),
        "rating": label(entry, "im:rating"),
        "version": label(entry, "im:version"),
        "title": label(entry, "title"),
        "content": label(entry, "content"),
        "updated": label(entry, "updated"),
    }


def has_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    raw = {}
    rows = []
    for name, app_id in APPS:
        try:
            payload = fetch(app_id, page=1)
            entries = as_list(payload.get("feed", {}).get("entry"))
            parsed = [parse_entry(name, app_id, e) for e in entries]
            raw[f"{name}_{app_id}"] = parsed
            rows.extend(parsed)
            print(f"{name}: {len(parsed)} 条")
        except Exception as exc:  # 单个 App 失败不影响其他 App
            print(f"{name}: ERROR {exc}")
            raw[f"{name}_{app_id}"] = []

    # 按 (app_id, review_id) 去重
    seen = set()
    unique = []
    for row in rows:
        key = (row["app_id"], row["review_id"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    with (DATA / "pilot_raw.json").open("w", encoding="utf-8") as fh:
        json.dump(raw, fh, ensure_ascii=False, indent=2)

    csv_path = DATA / "pilot_reviews.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(unique)

    # 统计
    by_app = defaultdict(list)
    for row in unique:
        by_app[(row["app_name"], row["app_id"])].append(row)

    app_stats = []
    for name, app_id in APPS:
        items = by_app.get((name, app_id), [])
        dates = sorted(x["updated"] for x in items if x["updated"])
        ratings = [int(x["rating"]) for x in items if x["rating"].isdigit()]
        ratings_ok = [r for r in ratings if 1 <= r <= 5]
        chinese = sum(1 for x in items if has_chinese(x["content"]))
        empty = sum(1 for x in items if not x["content"].strip())
        avg = round(sum(ratings_ok) / len(ratings_ok), 2) if ratings_ok else 0
        dist = Counter(ratings_ok)
        app_stats.append({
            "name": name,
            "count": len(items),
            "date_min": dates[0] if dates else "",
            "date_max": dates[-1] if dates else "",
            "chinese_ratio": round(chinese / len(items), 3) if items else 0,
            "empty_content": empty,
            "avg_rating": avg,
            "dist": {i: dist.get(i, 0) for i in range(1, 6)},
        })

    # 字段完整率
    completeness = {}
    for field in FIELDS:
        filled = sum(1 for x in unique if str(x[field]).strip())
        completeness[field] = round(filled / len(unique), 3) if unique else 0

    # 重复正文（刷评/水军信号）
    content_counts = Counter(x["content"].strip() for x in unique if x["content"].strip())
    duplicate_content = sum(1 for _, c in content_counts.items() if c > 1)

    total = len(unique)
    gate = "通过" if total >= 300 else "未通过，需要加 App 或加页"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# P0 试点报告：互联网医疗 App 公开评论",
        "",
        f"- 采集时间：{now}",
        "- 数据源：Apple App Store 中国区公开评论 RSS（每 App 第 1 页）",
        f"- 唯一评论总数：**{total}**",
        f"- Gate 1（总量 ≥ 300）：**{gate}**",
        "",
        "## 1. 各 App 数据量",
        "",
        "| App | 条数 | 日期范围 | 中文占比 | 空正文 | 平均评分 | 1 星 | 2 星 | 3 星 | 4 星 | 5 星 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in app_stats:
        lines.append(
            f"| {s['name']} | {s['count']} | {s['date_min'][:10]} 至 {s['date_max'][:10]} | "
            f"{s['chinese_ratio']:.0%} | {s['empty_content']} | {s['avg_rating']} | "
            f"{s['dist'][1]} | {s['dist'][2]} | {s['dist'][3]} | {s['dist'][4]} | {s['dist'][5]} |"
        )

    lines += [
        "",
        "## 2. 字段完整率",
        "",
        "| 字段 | 完整率 |",
        "|---|---:|",
    ]
    for field, ratio in completeness.items():
        lines.append(f"| {field} | {ratio:.0%} |")

    lines += [
        "",
        "## 3. 初步风险信号",
        "",
        f"- 重复正文（出现 2 次及以上）：{duplicate_content} 组；需在 P1 做去重与刷评检测。",
        f"- 空正文：{sum(s['empty_content'] for s in app_stats)} 条；建议 P1 过滤。",
        "- 评论只代表“愿意写评论的用户”，不能推断全体患者。",
        "- RSS 第 1 页只有约 50 条；P1 需取 3–5 页，并检查第 2 页之后是否仍是连续时间序列。",
        "",
        "## 4. 结论与下一步",
        "",
        f"- 当前总量 {total} 条，Gate 1 结果：**{gate}**。",
        "- 若通过：进入 P1，每 App 取 3–5 页，清洗后做 codebook。",
        "- 若不足：优先加“百度健康”“蚂蚁阿福”，或把每 App 页数加到 5 页。",
        "",
        "## 5. 文件",
        "",
        "- `data/pilot_raw.json`：原始 JSON（按 App 分组）",
        "- `data/pilot_reviews.csv`：清洗前评论表",
        "- `outputs/pilot_report.md`：本报告",
    ]

    report_path = OUT / "pilot_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n总唯一评论数：{total}")
    print(f"报告：{report_path}")
    print(f"CSV：{csv_path}")


if __name__ == "__main__":
    main()
