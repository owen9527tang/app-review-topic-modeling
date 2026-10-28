# -*- coding: utf-8 -*-
"""P1 全量采集：6 个互联网医疗 App 的 App Store 公开评论 + 总体评分元数据。

数据源：
- 评论：https://itunes.apple.com/cn/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json
- 元数据：https://itunes.apple.com/lookup?id={app_id}&country=cn
"""
import csv
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

APPS = [
    {"name": "平安好医生", "app_id": 1552354076, "pages": 4},
    {"name": "京东健康", "app_id": 1498018017, "pages": 5},
    {"name": "微医", "app_id": 595277934, "pages": 5},
    {"name": "好大夫在线", "app_id": 919502358, "pages": 5},
    {"name": "春雨医生", "app_id": 478834039, "pages": 5},
    {"name": "丁香医生", "app_id": 911261462, "pages": 5},
    {"name": "蚂蚁阿福", "app_id": 6743828427, "pages": 5},
    {"name": "微脉", "app_id": 958417066, "pages": 5},
]

REVIEW_URL = "https://itunes.apple.com/cn/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json"
LOOKUP_URL = "https://itunes.apple.com/lookup?id={app_id}&country=cn"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

FIELDS = ["app_name", "app_id", "source_page", "review_id", "author", "rating", "version", "title", "content", "updated"]
META_FIELDS = ["app_name", "app_id", "track_name", "average_user_rating", "user_rating_count", "version", "current_version_release_date", "track_view_url"]


def fetch_json(url, retries=3, pause=0.5):
    """带重试的 GET，返回解析后的 JSON。"""
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # 网络波动时重试
            last_error = exc
            time.sleep(pause * (attempt + 1))
    raise last_error


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def label(node, *keys):
    cur = node
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return ""
        cur = cur[key]
    return cur.get("label", "") if isinstance(cur, dict) else ""


def parse_entry(app_name, app_id, page, entry):
    return {
        "app_name": app_name,
        "app_id": app_id,
        "source_page": page,
        "review_id": label(entry, "id"),
        "author": label(entry, "author", "name"),
        "rating": label(entry, "im:rating"),
        "version": label(entry, "im:version"),
        "title": label(entry, "title"),
        "content": label(entry, "content"),
        "updated": label(entry, "updated"),
    }


def collect_reviews():
    all_rows = []
    summary = []
    for app in APPS:
        app_rows = []
        page_counts = []
        for page in range(1, app["pages"] + 1):
            url = REVIEW_URL.format(page=page, app_id=app["app_id"])
            try:
                payload = fetch_json(url)
            except Exception as exc:
                page_counts.append(f"p{page}=ERR")
                print(f"{app['name']} p{page}: ERROR {exc}")
                break
            entries = as_list(payload.get("feed", {}).get("entry"))
            rows = [parse_entry(app["name"], app["app_id"], page, e) for e in entries]
            app_rows.extend(rows)
            page_counts.append(f"p{page}={len(rows)}")
            if not rows:
                break
            time.sleep(0.4)
        all_rows.extend(app_rows)
        summary.append((app["name"], app["app_id"], len(app_rows), " ".join(page_counts)))
        print(f"{app['name']}: {len(app_rows)} 条 ({' '.join(page_counts)})")
    return all_rows, summary


def collect_metadata():
    rows = []
    for app in APPS:
        url = LOOKUP_URL.format(app_id=app["app_id"])
        try:
            payload = fetch_json(url)
            item = as_list(payload.get("results"))[0] if payload.get("results") else {}
        except Exception as exc:
            print(f"{app['name']} lookup: ERROR {exc}")
            item = {}
        rows.append({
            "app_name": app["name"],
            "app_id": app["app_id"],
            "track_name": item.get("trackName", ""),
            "average_user_rating": item.get("averageUserRating", ""),
            "user_rating_count": item.get("userRatingCount", ""),
            "version": item.get("version", ""),
            "current_version_release_date": item.get("currentVersionReleaseDate", ""),
            "track_view_url": item.get("trackViewUrl", ""),
        })
        time.sleep(0.3)
    return rows


def dedupe(rows):
    seen = set()
    unique = []
    for row in rows:
        key = (row["app_id"], row["review_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def main():
    print(f"开始采集：{datetime.now():%Y-%m-%d %H:%M:%S}")
    raw_rows, summary = collect_reviews()
    unique_rows = dedupe(raw_rows)
    metadata = collect_metadata()

    payload = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "apps": APPS,
        "page_summary": [{"app_name": s[0], "app_id": s[1], "rows": s[2], "pages": s[3]} for s in summary],
        "reviews": unique_rows,
        "metadata": metadata,
    }
    (DATA / "reviews_raw.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (DATA / "reviews_raw.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(unique_rows)

    with (DATA / "app_metadata.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)

    print("\n汇总：")
    for s in summary:
        print(f"  {s[0]}: {s[2]} 条 ({s[3]})")
    print(f"唯一评论总数：{len(unique_rows)}")
    print(f"原始文件：{DATA / 'reviews_raw.csv'}")
    print(f"元数据：{DATA / 'app_metadata.csv'}")


if __name__ == "__main__":
    main()
