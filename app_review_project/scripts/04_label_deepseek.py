# -*- coding: utf-8 -*-
"""P2 试点：用 DeepSeek 按 codebook 编码 App 评论。

用法：
  $env:DEEPSEEK_API_KEY="sk-..."
  python 04_label_deepseek.py --limit 50 --dry-run   # 只看抽样和 prompt
  python 04_label_deepseek.py --limit 50             # 跑 50 条试点

API key 只从环境变量读取，不写入文件、不打印。
"""
import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_INPUT = DATA / "reviews_analysis.csv"
DEFAULT_OUTPUT = DATA / "labels_pilot.csv"
DEFAULT_RAW = DATA / "labels_pilot_raw.json"
CODEBOOK_PATH = ROOT / "codebook.md"

API_URL = "https://api.deepseek.com/chat/completions"
ALLOWED_THEMES = {
    "fee", "doctor", "delivery", "prescription", "appointment", "tech", "privacy",
    "service", "promotion", "goods", "content", "other",
}
AI_CONTEXTS = {"feature_use", "feature_trust", "feature_failure", "competitor_compare", "metaphor", "none"}

SYSTEM_PROMPT = (
    "你是严谨的研究助理。严格按用户提供的 codebook 对 App 评论做多标签编码。"
    "只输出 JSON，不要 markdown，不要解释，不要添加 codebook 之外的字段。"
)


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [row for row in csv.DictReader(fh) if row.get("rating", "").isdigit()]


def stratified_sample(rows, n, seed=42):
    """按 App × 星级分层抽样：每个 App 尽量取 3 条 1 星、2 条 5 星、1 条中间。"""
    random.seed(seed)
    by_app = defaultdict(list)
    for row in rows:
        by_app[(row["app_name"], row["app_id"])].append(row)

    chosen = []
    chosen_ids = set()
    for _, items in by_app.items():
        random.shuffle(items)
        buckets = [("1", 3), ("5", 2), ("mid", 1)]
        for bucket, k in buckets:
            if bucket == "mid":
                picked = [r for r in items if r["rating"] in {"2", "3", "4"} and id(r) not in chosen_ids][:k]
            else:
                picked = [r for r in items if r["rating"] == bucket and id(r) not in chosen_ids][:k]
            for row in picked:
                chosen_ids.add(id(row))
                chosen.append(row)
        # 某个 App 分层不足时，用该 App 剩余评论补齐到 6 条
        app_chosen = sum(1 for r in chosen if r["app_id"] == items[0]["app_id"])
        if app_chosen < 6:
            fill = [r for r in items if id(r) not in chosen_ids][: 6 - app_chosen]
            for row in fill:
                chosen_ids.add(id(row))
                chosen.append(row)

    remaining = [r for r in rows if id(r) not in chosen_ids]
    random.shuffle(remaining)
    chosen.extend(remaining[: max(0, n - len(chosen))])
    return chosen[:n]


def build_prompt(batch, codebook):
    payload = [
        {
            "review_id": row["review_id"],
            "app": row["app_name"],
            "rating": int(row["rating"]),
            "text": (row.get("content") or "")[:800],
        }
        for row in batch
    ]
    user = (
        f"{codebook}\n\n"
        "## 待编码评论\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "请输出一个 JSON 对象：{\"labels\": [...]}。"
        "每个对象必须包含 review_id、complaint_themes、praise_themes、ai_mention、"
        "ai_context、ai_sentiment、empty_praise、overall_sentiment、evidence_quote。"
        "complaint_themes 和 praise_themes 只能使用 codebook 中的 key；没有则为空数组。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_api(api_key, model, messages, use_json=True, timeout=120):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4000,
    }
    if use_json:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def call_with_retry(api_key, model, messages, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            try:
                return call_api(api_key, model, messages, use_json=True)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                # 个别账号不支持 response_format，退化为普通 JSON 输出
                if exc.code == 400 and "response_format" in detail:
                    return call_api(api_key, model, messages, use_json=False)
                raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}")
        except Exception as exc:  # 网络或限流，退避重试
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last_error


def parse_labels(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    labels = obj.get("labels") if isinstance(obj, dict) else obj
    if isinstance(labels, dict):
        labels = [labels]
    if not isinstance(labels, list):
        raise ValueError("labels 不是数组")
    return labels


def normalize_label(item, row):
    def theme_list(value):
        if not isinstance(value, list):
            return []
        return [x for x in value if x in ALLOWED_THEMES]

    def tri(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return 0
        return value if value in (-1, 0, 1) else 0

    ai_context = item.get("ai_context", "none")
    if ai_context not in AI_CONTEXTS:
        ai_context = "none"
    return {
        "review_id": row["review_id"],
        "app_name": row["app_name"],
        "rating": row["rating"],
        "content": row.get("content", ""),
        "complaint_themes": "|".join(theme_list(item.get("complaint_themes"))),
        "praise_themes": "|".join(theme_list(item.get("praise_themes"))),
        "ai_mention": 1 if int(item.get("ai_mention", 0) or 0) == 1 else 0,
        "ai_context": ai_context,
        "ai_sentiment": tri(item.get("ai_sentiment")),
        "empty_praise": 1 if int(item.get("empty_praise", 0) or 0) == 1 else 0,
        "overall_sentiment": tri(item.get("overall_sentiment")),
        "evidence_quote": (item.get("evidence_quote") or "")[:120],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="0 表示全量")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只抽样并打印 prompt，不调用 API")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit("输入文件没有可用评论")

    sample = rows if args.limit == 0 else stratified_sample(rows, args.limit, args.seed)
    codebook = CODEBOOK_PATH.read_text(encoding="utf-8")

    if args.dry_run:
        dry_path = DATA / "pilot_sample.csv"
        with dry_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["review_id", "app_name", "rating", "content"])
            writer.writeheader()
            writer.writerows({k: row[k] for k in ["review_id", "app_name", "rating", "content"]} for row in sample)
        prompt = build_prompt(sample[: min(2, len(sample))], codebook)
        print(f"抽样 {len(sample)} 条，已写入 {dry_path}")
        print("Prompt 预览（前 1200 字）：")
        print(prompt[1]["content"][:1200])
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("未找到 DEEPSEEK_API_KEY。请在 PowerShell 中设置后重跑：")
        print('  $env:DEEPSEEK_API_KEY="sk-..."')
        print(f"  python {Path(__file__).name} --limit {args.limit}")
        raise SystemExit(1)

    all_labels = []
    raw_records = []
    errors = 0
    for start in range(0, len(sample), args.batch_size):
        batch = sample[start : start + args.batch_size]
        messages = build_prompt(batch, codebook)
        print(f"编码 {start + 1}-{start + len(batch)} / {len(sample)} ...", flush=True)
        try:
            text = call_with_retry(api_key, args.model, messages)
            parsed = parse_labels(text)
            by_id = {str(item.get("review_id", "")): item for item in parsed}
            for row in batch:
                item = by_id.get(str(row["review_id"]))
                if item is None:
                    errors += 1
                    continue
                all_labels.append(normalize_label(item, row))
            raw_records.append({"batch_start": start, "response": text})
        except Exception as exc:
            errors += len(batch)
            raw_records.append({"batch_start": start, "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)

    fields = [
        "review_id", "app_name", "rating", "content", "complaint_themes", "praise_themes",
        "ai_mention", "ai_context", "ai_sentiment", "empty_praise", "overall_sentiment",
        "evidence_quote",
    ]
    with Path(args.output).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_labels)
    Path(args.raw_output).write_text(
        json.dumps({"model": args.model, "labeled_at": datetime.now().isoformat(timespec="seconds"),
                    "records": raw_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    theme_counts = Counter()
    for row in all_labels:
        for key in filter(None, row["complaint_themes"].split("|")):
            theme_counts[key] += 1
    ai_count = sum(row["ai_mention"] for row in all_labels)
    print(f"\n完成：{len(all_labels)} 条，错误/缺失 {errors} 条")
    print(f"AI 提及：{ai_count} 条")
    print("抱怨主题分布：" + "，".join(f"{k}={v}" for k, v in theme_counts.most_common()))
    print(f"输出：{args.output}")


if __name__ == "__main__":
    main()
