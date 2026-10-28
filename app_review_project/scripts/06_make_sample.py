"""生成校准用分层样本：8 个 App 各 25 条，App 内按星级比例分配。

输出 data/sample_200.csv（review_id, app_name, rating, month, content）
固定 seed，可复现。
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "reviews_analysis.csv"
OUTPUT = DATA / "sample_200.csv"
PER_APP = 25
SEED = 42


def allocate(counts, n):
    """按比例分配 n 个名额，最大余数法。"""
    total = sum(counts.values())
    raw = {k: v * n / total for k, v in counts.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    rest = n - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[:rest]:
        alloc[k] += 1
    return alloc


def main():
    with INPUT.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("rating", "").isdigit()]

    by_app = defaultdict(list)
    for r in rows:
        by_app[r["app_name"]].append(r)

    random.seed(SEED)
    picked = []
    for app in sorted(by_app):
        items = by_app[app][:]
        random.shuffle(items)
        strata = defaultdict(list)
        for r in items:
            strata[r["rating"]].append(r)
        alloc = allocate({k: len(v) for k, v in strata.items()}, PER_APP)
        for rating, k in alloc.items():
            picked.extend(strata[rating][:k])

    picked.sort(key=lambda r: (r["app_name"], r["rating"], r["review_id"]))
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["review_id", "app_name", "rating", "month", "content"]
        )
        writer.writeheader()
        for r in picked:
            writer.writerow({
                "review_id": r["review_id"],
                "app_name": r["app_name"],
                "rating": r["rating"],
                "month": (r.get("updated") or "")[:7],
                "content": f"{r.get('title', '')} {r.get('content', '')}".strip(),
            })

    dist = defaultdict(int)
    for r in picked:
        dist[r["rating"]] += 1
    print(f"写入 {OUTPUT}，共 {len(picked)} 条")
    print("星级分布：" + "，".join(f"{k}星 {dist[k]}" for k in sorted(dist)))


if __name__ == "__main__":
    main()
