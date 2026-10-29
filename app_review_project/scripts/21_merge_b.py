"""B 表的半自动衔接：细项映射到 6 组 + 合并 B 的 spec 重判结果。

为什么这样分工：
- `primary` 由 12 类合并为 6 组是**粒度换算**，不改变任何判断，可以直接机械映射；
- `spec` 的判定标准从「有对象无细节 / 可核实细节」改成了「正文有无数量词」，两个维度不可映射，
  必须由第二人重新看一遍 —— 这一步不能由脚本代劳，否则 κ 失去意义。

用法：
  python scripts/21_merge_b.py --groups    # 生成 data/coding_B_groups.csv（只需一次）
  python scripts/21_merge_b.py --merge     # B 导出 coding_B_spec.csv 后，合成 data/coding_B.csv
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

M12 = {
    "doctor": "医疗服务", "appointment": "医疗服务", "prescription": "医疗服务",
    "fee": "费用与权益", "promotion": "费用与权益", "service": "费用与权益",
    "tech": "技术与账号", "privacy": "技术与账号",
    "delivery": "药品与商品", "goods": "药品与商品",
    "content": "内容", "other": "其他",
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def groups():
    src = DATA / "coding_B.csv"
    rows = read(src)
    out, blank = [], 0
    for r in rows:
        p = (r.get("primary") or "").strip()
        if p == "":
            blank += 1
            out.append({"review_id": r["review_id"], "primary_group": ""})
            continue
        if p not in M12:
            raise SystemExit(f"未知细项 {p!r}（review_id {r['review_id']}），先确认表是否为 12 类版本")
        out.append({"review_id": r["review_id"], "primary_group": M12[p]})
    write(DATA / "coding_B_groups.csv", out, ["review_id", "primary_group"])
    print(f"已写入 coding_B_groups.csv：{len(out)} 条，主议题空缺 {blank} 条")


def merge():
    spec = {r["review_id"]: (r.get("spec") or "").strip()
            for r in read(DATA / "coding_B_spec.csv")}
    grp = {r["review_id"]: r["primary_group"] for r in read(DATA / "coding_B_groups.csv")}
    rows = read(DATA / "coding_B.csv")
    blank = 0
    for r in rows:
        r["spec"] = spec.get(r["review_id"], "")
        r["primary"] = grp.get(r["review_id"], "")
        if r["spec"] == "":
            blank += 1
    write(DATA / "coding_B.csv", rows, ["review_id", "app_name", "content", "spec", "primary", "note"])
    print(f"已合成 coding_B.csv：{len(rows)} 条，spec 未填 {blank} 条")
    if blank:
        raise SystemExit("B 的 spec 还没编完，补齐后再跑")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.groups:
        groups()
    if a.merge:
        merge()
    if not (a.groups or a.merge):
        ap.print_help()
