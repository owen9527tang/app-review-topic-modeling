"""P2 替代方案：用 codebook v0.2 的关键词对全量评论做粗标。

valence 近似规则（关键词无法判断抱怨/表扬）：
  rating <= 3 → 命中的主题记为 complaint_themes
  rating >= 4 → 命中的主题记为 praise_themes
这只是一个可解释的近似，200 条人工精标用于校验它错到什么程度。

输出：data/labels_keyword.csv
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "reviews_analysis.csv"
OUTPUT = DATA / "labels_keyword.csv"

# codebook v0.2 的 12 类主题关键词
THEMES = {
    "fee": r"收费|退费|退款|扣费|乱收费|价格|费用|太贵|药费|挂号费|会员|付费|收钱|多少钱",
    "doctor": r"医生|大夫|回复|不理|敷衍|态度|专业|水平|响应|问诊|诊断|咨询|偏袒|规培",
    "delivery": r"配送|快递|发货|物流|送药|迟到|包装|送货|不发货",
    "prescription": r"处方|用药|剂量|副作用|药效|开药|药方|吃药",
    "appointment": r"挂号|预约|号源|候诊|门诊|复诊|排队|抢号|约不上",
    "tech": r"登录|闪退|卡顿|版本|更新|账号|验证码|绑定|支付|系统|bug|网络|打不开|崩溃|适配|连不上|刷新|搜索",
    "privacy": r"隐私|信息泄露|泄露|安全|实名|授权|骚扰|推销电话",
    "service": r"客服|人工服务|投诉|反馈|售后|没人理|踢皮球|不回复|态度差",
    "promotion": r"活动|提现|返现|优惠券|打卡|奖励|拉新|邀请|补贴|薅|减重|白嫖|体脂秤",
    "goods": r"商品|质量|赠品|礼品|正品|假货|过期|药箱|实物|发货",
    "content": r"广告|推送|软文|色情|文章|审核|内容|科普|中药|视频|软广",
    "other": r"骗子|垃圾|恶心|无语|离谱|倒闭",
}
AI_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z])AI(?![A-Za-z])|人工智能|智能助手|智能医生|机器人|大模型|"
    r"DeepSeek|ChatGPT|GPT|豆包|文心一言|通义千问|讯飞|智能问诊|AI问诊|AI医生|AI助手|智能客服"
)

COMPILED = {k: re.compile(v) for k, v in THEMES.items()}


def main():
    with INPUT.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("rating", "").isdigit()]

    fieldnames = [
        "review_id", "app_name", "rating", "month", "themes",
        "complaint_themes", "praise_themes", "ai_mention",
    ]
    out = []
    for row in rows:
        text = f"{row.get('title', '')} {row.get('content', '')}"
        hits = [k for k, pat in COMPILED.items() if pat.search(text)]
        rating = int(row["rating"])
        hits_wo_other = [h for h in hits if h != "other"]
        # 只有确实没有任何具体主题时，才保留 other 兜底
        themes = hits_wo_other if hits_wo_other else (["other"] if hits else [])
        complaint = "|".join(themes) if rating <= 3 else ""
        praise = "|".join(themes) if rating >= 4 else ""
        out.append({
            "review_id": row["review_id"],
            "app_name": row["app_name"],
            "rating": row["rating"],
            "month": (row.get("updated") or "")[:7],
            "themes": "|".join(themes),
            "complaint_themes": complaint,
            "praise_themes": praise,
            "ai_mention": 1 if AI_PATTERN.search(text) else 0,
        })

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)

    star1 = [r for r in out if r["rating"] == "1"]
    star5 = [r for r in out if r["rating"] == "5"]

    def dist(rs):
        c = Counter(t for r in rs for t in filter(None, r["themes"].split("|")))
        return c

    print(f"总样本 {len(out)} 条，写入 {OUTPUT}")
    d1, d5 = dist(star1), dist(star5)
    print(f"\n1 星 n={len(star1)} 的主题命中率：")
    for k, v in d1.most_common():
        print(f"  {k:12s} {v:4d}  {v/len(star1):5.1%}   (5星 {d5.get(k,0)/max(len(star5),1):5.1%})")
    print(f"\n5 星 n={len(star5)} 的主题命中率：")
    for k, v in d5.most_common():
        print(f"  {k:12s} {v:4d}  {v/len(star5):5.1%}")
    hit_any = sum(1 for r in out if r["themes"])
    print(f"\n至少命中一个主题：{hit_any} ({hit_any/len(out):.1%})")
    print(f"完全未命中：{len(out)-hit_any} ({(len(out)-hit_any)/len(out):.1%})")
    print(f"AI 关键词提及：{sum(r['ai_mention'] for r in out)} 条")


if __name__ == "__main__":
    main()
