"""P4 副线：68 条提到 AI 的评论定性精读。

背景：第 8 节只给了「AI 提及率」的量化结果——1 星 5.4% vs 5 星 2.0%。
但提及率回答不了「他们到底在说 AI 什么」。这里把 68 条全部读过一遍，
按评论的主要诉求编码，再按星级拆分，看两件事：

  1. 1 星的 AI 抱怨集中在哪一类（识别/记忆失败 vs 诊断质量 vs 语气）
  2. 5 星的 AI 好评集中在哪一类（报告解读、轻症咨询、减重记录）

编码是单人完成、按评论主旨取**唯一主码**，不做多标签，也不计算 κ。
编码结果写在 CODES 里，review_id 与 data/labels_keyword.csv 的 ai_mention=1 一一对应。

输出：outputs/ai_read.md、data/ai_codes.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
REPORT = OUT / "ai_read.md"
TABLE = DATA / "ai_codes.csv"

CODES = {
    "C1": ("识别与记忆失败", "动作/语音/图片识别失灵；记不住对话与数据；记录被改写"),
    "C2": ("诊断与解答质量", "回答错误、胡说、反复道歉、被判定为「假 AI 医生」"),
    "C3": ("与通用助手对比", "明确拿豆包 / DeepSeek 作基准，认为能力或形态不如"),
    "C4": ("语气、信任与隐私", "态度恶劣、爹味、被恶意揣测；不信任；实名与隐私担忧"),
    "C5": ("功能与交互设计", "不能发图、记录必须拍照、入口难找等产品设计问题"),
    "C6": ("技术故障", "闪退、崩溃、收不到消息、网络报错"),
    "C7": ("正面认可", "报告解读、轻症咨询、减重记录、情感陪伴等明确好用"),
    "C8": ("收费", "AI 能力需要付费，与免费助手比较"),
    "C9": ("修辞式提及", "AI 只作比喻（「跟 AI 自动回复似的」）或功能清单里的一个词"),
    "C0": ("与 AI 无关", "关键词误命中；抱怨的是退款、活动规则、医生质量等"),
}

# review_id → 主码（顺序与 labels_keyword.csv 中 ai_mention==1 的行一致）
ASSIGN = {
    "7353333001": "C0", "7149403498": "C0", "14614967715": "C0", "14606057431": "C9",
    "14600062809": "C1", "14587583147": "C1", "14587468950": "C2", "14579270399": "C1",
    "14575464732": "C1", "14574784187": "C0", "14569063007": "C1", "14557116548": "C0",
    "14549492109": "C2", "14548773134": "C1", "14548504218": "C1", "14547369971": "C1",
    "14547281707": "C1", "14545688139": "C1", "14545427687": "C1", "14543892775": "C1",
    "14538936368": "C9", "14536536008": "C1", "14533899481": "C0", "14533695284": "C2",
    "10965650618": "C0", "6866822831": "C7", "14161077165": "C0", "14158011504": "C9",
    "13811756527": "C9", "8792631324": "C0", "14077804085": "C9", "11810465350": "C9",
    "8126788288": "C0", "14708109773": "C3", "14695937126": "C7", "14693682505": "C1",
    "14687499435": "C4", "14684015932": "C4", "14682011784": "C1", "14668421384": "C4",
    "14662615233": "C5", "14661478778": "C3", "14660158193": "C5", "14657388347": "C1",
    "14648946758": "C7", "14648217925": "C1", "14644592207": "C7", "14638754030": "C7",
    "14626114058": "C7", "14605934226": "C1", "14603813309": "C5", "14603143507": "C6",
    "14579310008": "C7", "14575382580": "C7", "14559798220": "C2", "14558639524": "C7",
    "14558091638": "C3", "14556457017": "C7", "14556441301": "C7", "14555346247": "C7",
    "14552974291": "C1", "14549177105": "C4", "14528537500": "C0", "14523855511": "C1",
    "14523231498": "C2", "14508714881": "C8", "14502340792": "C1", "14500211414": "C1",
}

# 报告里要引用的原话（review_id → 截取片段）
QUOTES = [
    ("14600062809", "1", "京东健康", "「ai识别不灵」「页面设计极其混乱」——识别失败与界面混乱同时出现"),
    ("14587583147", "1", "京东健康", "「做五十个喝彩…因为贵软件太垃圾识别不到」——识别失败直接连着活动奖励"),
    ("14682011784", "1", "蚂蚁阿福", "「几分钟前自己记录的东西无法调用…和用户对话充满负能量」——记忆失败与语气问题叠加"),
    ("14552974291", "2", "蚂蚁阿福", "「不记得对话的内容，不记得一些数据…医疗类的 AI 软件必须要有的东西」——用户把记忆当底线功能"),
    ("14648217925", "2", "蚂蚁阿福", "「豆包的识别能力 95%…阿福的语音识别只有 20-30%」——唯一给出量化对比的评论"),
    ("14668421384", "1", "蚂蚁阿福", "「总以最坏的恶意揣测患者，每次都被气的半死」——语气问题而非能力问题"),
    ("14684015932", "1", "蚂蚁阿福", "「扫了她脖子上挂的二维码…我的隐私信息就上传了」——线下推广带来的信任问题"),
    ("14648946758", "5", "蚂蚁阿福", "「体检报告一堆专业术语看不懂，拍照上传就能解读…红黄蓝绿」——5 星的主场是报告解读"),
    ("14575382580", "5", "蚂蚁阿福", "「对于有点小病就嘀咕的人群太友好了，特别管用」——轻症咨询：把 AI 当第一道分诊"),
    ("14558639524", "5", "蚂蚁阿福", "「有点健康焦虑…阿福都耐心解答了，而且它记录了我的体检报告」——情绪价值 + 档案记忆是 5 星的两根支柱"),
    ("14556441301", "5", "蚂蚁阿福", "「定制食谱和识别每餐的热量；体重终于开始下降了」——减重场景的具体收益"),
    ("14606057431", "1", "京东健康", "「买药看病＋ai 咨询＋减重运动＋社交＋刷视频」——AI 只是被吐槽的功能堆砌之一"),
    ("14549492109", "1", "京东健康", "「ai医生还不如隔壁阿福，乱解答」——AI 被当作横向比较的锚点"),
]


def md_table(df, cols, headers, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c].format(v)
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    kw = pd.read_csv(DATA / "labels_keyword.csv", encoding="utf-8-sig", dtype=str)
    df = pd.read_csv(DATA / "reviews_analysis.csv", encoding="utf-8-sig", dtype=str)
    sub = kw[kw["ai_mention"] == "1"].copy()

    missing = [r for r in sub["review_id"] if r not in ASSIGN]
    extra = [r for r in ASSIGN if r not in set(sub["review_id"])]
    if missing or extra:
        print(f"编码与数据不一致：未编码 {len(missing)} 条，多余 {len(extra)} 条")
        sys.exit(1)

    sub["code"] = sub["review_id"].map(ASSIGN)
    sub["rating"] = sub["rating"].astype(int)
    merged = sub.merge(df[["review_id", "content"]], on="review_id", how="left")
    merged["label"] = merged["code"].map(lambda c: CODES[c][0])
    merged[["review_id", "app_name", "rating", "month", "code", "label", "content"]].to_csv(
        TABLE, index=False, encoding="utf-8-sig"
    )

    tab = (
        merged.groupby(["code", "label"])
        .agg(n=("review_id", "size"), 星1=("rating", lambda s: int((s == 1).sum())),
             星2=("rating", lambda s: int((s == 2).sum())),
             星3=("rating", lambda s: int((s == 3).sum())),
             星5=("rating", lambda s: int((s == 5).sum())))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    n1 = int((merged["rating"] == 1).sum())
    n5 = int((merged["rating"] == 5).sum())
    n_mid = len(merged) - n1 - n5

    by_app = merged.groupby("app_name").agg(n=("review_id", "size"), 相关=("code", lambda s: int((s != "C0").sum())))
    by_app["非误标占比"] = by_app["相关"] / by_app["n"]

    top1 = tab.sort_values("星1", ascending=False).head(1).iloc[0]
    top5 = tab.sort_values("星5", ascending=False).head(1).iloc[0]
    n_theme_fail = int(tab.loc[tab["code"] == "C1", "n"].iloc[0])
    n_quality = int(tab.loc[tab["code"].isin(["C2", "C3"]), "n"].sum())
    n_praise = int(tab.loc[tab["code"] == "C7", "n"].iloc[0])
    n_false = int(tab.loc[tab["code"] == "C0", "n"].iloc[0])
    two = ["蚂蚁阿福", "京东健康"]
    n_two = int(merged["app_name"].isin(two).sum())
    n_rest = len(merged) - n_two

    lines = [
        "# 68 条「提到 AI」的评论：定性精读",
        "",
        "> 第 8 节只回答了「多少评论提到 AI」；这一页回答「提到的时候在说什么」。"
        "全部 68 条逐条读过，按**主要诉求**取唯一主码，单次编码、未做第二人复核。",
        "",
        f"样本：`labels_keyword.csv` 中 `ai_mention = 1` 的 {len(merged)} 条；其中 1 星 {n1} 条、5 星 {n5} 条，"
        f"其余星级（2–4 星）共 {n_mid} 条。原始文本见 `data/ai_codes.csv`。",
        "",
        "## 1. 一句话结论",
        "",
        f"- 1 星谈 AI，**{top1['label']}** 是最大宗（{int(top1['星1'])}/{n1}）：不是「AI 不够聪明」，"
        "而是「识别不出来、记不住我刚才说的话」。这是可用性失败，不是智力失败。",
        f"- 5 星谈 AI，**{top5['label']}** 是最大宗（{int(top5['星5'])}/{n5}）：体检报告解读、轻症先问一句、"
        "减重记录——都是有具体交付物的场景。",
        f"- 有 {n_false} 条（{n_false / len(merged):.0%}）其实是关键词误命中：抱怨的是退款、活动规则或医生质量，"
        "AI 只是顺带被骂。这意味着「AI 提及率」这个指标本身高估了 AI 的真实存在感。",
        "",
        "## 2. 编码分布",
        "",
        md_table(tab, ["code", "label", "n", "星1", "星2", "星3", "星5"],
                 ["码", "类别", "合计", "1 星", "2 星", "3 星", "5 星"]),
        "",
        "编码口径：",
    ]
    for k, (name, desc) in CODES.items():
        lines.append(f"- `{k}` {name}：{desc}")
    lines += [
        "",
        "## 3. 按 App 拆分",
        "",
        md_table(by_app.reset_index(), ["app_name", "n", "相关", "非误标占比"],
                 ["App", "提到 AI", "与 AI 真相关", "占比"], {"非误标占比": "{:.0%}"}),
        "",
        f"{n_two} 条（{n_two / len(merged):.0%}）来自蚂蚁阿福与京东健康，与第 8 节的提及率排序一致；"
        f"其余 5 个 App 合计只有 {n_rest} 条，且多为误标或修辞式提及（「跟 AI 自动回复似的」），"
        "所以在其他 App 上不能说「用户对 AI 有意见」，只能说「用户拿 AI 当贬义词用」。",
        "",
        f"## 4. 原话（{len(QUOTES)} 条）",
        "",
        "| 星级 | App | 评论片段 | 读法 |",
        "|---|---|---|---|",
    ]
    for rid, star, app, note in QUOTES:
        snippet, sep, read = note.partition("——")
        read = read if sep else "（示例引语）"
        lines.append(f"| {star} | {app} | {snippet} | {read} |")
    lines += [
        "",
        "## 5. 这一页能支持什么、不能支持什么",
        "",
        "能支持：",
        f"- 「1 星的 AI 抱怨以可用性为主」：{n_theme_fail} 条属识别/记忆失败，"
        f"只有 {n_quality} 条明确说「诊断错/胡说」或「不如豆包、DeepSeek」。",
        f"- 「5 星的 AI 好评以具体任务为主」：{n_praise} 条正面提及里，"
        "报告解读、轻症咨询、减重记录三类占了绝大多数，几乎没有「AI 很聪明」这类抽象夸奖。",
        "",
        "不能支持：",
        f"- **不能说「AI 好用/难用」是普遍态度**：{len(merged)} 条只占全样本 {len(merged) / 1681:.0%}，"
        f"且 {n_two} 条（{n_two / len(merged):.0%}）来自两家 App。",
        "- **不能说这是产品之间的比较结论**：蚂蚁阿福是 AI 原生形态，京东健康是活动驱动的 AI 识别，"
        "两者的「AI」根本不是同一件事（一个是对话助手，一个是动作识别），不能并列排名。",
        "- **编码是单人的**，类别边界（尤其 C1 与 C5）靠判断；若要投稿或对外发布，需要第二人独立编码并报 κ。",
        "",
        "## 6. 对主报告的改动建议（已执行）",
        "",
        "第 8 节原来只有提及率表格，容易被读成「用户不喜欢 AI」。"
        "现在补一句方向性结论：**1 星骂的是识别与记忆，5 星夸的是具体任务交付**，"
        "并注明 {:.0%} 的提及是误标。这既纠正了方向，也保住了样本量不足的边界。".format(n_false / len(merged)),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(tab.to_string(index=False))
    print(f"1 星 {n1} / 5 星 {n5}；误标 {n_false}；识别失败 {n_theme_fail}；正面 {n_praise}")
    print(f"写出 {REPORT.name} 与 {TABLE.name}")


if __name__ == "__main__":
    main()
