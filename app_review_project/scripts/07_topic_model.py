"""P2 主路线：中文分词 + TF-IDF + NMF 主题发现。

步骤：
1. jieba 分词（自定义词典 + 停用词）
2. TF-IDF 向量化，NMF 分解，扫描 k 选主题数
3. 输出每个主题的 top 词，供人工命名并映射回 codebook v0.2

输出：
  data/topic_topwords.csv   各 k 的主题 top 词
  data/topic_doc_weights.csv 选定 k 的文档-主题权重
"""
import csv
import re
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "reviews_analysis.csv"
TOPWORDS_OUT = DATA / "topic_topwords.csv"
WEIGHTS_OUT = DATA / "topic_doc_weights.csv"
SELECTED_K = 10
KS = [6, 8, 10, 12, 14, 16]

CUSTOM_WORDS = [
    "挂号费", "问诊费", "药费", "退费", "退款", "提现", "返现", "闪退", "卡顿",
    "验证码", "客服", "人工客服", "处方", "电子处方", "复诊", "抢号", "号源",
    "候诊", "配送", "物流", "快递", "副作用", "剂量", "隐私", "实名", "授权",
    "推送", "软文", "会员", "自动扣费", "乱收费", "连不上", "打不开", "弹窗",
    "体脂秤", "打卡", "补贴", "优惠券", "拉新", "赠品", "过期", "假药", "正品",
    "药店", "药品", "医生", "大夫", "挂号", "预约", "广告", "科普", "直播",
    "问诊", "慢病", "医保", "报销", "病历", "检验报告", "体检", "图文问诊",
    "电话问诊", "视频问诊", "智能问诊", "机器人", "人工智能", "算法", "大数据",
    "处方药", "医保卡", "预约挂号", "在线问诊", "互联网医院", "三甲医院",
]

STOP = set("""
的 了 我 你 他 她 它 是 在 有 就 都 也 还 很 太 这 那 个 们 和 与 啊 吧 呢 吗
哦 呀 嘛 呗 一 不 没 没有 什么 怎么 为什么 现在 感觉 觉得 真的 非常 特别 而且
但是 因为 所以 如果 可以 一个 这个 那个 时候 一直 已经 又 再 能 会 要 想 说
去 来 上 下 多 少 好 差 只 让 给 被 把 从 到 对 最 更 挺 蛮 有些 有点
一点 一些 一次 一定 一般 一样 这么 那么 这样 那样 他们 我们 你们 自己
东西 事情 问题 情况 感觉 体验 使用 用了 用 用着 下载 安装 打开 进去 出来
希望 建议 要求 应该 需要 可能 好像 似乎 简直 实在 确实 完全 根本 反正 然后
之后 之前 以后 以前 目前 今天 昨天 明天 时间 第二 第一 两天 几天
软件 app 手机 版本 系统 里面 上面 下面 后面 前面 中间 左右 大家 别人
""".split())

# App 名与平台词：会把主题变成“某个 App 的评论”，必须剔除
BRAND_STOP = set("""
平安 好医生 京东 微医 微脉 春雨 丁香 好大夫 蚂蚁 阿福 苹果 ios 安卓 华为 小米
平板 ipad 客户端 官网 平台 互联网 医疗 健康 医生端 微信公众号 小程序
""".split())

# 纯极性/泛泛评价词：会把主题变成“好评类/差评类”，与星级重复，剔除
SENTIMENT_STOP = set("""
不错 好用 实用 方便 便捷 方便快捷 挺好 挺好用 好用极了 喜欢 推荐 值得 感谢 支持
满意 棒 完美 优秀 舒服 贴心 耐心 专业 负责 细心 认真 高效 及时 快速 很快 迅速
清晰 准确 正确 靠谱 可靠 良心 实在 真诚 热情 周到 感动 福音 加油 好评 五星
不好 不行 不满 失望 讨厌 恶心 无语 离谱 垃圾 骗子 坑人 骗人 骗钱 黑心 无耻
垃圾软件 差评 烂 废 恶心人 气人 生气 愤怒 气愤 崩溃 可怕 恐怖 恐怖 心累 累
超棒 超级 特别棒 非常棒 一般 一般般 马马虎虎 勉强 凑合 还凑合 尚可 欠佳
不是 不能 无法 不了 不会 不要 不到 不行 真是 怎么回事 每次 就是 还是 很多
起来 这种 的话 用来用去 好好 回来 出来 来到 过了 完了 算了 得了 越来越
""".split())

# 通用动词/泛指词：横跨所有主题，去掉才能让主题落在“具体对象”上
GENERIC_STOP = set("""
解决 帮助 提供 得到 能够 知道 找到 这里 一下 谢谢 赶紧 增加 其他 全面 各种
所有 开始 最后 成功 产品 一款 要是 千万 机会 就算 不用 再也 基本上 麻烦
小病 问问 发现 直接 全是 最近 提示 反应 请教 描述 选择 网上 一次 第一次
知名 国内 广大 很大 有用 有效 最好 很棒 认真负责 太有 真心 各种 老是
""".split())

STOP |= BRAND_STOP | SENTIMENT_STOP | GENERIC_STOP

CLEAN_RE = re.compile(r"[^\u4e00-\u9fa5A-Za-z]+")

for w in CUSTOM_WORDS:
    jieba.add_word(w)


def tokenize(text: str):
    text = CLEAN_RE.sub(" ", str(text))
    words = []
    for w in jieba.lcut(text):
        w = w.strip().lower()
        if len(w) >= 2 and w not in STOP and not w.isdigit():
            words.append(w)
    return words


def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", dtype=str)
    df = df[df["rating"].astype(float).notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()
    df["month"] = df["updated"].fillna("").str[:7]
    df["year"] = df["updated"].fillna("").str[:4]
    df["tokens"] = df["text"].map(tokenize)
    empty = (df["tokens"].map(len) == 0).sum()
    print(f"样本 {len(df)} 条，分词后为空 {empty} 条")

    vec = TfidfVectorizer(
        tokenizer=lambda t: t,
        preprocessor=lambda t: t,
        token_pattern=None,
        min_df=5,
        max_df=0.5,
        sublinear_tf=True,
    )
    X = vec.fit_transform(df["tokens"])
    terms = np.array(vec.get_feature_names_out())
    print(f"词表 {X.shape[1]} 个词，文档 {X.shape[0]} 条")

    rows = []
    for k in KS:
        nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=600)
        W = nmf.fit_transform(X)
        top10 = set()
        for i, comp in enumerate(nmf.components_):
            idx = np.argsort(comp)[::-1][:15]
            words = [terms[j] for j in idx]
            top10.update(words[:10])
            rows.append({
                "k": k,
                "topic": i,
                "top_words": " ".join(words),
                "recon_err": round(float(nmf.reconstruction_err_), 1),
            })
        print(f"k={k:2d} 重建误差 {nmf.reconstruction_err_:7.1f}  主题间 top10 去重率 {len(top10)/(k*10):.2f}")

    pd.DataFrame(rows).to_csv(TOPWORDS_OUT, index=False, encoding="utf-8-sig")
    print(f"top 词写入 {TOPWORDS_OUT}")

    # 选定 k 保存文档权重
    nmf = NMF(n_components=SELECTED_K, init="nndsvda", random_state=42, max_iter=600)
    W = nmf.fit_transform(X)
    share = W / np.clip(W.sum(axis=1, keepdims=True), 1e-9, None)
    out = df[["review_id", "app_name", "rating", "month", "year", "text"]].copy()
    for i in range(SELECTED_K):
        out[f"t{i:02d}"] = share[:, i]
    out.to_csv(WEIGHTS_OUT, index=False, encoding="utf-8-sig")
    print(f"k={SELECTED_K} 文档权重写入 {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
