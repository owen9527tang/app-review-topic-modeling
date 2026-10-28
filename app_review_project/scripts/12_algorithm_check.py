"""P4 算法对照：NMF vs LDA。

回答面试里最容易被问的一句「为什么用 NMF 不用 LDA」。做法是把算法换掉、
其他全部不动（同一批分词、同一 TF-IDF 矩阵、同一 k=10、同样随机种子），比较：

  1. 主题内部一致性：UMass coherence（top-10 词的共现）
  2. 主题可对应性：词分布余弦 + 匈牙利算法一一指派，看 10 个主题能否对上
  3. 核心结论是否复现：活动议题、医生合并主题的「1 星 − 5 星」份额差符号

LDA 也直接拟合在 TF-IDF 矩阵上，而不是词频矩阵，目的是只改变「分解算法」这一个变量；
代价是偏离 LDA 的生成假设（多项式计数），报告里要写明。

输出：outputs/algorithm_check.md、data/lda_vs_nmf.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
INPUT = DATA / "reviews_analysis.csv"
REPORT = OUT / "algorithm_check.md"
TABLE_OUT = DATA / "lda_vs_nmf.csv"

K = 10
SEED = 42
TOP_N = 10

TOPIC_NAME = {
    0: "医生解答与沟通",
    1: "医院挂号与就医流程",
    2: "活动打卡与 AI 识别",
    3: "专家资源与医术口碑",
    4: "付费问诊次数与退款",
    5: "免费咨询与买药",
    6: "App 功能与挂号操作",
    7: "客服售后与配送体检",
    8: "技术故障与更新",
    9: "患者获益与异地就医",
}


def load_tokenizer():
    spec = importlib.util.spec_from_file_location("tm", ROOT / "scripts" / "07_topic_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tokenize


def load_docs():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", dtype=str)
    df = df[df["rating"].astype(float).notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()
    tokenize = load_tokenizer()
    df["tokens"] = df["text"].map(tokenize)
    return df.reset_index(drop=True)


def shares(W):
    """每条评论的主题权重行归一化，得到「份额」；每行和为 1。"""
    return W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)


def top_words(H, terms, n=TOP_N):
    return [list(terms[np.argsort(-H[i])[:n]]) for i in range(H.shape[0])]


def umass(H, terms, Xbin):
    """UMass coherence：越接近 0 越好（负值越小越差）。"""
    t2i = {t: i for i, t in enumerate(terms)}
    df = np.asarray(Xbin.sum(axis=0)).ravel()
    scores = []
    for words in top_words(H, terms):
        idx = [t2i[w] for w in words if w in t2i]
        vals = []
        for i, wi in enumerate(idx):
            for wj in idx[:i]:
                co = float(Xbin[:, wi].multiply(Xbin[:, wj]).sum())
                vals.append(np.log((co + 1.0) / max(df[wj], 1.0)))
        if vals:
            scores.append(float(np.mean(vals)))
    return np.array(scores)


def match(H_from, H_to):
    A = H_from / np.clip(np.linalg.norm(H_from, axis=1, keepdims=True), 1e-9, None)
    B = H_to / np.clip(np.linalg.norm(H_to, axis=1, keepdims=True), 1e-9, None)
    S = A @ B.T
    r, c = linear_sum_assignment(-S)
    return {int(i): int(j) for i, j in zip(r, c)}, {int(i): float(S[i, j]) for i, j in zip(r, c)}


def main():
    df = load_docs()
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
    Xbin = (X > 0).astype(np.int8)
    print(f"样本 {len(df)} 条，词表 {X.shape[1]} 词，k={K}")

    nmf = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=600)
    W_nmf = nmf.fit_transform(X)
    H_nmf = nmf.components_

    lda = LatentDirichletAllocation(
        n_components=K,
        random_state=SEED,
        learning_method="batch",
        max_iter=40,
        evaluate_every=-1,
    )
    W_lda = lda.fit_transform(X)
    H_lda = lda.components_

    coh_nmf = umass(H_nmf, terms, Xbin)
    coh_lda = umass(H_lda, terms, Xbin)

    mapping, cos = match(H_nmf, H_lda)
    tw_nmf = top_words(H_nmf, terms)
    tw_lda = top_words(H_lda, terms)

    S_nmf = shares(W_nmf)
    S_lda = shares(W_lda)
    is1 = df["rating"].values == 1
    is5 = df["rating"].values == 5

    rows = []
    for i in range(K):
        j = mapping[i]
        d_nmf = float(S_nmf[is1, i].mean() - S_nmf[is5, i].mean())
        d_lda = float(S_lda[is1, j].mean() - S_lda[is5, j].mean())
        rows.append({
            "NMF主题": TOPIC_NAME[i],
            "余弦": round(cos[i], 3),
            "NMF一致性": round(float(coh_nmf[i]), 2),
            "LDA一致性": round(float(coh_lda[j]), 2),
            "NMF差值": round(d_nmf, 4),
            "LDA差值": round(d_lda, 4),
            "同号": "是" if np.sign(d_nmf) == np.sign(d_lda) else "否",
            "NMF_top5": " ".join(tw_nmf[i][:5]),
            "LDA_top5": " ".join(tw_lda[j][:5]),
        })
    tab = pd.DataFrame(rows)

    # 医生合并主题的方向
    doc_pair = [i for i, name in TOPIC_NAME.items() if name.startswith("医生") or name.startswith("专家")]
    j_pair = [mapping[i] for i in doc_pair]
    doc_nmf = float(S_nmf[is1][:, doc_pair].sum(1).mean() - S_nmf[is5][:, doc_pair].sum(1).mean())
    doc_lda = float(S_lda[is1][:, j_pair].sum(1).mean() - S_lda[is5][:, j_pair].sum(1).mean())

    n_same = int((tab["同号"] == "是").sum())
    n_match = int((tab["余弦"] >= 0.6).sum())
    n_coh_win = int((coh_nmf > coh_lda[[mapping[i] for i in range(K)]]).sum())
    med_cos = float(np.median(list(cos.values())))
    mean_coh_nmf = float(np.mean(coh_nmf))
    mean_coh_lda = float(np.mean(coh_lda))
    # 稀疏性：单条评论里份额 > 0.1 的主题数，以及份额分布的归一化熵
    conc_nmf = float((S_nmf > 0.1).sum(axis=1).mean())
    conc_lda = float((S_lda > 0.1).sum(axis=1).mean())
    ent_nmf = float(np.mean(-(S_nmf * np.log(np.clip(S_nmf, 1e-12, None))).sum(1)) / np.log(K))
    ent_lda = float(np.mean(-(S_lda * np.log(np.clip(S_lda, 1e-12, None))).sum(1)) / np.log(K))

    tab.to_csv(TABLE_OUT, index=False, encoding="utf-8-sig")

    lines = [
        "# 算法对照：NMF vs LDA",
        "",
        "> 目的：回答「为什么用 NMF 而不是 LDA」。只换分解算法，其他全部不动——同一批 jieba 分词与停用词表、"
        f"同一 TF-IDF 矩阵（min_df=5, max_df=0.5, sublinear_tf）、k={K}、random_state={SEED}。",
        "",
        f"样本 {len(df)} 条，词表 {X.shape[1]} 个词。LDA 同样拟合在 TF-IDF 上（而非词频矩阵），"
        "以便把差异归因于算法本身；代价是偏离 LDA 的多项式生成假设，这个偏离要在方法部分说明。",
        "",
        "## 1. 一句话结论",
        "",
        f"换算法不推翻主要结论，但会改变「主题颗粒度」：{n_same}/10 个主题方向同号，"
        "可一一对应的主题只有少数几个。选 NMF 的理由是可解释性，不是它更「准」。",
        "",
        f"- **主题内部一致性**：NMF 平均 UMass = {mean_coh_nmf:.2f}，LDA = {mean_coh_lda:.2f}"
        f"（越接近 0 越好）；{n_coh_win}/10 个主题 NMF 更连贯。差距不大，不足以单凭这一点选型。",
        f"- **主题可对应性**：匈牙利一一指派后，只有 {n_match}/10 个 NMF 主题能找到余弦 ≥ 0.6 的 LDA 对应主题"
        f"（余弦中位数 {med_cos:.2f}）。LDA 在 TF-IDF 上给出的主题更泛化、词更散，"
        "与 NMF 的细分主题（活动/付费/故障/客服）无法一一对上。**这是选 NMF 的实际理由：颗粒度更细、更好命名。**",
        f"- **结论复现**：{n_same}/10 个主题的「1 星 − 5 星」份额差在 LDA 下同号；"
        f"医生合并主题 NMF {doc_nmf:+.3f} vs LDA {doc_lda:+.3f}，活动议题 NMF "
        f"{tab.loc[tab['NMF主题'] == '活动打卡与 AI 识别', 'NMF差值'].iloc[0]:+.3f} vs LDA "
        f"{tab.loc[tab['NMF主题'] == '活动打卡与 AI 识别', 'LDA差值'].iloc[0]:+.3f}。"
        "LDA 下量级普遍更小，因为它把权重摊到更多主题上——这正是「原始差值含算法成分」的又一证据。",
        f"- **翻号的 {10 - n_same} 个**：" + "、".join(tab.loc[tab["同号"] == "否", "NMF主题"].tolist()) +
        "——正好是稳定性检验里边界或漂移的主题（`stability.md`），说明弱效应本身不可靠，"
        "而不是两个算法在打架。强效应（医生、活动、付费、App 功能、技术故障、免费咨询）方向全部一致。",
        f"- **稀疏性：不能作为选型理由**。两个指标方向相反——单条评论里份额 > 0.1 的主题数 NMF {conc_nmf:.2f} vs LDA {conc_lda:.2f}，"
        f"份额归一化熵 NMF {ent_nmf:.2f} vs LDA {ent_lda:.2f}（1 = 完全摊平）。"
        "LDA 是「少数主题拿大头 + 一堆极小权重」，NMF 是「两三个主题各拿三成」，谁更稀疏取决于怎么定义。"
        "所以面试里不要说「NMF 更稀疏」，只说「NMF 的主题切得更细、能对上 codebook」。",
        "",
        "## 2. 逐主题对照",
        "",
        "| NMF 主题 | 余弦 | NMF 一致性 | LDA 一致性 | NMF 差值 | LDA 差值 | 同号 | LDA top5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in tab.iterrows():
        lines.append(
            f"| {r['NMF主题']} | {r['余弦']:.3f} | {r['NMF一致性']:.2f} | {r['LDA一致性']:.2f} | "
            f"{r['NMF差值']:+.4f} | {r['LDA差值']:+.4f} | {r['同号']} | {r['LDA_top5']} |"
        )
    lines += [
        "",
        "差值 = 1 星平均份额 − 5 星平均份额；份额为单条评论主题权重的行归一化。",
        "",
        "## 3. 已补做与仍未做的对照",
        "",
        "- **BERTopic / 嵌入类方法**：已补做，见 `outputs/bertopic_check.md`。"
        "本地 bge-small-zh-v1.5 → UMAP → HDBSCAN，自动得到 33 个簇、26.7% 噪声；"
        "「两端对应不同语义簇」与「技术故障簇以 2–4 星为主」都复现了。"
        "但与 NMF 的 10 个主题不是一一映射，不能做主题级数值对比。",
        "- **LDA 拟合在词频而非 TF-IDF**：会影响与本文的可比性，未做。",
        "- **人工主题命名的一致性**：10 个主题由单人命名，没有第二人复核（κ 未计算）。",
        "",
        "## 4. 怎么在面试里用",
        "",
        "被问「凭什么选 NMF」时，按这个顺序答：",
        "1. 结论层面：换 LDA 后 " + f"{n_same}/10 个主题方向不变，医生合并主题方向也不变——所以选型不影响结论。",
        "2. 方法层面：NMF 的 10 个主题能一一映射到 codebook（活动/付费/故障/客服…），"
        "LDA 的主题更泛化，命名不到这一层，也就无法向产品团队说清「该修哪里」。",
        "3. 诚实层面：没做第二人编码（κ 未知）、LDA 用的是 TF-IDF 而非词频矩阵；这两条是明确的 next step。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"NMF 一致性 {mean_coh_nmf:.2f} / LDA {mean_coh_lda:.2f}")
    print(f"对应 {n_match}/10，方向同号 {n_same}/10")
    print(tab[["NMF主题", "余弦", "NMF差值", "LDA差值", "同号"]].to_string(index=False))
    print(f"医生合并：NMF {doc_nmf:+.4f} / LDA {doc_lda:+.4f}")
    print(f"写出 {REPORT.name} 与 {TABLE_OUT.name}")


if __name__ == "__main__":
    main()
