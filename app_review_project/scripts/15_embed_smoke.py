"""P4 嵌入模型连通性测试：把本地 bge-small-zh-v1.5 跑通。

只做三件事，不接 BERTopic：
  1. 从本地目录加载模型（不走网络，显式 HF_HUB_OFFLINE=1）
  2. 编码全部 1,681 条评论，确认维度、耗时与内存占用可接受
  3. 用几组「语义应该相近 / 应该无关」的评论对做合理性检查

为什么先做这一步：BERTopic 的失败大多出在嵌入环节（模型加载、维度、中文分词器），
先用最小代价把这一段跑通，再决定要不要接 UMAP + HDBSCAN。

用法：
    python scripts/15_embed_smoke.py
输出：data/embeddings_bge.npy、data/embeddings_bge_index.csv、outputs/embed_smoke.md
"""
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "bge-small-zh-v1.5"
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# (sentence_a, sentence_b, 期望关系)
PAIRS = [
    ("减重活动喝彩越来越难，而且故意不识别", "打卡任务根本识别不了，白做了", "相近"),
    ("体检报告一堆专业术语看不懂，拍照上传就能解读", "上传体检单，它会告诉你哪里有问题", "相近"),
    ("客服电话打不通，退款拖了半个月", "联系客服一直没人接，钱也不退", "相近"),
    ("更新之后一直闪退，进不去", "打开就崩溃，更新完就废了", "相近"),
    ("医生很耐心，解答得很清楚", "大夫专业，问诊回复细致", "相近"),
    ("减重活动喝彩越来越难，而且故意不识别", "医生很耐心，解答得很清楚", "无关"),
    ("体检报告一堆专业术语看不懂", "更新之后一直闪退，进不去", "无关"),
]


def main():
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(DATA / "reviews_analysis.csv", encoding="utf-8-sig", dtype=str)
    df["text"] = (df["title"].fillna("") + "。" + df["content"].fillna("")).str.strip()
    texts = df["text"].tolist()
    print(f"加载模型：{MODEL}")
    t0 = time.time()
    model = SentenceTransformer(str(MODEL), device="cpu")
    t_load = time.time() - t0
    dim = model.get_embedding_dimension()
    print(f"加载完成 {t_load:.1f}s，维度 {dim}")

    t0 = time.time()
    emb = model.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True
    )
    t_enc = time.time() - t0
    print(f"编码 {len(texts)} 条耗时 {t_enc:.1f}s（{len(texts) / t_enc:.0f} 条/秒），形状 {emb.shape}")

    pd.DataFrame({"review_id": df["review_id"], "app_name": df["app_name"], "rating": df["rating"]}).to_csv(
        DATA / "embeddings_bge_index.csv", index=False, encoding="utf-8-sig"
    )
    np.save(DATA / "embeddings_bge.npy", emb.astype(np.float32))
    print(f"已保存 {DATA / 'embeddings_bge.npy'}（{emb.nbytes / 1e6:.1f} MB）")

    # 合理性检查：同一批文本的相似度矩阵
    rows = []
    for a, b, expect in PAIRS:
        va, vb = model.encode([a, b], normalize_embeddings=True, show_progress_bar=False)
        cos = float(np.dot(va, vb))
        rows.append({"句A": a, "句B": b, "期望": expect, "余弦": round(cos, 3)})
    tab = pd.DataFrame(rows)

    pos = tab[tab["期望"] == "相近"]["余弦"]
    neg = tab[tab["期望"] == "无关"]["余弦"]
    ok = bool(pos.min() > neg.max())

    lines = [
        "# 嵌入模型连通性测试",
        "",
        f"模型：`BAAI/bge-small-zh-v1.5`，本地路径 `models/bge-small-zh-v1.5`，维度 **{dim}**，CPU 推理。",
        f"加载 {t_load:.1f}s；编码 {len(texts)} 条 {t_enc:.1f}s（约 {len(texts) / t_enc:.0f} 条/秒）。"
        f"向量已存为 `data/embeddings_bge.npy`（float32，{emb.nbytes / 1e6:.1f} MB）。",
        "",
        "## 合理性检查",
        "",
        "| 期望 | 句 A | 句 B | 余弦 |",
        "|---|---|---|---|",
    ]
    for _, r in tab.iterrows():
        lines.append(f"| {r['期望']} | {r['句A']} | {r['句B']} | {r['余弦']:.3f} |")
    lines += [
        "",
        f"「相近」组最小余弦 {pos.min():.3f}，「无关」组最大余弦 {neg.max():.3f}，"
        + ("**分组边界清晰**，模型能区分语义相近与无关的评论。" if ok else "**分组有重叠**，模型对本数据的区分力不足，需换更大模型或微调。"),
        "",
        "## 下一步的取舍",
        "",
        "- 全量 1,681 条 × 512 维是 3.4 MB，内存无压力，可以直接进 UMAP。",
        "- BERTopic 默认 `UMAP(n_neighbors=15, n_components=5)` + HDBSCAN，在 1.7k 短文本上通常会产出大量噪声点（-1 类）；"
        "若要严格对齐 k=10，需要用 `nr_topics=10` 或先 KMeans 定簇。",
        "- **这条线只能作为「换算法是否改变结论」的补充证据**：嵌入聚类出来的主题与 NMF 主题不是一一对应关系，"
        "比较的是「星级—主题份额差的方向是否一致」，不是逐主题对齐。",
        "",
    ]
    (OUT / "embed_smoke.md").write_text("\n".join(lines), encoding="utf-8")
    print(tab.to_string(index=False))
    print(f"分组边界清晰：{ok}（相近 min {pos.min():.3f} / 无关 max {neg.max():.3f}）")
    print(f"写出 {OUT / 'embed_smoke.md'}")


if __name__ == "__main__":
    main()
