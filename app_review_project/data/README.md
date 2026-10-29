# data/ 目录说明

本目录**只公开聚合与派生数据，不分发评论全文**。

## 为什么不分发全文

评论来自 Apple App Store 中国区公开 RSS。虽然接口公开，但评论正文可能包含用户自述的症状、用药、就医经历等敏感信息；即使 RSS 未提供用户身份，逐条转载仍可能造成间接识别。因此仓库只保留：

- **聚合结果**：主题份额、系数、p 值、稳定性指标、簇分布；
- **不含正文的逐条标记**：`review_id`、App、星级、月份、主题标签。

`review_id` 为 App Store 评论的公开 ID，可用于校验，但无法据此获得正文。

## 公开文件

| 文件 | 内容 | 由哪个脚本生成 |
|---|---|---|
| `app_metadata.csv` | 8 个 App 的官方评分、评分人数、版本、App Store 链接 | `02_collect_full.py` |
| `topic_topwords.csv` | k=6–16 各主题的 top 词与重建误差 | `07_topic_model.py` |
| `topic_compare.csv` | 10 个主题的 1 星/5 星份额、差值、三种检验的 p 值与 BH 校正 | `08_topic_compare.py` |
| `labels_keyword.csv` | 逐条关键词法标签（`review_id` + 主题 + AI 标记，**无正文**） | `05_keyword_label.py` |
| `stability_matches.csv` | 多种子 / Bootstrap / 跨 k 的主题匹配余弦与符号一致性 | `10_stability.py` |
| `lda_vs_nmf.csv` | NMF 与 LDA 主题的一一指派、余弦与差值对照 | `12_algorithm_check.py` |
| `heterogeneity.csv` | 四类产品内 1 星 − 5 星差值 | `17_heterogeneity.py` |
| `heterogeneity_fe.csv` | 三种固定效应口径（App+年 / App×年 / App×月）的系数与 p 值 | `17_heterogeneity.py` |
| `heterogeneity_shape.csv` | 分组后的星级剖面（仅供参考） | `17_heterogeneity.py` |
| `bertopic_topics.csv` | 33 个簇的规模、星级构成、top 词、中段占比 | `16_bertopic.py` |
| `bertopic_info.csv` | 簇的 UMAP/HDBSCAN 诊断信息 | `16_bertopic.py` |
| `embeddings_bge_index.csv` | 句向量矩阵的行顺序（`review_id` + App + 星级，**无正文**） | `15_embed_smoke.py` |
| `info_content.csv` | 逐条信息含量标记（`review_id` + App + 星级 + 月 + 中文字数 + `has_num` / `has_entity`，**无正文**） | `22_info_content.py` |

## 未公开文件（本地保留）

`reviews_raw.csv`、`reviews_clean.csv`、`reviews_analysis.csv`、`topic_doc_weights.csv`、`sample_200.csv`、`coding_A.csv`、`coding_B.csv`、`coding_worksheet.csv`、`ai_codes.csv`、`labels_pilot_manual.csv` 等含评论正文，已列入 `.gitignore`。

## 如何复现

```powershell
# 1. 重新采集（需联网；RSS 只提供最近约 10 页，结果会随日期变化）
python scripts/02_collect_full.py
python scripts/03_clean.py
python scripts/05_keyword_label.py
python scripts/07_topic_model.py

# 2. 分析（不依赖网络）
python scripts/08_topic_compare.py
python scripts/09_robustness.py
python scripts/10_stability.py
python scripts/11_figures.py
python scripts/12_algorithm_check.py
python scripts/13_ai_read.py
python scripts/17_heterogeneity.py
python scripts/22_info_content.py

# 3. 句向量与 BERTopic（需 Python 3.11 + torch，见根 README）
python scripts/14_download_embedding.py
python scripts/15_embed_smoke.py
python scripts/16_bertopic.py
```

> 注：`02_collect_full.py` 抓取的是「当日最近若干页」，与本文分析的 1,681 条不完全相同；正文级复现需自行采集。聚合结果的复现不受影响的前提是使用同一份本地数据。

### 关于重采集的结果差异

`02_collect_full.py` 抓的是「执行当天」App Store 返回的最近若干页，而 RSS 的可用页数会随时间漂移：实测部分 App 深页返回的是 2016–2021 年的评论，部分 App 十页全空。因此**重新采集得到的样本与本文的 1,681 条必然不同，重跑后的数字也会与仓库内 CSV 不一致**。

正确用法是：

- **核对数字** → 直接用仓库内已提交的聚合 CSV（`topic_compare.csv`、`heterogeneity_fe.csv`、`stability_matches.csv` 等），它们对应本文的分析样本；
- **验证流程** → 重采集 + 重跑脚本只用于确认流水线能走通，不要拿新结果与本文数字对比。

正文级数据的复现需要自行采集并用同一流程重算，仓库不提供评论全文。

## 合规

- 只使用公开 RSS 与 iTunes Lookup API，不抓取需登录或非公开页面；
- 不保存、不分发用户可识别信息；
- 报告仅呈现聚合结果与已匿名化的短引文。
