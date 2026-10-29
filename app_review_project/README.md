# 互联网医疗 App 的用户抱怨与评分-议题结构

用 8 个中国互联网医疗 App 的 1,681 条 App Store 公开评论，做中文短文本主题建模（jieba + TF-IDF + NMF），
检验「评分」与「用户谈什么」之间的关系。**不做逐条人工标注**：极性由星级分组承担，主题模型只负责发现议题。

核心结论：评分与议题不是单调关系，而是**分段结构**——1 星集中在活动返利与付费问诊/退款，
2–4 星集中在技术故障与 App 功能，5 星集中在医生与专家口碑。详见 `outputs/memo.md` 与 `outputs/topic_model.md`。

## 目录

```
app_review_project/
├── scripts/                 # 01 → 23，按序号执行即可复现
├── data/                    # 原始与派生数据（评论全文不对外分发）
├── models/                  # 本地嵌入模型 bge-small-zh-v1.5（96 MB，不入 git）
├── outputs/                 # 报告、图、演示、memo
│   ├── topic_model.md       # 主报告（10 节 + 附图清单）
│   ├── robustness.md        # 刷评/日期集中度排查 + 官方评分基准
│   ├── stability.md         # 主题稳定性：多种子 / Bootstrap / 跨 k
│   ├── algorithm_check.md   # 算法对照：NMF vs LDA（一致性 / 可对应性 / 结论复现）
│   ├── embed_smoke.md       # 嵌入模型连通性与合理性检查
│   ├── bertopic_check.md    # 范式对照：句向量 + UMAP + HDBSCAN（簇分布与星级倾向）
│   ├── ai_read.md           # 68 条提到 AI 的评论定性精读（编码分布与原话）
│   ├── heterogeneity.md     # 产品类型异质性 + App×年 / App×月 固定效应
│   ├── coding_check.md      # 抽样人工编码核查（负结果：两轮两版手册最低 κ 0.164，不采用）
│   ├── info_content.md      # 延伸：评分 × 可核对信息量（数量词 / 对象词 + 字数分层）
│   ├── info_content_audit.md # 规则指标的 30 条人工抽查（precision / specificity）
│   ├── memo.md              # 1 页 memo
│   ├── figures/             # 9 张图（PNG）
│   └── app-review-deck/     # 10 页 HTML 演示（app-review-deck.html）
├── codebook.md              # codebook v0.2（11 类主题 + other）
└── data_quality.md 等见 outputs/
```

## 流水线

| 脚本 | 作用 | 主要输出 |
|---|---|---|
| `01_collect_pilot.py` | 试点采集（每个 App 1 页） | `data/pilot_raw.json`、`outputs/pilot_report.md` |
| `02_collect_full.py` | 全量采集 8 个 App × 4–5 页 + 官方评分元数据 | `data/reviews_raw.csv`、`data/app_metadata.csv` |
| `03_clean.py` | 去重、过滤、爆发日标记 | `data/reviews_clean.csv`、`data/reviews_analysis.csv`、`outputs/data_quality.md` |
| `04_label_deepseek.py` | 早期 LLM 标注（已不用于主分析，保留对照） | `data/labels_pilot_manual.csv` |
| `05_keyword_label.py` | 关键词法全量粗标（用于交叉验证） | `data/labels_keyword.csv` |
| `06_make_sample.py` | 200 条分层抽样（供人工抽查） | `data/sample_200.csv` |
| `07_topic_model.py` | 分词 + TF-IDF + NMF，k=6–16 扫描 | `data/topic_topwords.csv`、`data/topic_doc_weights.csv` |
| `08_topic_compare.py` | 主分析：两端对比、形状检验、三种口径固定效应、分 App、文本具体性、AI | `outputs/topic_model.md`、`data/topic_compare.csv` |
| `09_robustness.py` | 爆发日排查、活动议题日期集中度与逐层剔除、样本与官方评分基准 | `outputs/robustness.md` |
| `10_stability.py` | 多种子（10）、Bootstrap（50，重拟合 TF-IDF）、跨 k=8/12/14 | `outputs/stability.md`、`data/stability_matches.csv` |
| `11_figures.py` | 8 张图 | `outputs/figures/*.png` |
| `12_algorithm_check.py` | NMF vs LDA：UMass 一致性、主题一一指派、结论方向复现 | `outputs/algorithm_check.md`、`data/lda_vs_nmf.csv` |
| `13_ai_read.py` | 68 条 AI 评论定性编码（10 类主码），输出分布与原话 | `outputs/ai_read.md`、`data/ai_codes.csv` |
| `14_download_embedding.py` | 从 ModelScope 下载 bge-small-zh-v1.5（11 个文件，96 MB） | `models/bge-small-zh-v1.5/` |
| `15_embed_smoke.py` | 本地加载嵌入模型、编码全部评论、相似度合理性检查 | `data/embeddings_bge.npy`、`outputs/embed_smoke.md` |
| `16_bertopic.py` | 句向量 + UMAP + HDBSCAN 主题发现，簇的星级倾向与噪声诊断 | `outputs/bertopic_check.md`、`data/bertopic_topics.csv` |
| `17_heterogeneity.py` | 四类产品分组的方向一致性；App×年、App×月 固定效应与有效识别单元诊断 | `outputs/heterogeneity.md`、`data/heterogeneity*.csv` |
| `18_manual_check.py` | 读取两份盲表、计算机械基线与 Cohen's κ | `outputs/coding_check.md` |
| `19_make_coding_sheets.py` | 从 200 条抽样取 100 条，生成两份互盲编码表（B 行序打乱、无星级列） | `data/coding_A.csv`、`data/coding_B.csv` |
| `20_make_coding_tool.py` | 把两份编码表打包成离线 HTML 点选工具（进度存 localStorage，一键导出 CSV） | `data/coding_tool_A.html`、`data/coding_tool_B.html` |
| `21_merge_b.py` | 把 B 的 12 细项机械映射为 6 组；`--merge` 时拼上新的 spec | `data/coding_B_groups.csv`、`data/coding_B.csv` |
| `22_info_content.py` | 信息含量三指标 × 三检验：原始差异、字数四档分层 + MH 合并 OR、条件在「谈到对象」 | `outputs/info_content.md`、`data/info_content.csv`、`fig9` |
| `23_audit_info_content.py` | 抽 30 条复核规则指标的标记对错，算 precision / specificity | `outputs/info_content_audit.md` |

## 复现

```powershell
# 主分析（Python 3.14 环境即可）
C:\Users\ASUS\miniconda3\python.exe scripts\02_collect_full.py   # 需要联网
C:\Users\ASUS\miniconda3\python.exe scripts\03_clean.py
C:\Users\ASUS\miniconda3\python.exe scripts\05_keyword_label.py
C:\Users\ASUS\miniconda3\python.exe scripts\07_topic_model.py
C:\Users\ASUS\miniconda3\python.exe scripts\08_topic_compare.py
C:\Users\ASUS\miniconda3\python.exe scripts\09_robustness.py
C:\Users\ASUS\miniconda3\python.exe scripts\10_stability.py      # 约 1–2 分钟
C:\Users\ASUS\miniconda3\python.exe scripts\11_figures.py
C:\Users\ASUS\miniconda3\python.exe scripts\12_algorithm_check.py # NMF vs LDA 对照
C:\Users\ASUS\miniconda3\python.exe scripts\13_ai_read.py         # AI 评论定性编码
C:\Users\ASUS\miniconda3\python.exe scripts\17_heterogeneity.py   # 产品类型异质性 + App×年/月 FE
C:\Users\ASUS\miniconda3\python.exe scripts\19_make_coding_sheets.py  # 生成两份互盲编码表
C:\Users\ASUS\miniconda3\python.exe scripts\18_manual_check.py         # 机械基线 + κ（填表后重跑）
C:\Users\ASUS\miniconda3\python.exe scripts\22_info_content.py         # 评分 × 可核对信息量（延伸）
C:\Users\ASUS\miniconda3\python.exe scripts\23_audit_info_content.py   # 生成 30 条抽查表（--check 读回）

# 嵌入与 BERTopic 对照：需要单独的 Python 3.11 环境（torch 无 cp314 轮子）
C:\Users\ASUS\miniconda3\envs\topic\python.exe scripts\14_download_embedding.py
C:\Users\ASUS\miniconda3\envs\topic\python.exe scripts\15_embed_smoke.py
C:\Users\ASUS\miniconda3\envs\topic\python.exe scripts\16_bertopic.py
```

依赖：主分析用 Python 3.14 + numpy、pandas、scipy、scikit-learn、statsmodels、jieba、matplotlib；
嵌入与 BERTopic 用 conda 环境 `topic`（Python 3.11）+ torch 2.14.1+cpu、sentence-transformers 6.2.0、
bertopic 0.17.4、jieba。两条线不混用环境，因为 torch 目前没有 Python 3.14 的 Windows 轮子。
固定随机种子（NMF/UMAP/HDBSCAN seed=42，Bootstrap rng seed=42），结果可复现。

## 数据与合规

- 只用 Apple App Store 中国区**公开评论 RSS**（`itunes.apple.com/cn/rss/customerreviews/...`）与 iTunes Lookup API；
  不抓取需登录或非公开页面，不保存用户可识别信息，仓库不分发评论全文。
- 官方评分（`data/app_metadata.csv`）用于披露选择偏差，不用于校正 prevalence。

## 主要局限

- 只代表「愿意写文字评论的用户」：样本均值 2.66 vs 官方加权 4.55，7/8 个 App 低于官方（符号检验 p=7.8e-03）。
- RSS 按时间倒序取页，评论涌入会把样本压到那一段：京东健康 214 条只覆盖 2026-09/10，2026-09 占全样本 19.5%。
- 中间星级仅 184 条且集中在 3 个 App，形状判定依赖控制 App 与年份后的系数。
- 4/10 个主题的边界不稳（会与相邻主题合并或漂移），中段结论按「功能与故障这一族」汇报。
- 样本量有硬上限：RSS 每个 App 最多 10 页；2026-10-27 实测深页返回的多是 2016–2021 年的老评论（微医 p4/p10、微脉 p4/p5），部分 App 深页为空。
  因此 1,681 条是「近期 + 时间干净」条件下的实际上限，不靠老评论凑数量；换 Android 商店数据源属另一次重建，不做。
- 换算法（LDA）7/10 个主题方向同号，换范式（BERTopic）两端与中段结构均复现，但两套主题都无法与 NMF 一一映射，只比对方向。
- 按产品类型分组后 7/8 个议题四类同号，`客服售后与配送体检` 只有 2/4（平台型反向），已降级为弱证据；
  收紧到 App×月后系数几乎不再变化（最大 0.014），但只保留 1,031 条，只能当方向证据。
- 「5 星更空泛」这一维**做不成可靠的人工编码**：两轮、两版手册下最低 κ = 0.164（12 类 0.068、6 组 0.173、
  两版混用 0.271、有对象 vs 空泛二元 0.164），因此不声称人工验证，该实验作为负结果保留。
  空泛程度改用规则指标：字符数中位数 44 vs 36、未命中主题关键词 29.1% vs 15.4%；
  「正文是否含数量词」1 星 30.4% vs 5 星 11.2%（字数四档分层后 MH OR 0.33，CI 0.24–0.45）。
  规则指标的 30 条人工抽查：数量词判「有」6/10 成立、判「无」20/20 成立，对象词 15/17 与 9/13。
  诊断见 `outputs/coding_check.md` 第 5、6 节与 `outputs/info_content_audit.md`。
- 「信息含量」是「有没有可核对细节」的代理，**不是医学质量**：判断建议对不对需要医生金标准，
  本仓库不具备该条件。
- 评分与文本同时产生，全部结果为关联，不是因果。
