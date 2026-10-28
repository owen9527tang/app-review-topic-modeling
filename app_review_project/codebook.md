# Codebook v0.2：互联网医疗 App 评论编码

> v0.2 变更（基于 50 条试点）：`other` 拆为 客服/活动营销/商品实物/内容推送；新增 `empty_praise` 标记；删除 `referral_mention`（试点 50 条仅命中 1 条）；`ai_context` 新增 `competitor_compare`。

## 0. 编码单位
一条评论 = 标题 + 正文。星级只作为上下文，不作为标签依据。

## 1. 基本原则
- 只编码评论中**明确出现**的内容，不推断。
- 同一条评论可以同时有 `complaint_themes` 和 `praise_themes`。
- 只提主题、没有评价（如“我在京东买药”）→ 不编码为抱怨或表扬。
- 泛泛的“垃圾/差评/骗子”但没有具体对象 → `complaint_themes=["other"]`，`overall_sentiment=-1`。
- 广告、刷评、无意义文本 → 两个主题列表都为空，`overall_sentiment=0`。
- **4 星的功能建议**（“如果能……会更方便”）→ 可编码 `complaint_themes`，但 `overall_sentiment=0`。
- **5 星但只有空泛好评**（“这软件太好了”“确实挺喜欢”）→ `praise_themes=[]`，`empty_praise=1`。

## 2. 主题（complaint / praise 共用）

| key | 名称 | 抱怨示例 | 表扬示例 |
|---|---|---|---|
| `fee` | 收费/退款 | 乱收费、退款不到账、自动扣费、药太贵、问诊次数受限 | 价格透明、退款快、性价比高 |
| `doctor` | 医生质量/响应 | 医生不回复、敷衍、态度差、不专业、偏袒医生 | 医生专业、回复快、解释清楚 |
| `delivery` | 药品配送 | 发货慢、送错药、物流不更新、包装破损 | 送药快、配送准时 |
| `prescription` | 处方/用药 | 处方不通过、开错药、剂量不对、副作用 | 用药指导清楚、处方审核快 |
| `appointment` | 预约/挂号 | 挂不上号、号源虚假、预约被取消、候诊太久、抢号加验证 | 挂号方便、预约顺利、复诊提醒 |
| `tech` | 技术/账号 | 闪退、卡顿、登录失败、验证码收不到、支付失败、功能被砍、设备连不上 | 版本流畅、操作简单、界面美观 |
| `privacy` | 隐私/安全 | 信息泄露、骚扰电话、强制实名、授权过多 | 隐私保护清楚、安全可靠 |
| `service` | 客服/售后 | 客服不回复、踢皮球、态度差、无投诉渠道 | 客服响应快、问题处理到位 |
| `promotion` | 活动/营销 | 活动虚假、提现失败、返现不到账、拉新奖励不兑现、商业化过重 | 活动实惠、补贴划算、打卡有用 |
| `goods` | 商品/实物 | 商品质量差、赠品不发货、实物设备难用、体检权益缩水 | 商品质量好、赠品实用 |
| `content` | 内容/推送 | 广告软文、推送骚扰、内容审核不当、文章消失、立场争议 | 内容专业、科普有用 |
| `other` | 其他（兜底） | 仅当确实无法归入上述任一类时使用 | 同左 |

## 3. AI 标记
- `ai_mention`：0/1。出现 AI、人工智能、智能助手、智能医生、大模型、DeepSeek、ChatGPT、GPT、豆包、文心一言、通义千问、讯飞、智能问诊、AI问诊、AI医生、AI助手、智能客服，或“客服像机器人”这类比喻。
- `ai_context`：
  - `feature_use`：实际使用本 App 的 AI 功能（如“AI 问诊很方便”）
  - `feature_trust`：对本 App AI 的信任/不信任（如“AI 看病不靠谱”）
  - `feature_failure`：本 App 的 AI 功能出错（如“AI 识别不灵”“AI 开错药”）
  - `competitor_compare`：拿其他 AI 产品做对比（如“豆包从来不掉线”）
  - `metaphor`：只是修辞比喻，不是真 AI 功能（如“客服像机器人”）
  - `none`：未提及
- `ai_sentiment`：-1 负面，0 中性/混合，1 正面。评价对象是被提及的那个 AI 实体。

## 4. 空泛好评标记
- `empty_praise`：0/1。仅对 5 星评论判定：`praise_themes` 为空或只有泛泛夸奖时记 1；有具体表扬对象记 0。其余星级一律 0。

## 5. 整体情绪
- `overall_sentiment`：-1 负面，0 中性/混合/建议，1 正面。以评论的主导语气为准，不直接照抄星级。

## 6. 输出 JSON
```json
{
  "labels": [
    {
      "review_id": "123",
      "complaint_themes": ["fee", "doctor"],
      "praise_themes": [],
      "ai_mention": 0,
      "ai_context": "none",
      "ai_sentiment": 0,
      "empty_praise": 0,
      "overall_sentiment": -1,
      "evidence_quote": "退款一直不到账"
    }
  ]
}
```

## 7. 边界规则
- “医生回复慢，但很专业” → `complaint_themes=["doctor"]`，`praise_themes=["doctor"]`。
- “客服像机器人” → `complaint_themes=["service"]`，`ai_mention=1`，`ai_context="metaphor"`，`ai_sentiment=-1`。
- “AI 问诊很方便，但不敢信” → `ai_mention=1`，`ai_context="feature_trust"`，`ai_sentiment=0`。
- “药送得快，但医生不回复” → `complaint_themes=["doctor"]`，`praise_themes=["delivery"]`。
- “客服不理我，退款也不到” → `complaint_themes=["service","fee"]`。
- “打卡返现不到账” → `complaint_themes=["promotion"]`，不要同时记 `fee`，除非另提到收费金额。
- “买了体脂秤连不上” → `complaint_themes=["goods","tech"]`。
- “主页全是广告和软文” → `complaint_themes=["content"]`。
- “升级后把历史问诊藏了，就是想多赚钱” → `complaint_themes=["tech","promotion"]`。
- 纯情绪“垃圾”“差评” → `complaint_themes=["other"]`，`overall_sentiment=-1`。

## 8. 人工编码：200 条抽样（可核实性 + 主议题）

用于核验「5 星更空泛」这一条，**只编码 200 条分层抽样（`data/sample_200.csv`）**，不编码全量。
编码表在 `data/coding_worksheet.csv`，判定只看文字、不看星级。

### 8.1 `spec`（文本具体性，三值）
- `2` 可核实细节：出现金额、次数、时长、日期、具体功能名、具体症状/药品、具体操作步骤。例：「挂号费 99 到 199」「更新后闪退，点了三次都退」「吃完第二天血压 150」。
- `1` 有明确对象但无可核实细节：能说清在谈什么，但没有可核对的量或步骤。例：「客服态度差」「医生有点敷衍」「药太贵」。
- `0` 无具体对象：只有情绪、口号或空泛夸奖。例：「垃圾软件」「太好了，推荐」「这软件真不错」。

### 8.2 `primary`（主议题，单选）
从第 2 节的 11 个 key + `other` 中选一个，取**占篇幅最多**的议题；若全篇无具体对象则记 `other`。
一条评论谈多件事时，只记最主要的那个，不做多标签——这一列的目的是核对「谈什么」，不是重建全量权重。

### 8.3 一致性口径
- 必须由**两名编码者独立**完成（A 填 `spec_A`/`primary_A`，B 填 `spec_B`/`primary_B`），A 只解释规则本身，不给判例提示。
- 具体性用**二次加权 Cohen's κ**（三值有序），主议题用普通 κ。
- 参考区间：<0.20 差，0.21–0.40 一般，0.41–0.60 中等，0.61–0.80 好，>0.80 很好；**低于 0.6 先改手册边界，不改数字**。
- 只有一名编码者时，只报「人工 vs 机械基线」的一致率，不报 κ、也不写「一致性良好」。

### 8.4 机检
`scripts/18_manual_check.py` 生成并校验编码表、计算机械基线与 κ，输出 `outputs/coding_check.md`。
