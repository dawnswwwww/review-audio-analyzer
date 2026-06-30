"""Jinja2 prompt templates for the LLM calls.

The system prompt is constant across calls; the user prompt is rendered from
the per-call data. All prompts are in Chinese and require JSON output.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)

ANALYZE_PAIR_SYSTEM = (
    "你是一位资深的技术面试官，正在评审一位候选人的面试表现。"
    "你要逐题分析一道「面试官提问 → 候选人回答」的片段。\n\n"
    "严格要求：\n"
    "1. 严格基于提供的文本做出判断；不要凭空补充未提及的内容。\n"
    "2. 评价要具体、客观、可操作；避免空泛的「不错」「一般」等评价。\n"
    "3. 知识点的抽取要忠于原文中实际出现或明确暗示的概念；避免编造。\n"
    "4. 严格按 JSON 格式输出，字段名和类型与给出的 schema 一致。"
)

ANALYZE_PAIR_USER_TEMPLATE = """## 面试官提问
{{ question }}

## 候选人回答
{{ answer }}

## 任务
请以 JSON 格式输出以下字段：
- `question_summary`（字符串）：用一句话概括这道题在考察什么知识点。
- `answer_evaluation`（字符串）：2-4 句评价候选人的回答（正确性 / 思路 / 完整性）。
- `knowledge_points`（字符串数组）：这道题涉及的知识点（候选词 / 术语）。
- `highlights`（字符串数组）：候选人回答中的亮点（可为空数组）。
- `weaknesses`（字符串数组）：候选人回答中的不足或错误（可为空数组）。
- `improvement_suggestions`（字符串）：如何更好地回答这道题（1-2 句）。

仅输出 JSON，不要任何额外文本。
"""


def analyze_pair_prompt(*, question: str, answer: str) -> str:
    """Render the user prompt for analyzing a single Q+A pair."""
    tmpl = _ENV.from_string(ANALYZE_PAIR_USER_TEMPLATE)
    return tmpl.render(question=question, answer=answer)


AGGREGATE_SYSTEM = (
    "你是一位资深的技术面试官，要根据多道「逐题分析」的结果，"
    "给出整场面试的总评、亮点、不足，并整理一份去重、带解释的「知识点总结」。\n\n"
    "严格要求：\n"
    "1. 总评分要分维度（技术深度、思路结构、表达清晰度、反应速度），每项 0-100。\n"
    "2. 整体得分是上述维度的加权平均（你决定权重，但要合理）。\n"
    "3. 知识点要合并同义项（如「红黑树」和「RB-Tree」是同一个），按类别分组，"
    "给出 2-4 句的中文解释和 1-3 条拓展学习方向。\n"
    "4. 亮点 / 不足要去重、按主题归类。\n"
    "5. 严格按 JSON 格式输出，字段名和类型与给出的 schema 一致。"
)

AGGREGATE_USER_TEMPLATE = """## 候选人简历背景（可为空）
{{ candidate_background or "未提供" }}

## 整场面试的逐题分析
{% for p in pair_analyses %}
### 第 {{ p.pair_index }} 题
- 题目摘要：{{ p.question_summary }}
- 答案评价：{{ p.answer_evaluation }}
- 涉及知识点：{{ p.knowledge_points | join("、") }}
- 亮点：{{ p.highlights | join("；") }}
- 不足：{{ p.weaknesses | join("；") }}
- 改进建议：{{ p.improvement_suggestions }}
{% endfor %}

## 任务
请以 JSON 格式输出：
- `overall_evaluation`（字符串）：3-5 句整场面试总评。
- `overall_score`（整数 0-100)：整体得分。
- `dimension_scores`（对象）：{"技术深度": int, "思路结构": int, "表达清晰度": int, "反应速度": int}。
- `consolidated_highlights`（字符串数组）：合并去重后的亮点。
- `consolidated_weaknesses`（字符串数组）：合并去重后的不足。
- `consolidated_knowledge_points`（对象数组）：每个对象含
  `name`（规范化名称）, `category`（类别，如 数据结构 / 算法 / 系统设计 / 编程语言 / 计算机网络 / 操作系统 / 数据库 / 其它）,
  `explanation`（2-4 句中文解释）, `learning_directions`（1-3 条拓展学习方向）。

仅输出 JSON，不要任何额外文本。
"""


def aggregate_prompt(*, pair_analyses: list[dict[str, Any]], candidate_background: str) -> str:
    """Render the user prompt for the aggregator."""
    tmpl = _ENV.from_string(AGGREGATE_USER_TEMPLATE)
    return tmpl.render(
        pair_analyses=pair_analyses,
        candidate_background=candidate_background,
    )


SUMMARIZE_PAIRS_SYSTEM = (
    "你是一位助手，负责将多道「逐题分析」压缩成简短形式，"
    "保留每道题的题目摘要、答案评价一行、知识点列表。严格输出 JSON 数组。"
)

SUMMARIZE_PAIRS_USER_TEMPLATE = """## 待压缩的逐题分析
{% for p in pair_analyses %}
### 第 {{ p.pair_index }} 题
- 题目摘要：{{ p.question_summary }}
- 答案评价：{{ p.answer_evaluation }}
- 涉及知识点：{{ p.knowledge_points | join("、") }}
{% endfor %}

## 任务
请以 JSON 数组输出，每个元素是一个对象：
- `pair_index`（整数）
- `question_summary`（字符串）
- `answer_evaluation`（字符串，一句话）
- `knowledge_points`（字符串数组）

仅输出 JSON 数组，不要任何额外文本。
"""


def summarize_pairs_prompt(*, pair_analyses: list[dict[str, Any]]) -> str:
    """Render the user prompt for the token-overflow summarizer (Prompt 2.5)."""
    tmpl = _ENV.from_string(SUMMARIZE_PAIRS_USER_TEMPLATE)
    return tmpl.render(pair_analyses=pair_analyses)


CLEAN_TRANSCRIPT_SYSTEM = (
    "你是一位技术面试转写校对员。原始音频已经过 ASR（语音识别）转写，"
    "但 ASR 在处理技术术语时经常出错（音近字误识别），例如：\n"
    "- 英文技术术语被听成中文近音字（如 \"BUD\" → \"Vue\"、\"雷啊特\" → \"React\"、\"撕扣\" → \"Sequelize\"）\n"
    "- 中文技术术语被听成错误字（如 \"灵火\" 可能本意是 \"流火 / 零和 / 林火\"，但要靠上下文判断）\n"
    "- 中英混说时，ASR 可能把英文术语识别成无意义的中文词\n\n"
    "你的任务：\n"
    "1. **只修术语** —— 修正音近字误识别的技术术语（人名、框架名、算法名、协议名、库名、概念名）。\n"
    "2. **保留原意** —— 不润色、不改写、不补全内容。候选人的口语化、语法错误、停顿都保留。\n"
    "3. **靠上下文判断** —— 利用整段对话的主题，判断哪些词可能是技术术语。\n"
    "4. **保守原则** —— 不确定时宁可保留原文；不要凭空猜。\n"
    "5. **记录改动** —— 对每处修改，在 `corrections` 字段里说明 `原词 → 改后词：依据`。\n"
    "6. **不修改其他字段** —— `speaker` / `start` / `end` 必须保持原样，只改 `text`。"
)

CLEAN_TRANSCRIPT_USER_TEMPLATE = """## 候选人面试方向
{{ domain }}

## 待校对的转写（JSON 数组）
{{ transcript_json }}

## 任务
对每条 utterance 校对 `text` 字段，输出相同结构的 JSON 数组：

- `index`（整数）：原始数组的下标
- `speaker`（字符串）：与原文一致，不要修改
- `start`（数字）：与原文一致，不要修改
- `end`（数字）：与原文一致，不要修改
- `text`（字符串）：校对后的文本；若无修改则与原文相同
- `corrections`（字符串数组）：每条形如 "原词 → 改后词：依据"；若无修改则为空数组

严格按 JSON 输出，不要任何额外文本。
"""


def clean_transcript_prompt(*, domain: str, transcript_json: str) -> str:
    """Render the user prompt for transcript proofreading."""
    tmpl = _ENV.from_string(CLEAN_TRANSCRIPT_USER_TEMPLATE)
    return tmpl.render(domain=domain, transcript_json=transcript_json)