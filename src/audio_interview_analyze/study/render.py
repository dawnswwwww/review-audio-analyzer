"""Render a study-guide dict as Markdown."""

from __future__ import annotations

from collections import defaultdict


def render_study_guide(guide: dict) -> str:
    """Render the LLM-produced study guide as a Markdown string.

    Groups knowledge points by category (e.g., "前端框架", "系统设计")
    and within each category lists the per-KP "study card" with the
    required sections.
    """
    sections: list[str] = []
    sections.append(f"# {guide.get('guide_title', '面试学习指南')}")
    sections.append("")

    preface = guide.get("preface", "").strip()
    if preface:
        sections.append(preface)
        sections.append("")

    kps: list[dict] = guide.get("knowledge_points", [])
    if not kps:
        return "\n".join(sections)

    # Group by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for kp in kps:
        by_category[kp.get("category", "其它")].append(kp)

    for category, items in by_category.items():
        sections.append(f"## {category}")
        sections.append("")
        for kp in items:
            sections.extend(_render_kp(kp))
            sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def _render_kp(kp: dict) -> list[str]:
    """Render a single knowledge-point study card."""
    out: list[str] = []
    name = kp.get("name", "未命名")
    out.append(f"### {name}")
    out.append("")

    if kp.get("one_liner"):
        out.append(f"**一句话核心**：{kp['one_liner']}")
        out.append("")

    if kp.get("why_interview"):
        out.append(f"**为什么面试爱问**：{kp['why_interview']}")
        out.append("")

    concepts = kp.get("core_concepts", [])
    if concepts:
        out.append("**核心概念**：")
        for c in concepts:
            out.append(f"- {c}")
        out.append("")

    code = kp.get("code_example", "").strip()
    if code:
        lang = kp.get("code_language", "").strip() or "text"
        out.append(f"**代码示例**（{lang}）：")
        out.append("")
        out.append(f"```{lang}")
        out.append(code)
        out.append("```")
        out.append("")

    mistakes = kp.get("common_mistakes", [])
    if mistakes:
        out.append("**常见误区（面试中常错答的）**：")
        for i, m in enumerate(mistakes, 1):
            out.append(f"{i}. {m}")
        out.append("")

    self_check = kp.get("self_check", [])
    if self_check:
        out.append("**自检**：")
        for q in self_check:
            out.append(f"- Q: {q.get('question', '')}")
            out.append(f"  - A: {q.get('answer', '')}")
        out.append("")

    further = kp.get("further_reading", [])
    if further:
        out.append("**进一步学习**：")
        for r in further:
            out.append(f"- {r}")
        out.append("")

    return out
