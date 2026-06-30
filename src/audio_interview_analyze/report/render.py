"""Render a FinalReport as Markdown."""

from __future__ import annotations

from audio_interview_analyze.report.model import FinalReport, Utterance

SPEAKER_LABEL = {
    "interviewer": "[面试官]",
    "candidate": "[候选人]",
    "unknown": "[未知]",
}


def _format_transcript_line(u: Utterance) -> str:
    label = SPEAKER_LABEL.get(u.speaker, f"[{u.speaker}]")
    ts = f"[{u.start:0.1f}s-{u.end:0.1f}s]"
    return f"- {ts} {label} {u.text}"


def render_markdown(report: FinalReport) -> str:
    """Render the report as a Markdown string."""
    sections: list[str] = []

    # Title
    sections.append("# 面试分析报告")
    sections.append("")

    # 1. Overall evaluation
    sections.append("## 1. 面试总评")
    sections.append("")
    sections.append(f"**整体得分：{report.overall_score} / 100**")
    sections.append("")
    sections.append(report.overall_evaluation.strip())
    sections.append("")
    sections.append("**分维度得分：**")
    sections.append("")
    for dim, score in report.dimension_scores.items():
        sections.append(f"- {dim}：{score}")
    sections.append("")

    # 2. Highlights
    sections.append("## 2. 亮点")
    sections.append("")
    if report.consolidated_highlights:
        for h in report.consolidated_highlights:
            sections.append(f"- {h}")
    else:
        sections.append("（无）")
    sections.append("")

    # 3. Weaknesses
    sections.append("## 3. 不足")
    sections.append("")
    if report.consolidated_weaknesses:
        for w in report.consolidated_weaknesses:
            sections.append(f"- {w}")
    else:
        sections.append("（无）")
    sections.append("")

    # 4. Per-question analysis
    sections.append("## 4. 逐题分析")
    sections.append("")
    if report.pair_analyses:
        for p in report.pair_analyses:
            sections.append(f"### 第 {p.pair_index} 题")
            sections.append("")
            sections.append(f"**题目摘要：** {p.question_summary}")
            sections.append("")
            sections.append(f"**答案评价：** {p.answer_evaluation}")
            sections.append("")
            if p.highlights:
                sections.append("**亮点：**")
                for h in p.highlights:
                    sections.append(f"- {h}")
                sections.append("")
            if p.weaknesses:
                sections.append("**不足：**")
                for w in p.weaknesses:
                    sections.append(f"- {w}")
                sections.append("")
            if p.improvement_suggestions:
                sections.append(f"**改进建议：** {p.improvement_suggestions}")
                sections.append("")
    else:
        sections.append("（无）")
        sections.append("")

    # 5. Knowledge points
    sections.append("## 5. 知识点总结")
    sections.append("")
    if report.consolidated_knowledge_points:
        # Group by category
        by_cat: dict[str, list] = {}
        for kp in report.consolidated_knowledge_points:
            by_cat.setdefault(kp.category, []).append(kp)
        for cat, kps in by_cat.items():
            sections.append(f"### {cat}")
            sections.append("")
            for kp in kps:
                sections.append(f"#### {kp.name}")
                sections.append("")
                sections.append(kp.explanation.strip())
                sections.append("")
                if kp.learning_directions:
                    sections.append("**拓展学习方向：**")
                    for d in kp.learning_directions:
                        sections.append(f"- {d}")
                    sections.append("")
    else:
        sections.append("（无）")
        sections.append("")

    # Appendix: full transcript
    sections.append("## 附录：完整转写")
    sections.append("")
    if report.full_transcript.utterances:
        for u in report.full_transcript.utterances:
            sections.append(_format_transcript_line(u))
    else:
        sections.append("（无）")
    sections.append("")

    return "\n".join(sections)
