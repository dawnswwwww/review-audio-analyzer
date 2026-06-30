"""Pydantic data models for the analysis pipeline."""

from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """A single transcribed utterance with speaker label and timing."""

    speaker: str  # "interviewer" | "candidate" | "unknown"
    text: str
    start: float
    end: float


class Transcript(BaseModel):
    """Ordered list of utterances covering the full interview."""

    utterances: list[Utterance]
    duration: float


class QAPair(BaseModel):
    """A single interviewer question and the candidate's answer."""

    question: str
    answer: str
    q_start: float
    q_end: float
    a_start: float
    a_end: float
    pair_index: int = Field(ge=1)


class PairAnalysis(BaseModel):
    """LLM-generated analysis of a single QAPair."""

    pair_index: int = Field(ge=1)
    question_summary: str
    answer_evaluation: str
    knowledge_points: list[str]
    highlights: list[str]
    weaknesses: list[str]
    improvement_suggestions: str


class KnowledgePoint(BaseModel):
    """A canonicalized knowledge point with explanation and learning directions."""

    name: str
    category: str
    explanation: str
    learning_directions: list[str]


class FinalReport(BaseModel):
    """The complete report consumed by the Markdown renderer."""

    overall_evaluation: str
    overall_score: int = Field(ge=0, le=100)
    dimension_scores: dict[str, int]
    consolidated_highlights: list[str]
    consolidated_weaknesses: list[str]
    consolidated_knowledge_points: list[KnowledgePoint]
    pair_analyses: list[PairAnalysis]
    full_transcript: Transcript
