"""Transcription with speaker alignment.

Run ``faster-whisper`` on the audio, then intersect the word-level
timestamps with the diarization segments to produce ordered
``Utterance(speaker, text, start, end)`` objects.
"""

from __future__ import annotations

from faster_whisper import WhisperModel  # type: ignore[import-untyped]

from audio_interview_analyze.report.model import Transcript, Utterance


def transcribe(
    audio_path: str | bytes,
    *,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "default",
) -> list[dict]:
    """Run Whisper and return a list of ``{"start", "end", "text", "words"}``.

    Each ``words`` entry is ``{"start", "end", "word"}``. The model is
    loaded once per process; callers should reuse the returned list rather
    than re-running on the same file.
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(audio_path, vad_filter=True, word_timestamps=True)
    out: list[dict] = []
    for seg in segments:
        out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": [
                    {"start": float(w.start), "end": float(w.end), "word": w.word}
                    for w in (seg.words or [])
                ],
            }
        )
    return out


def assign_speakers(
    whisper_segments: list[dict],
    diarization: list[tuple[float, float, str]],
) -> Transcript:
    """Intersect Whisper word timestamps with diarization segments to produce
    a labeled ``Transcript``.

    Strategy: for each Whisper segment, find the diarization segment that
    overlaps the most (by total overlapping duration), and use its speaker
    label for the whole segment. This is good enough when Whisper segments
    are sentence-sized.
    """
    utterances: list[Utterance] = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        text = seg["text"]
        if not text:
            continue
        best_label = "unknown"
        best_overlap = 0.0
        for d_start, d_end, label in diarization:
            overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        utterances.append(
            Utterance(
                speaker=best_label,
                text=text,
                start=seg_start,
                end=seg_end,
            )
        )
    duration = max((u.end for u in utterances), default=0.0)
    return Transcript(utterances=utterances, duration=duration)