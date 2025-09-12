from dataclasses import dataclass
from typing import Literal


@dataclass
class Evaluation:
    """Wrapper for evaluation instances."""

    model: str
    idx: str
    pairwise_winner: Literal["zero-shot", "rag", "tie"] | None = None
    pairwise_judgement_rag_zero_shot: str | None = None
    pairwise_judgement_zero_shot_rag: str | None = None
    score: int | None = None
    rag_judgement: str | None = None
    context_relevance: int | None = None
    answer_relevance: int | None = None
    groundedness: int | None = None

    def as_csv_line(self, sep: str = ",") -> str:
        """Return eval as line writable to a csv."""
        for attr, value in self.__dict__.items():
            if isinstance(value, str):
                setattr(self, attr, value.replace("\n", "").replace('"', '\\"'))

        return (
            f'\n"{self.model}"{sep}"{self.idx}"'
            f'{sep}"{self.pairwise_winner or ""}"'
            f'{sep}"{self.pairwise_judgement_rag_zero_shot or ""}"'
            f'{sep}"{self.pairwise_judgement_zero_shot_rag or ""}"'
            f'{sep}{self.score or ""}'
            f'{sep}"{self.rag_judgement or ""}"'
            f'{sep}{self.context_relevance or ""}'
            f'{sep}{self.answer_relevance or ""}'
            f'{sep}{self.groundedness or ""}'
        )
