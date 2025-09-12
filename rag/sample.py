from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    """Wrapper class for question-answer pairs with metadata."""

    model: str
    temperature: float
    question: str
    aql_params: dict[str, float]
    query: str
    aql_results: list[dict[str, Any]]
    context: str
    rag_answer: str
    zero_shot_answer: str
    timestamp: str

    def as_dict(self) -> dict[str, str | float | dict[str, float] | list[dict[str, Any]]]:
        """Get the sample as dict."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "question": self.question,
            "aql_params": self.aql_params,
            "query": self.query,
            "aql_results": self.aql_results,
            "context": self.context,
            "rag_answer": self.rag_answer,
            "zero_shot_answer": self.zero_shot_answer,
            "timestamp": self.timestamp,
        }
