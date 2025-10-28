# Create a list with models that should be evaluated (qwen 7, 32, 72 and something large)
# Load list with questions
# Ask all questions with RAG and as zero shot
# LLM-as-a-judge comparison (with position swap)
# Save results to DataFrame and do plotting / display results

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from pydantic import SecretStr
from tqdm import tqdm

from backend.app.sample import Sample


def get_samples(filepath: Path = Path("evaluation/samples.json")) -> list[Sample]:
    """Get the list of all generated samples."""
    with filepath.open("r") as fp:
        return [Sample(**item) for item in json.load(fp)]


def get_prompts(directory: Path = Path("evaluation/prompts/")) -> dict[str, str]:
    """Get all prompts in given dir."""
    prompts = {}
    for file in directory.glob("*.txt"):
        with file.open("r") as f:
            prompts[file.stem] = f.read()

    return prompts


def eval_pairwise(
    sample: Sample, model: BaseChatModel, prompt: str
) -> dict[str, str | tuple[str, str] | None]:
    """Let the LLM-as-a-judge choose which answer it prefers.

    To deal with position bias run twice with switched position. If decision is
    based on position call it a tie.

    Output:

    .. code-block:: python
        {
            "judgement": Literal["rag", "zero-shot", "tie"],
            "decision_1": str, # RAG is [[A]]
            "decision_2: str,  # RAG is [[B]]
        }
    """
    values = {
        "question": sample.question,
        "answer_a": sample.rag_answer,
        "answer_b": sample.zero_shot_answer,
    }

    values_inverted = {
        "question": sample.question,
        "answer_a": sample.zero_shot_answer,
        "answer_b": sample.rag_answer,
    }

    decision_1 = model.invoke(prompt.format(**values)).content.__str__()
    decision_2 = model.invoke(prompt.format(**values_inverted)).content.__str__()

    end_1 = "A" if decision_1.endswith("[[A]]") else "B"
    end_2 = "A" if decision_2.endswith("[[A]]") else "B"

    if end_1 == end_2:
        return {
            "judgement": "tie",
            "decision_1": decision_1,

        }

    if end_1 == "A" and end_2 == "B":
        return {
            "judgement": "rag",
            "decision_1": decision_1,
            "decision_2": decision_2,
        }

    if end_1 == "B" and end_2 == "A":
        return {
            "judgement": "zero-shot",
            "decision_1": decision_1,
            "decision_2": decision_2,
        }

    msg = "Unable to extract decision"
    raise RuntimeError(msg)


def eval_hf_question_answer(sample: Sample, model: BaseChatModel, prompt: str) -> dict[str, float]:
    """Rates on a scale of 0 to 10 how well the users question gets answered.

    Output:

    .. code-block:: python
        {
            "rag_answer_score": float,
            "zero_shot_answer_score": float,
        }
    """
    rag_values = {"question": sample.question, "answer": sample.rag_answer}
    zero_shot_values = {"question": sample.question, "answer": sample.zero_shot_answer}

    rag_judgement = model.invoke(prompt.format(**rag_values)).content.__str__()
    zero_shot_judgement = model.invoke(prompt.format(**zero_shot_values)).content.__str__()

    split_str = "Total rating: "

    rag_rating = rag_judgement.split(split_str)[-1]
    zero_shot_rating = zero_shot_judgement.split(split_str)[-1]

    return {
        "rag_answer_score": float(rag_rating),
        "zero_shot_answer_score": float(zero_shot_rating),
    }


def eval_hf_question_answer_improved(
    sample: Sample, model: BaseChatModel, prompt: str
) -> dict[str, float | str]:
    """Rates answers on a range of 0 to 4.

    1: The system_answer is terrible: completely irrelevant to the question
        asked, or very partial
    2: The system_answer is mostly not helpful: misses some key aspects of the
        question
    3: The system_answer is mostly helpful: provides support, but still could be
        improved
    4: The system_answer is excellent: relevant, direct, detailed, and addresses
        all the concerns raised in the question

    Output:

    .. code-block:: python
        {
            "rag_answer_score": int,
            "rag_answer_reasoning": str,
            "zero_shot_answer_score": int,
            "zero_shot_answer_reasoning": str,
        }
    """
    rag_values = {"question": sample.question, "answer": sample.rag_answer}
    zero_shot_values = {"question": sample.question, "answer": sample.zero_shot_answer}

    rag_judgement = model.invoke(prompt.format(**rag_values)).content.__str__()
    zero_shot_judgement = model.invoke(prompt.format(**zero_shot_values)).content.__str__()

    rag_split = rag_judgement.split("Total rating: ")
    zero_shot_split = zero_shot_judgement.split("Total rating: ")

    return {
        "rag_answer_score": int(rag_split[-1]),
        "rag_answer_reasoning": rag_split[0].split("Evaluation: ")[-1],
        "zero_shot_answer_score": int(zero_shot_split[-1]),
        "zero_shot_answer_reasoning": zero_shot_split[0].split("Evaluation: ")[-1],
    }


def eval_mistral_rag(sample: Sample, model: BaseChatModel, prompt: str) -> dict[str, str | float]:
    """Evaluate context / answer relevance and groundedness.

    https://github.com/mistralai/cookbook/blob/main/mistral/evaluation/RAG_evaluation.ipynb
        -> Added instruction for structured output
    """
    values = {"query": sample.question, "answer": sample.rag_answer, "context": sample.context}
    judgement = model.invoke(prompt.format(**values)).content.__str__()

    scores = re.findall(r"score:\s*\D*(\d+)", string=judgement, flags=re.IGNORECASE)
    scores = [int(score) for score in scores]

    return {
        "judgement": judgement,
        "context_relevance": scores[0],
        "answer_relevance": scores[1],
        "groundedness": scores[2],
    }

load_dotenv()

judge = ChatOpenAI(
    model="o4-mini",
    api_key=SecretStr(os.getenv("OPENAI_API_KEY_ROXANNE") or "")
)

prompts = get_prompts()
samples = get_samples(Path("evaluation/samples_2.json"))

test = False
if test:
    eval_mistral_rag(samples[12], judge, prompts["mistral_rag_eval"])
    import sys
    sys.exit()

results = []
num_evals = 4

with tqdm(total=len(samples) * num_evals) as pbar:
    for sample in samples:

        pairwise = eval_pairwise(
            sample=sample,
            model=judge,
            prompt=prompts["pairwise"],
        )
        pbar.update()

        hf_question_answer = eval_hf_question_answer(
            sample=sample,
            model=judge,
            prompt=prompts["hf_question_answer"],
        )
        pbar.update()

        hf_question_answer_improved = eval_hf_question_answer_improved(
            sample=sample,
            model=judge,
            prompt=prompts["hf_question_answer_improved"],
        )
        pbar.update()

        mistral_rag = eval_mistral_rag(
            sample=sample,
            model=judge,
            prompt=prompts["mistral_rag_eval"],
        )
        pbar.update()

        results.append(
            {
                "question": sample.question,
                "rag_answer": sample.rag_answer,
                "zero_shot_answer": sample.zero_shot_answer,
                "pairwise": {
                    "judgement": pairwise["judgement"],
                    "decision_1": pairwise["decision_1"],
                    "decision_2": pairwise["decision_2"],
                },
                "hf_question_answer": {
                    "rag_answer_score": hf_question_answer["rag_answer_score"],
                    "zero_shot_answer_score": hf_question_answer["zero_shot_answer_score"],
                },
                "hf_question_answer_improved": {
                    "rag_answer_score": hf_question_answer_improved["rag_answer_score"],
                    "rag_answer_reasoning": hf_question_answer_improved["rag_answer_reasoning"],
                    "zero_shot_answer_score": hf_question_answer_improved["zero_shot_answer_score"],
                    "zero_shot_answer_reasoning": hf_question_answer_improved["zero_shot_answer_reasoning"],
                },
                "rag_metrics": {
                    "judgement": mistral_rag["judgement"],
                    "context_relevance": mistral_rag["context_relevance"],
                    "answer_relevance": mistral_rag["answer_relevance"],
                    "groundedness": mistral_rag["groundedness"],
                }
            }
        )

with Path("evaluation/eval_2.json").open("w") as fp:
    json.dump(results, fp=fp, ensure_ascii=False, indent=4)


# TODO
# - mehr fragen
