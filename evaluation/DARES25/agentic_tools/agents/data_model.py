from typing_extensions import  List, Literal
from pydantic import BaseModel, Field


class AnswerToAssess(BaseModel):
    """A question and a generated answer to assess"""

    question: str = Field(
        description="A question or query."
    )

    answer: str = Field(
        description="A generated answer to the question."
    )


class CriticalQuestion(BaseModel):
    """A critical question that challenges an argument, with the ranking id and a very brief reasoning"""
    id: int = Field(
        description="The numeric rank of the generated question, starting with 0. The values reflect the usefulness of the question, where 0 is the most useful."
    )
    critical_question: str = Field(
        description="A critical question to challenge and argument's *acceptability* or *fallaciousness*."
    )
    reason: str = Field(description="Very brief reasoning for the critical question")


class CriticalQuestionList(BaseModel):
    """The list of the three critical questions generated"""
    critical_questions: List[CriticalQuestion] = Field(
        description="The list of all the critical questions."
    )

class QueryTopicsIntentions(BaseModel):#
    """The list of topics, user intentions and metadata of a research query"""
    topic_lst: list = Field(
        description="The list of topics of the research query."
    )
    intention_lst: list = Field(
        description="The list of intentions of the research query"
    )



class Confirmation(BaseModel):
    """Confirmation that the model confirm their role"""

    confirmation: str = Field(
        description="The LLM confirms the role assigned to it by saying 'ok'."
    )


class SocialAgentAnswer(BaseModel):
    critical_question_list: CriticalQuestionList = Field(
        description="The list of all the critical questions and their ranks to criticize and reveal the weaknesses of an argument."
    )
    question_type: Literal["debate", "reflect", "question", "validate"]
    prompt: dict
    trait: str


class Validator4Agent(BaseModel):
    """The validator that takes an argument and a critical question and scores several criteria"""

    feedback: str = Field(
        description="reason behind the scores"
    )

    depth: int = Field(
        description="A score from 1 (low) to 5 (high) if the question deeply probe the core reasoning or assumptions behind the argument"
    )

    relevance: int = Field(
        description="A score from 1 (low) to 5 (high) if the question stay within the scope of the original argument without introducing unrelated ideas"
    )

    reasoning: int = Field(
        description="A score from 1 (low) to 5 (high) if the question is logically coherent and fair, avoiding misrepresentations or fallacies"
    )

    specificity: int = Field(
        description="A score from 1 (low) to 5 (high) if the question is targeted to the particular argument (5) rather than being vague or overly general (1)"
    )


class CriticalQuestionRank(BaseModel):
    """The rank of a critical question"""

    rank: int = Field(
        description="The rank of a critical question with respect to a criteria, where 1 means highly ranked."
    )

    cq: str = Field(
        description="The critical question that is being ranked - AS PROVIDED AND UNCHANGED."
    )


class CriteriaRank(BaseModel):
    """The ranker that takes an argument and a set of critical question and rank them based on a criteria"""

    feedback: str = Field(
        description="The reason behind the ranking"
    )

    cq_ranking_lst: list[CriticalQuestionRank] = Field(
        description="a list of 'rank' as integer (1 means highly ranked) and its corresponding 'cq' (critical question)"
    )


class CriteriaScore(BaseModel):
    """The score for a critical question of a criteria from 1 to 10 """

    score: int = Field(
        description="A score from 1 (low) to 10 (high) if the critical question meets the described criterion,"
    )
