
import operator
from typing import Any, TypedDict
from typing_extensions import Annotated

from agentic_tools import utils
from .data_model import CriteriaRank, CriticalQuestionList, SocialAgentAnswer


# state
class BasicState(TypedDict):
    input_arg: str

    # OutPut
    final_cq: CriticalQuestionList


class SocialAgentState(TypedDict):
    # INPUT
    input_arg: str
    collaborative_strategy: list

    # Runtime vars
    roles_confirmed: Annotated[list, operator.add]
    round_answer_dict: Annotated[dict[str, list[SocialAgentAnswer]],
                                 utils.dict_reducer]
    current_round: int = -2

    # OUTPUT
    final_cq: CriticalQuestionList
    validation_instruction: str


class ScoreAgentState(TypedDict):
    # INPUT
    question: str
    answer: str

    answer_scores_dict: Annotated[dict[str, float], operator.or_]
    answer_criteria_scores: Annotated[dict[str, dict], utils.dict_of_dict]

    # OUTPUT
    best_answer: str  # make it dynamic
    best_answer_index: int  # make it dynamic
    instruction_template: str


class RankerAgentState(TypedDict):
    # INPUT
    input_arg: str
    cqs: list[str]

    criteria_cqs_rank_dict: Annotated[dict[str, CriteriaRank], operator.or_] 
    # OUTPUT
    final_cq: CriticalQuestionList
    validation_instruction: str


class TwoStageValState(TypedDict):
    # INPUT
    input_arg: str
    cqs: list[str]

    cq_scores_dict: Annotated[dict[Any:dict], utils.dict_of_dict]
    # OUTPUT
    final_cq: CriticalQuestionList
    validation_instruction: str