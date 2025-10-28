import itertools
from typing import Optional, Union
import uuid

import pandas as pd

from agentic_tools.agents.state import BasicState, SocialAgentState

# -------- #
#  HELPERS #
#  ------- #


def _get_strategy_permutation(
    n: Union[list[int], int],
    elements: Optional[list[str]] = ["debate", "reflect"]
):
    permutations = []

    def _get_permutation(r: int):
        for p in itertools.product(elements, repeat=r):
            permutations.append(p)

    if isinstance(n, int):
        _get_permutation(n)
    else:
        for r in n:
            _get_permutation(r)
    return permutations


def _get_traits_combos(
    n: Union[list[int], int],
    elements: Optional[list[str]] = ["overconfident", "easy_going"],
):
    combos = []

    def _get_combinations(r: int):
        for p in itertools.combinations_with_replacement(elements, r):
            combos.append(p)

    if isinstance(n, int):
        _get_combinations(n)
    else:
        for r in n:
            _get_combinations(r)
    return combos


def generate_experiment_settings(
    rounds: int = 3,
    number_of_agents: int = 3,
    traits: list = ["overconfident", "easy_going"],
    strategies_: list = ["debate", "reflect"],
):
    all_expr = []
    all_traits_combos = _get_traits_combos(
        range(1, number_of_agents + 1), elements=traits
    )
    all_strategy_permutations = _get_strategy_permutation(
        range(1, rounds + 1), elements=strategies_
    )
    pairs = list(itertools.product(all_traits_combos,
                                   all_strategy_permutations))
    for pair in pairs:
        if len(pair[0]) == 1 and "debate" in pair[1]:
            continue

        trait_str = "".join(sorted([x[0] for x in pair[0]]))
        strategy_str = "".join([x[0] for x in pair[1]])
        all_expr.append(
            {
                "traits": pair[0],
                "strategies": pair[1],
                "rounds": len(pair[1]),
                "number_of_agents": len(pair[0]),
                "has_debate": "debate" in pair[1],
                "thread_id": uuid.uuid4(),
                "experiment_name": "{llm_name}"
                + f"social_n{len(pair[0])}_T{trait_str}_S{strategy_str}",
            }
        )
    for traits in all_traits_combos:
        trait_str = "".join(sorted([x[0] for x in traits]))
        all_expr.append(
            {
                "traits": traits,
                "strategies": tuple(),
                "rounds": 0,
                "number_of_agents": len(traits),
                "has_debate": False,
                "thread_id": uuid.uuid4(),
                "experiment_name": "{llm_name}"
                + f"social_n{len(traits)}_T{trait_str}_S",
            }
        )

    all_exp_settings_df = pd.DataFrame(all_expr)

    all_exp_settings_df.sort_values(
        by=["rounds", "number_of_agents"], ascending=True
    )
    return pd.DataFrame(all_expr)


# FROM PYDANTIC OUTPUT TO JSON
def _basic_state_to_serializable(state: BasicState, final_answer: str = "final_cq") -> dict:
    serializable_state = {}
    for key, value in state.items():
        if key == final_answer:
            # Convert final_cq if it's a Pydantic model
            if hasattr(value, "model_dump"):
                serializable_state[key] = value.model_dump()
            else:
                serializable_state[key] = value
        else:
            serializable_state[key] = value
    return serializable_state


def _state_to_serializable(state: SocialAgentState,
                           attr_name_dict_lst: list = ["round_answer_dict"],
                           attr_name_dict: list = ["criteria_cqs_rank_dict"],
                           final_answer: str = "final_cq"
                           ) -> dict:
    serializable_state = {}
    for key, value in state.items():

        if key in attr_name_dict_lst:
            # Convert each SocialAgentAnswer in the dictionary to a dict
            new_dict = {}
            for k, answer_list in value.items():
                new_dict[k] = [answer.model_dump() for answer in answer_list]
            serializable_state[key] = new_dict
        elif key in attr_name_dict:
            # Convert each SocialAgentAnswer in the dictionary to a dict
            new_dict = {}
            for k, answer_list in value.items():
                new_dict[k] = answer_list.model_dump()  #[answer.model_dump() for answer in answer_list]
            serializable_state[key] = new_dict
        elif key == final_answer:
            # Convert final_cq if it's a Pydantic model
            if hasattr(value, "model_dump"):
                serializable_state[key] = value.model_dump()
            else:
                serializable_state[key] = value
        else:
            serializable_state[key] = value
    return serializable_state
