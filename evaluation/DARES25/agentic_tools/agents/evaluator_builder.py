import json
import statistics
from typing import ClassVar, Optional
import pandas as pd
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, create_model


from agentic_tools import helper
from agentic_tools.agents.agents import AbstractAgent
from agentic_tools.agents.data_model import (
    CriteriaRank,
    CriteriaScore,
    CriticalQuestion,
    CriticalQuestionList,
    Validator4Agent,
)

from agentic_tools.agents.state import (

    RankerAgentState,

    TwoStageValState,
    ScoreAgentState,
)


class ScoreCriteriaPerAnswerBuilder(AbstractAgent):
    answer_lst: list[tuple]
    criteria_list: dict = None
    weights = None # {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}
    scoresClassModel: BaseModel = None
    score_prompt: str = ""
    criteria_str: str = ""

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = helper._basic_state_to_serializable,
        answer_lst: list[tuple] = None,
        weights: Optional[list[str]] = None

    ):
        self.answer_lst = answer_lst if answer_lst is not None else []
        

        with open("agentic_tools/prompts/criteria_scorer_desc.json", "r") as file:
            default_criteria = json.load(file)
        with open("agentic_tools/prompts/score_answer_criteria.txt", "r") as f:
            self.score_prompt = f.read()

        self.criteria_list = self.criteria_list if self.criteria_list is not None else default_criteria 
        self.criteria_str = "\n\n".join([f"- {criterion['criteria']}\n   {criterion['desc']}\n    {'\n    '.join([str(s) + ": "+ str(d) for s,d in criterion['scores'].items()])}" for criterion in self.criteria_list])
            
        self.weights = (
            weights
            if weights is not None
            else {c["criteria"].lower(): 1 for c in self.criteria_list}
        )

        def _create_base_model():
            attributes = {
                "feedback": {
                    "type": str,
                    "default": "",
                    "description": "feedback that assesses the quality of the **answer** strictly based on the given score rubrics."
                }
            }

            for item in self.criteria_list:
                field_name = item["criteria"].lower()  # make attribute name lowercase
                attributes[field_name] = {
                    "type": int,
                    "default": 1,
                    "description": item["attr_description"]
                }

            fields = {
                name: (spec["type"], Field(default=spec["default"], description=spec["description"]))
                for name, spec in attributes.items()
            }

            # Create the model dynamically
            return create_model(
                "AgentScores", 
                __base__=BaseModel,
                **fields
            )
        self.scoresClassModel = _create_base_model()

        super().__init__(
            model_thread_id=model_thread_id,
            llm_name=llm_name,
            llm_num=llm_num,
            experiment_name=experiment_name,
            temperature=temperature,
            serialize_func=serialize_func,
            #state_class = ScoreAgentState
        )

    def calculate_score(self, criteria):
        
        score = (
            sum([getattr(criteria, field_name) * self.weights[field_name] for field_name in criteria.__fields__ if field_name != "feedback"])
        ) / sum(self.weights.values())
        return score

    def build_agent(self) -> StateGraph:
        # llm_num = len(self.agent_trait_lst)
        print("building workflow")
        validator_llm = AbstractAgent._init_llm(self.llm_name,
                                                self.temperature)
        repeat = 1 if self.temperature == 0 else 3
        print(f"the call will be repeated {repeat} times")

        def _score_answer_node(state: ScoreAgentState, answer: str, answer_source: str, context: str = ""):
            
            instr = self.score_prompt.format(question=state["question"], answer=answer, criteria= self.criteria_str, context= context)
            criteria_scores = {field_name: [] for field_name in self.scoresClassModel.model_fields if field_name != "feedback"}

            criteria_score_means  = {}
            scores = []

            while len(scores) < repeat:
                response = validator_llm.with_structured_output(self.scoresClassModel).invoke(
                    [
                        SystemMessage(
                            """
You are an expert Earth Observation (EO) scientist and evaluation specialist.
You assess AI-generated answer to a scientific EO question based on several criteria, using a 1–5 scale for each criterion. Your evaluation must follow the provided rubrics.
"""
                        ),
                        HumanMessage(
                            instr
                        ),
                    ],
                )
                if response is not None:
                    scores.append(self.calculate_score(response))

                    for field_name in response.model_dump().keys():
                        if field_name == "feedback": continue
                        vls_ = getattr(response, field_name)
                        criteria_scores[field_name].append(vls_) 

                        criteria_score_means[field_name] = round(statistics.mean(criteria_scores[field_name]), 2)

            return {"answer_scores_dict": {answer_source: round(statistics.mean(scores), 2)},
                    "answer_criteria_scores": {answer_source: criteria_score_means}
                    }

        def aggregator(state: ScoreAgentState):
            
            best_answer = sorted(
                state["answer_scores_dict"], key=state["answer_scores_dict"].get, reverse=True
            )[0]  # [:1]  # TO DO make it dynamic

            best_answer_index = 0 
            for i, (_, v) in enumerate(state["answer_scores_dict"].items()):
                if v == state["answer_scores_dict"][best_answer]:
                    best_answer_index = i
            
            return {
                "best_answer": best_answer,
                "best_answer_index": best_answer_index,
                "instruction_template": self.score_prompt.format(question="{question}", context="{context}", answer="{answer}", criteria= self.criteria_str)
            }

        workflow = StateGraph(ScoreAgentState)
        workflow.add_node("aggregator", aggregator)
        for i, answer_context in enumerate(self.answer_lst):
            
            answer_source = answer_context[0] if len(answer_context) > 0 else f"answer_{i}"
            answer = answer_context[1] 
            context = answer_context[2] if type(answer_context) != str and len(answer_context) > 2 else ""

            workflow.add_node(
                f"{i}_score_answer_node",
                lambda state, answer=answer, answer_source=answer_source, context=context: 
                    _score_answer_node(state, answer, answer_source, context=context)
            )
            workflow.add_edge(START, f"{i}_score_answer_node")

            workflow.add_edge(f"{i}_score_answer_node", "aggregator")
        workflow.add_edge("aggregator", END)
        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)


class RankerAgentBuilder(AbstractAgent):
    # cqs: list[str]
    weights = {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}

    # do not set
    criteria_desc: dict = None
    
    _CRITERIA_DESC_JSON_FILE_: ClassVar = "prompts/validators/criteria_ranker_desc.json"

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = helper._state_to_serializable,
        weights: Optional[list[str]] = {
            "depth": 1,
            "relevance": 1,
            "specificity": 1,
            "reasoning": 1,
        },
    ):
        self.weights = (
            weights
            if weights is not None
            else {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}
        )
        with open("prompts/validators/criteria_ranker_desc.json", "r") as f:
            self.criteria_desc = json.load(f)

        super().__init__(
            model_thread_id=model_thread_id,
            llm_name=llm_name,
            llm_num=llm_num,
            experiment_name=experiment_name,
            temperature=temperature,
            serialize_func=serialize_func,
        )

    def build_agent(self) -> StateGraph:
        # llm_num = len(self.agent_trait_lst)
        print("building workflow")
        validator_llm = CQSTAbstractAgent._init_llm(self.llm_name,
                                                    self.temperature)

        def rank_criteria_node(state: RankerAgentState, criterion: str):
            criterion_desc = self.criteria_desc[criterion]
            with open("prompts/validators/criteria_ranker.txt", "r") as f:
                prompt = f.read().format(
                    criteria_name=criterion_desc["name"],
                    criteria_adj=criterion_desc["adj"],
                    criteria_desc=criterion_desc["desc"],
                    input_arg=state["input_arg"],
                    cqs=state["cqs"]
                )

            response = None
            while response is None:
                response = validator_llm.with_structured_output(CriteriaRank).invoke(
                    [
                        SystemMessage("You are an assistant that evaluates critical questions based on a specific quality."),
                        HumanMessage(
                            prompt
                        ),
                    ],
                )
            assert response is not None

            #print("RESPONSE", response)
            return {"criteria_cqs_rank_dict": {criterion: response}}

        def aggregator(state: RankerAgentState):
            cqs_ranks = state["criteria_cqs_rank_dict"]
            #print(state["criteria_cqs_rank_dict"])
            scores = {}

            for _, items in cqs_ranks.items():
                for item in items.cq_ranking_lst:
                    if item.cq not in scores.keys(): scores[item.cq] = []
                    scores[item.cq].append(item.rank)

            # Compute mean rank for each question
            mean_scores = {cq: sum(ranks) / len(ranks) for cq, ranks in scores.items()}

            # Sort by mean score (lower is better)
            cqs_items = sorted(mean_scores.items(), key=lambda x: x[1])
            cqs = [x[0] for x in cqs_items][:3]

            return {
                "final_cq": CriticalQuestionList(
                    critical_questions=[
                        CriticalQuestion(id=i, critical_question=cq, reason="")
                        for i, cq in enumerate(cqs)
                    ]
                )
            }

        workflow = StateGraph(RankerAgentState)
        workflow.add_node("aggregator", aggregator)
        for criterion in self.criteria_desc.keys():
            workflow.add_node(
                f"{criterion}_validation_node",
                lambda state, criterion=criterion:
                rank_criteria_node(state, criterion),
            )
            workflow.add_edge(START, f"{criterion}_validation_node")

            workflow.add_edge(f"{criterion}_validation_node", "aggregator")
        workflow.add_edge("aggregator", END)
        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)


class TwoStepsCriteriaScorer(AbstractAgent):
    # cqs: list[str]
    weights = {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}    

    def __init__(
        self,
        model_thread_id: str,
        llm_name: str,
        llm_num: int = 1,
        experiment_name: Optional[str] = None,
        temperature: Optional[float] = None,
        serialize_func: callable = helper._state_to_serializable,
        weights: Optional[list[str]] = {
            "depth": 1,
            "relevance": 1,
            "specificity": 1,
            "reasoning": 1,
        },
    ):
        self.weights = (
            weights
            if weights is not None
            else {"depth": 1, "relevance": 1, "specificity": 1, "reasoning": 1}
        )
       
        super().__init__(
            model_thread_id=model_thread_id,
            llm_name=llm_name,
            llm_num=llm_num,
            experiment_name=experiment_name,
            temperature=temperature,
            serialize_func=serialize_func,
        )

    def build_agent(self) -> StateGraph:
        # llm_num = len(self.agent_trait_lst)
        validator_llm = AbstractAgent._init_llm(self.llm_name,
                                                    self.temperature)

        def score_criteria_node(state: TwoStageValState, criterion: str):
            with open(f"prompts/validators/system_2step_{criterion}.txt", "r") as f:
                system = f.read()
            with open("prompts/validators/step1.txt", "r") as f:
                prompt_1 = f.read()
            with open("prompts/validators/step2.txt", "r") as f:
                prompt_2 = f.read()
            response = None
            cqs = state["cqs"]
            results: dict[str:dict] = {i: {} for i in range(len(cqs))}
            for i, cq in enumerate(cqs):
                response = None
                while response is None:
                    try:
                        messages: list = [
                                SystemMessage(system),
                                HumanMessage(
                                    prompt_1
                                )
                            ]
                        response1 = validator_llm.invoke(
                            messages
                        )
                        #print(response1)
                        messages = messages + [{"role": "assistant",
                                                "content": response1.content},
                                               HumanMessage(prompt_2.format(
                                                   input_arg=state["input_arg"],
                                                   cq=cq))]
                        response = validator_llm.with_structured_output(CriteriaScore).invoke(
                            messages
                        )
                        results[i][criterion] = response.score

                    except Exception as e:
                        print(f"Error: {e}")
                        response = None
            return {"cq_scores_dict": results}

        def aggregator(state: TwoStageValState):
            cq_score_dict = state["cq_scores_dict"]

            # Compute mean rank for each question
            mean_scores = {cq: sum(scores.values()) / len(scores.values()) for cq, scores in cq_score_dict.items()}

            # Sort by mean score (lower is better)
            cqs_items = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
            print(cqs_items)
            cqs_idx = list([x[0] for x in cqs_items])
            print("indices", cqs_idx)
            cqs = [state["cqs"][i] for i in cqs_idx][:3]
            print(cqs)
            return {
                "final_cq": CriticalQuestionList(
                    critical_questions=[
                        CriticalQuestion(id=i, critical_question=cq, reason="")
                        for i, cq in enumerate(cqs)
                    ]
                )
            }

        workflow = StateGraph(TwoStageValState)
        workflow.add_node("aggregator", aggregator)
        for criterion in ["depth", "reasoning", "specificity"]:
            workflow.add_node(
                f"{criterion}_validation_node",
                lambda state, criterion=criterion:
                score_criteria_node(state, criterion),
            )
            workflow.add_edge(START, f"{criterion}_validation_node")

            workflow.add_edge(f"{criterion}_validation_node", "aggregator")
        workflow.add_edge("aggregator", END)
        memory = MemorySaver()
        print("Building Completed!")
        return workflow.compile(checkpointer=memory)
