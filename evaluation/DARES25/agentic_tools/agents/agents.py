from abc import ABC
import dataclasses
import json
import os
from typing import ClassVar, Optional, TypedDict, Type
from langchain_openai import ChatOpenAI
import pandas as pd
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pyparsing import abstractmethod
import tqdm
from langchain.chat_models import init_chat_model

from agentic_tools import helper
from agentic_tools.agents.state import ScoreAgentState
from agentic_tools.utils import get_st_data, timer


@dataclasses.dataclass
class AbstractAgent(ABC):
    model_thread_id: str
    llm_name: str
    llm_num: int = 1
    experiment_name: Optional[str] = None
    temperature: Optional[float] = None

    serialize_func: callable = helper._basic_state_to_serializable
    best_answer_key: str = "best_answer"

    # DO NOT SET
    out: json = None
    graph: StateGraph = None
    time_log: pd.DataFrame = None
    llm_lst: list = None

    ROOT_FOLDER: ClassVar[str] = "output/"
    _SUB_FOLDER_: ClassVar[str] = "test_set/" # REMOVE THIS

    def __post_init__(self):
        print(f"Initializing LLM {self.llm_name}")
        self.llm_lst = [
            AbstractAgent._init_llm(self.llm_name, self.temperature)
            for _ in range(self.llm_num)
        ]
        print("Building Agentic Graph")
        self.graph = self.build_agent()
        print(
            "run `display(Image(graph.get_graph(xray=1).draw_mermaid_png()))` to see your Agentic Graph"
        )
        self.experiment_name = (
            self.experiment_name
            if self.experiment_name is not None
            else f"{self.llm_name}_temperature{self.temperature}_{self.__class__.__name__.lower()}"
        )
        print("experiment name: ", self.experiment_name)

    @staticmethod
    def _init_llm(llm_name: str, temperature: int = 0):
        if llm_name == "o3-mini-2025-01-31":
            return ChatOpenAI(model=llm_name)
        elif llm_name.startswith("gpt") or llm_name.startswith("openai"):
            return ChatOpenAI(model=llm_name, temperature=temperature)
        elif llm_name.lower().startswith(
            "meta-llama".lower()
        ) or llm_name.lower().startswith("mistralai".lower()):
            return init_chat_model(
                llm_name,
                model_provider="together",
                temperature=temperature,
            )
        elif llm_name.lower().startswith("mistral"):
            return init_chat_model(
                llm_name,
                model_provider="mistralai",
                temperature=temperature,
            )

    @abstractmethod
    def build_agent(self) -> StateGraph:
        pass

    def _invoke_graph(self, params, id_):
        print("invoking")
        questions = []
        dict_ = None
        config = {"configurable": {"thread_id": f"{self.model_thread_id}_{id_}"}}
        fname = f"{AbstractAgent.ROOT_FOLDER}last_state/{self.experiment_name}_{id_}.json"
        if os.path.exists(fname):
            print("already assessed")
            with open(fname, "r") as f:
                dict_ = json.load(f)
        else:
            for event in self.graph.stream(params, config, stream_mode="values"):
                # Review
                _labels = event.get(self.best_answer_key, "")
                if _labels:
                    questions.append(_labels)
                    # Save the final state

                    with open(
                        fname,
                        "w",
                    ) as f:
                        try:
                            if self.serialize_func is None:
                                dict_ = dict(event)
                                json.dump(dict(event), f, indent=2)
                            else:
                                dict_ =self.serialize_func(event) 
                                json.dump(self.serialize_func(event), f, indent=2)
                        except Exception:
                            print("error whiles saving file: ", fname)
        
        return dict_

    def run_experiment(self, data_type: str = "validation", save: bool = True):
        out = {}
        time_in_seconds_arr = []
        for _, line in tqdm.tqdm(get_st_data(data_type).items()):
            with timer(f"Iteration {line['intervention_id']}",
                       time_in_seconds_arr):
                input_arg = (
                    line["intervention"].encode("utf-8").decode("unicode-escape")
                )
                cqs = self._invoke_graph(
                    {"input_arg": input_arg}, line["intervention_id"]
                ).model_dump()["critical_questions"]

                # postprocessing data: replacing critical_question with cq to match the ST format
                for e in cqs:
                    e["cq"] = e.pop("critical_question")
                line["cqs"] = cqs

                out[line["intervention_id"]] = line
        if save:
            with open(
                f"{AbstractAgent.ROOT_FOLDER}output_{self.experiment_name}.json",
                "w",
            ) as o:
                json.dump(out, o, indent=4)
        # Log TIME
        time_log_df = pd.read_csv(f"{AbstractAgent.ROOT_FOLDER}time_log.csv")
        time_log_df[f"{self.experiment_name}"] = time_in_seconds_arr
        time_log_df.to_csv(f"{AbstractAgent.ROOT_FOLDER}time_log.csv", index=False)
        self.time_log = time_log_df

        self.out = out
        return out

