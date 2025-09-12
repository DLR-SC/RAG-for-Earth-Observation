import os

import pandas as pd
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from pydantic import SecretStr
from tqdm import tqdm

from rag.app import InterfaceRAG

questions = pd.read_csv("prompt_generator/generated_questions.csv", index_col=0)

load_dotenv()

embedding = MistralAIEmbeddings(api_key=SecretStr(os.getenv("MISTRAL_API_KEY_ROXANNE") or ""))

models = [
    ChatOllama(
        model="DLR_FM_2.mistral-small:24b-instruct-2501-q8_0",
        base_url=os.getenv("FM_OLLAMA_BASE_URL"),
        client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('FM_OLLAMA_API_KEY')}"}},
    ),
    ChatOllama(
        model="DLR_FM_1.llama3.3:latest",
        base_url=os.getenv("FM_OLLAMA_BASE_URL"),
        client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('FM_OLLAMA_API_KEY')}"}},
    ),
]

aql_params = {"n": 16, "k": 3, "k_threshold": 0.3, "s": 2}

samples: list[tuple] = []

with tqdm(total=len(models) * len(questions)) as pbar:
    for model in models:

        proto = InterfaceRAG(
            model=model,
            embedding=embedding,
            arango_root_password=SecretStr(os.getenv("CAG_ARANGO_ROOT_PASSWORD") or ""),
        )

        for _, series in questions.iterrows():

            question = series["question"]
            sample = proto.generate_sample(question=question, aql_params=aql_params)

            if sample is None:
                print(f"Unable to find any results for '{series["idx"]}'")
                pbar.update()
                continue

            samples.append((
                series["idx"],
                sample.model,
                sample.temperature,
                sample.question,
                sample.aql_params,
                sample.query,
                sample.aql_results,
                sample.context,
                sample.rag_answer,
                sample.zero_shot_answer,
                sample.timestamp,
            ))
            pbar.update()


columns = [
    "idx",
    "model",
    "temperature",
    "question",
    "aql_params",
    "query",
    "aql_results",
    "context",
    "rag_answer",
    "zero_shot_answer",
    "timestamp"
]

results = pd.DataFrame(data=samples, columns=columns)
results.to_csv("big_eval_new.csv", index=False)
