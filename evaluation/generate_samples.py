import json
import os
from pathlib import Path

from chromadb.utils.embedding_functions import MistralEmbeddingFunction
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from pydantic import SecretStr
from tqdm import tqdm

from rag.app import InterfaceRAG

load_dotenv()

embedding = MistralAIEmbeddings(
    api_key=SecretStr(os.getenv("MISTRAL_API_KEY_ROXANNE") or "")
)

models = [
    ChatOllama(
        model="DLR_FM_2.mistral-small:24b-instruct-2501-q8_0",
        # temperature=0.1,
        base_url=os.getenv("FM_OLLAMA_BASE_URL"),
        client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('FM_OLLAMA_API_KEY')}"}},
    ),
    ChatOllama(
        model="DLR_FM_1.llama3.3:latest",
        # temperature=0.1,
        base_url=os.getenv("FM_OLLAMA_BASE_URL"),
        client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('FM_OLLAMA_API_KEY')}"}},
    ),
    # ChatTogether(
    #     model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    #     temperature=0.1,
    #     api_key=SecretStr(os.getenv("TOGETHER_AI_API_KEY") or ""),
    # ),
]


with Path("evaluation/questions.json").open("r") as fp:
    questions: list[str] = json.load(fp)

eval_file = Path("evaluation/samples/samples_chroma_max_marginal_relevance_search.json")

aql_params = {"n": 16, "k": 3, "k_threshold": 0.3, "s": 2}

with tqdm(total=len(models) * len(questions)) as pbar:
    for model in models:
        interface = InterfaceRAG(
            model=model,
            embedding=embedding,
            arango_root_password=SecretStr(os.getenv("CAG_ARANGO_ROOT_PASSWORD") or ""),
        )

        for question in questions:
            sample = interface.generate_sample(question, aql_params)

            if sample is None:
                print(f"No results for '{question}' found in database")
                continue

            with eval_file.open("r") as fp:
                data = json.load(fp)

            data.append(sample.as_dict())

            with eval_file.open("w") as fp:
                json.dump(data, fp=fp, ensure_ascii=False, indent=4)

            pbar.update()
