"""Allows for CLI interaction with the model."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from pydantic import SecretStr

from rag.app import InterfaceRAG

load_dotenv()

print("HF_TOKEN from env:", os.getenv("HF_TOKEN"))


model = ChatOllama(
    model="DLR_FM_1.llama3.3:latest",
    # temperature=0.4,
    base_url=os.getenv("FM_OLLAMA_BASE_URL"),
    client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv("FM_OLLAMA_API_KEY")}"}},
)

embedding = MistralAIEmbeddings(
    api_key=SecretStr(os.getenv("MISTRAL_API_KEY_ROXANNE") or "")
)

logfile = Path("rag/logfile_new_prompts.json")

proto = InterfaceRAG(
    model=model,
    embedding=embedding,
    arango_root_password=SecretStr(os.getenv("CAG_ARANGO_ROOT_PASSWORD") or ""),
    logfile=logfile,
)

while True:
    print()
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    print(f"RAG: {proto.ask(user_input)}")
    # print(f"zero-shot: {proto.ask_zero_shot(user_input)}")
