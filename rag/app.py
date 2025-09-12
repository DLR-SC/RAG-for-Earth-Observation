# Tool calling might be an option but probably not important since the retrieval
# is called every time.
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arango.client import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together.chat_models import ChatTogether
from pydantic import SecretStr

from rag.processing.context_processor import structure_context
from rag.sample import Sample


class InterfaceRAG:
    """Interface to RAG system."""

    def __init__(
        self,
        model: BaseChatModel,
        embedding: Embeddings,
        arango_root_password: SecretStr,
        logfile: Path | None = None,
    ) -> None:
        """Initialize the interface."""
        self.llm = model
        self.embedding = embedding
        self.db = self._connect_arangodb(arango_root_password)
        self.aql = self._load_aql()
        self.logfile = logfile
        self.prompts = self._get_prompts()

    def _connect_arangodb(self, arango_root_password: SecretStr) -> StandardDatabase:
        """Establish a connection to an ArangoDB."""
        return ArangoClient().db(
            name="ScienceSearch",
            password=arango_root_password.get_secret_value(),
        )

    def _load_aql(self, filepath: Path = Path("rag/query.aql")) -> str:
        """Read AQL from file."""
        with filepath.open("r") as f:
            return f.read()

    def _get_prompts(self, directory: Path = Path("rag/prompts/")) -> dict[str, str]:
        """Get all prompts in given dir."""
        prompts = {}
        for file in directory.glob("*.txt"):
            with file.open("r") as f:
                prompts[file.stem] = f.read()

        return prompts

    def _run_aql(
        self, query: str, n: int = 2, k: int = 2, k_threshold: float = 0.3, s: int = 2
    ) -> list[dict[str, Any]] | None:
        """Run query against database.

        :param str query: Query optimized for fulltext search
        :param int n: Number of primary nodes returned by query, defaults to 2
        :param int k: Number of associated keywords, defaults to 2
        :param float k_threshold: Threshold for associated keywords, defaults to
            0.3
        :param int s: Number of secondary nodes per keyword (`s` nodes with the
            highest score for a assocaietd keywords), defaults to 2
        :return list[dict[str, Any]] | None: List of nested result dict. Each
            dict contains `n` primary nodes, associated with `k` keywords (with
            a minimum score of `k_threshold`) and s secondary_nodes for each
            associated keyword
        """
        bind_vars = {"query": query, "n": n, "k": k, "k_threshold": k_threshold, "s": s}
        cursor = self.db.aql.execute(self.aql, count=True, bind_vars=bind_vars)

        # No matches found in database
        if not cursor:
            print("No results found")
            return None

        if not isinstance(cursor, Cursor):
            msg = "Malformed cursor"
            raise TypeError(msg)

        return [
            {
                "title": doc.get("title"),
                "abstract": doc.get("abstract"),
                "uri": doc.get("uri"),
                "science_keywords": [
                    {
                        "name": sk.get("name"),
                        "description": sk.get("description"),
                        "secondary_nodes": [
                            {
                                "title": sn.get("title"),
                                "abstract": sn.get("abstract"),
                                "uri": sn.get("uri"),
                            }
                            for sn in sk.get("secondary_nodes")
                        ],
                    }
                    for sk in doc.get("science_keywords")
                ],
            }
            for doc in cursor
        ]

    def _create_query(self, question: str) -> str:
        """Create a query optimized for fulltext search."""
        values = {"question": question}
        raw_query = self.llm.invoke(self.prompts["query"].format(**values))
        return str(raw_query.content)

    def ask_zero_shot(self, question: str) -> str:
        """Zero-shot as reference."""
        values = {"question": question}
        # raw_answer = self.llm.invoke(self.prompts["zero-shot"].format(**values))
        raw_answer = self.llm.invoke(
            [
                SystemMessage(self.prompts["system_zero-shot"]),
                HumanMessage(self.prompts["zero-shot"].format(**values)),
            ],
        )
        return str(raw_answer.content)

    def _as_documents(self, aql_results: list[dict[str, Any]]) -> list[Document]:
        docs_list: list[Document] = []

        # TODO: Maybe 'type' as metadata so the LLM can differenciate between publications and datasets?
        for doc in aql_results:
            docs_list.append(
                Document(
                    page_content=doc.get("abstract", ""),
                    metadata={
                        "title": doc.get("title"),
                        "source": doc.get("uri"),
                    },
                )
            )
            for science_keyword in doc.get("science_keywords", []):
                # Some ScienceKeywords got no description
                if science_keyword.get("description") is not None:
                    docs_list.append(
                        Document(
                            page_content=science_keyword.get("description"),
                            metadata={
                                "name": science_keyword.get("name"),
                                "source": science_keyword.get("reference"),
                                "type": "keyword",
                            },
                        )
                    )
                for secondary_node in science_keyword.get("secondary_nodes", []):
                    docs_list.append(  # noqa: PERF401
                        Document(
                            page_content=secondary_node.get("abstract", ""),
                            metadata={
                                "title": secondary_node.get("title"),
                                "source": secondary_node.get("uri"),
                            },
                        )
                    )

        return docs_list

    def _create_context_chroma(
        self,
        aql_results: list[dict[str, Any]],
        query: str,
        k: int = 4,
    ) -> list[Document]:

        # TODO: Change to mistral tokenizer
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=256, chunk_overlap=64
        )

        # TODO: Hier irgendwo 'as batch' um das ganze zu beschleunigen?
        doc_splits = text_splitter.split_documents(self._as_documents(aql_results))

        vector_store = Chroma.from_documents(
            documents=doc_splits,
            embedding=self.embedding,
        )

        return vector_store.max_marginal_relevance_search(query=query, k=k)

    def _create_context(self, aql_results: list[dict[str, Any]], query: str) -> Any:

        # TODO: Change to mistral tokenizer
        # https://api.python.langchain.com/en/latest/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain_text_splitters.character.RecursiveCharacterTextSplitter.from_huggingface_tokenizer
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=256, chunk_overlap=64
        )

        doc_splits = text_splitter.split_documents(self._as_documents(aql_results))

        vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits,
            embedding=self.embedding,
        )

        retriever = vectorstore.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever=retriever,
            name="retrieve_eo_documents",
            description="Search and return scientific Earth Observation publications and datasets.",
        )

        return retriever_tool.invoke({"query": query})

    def ask(
        self,
        question: str,
        aql_params: dict[str, int | float] | None = None,
        metadata: bool = False,
    ) -> str | dict[str, str | list[dict[str, Any]]] | None:
        """Ask an EO related question.

        :param question: question
        :type question: str
        :param aql_params: AQL parameters, defaults to None
        :type aql_params: dict[str, int  |  float] | None, optional

        .. code-block:: python
            {
                "n": int,  # Defaults to 16
                "k": int,  # Defaults to 3
                "k_threshold": float,  # Defaults to 0.3
                "s": int,  # Defaults to 2
            }

        :param metadata: also return internal parameters (`query`, `aql_results`
            and `context`), defaults to False
        :type metadata: bool, optional
        :return: answer (if database contains matches) or dict containing answer
            and metadata
        :rtype: str | dict[str, str | list[dict[str, Any]]] | None

        Structure for return with metdata:

        .. code-block:: python
            {
                "answer": str,
                "query": str,
                "aql_results": list[dict[str, Any]],
                "context": str,
            }
        """
        if aql_params is None:
            aql_params = {}

        # Set parameters for query
        aql_params = {
            "n": aql_params.get("n", 16),  # Primary nodes
            "k": aql_params.get("k", 3),  # SecienceKeywords
            "k_threshold": aql_params.get(
                "k_threshold", 0.3
            ),  # Threshold for ScienceKeywords
            "s": aql_params.get("s", 2),  # Secondary nodes
        }

        query = self._create_query(question)

        aql_results = self._run_aql(
            query,
            n=round(aql_params["n"]),
            k=round(aql_params["k"]),
            k_threshold=aql_params["k_threshold"],
            s=round(aql_params["s"]),
        )

        # No matching entries in database
        if aql_results is None:
            return None

        # Get context from vectorstore database
        # context = self._create_context(aql_results=aql_results, query=query)
        context = self._create_context_chroma(aql_results=aql_results, query=query)

        values = {"context": context, "question": question}

        # TODO: Manually add sources to end of output and also give it to LLM
        # so it can reference. Otherise list of references often is borked.

        raw_answer = self.llm.invoke(
            [
                SystemMessage(self.prompts["system_rag"]),
                HumanMessage(self.prompts["rag"].format(**values)),
            ],
        )
        answer = str(raw_answer.content)

        if not metadata:
            return answer

        return {
            "answer": answer,
            "query": query,
            "aql_results": aql_results,
            "context": str(context),
        }

    def generate_sample(
        self, question: str, aql_params: dict[str, int | float] | None = None
    ) -> Sample | None:
        """Generate a sample for a a question."""
        rag_answer_dict = self.ask(
            question=question, aql_params=aql_params, metadata=True
        )
        zero_shot_answer = self.ask_zero_shot(question=question)

        if isinstance(rag_answer_dict, str):
            msg = "'generate_sample()' requires metadata from 'ask()'"
            raise TypeError(msg)

        if rag_answer_dict is None:
            return None

        # Get the models name. Method varies between chat models
        if isinstance(self.llm, ChatOllama):
            model_name = self.llm.model
        elif isinstance(self.llm, ChatOpenAI | ChatTogether):
            model_name = self.llm.model_name
        else:
            msg = f"Unknowm chat model '{type(self.llm)}'"
            raise NotImplementedError(msg)

        rag_answer = rag_answer_dict.get("answer")
        query = rag_answer_dict.get("query")
        aql_results = rag_answer_dict.get("aql_results")
        context = rag_answer_dict.get("context")

        return Sample(
            model=model_name,
            temperature=self.llm.temperature or -1,
            question=question,
            aql_params=(
                aql_params if isinstance(aql_params, dict) else {}
            ),  # TODO: Will default to {} if not params are passed
            query=query if isinstance(query, str) else "",
            aql_results=aql_results if isinstance(aql_results, list) else [],
            context=context if isinstance(context, str) else "",
            rag_answer=rag_answer if isinstance(rag_answer, str) else "",
            zero_shot_answer=zero_shot_answer,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    def _save_interaction(  # noqa: PLR0913
        self,
        question: str,
        aql_params: dict[str, float],
        query: str,
        aql_results: list[dict[str, Any]],
        context: str,
        answer: str,
        zero_shot_answer: str,
    ) -> None:
        """Save an interaction for evaluation purposes."""
        if self.logfile is None:
            print("WARNING: No logfile defined, unable to save sample")
            return

        if isinstance(self.llm, ChatOllama):
            model_name = self.llm.model
        elif isinstance(self.llm, ChatOpenAI):
            model_name = self.llm.model_name
        else:
            msg = f"Unknowm chat model '{type(self.llm)}'"
            raise NotImplementedError(msg)

        sample = Sample(
            model=model_name,
            temperature=self.llm.temperature or -1,
            question=question,
            aql_params=aql_params,
            query=query,
            aql_results=aql_results,
            context=context,
            rag_answer=answer,
            zero_shot_answer=zero_shot_answer,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

        with self.logfile.open("r") as f:
            data = json.load(f)

        data.append(sample.as_dict())
        with self.logfile.open("w") as f:
            json.dump(data, fp=f, ensure_ascii=False, indent=4)
