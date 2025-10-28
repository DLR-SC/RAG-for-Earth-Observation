"""Main application logic."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arango.cursor import Cursor
from arango.database import StandardDatabase
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together.chat_models import ChatTogether

from app.sample import Sample


class InterfaceRAG:
    """Interface to RAG system."""

    def __init__(
        self,
        model: BaseChatModel,
        embedding: Embeddings,
        database: StandardDatabase,
        logfile: Path | None = None,
    ) -> None:
        """Initialize the interface."""
        self.llm = model
        self.embedding = embedding
        self.db = database
        self.aql = self._load_aql()
        self.logfile = logfile
        self.prompts = self._get_prompts()

    def _load_aql(self, filepath: Path = Path("app/query.aql")) -> str:
        """Read AQL from file."""
        with filepath.open("r") as f:
            return f.read()

    def _get_prompts(self, directory: Path = Path("app/prompts/")) -> dict[str, str]:
        """Get all prompts in given dir."""
        prompts = {}
        for file in directory.glob("*.txt"):
            with file.open("r") as f:
                prompts[file.stem] = f.read()

        return prompts

    def _run_aql(
        self, query: str, kappa: int = 16, phi: int = 3, psi: int = 2, theta_expand: float = 0.3
    ) -> list[dict[str, Any]] | None:
        """Run query against database.

        :param str query: Query optimized for fulltext search
        :param int kappa: Number of primary nodes returned by query,
            defaults to 16
        :param int phi: Max. number of associated keywords,
            defaults to 3
        :param int psi: Max. number of secondary nodes per keyword,
            defaults to 2
        :param float theta_expand: Threshold for associated keywords,
            defaults to 0.3

        :return list[dict[str, Any]] | None: List of nested result dict. Each
            dict contains exactly `kappa` primary nodes, as well as up to `phi`
            keywords and `psi`secondary nodes.
        """
        bind_vars = {
            "query": query,
            "kappa": kappa,
            "phi": phi,
            "theta_expand": theta_expand,
            "psi": psi
        }
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
        raw_query = self.llm.invoke(self.prompts["query"].format(question=question))
        return str(raw_query.content)

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
        k: int = 10,
    ) -> list[Document]:
        # TODO: Change to mistral tokenizer
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=256, chunk_overlap=64
        )

        # TODO: Is there any option to run 'as batch'?
        doc_splits = text_splitter.split_documents(self._as_documents(aql_results))

        vector_store = Chroma.from_documents(
            documents=doc_splits,
            embedding=self.embedding,
        )

        return vector_store.max_marginal_relevance_search(query=query, k=k)

    def _structure_context(self, documents: list[Document]) -> str:
        """Structure the context to a single string to parse to the LLM."""
        # NOTE: Datasets are also handeld as publication as they are not
        # differenciated when retrieved. Seperate types would not matter anyway
        # since both have aa title and an abstract.
        keywords: set[str] = set()
        publications: set[str] = set()

        keyword_str = "- Topic: {topic}\n- Description: {description}"
        publication_str = "- Title: {title}\n- Content: {content}"

        for doc in documents:
            if doc.metadata.get("type", "") == "keyword":
                keywords.add(
                    keyword_str.format(
                        topic=doc.metadata.get("name"),
                        description=doc.page_content,
                    )
                )
            else:
                publications.add(
                    publication_str.format(
                        title=doc.metadata.get("title"),
                        content=doc.page_content,
                    )
                )

        if len(publications) > 0:
            publication_str = "# Related Documents\n\n" + "\n\n".join(publications)
        else:
            publication_str = ""

        if len(keywords) > 0:
            keyword_str = "\n\n# Related Topics\n\n" + "\n\n".join(keywords)
        else:
            keyword_str = ""

        return publication_str + keyword_str

    def generate_zero_shot_answer(self, question: str) -> str:
        """Generate a zero-shot answer."""
        zero_shot_answer = self.llm.invoke(
            [
                SystemMessage(self.prompts["system"]),
                HumanMessage(self.prompts["zero_shot"].format(question=question)),
            ],
        )
        return str(zero_shot_answer.content)

    def _generate_rag_answer(self, question: str, context: str) -> str:
        """Generate a single-step rag answer."""
        rag = self.llm.invoke(
            [
                SystemMessage(self.prompts["system"]),
                HumanMessage(self.prompts["rag"].format(question=question, context=context)),
            ]
        )

        return str(rag.content)

    def _generate_2rag_answer(
        self, question: str, context: str, zero_shot_answer: str | None = None
    ) -> str:
        """Generate a two-step rag answer."""
        # Prevent unnecessary invocation if already done when generating sample
        if not zero_shot_answer:
            zero_shot_answer = self.generate_zero_shot_answer(question=question)

        revised_answer = self.llm.invoke(
            [
                SystemMessage(self.prompts["system"]),
                HumanMessage(
                    self.prompts["redefine_rag"].format(
                        question=question, context=context, draft_answer=zero_shot_answer
                    )
                ),
            ]
        )

        return str(revised_answer.content)

    def ask(
        self,
        question: str,
        aql_params: dict[str, int | float] | None = None,
        zero_shot_answer: str | None = None,
        metadata: bool = False,
    ) -> str | dict[str, str | list[dict[str, Any]]] | None:
        """Ask an EO related question.

        :param question: question
        :type question: str
        :param aql_params: AQL parameters, defaults to None
        :type aql_params: dict[str, int  |  float] | None, optional

        .. code-block:: python
            {
                "kappa": int,  # Defaults to 16
                "phi": int,  # Defaults to 3
                "psi": int,  # Defaults to 2
                "theta_expand": float,  # Defaults to 0.3
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
            "kappa": aql_params.get("kappa", 16),  # Primary nodes
            "phi": aql_params.get("phi", 3),  # science keywords
            "psi": aql_params.get("psi", 2),  # Secondary nodes
            "theta_expand": aql_params.get("theta_expand", 0.3),  # Threshold for science keyword
        }

        query = self._create_query(question)

        aql_results = self._run_aql(
            query,
            kappa=round(aql_params["kappa"]),
            phi=round(aql_params["phi"]),
            psi=round(aql_params["psi"]),
            theta_expand=aql_params["theta_expand"],
        )

        # No matching entries in database
        if aql_results is None:
            return None

        # Get context from vectorstore
        context = self._create_context_chroma(aql_results=aql_results, query=query, k=10)
        structured_context = self._structure_context(context)

        answer = self._generate_2rag_answer(
            question=query,
            context=structured_context,
            zero_shot_answer=zero_shot_answer
        )

        # TODO: Manually add sources to end of output and also give it to LLM
        # so it can reference. Othweise list of references often is borked.

        if not metadata:
            return answer

        return {
            "answer": answer,
            "query": query,
            "aql_results": aql_results,
            "context": structured_context,
        }

    def generate_sample(
        self, question: str, aql_params: dict[str, int | float] | None = None
    ) -> Sample | None:
        """Generate a sample for a a question."""
        zero_shot_answer = self.generate_zero_shot_answer(question=question)

        two_rag_answer_dict = self.ask(
            question=question,
            aql_params=aql_params,
            metadata=True,
            zero_shot_answer=zero_shot_answer,
        )

        # Ensure existence of metadata
        if isinstance(two_rag_answer_dict, str):
            msg = "'generate_sample()' requires metadata from 'ask()'"
            raise TypeError(msg)

        if two_rag_answer_dict is None:
            return None

        # Extract context from metadata
        context = two_rag_answer_dict.get("context")
        if not isinstance(context, str):
            msg = "'context' must be of 'str' type"
            raise TypeError(msg)

        rag_answer = self._generate_rag_answer(question=question, context=context)

        # Get the models name. Method varies between chat models
        if isinstance(self.llm, ChatOllama):
            model_name = self.llm.model
        elif isinstance(self.llm, ChatOpenAI | ChatTogether):
            model_name = self.llm.model_name
        else:
            msg = f"Unknowm chat model '{type(self.llm)}'"
            raise NotImplementedError(msg)

        query = two_rag_answer_dict.get("query")
        aql_results = two_rag_answer_dict.get("aql_results")
        two_rag_answer = two_rag_answer_dict.get("answer")

        return Sample(
            model=model_name,
            temperature=self.llm.temperature or -1,
            question=question,
            aql_params=aql_params if isinstance(aql_params, dict) else {},
            query=query if isinstance(query, str) else "",
            aql_results=aql_results if isinstance(aql_results, list) else [],
            context=context,
            zero_shot_answer=zero_shot_answer,
            rag_answer=rag_answer,
            two_rag_answer=two_rag_answer if isinstance(two_rag_answer, str) else "",
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    def _save_interaction(  # noqa: PLR0913
        self,
        question: str,
        aql_params: dict[str, float],
        query: str,
        aql_results: list[dict[str, Any]],
        context: str,
        zero_shot_answer: str,
        rag_answer: str,
        two_rag_answer: str,
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
            zero_shot_answer=zero_shot_answer,
            rag_answer=rag_answer,
            two_rag_answer=two_rag_answer,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

        with self.logfile.open("r") as f:
            data = json.load(f)

        data.append(sample.as_dict())
        with self.logfile.open("w") as f:
            json.dump(data, fp=f, ensure_ascii=False, indent=4)
