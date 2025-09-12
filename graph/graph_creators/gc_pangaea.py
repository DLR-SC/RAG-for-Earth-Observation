import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from better_cag import upsert_vertex
from cag.framework import GraphCreatorBase
from cag.graph_elements.base_graph import BaseGraph
from cag.graph_elements.nodes import AbstractNode
from cag.graph_elements.relations import HasAuthor
from edges import HasKeyword
from nodes import EarthObservationAuthor, EarthObservationDataset, ScienceKeyword
from pyArango.collection import Document
from pyArango.theExceptions import CreationError
from tqdm import tqdm


class PangaeaGraphCreator(GraphCreatorBase):
    """Graph creator for the Pangaea data."""

    _name = "Pangaea Graph Creator"
    _description = "Creates a graph based on Pangaea metadata"

    _ABSTRACT = AbstractNode
    _AUTHOR = EarthObservationAuthor
    _DATASET = EarthObservationDataset
    _SCIENCE_KEYWORD = ScienceKeyword

    _HAS_AUTHOR = HasAuthor
    _HAS_KEYWORD = HasKeyword

    _edge_definitions = [
        {
            "relation": GraphCreatorBase._BELONGS_TO_RELATION_NAME,
            "from_collections": [_DATASET],
            "to_collections": [GraphCreatorBase._CORPUS_NODE_NAME],
        },
        {
            "relation": _HAS_AUTHOR,
            "from_collections": [_DATASET],
            "to_collections": [_AUTHOR],
        },
        {
            "relation": _HAS_KEYWORD,
            "from_collections": [_DATASET],
            "to_collections": [_SCIENCE_KEYWORD],
        },
    ]

    def init_graph(self) -> None:
        """Initialize the graph."""
        self._set_corpus_node(datetime.now(tz=UTC))

        with Path("gcmd_science_keywords/independent_vertices.json").open("r") as f:
            self.science_keywords: dict[str, dict[str, str | bool | None]] = {
                item["uuid"]: item for item in json.load(f)
            }

        pangaea_files = sorted(Path(self.corpus_file_or_dir).rglob("*.json"))

        # Count files
        total = sum(len(json.load(file.open("r"))) for file in pangaea_files)

        with tqdm(desc="Processing Pangaea datasets", total=total) as pbar:
            for file in pangaea_files:
                with file.open("r") as f:
                    for dataset in json.load(f):
                        with contextlib.suppress(CreationError):
                            self._create_update_dataset_node(dataset)
                        pbar.update()

    def update_graph(self, timestamp: datetime) -> None:
        """Update an exisiting graph."""
        self.init_graph()

    def _set_corpus_node(self, timestamp: datetime) -> None:
        dict_ = {
            "_key": "PangaeaData",
            "created_on": datetime.now(tz=UTC),
            "description": "Datasets provided by the PANGAEA® Data Publisher",
            "name": "Pangaea",
            "timestamp": timestamp.isoformat(),
            "type": "geospatial_data",
        }

        self.corpus_vertex = upsert_vertex(self, GraphCreatorBase._CORPUS_NODE_NAME, dict_)

    def _create_dataset_node(self, dataset: dict[str, Any], timestamp: datetime) -> Document:

        extent = dataset.get("coverage") or {}
        geographic_boundaries = extent.get("geographic_boundaries") or {}
        temporal = extent.get("temporal") or {}

        coverage = {
            "geographic_boundaries": {
                "west": geographic_boundaries.get("west"),
                "south": geographic_boundaries.get("south"),
                "east": geographic_boundaries.get("east"),
                "north": geographic_boundaries.get("north"),
            },
            "temporal": {
                "start": datetime.fromtimestamp(ts, tz=UTC) if (ts := temporal.get("start")) else None,
                "stop": datetime.fromtimestamp(ts, tz=UTC) if (ts := temporal.get("stop")) else None,
            }
        }

        published = datetime.fromtimestamp(ts, tz=UTC) if (ts := temporal.get("published")) else None

        dict_ = {
            "abstract": dataset.get("abstract"),
            "coverage": coverage,
            "datasource": "pangaea",
            "doi": dataset.get("doi"),
            "keywords": dataset.get("specific_keywords"),
            "license": dataset.get("license"),
            "openalex_id": None,
            "published": published,
            "timestamp": timestamp.isoformat(),
            "title": dataset.get("title"),
            "uri": dataset.get("uri"),
        }

        return upsert_vertex(
            self,
            collection_name=PangaeaGraphCreator._DATASET.__name__,
            data=dict_,
            alt_key="doi",
        )

    def _create_author_node(self, author: dict[str, str], timestamp: datetime) -> Document:
        dict_ = {
            "name": f"{author.get("first_name")} {author.get("last_name")}",
            "openalex_id": None,
            "orcid": author.get("orcid"),
            "timestamp": timestamp.isoformat(),
        }

        # Match to ORCiD if exists, otherwise fall back to name
        alt_key = "orcid" if dict_.get("orcid") else "name"

        return upsert_vertex(
            self,
            collection_name=PangaeaGraphCreator._AUTHOR.__name__,
            data=dict_,
            alt_key=alt_key
        )

    def _create_science_keyword_node(
        self, keyword: dict[str, str | bool | None], timestamp: datetime
    ) -> Document:
        dict_ = {
            "description": keyword.get("description"),
            "is_leaf": keyword.get("is_leaf"),
            "level": keyword.get("level"),
            "name": keyword.get("name"),
            "parent_uuid": keyword.get("parent_uuid"),
            "reference": keyword.get("reference"),
            "timestamp": timestamp.isoformat() ,
            "uuid": keyword.get("uuid"),
        }

        return upsert_vertex(
            self,
            collection_name=PangaeaGraphCreator._SCIENCE_KEYWORD.__name__,
            data=dict_,
            alt_key="uuid",
        )

    def _create_update_dataset_node(
        self, dataset_dict: dict[str, Any]
    ) -> BaseGraph:
        """Create or update a dataset node."""
        timestamp = dataset_dict.get("timestamp") or datetime.now(tz=UTC)

        dataset = self._create_dataset_node(dataset_dict, timestamp=timestamp)

        # DatasetNode --BelongsTo--> Corpus
        self.upsert_edge(
            relationName=GraphCreatorBase._BELONGS_TO_RELATION_NAME,
            from_doc=dataset,
            to_doc=self.corpus_vertex,
            edge_attrs={"timestamp": timestamp.isoformat()},
        )

        # DatasetNode --HasAuthor--> AuthorNode
        for author in dataset_dict.get("authors", []):
            assigned_author = self._create_author_node(author, timestamp=timestamp)
            self.upsert_edge(
                relationName=self._HAS_AUTHOR.__name__,
                from_doc=dataset,
                to_doc=assigned_author,
                edge_attrs={"timestamp": timestamp.isoformat()},
            )

        # DatasetNode --HasKeyword--> ScienceKeywordNode
        for sk in dataset_dict.get("science_keywords", []):
            if assigned_keyword := self.science_keywords.get(sk[1]):
                self.upsert_edge(
                    relationName=self._HAS_KEYWORD.__name__,
                    from_doc=dataset,
                    to_doc=self._create_science_keyword_node(assigned_keyword, timestamp=timestamp),
                    edge_attrs={"timestamp": timestamp.isoformat(), "score": sk[2]},
                )
            else:
                print(f"Unable to upsert edge, unknow uuid '{sk[1]}'")

        return self.graph
