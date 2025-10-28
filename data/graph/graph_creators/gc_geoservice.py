import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from better_cag import upsert_vertex
from cag.framework import GraphCreatorBase
from cag.graph_elements.base_graph import BaseGraph
from edges import HasKeyword
from nodes import EarthObservationDataset, ScienceKeyword
from pyArango.collection import Document
from tqdm import tqdm


class GeoserviceGraphCreator(GraphCreatorBase):
    """Graph creator for the EOC Geoservice data."""

    _name = "EOC Geoservice Graph Creator"
    _description = "Creates a graph based on EOC Geoservice metadata"

    _DATASET = EarthObservationDataset
    _SCIENCE_KEYWORD = ScienceKeyword

    _HAS_KEYWORD = HasKeyword

    _edge_definitions = [
        {
            "relation": GraphCreatorBase._BELONGS_TO_RELATION_NAME,
            "from_collections": [_DATASET],
            "to_collections": [GraphCreatorBase._CORPUS_NODE_NAME],
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

        geoservice_files = sorted(Path(self.corpus_file_or_dir).rglob("*.json"))

        # Count files
        total = sum(len(json.load(file.open("r"))) for file in geoservice_files)

        with tqdm(geoservice_files, desc="Processing Geoservice files", total=total) as pbar:
            for file in geoservice_files:
                with file.open("r") as f:
                    for dataset in json.load(f):
                        self._create_update_dataset_node(dataset)
                        pbar.update()

    def update_graph(self, timestamp: datetime | None = None) -> None:
        """Update an eisiting graph."""
        self.init_graph()

    def _set_corpus_node(self, timestamp: datetime) -> None:
        dict_ = {
            "_key": "GeoserviceData",
            "created_on": datetime.now(tz=UTC).isoformat(),
            "description": "Geospatial datasets provided by the Earth Observation Center (EOC) of the German Aerospace Center (DLR)",
            "name": "Geoservice",
            "timestamp": timestamp.isoformat(),
            "type": "geospatial_data",
        }

        self.corpus_vertex = upsert_vertex(self, GraphCreatorBase._CORPUS_NODE_NAME, dict_)

    def _create_dataset_node(self, dataset: dict[str, Any], timestamp: datetime) -> Document:
        dict_ = {
            "abstract": dataset.get("abstract"),
            "coverage": {
                "geographic_boundaries": {
                    "west": dataset.get("coverage", {}).get("geographic_boundaries", {}).get("west"),
                    "south": dataset.get("coverage", {}).get("geographic_boundaries", {}).get("south"),
                    "east": dataset.get("coverage", {}).get("geographic_boundaries", {}).get("east"),
                    "north": dataset.get("coverage", {}).get("geographic_boundaries", {}).get("north"),
                },
                "temporal": {
                    "start": dataset.get("coverage", {}).get("temporal", {}).get("start"),
                    "stop": dataset.get("coverage", {}).get("temporal", {}).get("stop"),
                },
            },
            "datasource": "eoc-geoservice",
            "doi": None,
            "keywords": dataset.get("local_keywords"),
            "license": dataset.get("license"),
            "openalex_id": None,
            "published": None,
            "timestamp": timestamp.isoformat(),
            "title": dataset.get("title"),
            "uri": dataset.get("uri"),
        }

        return upsert_vertex(
            self,
            collection_name=GeoserviceGraphCreator._DATASET.__name__,
            data=dict_,
            alt_key=["abstract", "datasource", "title", "published"],
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
            "timestamp": timestamp.isoformat(),
            "uuid": keyword.get("uuid"),
        }

        return upsert_vertex(
            self,
            collection_name=GeoserviceGraphCreator._SCIENCE_KEYWORD.__name__,
            data=dict_,
            alt_key="uuid",
        )

    def _find_dict_by_uuid(
        self, data: list[dict[str, str | bool | None]], target_uuid: str
    ) -> dict[str, str | bool | None] | None:
        """In a list of dicts find the one with matching UUID."""
        return next((item for item in data if item.get("uuid") == target_uuid), None)

    def _create_update_dataset_node(self, dataset_dict: dict[str, Any]) -> BaseGraph:
        """Create or update a dataset node.

        Datasets are only coinnected to Science Keywords. The author is always
        the DLR, which has no author node since author are always natural
        persons with ORCiDs if aviable.

        The EOC Geoservice is the only datasource which is allowed to have nodes
        not connected to any Sciene Keywords. This restriction usually ensures
        that the data is related to the domain of earth observation, but since
        the Geoservice datasets are STAC collections this is already ensured.
        """
        timestamp = dataset_dict.get("timestamp") or datetime.now(tz=UTC)

        dataset = self._create_dataset_node(dataset_dict, timestamp=timestamp)

        # DatasetNode --BelongsTo--> Corpus
        self.upsert_edge(
            relationName=GraphCreatorBase._BELONGS_TO_RELATION_NAME,
            from_doc=dataset,
            to_doc=self.corpus_vertex,
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
