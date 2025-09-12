import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from better_cag import upsert_vertex
from cag.framework import GraphCreatorBase
from cag.graph_elements.base_graph import BaseGraph
from cag.graph_elements.relations import HasAuthor
from edges import HasKeyword, IsReferencedBy, IsRelatedTo
from nodes import (
    EarthObservationAuthor,
    EarthObservationDataset,
    EarthObservationPublication,
    ScienceKeyword,
)
from pyArango.collection import Document
from pyArango.theExceptions import CreationError, UpdateError
from tqdm import tqdm
from util import count_lines_dir, count_lines_file


class OpenAlexGraphCreator(GraphCreatorBase):
    """Graph creator for OpenAlex data."""

    _name = "OpenAlex Graph Creator"
    _description = "Creates a graph based on the OpenAlex dataset"

    _AUTHOR = EarthObservationAuthor
    _DATASET = EarthObservationDataset
    _PUBLICATION = EarthObservationPublication
    _SCIENCE_KEYWORD = ScienceKeyword

    _HAS_AUTHOR = HasAuthor
    _HAS_KEYWORD = HasKeyword
    _IS_RELATED_TO = IsRelatedTo
    _IS_REFERENCED_BY = IsReferencedBy

    _edge_definitions = [
        {
            "relation": GraphCreatorBase._BELONGS_TO_RELATION_NAME,
            "from_collections": [_DATASET],
            "to_collections": [GraphCreatorBase._CORPUS_NODE_NAME],
        },
        {
            "relation": _HAS_AUTHOR,
            "from_collections": [_DATASET, _PUBLICATION],
            "to_collections": [_AUTHOR],
        },
        {
            "relation": _HAS_KEYWORD,
            "from_collections": [_DATASET, _PUBLICATION],
            "to_collections": [_SCIENCE_KEYWORD],
        },
        {
            "relation": _IS_RELATED_TO,
            "from_collections": [_DATASET, _PUBLICATION],
            "to_collections": [_DATASET, _PUBLICATION],
        },
        {
            "relation": _IS_REFERENCED_BY,
            "from_collections": [_DATASET, _PUBLICATION],
            "to_collections": [_DATASET, _PUBLICATION],
        },
    ]

    def init_graph(self) -> None:
        """Initialize the graph."""
        print("Setting up OpenAlex graph creator ...")
        self._set_corpus_node(self.now)
        self.logfile = Path("/localdata1/proj_ows/openalex/log")

        # Logger for failed upserts
        logging.basicConfig(
            filename="/home/schl_b2/code/ows_eo_kg/graph/log",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # Lookup table, guarantees the option for an upsert
        with Path("/localdata1/proj_ows/openalex/lookup_table_375.json").open("r") as f:
            self.lookup_table = json.load(f)

        with self.logfile.open("r") as f:
            processed_works = [Path(line.strip()) for line in f]
            if len(processed_works) >= 1:
                print(f"Log: {processed_works[0]} ... {processed_works[-1]}")

        with Path("gcmd_science_keywords/independent_vertices.json").open("r") as f:
            self.science_keywords: dict[str, dict[str, str | bool | None]] = {
                item["uuid"]: item for item in json.load(f)
            }

        openalex_files = sorted(Path(self.corpus_file_or_dir).rglob("*.jsonl"))
        total_works = count_lines_dir(self.corpus_file_or_dir, filetype="jsonl")

        with tqdm(desc="Processing OpenAlex works", total=total_works) as pbar:
            for file in openalex_files:

                # Works have been already added to graph
                if file in processed_works:
                    pbar.update(count_lines_file(file))
                    continue

                with file.open("r") as f:
                    for line in f:
                        work = json.loads(line)
                        try:
                            self._create_update_work(work)
                        except CreationError as e:
                            logging.warning(f"Unable to create vertex - {work.get("id")} {e}")
                        except UpdateError as e:
                            logging.warning(f"Unable to update vertex - {work.get("id")} {e}")
                        pbar.update()

                self._save_progress(file)

    def update_graph(self, timestamp: datetime | None = None) -> None:
        """Update an exisiting graph."""
        self.init_graph()

    def _set_corpus_node(self, timestamp: datetime) -> None:
        dict_ = {
            "_key": "OpenAlexData",
            "created_on": self.now,
            "description": "Dataset and publication metadata provided by OpenAlex",
            "name": "OpenAlex",
            "timestamp": timestamp.isoformat(),
            "type": "data",
        }

        self.corpus_vertex = upsert_vertex(self, GraphCreatorBase._CORPUS_NODE_NAME, dict_)

    def _save_progress(self, filepath: Path) -> None:
        """Save the last completed file to log.

        Logfile needed if process crashes or machine is restarted before
        completion.
        """
        with self.logfile.open("a") as f:
            f.write(f"{filepath!s}\n")

    def _create_work_node(self, work: dict[str, Any], timestamp: datetime) -> Document:
        doi = raw.split("https://doi.org/")[-1] if (raw := work.get("doi")) else None

        published = (
            datetime.strptime(publication, r"%Y-%m-%d").astimezone(tz=UTC).isoformat()
            if (publication := work.get("publication_date"))
            else None
        )

        # Keywords are common across works, ignore this trait for now
        keywords = [k.get("display_name") for k in work.get("keywords", [])]
        keywords = None if len(keywords) == 0 else keywords

        dict_ = {
            "abstract": work.get("abstract"),
            "coverage": {  # OpenAlex des not provide geospatial data
                "geographic_boundaries": {"west": None, "south": None, "east": None,  "north": None},
                "temporal": {"start": None, "stop": None}
            },
            "datasource": "openalex",
            "doi": doi,
            "keywords": keywords,
            "license": work.get("primary_lcoation", {}).get("license"),
            "openalex_id": work.get("id"),
            "published": published,
            "timestamp": timestamp.isoformat(),
            "title": work.get("title"),
            "uri": work.get("id"),
        }

        if work.get("type") == "article":
            return upsert_vertex(
                self,
                collection_name=OpenAlexGraphCreator._PUBLICATION.__name__,
                data=dict_,
                alt_key="openalex_id",
            )

        if work.get("type") == "dataset":
            return upsert_vertex(
                self,
                collection_name=OpenAlexGraphCreator._DATASET.__name__,
                data=dict_,
                alt_key="openalex_id",
            )

        msg = f"Unknown work type '{type}'"
        raise ValueError(msg)

    def _create_author_node(self, authorship: dict[str, Any], timestamp: datetime) -> Document:
        dict_ = {
            "name": authorship.get("display_name"),
            "openalex_id": authorship.get("id"),
            "orcid": authorship.get("orcid"),
            "timestamp": timestamp.isoformat(),
        }

        # Match to ORCiD if exists, otherwise fall back to openalex_id
        alt_key = "orcid" if dict_.get("orcid") else "openalex_id"

        return upsert_vertex(
            self,
            collection_name=OpenAlexGraphCreator._AUTHOR.__name__,
            data=dict_,
            alt_key=alt_key,
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
            collection_name=OpenAlexGraphCreator._SCIENCE_KEYWORD.__name__,
            data=dict_,
            alt_key="uuid",
        )

    def _create_dummy_node(self, id_: str, type_: str, timestamp: datetime) -> Document:
        """Create and upsert a dummy vertex."""
        collection = self._DATASET if type_ == "dataset" else self._PUBLICATION

        data = {
            "abstract": None,
            "coverage": {
                "geographic_boundaries": {
                    "west": None,
                    "south": None,
                    "east": None,
                    "north": None,
                },
                "temporal": {"start": None, "stop": None},
            },
            "datasource": "openalex",
            "doi": None,
            "keywords": None,
            "license": None,
            "openalex_id": id_,
            "published": None,
            "timestamp": timestamp.isoformat(),
            "title": None,
            "uri": None,
        }

        return upsert_vertex(
            self,
            collection_name=collection.__name__,
            data=data,
            alt_key="openalex_id"
        )

    def _create_update_work(self, work_dict: dict[str, Any]) -> BaseGraph:
        """Create or update an OpenAlex work -> Dataset or Publication."""
        # Work does not meet the criteria
        if work_dict.get("id") not in self.lookup_table:
            return self.graph

        timestamp = work_dict.get("timestamp") or datetime.now(tz=UTC)

        work = self._create_work_node(work_dict, timestamp=timestamp)

        # (Dataset|Publication)Node --BelongsTo--> Corpus
        self.upsert_edge(
            GraphCreatorBase._BELONGS_TO_RELATION_NAME,
            from_doc=work,
            to_doc=self.corpus_vertex,
            edge_attrs={"timestamp": timestamp.isoformat()},
        )

        # (Dataset|Publication)Node --HasAuthor--> AuthorNode
        for authorship in work_dict.get("authorships", []):
            author = self._create_author_node(authorship, timestamp=timestamp)
            self.upsert_edge(
                self._HAS_AUTHOR.__name__,
                from_doc=work,
                to_doc=author,
                edge_attrs={"timestamp": timestamp.isoformat()},
            )

        # (Dataset|Publication) --HasKeyword--> ScienceKeywordNode
        for sk in work_dict.get("science_keywords", []):
            if assigned_keyword := self.science_keywords.get(sk[1]):
                self.upsert_edge(
                    self._HAS_KEYWORD.__name__,
                    from_doc=work,
                    to_doc=self._create_science_keyword_node(assigned_keyword, timestamp=timestamp),
                    edge_attrs={"timestamp": timestamp.isoformat(), "score": sk[2]},
                )
            else:
                print(f"Unable to upsert edge, unknow uuid '{sk[1]}'")

        # (Dataset|Publication) --IsReferencedBy--> (Dataset|Publiaction)
        for id_ in work_dict.get("referenced_works", []):
            if ref := self.lookup_table.get(id_):
                self.upsert_edge(
                    self._IS_REFERENCED_BY.__name__,
                    from_doc=self._create_dummy_node(id_, type_=ref, timestamp=timestamp),
                    to_doc=work,
                    edge_attrs={"timestamp": timestamp.isoformat()},
                )

        # (Dataset|Publication) --IsRelatedTo--> (Dataset|Publiaction)
        for id_ in work_dict.get("related_works", []):
            if rel := self.lookup_table.get(id_):
                self.upsert_edge(
                    self._IS_RELATED_TO.__name__,
                    from_doc=work,
                    to_doc=self._create_dummy_node(id_, type_=rel, timestamp=timestamp),
                    edge_attrs={"timestamp": timestamp.isoformat()},
                )

        return self.graph
