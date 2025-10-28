"""Process a database dump / snapshot for use in the knowledge graph."""

import gzip
import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import simdjson
from tqdm import tqdm


class OpenAlexProcessor:
    """OpenAlex Processor class."""

    def __init__(self, snapshot: Path, topics: set[str], output_dir: Path) -> None:
        """Initialize an OA-Processor."""
        self.parser: simdjson.Parser = simdjson.Parser()
        self.snapshot: Path = snapshot
        self.topics: set[str] = topics  # Set with topic IDs

        if not output_dir.exists():
            msg = f"{output_dir} does not exist"
            raise FileNotFoundError(msg)

        if not output_dir.is_dir():
            msg = "'output_dir' must be a directory"
            raise TypeError(msg)

        self.output_dir: Path = output_dir

    def __validate_work(self, work: simdjson.Object) -> bool:  # noqa: PLR0911
        """Check if a work meets the defined criteria."""
        # Work has to be either an article or dataset
        if (work_type := work.get("type")) not in {"article", "dataset"}:
            return False

        # Primary Topics has to match given set
        topic = work.get("primary_topic")
        if not isinstance(topic, simdjson.Object) or topic.get("id") not in self.topics:
            return False

        # Work must have an abstract
        if not work.get("abstract_inverted_index"):
            return False

        # Work must be in english
        if work.get("language") != "en":
            return False

        # Only works published after the first STAC Item
        pub_date = work.get("publication_date")
        if not isinstance(pub_date, str) or pub_date < "1984-04-07":
            return False

        # Check for datasets completed, work validated
        if work_type == "dataset":
            return True

        # Articles have to be peer reviewed
        locations = work.get("locations", [])
        return (
            isinstance(locations, Iterable)
            and any(location.get("is_accepted")
            for location in locations
            if isinstance(location, simdjson.Object))
        )

    def __restore_abstract(self, abstract_inverted_index: simdjson.Object) -> str:
        """Restore an abstract from the inverted index format.

        NOTE: Code is from Roxanne, worked so far.
        """
        positions_tmp = []

        for lst in list(abstract_inverted_index.values()):
            positions_tmp += lst

        lst = [""] * (max(positions_tmp) + 1)

        for term in abstract_inverted_index:
            for posistions in abstract_inverted_index[term]:
                lst[posistions] = term

        return " ".join(lst)

    def __extract_authors(self, authorships: Iterable[simdjson.Object]) -> list[dict[str, str]] | None:
        """Extract author information from the authorship list."""
        authors = []

        for authorship in authorships:
            author = authorship.get("author")
            if not isinstance(author, simdjson.Object):
                continue

            name = author.get("display_name")
            if not isinstance(name, str):
                continue

            # Try to split the name in a first and last
            m = re.match(r"(.+)\s(.+)", string=name)
            last_name = m.group(2) if m else None
            first_name = m.group(1) if m else None

            url_orcid = author.get("orcid")
            orcid = url_orcid.split("https://orcid.org/")[-1] if isinstance(url_orcid, str) else None

            # Unique OpenAlex ID, needed later for graph creation
            oa_id = author.get("id")

            authors.append({
                "last_name": last_name,
                "first_name": first_name,
                "original_name": name,  # Fallback for failed / incorrect namesplit
                "orcid": orcid,
                "oa_id": oa_id
            })

        return authors or None

    def __work_to_dict(self, work: simdjson.Object) -> dict:

        # DOI might not be provided
        # NOTE: DOI and URI are redundant, keep anyway for now
        w_uri = work.get("doi")
        w_doi = w_uri.split("https://doi.org/")[-1] if isinstance(w_uri, str) else None

        # OpenAlex traits
        w_oa_id = work.get("id")
        w_oa_type = work.get("type")

        w_title = work.get("title")

        # Get abstract
        w_abstract_inverted_index = work.get("abstract_inverted_index")
        if not isinstance(w_abstract_inverted_index, simdjson.Object):
            msg = f"Abstract of work {w_uri} malformed (you should never see this)"
            raise TypeError(msg)
        w_abstract = self.__restore_abstract(w_abstract_inverted_index)

        # Get license information from best open access source
        w_best_oa = work.get("best_oa_location")
        w_license = w_best_oa.get("license") if isinstance(w_best_oa, simdjson.Object) else None

        # Get date of publication
        w_pub = work.get("publication_date")
        w_published = int(datetime.fromisoformat(w_pub).timestamp()) if isinstance(w_pub, str) else None

        # Get work specific keywords
        w_keywords_list = work.get("keywords")
        w_specific_keywords = (
            [kw.get("display_name") for kw in w_keywords_list if isinstance(kw, simdjson.Object)]
            if isinstance(w_keywords_list, Iterable) else None
        )

        # Get authors
        w_authorships = work.get("authorships")
        w_authors = (
            self.__extract_authors(w_authorships) # type: ignore (type check below seems to be too advanced for system)
            if isinstance(w_authorships, Iterable)
            and all(isinstance(item, simdjson.Object)
            for item in w_authorships)
            else None
        )

        # Get referenced works
        w_ref_arr = work.get("referenced_works")
        w_referenced_works = (
            w_ref_arr.as_list()
            if isinstance(w_ref_arr, simdjson.Array)
            and len(w_ref_arr) > 0
            else None
        )

        # Get related works
        w_rel_arr = work.get("related_works")
        w_related_works = (
            w_rel_arr.as_list()
            if isinstance(w_rel_arr, simdjson.Array)
            and len(w_rel_arr) > 0
            else None
        )

        return {
            "doi": w_doi,
            "uri": w_uri,
            "oa_id": w_oa_id,  # Relevant for linking OA works
            "oa_type": w_oa_type,  # Dataset or article, needed for GUI filtering
            "title": w_title,
            "abstract": w_abstract,
            "license": w_license,
            "coverage": {
                "geographic_boundaries": None,  # Not provided
                "temporal": {
                    "start": None,  # Not provided
                    "stop": None,  # Not provided
                    "published": w_published
                }
            },
            "specific_keywords": w_specific_keywords,
            "science_keywords": None,  # Will later be added by the taxo tagger
            "authors": w_authors,
            "referenced_works": w_referenced_works,
            "related_works": w_related_works
        }

    def process_works(self) -> None:
        """Process all work items in snapshot."""
        works_dir = self.snapshot.joinpath("data", "works")
        archives = sorted(works_dir.rglob("*.gz"))

        # Iterate over all work archives
        for archive in tqdm(archives, desc="Processing works", total=len(archives)):

            # Check if output file already exists, skip if so
            out_file = self.output_dir.joinpath(
                archive.parts[-2],
                Path(archive.parts[-1]).with_suffix("").with_suffix(".jsonl")
            )

            # Prevoius (crashed) run may already have processed the current file
            if out_file.exists():
                continue

            # Save all works of archive and save after each iteration
            works = []

            # Unpack current archive (part_xxx.gz)
            with gzip.open(archive) as f:
                for line in f.read().splitlines():
                    document = self.parser.parse(line)
                    if isinstance(document, simdjson.Object) and self.__validate_work(document):
                        works.append(self.__work_to_dict(document))
                    del document

            if len(works) > 0:
                out_file.parent.mkdir(parents=True, exist_ok=True)  # Make sure parent dirs exist
                with out_file.open("w") as f:
                    for w in works:
                        json.dump(w, f, ensure_ascii=False)
                        f.write("\n")


with Path("openalex/topics/tagged_topics.json").open("r") as f:
    topics = json.load(f)

# Only use topics where the threshhold is greater or equal to 0.4
ids = {topic.get("topic_id") for topic in topics if topic.get("highest_keyword_score") >= 0.4}

oa_processor = OpenAlexProcessor(
    Path("/localdata1/proj_ows/openalex/snapshot/"),
    topics=ids,
    output_dir=Path("/localdata1/proj_ows/openalex/processed/")
)

oa_processor.process_works()
