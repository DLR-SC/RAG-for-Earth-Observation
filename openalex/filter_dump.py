"""Filter an OpenAlex snapshot."""

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
                        works.append(document.as_dict())
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
