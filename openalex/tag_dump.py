"""Assign GCMD Science Keywords to a processed OpenAlex database snapshot."""

import json
import os
import sys
from pathlib import Path

from arango.client import ArangoClient
from arango.database import StandardDatabase
from dotenv import load_dotenv
from tqdm import tqdm

from gcmd_science_keywords.tagger import KeywordTagger

# TODO: It would be very good to have this code parallelized

class OpenAlexTagger:
    """OpenAlex Tagger class."""

    def __init__(self, processed_snapshot: Path, db: StandardDatabase, destination: Path) -> None:
        """Initialize an OpenAlex Tagger."""
        self.files = sorted(processed_snapshot.rglob("*.jsonl"))
        self.db = db

        if not destination.exists():
            msg = f"{destination} does not exist"
            raise FileNotFoundError(msg)

        if not destination.is_dir():
            msg = "'destination' must be a directory"
            raise TypeError(msg)

        self.dest = destination
        self.logfile = destination.joinpath("log")
        self.logfile.touch()

        self.tagger = KeywordTagger(db, min_sen=0.1, min_agg=0.1)

    def __save_progress(self, filepath: Path) -> None:
        """Save the last completed file to log.

        Logfile needed if process crashes or machine is restarted before
        completion.
        """
        with self.logfile.open("a") as f:
            f.write(f"{filepath!s}\n")

    def __restore_abstract(self, abstract_inverted_index: str) -> str:
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

    def __assign_tags(self, work: dict) -> dict:
        aiv = work.get("abstract_inverted_index")
        if not aiv:
            # Should never happen since the works are already filtered
            return work

        abstract = self.__restore_abstract(aiv)
        work["abstract"] = abstract
        work["science_keywords"] = self.tagger.get_keywords(abstract)
        return work


    def process(self) -> None:
        """Process the dump."""
        with self.logfile.open("r") as f:
            tagged_works = [Path(line.strip()) for line in f]
            print(f"Log: {tagged_works[0]} ... {tagged_works[-1]}")

        for file in tqdm(self.files):

            outfile = self.dest.joinpath(*file.parts[-2:])

            # Skip already tagged works
            if file in tagged_works:
                continue

            # Assign tags
            with file.open("r") as f:
                out = [self.__assign_tags(json.loads(line)) for line in f]

            # Save output
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with outfile.open("w") as f:
                for o in out:
                    json.dump(o, fp=f, ensure_ascii=False)
                    f.write("\n")

            self.__save_progress(file)

load_dotenv()
client = ArangoClient(os.getenv("HOST") or "127.0.0.1:8529")
db = client.db(
    "taxo",
    username="root",
    password=os.getenv("ARANGO_ROOT_PASSWORD") or "root"
)

oat = OpenAlexTagger(
    processed_snapshot=Path("/localdata1/proj_ows/openalex/filtered/"),
    db=db,
    destination=Path("/localdata1/proj_ows/openalex/tagged/")
)

oat.process()
