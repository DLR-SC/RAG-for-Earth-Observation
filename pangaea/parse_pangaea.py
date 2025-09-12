import gzip
import json
import os
import re
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from arango import ArangoClient
from arango.database import StandardDatabase
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord
from warcio.statusandheaders import StatusAndHeaders

from gcmd_science_keywords.tagger import KeywordTagger


class GeographicExtent(TypedDict):
    """Geopgraphic coordinates."""

    west: float
    south: float
    east: float
    north: float


class TemporalExtent(TypedDict):
    """Temporal information."""

    start: int | None
    stop: int | None
    published: int | None


class Coverage(TypedDict):
    """Combines geographic and temporal extent."""

    geographic_boundaries: GeographicExtent | None
    temporal: TemporalExtent | None


class Author(TypedDict):
    """Author name and ORCiD."""

    last_name: str
    first_name: str
    orcid: str | None


class Dataset(TypedDict):
    """Pangaea datasets containing all processed data."""

    doi: str
    uri: str
    title: str
    abstract: str
    license: str
    covergae: Coverage | None
    specific_keywords: list[str] | None
    science_keywords: list[tuple[str, float]] | None
    authors: list[Author] | None


# TODO: Change to a class based approach with self variables
tagger: KeywordTagger

total_datasets: int = 0
discarded_datasets: int = 0


def main(args: Namespace) -> None:
    # Create a list containing the paths of all .warc.gz files
    crawls: list[Path] = list(args.input.glob(pattern="*.warc.gz"))

    # Init taxonomy tagger
    print("Initializing taxonomy tagger ...")
    load_dotenv()
    arango_db: StandardDatabase = ArangoClient(hosts=os.getenv("HOST")).db(
        name="taxo",
        username="root",
        password=os.getenv("ARANGO_ROOT_PASSWORD"),
    )
    global tagger
    tagger = KeywordTagger(db=arango_db)
    print("Initialized tagger, continuing with processing crawled datasets")

    # Process all datasets and save them to file
    datasets: list[Dataset] = process_files(crawls)
    save_data(data=datasets, output=args.output)

    print("Statistics of processed Pangaea datasets:")
    print(f"\tProcessed: {total_datasets}")
    print(f"\tDiscarded: {discarded_datasets} (missing abstract)")
    print(f"\tLoss: {((discarded_datasets * 100) / total_datasets):.2f}%")
    print(
        f"\tSaved {total_datasets - discarded_datasets} datasets to {args.output}"
    )


def process_files(files: list[Path]) -> list[Dataset]:
    """Process list of .warc.gz files containg crawled Pangaea data.

    Iterates of a list of Paths pointinf torwards Pangaea data in
    .warc.gz file format obtained by running the"OpenSearch fill-in
    crawls" crawler. Extracts useful data and assigns keywords from the
    GCMD Science Keyword taxonomy using the Taxonomy Tagger.

    Args:
        files (list[Path]): List of paths to the .warc.gz files.

    Returns:
        list[Dataset]: List containing all datasets from the crawl and
        additionally science-keywords.

    """
    datasets: list[Dataset] = []

    for file in tqdm(iterable=files, desc="Processing crawls"):
        with gzip.open(filename=file, mode="rb") as stream:
            for record in ArchiveIterator(fileobj=stream):
                global total_datasets
                total_datasets += 1

                # Processing the record might return None
                # if it has no abstract
                if dataset := process_record(record=record):
                    datasets.append(dataset)
                else:
                    global discarded_datasets
                    discarded_datasets += 1

    # NOTE: Technically it is possible to have an empty list but it is
    # highly unlikeley
    return datasets


def process_record(record: ArcWarcRecord) -> Dataset | None:
    """Process a given record (dataset in raw crawled format).

    If a dataset does not contain an abstract it will be discarded.

    Args:
        record (ArcWarcRecord): Record to be processed.

    Returns:
        Dataset | None: Processed dataset.

    """
    # Parse warc headers and get DOI and URI
    header: StatusAndHeaders = record.rec_headers
    uri: str = header.get_header("warc-target-uri")
    doi: str = uri.split("https://doi.org/")[-1]

    # Create soup from record body
    html_body: str = record.content_stream().read().decode()
    soup: BeautifulSoup = BeautifulSoup(
        markup=html_body, features="html.parser"
    )

    # Extract all data from record body
    content: dict[str, str] = {
        key: value.strip(' "')
        for paragraph in soup.find_all(name="p")
        if (text := paragraph.get_text())
        and (parts := text.split(sep=":", maxsplit=1))
        for key, value in [parts]
    }

    # Datasets without an abstract must be discarded
    if not (abstract := content.get("abstract")):
        return None

    # Parse extent (geographic and temporal)
    extent: Coverage | None = parse_extent(content=content, soup=soup)

    # Get dataset specific keywords
    specific_keywords: list[str] | None = parse_specific_keywords(soup=soup)

    # Get authors and ther ORCiD
    authors: list[dict[str, str]] | None = parse_authors(soup=soup)

    # Assign Science Keywords with taxo tagger
    science_keywords: list[tuple[str, str, float]] | None = (
        assign_science_keywords(abstract=abstract)
    )

    # No science keywords found, discard dataset
    if not science_keywords:
        return None

    return Dataset(
        {
            "doi": doi,
            "uri": uri,
            "title": content.get("title"),
            "abstract": abstract,
            "license": content.get("license"),
            "coverage": extent,
            "specific_keywords": specific_keywords,
            "science_kewords": science_keywords,
            "authors": authors,
        }
    )


def parse_extent(
    content: dict[str, str],
    soup: BeautifulSoup,
) -> Coverage | None:
    """Parse the geograpic and temporal extent of a given dataset.

    Args:
        content (dict[str, str]): Website body containing the dataset extent.
        soup (BeautifulSoup): Soup containing the date of publication.

    Returns:
        Coverage | None: Processed extent, some datasets do not contain
        any information so may be None.

    """
    if not (extent := json.loads(s=content.get("extent", {}))):
        return None

    # extent = json.loads(s=content.get("extent", {}))

    # Try to get extents
    geographic: GeographicExtent | None = parse_geographic_extent(
        extent=extent
    )
    temporal: TemporalExtent | None = parse_remporal_extent(
        soup=soup, extent=extent
    )

    # No coverage specified at all
    if not geographic and not temporal:
        return None

    return Coverage(
        {"geographic_boundaries": geographic, "temporal": temporal}
    )


def parse_geographic_extent(extent: dict) -> GeographicExtent | None:
    """Parse specifically the geographic extent.

    Args:
        extent (dict): Extent containing geographic extent.

    Returns:
        GeographicExtent | None: Processed geographic extent, might be None.

    """
    geographic_extent = extent.get("geographic")

    # There may be no geographic extent aviable
    if not geographic_extent:
        return None

    return GeographicExtent(
        {
            "west": geographic_extent.get("west_bound_longitude"),
            "south": geographic_extent.get("south_bound_latitude"),
            "east": geographic_extent.get("east_bound_longitude"),
            "north": geographic_extent.get("north_bound_latitude"),
        }
    )


def parse_remporal_extent(
    soup: BeautifulSoup, extent: dict
) -> TemporalExtent | None:
    """Parse specifically the temporal extent.

    The date of publication is stored at the metadata section so the
    soup is also needed.

    Args:
        soup (BeautifulSoup): Soup containing the date of publication.
        extent (dict): Extent containing temporal extent.

    Returns:
        TemporalExtent | None: Processed temporal extent, might be None.

    """
    temporal_extent = extent.get("temporal")
    published_tag = soup.find(name="meta", attrs={"name": "published"})

    # There may be no temporal data aviable at all
    if not temporal_extent and not published_tag:
        return None

    # Try to get timestamps
    start: int | None = get_timestamp(
        temporal_extent.get("min_date_time") if temporal_extent else None
    )
    stop: int | None = get_timestamp(
        temporal_extent.get("max_date_time") if temporal_extent else None
    )
    published: int | None = get_timestamp(
        published_tag.get("content") if published_tag else None
    )

    return TemporalExtent(
        {"start": start, "stop": stop, "published": published}
    )


def get_timestamp(time: str | None) -> int | None:
    """Convert a string to unix time format (integer)."""
    return int(datetime.fromisoformat(time).timestamp()) if time else None


def parse_specific_keywords(soup: BeautifulSoup) -> list[str] | None:
    """Parse the keywords specific to a dataset.

    These datasets are asigned by the authors and do not belong to a
    unified concept.

    Args:
        soup (BeautifulSoup): Soup containing the keywords.

    Returns:
        list[str] | None: Specific keyword, might be None.

    """
    keywords_tag = soup.find(name="meta", attrs={"name": "keywords"})
    kw_content: str = keywords_tag.get("content")
    specific_keywords: list[str] = (
        [kw.strip() for kw in kw_content.split(",") if kw.strip()]
        if kw_content
        else None
    )

    return specific_keywords


def parse_authors(soup: BeautifulSoup) -> list[dict[str, str]] | None:
    """Parse the dataset authors.

    Ideally an author has a first and last name and an ORCiD. Some
    authors may not have an ORCiD assigned. Institutions do not match
    the regex in this method and are ignored, other usecases may want
    to include them also.

    Args:
        soup (BeautifulSoup): Soup containing the authors.

    Returns:
        list[dict[str, str]] | None: List auf authors (last_name, first_name, orcid)

    """
    # NOTE: When no match is found the author is ignored
    # (only insitutions so far, maybe include in log)
    author_tags = soup.find_all(name="meta", attrs={"name": "author"})
    authors: list[dict[str, str]] = [
        {
            "last_name": m.group(2),
            "first_name": m.group(1),
            "orcid": m.group(4) or None,
        }
        for tag in author_tags
        if (
            m := re.match(
                pattern=r"(.+)\s(.+)\s(\((.*)\))",
                string=tag.get("content").strip(),
            )
        )
    ]

    return authors


def assign_science_keywords(
    abstract: str,
) -> list[tuple[str, str, float]] | None:
    """Assign science-keywords to a gven abstract.

    These keywords are from the NASA GCMD-Science-Keywords taxonomy and
    are assigned by a taxonomy tagger.

    Args:
        abstract (str): Abstract of a dataset.

    Returns:
        list[tuple[str, str, float]] | None:
        Assigned keywords (name, uuid, score).

    """
    global tagger
    return tagger.get_keywords(text=abstract)


def save_data(data: list[Dataset], output: Path) -> None:
    """Save data to a specified location.

    If the file already exists the user will be promted if he wants to
    override it.

    Args:
        data (list[Dataset]): Datasets to be stored.
        output (Path): Location of the output file.

    """
    # TODO: Make sure specified output file is .json

    if output.exists():
        response: str = (
            input(f"File {output} already exists. Overwrite? (y/N): ")
            .strip()
            .lower()
        )
        if response != "y":
            print("Aborted. File not overwritten.")
            return

    with output.open(mode="w") as fp:
        json.dump(obj=data, fp=fp, indent=4, ensure_ascii=False)

    # with Path.open(file=output, mode="w") as fp:
    #    json.dump(obj=data, fp=fp, indent=4, ensure_ascii=False)


def parse_args() -> Namespace:
    """Define and parse arguments."""
    parser: ArgumentParser = ArgumentParser(
        description="Parser for Pangaea data in .warc format."
    )

    parser.add_argument(
        "--input",
        default=Path(),
        type=Path,
        help="Directory with .warc files.",
    )
    parser.add_argument(
        "--output",
        default=Path(),
        type=Path,
        help="Directory for output files..",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
