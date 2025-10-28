import json
from pathlib import Path

from pystac import Collection
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry


def get_catalog(url: str) -> Client:
    """Get a STAC catalog."""
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
    stac_api_io = StacApiIO(max_retries=retry)
    return Client.open(url=url, stac_io=stac_api_io)


def get_temporal_extent(collection: Collection) -> dict:
    """Get the temporal extent of a collection."""
    extent = collection.extent.temporal.intervals[0]

    start = extent[0].isoformat() if extent[0] is not None else None
    stop = extent[1].isoformat() if extent[1] is not None else None

    return {"start": start, "stop": stop}


def get_dataset(catalog: Client, id_: str) -> dict:
    """Get a dataset from a STAC catalog/client."""
    collection = catalog.get_collection(id_)

    return {
        "abstract": collection.description,
        "coverage": {
            "geographic_boundaries": {
                "west": collection.extent.spatial.bboxes[0][0],
                "south": collection.extent.spatial.bboxes[0][1],
                "east": collection.extent.spatial.bboxes[0][2],
                "north": collection.extent.spatial.bboxes[0][3],
            },
            "temporal": get_temporal_extent(collection=collection),
        },
        "license": collection.license,
        "local_keywords": collection.keywords,
        "science_keywords": None,  # Added later on
        "title": collection.title,
        "uri": collection.self_href.removesuffix("?f=application/json"),
    }


def main() -> None:
    catalog = get_catalog(url="https://geoservice.dlr.de/eoc/ogc/stac/v1/")
    data = [get_dataset(catalog=catalog, id_=c.id) for c in catalog.get_collections()]

    with Path("/localdata1/proj_ows/eoc_geoservice/datasets.json").open("w") as f:
        json.dump(data, fp=f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
