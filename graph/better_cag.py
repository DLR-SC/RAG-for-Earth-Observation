"""Improved CAG functions."""
from datetime import UTC, datetime
from typing import Any

from cag.framework import GraphCreatorBase
from pyArango.collection import Document


def upsert_vertex(
    graph_creator: GraphCreatorBase,
    collection_name: str,
    data: dict[str, Any],
    alt_key: str | list[str] | None = None
) -> Document:
    """Improved vertex upsertion, does not overwrite but update values.

    Args:
        graph_creator (GraphCreatorBase): sadf
        collection_name (str): the collection to work on
        data (dict[str, Any]): a dictionary with your data
        alt_key (str | list | None): on which key the upsert should look for
            existing data (if there are multiple, the first match will
            return and it combines all key into a fetch-by-example-query),
            defaults to None

    Returns:
        Document: the upserted document

    """
    if data.get("timestamp") is None:
        data["timestamp"] = datetime.now(tz=UTC).isoformat()

    if vertex := graph_creator.get_document(collection_name, data, alt_key):
        # Update existing vertex
        for key, new_value in data.items():

            # Do not overwrite if the new value is null
            if new_value is None:
                continue

            old_value = vertex[key]

            if key == "datasource":
                # Check if datasources differ and create list from both if
                # doc_datasource is a single one
                if isinstance(old_value, str) and new_value != old_value:
                    vertex[key] = [old_value, new_value]
                    continue

                # If doc already contains multiple datasources append new
                # datasource (check for duplication included)
                if isinstance(old_value, list) and new_value not in old_value:
                    old_value.append(new_value)
                    vertex[key] = old_value
                    continue

            if isinstance(old_value, list) and isinstance(new_value, list):
                vertex[key] = old_value + [item for item in new_value if item not in old_value]
            else:
                vertex[key] = new_value

        # Save updated vertex
        vertex.patch()

        # TODO: Logging

    else:
        # Create a new vertex
        vertex = graph_creator.graph.createVertex(collection_name, data)

    if isinstance(vertex, Document):
        return vertex

    msg = f"Vertex of unkknown type: {type(vertex)}"
    raise TypeError(msg)
