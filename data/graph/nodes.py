"""Node definitions for the Science Search knowledge graph."""

from cag.graph_elements.nodes import Field, GenericOOSNode


class EarthObservationAuthor(GenericOOSNode):
    """Authors. May have an openalex_id and/or ORCiD."""

    _fields = {"name": Field(), "openalex_id": Field(), "orcid": Field(), **GenericOOSNode._fields}


class EarthObservationDataset(GenericOOSNode):
    """Datasets belogning to the domain of earth observation."""

    _fields = {
        "abstract": Field(),
        "coverage": {
            "geographic_boundaries": {
                "west": Field(),
                "south": Field(),
                "east": Field(),
                "north": Field(),
            },
            "temporal": {"start": Field(), "stop": Field()},
        },
        "datasource": Field(),
        "doi": Field(),
        "keywords": Field(),
        "license": Field(),
        "openalex_id": Field(),
        "published": Field(),
        "title": Field(),
        "uri": Field(),
        **GenericOOSNode._fields,
    }

    def __init__(self, database, jsonData):
        super().__init__(database, jsonData)
        self.ensureFulltextIndex(["abstract"])
        self.ensureFulltextIndex(["title"])


class EarthObservationPublication(GenericOOSNode):
    """Publications belogning to the domain of earth observation."""

    _fields = {
        "abstract": Field(),
        "datasource": Field(),
        "doi": Field(),
        "keywords": Field(),
        "license": Field(),
        "openalex_id": Field(),
        "published": Field(),
        "title": Field(),
        "uri": Field(),
        **GenericOOSNode._fields,
    }

    def __init__(self, database, jsonData):
        super().__init__(database, jsonData)
        self.ensureFulltextIndex(["abstract"])
        self.ensureFulltextIndex(["title"])


class ScienceKeyword(GenericOOSNode):
    """Keywords from the NASA GCMD Science Keywords taxonomy."""

    _fields = {
        "description": Field(),
        "is_leaf": Field(),
        "level": Field(),
        "name": Field(),
        "parent_uuid": Field(),
        "reference": Field(),
        "uuid": Field(),
        **GenericOOSNode._fields,
    }
