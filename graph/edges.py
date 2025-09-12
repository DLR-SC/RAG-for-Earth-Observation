"""Edge definitions for the Science Search knowledge graph."""
from cag.graph_elements.relations import Field, GenericEdge


class IsParent(GenericEdge):
    """Science Keywords are part of a taxonomy and therefore have parents."""

    _fields = GenericEdge._fields


class HasKeyword(GenericEdge):
    """Datasets and publications get science keywords assigned."""

    _fields = {
        "score": Field(),
        **GenericEdge._fields
    }


class IsReferencedBy(GenericEdge):
    """Datasets and Publications can reference each other."""

    _fields = GenericEdge._fields


class IsRelatedTo(GenericEdge):
    """Datasets and publications can be related to one another."""

    _fields = GenericEdge._fields
