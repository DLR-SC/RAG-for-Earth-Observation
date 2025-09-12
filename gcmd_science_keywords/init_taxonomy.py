"""Fetches and stores the NASA GCMD Science Keywords taxonomy."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from rdflib import Graph
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean text from various encodings.

    - first_pass:     remove sequences and HTML tags
    - second_pass:    remove HTML embeddings for links
    - third_pass:     remove resulting whitespaces if more than one and strip outer edges
    """
    first_pass = re.sub(r"<br>|<p>|<i>|<\/i>|<b>|<\/b>|<\/body>|<ul>|<\/ul>|<li>|<\/u>|\n|\r|\\", " ", text, flags=re.IGNORECASE)
    second_pass = re.sub(r"(<a\s+href=(.*)>)(.*)(<\/a>|<\/HTML>)", r"\3", first_pass, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", second_pass).strip()


g = Graph()

# Fetch the taxonomy in .rdf format from NASA
print("Fetching taxonomy ...")
page_num = 1
while True:
    url = f"https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/sciencekeywords/?format=rdf&page_size=2000&page_num={page_num}"
    response = requests.get(url, timeout=60)

    if response.status_code == 200:
        content = response.text
        g.parse(data=content, format="xml")
        page_num += 1
    else:
        # Fails after iterating over all pages
        print(f"Failed to fetch page {page_num}: {response.status_code}")
        break

# Combine pages into one big graph and define root
taxonomy =  g.serialize(format="xml")
root = ET.fromstring(text=taxonomy)

# Set namespaces for XML parsing
namespaces = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "gcmd": "https://gcmd.earthdata.nasa.gov/kms#"
}

independent_vertices = []
vertices = []
edges = []

# Identify all keywords and edges
for description in tqdm(root.findall(path="rdf:Description", namespaces=namespaces), desc="Parsing keywords"):

    # Get UUID
    url = description.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
    if url is None:
        # Current item is not a keyword
        continue
    else:
        # Split UUID from URL
        uuid = url.rsplit(sep="/", maxsplit=1)[1]

    # Get name (and force capital letters to ensure consistency)
    name = description.find("skos:prefLabel", namespaces=namespaces).text.upper()

    # Get parent UUID
    broader = description.find("skos:broader", namespaces=namespaces)
    if broader is None:
        # Found top keyword, no parent above
        parent_uuid = None
    else:
        parent_url = broader.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
        parent_uuid = parent_url.rsplit("/", maxsplit=1)[1]

    # Get descrption (here "definition" to avoid name conflict)
    raw_definition = description.find("skos:definition", namespaces=namespaces)
    definition = clean_text(raw_definition.text) if raw_definition is not None else None

    # Get reference (p_reference -> pointer to reference)
    p_reference = description.find("gcmd:reference", namespaces=namespaces)
    if p_reference is None:
        # Keyword has no reference
        reference = None
    else:
        # The reference is stored inside another node
        node_id = p_reference.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}nodeID")
        reference_node = root.find(f".//rdf:Description[@rdf:nodeID='{node_id}']", namespaces=namespaces)
        reference = clean_text(reference_node.find("gcmd:text", namespaces=namespaces).text)

    # Check if keyword has any children
    is_leaf = description.find(path="skos:narrower", namespaces=namespaces) is None

    # Create independet vertices for use with CAG
    independent_vertices.append({
        "description": definition,
        "is_leaf": is_leaf,
        "level": None,
        "name": name,
        "parent_uuid": parent_uuid,
        "reference": reference,
        "uuid": uuid,
    })

    # Create vertices and edges for the taxonomy tagger
    vertices.append({
        "_key": uuid,
        "name": name,
        "description": definition,
        "reference": reference,
        "is_leaf": is_leaf
    })

    edges.append({
        "_from": f"keywords/{uuid}",
        "_to": f"keywords/{parent_uuid}"
    })

with Path("gcmd_science_keywords/vertices.json").open("w") as f:
    json.dump(vertices, fp=f, ensure_ascii=False, indent=4)

with Path("gcmd_science_keywords/edges.json").open("w") as f:
    json.dump(edges, fp=f, ensure_ascii=False, indent=4)

def find_dict_by_key(data: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    """Given a list of dicts and an unique key find the matching dict."""
    return next((item for item in data if item.get(key) == value), None)

# Get the keyword levels
for keyword in tqdm(independent_vertices, desc="Identify keyword levels"):

    steps = 0
    current = keyword

    while keyword.get("level") is None:

        parent_uuid = current.get("parent_uuid")
        if parent_uuid is None:
            # Found top level item, no level to assign
            keyword["level"] = "TOP"
            break

        # EARTH SCIENCE
        if current.get("uuid") == "e9f67a66-e9fc-435c-b720-ae32a2c3d8f5":
            levels = [
                "Category",
                "Topic",
                "Term",
                "Variable Level 1",
                "Variable Level 2",
                "Variable Level 3",
                "Detailed Variable",
            ]

            keyword["level"] = levels[steps]

        # EARTH SCIENCE SERVCIES
        elif current.get("uuid") == "894f9116-ae3c-40b6-981d-5113de961710":
            levels = [
                "Service Category",
                "Service Topic",
                "Service Term",
                "Service Variable",
                "Detailed Variable",
            ]

            keyword["level"] = levels[steps]

        parent = find_dict_by_key(independent_vertices, "uuid", parent_uuid)
        if parent is None:
            msg = "Unable to get parent for non-top-level item"
            raise RuntimeError(msg)

        current = parent
        steps += 1


with Path("gcmd_science_keywords/independent_vertices.json").open("w") as f:
    json.dump(independent_vertices, fp=f, ensure_ascii=False, indent=4)
