import ast
import json
import re

meta_keys = []
all_keys = []


def parse_documents(raw_str):
    """
    Loads documents from either:
    - JSON array (produced by json.dumps)
    - Old Python str(docs) representation.
    Returns a list of dicts.
    """
    raw_str = raw_str.strip()

    # 1. Try JSON first
    try:
        data = json.loads(raw_str)
        if isinstance(data, list) and all(isinstance(d, dict) for d in data):
            return data
    except json.JSONDecodeError:
        pass  # Not valid JSON

    # 2. Fallback to old format
    return parse_str_documents(raw_str)


def parse_str_documents(raw_str):
    """
    Parse a stringified Python list of Document(...) into a list of dictionaries.
    Handles multi-line content, mixed quotes, and metadata safely.
    """
    # Remove surrounding brackets if present
    raw_str = raw_str.strip()
    if raw_str.startswith("[") and raw_str.endswith("]"):
        raw_str = raw_str[1:-1]

    # Split by document boundaries
    blocks = re.split(r"\),\s*Document\(", raw_str)
    # Clean start and end
    blocks[0] = blocks[0].lstrip("Document(")
    blocks[-1] = blocks[-1].rstrip(")")

    documents = []
    for block in blocks:
        # --- Extract id ---
        id_match = re.search(r"id=['\"](.*?)['\"]", block)
        doc_id = id_match.group(1) if id_match else None

        # --- Extract metadata dict ---
        meta_match = re.search(r"metadata=(\{.*?\})", block, re.DOTALL)
        metadata = {}
        if meta_match:
            try:
                metadata = ast.literal_eval(meta_match.group(1))
            except Exception:
                metadata = {"_metadata_parse_error": meta_match.group(1)}

        # --- Extract page_content ---
        page_start = block.find("page_content=")
        page_content = ""
        if page_start != -1:
            raw_content = block[page_start + len("page_content=") :].strip()
            if raw_content.startswith(("'", '"')):
                quote_char = raw_content[0]
                end_idx = raw_content.rfind(quote_char)
                page_content = raw_content[1:end_idx]
            else:
                page_content = raw_content

        documents.append(
            {"id": doc_id, "metadata": metadata, "page_content": page_content}
        )

    return documents


par_keywords = """
# Related Topics' Descriptions
"""

par_keywords_content = """

Topic: {topic}
Description: {description}
---
"""


par_pubs = """
# Related Documents
"""

par_pubs_content = """
- Title: {title}
- Content: {content}
---
"""


def structure_context(unstructured_context_str):
    # convert context to a list
    context_lst = parse_documents(unstructured_context_str)

    # Loop over each entry and build the string
    all_keywords = []
    unique_keywords = []

    all_pubs = []
    unique_pubs = []

    for x in context_lst:
        # metadata
        meta_keys.extend(x["metadata"].keys())
        all_keys.extend(x.keys())
        if "type" in x["metadata"].keys() and x["metadata"]["type"] == "keyword":
            if (
                x["metadata"]["name"] not in unique_keywords
                and x["page_content"].strip() != ""
            ):
                all_keywords.append(
                    par_keywords_content.format(
                        topic=x["metadata"]["name"], description=x["page_content"]
                    )
                )
                unique_keywords.append(x["metadata"]["name"])
        elif "title" in x["metadata"].keys():
            if (
                x["metadata"]["title"] not in unique_pubs
                and x["page_content"].strip() != ""
            ):
                all_pubs.append(
                    par_pubs_content.format(
                        title=x["metadata"]["title"], content=x["page_content"]
                    )
                )
                unique_pubs.append(x["metadata"]["title"])
        else:
            print(x["metadata"].keys())

    keywords_str = (
        par_keywords.upper() + "".join(all_keywords) if len(all_keywords) > 0 else ""
    )
    pubs_str = par_pubs.upper() + "".join(all_pubs) if len(all_pubs) > 0 else ""
    context_str = pubs_str + keywords_str
    return context_str


def _add_structure_context(row):
    row["structured_context"] = structure_context(row["context"])
    return row
