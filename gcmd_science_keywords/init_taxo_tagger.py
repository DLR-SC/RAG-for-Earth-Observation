import json
import os
from pathlib import Path

from arango import ArangoClient
from dotenv import load_dotenv

# Load password from .env file
load_dotenv()
host = os.getenv("HOST")
password = os.getenv("ARANGO_ROOT_PASSWORD")

# Connect to "_system" database
client = ArangoClient(hosts=host)
sys_db = client.db(name="_system", username="root", password=password)

# TODO: Check if database already exists

# Create new database
sys_db.create_database("taxo")
taxo_db = client.db(name="taxo", username="root", password=password)

# Create graph for keywords and edges
graph = taxo_db.create_graph("science_keywords")
vertices_collection = graph.create_vertex_collection("keywords")
edge_collection = graph.create_edge_definition(
    edge_collection="has_parent",
    from_vertex_collections=["keywords"],
    to_vertex_collections=["keywords"]
)

# Read vertices and edges from disk
vertices = Path("gcmd_science_keywords/vertices.json")
with vertices.open("r") as f:
    vertices = json.load(f)

edges = Path("gcmd_science_keywords/edges.json")
with edges.open("r") as f:
    edges = json.load(f)

# Insert vertices and edges
vertices_collection.insert_many(documents=vertices)
edge_collection.insert_many(documents=edges)
