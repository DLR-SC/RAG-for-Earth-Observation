import json
import os
from pathlib import Path

from arango.client import ArangoClient
from dotenv import load_dotenv
from tqdm import tqdm

from gcmd_science_keywords.tagger import KeywordTagger

load_dotenv()
client = ArangoClient(os.getenv("HOST") or "127.0.0.1:8529")
db = client.db("taxo", username="root", password=os.getenv("ARANGO_ROOT_PASSWORD") or "root")
tagger = KeywordTagger(db)

filepath = Path("/localdata1/proj_ows/eoc_geoservice/datasets.json")


with filepath.open("r") as f:
    datasets = json.load(f)


for dataset in tqdm(datasets, desc="Assigning Science Keywords"):
    dataset["science_keywords"] = tagger.get_keywords(dataset.get("abstract"))


with filepath.open("w") as f:
    json.dump(datasets, fp=f, ensure_ascii=False, indent=4)
