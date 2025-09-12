import json
import os
from pathlib import Path

import pandas as pd
from arango.client import ArangoClient
from dotenv import load_dotenv
from tqdm import tqdm

from gcmd_science_keywords.tagger import KeywordTagger

load_dotenv()

client = ArangoClient(hosts=os.getenv("HOST") or "127.0.0.1:8529")
db = client.db(name="taxo", username="root", password=os.getenv("ARANGO_ROOT_PASSWORD") or "")

print("Setting up tagger ...")
tagger = KeywordTagger(db)
print("Tagger setup complete!")

results = []

with Path("evaluation/samples/big_eval_new.csv").open("r") as f:
    data = pd.read_csv(f)

for _, series in tqdm(data.iterrows(), total=len(data)):

    rag_answer = series["rag_answer"]
    zero_shot_answer = series["zero_shot_answer"]

    rag_answer_keywords = tagger.get_keywords(rag_answer, traceback=True)
    zero_shot_answer_keywords = tagger.get_keywords(zero_shot_answer, traceback=True)

    results.append(
        {
            "idx": series["idx"],
            "model": series["model"],
            "rag_answer": {
                "topics": list({kt[1] for kt in (rag_answer_keywords or []) if len(kt) > 1}),
                "terms": list({kt[2] for kt in (rag_answer_keywords or []) if len(kt) > 2}),
                "keywords": [kt[-1] for kt in rag_answer_keywords or []],
            },
            "zero_shot_answer": {
                "topics": list({kt[1] for kt in (zero_shot_answer_keywords or []) if len(kt) > 1}),
                "terms": list({kt[2] for kt in (zero_shot_answer_keywords or []) if len(kt) > 2}),
                "keywords": [kt[-1] for kt in zero_shot_answer_keywords or []],
            },
        }
    )

with Path("evaluation/tagged_results.json").open("w") as f:
    json.dump(obj=results, fp=f, ensure_ascii=False, indent=4)
