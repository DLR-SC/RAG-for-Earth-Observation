"""Preprocessing to isolate relevant works."""
import json
from contextlib import suppress
from pathlib import Path

from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm

from graph.util import count_lines_dir

snapshot = Path("/localdata1/proj_ows/openalex/tagged_and_filtered_snapshot")
files = sorted(snapshot.rglob("*.jsonl"))

print("Calculating total number of works ...")
total_works = count_lines_dir(snapshot, filetype="jsonl")

threshold = 0.375
lookup_table: dict[str, str] = {}

detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH).build()

disc_abstract_lang = 0
disc_abstract_null = 0
disc_threshold = 0
disc_title_lang = 0
disc_title_null = 0
keep = 0

with tqdm(desc="Processing works", total=total_works) as pbar:

    for file in files:
        with file.open("r") as f:
            for line in f:
                work = json.loads(line)

                # Supress index error originating from works without keywords
                with suppress(IndexError):

                    # Highest keyword score does not exceed threshold
                    if work.get("science_keywords", [])[0][2] < threshold:
                        disc_threshold += 1
                        pbar.update()
                        continue

                # Title must not be null
                if not (title := work.get("title")):
                    disc_title_null += 1
                    pbar.update()
                    continue

                # Abstract must not be null
                if not (abstract := work.get("abstract")):
                    disc_abstract_null += 1
                    pbar.update()
                    continue

                # Title language must be ENGLISH
                if detector.detect_language_of(title) != Language.ENGLISH:
                    disc_title_lang += 1
                    pbar.update()
                    continue

                # Abstract language must be ENGLISH
                if detector.detect_language_of(abstract) != Language.ENGLISH:
                    disc_abstract_lang += 1
                    pbar.update()
                    continue

                # Add to lookup table if all criteria are met
                keep += 1
                lookup_table[work.get("id")] = work.get("type")
                pbar.update()


loss = (
    100
    * (disc_abstract_lang + disc_abstract_null + disc_title_lang + disc_title_null + disc_threshold)
    / total_works
)

print("Discarded stats:")
print(f"\tThreshold: {disc_threshold}")
print(f"\tAbtract null: {disc_abstract_null}")
print(f"\tAbstract language: {disc_abstract_lang}")
print(f"\tTitle null: {disc_title_null}")
print(f"\tTitle language: {disc_title_lang}\n")


print(f"Kept a total of {keep} works, loss {loss}%")

with Path(f"/localdata1/proj_ows/openalex/lookup_table_{str(threshold).split(".")[1]}_alt.json").open("w") as f:
    json.dump(lookup_table, fp=f, indent=4)
