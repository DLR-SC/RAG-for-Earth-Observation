#!/bin/bash
# Remove all indents from all .jsonl files in directory.
# Needed since tagging was accidentally done with indents which is not valid .jsonl format.

find /localdata1/proj_ows/openalex-processed/tagged -type f -name "*.jsonl" | while read file; do
  jq -c . "$file" > "$file.fixed" && mv "$file.fixed" "$file"
done
