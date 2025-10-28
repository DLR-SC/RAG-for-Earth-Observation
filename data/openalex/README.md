# OpenAlex Processing Steps

## 1. Filtering Topics

`openalex/topics/evaluation.ipynb`

OpenAlex works are classified by "Topics". We can use this to our advantage by assigning Science Keywords to these topcis. Each topic with a highest keyword score of $\geq 0.4$ is deemed relevant to our taxonomy.

## 2. Filtering works

`openalex/filter_dump.py`

We are only interested in a small subset of the OpenAlex snapshot so only works meeting the following criteria are saved:

- `type`: work has to be either a `dataset` or `article`
- `primary_topic` has to match the set of relevant topics
- work must have an abstract
- `language` must be `en` (english)
- `publication_date`: must be published after `1984-04-07` (date of first STAC item)
- if `type` is `article` the work has to be peer reviewed

## 3. Tagging works

`openalex/tag_dump.py`

Since even filtered works include irrelevant works each work also gets Science Keywords assigned. Works with a highest keyword score of $\geq 0.375$ will be considered relevant.

## 4. Additional filtering

`openalex/preprocess.py`

In a final step the works will undergo a final round of filtering. A final dataset will be created where each work has to meet the defined threshold of $\geq 0.375$. Additionally language detection is being run. This needs to be done since the OpenAlex detection is not really reliable. Works without a title or a title too short to have a language assigned are also discarded.

All works passing that inspection will be added to a lookup table. This table consisting of `id` and `type` decides if a work will be added or not.
