# Statistics and Notes

## EOC Geoservice

Data age: `2025-06-23`

- 65 datasets
- Downloaded via public STAC API `https://geoservice.dlr.de/eoc/ogc/stac/v1/`
- 13 datasets got no ScienceKeywords assigned, this happens when the abstract is very short
- In graph only connected to ScienceKeywords (13 without no connections), no authors

## PANGAEA

Data age: `2025-04-07`

- Crawled using [OpenSearch fill-in crawl](https://gitlab.dlr.de/sc/ivs-open/opensearch-fill-in-crawls)
- API restrictions allowed only crawling of most recent `10,000` pages
- Tagged and removed all datasets without ScienceKeywords, left with `885`

## OpenAlex

Data age: `2025-02-26`

- Database aviable as snapshot in aws s3 bucket [donwload instructions](https://docs.openalex.org/download-all-data/download-to-your-machine)
- Currently `269,044,575` works in database (as of `2025-07-14 16:54 CET`)

### Filtering

#### 1. Topic Filtering

- Underlying taxonomy (Domains -> Fields -> Subfields -> Topics) [source](https://docs.google.com/document/d/1bDopkhuGieQ4F8gGNj7sEc8WSE8mvLZS/)
- Using topics as first filter layer. Tag all topics and decide on minimum threshold
- Threshold being the score of the highest scoring ScienceKeyword
- `4,516` topics in total
- Decided on threshold of `0.4` using manual eval (gut feeling more or less)

| Threshold | Topics | Remaining Works | Works loss (%) |
| --------- | ------ | --------------- | -------------- |
|     0.000 |  4,516 |      28,398,134 |          0.00% |
|     0.400 |    563 |       4,236,715 |         85.08% |

As you probably noticed are 28 million not quite the 269 million mentioned previously. This roots back to the requirements each work has to fullfill to be considered:

- `type`: work has to be either a `dataset` or `article`
- `primary_topic` has to match the set of relevant topics
- work must have an abstract
- `language` must be `en` (english)
- `publication_date`: must be published after `1984-04-07` (date of first STAC item)
- if `type` is `article` the work has to be peer reviewed

**NOTE**: Unsure about the oldest STAC item. No catalog including every collection and item ever recoreded. Decided for oldest STAC item of [Copernicus STAC catalog](https://catalogue.dataspace.copernicus.eu/stac/), but there is no gurantee this is really the oldest.

#### 2. Work Filtering

- Topic filtering removed already the majority of the works but is not granular enough
- Tagged every remaining work and decided on theshold of `0.375`, did sampling and verified again manually

| Threshold | Works     | Works loss (%) |
| --------- | --------- | -------------- |
|     0.000 | 4,236,715 |          0.00% |
|     0.375 | 2,105,943 |         49.43% |

#### 3. Additional Loss during Processing

- OpenAlex language detection fails from time to time, additional check using [lingua-py](https://github.com/pemistahl/lingua-py)
  - `696` title not english
  - `26,783` abstract not english
- `5,195` lost because title was `null` (nullable titles were not forseen and thus have to be processed this late)
- Left with `2,073,269` works

**NOTE**: There are some deviations between these calculations and the actual results. Unable to trace back, margin however is approx. below `1,000` works.

## ArangoDB

Some more loss due to unique contstrains. Some datasets are present with mutliple revisions and only differ in e.g. `doi`. Unknown how many exactly but the number cann be approximated when looking at the composition of the database:

| Type        | Number of Items | Source(s)                         |
| ----------- | --------------- | --------------------------------- |
| Author      |           2,850 | PANGAEA, OpenAlex                 |
| Dataset     |          47,883 | EOC Geoservice, PANGAEA, OpenAlex |
| Publication |       2,021,267 | OpenAlex                          |
