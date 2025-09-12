Pangaea data is crawled using the [OpenSearch fill-in crawl](https://gitlab.dlr.de/sc/ivs-open/opensearch-fill-in-crawls). The Pangaea API limits the crwal to the most recent 10,000 pages, any pages older than that are currently ignored. A future fix probably needs contact with the Pangaea team.

A cron job will be required to regulary update, tag and insert new data into the graph.