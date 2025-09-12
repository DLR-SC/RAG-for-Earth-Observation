import json  # noqa: D100
import sys
from argparse import ArgumentParser, FileType, Namespace

import requests


def main(args: Namespace) -> None:
    """Retrieve Topics from OpenAlex and store inside .json."""
    url_template = "https://api.openalex.org/topics?cursor={}&per-page=200"
    cursor = "*"

    data: list = []

    while cursor:

        url = url_template.format(cursor)
        page_with_results = requests.get(url=url, timeout=10).json()
        results = page_with_results["results"]

        data = data + results

        cursor = page_with_results["meta"]["next_cursor"]

    topics = {"topics": data}

    if args.outfile is sys.stdout:
        print(topics)
    else:
        # TODO: Ensure .json and ask to overwrite if exists
        json.dump(obj=topics, fp=args.outfile, indent=4)
        args.outfile.close()


def parse_args() -> Namespace:
    """Parse and define output argument."""
    parser = ArgumentParser()
    parser.add_argument(
        "outfile", nargs="?", type=FileType("w"), default=sys.stdout
    )
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
