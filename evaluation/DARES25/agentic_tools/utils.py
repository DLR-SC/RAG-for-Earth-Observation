import json
import time
from contextlib import contextmanager

import subprocess
import ast
import re
from collections import Counter
from typing import Optional, Dict, Any
import copy

from pydantic import BaseModel, Field, create_model


@contextmanager
def timer(label: str, time_in_seconds_arr: list):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start:.4f} seconds")
        time_in_seconds_arr.append(end - start)


def get_st_data(name: str = "validation"):
    dataset = None
    with open(f"data_splits/{name}.json") as f:
        dataset = json.load(f)
    return dataset


def dict_reducer(left: dict | None, right: dict | None) -> dict:
    """
    Merge two dictionaries with list values.

    If a key exists in both dictionaries, append the elements from the right
    dictionary's list to the left dictionary's list. If either input is None,
    it's treated as an empty dictionary.

    Args:
        left (dict or None): The first dictionary. If None, treated as {}.
        right (dict or None): The second dictionary. If None, treated as {}.

    Returns:
        dict: A dictionary containing the merged lists for each key.
              For example, merging {'key1': ['e1', 'e2']} with {'key1': ['e_new']}
              will yield {'key1': ['e1', 'e2', 'e_new']}.
    """
    left = left if left is not None else {}
    right = right if right is not None else {}
    merged = left.copy()  # Start with a copy of left

    for key, right_value in right.items():
        if not isinstance(right_value, list):
            right_value = [right_value]

        if key in merged:
            merged[key] = merged[key] + right_value
        else:
            merged[key] = right_value

    return merged


def dict_of_dict(
    left: Optional[Dict[str, Dict[str, Any]]],
    right: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Merge two dictionaries whose values are dictionaries.

    If a key exists in both dictionaries, merge the inner dictionaries.
    If either input is None, it's treated as an empty dictionary.

    Args:
        left: The first dictionary or None.
        right: The second dictionary or None.

    Returns:
        A dictionary containing the merged key-value pairs.
    """
    left = left or {}
    right = right or {}
    merged = copy.deepcopy(left)  # deep copy to avoid mutating original dicts

    for key, right_value in right.items():
        if not isinstance(right_value, dict):
            raise TypeError(f"Expected dict for value of key '{key}', got {type(right_value)}")

        if key in merged:
            merged[key] = {**merged[key], **right_value}
        else:
            merged[key] = copy.deepcopy(right_value)

    return merged


# EVALUATION Utils
def _extract_eval_from_str(stdout_text: str):
    """
    Turns the eval script str output into a dictionary
    """
    print(stdout_text)
    # --- Extract the labels counter ---
    label_dstr = r"Distribution of the labels:\s*Counter\((\{.*?\})\)"
    punc_dstr = r"Distribution of the intervention punctuation:\s*Counter\((\{.*?\})\)"

    def _match_labels(pattern_labels):
        labels_counter = None
        match_labels = re.search(pattern_labels, stdout_text)
        if match_labels:
            # match_labels.group(1) will be the dictionary string
            labels_dict_str = match_labels.group(1)
            # Safely evaluate the string to a Python dict
            labels_dict = ast.literal_eval(labels_dict_str)
            # Optionally wrap it in a Counter (if you want Counter behavior)
            labels_counter = Counter(labels_dict)
        return labels_counter

    labels_counter = _match_labels(label_dstr)
    punc_counter = _match_labels(punc_dstr)
    # --- Extract the overall punctuation value ---
    # Look for a line starting with "Overall punctuation" followed by a number
    pattern_overall = r"Overall punctuation\s+([0-9\.eE+-]+)"
    match_overall = re.search(pattern_overall, stdout_text)
    if match_overall:
        overall_punctuation = float(match_overall.group(1))
    else:
        overall_punctuation = None

    # Now you have your variables:
    result = {
        "labels_counter": labels_counter,
        "punctuation_counter": punc_counter,
        "overall_punctuation": overall_punctuation,
    }
    return result


def eval_experiment(
    submission_path: str,
    data_split: str = "validation",
    threshold: float = 0.6,
    metric: str = "similarity",
):
    """
    Takes as input the path of the generated answers wit other attributes and returns a dict
    """
    # Build the command as a list of strings
    command = [
        "python",
        "eval_scripts/evaluation.py",
        "--metric",
        metric,
        "--input_path",
        f"data_splits/{data_split}.json",
        "--submission_path",
        submission_path,
        "--threshold",
        str(threshold),
    ]
    print("Running command:", " ".join(command))
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    # Print the standard output and error (if any)
    eval_res = _extract_eval_from_str(result.stdout)

    # generate result
    labels_dstr = dict(eval_res["labels_counter"])
    sum_ = sum(labels_dstr.values())
    labels_dstr_ratio = {
        f"{k}_ratio": round(v / sum_, 2) for k, v in labels_dstr.items()
    }

    punctuation_counter = dict(eval_res["punctuation_counter"])

    rename_mapping = {"1.0": "3/3", "0.6": "2/3", "0.3": "1/3", "0": "0/3"}

    punctuation_counter_proc = {
        rename_mapping[str(k)[: min(3, len(str(k)))]]: v
        for k, v in punctuation_counter.items()
    }

    for _, v in rename_mapping.items():
        if v not in punctuation_counter_proc.keys():
            punctuation_counter_proc[v] = 0.0

    punctuation_counter_ratio = {
        f"{k}_ratio": round(v / sum(punctuation_counter.values()), 2)
        for k, v in punctuation_counter_proc.items()
    }
    merged = (
        labels_dstr
        | labels_dstr_ratio
        | punctuation_counter_proc
        | punctuation_counter_ratio
    )
    merged["overall_punctuation"] = eval_res["overall_punctuation"]
    return merged

def create_dynamic_object_with_schema(className: str, attributes: Dict[str, Dict[str, Any]]):  #, values: Dict[str, Any] = None):
    """
    attributes: dict like:
        {
            'a': {'type': int, 'default': 1, 'description': 'desc a'},
            'b': {'type': float, 'default': 0.5, 'description': 'desc b'}
        }
    values: values to populate the instance
    """
    # Build fields dynamically
    fields = {
        name: (
            spec["type"],
            Field(default=spec.get("default", ...), description=spec.get("description", ""))
        )
        for name, spec in attributes.items()
    }

    # Create model dynamically
    DynamicClass = create_model(className, **fields, __base__=BaseModel)

    # Return instance
    return DynamicClass# DynamicClass(**(values or {}))
