import json
import os
from data_structures import Dataset


def load_jsonl(jsonl_path: str) -> list:
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ========= usage example for load_jsonl function with LLM input =======
# dataset = load_jsonl('./SciER/LLM/test.jsonl')
# sent = dataset[0]
# print(sent.keys())
# print(sent['sentence'])
# print('----------------')
# print(sent['ner'])
# print('----------------')
# print(sent['rel'])
# print('----------------')
# print(sent['rel_plus'])
