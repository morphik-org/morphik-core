import json

entity_count = 0
relation_count = 0
document_count = 0

with open("test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        document_count += 1
        data = json.loads(line)
        if "ner" in data:
            entity_count += len(data["ner"])
        if "rel" in data:
            relation_count += len(data["rel"])

print(f"Total documents: {document_count}")
print(f"Total entities: {entity_count}")
print(f"Total relationships: {relation_count}")
