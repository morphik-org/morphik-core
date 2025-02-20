import os
import json
from collections import Counter

# Get all JSON files in samples/results
results_dir = "samples/results"
json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

# Count number of contacts in each file
contact_counts = []
for json_file in json_files:
    file_path = os.path.join(results_dir, json_file)
    with open(file_path) as f:
        data = json.load(f)
        num_contacts = len(data["contacts"])
        contact_counts.append(num_contacts)

# Calculate statistics
total_files = len(contact_counts)
count_stats = Counter(contact_counts)

# Print results
print(f"\nContact count statistics across {total_files} companies:")
print("-" * 50)

for num_contacts in range(4):  # 0-3 contacts
    count = count_stats[num_contacts]
    percentage = (count / total_files) * 100
    print(f"{num_contacts} contacts: {count} companies ({percentage:.1f}%)")
