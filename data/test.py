import json

with open('data_full.json', 'r') as f:
    data = json.load(f)

print(data)
