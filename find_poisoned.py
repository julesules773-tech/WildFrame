import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/osm_road_cache.json"
d = json.load(open(path))
poisoned = [k for k, v in d.items() if not v.get("segments")]
print(f"{len(poisoned)} poisoned (0-segment) entries out of {len(d)} total:")
for k in poisoned:
    print(" -", k)