import json

with open("new.json") as f:
    data = json.load(f)
new_data = {}
for i in data:
    try:
        new_data[i["name"]] = i
        i["color"] = [int(i) for i in i["color"]]
        if "active" in i:
            i["active"] = [int(i) for i in i["active"]]
        i["tiling"] = int(i["tiling"])
    except KeyError as e:
        e.add_note(f"name is {i["name"]}")
        raise
with open("new.json", "w") as f:
    json.dump(new_data, f)