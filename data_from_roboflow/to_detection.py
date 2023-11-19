# Convert segmentation to detection labelme
input_folder = "data/data_02"

import os
import json
import numpy as np

label_map = {
    "Knight": "ma",
    "Bishop": "tuong",
    "Chariot": "xe",
    "Cannon": "phao",
    "Guard": "si",
    "King": "vua",
    "Soldier": "tot",
}

label_files = os.listdir(input_folder)
for label_file in label_files:
    if not label_file.endswith(".json"):
        continue

    with open(os.path.join(input_folder, label_file), "r") as f:
        data = json.load(f)

    for shape in data["shapes"]:
        if shape["label"] not in label_map.keys():
            print(f"Unknown label: {shape['label']}")
            continue
        shape["shape_type"] = "rectangle"
        shape["points"] = [
            np.min(shape["points"], axis=0).tolist(),
            np.max(shape["points"], axis=0).tolist(),
        ]
        shape["label"] = label_map[shape["label"]]

    # Remove unknown labels
    data["shapes"] = [shape for shape in data["shapes"] if shape["label"] in label_map.values()]

    with open(os.path.join(input_folder, label_file), "w") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
