import os
import re
from typing import Dict


def parse_reso_file(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    model_data: Dict[str, object] = {}
    model_match = re.search(r"model\s+(\w+)\s*\{", text)
    if model_match:
        model_data["name"] = model_match.group(1)

    path_match = re.search(r'path\s*:\s*"([^"]+)"', text)
    if path_match:
        model_data["path"] = path_match.group(1)

    j_match = re.search(r"J\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if j_match:
        model_data["J"] = float(j_match.group(1))

    grid_match = re.search(r"grid\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if grid_match:
        rows = int(grid_match.group(1))
        cols = int(grid_match.group(2))
        model_data["size"] = rows * cols
    else:
        size_match = re.search(r"grid\s*\(\s*(\d+)\s*\)", text)
        if size_match:
            model_data["size"] = int(size_match.group(1))

    if "size" not in model_data:
        model_data["size"] = 64

    if "J" not in model_data:
        model_data["J"] = -1.0

    model_data["type"] = "ising"
    return model_data
