"""Convert a concept tree JSON file into a YAML file containing only hierarchical names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import yaml


def build_structure(node: dict[str, Any]) -> Tuple[str, Any]:
    """Return the node name and its hierarchical payload (list or dict)."""

    name = node.get("name", "")
    children = node.get("sub_concepts") or []

    if not children:
        return name, None

    processed: list[Tuple[str, Any]] = [build_structure(child) for child in children]

    if all(payload is None for _, payload in processed):
        return name, [child_name for child_name, _ in processed]

    value: dict[str, Any] = {}
    for child_name, payload in processed:
        value[child_name] = [] if payload is None else payload

    return name, value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a concept tree JSON file into a YAML hierarchy with names only."
    )
    parser.add_argument("input_json", type=Path, help="Path to the JSON file containing the concept tree.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output path for the YAML file. Defaults to the input path with a .yaml extension.",
    )

    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    root_name, payload = build_structure(data)
    structure: dict[str, Any]
    if payload is None:
        structure = {root_name: []}
    else:
        structure = {root_name: payload}

    output_path = args.output or args.input_json.with_suffix(".yaml")

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(structure, handle, sort_keys=False, allow_unicode=True)

    print(f"YAML hierarchy written to {output_path}")


if __name__ == "__main__":
    main()
