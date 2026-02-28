"""
Convert EG entailment dataset to alpaca format for LLaMA Factory SFT.

Input format:
    [{"prem": "...", "hypo": "...", "label": true/false}, ...]

Output format (alpaca):
    [{"instruction": "...", "input": "", "output": "True"/"False"}, ...]
"""

import json
import argparse
from pathlib import Path


INSTRUCTION_TEMPLATE = (
    'If "{prem}", then "{hypo}", is that true or false? '
    'Answer with only "True" or "False".'
)


def convert(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        raw = json.load(f)

    converted = []
    for item in raw:
        converted.append({
            "instruction": INSTRUCTION_TEMPLATE.format(
                prem=item["prem"],
                hypo=item["hypo"],
            ),
            "input": "",
            "output": "True" if item["label"] else "False",
        })

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    true_count = sum(1 for d in raw if d["label"])
    false_count = len(raw) - true_count
    print(f"Converted {len(converted)} examples  (True: {true_count}, False: {false_count})")
    print(f"Saved to: {output_path}")
    print("\nSample (first example):")
    print(json.dumps(converted[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/EGs/EG_train-balanced.json")
    parser.add_argument("--output", default="data/EG_entailment_train.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    convert(args.input, args.output)
