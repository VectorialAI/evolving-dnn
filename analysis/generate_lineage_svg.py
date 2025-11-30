#!/usr/bin/env python3
"""Generate lineage SVG for an existing experiment."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from experiment_recorder import ExperimentRecorder


def main():
    parser = argparse.ArgumentParser(description="Generate lineage SVG for an existing experiment")
    parser.add_argument("--experiment_path", help="Path to the experiment directory containing experiment_summary.json")
    args = parser.parse_args()

    summary_path = os.path.join(args.experiment_path, "experiment_summary.json")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create a minimal recorder instance without triggering __init__'s file creation
    recorder = object.__new__(ExperimentRecorder)
    recorder.experiment_path = args.experiment_path
    recorder.data = data

    recorder._render_lineage_svg()
    print(f"Generated: {os.path.join(args.experiment_path, 'lineage-codex.svg')}")


if __name__ == "__main__":
    main()

