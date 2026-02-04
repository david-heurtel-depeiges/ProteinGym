#!/usr/bin/env python3
"""
Generate a config.json file for AMPLIFY models based on score files.

This script scans your score folders and creates the necessary config.json
entries to run performance aggregation on your AMPLIFY results.
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path


def detect_score_column(csv_file):
    """Detect the score column name from a CSV file."""
    try:
        df = pd.read_csv(csv_file, nrows=5)

        # Common score column names in AMPLIFY scoring scripts
        possible_columns = ["avg_score", "log_likelihood", "score", "masked_marginals"]
        for col in possible_columns:
            if col in df.columns:
                return col

        # If none found, look for columns that aren't standard DMS columns
        standard_cols = {
            "mutant",
            "mutated_sequence",
            "DMS_score",
            "DMS_score_bin",
            "mutation_depth",
            "mutation_depth_grouped",
        }

        # Filter out obvious non-score columns
        def is_valid_score_column(col_name):
            col_lower = str(col_name).lower()
            # Exclude common metadata columns
            if col_lower in ["index", "id", "name", "sequence", "position"]:
                return False
            # Exclude columns that look like directory paths (not filenames)
            if "/" in col_lower or "\\" in col_lower:
                return False
            # Note: We DON'T exclude file extensions like .pt, .pth because
            # some models actually use these as column names (e.g., mp_rank_00_model_states.pt)
            return True

        other_cols = [c for c in df.columns if c not in standard_cols and is_valid_score_column(c)]

        if other_cols:
            # Prefer columns with "score", "likelihood", model names, or model-related keywords
            priority_keywords = [
                "score",
                "likelihood",
                "pred",
                "esm",
                "amplify",
                "progen",
                "carp",
                "model",
                "states",
                "mp_rank",
            ]
            priority_cols = [c for c in other_cols if any(keyword in str(c).lower() for keyword in priority_keywords)]
            if priority_cols:
                return priority_cols[0]
            return other_cols[0]

    except Exception as e:
        print(f"Warning: Could not read {csv_file}: {e}")
    return None


def scan_score_folders(base_folder, model_type="Single sequence", base_for_location=None):
    """Scan score folders recursively and generate config entries.

    Args:
        base_folder: Root folder to scan for score files
        model_type: Model type classification
        base_for_location: Base folder for computing relative paths in config
    """
    base_path = Path(base_folder)

    if not base_path.exists():
        print(f"Error: Base folder {base_folder} does not exist")
        return {}

    # Use the base folder for computing relative paths if not specified
    if base_for_location is None:
        base_for_location = base_path

    models = {}

    # Recursively walk through all directories
    for dirpath, dirnames, filenames in os.walk(base_path):
        # Filter out directories to ignore
        dirnames[:] = [d for d in dirnames if not d.startswith("_analysis_cache")]

        current_path = Path(dirpath)

        # Check if current directory contains CSV files
        csv_files = [f for f in filenames if f.endswith(".csv")]

        if csv_files:
            # Found a directory with CSV files
            csv_path = current_path / csv_files[0]
            score_col = detect_score_column(csv_path)

            if score_col:
                # Generate a model name from the relative path
                rel_path = current_path.relative_to(base_path)

                # If we're at the root level, use the folder name
                if str(rel_path) == ".":
                    model_name = base_path.name
                    location = str(base_path.relative_to(base_for_location))
                else:
                    # Use the relative path as model name, replacing / with _
                    model_name = str(rel_path).replace("/", "_").replace(os.sep, "_")
                    location = str(current_path.relative_to(base_for_location))

                models[model_name] = {
                    "input_score_name": score_col,
                    "location": location,
                    "directionality": 1,
                    "key": "mutant",
                    "model_type": model_type,
                }
                print(f"✓ Found model: {model_name}")
                print(f"    Score column: '{score_col}'")
                print(f"    Location: {location}")
            else:
                print(f"⚠ Skipping {current_path.relative_to(base_path)}: Could not detect score column")
                print(f"    Checked file: {csv_files[0]}")

    return models


def main():
    parser = argparse.ArgumentParser(description="Generate config.json for AMPLIFY models")
    parser.add_argument("--score-folder", type=str, required=True, help="Base folder containing model score files")
    parser.add_argument("--output", type=str, default="config_amplify.json", help="Output config file path")
    parser.add_argument(
        "--model-type",
        type=str,
        default="Single sequence",
        choices=[
            "Single sequence",
            "MSA",
            "Structure",
            "Single sequence & Structure",
            "Structure & MSA",
            "Single sequence, Structure & Function annotations",
        ],
        help="Model type classification",
    )
    parser.add_argument("--merge-with", type=str, default=None, help="Merge with an existing config.json file")
    parser.add_argument(
        "--base-for-location",
        type=str,
        default=None,
        help="Base path for computing relative 'location' in config (default: parent of score-folder)",
    )

    args = parser.parse_args()

    # Determine base for location paths
    if args.base_for_location:
        base_for_location = Path(args.base_for_location)
    else:
        # Default: use parent directory of score folder (to match ProteinGym structure)
        base_for_location = Path(args.score_folder).parent

    # Scan for models
    print("=" * 80)
    print(f"Scanning {args.score_folder} recursively for models...")
    print(f"Ignoring directories starting with '_analysis_cache'")
    print(f"Computing relative paths from: {base_for_location}")
    print("=" * 80)
    print()
    models = scan_score_folders(args.score_folder, args.model_type, base_for_location)

    if not models:
        print("\n❌ No models found! Make sure the folder contains CSV files with scores.")
        return

    # Analyze score columns
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    score_col_groups = {}
    for model_name, model_config in models.items():
        score_col = model_config["input_score_name"]
        if score_col not in score_col_groups:
            score_col_groups[score_col] = []
        score_col_groups[score_col].append(model_name)

    print(f"\nFound {len(models)} model(s) with {len(score_col_groups)} unique score column(s):")
    for score_col, model_list in sorted(score_col_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n  Score column: '{score_col}' ({len(model_list)} models)")
        if len(model_list) <= 5:
            for m in model_list:
                print(f"    - {m}")
        else:
            for m in model_list[:3]:
                print(f"    - {m}")
            print(f"    ... and {len(model_list) - 3} more")

    if any(len(models) > 1 for models in score_col_groups.values()):
        print("\n💡 NOTE: Multiple models sharing the same score column is NORMAL.")
        print("   This happens when different model versions output the same column name.")
        print("   The 'location' field distinguishes them.")

    # Create config structure
    config = {"model_list_zero_shot_substitutions_DMS": models}

    # Merge with existing config if requested
    if args.merge_with and os.path.exists(args.merge_with):
        print(f"\nMerging with existing config: {args.merge_with}")
        with open(args.merge_with, "r") as f:
            existing_config = json.load(f)
        if "model_list_zero_shot_substitutions_DMS" in existing_config:
            existing_config["model_list_zero_shot_substitutions_DMS"].update(models)
            config = existing_config
        else:
            config.update(existing_config)

    # Save config
    with open(args.output, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n{'=' * 80}")
    print(f"✓ Config file saved to: {args.output}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
