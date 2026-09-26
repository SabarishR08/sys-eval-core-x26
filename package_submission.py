#!/usr/bin/env python3
"""
Automates packaging the final submission zip adhering strictly
to the required submission structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import sys
import argparse
import zipfile
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.validate_submission import validate


def package_submission(team_name: str, test_dir: str = "dataset/test"):
    repo_dir = REPO_ROOT
    output_dir = repo_dir / "output"
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"
    code_dir = repo_dir / "code" / "business_entity_resolution"
    doc_file = repo_dir / "Documentation_template.md"

    print("=" * 60)
    print(f"Creating Submission Package for Team: {team_name}")
    print("=" * 60)

    # 1. Run validation
    print("\n[Step 1/3] Running pre-packaging submission validation...")
    errors, warnings = validate(str(matching_file), str(candidate_file), test_dir)
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    if errors:
        print("\nERROR: Cannot package submission! Validation failed:")
        for i, issue in enumerate(errors, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)
    print("Pre-validation passed successfully!")

    # 2. Check existence of code and documentation
    print("\n[Step 2/3] Checking required deliverables...")
    if not code_dir.exists():
        print(f"ERROR: Missing code directory: {code_dir}")
        sys.exit(1)
    if not doc_file.exists():
        print(f"ERROR: Missing documentation template: {doc_file}")
        sys.exit(1)

    # 3. Create Zip
    zip_name = f"{team_name}_submission.zip"
    zip_path = repo_dir / zip_name
    print(f"\n[Step 3/3] Creating zip archive: {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add output files
        zipf.write(matching_file, arcname="output/matching_results.tsv")
        zipf.write(candidate_file, arcname="output/candidate_pairs.tsv")

        # Add documentation
        zipf.write(doc_file, arcname="Documentation_template.md")

        # Add code directory recursively
        for file in code_dir.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts and not file.name.endswith(".pyc"):
                rel_path = file.relative_to(repo_dir)
                zipf.write(file, arcname=str(rel_path).replace("\\", "/"))

    print("\nSubmission Package Summary:")
    with zipfile.ZipFile(zip_path, "r") as zipf:
        for info in zipf.infolist():
            print(f"  - {info.filename} ({info.file_size:,} bytes)")

    print(f"\nSUCCESS! Package ready: {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge Submission")
    parser.add_argument("--team-name", default="Team_Antigravity", help="Your team name")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to dataset/test directory")
    args = parser.parse_args()

    package_submission(args.team_name, args.test_dir)


if __name__ == "__main__":
    main()
