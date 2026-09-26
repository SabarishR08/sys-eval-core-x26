#!/usr/bin/env python3
"""
Official submission validator script (standard library only).
Matches the specifications in the Amazon ML Challenge 2026 problem statement.
"""
import sys
import argparse
from pathlib import Path


def parse_tsv_ids(filepath, id_col_name):
    """
    Parses a tab-separated file and returns a mapping:
    source1_entity_id -> list of candidate/matched entity_ids.
    """
    if not filepath.exists():
        return None, f"File does not exist: {filepath}"

    records = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if len(header) != 2:
            return None, f"{filepath.name}: Invalid header columns. Expected exactly 2 tab-separated columns, got {header}"
        if header[0] != "source1_entity_id" or header[1] != id_col_name:
            return None, f"{filepath.name}: Header must be exactly ['source1_entity_id', '{id_col_name}'], got {header}"

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 1:
                # Could be empty second column (singleton)
                s1_id = parts[0].strip()
                ids_str = ""
            elif len(parts) == 2:
                s1_id, ids_str = parts[0].strip(), parts[1].strip()
            else:
                return None, f"{filepath.name} Line {line_num}: Contains {len(parts)} columns (expected 2 tab-separated columns). Check for unescaped tabs."

            if s1_id in records:
                return None, f"{filepath.name} Line {line_num}: Duplicate source1_entity_id '{s1_id}'."

            if ids_str:
                id_list = [i.strip() for i in ids_str.split(",") if i.strip()]
                # Check for duplicates in list
                if len(id_list) != len(set(id_list)):
                    return None, f"{filepath.name} Line {line_num}: Contains duplicate IDs in list: {ids_str}"
                records[s1_id] = id_list
            else:
                records[s1_id] = []

    return records, None


def load_test_source_ids(test_dir):
    """Loads all valid entity IDs from test_source1, test_source2, and test_source3."""
    test_dir = Path(test_dir)
    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"

    for f in [s1_file, s2_file, s3_file]:
        if not f.exists():
            return None, None, f"Required test file not found: {f}"

    def extract_ids(fpath):
        ids = set()
        with open(fpath, "r", encoding="utf-8") as fp:
            _ = fp.readline()
            for line in fp:
                p = line.rstrip("\r\n").split("\t")
                if p and p[0].strip():
                    ids.add(p[0].strip())
        return ids

    s1_ids = extract_ids(s1_file)
    valid_target_ids = extract_ids(s2_file) | extract_ids(s3_file)
    return s1_ids, valid_target_ids, None


def validate_submission(matching_file, candidate_file, test_dir):
    issues = []
    print(f"Validating matching file: {matching_file}")
    print(f"Validating candidate file: {candidate_file}")
    print(f"Against test directory: {test_dir}\n")

    matching_path = Path(matching_file)
    candidate_path = Path(candidate_file)
    test_dir_path = Path(test_dir)

    s1_ids, valid_targets, err = load_test_source_ids(test_dir_path)
    if err:
        return [f"[FATAL] Could not read test files: {err}"]

    matches, err_m = parse_tsv_ids(matching_path, "matched_entity_ids")
    if err_m:
        return [f"[FORMAT ERROR] {err_m}"]

    candidates, err_c = parse_tsv_ids(candidate_path, "candidate_entity_ids")
    if err_c:
        return [f"[FORMAT ERROR] {err_c}"]

    # 1. Check all Source 1 entities are present
    missing_in_matches = s1_ids - set(matches.keys())
    if missing_in_matches:
        issues.append(f"{len(missing_in_matches)} Source 1 test entities missing from {matching_path.name} (e.g. {list(missing_in_matches)[:3]})")

    extra_in_matches = set(matches.keys()) - s1_ids
    if extra_in_matches:
        issues.append(f"{len(extra_in_matches)} unrecognized Source 1 entities in {matching_path.name} (e.g. {list(extra_in_matches)[:3]})")

    missing_in_cands = s1_ids - set(candidates.keys())
    if missing_in_cands:
        issues.append(f"{len(missing_in_cands)} Source 1 test entities missing from {candidate_path.name} (e.g. {list(missing_in_cands)[:3]})")

    # 2. Check candidate superset invariant: every matched entity MUST be in candidate list
    violation_count = 0
    invalid_target_count = 0
    self_match_count = 0

    for s1_id, m_list in matches.items():
        c_set = set(candidates.get(s1_id, []))
        for mid in m_list:
            # Self match check
            if mid.startswith("S1-") or mid == s1_id:
                self_match_count += 1
            # Valid test target check
            if mid not in valid_targets:
                invalid_target_count += 1
            # Candidate superset check
            if mid not in c_set:
                violation_count += 1

    if violation_count > 0:
        issues.append(f"{violation_count} matched IDs were not present in candidate_pairs.tsv! (Final matches must be a subset of candidates)")
    if invalid_target_count > 0:
        issues.append(f"{invalid_target_count} matched IDs do not exist in test Source 2 or Source 3 files!")
    if self_match_count > 0:
        issues.append(f"{self_match_count} matched IDs contain Source 1 self-matches!")

    # Check candidates for valid targets
    invalid_cands = 0
    for s1_id, c_list in candidates.items():
        for cid in c_list:
            if cid not in valid_targets:
                invalid_cands += 1
    if invalid_cands > 0:
        issues.append(f"{invalid_cands} candidate IDs do not exist in test Source 2 or Source 3 files!")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate Amazon ML Challenge 2026 submissions")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to dataset/test directory")

    args = parser.parse_args()

    issues = validate_submission(args.matching, args.candidate, args.test_dir)
    if issues:
        print("VALIDATION FAILED with the following issues:")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        sys.exit(1)
    else:
        print("PASS (exit 0): All submission rules verified successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
