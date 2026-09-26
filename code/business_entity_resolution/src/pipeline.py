"""
End-to-End Entity Resolution Execution Pipeline.
Processes data from TSV files -> Blocking -> Pairwise Feature Extraction -> ML Scoring -> Submission Output.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add repo root to sys.path so modules can be imported directly
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blocking import CandidateGenerator
from features import build_pair_features
from model import EntityMatcherModel
from utils.metrics import compute_macro_f05


def load_ground_truth(gt_path: Path) -> Dict[str, List[str]]:
    """Loads ground truth file mapping source1_entity_id -> list of matched_entity_ids."""
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                gt[s1_id] = [i.strip() for i in parts[1].split(",") if i.strip()]
            else:
                gt[s1_id] = []
    return gt


def run_pipeline(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    top_k_candidates: int = 40,
    decision_threshold: float = 0.65,
):
    """Executes the complete entity resolution pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Starting Amazon ML Challenge 2026 - Business Entity Resolution")
    print("=" * 60)

    # 1. Load Training Data
    print("\n[Step 1/5] Loading training data...")
    train_s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t")
    train_s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t")
    train_s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t")
    train_gt = load_ground_truth(train_dir / "train_ground_truth.tsv")

    train_target = pd.concat([train_s2, train_s3], ignore_index=True)
    print(f"Loaded Train: Source 1 ({len(train_s1)}), Source 2 ({len(train_s2)}), Source 3 ({len(train_s3)})")
    print(f"Total Reference Records: {len(train_gt)}")

    # 2. Candidate Generation on Training Set
    print("\n[Step 2/5] Candidate generation (Blocking) on train set...")
    blocker = CandidateGenerator(top_k=top_k_candidates)
    train_candidates = blocker.generate_candidates(train_s1, train_target)

    # Convert target df to lookup dictionary for fast feature generation
    train_s1_dict = train_s1.set_index("entity_id").to_dict(orient="index")
    train_target_dict = train_target.set_index("entity_id").to_dict(orient="index")

    # Evaluate blocking recall ceiling
    total_true_links = sum(len(matches) for matches in train_gt.values())
    recalled_links = 0
    for s1_id, true_matches in train_gt.items():
        cand_set = set(train_candidates.get(s1_id, []))
        recalled_links += len(set(true_matches) & cand_set)

    recall_ceiling = (recalled_links / total_true_links) if total_true_links > 0 else 1.0
    print(f"Blocking Recall Ceiling on Train: {recall_ceiling * 100:.2f}% ({recalled_links}/{total_true_links} links)")

    # 3. Feature Extraction & Model Training
    print("\n[Step 3/5] Extracting pairwise features & training classifier...")
    X_train = []
    y_train = []

    for s1_id, cands in train_candidates.items():
        s1_row = train_s1_dict.get(s1_id, {})
        true_set = set(train_gt.get(s1_id, []))

        for cid in cands:
            c_row = train_target_dict.get(cid, {})
            feats = build_pair_features(s1_row, c_row)
            label = 1 if cid in true_set else 0

            X_train.append(feats)
            y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Built training dataset: {X_train.shape[0]} pairs ({np.sum(y_train)} positive matches)")

    matcher = EntityMatcherModel(threshold=decision_threshold)
    matcher.train(X_train, y_train)

    # 4. Process Test Data
    print("\n[Step 4/5] Processing test dataset...")
    test_s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t")
    test_s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t")
    test_s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t")
    test_target = pd.concat([test_s2, test_s3], ignore_index=True)

    print(f"Loaded Test: Source 1 ({len(test_s1)}), Source 2 ({len(test_s2)}), Source 3 ({len(test_s3)})")

    # Generate Test Candidates
    print("Generating candidates for test set...")
    test_candidates = blocker.generate_candidates(test_s1, test_target)

    test_s1_dict = test_s1.set_index("entity_id").to_dict(orient="index")
    test_target_dict = test_target.set_index("entity_id").to_dict(orient="index")

    # Predict Matches on Test Candidates
    final_matches = {s1_id: [] for s1_id in test_s1["entity_id"]}

    # Collect pairs to score in batch
    pair_s1 = []
    pair_cid = []
    pair_feats = []

    for s1_id in test_s1["entity_id"]:
        s1_row = test_s1_dict.get(s1_id, {})
        cands = test_candidates.get(s1_id, [])
        for cid in cands:
            c_row = test_target_dict.get(cid, {})
            feats = build_pair_features(s1_row, c_row)
            pair_s1.append(s1_id)
            pair_cid.append(cid)
            pair_feats.append(feats)

    if pair_feats:
        X_test = np.array(pair_feats, dtype=np.float32)
        test_probs = matcher.predict_proba(X_test)

        for s1_id, cid, prob in zip(pair_s1, pair_cid, test_probs):
            if prob >= matcher.threshold:
                final_matches[s1_id].append(cid)

    # Guarantee: Final matches are strictly a subset of candidates
    for s1_id in test_candidates:
        c_set = set(test_candidates[s1_id])
        final_matches[s1_id] = [mid for mid in final_matches[s1_id] if mid in c_set]

    # 5. Write Submission Files
    print("\n[Step 5/5] Writing final submission files...")
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # Write candidate_pairs.tsv
    with open(candidate_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in test_s1["entity_id"]:
            cands = test_candidates.get(s1_id, [])
            cands_str = ",".join(cands)
            f.write(f"{s1_id}\t{cands_str}\n")

    # Write matching_results.tsv
    with open(matching_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1["entity_id"]:
            matches = final_matches.get(s1_id, [])
            matches_str = ",".join(matches)
            f.write(f"{s1_id}\t{matches_str}\n")

    print(f"Generated {matching_file}")
    print(f"Generated {candidate_file}")
    print("\nPipeline execution complete!")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 ER Pipeline")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory containing training files")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory containing test files")
    parser.add_argument("--output-dir", default="output", help="Directory to save output TSVs")
    parser.add_argument("--threshold", type=float, default=0.65, help="F0.5 precision threshold")

    args = parser.parse_args()

    run_pipeline(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        output_dir=Path(args.output_dir),
        decision_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
