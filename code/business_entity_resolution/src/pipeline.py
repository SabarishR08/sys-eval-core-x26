"""
Ultra-High Performance End-to-End Entity Resolution Execution Pipeline.
Processes in streaming memory-bounded chunks with invariant matching & LightGBM scoring.
"""
import os
import sys
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

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
    top_k_candidates: int = 35,
    decision_threshold: float = 0.70,
    train_sample_size: int = 50000,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Starting Amazon ML Challenge 2026 - PERCEPTRON Entity Resolution")
    print("=" * 60)

    # 1. Load Training Data (Sampled for lightning-fast training with full pattern coverage)
    print("\n[Step 1/5] Loading training samples...")
    train_s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", nrows=train_sample_size)
    s1_ids_sample = set(train_s1["entity_id"])

    train_gt_full = load_ground_truth(train_dir / "train_ground_truth.tsv")
    train_gt = {s1_id: train_gt_full.get(s1_id, []) for s1_id in s1_ids_sample}

    # Extract all target records referenced in ground truth sample
    needed_targets = set()
    for s1_id, m_list in train_gt.items():
        for mid in m_list:
            needed_targets.add(mid)

    print(f"Loaded {len(train_s1):,} S1 training entities ({len(needed_targets):,} target links)")

    # Read S2 and S3 efficiently
    train_s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", nrows=train_sample_size * 2)
    train_s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", nrows=train_sample_size * 2)
    train_target = pd.concat([train_s2, train_s3], ignore_index=True)

    # 2. Candidate Generation on Training
    print("\n[Step 2/5] Inverted Index Candidate Generation on training...")
    blocker = CandidateGenerator(top_k=top_k_candidates)
    train_candidates = blocker.generate_candidates(train_s1, train_target)

    train_s1_dict = train_s1.set_index("entity_id").to_dict(orient="index")
    train_target_dict = train_target.set_index("entity_id").to_dict(orient="index")

    # 3. Train Matcher Model
    print("\n[Step 3/5] Extracting features & training LightGBM Matcher...")
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
    print(f"Built training dataset: {X_train.shape[0]:,} pairs ({np.sum(y_train):,} positive matches)")

    matcher = EntityMatcherModel(threshold=decision_threshold)
    matcher.train(X_train, y_train)

    del train_s1, train_s2, train_s3, train_target, X_train, y_train
    gc.collect()

    # 4. Stream & Process Test Set in Manageable Country Batches
    print("\n[Step 4/5] Processing test dataset...")
    test_s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t")
    test_s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t")
    test_s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t")
    test_target = pd.concat([test_s2, test_s3], ignore_index=True)

    print(f"Loaded Test: Source 1 ({len(test_s1):,}), Target Records ({len(test_target):,})")

    # Block on Test
    print("Generating candidates for test set...")
    test_candidates = blocker.generate_candidates(test_s1, test_target)

    test_s1_dict = test_s1.set_index("entity_id").to_dict(orient="index")
    test_target_dict = test_target.set_index("entity_id").to_dict(orient="index")

    # Batch Predict Matches
    print("Scoring candidate pairs with precision thresholding...")
    final_matches = {s1_id: [] for s1_id in test_s1["entity_id"]}

    batch_s1 = []
    batch_cid = []
    batch_feats = []

    for s1_id in test_s1["entity_id"]:
        s1_row = test_s1_dict.get(s1_id, {})
        cands = test_candidates.get(s1_id, [])
        for cid in cands:
            c_row = test_target_dict.get(cid, {})
            feats = build_pair_features(s1_row, c_row)
            batch_s1.append(s1_id)
            batch_cid.append(cid)
            batch_feats.append(feats)

            if len(batch_feats) >= 100000:
                X_batch = np.array(batch_feats, dtype=np.float32)
                probs = matcher.predict_proba(X_batch)
                for sid, cid_val, prob in zip(batch_s1, batch_cid, probs):
                    if prob >= matcher.threshold:
                        final_matches[sid].append(cid_val)
                batch_s1, batch_cid, batch_feats = [], [], []

    if batch_feats:
        X_batch = np.array(batch_feats, dtype=np.float32)
        probs = matcher.predict_proba(X_batch)
        for sid, cid_val, prob in zip(batch_s1, batch_cid, probs):
            if prob >= matcher.threshold:
                final_matches[sid].append(cid_val)

    # Invariant guarantee: matches must be subset of candidates
    for s1_id in test_candidates:
        c_set = set(test_candidates[s1_id])
        final_matches[s1_id] = [mid for mid in final_matches[s1_id] if mid in c_set]

    # 5. Write Output Files
    print("\n[Step 5/5] Writing final submission files...")
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    with open(candidate_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in test_s1["entity_id"]:
            cands = test_candidates.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    with open(matching_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1["entity_id"]:
            matches = final_matches.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(matches)}\n")

    print(f"Generated: {matching_file}")
    print(f"Generated: {candidate_file}")
    print("\nPipeline execution complete! Ready for packaging.")


def main():
    parser = argparse.ArgumentParser(description="PERCEPTRON ER Pipeline")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train dir")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dir")
    parser.add_argument("--output-dir", default="output", help="Path to output dir")
    parser.add_argument("--threshold", type=float, default=0.70, help="F0.5 precision threshold")
    parser.add_argument("--sample-size", type=int, default=50000, help="Training sample size")

    args = parser.parse_args()

    run_pipeline(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        output_dir=Path(args.output_dir),
        decision_threshold=args.threshold,
        train_sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
