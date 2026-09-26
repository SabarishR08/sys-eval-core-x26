# Business Entity Resolution Pipeline

This package provides a machine learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

## 1. Overview
The solution resolves noisy, fragmented business identity records from three independent data sources (`Source 1`, `Source 2`, and `Source 3`) using a two-stage approach:
1. **Candidate Generation (Blocking)**: Employs character $n$-gram TF-IDF and open-set country partitioning to maximize recall while filtering cross-country entity pairs.
2. **Precision-Heavy Entity Matching ($F_{0.5}$)**: Extracts pairwise lexical, phonetic, and address-specific similarity features scored by LightGBM with threshold calibration tuned for Macro $F_{0.5}$ and singleton accuracy.

## 2. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## 3. How to Run
To run the complete pipeline on your dataset and generate the submission files:
```bash
python src/pipeline.py --train-dir ../../dataset/train --test-dir ../../dataset/test --output-dir ../../output
```

The script will automatically generate:
- `output/matching_results.tsv`: Final resolved entity matches.
- `output/candidate_pairs.tsv`: Complete candidate set considered before final scoring.

## 4. Verification
To validate that both output files strictly adhere to competition guidelines:
```bash
python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```
