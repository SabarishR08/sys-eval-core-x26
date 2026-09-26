# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains the end-to-end solution for the **Amazon ML Challenge 2026: Business Entity Resolution**.

## 🚀 Overview
The objective is to resolve entities across three data sources (`Source 1`, `Source 2`, and `Source 3`) where `Source 1` is the deduplicated reference catalog and `Source 2`/`Source 3` contain noisy, partial business records.

### Evaluation Metric
Submissions are evaluated on **Macro $F_{0.5}$** across all `Source 1` entities:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Precision is weighted $2\times$ over recall** (false merges are penalized heavily).
- **Singletons** (entities with no matches) are included: correctly predicting an empty match earns a score of 1.0, while false merges score 0.0.

---

## 📁 Repository Structure
```
amazon-ml-challenge-2026/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py    # Legal suffix & address normalization
│       │   ├── blocking.py         # Sub-word TF-IDF candidate generation
│       │   ├── features.py         # Fuzzy string & PIN/address agreement
│       │   ├── model.py            # LightGBM matcher & F0.5 calibration
│       │   └── pipeline.py         # End-to-end execution runner
│       ├── requirements.txt        # Pinned dependencies
│       └── README.md               # Reproducibility instructions
├── utils/
│   ├── validate_submission.py      # Official submission compliance validator
│   └── metrics.py                  # Macro F0.5 evaluation tool
├── Documentation_template.md       # Detailed technical methodology write-up
├── package_submission.py           # Automated packaging script
└── .gitignore
```

---

## 🛠️ Quickstart

### 1. Install Dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Place Competition Dataset
Place your official competition data into `dataset/`:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Run Pipeline
```bash
python code/business_entity_resolution/src/pipeline.py --train-dir dataset/train --test-dir dataset/test --output-dir output
```

### 4. Validate and Package Submission
```bash
python package_submission.py --team-name <your_team_name>
```
This produces `<your_team_name>_submission.zip` matching all requirements:
1. `output/matching_results.tsv` (Leaderboard upload)
2. `output/candidate_pairs.tsv`
3. `code/business_entity_resolution/`
4. `Documentation_template.md`
