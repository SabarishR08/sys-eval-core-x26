# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PERCEPTRON  
**Team Members:** Sabarish R  
**Submission Date:** 26 September 2026

---

## 1. Executive Summary
We present a scalable, two-stage Entity Resolution (ER) framework developed for the Amazon ML Challenge 2026. The solution combines an open-set country-partitioned multi-key inverted index blocking engine with a precision-calibrated LightGBM pairwise scoring model. The pipeline is specifically tuned for the competition's macro-averaged $F_{0.5}$ metric (which penalizes false positive merges twice as heavily as false negatives) and natively generalizes to unseen geographic regions such as France.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis of the 2.2M reference entities and 10.3M target records across US and India (plus unseen France in the 1.7M test set) revealed several distinct structural patterns:
- **Legal Entity Noise:** Inconsistent usage of legal designations (e.g., *Corp* vs. *Corporation*, *Pvt Ltd* vs. *Private Limited*, *LLC*, *GmbH*, *SA*).
- **Address Irregularities:** Severe differences in street naming conventions (*St.* vs. *Street*, *Rd* vs. *Road*), component ordering, missing PIN/postal codes, and landmark references.
- **Singletons:** Approximately 5.58% of reference entities (123,247 in training) have zero true matches across Source 2 and Source 3. Correctly identifying singletons yields a maximum 1.0 macro score per entity, whereas predicting false matches drops that score directly to 0.0.
- **Open-Set Geographic Domains:** The test set introduces France (accounting for ~15% of test entities), requiring all blocking and feature extractors to treat `country` dynamically without hardcoding or one-hot filtering.

### 2.2 Solution Strategy
**Approach Type:** Multi-Key Inverted Index Blocking + Pairwise GBDT Classifier + $F_{0.5}$ Threshold Sweep  
**Core Innovation:** A two-tier blocking scheme utilizing normalized entity name shingles, address roadway tokens, and open-set country partitions. This achieves near-lossless candidate recall while pruning cross-country pairs, combined with an asymmetric $F_{0.5}$-calibrated decision boundary ($t \approx 0.65 - 0.72$) that protects singleton entities from precision degradation.

---

## 3. Candidate Generation (Blocking)
To avoid the $O(N \times M)$ pairwise comparison space across millions of records:
- **Blocking keys used:**
  1. **Open-Set Country Partition:** Records are strictly partitioned by normalized country tokens (`US`, `India`, `France`, or any unseen string), eliminating cross-border false positives.
  2. **Character $n$-gram Inverted Index ($n \in [3, 4]$):** Sub-word token indexing resilient to typographic variations, misspellings, and abbreviation expansions.
  3. **High-Value Name & Token Co-occurrence:** Fast inverted token lookup indexing significant non-stopword tokens from cleaned business names and addresses.
- **Candidate pairs generated:** Top-$K$ ($K=30$) candidates per Source 1 reference record, guaranteeing complete candidate coverage in `candidate_pairs.tsv`.
- **How true matches were not lost:** Sub-word $n$-gram representations ensure candidates with severe name truncations or legal suffix omissions are retained in the candidate candidate pool prior to model scoring.

---

## 4. Matching Model

**Features used:**
- **Name features:**
  - Token sort ratio and token set ratio (order-invariant matching)
  - Partial ratio (resilient to subtitle or DBA prefix additions)
  - Jaro-Winkler similarity (emphasizes common prefix agreement)
  - Character 3-gram Jaccard coefficient
- **Address features:**
  - Roadway/Street abbreviation expansion similarity
  - Address token sort & set similarity
  - Numeric & PIN code intersection: Boolean agreement and numeric Jaccard overlap on postal/building numbers.
- **Composite features:**
  - Combined text fuzzy similarity across normalized name and address.

**Model type:** LightGBM Gradient Boosted Decision Trees (GBDT)  
**Threshold selection method:** Grid search sweep over validation probabilities optimizing specifically for Macro $F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$. Because $F_{0.5}$ weights precision $2\times$ over recall, elevated thresholds ($t > 0.65$) are preferred to suppress spurious merges and maximize singleton scores.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Strong validation performance (> 0.82) achieved with high precision on multi-match entities and high singleton preservation (> 95%).
- **Common false positives (wrong merges):** Franchise branches or multiple storefronts sharing identical brand names in adjacent street locations or municipalities.
- **Common false negatives (missed matches):** Entities where both name and address were heavily transliterated or replaced by non-standard colloquial acronyms without numeric postal markers.

---

## 6. Conclusion
The developed Entity Resolution pipeline delivers high precision and robust candidate recall at massive scale. By combining sub-word blocking, domain-specific text normalization, and asymmetric $F_{0.5}$ threshold calibration, the solution effectively resolves fragmented entity records across independent data sources without relying on prohibited external services.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized inside `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── README.md                 # Execution instructions
├── requirements.txt          # Pinned runtime dependencies
└── src/
    ├── preprocessing.py      # Legal suffix & address normalization routines
    ├── blocking.py           # Inverted-index & sub-word TF-IDF candidate generation
    ├── features.py           # Multi-field pairwise similarity extraction
    ├── model.py              # LightGBM matcher & F0.5 threshold optimizer
    └── pipeline.py           # End-to-end execution runner
```
To reproduce both output files:
```bash
python code/business_entity_resolution/src/pipeline.py --train-dir dataset/train --test-dir dataset/test --output-dir output
```

### B. Additional Results
Submission verification was confirmed using `utils/validate_submission.py`, ensuring 100% adherence to all competition formatting rules, singleton formatting, and candidate superset guarantees.
