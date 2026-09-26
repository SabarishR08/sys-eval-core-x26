# Amazon ML Challenge 2026: Methodology Document
**Team Name**: Team Antigravity  
**Problem Statement**: Business Entity Resolution Challenge  

---

## 1. Executive Summary
Entity Resolution (ER) in commercial catalogs involves identifying records referring to identical real-world entities across diverse, noisy, and unstructured sources without shared primary keys. In this challenge, deduplicated reference records from **Source 1** must be matched against records in **Source 2** and **Source 3**. 

Our solution implements an end-to-end two-stage architecture:
1. **Recall-Maximizing Blocking Stage**: Partitions the candidate search space using open-set country grouping and sub-word character $n$-gram TF-IDF inverted index to achieve high candidate recall ceiling while pruning infeasible pairs.
2. **Precision-Optimized Matching Stage ($F_{0.5}$)**: Computes fine-grained lexical, phonetic, and address-specific similarity features scored by LightGBM, coupled with threshold calibration specifically tuned for the $F_{0.5}$ metric (which penalizes false positive merges $2\times$ more than false negatives) and singleton accuracy.

---

## 2. Preprocessing & Normalization
Commercial records present multiple noise patterns: legal suffix variations (e.g., *Corp* vs. *Corporation*, *Pvt* vs. *Private*), roadway abbreviations (*Rd* vs. *Road*, *St* vs. *Street*), punctuation, and transliterations.

- **Legal Suffix Canonicalization**: Standardizes company designations (`Corp`, `Ltd`, `Inc`, `Pvt`, `LLC`, `LLP`, `Enterprises`) to standard base forms.
- **Address & Roadway Normalization**: Expands abbreviations (`Rd`, `St`, `Ave`, `Blvd`, `Apt`, `Ste`) and cleans non-alphanumeric noise while retaining numeric tokens (PIN/postal codes and building numbers).
- **Open-Set Country Handling**: The test dataset introduces unseen countries (such as France). The pipeline treats country strictly as an open string attribute without hardcoding or one-hot filtering, ensuring complete coverage.

---

## 3. Candidate Generation (Blocking) Strategy
Evaluating the Cartesian product of Source 1 and target sources ($O(N \times M)$) is computationally prohibitive.
- **Country Partitioning**: Pairs are constrained to matching normalized country tokens, drastically shrinking candidate pairs.
- **Sub-word TF-IDF Vectorization**: Characters $n$-grams ($n \in [3, 4]$) capture typographic errors, phonetic variants, and word order permutations.
- **Top-$K$ Candidate Emission**: For each Source 1 entity, the top-$K$ candidates above an adaptive threshold are captured and emitted directly to `candidate_pairs.tsv`. This satisfies the competition requirement that final matches must be a subset of this file.

---

## 4. Feature Engineering
For each candidate pair $(s_1, t)$, a feature vector is constructed across several orthogonal similarity signals:
1. **Name Matching**:
   - Normalized Levenshtein ratio
   - Partial ratio & token sort ratio (resilient to word-order inversion)
   - Jaro-Winkler similarity (rewards shared prefixes)
   - Token Jaccard overlap
2. **Address & Location Matching**:
   - Address token sort & set ratios
   - Jaro-Winkler address similarity
   - Postal code / PIN overlap: Numeric token intersection between addresses to verify geographic consistency.
3. **Combined Representation**:
   - Composite token similarity across name and address.

---

## 5. Model Architecture & Metric Optimization
- **Gradient Boosted Decision Trees (LightGBM)**: Fast, non-linear classifier resilient to feature scale differences and missing fields.
- **$F_{0.5}$ Aligned Thresholding**: The competition metric is Macro $F_{0.5}$:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Because precision is prioritized over recall by a factor of 2, standard 0.5 classification probability leads to sub-optimal scores due to false merges. We perform a validation sweep to select an elevated decision threshold ($t \approx 0.65 - 0.72$), eliminating dubious merges and maximizing singletons score (worth 1.0 each).

---

## 6. Submission Integrity & Validation
The pipeline strictly conforms to all submission constraints:
- Produces valid tab-separated files: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Guaranteed candidate superset: All predictions in `matching_results.tsv` are verified to be subsets of `candidate_pairs.tsv`.
- Validated via `utils/validate_submission.py` ensuring zero duplicate IDs, complete Source 1 coverage, and exclusion of self-matches.
