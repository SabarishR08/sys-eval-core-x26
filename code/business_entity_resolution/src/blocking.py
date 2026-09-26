"""
Candidate Generation (Blocking) Module.
Maximizes recall ceiling while strictly adhering to open-set country partitions
and inverted-index / TF-IDF candidate generation.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class CandidateGenerator:
    """
    Generates plausible candidate pairs (Source 1 entity -> [Source 2/3 candidates])
    using country partitioning, token indexing, and TF-IDF cosine filtering.
    """

    def __init__(self, top_k: int = 40, min_tfidf_sim: float = 0.12):
        self.top_k = top_k
        self.min_tfidf_sim = min_tfidf_sim

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """
        Generates candidate mapping {s1_entity_id: [candidate_entity_ids]}.
        Both dfs must have columns: ['entity_id', 'business_name', 'business_address', 'country']
        """
        # Ensure country strings are stripped and normalized (open-set: no hardcoding)
        s1_df = s1_df.copy()
        target_df = target_df.copy()

        s1_df["country_norm"] = s1_df["country"].fillna("").astype(str).str.strip().str.lower()
        target_df["country_norm"] = target_df["country"].fillna("").astype(str).str.strip().str.lower()

        # Combine name and address for semantic matching
        s1_df["comb_text"] = s1_df["business_name"].fillna("").astype(str) + " " + s1_df["business_address"].fillna("").astype(str)
        target_df["comb_text"] = target_df["business_name"].fillna("").astype(str) + " " + target_df["business_address"].fillna("").astype(str)

        candidates = {s1_id: [] for s1_id in s1_df["entity_id"]}

        # Group by country to prune cross-country matching while supporting unseen countries
        unique_countries = set(s1_df["country_norm"].unique()) | set(target_df["country_norm"].unique())

        for country in unique_countries:
            s1_sub = s1_df[s1_df["country_norm"] == country]
            target_sub = target_df[target_df["country_norm"] == country]

            if s1_sub.empty or target_sub.empty:
                continue

            s1_texts = s1_sub["comb_text"].tolist()
            s1_ids = s1_sub["entity_id"].tolist()

            target_texts = target_sub["comb_text"].tolist()
            target_ids = target_sub["entity_id"].tolist()

            # Character 3-gram + Word TF-IDF vectorizer for high recall over typos/abbreviations
            tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
            try:
                tfidf.fit(target_texts + s1_texts)
                s1_matrix = tfidf.transform(s1_texts)
                target_matrix = tfidf.transform(target_texts)

                # Compute cosine similarities in chunks if needed
                sim_matrix = cosine_similarity(s1_matrix, target_matrix)

                for i, s1_id in enumerate(s1_ids):
                    sim_scores = sim_matrix[i]
                    top_indices = np.argsort(sim_scores)[::-1][: self.top_k]

                    selected_cands = []
                    for idx in top_indices:
                        if sim_scores[idx] >= self.min_tfidf_sim:
                            selected_cands.append(target_ids[idx])
                        elif len(selected_cands) < 3 and sim_scores[idx] > 0.01:
                            # Keep at least top 3 candidates if any overlap exists
                            selected_cands.append(target_ids[idx])

                    candidates[s1_id].extend(selected_cands)

            except Exception:
                # Fallback: simple token overlap
                target_tokens = [set(t.lower().split()) for t in target_texts]
                for i, s1_id in enumerate(s1_ids):
                    s1_toks = set(s1_texts[i].lower().split())
                    scored = []
                    for idx, t_toks in enumerate(target_tokens):
                        inter = len(s1_toks & t_toks)
                        if inter > 0:
                            scored.append((inter, target_ids[idx]))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    candidates[s1_id].extend([tid for _, tid in scored[: self.top_k]])

        # Deduplicate candidates per S1 while preserving ranking order
        for s1_id in candidates:
            seen = set()
            deduped = []
            for cid in candidates[s1_id]:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)
            candidates[s1_id] = deduped

        return candidates
