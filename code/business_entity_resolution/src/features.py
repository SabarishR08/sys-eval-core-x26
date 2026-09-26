"""
Pairwise Feature Engineering Module for Entity Resolution.
Extracts name similarities, address similarities, token sets, and numerical agreement.
"""
from typing import Dict, List, Any
import numpy as np
import rapidfuzz
from rapidfuzz import fuzz, distance
try:
    from .preprocessing import (
        normalize_business_name,
        normalize_address,
        extract_numbers_and_pins,
    )
except ImportError:
    from preprocessing import (
        normalize_business_name,
        normalize_address,
        extract_numbers_and_pins,
    )


def compute_string_features(name1: str, name2: str, addr1: str, addr2: str) -> Dict[str, float]:
    """
    Computes fine-grained lexical and fuzzy string similarity features
    between a reference record (1) and candidate record (2).
    """
    n1_norm = normalize_business_name(name1)
    n2_norm = normalize_business_name(name2)
    a1_norm = normalize_address(addr1)
    a2_norm = normalize_address(addr2)

    # 1. Name Similarities
    name_ratio = fuzz.ratio(n1_norm, n2_norm) / 100.0
    name_partial_ratio = fuzz.partial_ratio(n1_norm, n2_norm) / 100.0
    name_token_sort_ratio = fuzz.token_sort_ratio(n1_norm, n2_norm) / 100.0
    name_token_set_ratio = fuzz.token_set_ratio(n1_norm, n2_norm) / 100.0
    name_jw = distance.JaroWinkler.similarity(n1_norm, n2_norm)

    # Name token overlap
    n1_toks = set(n1_norm.split())
    n2_toks = set(n2_norm.split())
    name_jaccard = len(n1_toks & n2_toks) / max(1, len(n1_toks | n2_toks))

    # Prefix match
    name_prefix_match = 1.0 if (n1_norm and n2_norm and (n1_norm[:4] == n2_norm[:4])) else 0.0

    # 2. Address Similarities
    addr_ratio = fuzz.ratio(a1_norm, a2_norm) / 100.0
    addr_token_sort_ratio = fuzz.token_sort_ratio(a1_norm, a2_norm) / 100.0
    addr_token_set_ratio = fuzz.token_set_ratio(a1_norm, a2_norm) / 100.0
    addr_jw = distance.JaroWinkler.similarity(a1_norm, a2_norm)

    a1_toks = set(a1_norm.split())
    a2_toks = set(a2_norm.split())
    addr_jaccard = len(a1_toks & a2_toks) / max(1, len(a1_toks | a2_toks))

    # 3. Numeric / PIN code Agreement
    num1 = extract_numbers_and_pins(addr1)
    num2 = extract_numbers_and_pins(addr2)
    if num1 and num2:
        exact_pin_match = 1.0 if (num1 & num2) else 0.0
        pin_jaccard = len(num1 & num2) / len(num1 | num2)
        pin_diff_count = float(abs(len(num1) - len(num2)))
    elif not num1 and not num2:
        exact_pin_match = 0.5  # Neutral
        pin_jaccard = 0.5
        pin_diff_count = 0.0
    else:
        exact_pin_match = 0.0
        pin_jaccard = 0.0
        pin_diff_count = float(abs(len(num1) - len(num2)))

    # First token match (very strong signal in corporate & commercial entity resolution)
    tok1_first = n1_toks.pop() if len(n1_toks) == 1 else (n1_norm.split()[0] if n1_norm.split() else "")
    tok2_first = n2_toks.pop() if len(n2_toks) == 1 else (n2_norm.split()[0] if n2_norm.split() else "")
    first_word_match = 1.0 if (tok1_first and tok2_first and tok1_first == tok2_first) else 0.0

    # Length ratios
    name_len_diff = abs(len(n1_norm) - len(n2_norm)) / max(1, max(len(n1_norm), len(n2_norm)))
    addr_len_diff = abs(len(a1_norm) - len(a2_norm)) / max(1, max(len(a1_norm), len(a2_norm)))

    # 4. Combined Name + Address Text
    c1 = f"{n1_norm} {a1_norm}"
    c2 = f"{n2_norm} {a2_norm}"
    comb_token_sort = fuzz.token_sort_ratio(c1, c2) / 100.0

    return {
        "name_ratio": name_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_jw": name_jw,
        "name_jaccard": name_jaccard,
        "name_prefix_match": name_prefix_match,
        "first_word_match": first_word_match,
        "name_len_diff": name_len_diff,
        "addr_ratio": addr_ratio,
        "addr_token_sort_ratio": addr_token_sort_ratio,
        "addr_token_set_ratio": addr_token_set_ratio,
        "addr_jw": addr_jw,
        "addr_jaccard": addr_jaccard,
        "addr_len_diff": addr_len_diff,
        "exact_pin_match": exact_pin_match,
        "pin_jaccard": pin_jaccard,
        "pin_diff_count": pin_diff_count,
        "comb_token_sort": comb_token_sort,
    }


def build_pair_features(
    s1_row: Dict[str, Any],
    target_row: Dict[str, Any]
) -> List[float]:
    """Generates an ordered vector of numerical features for a pair."""
    feat_dict = compute_string_features(
        str(s1_row.get("business_name", "")),
        str(target_row.get("business_name", "")),
        str(s1_row.get("business_address", "")),
        str(target_row.get("business_address", "")),
    )
    return list(feat_dict.values())
