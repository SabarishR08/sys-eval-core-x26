"""
Evaluation metric implementation for Macro-averaged F_0.5 score
with proper singleton handling as specified by the Amazon ML Challenge 2026 rubric.
"""
from typing import Dict, List, Set, Union


def compute_entity_f05(pred_ids: Union[List[str], Set[str]], true_ids: Union[List[str], Set[str]]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Singleton handling:
    - If true_ids is empty and pred_ids is empty -> 1.0 (correct singleton)
    - If true_ids is empty and pred_ids is non-empty -> 0.0 (false merge on singleton)
    - If true_ids is non-empty and pred_ids is empty -> 0.0 (missed match)
    """
    pred_set = set(pred_ids)
    true_set = set(true_ids)

    # Singleton case
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    if len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision == 0.0 or recall == 0.0:
        return 0.0

    f_05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f_05


def compute_macro_f05(
    predictions: Dict[str, Union[List[str], Set[str]]],
    ground_truth: Dict[str, Union[List[str], Set[str]]]
) -> Dict[str, float]:
    """
    Computes macro-average F_0.5 across all Source 1 entities in ground_truth.
    Returns:
        {
            "macro_f05": float,
            "singleton_accuracy": float,
            "non_singleton_f05": float,
            "total_entities": int,
            "singleton_count": int
        }
    """
    total = len(ground_truth)
    if total == 0:
        return {"macro_f05": 0.0}

    f05_scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        score = compute_entity_f05(pred_matches, true_matches)
        f05_scores.append(score)

        if len(true_matches) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    macro_f05 = sum(f05_scores) / total
    singleton_acc = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 1.0
    non_singleton_avg = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_avg,
        "total_entities": total,
        "singleton_count": len(singleton_scores),
    }
