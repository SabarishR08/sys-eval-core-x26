"""
Matching Model and Decision Threshold Optimizer.
Trains a LightGBM/Random Forest classifier on candidate pair features
and optimizes decision threshold specifically for Macro F0.5.
"""
from typing import Dict, List, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier


class EntityMatcherModel:
    """
    Supervised pairwise matching model with F_0.5 threshold calibration.
    """

    def __init__(self, threshold: float = 0.68):
        # Default high threshold to enforce high precision (F0.5 penalizes false merges 2x)
        self.threshold = threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=7,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """Fits the GBDT classifier on pairwise features."""
        if len(y) == 0:
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns match probabilities for candidate pairs."""
        if not self.is_fitted or len(X) == 0:
            return np.zeros(len(X))
        return self.model.predict_proba(X)[:, 1]

    def calibrate_threshold_f05(
        self,
        val_s1_ids: List[str],
        val_cand_ids: List[str],
        val_X: np.ndarray,
        ground_truth: Dict[str, List[str]],
        search_range: Tuple[float, float, int] = (0.40, 0.90, 51)
    ) -> float:
        """
        Grid searches for the optimal decision threshold that maximizes Macro F0.5
        on the validation set.
        """
        try:
            from utils.metrics import compute_macro_f05
        except ImportError:
            from ...utils.metrics import compute_macro_f05

        probs = self.predict_proba(val_X)

        # Pre-group by S1 ID
        cand_map = {}
        for s1_id, cid, prob in zip(val_s1_ids, val_cand_ids, probs):
            if s1_id not in cand_map:
                cand_map[s1_id] = []
            cand_map[s1_id].append((cid, prob))

        best_f05 = -1.0
        best_thresh = self.threshold

        thresholds = np.linspace(search_range[0], search_range[1], search_range[2])
        for t in thresholds:
            preds = {}
            for s1_id in ground_truth:
                cands = cand_map.get(s1_id, [])
                matched = [cid for cid, p in cands if p >= t]
                preds[s1_id] = matched

            scores = compute_macro_f05(preds, ground_truth)
            f05 = scores["macro_f05"]
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = t

        print(f"Optimal F_0.5 Threshold: {best_thresh:.4f} (Validation Macro F0.5: {best_f05:.4f})")
        self.threshold = best_thresh
        return best_thresh
