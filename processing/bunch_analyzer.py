"""
bunch_analyzer.py
-----------------
Aggregates per-banana ripeness predictions into a bunch-level result.

Two aggregation methods are provided:

  Method A — Majority vote
      The most common ripeness class among all bananas is the final result.
      This is the OFFICIAL bunch classification for version 1.

  Method B — Average ripeness score
      Unripe=0, Ripe=1, Overripe=2.
      The numerical mean is computed and reported for experimentation.
      Thresholds for score→class conversion are configurable constants.

Note on class ordering
----------------------
The score mapping uses canonical class names (lowercase, no space):
  "unripe"   → 0
  "ripe"     → 1
  "overripe" → 2

These are separate from the model's class INDEX (which is {0:OverRipe, 1:Ripe, 2:Unripe}).
The canonical mapping is defined here and is independent of the model index.
"""

from collections import Counter
from typing import List, Dict, Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Configurable constants
# ──────────────────────────────────────────────────────────────────────────────

# Numerical score assigned to each canonical ripeness class
RIPENESS_SCORE: Dict[str, int] = {
    "unripe":   0,
    "ripe":     1,
    "overripe": 2,
}

# Thresholds for score→class mapping (Method B).
# Calibrate these after real-world validation.
# score < SCORE_UNRIPE_MAX  → Unripe
# score > SCORE_OVERRIPE_MIN → Overripe
# otherwise                 → Ripe
SCORE_UNRIPE_MAX: float = 0.40
SCORE_OVERRIPE_MIN: float = 1.60

# Threshold for problematic bunch (percentage of unripe + overripe bananas)
# If problematic_percentage >= PROBLEMATIC_THRESHOLD → HIGH RISK
PROBLEMATIC_THRESHOLD: float = 0.30  # 30%


class BunchAnalyzer:
    """
    Aggregates individual banana ripeness predictions into a bunch-level result.

    Parameters
    ----------
    unripe_threshold : float
        Maximum average score to classify a bunch as Unripe.  Default: 0.40.
    overripe_threshold : float
        Minimum average score to classify a bunch as Overripe.  Default: 1.60.
    """

    def __init__(
        self,
        unripe_threshold: float = SCORE_UNRIPE_MAX,
        overripe_threshold: float = SCORE_OVERRIPE_MIN,
    ):
        self.unripe_threshold = unripe_threshold
        self.overripe_threshold = overripe_threshold

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        banana_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate per-banana predictions into a bunch-level result.

        Parameters
        ----------
        banana_results : list of dict
            Each dict must contain at least:
              {
                "banana_id"       : int,
                "bbox"            : [x1, y1, x2, y2],
                "confidence"      : float,  # detection confidence
                "ripeness_class"  : str,    # e.g. "Ripe"
                "canonical"       : str,    # e.g. "ripe"
                "ripe_confidence" : float,  # classification confidence
              }
            If the classifier was not available, canonical may be None.

        Returns
        -------
        dict
            {
                "total_bananas"     : int,
                "counts"            : {"unripe": int, "ripe": int, "overripe": int},
                "percentages"       : {"unripe": float, "ripe": float, "overripe": float},
                "problematic_count" : int,
                "problematic_percentage": float,
                "bunch_status"      : str,  # "ACCEPTABLE" or "HIGH RISK"
                "dominant_problem"  : str,  # e.g. "Too many Unripe bananas"
                "majority_result"   : str,   # canonical, e.g. "ripe"
                "avg_ripeness_score": float,
                "score_result"      : str,   # canonical, from score thresholds
                "final_result"      : str,   # official result = majority_result
            }
        """
        if not banana_results:
            result = self._empty_result()
            # Add problematic fields to empty result
            result.update({
                "problematic_count": 0,
                "problematic_percentage": 0.0,
                "bunch_status": "ACCEPTABLE",
                "dominant_problem": "None"
            })
            return result

        # Extract canonical classes; skip None (classifier not available)
        canonicals = [
            b.get("canonical") or b.get("ripeness_class", "").lower().replace(" ", "")
            for b in banana_results
        ]
        # Normalise "overripe" / "over ripe" / "over_ripe"
        canonicals = [self._normalise(c) for c in canonicals]

        total = len(canonicals)
        counts = {
            "unripe":   sum(1 for c in canonicals if c == "unripe"),
            "ripe":     sum(1 for c in canonicals if c == "ripe"),
            "overripe": sum(1 for c in canonicals if c == "overripe"),
        }

        percentages = {
            cls: round(cnt / total * 100, 2) if total > 0 else 0.0
            for cls, cnt in counts.items()
        }

        # Calculate problematic bananas (unripe + overripe)
        problematic_count = counts["unripe"] + counts["overripe"]
        problematic_percentage = round(problematic_count / total * 100, 2) if total > 0 else 0.0

        # Determine bunch status based on problematic threshold
        bunch_status = "HIGH RISK" if problematic_percentage >= (PROBLEMATIC_THRESHOLD * 100) else "ACCEPTABLE"

        # Determine dominant problem
        unripe_pct = percentages["unripe"]
        overripe_pct = percentages["overripe"]

        if unripe_pct == 0 and overripe_pct == 0:
            dominant_problem = "None"
        elif unripe_pct > overripe_pct:
            dominant_problem = "Too many Unripe bananas"
        elif overripe_pct > unripe_pct:
            dominant_problem = "Too many Over Ripe bananas"
        else:  # equal and non-zero
            dominant_problem = "Both Unripe and Over Ripe bananas"

        # ── Method A: Majority vote ────────────────────────────────────────
        majority_result = self._majority_vote(canonicals)

        # ── Method B: Average ripeness score ──────────────────────────────
        scores = [RIPENESS_SCORE.get(c, 1) for c in canonicals]   # unknown → 1 (ripe)
        avg_score = round(sum(scores) / total, 4) if total > 0 else 0.0
        score_result = self._score_to_class(avg_score)

        return {
            "total_bananas":      total,
            "counts":             counts,
            "percentages":        percentages,
            "problematic_count":  problematic_count,
            "problematic_percentage": problematic_percentage,
            "bunch_status":       bunch_status,
            "dominant_problem":   dominant_problem,
            "majority_result":    majority_result,
            "avg_ripeness_score": avg_score,
            "score_result":       score_result,
            "final_result":       majority_result,   # v1 official = majority
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(canonical: Optional[str]) -> str:
        """Normalise various "overripe" spellings → canonical form."""
        if not canonical:
            return "ripe"   # default for missing predictions
        c = canonical.lower().strip()
        if "over" in c:
            return "overripe"
        if "unripe" in c or "un ripe" in c:
            return "unripe"
        return "ripe"

    @staticmethod
    def _majority_vote(canonicals: List[str]) -> str:
        """Return the most common class among a list of canonical names."""
        if not canonicals:
            return "ripe"
        counter = Counter(canonicals)
        return counter.most_common(1)[0][0]

    def _score_to_class(self, avg_score: float) -> str:
        """
        Map an average ripeness score → canonical class.

        Uses configurable thresholds:
          score < unripe_threshold   → "unripe"
          score > overripe_threshold → "overripe"
          otherwise                  → "ripe"
        """
        if avg_score < self.unripe_threshold:
            return "unripe"
        if avg_score > self.overripe_threshold:
            return "overripe"
        return "ripe"

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "total_bananas":      0,
            "counts":             {"unripe": 0, "ripe": 0, "overripe": 0},
            "percentages":        {"unripe": 0.0, "ripe": 0.0, "overripe": 0.0},
            "majority_result":    "unknown",
            "avg_ripeness_score": 0.0,
            "score_result":       "unknown",
            "final_result":       "unknown",
        }

    def __repr__(self) -> str:
        return (
            f"BunchAnalyzer("
            f"unripe_threshold={self.unripe_threshold}, "
            f"overripe_threshold={self.overripe_threshold})"
        )
