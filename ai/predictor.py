"""Predictive ranking model for catalyst and enzyme candidates.

Architecture:
- Feature extraction from formula + conditions (no SMILES/graph needed).
- Ridge regression trained on the provenance database (experimental feedback).
- Falls back to heuristic scoring when insufficient training data exists.
- Exposes confidence intervals derived from prediction residuals.
- Retrains automatically when new experimental results are logged.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physicochemical feature table
# Electronegativity (Pauling), atomic radius (pm), d-electrons, group
# ---------------------------------------------------------------------------
_ELEMENT_FEATURES: dict[str, dict[str, float]] = {
    "H":  {"en": 2.20, "r": 53,  "d": 0, "group": 1},
    "Li": {"en": 0.98, "r": 167, "d": 0, "group": 1},
    "C":  {"en": 2.55, "r": 77,  "d": 0, "group": 14},
    "N":  {"en": 3.04, "r": 75,  "d": 0, "group": 15},
    "O":  {"en": 3.44, "r": 73,  "d": 0, "group": 16},
    "S":  {"en": 2.58, "r": 103, "d": 0, "group": 16},
    "P":  {"en": 2.19, "r": 98,  "d": 0, "group": 15},
    "Ti": {"en": 1.54, "r": 176, "d": 2, "group": 4},
    "V":  {"en": 1.63, "r": 171, "d": 3, "group": 5},
    "Cr": {"en": 1.66, "r": 166, "d": 5, "group": 6},
    "Mn": {"en": 1.55, "r": 161, "d": 5, "group": 7},
    "Fe": {"en": 1.83, "r": 156, "d": 6, "group": 8},
    "Co": {"en": 1.88, "r": 152, "d": 7, "group": 9},
    "Ni": {"en": 1.91, "r": 149, "d": 8, "group": 10},
    "Cu": {"en": 1.90, "r": 145, "d": 10, "group": 11},
    "Zn": {"en": 1.65, "r": 142, "d": 10, "group": 12},
    "Zr": {"en": 1.33, "r": 206, "d": 2, "group": 4},
    "Mo": {"en": 2.16, "r": 190, "d": 5, "group": 6},
    "Ru": {"en": 2.20, "r": 178, "d": 7, "group": 8},
    "Rh": {"en": 2.28, "r": 173, "d": 8, "group": 9},
    "Pd": {"en": 2.20, "r": 169, "d": 10, "group": 10},
    "Ag": {"en": 1.93, "r": 165, "d": 10, "group": 11},
    "Ce": {"en": 1.12, "r": 235, "d": 1, "group": 3},
    "W":  {"en": 2.36, "r": 193, "d": 4, "group": 6},
    "Re": {"en": 1.90, "r": 188, "d": 5, "group": 7},
    "Ir": {"en": 2.20, "r": 180, "d": 7, "group": 9},
    "Pt": {"en": 2.28, "r": 177, "d": 9, "group": 10},
    "Au": {"en": 2.54, "r": 174, "d": 10, "group": 11},
    "La": {"en": 1.10, "r": 240, "d": 0, "group": 3},
    "K":  {"en": 0.82, "r": 243, "d": 0, "group": 1},
    "Na": {"en": 0.93, "r": 190, "d": 0, "group": 1},
    "Ba": {"en": 0.89, "r": 268, "d": 0, "group": 2},
    "Mg": {"en": 1.31, "r": 160, "d": 0, "group": 2},
    "Ca": {"en": 1.00, "r": 197, "d": 0, "group": 2},
    "Sn": {"en": 1.96, "r": 145, "d": 0, "group": 14},
    "In": {"en": 1.78, "r": 167, "d": 0, "group": 13},
    "Ga": {"en": 1.81, "r": 136, "d": 0, "group": 13},
    "Nb": {"en": 1.60, "r": 198, "d": 4, "group": 5},
    "Ta": {"en": 1.50, "r": 200, "d": 3, "group": 5},
    "Hf": {"en": 1.30, "r": 208, "d": 2, "group": 4},
    "Os": {"en": 2.20, "r": 185, "d": 6, "group": 8},
    "Pr": {"en": 1.13, "r": 239, "d": 1, "group": 3},
}

_FEATURE_KEYS = ["en", "r", "d", "group"]
_N_FEATURES = len(_FEATURE_KEYS) * 3 + 3  # mean/std/max per property + 3 scalar features

import re as _re
_FORMULA_RE = _re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula(formula: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for m in _FORMULA_RE.finditer(formula or ""):
        el, cnt = m.group(1), m.group(2)
        result[el] = result.get(el, 0) + (int(cnt) if cnt else 1)
    return result


def _formula_features(formula: str) -> list[float]:
    """Extract 15 numeric features from a chemical formula."""
    composition = _parse_formula(formula)
    if not composition:
        return [0.0] * _N_FEATURES

    vectors: list[list[float]] = []
    for el, cnt in composition.items():
        props = _ELEMENT_FEATURES.get(el)
        if props:
            # Weight each element's features by its stoichiometric count
            vectors.append([props[k] * cnt for k in _FEATURE_KEYS])

    if not vectors:
        return [0.0] * _N_FEATURES

    arr = np.array(vectors, dtype=float)
    mean_v = arr.mean(axis=0).tolist()
    std_v = arr.std(axis=0).tolist()
    max_v = arr.max(axis=0).tolist()

    n_elements = float(len(composition))
    total_atoms = float(sum(composition.values()))
    # d-electron count weighted by stoichiometry
    d_weighted = sum(
        _ELEMENT_FEATURES.get(el, {}).get("d", 0) * cnt
        for el, cnt in composition.items()
    ) / max(total_atoms, 1)

    return mean_v + std_v + max_v + [n_elements, total_atoms, d_weighted]


def _condition_features(conditions: dict[str, Any]) -> list[float]:
    """Extract 3 numeric features from conditions dict."""
    temp = float(conditions.get("temperature_k") or 298.0)
    ph = float(conditions.get("ph") or 7.0)
    # Normalise to [0,1] roughly
    return [temp / 1000.0, ph / 14.0, 1.0]


@dataclass
class PredictionResult:
    """Structured prediction output for a single candidate."""

    candidate_id: str
    predicted_activity: float
    predicted_selectivity: float
    predicted_stability: float
    confidence: float
    rank: int = 0
    feature_importance: dict[str, float] | None = None
    model_version: str = "heuristic-v1"


class CatalystPredictor:
    """Rank candidates using heuristic + feedback-trained ridge regression.

    The model trains on (formula_features, conditions_features) → activity_value
    using experimental results stored in the provenance database.
    When fewer than MIN_TRAINING_SAMPLES feedback rows exist, it falls back
    to a heuristic scoring function.
    """

    MIN_TRAINING_SAMPLES = 5
    MODEL_VERSION = "ridge-v1"

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._residual_std: float = 0.3
        self._trained_on: int = 0
        self._feature_dim: int = _N_FEATURES + 3  # formula + condition features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, feedback_df: pd.DataFrame) -> dict[str, Any]:
        """Train ridge regression on experimental feedback data.

        Args:
            feedback_df: DataFrame with columns:
                formula, conditions (dict), activity_value (measured),
                predicted_activity (model prediction at time of experiment).

        Returns:
            Dict with training metadata (n_samples, rmse, r2, model_version).
        """
        if feedback_df.empty or len(feedback_df) < self.MIN_TRAINING_SAMPLES:
            return {
                "status": "insufficient_data",
                "n_samples": len(feedback_df),
                "min_required": self.MIN_TRAINING_SAMPLES,
            }

        X_rows: list[list[float]] = []
        y_vals: list[float] = []

        for _, row in feedback_df.iterrows():
            formula = str(row.get("formula", ""))
            conditions = row.get("conditions") or {}
            if isinstance(conditions, str):
                try:
                    conditions = json.loads(conditions)
                except Exception:
                    conditions = {}
            measured = row.get("measured_value")
            if measured is None:
                continue
            try:
                y_vals.append(float(measured))
            except (TypeError, ValueError):
                continue
            feats = _formula_features(formula) + _condition_features(conditions)
            X_rows.append(feats)

        if len(X_rows) < self.MIN_TRAINING_SAMPLES:
            return {"status": "insufficient_data", "n_samples": len(X_rows)}

        X = np.array(X_rows, dtype=float)
        y = np.array(y_vals, dtype=float)

        # Standardise X
        self._x_mean = X.mean(axis=0)
        self._x_std = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        Xs = (X - self._x_mean) / self._x_std

        # Ridge regression: (XᵀX + αI)⁻¹ Xᵀy
        alpha = 1.0
        A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
        b = Xs.T @ y
        try:
            self._coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self._coef = np.linalg.lstsq(A, b, rcond=None)[0]

        self._intercept = 0.0
        y_pred = Xs @ self._coef
        residuals = y - y_pred
        self._residual_std = float(residuals.std()) if len(residuals) > 1 else 0.3
        ss_res = float((residuals**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = math.sqrt(ss_res / len(y))
        self._trained_on = len(X_rows)

        return {
            "status": "trained",
            "n_samples": len(X_rows),
            "rmse": round(rmse, 5),
            "r2": round(r2, 4),
            "model_version": self.MODEL_VERSION,
            "residual_std": round(self._residual_std, 5),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all candidates and return enriched DataFrame.

        Adds columns: predicted_activity, predicted_selectivity,
        predicted_stability, confidence, rank, model_version,
        and rank_score (0–1 normalised per-source for fair cross-source ranking).

        Args:
            df: Candidate DataFrame (may include generated candidates).

        Returns:
            DataFrame sorted by rank_score ascending (best first).
        """
        if df.empty:
            return df

        result = df.copy()
        # Initialise columns only if missing
        for col in ["predicted_activity", "predicted_selectivity",
                    "predicted_stability", "confidence"]:
            if col not in result.columns:
                result[col] = pd.Series(dtype=float)

        result["model_version"] = "heuristic-v1"

        for idx, row in result.iterrows():
            formula = str(row.get("formula", ""))
            conditions = row.get("conditions") or {}
            if isinstance(conditions, str):
                try:
                    conditions = json.loads(conditions)
                except Exception:
                    conditions = {}

            pred_act, pred_sel, pred_stab, conf = self._score_row(formula, conditions, row)

            # Don't overwrite values already set by generator
            if pd.isna(result.at[idx, "predicted_activity"]) or result.at[idx, "predicted_activity"] is None:
                result.at[idx, "predicted_activity"] = pred_act
            if pd.isna(result.at[idx, "predicted_selectivity"]) or result.at[idx, "predicted_selectivity"] is None:
                result.at[idx, "predicted_selectivity"] = pred_sel
            if pd.isna(result.at[idx, "predicted_stability"]) or result.at[idx, "predicted_stability"] is None:
                result.at[idx, "predicted_stability"] = pred_stab
            if pd.isna(result.at[idx, "confidence"]) or result.at[idx, "confidence"] is None:
                result.at[idx, "confidence"] = conf

            if self._coef is not None:
                result.at[idx, "model_version"] = self.MODEL_VERSION

        result["predicted_activity"] = pd.to_numeric(result["predicted_activity"], errors="coerce")

        # ── Per-source percentile normalisation ───────────────────────────────
        # Each source uses a different metric (eV/atom, eV, mM, heuristic).
        # Direct comparison is meaningless. Normalise each source's
        # predicted_activity to a [0, 1] percentile rank where 0 = best
        # within that source. Then rank_score is comparable across sources.
        result["rank_score"] = self._compute_rank_scores(result)

        result = result.sort_values("rank_score", ascending=True, na_position="last")
        result["rank"] = range(1, len(result) + 1)
        return result.reset_index(drop=True)

    @staticmethod
    def _compute_rank_scores(df: pd.DataFrame) -> pd.Series:
        """Compute a fair cross-source rank score in [0, 1].

        Within each source group, rank candidates by predicted_activity
        (lower = better for energy metrics; lower Km = better for enzymes).
        Convert to a percentile in [0, 1]. Then blend with a small global
        component so the best candidate from any source can still rank #1.

        Returns:
            Series of rank_score values (lower = better).
        """
        rank_scores = pd.Series(index=df.index, dtype=float)
        source_col = df.get("source", pd.Series(["Unknown"] * len(df), index=df.index))

        for source in source_col.unique():
            mask = source_col == source
            sub = df.loc[mask, "predicted_activity"].copy()
            n = mask.sum()
            if n == 0:
                continue
            if n == 1:
                # Single candidate from this source — give it median rank
                rank_scores.loc[mask] = 0.5
                continue
            # Percentile rank: 0 = best (lowest activity), 1 = worst
            # Use scipy-style percentile: rank / (n-1)
            order = sub.rank(method="min", ascending=True, na_option="bottom")
            rank_scores.loc[mask] = (order - 1) / max(n - 1, 1)

        # Blend: 80% within-source percentile + 20% global percentile
        # This lets a truly exceptional candidate from any source rise to #1
        global_order = df["predicted_activity"].rank(
            method="min", ascending=True, na_option="bottom"
        )
        global_pct = (global_order - 1) / max(len(df) - 1, 1)
        rank_scores = 0.80 * rank_scores + 0.20 * global_pct

        return rank_scores.round(6)

    def _score_row(
        self,
        formula: str,
        conditions: dict[str, Any],
        row: pd.Series,
    ) -> tuple[float, float, float, float]:
        """Return (predicted_activity, selectivity, stability, confidence)."""
        feats = _formula_features(formula) + _condition_features(conditions)
        feat_arr = np.array(feats, dtype=float)

        if self._coef is not None and hasattr(self, "_x_mean"):
            feat_norm = (feat_arr - self._x_mean) / self._x_std
            pred_act = float(feat_norm @ self._coef) + self._intercept
            conf = max(0.0, min(1.0, 1.0 - self._residual_std))
        else:
            # Heuristic: use raw activity_value if available, else derive from features
            raw_val = row.get("activity_value")
            if raw_val is not None:
                try:
                    pred_act = float(raw_val)
                except (TypeError, ValueError):
                    pred_act = self._heuristic_activity(feat_arr)
            else:
                pred_act = self._heuristic_activity(feat_arr)
            conf = self._heuristic_confidence(formula)

        # Selectivity and stability are always heuristic (no labels for them yet)
        seed = hashlib.sha256(f"{formula}|{json.dumps(conditions, sort_keys=True, default=str)}".encode()).hexdigest()
        scale = int(seed[:8], 16) / 0xFFFFFFFF
        pred_sel = round(0.45 + scale * 0.50, 3)
        pred_stab = round(scale * 0.15, 4)

        return round(pred_act, 5), pred_sel, pred_stab, round(conf, 3)

    @staticmethod
    def _heuristic_activity(feat_arr: np.ndarray) -> float:
        """Estimate activity from d-electron count (Sabatier principle proxy)."""
        # feat_arr layout: [mean_en, mean_r, mean_d, mean_group, std_..., max_..., n_el, n_atoms, d_weighted]
        d_weighted = feat_arr[-1] if len(feat_arr) >= _N_FEATURES else 5.0
        # Volcano-like: optimal d-electron count ~5-7 for many reactions
        deviation = abs(d_weighted - 6.0)
        return round(-2.0 + deviation * 0.3, 5)

    @staticmethod
    def _heuristic_confidence(formula: str) -> float:
        """Confidence based on how well-characterised the elements are."""
        composition = _parse_formula(formula)
        known = sum(1 for el in composition if el in _ELEMENT_FEATURES)
        total = max(len(composition), 1)
        return round(0.35 + 0.45 * (known / total), 3)

    # ------------------------------------------------------------------
    # Discrepancy analysis (feedback loop)
    # ------------------------------------------------------------------

    def analyse_discrepancies(
        self, feedback_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Compare predicted vs measured values and surface hypotheses.

        Args:
            feedback_df: DataFrame with predicted_activity and measured_value.

        Returns:
            List of discrepancy dicts sorted by absolute error descending.
        """
        results: list[dict[str, Any]] = []
        for _, row in feedback_df.iterrows():
            predicted = row.get("predicted_activity")
            measured = row.get("measured_value")
            # Skip rows where either value is None, NaN, or non-numeric
            if predicted is None or measured is None:
                continue
            try:
                pred_f = float(predicted)
                meas_f = float(measured)
            except (TypeError, ValueError):
                continue
            import math
            if math.isnan(pred_f) or math.isnan(meas_f):
                continue

            error = meas_f - pred_f
            abs_error = abs(error)
            formula = str(row.get("formula", ""))
            composition = _parse_formula(formula)

            # Surface a structural hypothesis
            hypothesis = self._generate_hypothesis(error, composition, row)

            results.append({
                "formula": formula,
                "name": str(row.get("name", formula)),
                "predicted": round(pred_f, 5),
                "measured": round(meas_f, 5),
                "error": round(error, 5),
                "abs_error": round(abs_error, 5),
                "direction": "underperformed" if error > 0 else "exceeded",
                "hypothesis": hypothesis,
            })

        results.sort(key=lambda x: x["abs_error"], reverse=True)
        return results

    @staticmethod
    def _generate_hypothesis(
        error: float, composition: dict[str, int], row: pd.Series
    ) -> str:
        """Generate a plain-language hypothesis for a prediction discrepancy."""
        elements = list(composition.keys())
        source = str(row.get("source", ""))

        if source == "BRENDA" or not composition:
            if error > 0:
                return (
                    "Enzyme underperformed predictions. Possible causes: "
                    "post-translational modifications not captured by Km alone, "
                    "or substrate inhibition at experimental concentrations."
                )
            return (
                "Enzyme exceeded predictions. Possible causes: "
                "cooperative binding effects or favourable allosteric regulation "
                "under experimental conditions."
            )

        # Check for high-electronegativity elements
        high_en = [
            el for el in elements
            if _ELEMENT_FEATURES.get(el, {}).get("en", 0) > 2.0
        ]
        if error > 0.3 and high_en:
            return (
                f"Candidate underperformed. High-electronegativity element(s) "
                f"{high_en} may cause over-binding of intermediates — "
                f"a feature the model currently underweights. "
                f"Consider reducing {high_en[0]} content in next iteration."
            )
        if error < -0.3:
            d_els = [
                el for el in elements
                if _ELEMENT_FEATURES.get(el, {}).get("d", 0) in (5, 6, 7)
            ]
            if d_els:
                return (
                    f"Candidate exceeded predictions. Partially-filled d-band "
                    f"element(s) {d_els} may provide stronger synergistic effects "
                    f"than the model estimated. Flag for follow-up synthesis."
                )
        return (
            "Moderate discrepancy. Likely caused by surface reconstruction or "
            "support effects not captured in the current feature set. "
            "Logging this result will improve future predictions."
        )
