"""Standalone 4-state regime HMM inference.

Loads the trained artifacts (GaussianHMM + StandardScaler + PCA whitener) and
scores a feature matrix into per-bar regime probabilities.

Inference math (ported verbatim from the original service): for a single
observation we use equal priors over states and let the Gaussian emission
likelihoods be the only evidence —

    P(state_i | x) ∝ N(x; μ_i, Σ_i)

computed in the whitened PCA space. We deliberately do NOT call the HMM's own
``predict_proba`` because that multiplies by ``startprob_`` (the distribution of
the first training observation), which would bias every single-step "what
regime now?" call toward whatever regime the training window opened in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_TR_ELEVATED_CONF = 0.70  # top posterior below this -> "Elevated" transition risk
_TR_MODERATE_CONF = 0.90  # below this (but above Elevated) -> "Moderate"

_BUCKETS = ["risk_on_pct", "risk_on_retreat_pct", "risk_off_pct", "risk_off_stable_pct"]
_LABEL_TO_BUCKET = {
    "Risk-On": "risk_on_pct",
    "Risk-On-Retreat": "risk_on_retreat_pct",
    "Risk-Off": "risk_off_pct",
    "Risk-Off-Stable": "risk_off_stable_pct",
}
_DOMINANT_LABEL = {
    "risk_on_pct": "Risk-On",
    "risk_on_retreat_pct": "Risk-On-Retreat",
    "risk_off_pct": "Risk-Off",
    "risk_off_stable_pct": "Risk-Off-Stable",
}


class RegimeHMM:
    """Loads the trained HMM artifacts and scores feature matrices."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        import joblib

        self.model = joblib.load(models_dir / "hmm_model.pkl")
        self.scaler = joblib.load(models_dir / "feature_scaler.pkl")
        pca_path = models_dir / "feature_pca_whitener.pkl"
        self.pca = joblib.load(pca_path) if pca_path.exists() else None
        with open(models_dir / "training_meta.json") as f:
            self.meta = json.load(f)
        self.state_labels: Dict[int, str] = {
            int(k): v for k, v in self.meta.get("state_labels", {}).items()
        }
        self.n_states = int(self.meta.get("n_regimes", 4))

    def _posteriors(self, feat: pd.DataFrame) -> np.ndarray:
        """Return (n_bars, n_states) Bayesian posteriors for a feature frame."""
        from scipy.stats import multivariate_normal as MVN

        x = self.scaler.transform(feat[FEATURE_NAMES].to_numpy(dtype=np.float64))
        if self.pca is not None:
            x = self.pca.transform(x)  # (n, n_pca_components)

        # Per-state Gaussian log-likelihood, vectorised across all bars.
        log_liks = np.column_stack(
            [
                MVN(mean=self.model.means_[i], cov=self.model.covars_[i]).logpdf(x)
                for i in range(self.n_states)
            ]
        )  # (n_bars, n_states)

        log_liks -= log_liks.max(axis=1, keepdims=True)
        post = np.exp(log_liks)
        post /= post.sum(axis=1, keepdims=True)
        return post

    def predict_history(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Score every bar in ``feat`` into a tidy regime-probability frame.

        Columns: date, regime, {four *_pct}, neutral_pct, transition_risk,
        confidence. Probabilities are percentages summing to 100.
        """
        post = self._posteriors(feat)  # (n, n_states)

        # Aggregate state-index posteriors into the four named buckets.
        buckets = {b: np.zeros(len(feat)) for b in _BUCKETS}
        for idx, label in self.state_labels.items():
            bucket = _LABEL_TO_BUCKET.get(label)
            if bucket is None:
                # Unrecognised label — fall back by substring, else split.
                if "Risk-On" in label:
                    bucket = "risk_on_pct"
                elif "Risk-Off" in label:
                    bucket = "risk_off_pct"
                else:
                    buckets["risk_on_retreat_pct"] += post[:, idx] * 50.0
                    buckets["risk_off_stable_pct"] += post[:, idx] * 50.0
                    continue
            buckets[bucket] += post[:, idx] * 100.0

        top = post.max(axis=1)
        tr = np.where(
            top < _TR_ELEVATED_CONF, "Elevated",
            np.where(top < _TR_MODERATE_CONF, "Moderate", "Low"),
        )

        out = pd.DataFrame({b: np.round(buckets[b], 1) for b in _BUCKETS})
        out.insert(0, "date", pd.to_datetime(feat.index).strftime("%Y-%m-%d"))
        # Dominant bucket -> regime label.
        dom = out[_BUCKETS].to_numpy().argmax(axis=1)
        out.insert(1, "regime", [_DOMINANT_LABEL[_BUCKETS[i]] for i in dom])
        out["neutral_pct"] = np.round(
            out["risk_on_retreat_pct"] + out["risk_off_stable_pct"], 1
        )
        out["transition_risk"] = tr
        out["confidence"] = np.round(top, 3)
        cols = ["date", "regime"] + _BUCKETS + ["neutral_pct", "transition_risk", "confidence"]
        return out[cols]
