"""
ev_models.py
─────────────
Manages loading and access to the trained EV brake telemetry ML models.
Models are saved in models/ev/ by train_ev_models.py.
"""

import os
import pickle
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "ev")


class EVModelManager:
    """Manages loading and access to the EV ML models, encoders, and feature list."""

    def __init__(self):
        self.health_model      = None
        self.failure_model     = None
        self.health_encoder    = None
        self.failure_encoder   = None
        self.feature_columns   = None   # list of column names used at training time
        self.is_loaded         = False

    def load_models(self):
        """Loads EV models from the models/ev/ directory."""
        try:
            with open(os.path.join(MODEL_DIR, "ev_health_model.pkl"),    "rb") as f:
                self.health_model    = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "ev_failure_model.pkl"),   "rb") as f:
                self.failure_model   = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "ev_health_encoder.pkl"),  "rb") as f:
                self.health_encoder  = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "ev_failure_encoder.pkl"), "rb") as f:
                self.failure_encoder = pickle.load(f)

            # Optional: feature column list (safer inference)
            feat_path = os.path.join(MODEL_DIR, "ev_feature_columns.pkl")
            if os.path.exists(feat_path):
                with open(feat_path, "rb") as f:
                    self.feature_columns = pickle.load(f)
                logger.info(f"EV feature columns loaded: {len(self.feature_columns)} features")

            self.is_loaded = True
            logger.info("EV ML models loaded successfully.")
        except FileNotFoundError:
            self.is_loaded = False
            logger.warning(
                "EV model files not found in models/ev/. "
                "EV endpoint will use heuristic + LLM mode."
            )
        except Exception as e:
            self.is_loaded = False
            logger.error(f"Failed to load EV models: {e}")


ev_model_manager = EVModelManager()
