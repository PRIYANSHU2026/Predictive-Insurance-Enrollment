#!/usr/bin/env python3
# run_pipeline.py
# ============================================================
# End-to-End ML Pipeline Orchestrator
# ============================================================
"""
Single entry point that runs the full machine learning pipeline:

    1. Load & clean the raw employee data.
    2. Split into train / test sets.
    3. Train multiple models with hyperparameter tuning.
    4. Log all experiments to MLflow.
    5. Persist the champion model to disk.

Usage:
    python run_pipeline.py

After training, inspect results:
    mlflow ui --backend-store-uri mlruns
    # then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import logging
import sys
import time

from src.config import BEST_MODEL_PATH, MLFLOW_TRACKING_URI
from src.data_processing import prepare_data
from src.model_training import train_and_evaluate


def main() -> None:
    """Run the complete training pipeline."""
    # --- Configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  │  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pipeline")

    logger.info("=" * 70)
    logger.info("  Insurance Enrollment Prediction — Training Pipeline")
    logger.info("=" * 70)

    start = time.time()

    # ── Step 1: Data preparation ──────────────────────────
    logger.info("STEP 1/2 — Preparing data …")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    logger.info(
        "  Train samples: %d  |  Test samples: %d", len(X_train), len(X_test)
    )

    # ── Step 2: Model training & evaluation ───────────────
    logger.info("STEP 2/2 — Training & evaluating models …")
    best_pipeline = train_and_evaluate(
        X_train, X_test, y_train, y_test, preprocessor
    )

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("  ✅ Pipeline complete in %.1f seconds.", elapsed)
    logger.info("  Champion model saved to: %s", BEST_MODEL_PATH)
    logger.info("  MLflow experiments at:   %s", MLFLOW_TRACKING_URI)
    logger.info("=" * 70)
    logger.info(
        "\n  Next steps:\n"
        "    1. View experiments:  mlflow ui --backend-store-uri mlruns\n"
        "    2. Serve the API:     uvicorn src.api:app --reload\n"
    )


if __name__ == "__main__":
    main()
