#!/usr/bin/env python3
# generate_plots.py
# ============================================================
# Generate All Visualizations for README & Report
# ============================================================
"""
Produces every plot needed for documentation:
  1. Target distribution (bar + pie)
  2. Numerical feature distributions by enrollment
  3. Box plots for outlier inspection
  4. Categorical enrollment rates
  5. Correlation heatmap
  6. Model comparison bar chart
  7. ROC curves for all trained models
  8. Confusion matrices
  9. Feature importance (Random Forest)
  10. Classification report heatmap
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import (
    ALL_FEATURES, BEST_MODEL_PATH, CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES, RANDOM_STATE, TARGET_COLUMN,
)
from src.data_processing import build_preprocessor, clean_data, load_raw_data, split_data

# ── Setup ─────────────────────────────────────────────────
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Premium color palette
COLORS = {
    "primary": "#6366f1",      # Indigo
    "secondary": "#ec4899",    # Pink
    "success": "#10b981",      # Emerald
    "warning": "#f59e0b",      # Amber
    "danger": "#ef4444",       # Red
    "info": "#06b6d4",         # Cyan
    "enrolled": "#10b981",
    "not_enrolled": "#ef4444",
    "bg_dark": "#1e1b4b",
    "text": "#e2e8f0",
}
PALETTE = [COLORS["not_enrolled"], COLORS["enrolled"]]

# Apply premium style
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

# ── Load Data ─────────────────────────────────────────────
print("Loading and preparing data...")
df_raw = load_raw_data()
df = clean_data(df_raw)
X_train_df, X_test_df, y_train, y_test = split_data(df)
preprocessor = build_preprocessor()

# ── 1. Target Distribution ───────────────────────────────
print("1/10  Target distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

target_counts = df[TARGET_COLUMN].value_counts().sort_index()
bars = axes[0].bar(
    ["Not Enrolled (0)", "Enrolled (1)"],
    target_counts.values,
    color=PALETTE,
    edgecolor="white",
    linewidth=0.8,
    width=0.5,
    zorder=3,
)
for bar, val in zip(bars, target_counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, val + 80,
        f"{val:,}", ha="center", fontweight="bold", fontsize=13, color="#e2e8f0",
    )
axes[0].set_title("Target Class Distribution")
axes[0].set_ylabel("Count")
axes[0].grid(axis="y", zorder=0)

wedges, texts, autotexts = axes[1].pie(
    target_counts.values,
    labels=["Not Enrolled (0)", "Enrolled (1)"],
    autopct="%1.1f%%",
    colors=PALETTE,
    startangle=90,
    explode=(0.04, 0.04),
    textprops={"color": "#e2e8f0", "fontsize": 12},
    wedgeprops={"edgecolor": "#0f172a", "linewidth": 2},
)
for at in autotexts:
    at.set_fontweight("bold")
axes[1].set_title("Target Class Proportion")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_target_distribution.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 2. Numerical Feature Distributions ───────────────────
print("2/10  Numerical distributions...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(NUMERICAL_FEATURES):
    for enrolled_val, color, label in [(0, COLORS["not_enrolled"], "Not Enrolled"), (1, COLORS["enrolled"], "Enrolled")]:
        subset = df[df[TARGET_COLUMN] == enrolled_val][col]
        axes[i].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor="none")
    axes[i].set_title(f"{col.replace('_', ' ').title()} Distribution")
    axes[i].set_xlabel(col.replace("_", " ").title())
    axes[i].set_ylabel("Frequency")
    axes[i].legend(framealpha=0.3)
    axes[i].grid(axis="y", zorder=0)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_numerical_distributions.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 3. Box Plots ─────────────────────────────────────────
print("3/10  Box plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(NUMERICAL_FEATURES):
    data_0 = df[df[TARGET_COLUMN] == 0][col]
    data_1 = df[df[TARGET_COLUMN] == 1][col]
    bp = axes[i].boxplot(
        [data_0, data_1],
        labels=["Not Enrolled", "Enrolled"],
        patch_artist=True,
        widths=0.4,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "#94a3b8"},
        capprops={"color": "#94a3b8"},
        flierprops={"marker": "o", "markerfacecolor": "#f59e0b", "markersize": 4, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")
    axes[i].set_title(f"{col.replace('_', ' ').title()} by Enrollment")
    axes[i].grid(axis="y", zorder=0)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_numerical_boxplots.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 4. Categorical Feature Analysis ─────────────────────
print("4/10  Categorical analysis...")
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes_flat = axes.flatten()

for i, col in enumerate(CATEGORICAL_FEATURES):
    ct = pd.crosstab(df[col], df[TARGET_COLUMN], normalize="index") * 100
    ct.columns = ["Not Enrolled %", "Enrolled %"]
    ct.plot(
        kind="bar", stacked=True, ax=axes_flat[i],
        color=PALETTE, alpha=0.85, edgecolor="none", width=0.6,
    )
    axes_flat[i].set_title(f"Enrollment Rate by {col.replace('_', ' ').title()}")
    axes_flat[i].set_ylabel("Percentage (%)")
    axes_flat[i].set_xlabel("")
    axes_flat[i].legend(framealpha=0.3, fontsize=9)
    axes_flat[i].tick_params(axis="x", rotation=0)
    axes_flat[i].set_ylim(0, 100)
    axes_flat[i].grid(axis="y", zorder=0)

axes_flat[-1].set_visible(False)  # Hide unused 6th subplot
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_categorical_analysis.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 5. Correlation Heatmap ───────────────────────────────
print("5/10  Correlation heatmap...")
corr_cols = NUMERICAL_FEATURES + [TARGET_COLUMN]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 10, s=80, l=55, as_cmap=True)
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".3f",
    cmap=cmap, center=0, linewidths=2, linecolor="#0f172a",
    square=True, cbar_kws={"shrink": 0.8},
    annot_kws={"fontsize": 12, "fontweight": "bold"},
    ax=ax,
)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_correlation_heatmap.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ══════════════════════════════════════════════════════════
# MODEL EVALUATION PLOTS
# ══════════════════════════════════════════════════════════
print("\nTraining models for evaluation plots...")

# Train both models fresh for plotting
models = {}
for name, clf in [
    ("Logistic Regression", LogisticRegression(C=1, max_iter=1000, random_state=RANDOM_STATE)),
    ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE)),
]:
    pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
    pipe.fit(X_train_df, y_train)
    models[name] = pipe

# ── 6. Model Comparison Bar Chart ────────────────────────
print("6/10  Model comparison...")
metrics_data = {}
for name, pipe in models.items():
    y_pred = pipe.predict(X_test_df)
    y_proba = pipe.predict_proba(X_test_df)[:, 1]
    metrics_data[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

metrics_df = pd.DataFrame(metrics_data).T
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(metrics_df.columns))
width = 0.3
model_colors = [COLORS["primary"], COLORS["success"]]

for i, (model_name, row) in enumerate(metrics_df.iterrows()):
    bars = ax.bar(
        x + i * width, row.values, width,
        label=model_name, color=model_colors[i],
        edgecolor="white", linewidth=0.5, zorder=3,
    )
    for bar, val in zip(bars, row.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold", color="#e2e8f0",
        )

ax.set_xticks(x + width / 2)
ax.set_xticklabels(metrics_df.columns)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend(framealpha=0.3, fontsize=11)
ax.grid(axis="y", zorder=0)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_model_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 7. ROC Curves ────────────────────────────────────────
print("7/10  ROC curves...")
fig, ax = plt.subplots(figsize=(9, 7))

for name, pipe, color in zip(
    models.keys(), models.values(),
    [COLORS["primary"], COLORS["success"]],
):
    y_proba = pipe.predict_proba(X_test_df)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1, label="Random Classifier")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="white")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right", framealpha=0.3, fontsize=11)
ax.grid(alpha=0.2)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_roc_curves.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 8. Confusion Matrices ───────────────────────────────
print("8/10  Confusion matrices...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for i, (name, pipe) in enumerate(models.items()):
    y_pred = pipe.predict(X_test_df)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Enrolled", "Enrolled"],
        yticklabels=["Not Enrolled", "Enrolled"],
        ax=axes[i], linewidths=2, linecolor="#0f172a",
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    axes[i].set_title(f"{name}\nConfusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_confusion_matrices.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 9. Feature Importance (Random Forest) ────────────────
print("9/10  Feature importance...")
rf_pipe = models["Random Forest"]
rf_clf = rf_pipe.named_steps["classifier"]
preprocessor_fitted = rf_pipe.named_steps["preprocessor"]

# Get feature names after transformation
num_names = NUMERICAL_FEATURES
cat_encoder = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = num_names + cat_names

importances = rf_clf.feature_importances_
sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(
    range(len(sorted_idx)), importances[sorted_idx],
    color=COLORS["primary"], edgecolor="white", linewidth=0.3, zorder=3,
)
# Highlight top 5
for bar in bars[-5:]:
    bar.set_color(COLORS["success"])

ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([all_feature_names[i].replace("_", " ").title() for i in sorted_idx], fontsize=10)
ax.set_xlabel("Importance (Gini)")
ax.set_title("Random Forest — Feature Importance")
ax.grid(axis="x", zorder=0)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_feature_importance.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── 10. Cross-Validation Score Distribution ──────────────
print("10/10 Cross-validation scores...")
fig, ax = plt.subplots(figsize=(12, 5))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, pipe_template, color in [
    ("Logistic Regression",
     Pipeline([("preprocessor", build_preprocessor()),
               ("classifier", LogisticRegression(C=1, max_iter=1000, random_state=RANDOM_STATE))]),
     COLORS["primary"]),
    ("Random Forest",
     Pipeline([("preprocessor", build_preprocessor()),
               ("classifier", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE))]),
     COLORS["success"]),
]:
    scores = cross_val_score(pipe_template, X_train_df, y_train, cv=cv, scoring="roc_auc")
    cv_results[name] = scores

positions = [1, 2]
bp_data = [cv_results["Logistic Regression"], cv_results["Random Forest"]]
bp = ax.boxplot(
    bp_data, positions=positions, widths=0.4, patch_artist=True,
    medianprops={"color": "white", "linewidth": 2},
    whiskerprops={"color": "#94a3b8", "linewidth": 1.5},
    capprops={"color": "#94a3b8", "linewidth": 1.5},
)
for patch, color in zip(bp["boxes"], [COLORS["primary"], COLORS["success"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor("white")

# Overlay individual fold scores
for pos, scores, color in zip(positions, bp_data, [COLORS["primary"], COLORS["success"]]):
    ax.scatter([pos] * len(scores), scores, color="white", zorder=5, s=40, edgecolors=color, linewidths=1.5)

ax.set_xticks(positions)
ax.set_xticklabels(["Logistic Regression", "Random Forest"])
ax.set_ylabel("ROC-AUC Score")
ax.set_title("5-Fold Cross-Validation Score Distribution")
ax.grid(axis="y", zorder=0)
ax.set_ylim(0.9, 1.01)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "10_cv_scores.png", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── Summary ──────────────────────────────────────────────
print(f"\n✅ All 10 plots saved to {PLOTS_DIR.resolve()}/")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"   {f.name}")
