"""
============================================================
  Supplier Scoring + Part Failure Prediction Pipeline
  Dataset : BIW_BPillar_Supplier_Dataset.csv
============================================================
Steps
  1.  Load & clean data
  2.  Encode target  (REJECTED → "fail")
  3.  Build Supplier Score  (composite quality index per supplier)
  4.  Train failure-prediction model  (uses supplier score + part features)
  5.  Evaluate, save model & score table
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline          import Pipeline
from sklearn.impute            import SimpleImputer
import joblib

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "BIW_BPillar_Supplier_Dataset.csv")
OUT_DIR    = os.path.join(BASE_DIR, "model_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 – Loading dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
print(f"  Disposition counts:\n{df['Overall_Disposition'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENCODE TARGET
#     REJECTED          → fail  = 1
#     APPROVED/CONDITIONAL → fail = 0
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 2 – Encoding target label")
print("=" * 60)

df["fail"] = (df["Overall_Disposition"] == "REJECTED").astype(int)
print(f"  Label distribution:\n{df['fail'].value_counts().rename({0:'pass (0)', 1:'fail (1)'}).to_string()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BUILD SUPPLIER SCORE
#     Score = weighted average of per-supplier pass rates on every
#             quality/compliance column.  Higher score = better supplier.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 3 – Computing Supplier Score")
print("=" * 60)

# --- 3a. Binary compliance columns (OK / NOT OK) ---
ok_cols = [c for c in df.columns if df[c].dtype == object
           and set(df[c].dropna().unique()).issubset({"OK", "NOT OK"})]

for c in ok_cols:
    df[c + "_bin"] = (df[c] == "OK").astype(float)

ok_bin_cols = [c + "_bin" for c in ok_cols]

# --- 3b. Numeric quality indicators ----------------------------------------
#  Higher is BETTER  : UTS_MPa, YS_MPa, Elongation_%, n_value, r_value,
#                       Salt_Spray_hrs, Fatigue_Cycles, Charpy_Impact_J
#  Lower is BETTER   : Hardness deviation (keep as-is for now),
#                       Surface_Roughness_Ra_um, P_wt%, S_wt%

numeric_quality_cols = [
    "UTS_MPa", "YS_MPa", "Elongation_%",
    "n_value", "r_value",
    "Salt_Spray_hrs", "Fatigue_Cycles", "Charpy_Impact_J",
]
penalty_cols = ["Surface_Roughness_Ra_um", "P_wt%", "S_wt%"]

# Min-max normalise each quality metric → [0, 1]
df_norm = df[numeric_quality_cols].copy()
for c in numeric_quality_cols:
    mn, mx = df[c].min(), df[c].max()
    df_norm[c] = (df[c] - mn) / (mx - mn + 1e-9)

# Penalty cols: invert so lower raw = higher score
df_pen = df[penalty_cols].copy()
for c in penalty_cols:
    mn, mx = df_pen[c].min(), df_pen[c].max()
    df_pen[c] = 1 - (df_pen[c] - mn) / (mx - mn + 1e-9)   # inverted

df_quality = pd.concat([df_norm, df_pen], axis=1)

# --- 3c. Aggregate to supplier level ---------------------------------------
supplier_agg = pd.DataFrame()

# Pass-rates on OK/NOT-OK columns
for c in ok_bin_cols:
    supplier_agg[c] = df.groupby("Supplier_ID")[c].mean()

# Mean normalised quality metrics
for c in df_quality.columns:
    supplier_agg[c] = df.groupby("Supplier_ID")[df_quality.columns].mean()[c]

# Rejection rate (lower is better → invert)
rej_rate = df.groupby("Supplier_ID")["fail"].mean()
supplier_agg["pass_rate"] = 1 - rej_rate

# --- 3d. Composite score  (equal weight, scale 0-100) ----------------------
supplier_agg["Supplier_Score"] = supplier_agg.mean(axis=1) * 100
# Clip to 0-100 in case of floating point drift
supplier_agg["Supplier_Score"] = supplier_agg["Supplier_Score"].clip(0, 100)
supplier_agg = supplier_agg[["Supplier_Score"]].reset_index()

# Name lookup
name_map = df.groupby("Supplier_ID")["Supplier_Name"].first()
supplier_agg["Supplier_Name"] = supplier_agg["Supplier_ID"].map(name_map)
supplier_agg = supplier_agg.sort_values("Supplier_Score", ascending=False).reset_index(drop=True)
supplier_agg["Rank"] = supplier_agg.index + 1

print(supplier_agg[["Rank", "Supplier_ID", "Supplier_Name", "Supplier_Score"]]
      .to_string(index=False))
print()

# Save supplier scores
score_path = os.path.join(OUT_DIR, "supplier_scores.csv")
supplier_agg.to_csv(score_path, index=False)
print(f"  Supplier scores saved -> {score_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MERGE SCORE BACK & BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 4 – Building feature matrix")
print("=" * 60)

df = df.merge(supplier_agg[["Supplier_ID", "Supplier_Score"]],
              on="Supplier_ID", how="left")

# Part-level numeric features + supplier score
feature_cols = [
    "Supplier_Score",          # ← the key engineered feature
    "UTS_MPa", "YS_MPa", "Elongation_%",
    "n_value", "r_value",
    "Hardness_HV", "Hardness_HRB",
    "Thickness_mm",
    "Surface_Roughness_Ra_um",
    "Zinc_Coating_gsm",
    "Salt_Spray_hrs",
    "C_wt%", "Mn_wt%", "Si_wt%", "P_wt%", "S_wt%",
    "Cr_wt%", "Nb_wt%",
    "Bend_Ratio_d/t",
    "Weld_Nugget_Dia_mm",
    "Fatigue_Cycles",
    "Charpy_Impact_J",
] + ok_bin_cols

X = df[feature_cols].copy()
y = df["fail"].values

print(f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
print(f"  Class balance – pass: {(y==0).sum()}  fail: {(y==1).sum()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN  (3 models, pick best by ROC-AUC)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 5 – Training & evaluation (5-fold Stratified CV)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "Random Forest": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced", random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42)),
    ]),
    "Logistic Regression": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42)),
    ]),
}

results = {}
for name, pipe in candidates.items():
    aucs = cross_val_score(pipe, X, y, cv=cv,
                           scoring="roc_auc", n_jobs=-1)
    f1s  = cross_val_score(pipe, X, y, cv=cv,
                           scoring="f1",      n_jobs=-1)
    results[name] = {"roc_auc": aucs, "f1": f1s}
    print(f"  {name:25s}  ROC-AUC = {aucs.mean():.4f} ± {aucs.std():.4f}"
          f"  |  F1 = {f1s.mean():.4f} ± {f1s.std():.4f}")

# ─── pick winner ───
best_name = max(results, key=lambda n: results[n]["roc_auc"].mean())
print(f"\n  [OK] Best model: {best_name}\n")

# ─── train final model on full dataset ───
best_pipe = candidates[best_name]
best_pipe.fit(X, y)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DETAILED EVALUATION ON FULL DATA
#     (train-set report – for a real project use a hold-out test set)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 6 – Full-dataset report")
print("=" * 60)

y_pred  = best_pipe.predict(X)
y_proba = best_pipe.predict_proba(X)[:, 1]

print(classification_report(y, y_pred, target_names=["pass", "fail"]))
print(f"  ROC-AUC (full data) : {roc_auc_score(y, y_proba):.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE MODEL & OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
model_path = os.path.join(OUT_DIR, "failure_model.joblib")
joblib.dump(best_pipe, model_path)
print(f"  Model saved -> {model_path}")

meta = {
    "best_model"      : best_name,
    "feature_cols"    : feature_cols,
    "cv_roc_auc_mean" : float(results[best_name]["roc_auc"].mean()),
    "cv_f1_mean"      : float(results[best_name]["f1"].mean()),
}
meta_path = os.path.join(OUT_DIR, "model_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata saved -> {meta_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 7 – Generating charts")
print("=" * 60)

plt.style.use("dark_background")
ACCENT = "#00E5FF"
GREEN  = "#69FF47"
RED    = "#FF4B6E"
GOLD   = "#FFD700"

# ── 8a. Supplier Score Bar Chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

colors = [GREEN if s >= 70 else (GOLD if s >= 55 else RED)
          for s in supplier_agg["Supplier_Score"]]
bars = ax.barh(supplier_agg["Supplier_Name"],
               supplier_agg["Supplier_Score"],
               color=colors, edgecolor="none", height=0.6)

for bar, score in zip(bars, supplier_agg["Supplier_Score"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}", va="center", ha="left",
            fontsize=9, color="white", fontweight="bold")

ax.set_xlabel("Supplier Score (0–100)", color="white", fontsize=11)
ax.set_title("Supplier Quality Score",
             color=ACCENT, fontsize=14, fontweight="bold", pad=12)
ax.tick_params(colors="white")
ax.spines[:].set_visible(False)
ax.set_xlim(0, 115)
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend = [Patch(color=GREEN, label="High  (≥70)"),
          Patch(color=GOLD,  label="Medium (55–70)"),
          Patch(color=RED,   label="Low  (<55)")]
ax.legend(handles=legend, loc="lower right",
          facecolor="#0D1117", edgecolor="#444", labelcolor="white")

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "supplier_scores.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Chart saved -> {p1}")

# ── 8b. ROC Curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

fpr, tpr, _ = roc_curve(y, y_proba)
auc_val      = roc_auc_score(y, y_proba)
ax.plot(fpr, tpr, color=ACCENT, lw=2,
        label=f"ROC  (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)

ax.set_xlabel("False Positive Rate", color="white")
ax.set_ylabel("True Positive Rate",  color="white")
ax.set_title("ROC Curve – Part Failure Prediction",
             color=ACCENT, fontsize=12, fontweight="bold")
ax.tick_params(colors="white")
ax.spines[:].set_color("#444")
ax.legend(facecolor="#0D1117", edgecolor="#444", labelcolor="white")

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "roc_curve.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Chart saved -> {p2}")

# ── 8c. Confusion Matrix ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

cm   = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["pass", "fail"])
disp.plot(ax=ax, colorbar=False,
          cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix", color=ACCENT,
             fontsize=12, fontweight="bold")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
for text in disp.text_.ravel():
    text.set_color("white")
    text.set_fontsize(14)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Chart saved -> {p3}")

# ── 8d. Feature Importance (if tree-based) ───────────────────────────────
if hasattr(best_pipe["clf"], "feature_importances_"):
    importances = best_pipe["clf"].feature_importances_
    fi_df = (pd.DataFrame({"feature": feature_cols, "importance": importances})
               .sort_values("importance", ascending=True)
               .tail(20))

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.barh(fi_df["feature"], fi_df["importance"],
                   color=ACCENT, edgecolor="none", height=0.65)

    # Highlight Supplier_Score
    for bar, feat in zip(bars, fi_df["feature"]):
        if feat == "Supplier_Score":
            bar.set_color(GOLD)
            bar.set_linewidth(2)

    ax.set_xlabel("Importance", color="white")
    ax.set_title("Feature Importances  (* = Supplier Score)",
                 color=ACCENT, fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    plt.tight_layout()
    p4 = os.path.join(OUT_DIR, "feature_importances.png")
    plt.savefig(p4, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {p4}")

# ── 8e. Score vs Failure Rate scatter ───────────────────────────────────
sup_stats = df.groupby("Supplier_ID").agg(
    Supplier_Name   = ("Supplier_Name", "first"),
    Supplier_Score  = ("Supplier_Score", "first"),
    Fail_Rate       = ("fail", "mean"),
    Sample_Count    = ("fail", "count"),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

sc = ax.scatter(sup_stats["Supplier_Score"],
                sup_stats["Fail_Rate"] * 100,
                s=sup_stats["Sample_Count"] * 15,
                c=sup_stats["Supplier_Score"],
                cmap="RdYlGn", vmin=40, vmax=90,
                edgecolors="white", linewidths=0.8, zorder=3)

for _, row in sup_stats.iterrows():
    ax.annotate(row["Supplier_Name"].split()[0],
                (row["Supplier_Score"], row["Fail_Rate"] * 100),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color="white")

cb = plt.colorbar(sc, ax=ax)
cb.set_label("Supplier Score", color="white")
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

ax.set_xlabel("Supplier Score", color="white")
ax.set_ylabel("Failure Rate (%)", color="white")
ax.set_title("Supplier Score  vs  Part Failure Rate",
             color=ACCENT, fontsize=12, fontweight="bold")
ax.tick_params(colors="white")
ax.spines[:].set_color("#444")
ax.grid(alpha=0.15)

plt.tight_layout()
p5 = os.path.join(OUT_DIR, "score_vs_failure.png")
plt.savefig(p5, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Chart saved -> {p5}")

# ── 8f. CV Results Bar ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor("#0D1117")

metric_labels = {"roc_auc": "ROC-AUC", "f1": "F1 Score"}
for ax, metric in zip(axes, ["roc_auc", "f1"]):
    ax.set_facecolor("#161B22")
    names  = list(results.keys())
    means  = [results[n][metric].mean() for n in names]
    stds   = [results[n][metric].std()  for n in names]
    colors_ = [GREEN if n == best_name else ACCENT for n in names]
    bars = ax.bar(range(len(names)), means, yerr=stds,
                  color=colors_, capsize=5,
                  error_kw={"ecolor": "white", "alpha": 0.6})
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names],
                       color="white", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric_labels[metric], color="white")
    ax.set_title(f"{metric_labels[metric]} (5-fold CV)",
                 color=ACCENT, fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                f"{m:.3f}", ha="center", fontsize=9,
                color="white", fontweight="bold")

plt.tight_layout()
p6 = os.path.join(OUT_DIR, "model_comparison.png")
plt.savefig(p6, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Chart saved -> {p6}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PREDICTION DEMO — score a new hypothetical part
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8 – Prediction demo on sample parts")
print("=" * 60)

# Take first 5 rows as "new" parts and re-predict
demo = df[feature_cols].head(5).copy()
preds  = best_pipe.predict(demo)
probas = best_pipe.predict_proba(demo)[:, 1]

print(f"\n  {'Sample':<8} {'Supplier_Score':>16} {'Fail_Prob':>12} {'Verdict':>10}")
print(f"  {'-'*50}")
for i, (pred, prob) in enumerate(zip(preds, probas)):
    score = df.loc[i, "Supplier_Score"]
    verdict = "[FAIL]" if pred == 1 else "[PASS]"
    print(f"  S{i+1:<7} {score:>16.1f} {prob:>11.1%}  {verdict:>10}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [DONE] Pipeline complete!")
print(f"  All outputs saved to: {OUT_DIR}")
print("=" * 60)
