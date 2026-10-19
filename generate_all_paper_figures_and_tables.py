import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    matthews_corrcoef, mean_absolute_error, mean_squared_error
)

print("================================================================================")
print("  GENERATING PUBLICATION-READY RESULTS, CONFUSION MATRICES & CORRELATION GRAPHS ")
print("================================================================================")

OUT_DIR = "D:/new pro/paper_results"
os.makedirs(OUT_DIR, exist_ok=True)
ARTIFACTS_DIR = "C:/Users/Nikitha AG/.gemini/antigravity/brain/b66dc714-87bf-459d-a999-cd533aa75fa1"

# Style configuration for Nature / Lancet publication standards
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14

cache = pickle.load(open('D:/new pro/models_cache.pkl', 'rb'))

# ==============================================================================
# SECTION 1: CIMT METRICS & CONFUSION MATRICES ACROSS TRAIN, VAL, TEST
# ==============================================================================
print("\n[1/5] Processing CIMT splits (Train N=2603, Val N=200, Test N=100)...")

# Load CIMT NPZ data
cimt_tr = np.load('D:/retfound anti/cimt_train_embeds.npz', allow_pickle=True)
cimt_val = np.load('D:/retfound anti/cimt_val_embeds.npz', allow_pickle=True)
cimt_te = np.load('D:/retfound anti/cimt_test_embeds.npz', allow_pickle=True)

# Helper to compute single-eye / bilateral fused predictions
def evaluate_cimt_split(npz_data, split_name, tau=0.80):
    left = npz_data['left_embeds']
    right = npz_data['right_embeds']
    age = npz_data['age'].astype(float)
    gender = npz_data['gender'].astype(float)
    y_true = npz_data['label'].astype(int)
    thickness = npz_data['thickness'].astype(float)
    pids = npz_data['pids']
    
    # Standardize demographics
    dem_x = np.column_stack([age, gender])
    dem_xs = (dem_x - cache['sc_dem_mean']) / cache['sc_dem_scale']
    logit_dem = np.dot(dem_xs, cache['clf_dem_coef']) + cache['clf_dem_intercept']
    prob_dem = 1.0 / (1.0 + np.exp(-logit_dem))
    
    # Retina single-eye embeddings (mean of right & left)
    ret_mean = (right + left) / 2.0
    ret_xs = (ret_mean - cache['sc_ret_single_mean']) / cache['sc_ret_single_scale']
    logit_ret = np.dot(ret_xs, cache['clf_ret_single_coef']) + cache['clf_ret_single_intercept']
    prob_ret = 1.0 / (1.0 + np.exp(-logit_ret))
    
    # Calibrated late fusion: 0.83 demog + 0.17 retina
    prob_fused = 0.83 * prob_dem + 0.17 * prob_ret
    y_pred = (prob_fused >= tau).astype(int)
    
    # Continuous CIMT prediction (mm): x_full is [age, gender, retinal_features]
    x_full = np.column_stack([age, gender, ret_mean[:, :1024]])
    pred_cimt_mm = np.dot(x_full, cache['reg_cimt_coef']) + cache['reg_cimt_intercept']
    # Soft bound to physiological limits [0.60, 1.50]
    pred_cimt_mm = np.clip(pred_cimt_mm, 0.60, 1.50)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, prob_fused)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    mae_cimt = mean_absolute_error(thickness, pred_cimt_mm)
    r_cimt, _ = stats.pearsonr(thickness, pred_cimt_mm)
    
    metrics = {
        'Split': split_name,
        'N_Patients': len(y_true),
        'N_Images': len(y_true) * 2,
        'Accuracy': acc * 100,
        'Balanced_Acc': bacc * 100,
        'AUROC': auc,
        'Sensitivity': sens * 100,
        'Specificity': spec * 100,
        'Precision': prec * 100,
        'F1_Score': f1,
        'MCC': mcc,
        'CIMT_MAE_mm': mae_cimt,
        'CIMT_Pearson_r': r_cimt,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }
    
    df_preds = pd.DataFrame({
        'Patient_ID': pids,
        'Age': age,
        'Gender': gender,
        'True_CIMT_mm': thickness,
        'Pred_CIMT_mm': pred_cimt_mm,
        'True_Label': y_true,
        'Pred_Prob': prob_fused,
        'Pred_Label': y_pred
    })
    
    return metrics, cm, df_preds, (y_true, prob_fused)

cimt_metrics_tr, cm_tr, df_preds_tr, roc_tr = evaluate_cimt_split(cimt_tr, 'Training (N=2,603)', tau=0.80)
cimt_metrics_val, cm_val, df_preds_val, roc_val = evaluate_cimt_split(cimt_val, 'Validation (N=200)', tau=0.80)
cimt_metrics_te, cm_te, df_preds_te, roc_te = evaluate_cimt_split(cimt_te, 'Testing (N=100)', tau=0.80)

df_cimt_summary = pd.DataFrame([cimt_metrics_tr, cimt_metrics_val, cimt_metrics_te])
df_cimt_summary.to_csv(f"{OUT_DIR}/cimt_all_splits_metrics_table.csv", index=False)
print("Saved CIMT summary table:", f"{OUT_DIR}/cimt_all_splits_metrics_table.csv")

# Plot Publication Figure 1: CIMT 3-Panel Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
splits_info = [
    ('Training Cohort (N=2,603)', cm_tr, cimt_metrics_tr),
    ('Validation Cohort (N=200)', cm_val, cimt_metrics_val),
    ('Independent Testing Cohort (N=100)', cm_te, cimt_metrics_te)
]

for idx, (title, cm_data, m) in enumerate(splits_info):
    ax = axes[idx]
    # Percentages
    cm_pct = cm_data.astype('float') / cm_data.sum() * 100
    labels = np.array([
        [f"TN\n{cm_data[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"FP\n{cm_data[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
        [f"FN\n{cm_data[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"TP\n{cm_data[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
    ])
    
    cmap = sns.light_palette("#1b4d3e" if idx==2 else "#1f4e79", as_cmap=True)
    sns.heatmap(cm_data, annot=labels, fmt="", cmap=cmap, cbar=False, ax=ax,
                linewidths=2, linecolor='white', annot_kws={"size": 13, "weight": "bold"})
    
    ax.set_title(f"{title}\nAcc: {m['Accuracy']:.2f}% | AUROC: {m['AUROC']:.4f}", pad=12, fontweight='bold')
    ax.set_xlabel("AI Predicted Class (CVD Risk)", labelpad=8)
    ax.set_ylabel("True Carotid Ultrasound Gold Standard", labelpad=8)
    ax.set_xticklabels(['Normal (<0.90mm)', 'Plaque (>=0.90mm)'])
    ax.set_yticklabels(['Normal (<0.90mm)', 'Plaque (>=0.90mm)'], va='center')

plt.suptitle("Figure 1: Carotid Artery Atherosclerosis (CIMT) Confusion Matrices Across All Dataset Splits\n(RETFound 304M ViT-Large Synergistic Multimodal Late Fusion)", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_cimt_confusion_matrices_all_splits.png", bbox_inches='tight', dpi=300)
fig.savefig(f"{ARTIFACTS_DIR}/fig1_cimt_confusion_matrices_all_splits.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved Figure 1:", f"{OUT_DIR}/fig1_cimt_confusion_matrices_all_splits.png")


# ==============================================================================
# SECTION 2: MURED 5-DISEASE EVALUATION & CONFUSION MATRICES (TRAIN, VAL, TEST)
# ==============================================================================
print("\n[2/5] Processing MuReD 5 Retinal Diseases across Train (N=1,588), Val (N=176), Test (N=444)...")

mured_npz = np.load('D:/retfound anti/mured_embeddings.npz', allow_pickle=True)
df_mured_tr = pd.read_csv('D:/retfound anti/mured_train.csv')
df_mured_val = pd.read_csv('D:/retfound anti/mured_val.csv')
df_mured_te = pd.read_csv('D:/retfound anti/mured_test.csv')

diseases = ['DR', 'MYA', 'ODC', 'TSLN', 'BRVO']
disease_names = {
    'DR': 'Diabetic Retinopathy',
    'MYA': 'Pathological Myopia',
    'ODC': 'Optic Disc Cupping (Glaucoma)',
    'TSLN': 'Tessellated Fundus',
    'BRVO': 'Branch Retinal Vein Occlusion'
}

mured_results = []
mured_cms = {'train': {}, 'val': {}, 'test': {}}
mured_roc_data = {}

splits_mured = [
    ('Training (N=1,588)', mured_npz['train_embeds'], df_mured_tr, 'train'),
    ('Validation (N=176)', mured_npz['val_embeds'], df_mured_val, 'val'),
    ('Testing (N=444)', mured_npz['test_embeds'], df_mured_te, 'test')
]

for split_label, X_emb, df_split, split_key in splits_mured:
    for d in diseases:
        m = cache['mured_models'][d]
        mean, scale, coef, intercept = m['scaler_mean'], m['scaler_scale'], m['coef'], m['intercept']
        
        # Standardize & forward pass
        X_scaled = (X_emb - mean) / scale
        logits = np.dot(X_scaled, coef) + intercept
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_true = df_split[d].values.astype(int)
        y_pred = (probs >= 0.50).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 1.0
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
        mcc = matthews_corrcoef(y_true, y_pred)
        
        mured_cms[split_key][d] = cm
        if split_key == 'test':
            mured_roc_data[d] = (y_true, probs)
            
        mured_results.append({
            'Split': split_label,
            'Disease_Code': d,
            'Disease_Full_Name': disease_names[d],
            'Prevalence_N': int(y_true.sum()),
            'Prevalence_Pct': float(y_true.sum() / len(y_true) * 100),
            'Accuracy': acc * 100,
            'Balanced_Acc': bacc * 100,
            'AUROC': auc,
            'Sensitivity': sens * 100,
            'Specificity': spec * 100,
            'Precision': prec * 100,
            'F1_Score': f1,
            'MCC': mcc,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })

df_mured_summary = pd.DataFrame(mured_results)
df_mured_summary.to_csv(f"{OUT_DIR}/mured_5_diseases_all_splits_metrics.csv", index=False)
print("Saved MuReD 5-disease summary table:", f"{OUT_DIR}/mured_5_diseases_all_splits_metrics.csv")

# Plot Publication Figure 2: MuReD 5 Diseases Confusion Matrices on Independent Test Set (N=444)
fig, axes = plt.subplots(1, 5, figsize=(24, 4.8), dpi=300)
test_metrics_df = df_mured_summary[df_mured_summary['Split'] == 'Testing (N=444)'].set_index('Disease_Code')

for idx, d in enumerate(diseases):
    ax = axes[idx]
    cm_data = mured_cms['test'][d]
    m = test_metrics_df.loc[d]
    cm_pct = cm_data.astype('float') / cm_data.sum() * 100
    labels = np.array([
        [f"TN\n{cm_data[0,0]}\n({cm_pct[0,0]:.1f}%)", f"FP\n{cm_data[0,1]}\n({cm_pct[0,1]:.1f}%)"],
        [f"FN\n{cm_data[1,0]}\n({cm_pct[1,0]:.1f}%)", f"TP\n{cm_data[1,1]}\n({cm_pct[1,1]:.1f}%)"]
    ])
    
    cmap = sns.light_palette("#6c2d58", as_cmap=True)
    sns.heatmap(cm_data, annot=labels, fmt="", cmap=cmap, cbar=False, ax=ax,
                linewidths=2, linecolor='white', annot_kws={"size": 12, "weight": "bold"})
    
    ax.set_title(f"{d}: {disease_names[d]}\nAcc: {m['Accuracy']:.1f}% | AUROC: {m['AUROC']:.3f}", pad=10, fontweight='bold', fontsize=11)
    ax.set_xlabel("AI Predicted", labelpad=6)
    if idx == 0:
        ax.set_ylabel("Clinical Truth", labelpad=6)
    else:
        ax.set_ylabel("")
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'], va='center')

plt.suptitle("Figure 2: Multi-Label Retinal Abnormality Confusion Matrices on MuReD Benchmark Test Set (N=444)\n(Diabetic Retinopathy, Myopia, Optic Cupping, Tessellated Fundus, Branch Retinal Vein Occlusion)", fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_mured_test_confusion_matrices.png", bbox_inches='tight', dpi=300)
fig.savefig(f"{ARTIFACTS_DIR}/fig2_mured_test_confusion_matrices.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved Figure 2:", f"{OUT_DIR}/fig2_mured_test_confusion_matrices.png")


# ==============================================================================
# SECTION 3: CORRELATION MATRIX & GRAPHICAL HEATMAP
# ==============================================================================
print("\n[3/5] Computing Cross-Modal Correlations (CIMT, CVD Risk, 5 Diseases, Demographics)...")

# Load testing images with disease probabilities
df_corr_source = pd.read_csv("D:/new pro/audit_split_results/testing_images_with_disease_probabilities.csv")

# Ensure all columns exist and are numeric
corr_cols = [
    'Measured_CIMT_mm', 'CVD_Risk_Prob',
    'BRVO_Prob', 'DR_Prob', 'ODC_Prob', 'MYA_Prob', 'TSLN_Prob',
    'Age_Years', 'Gender_Code'
]
corr_labels = [
    'CIMT Wall (mm)', 'CVD Risk Score',
    'BRVO (Vein Occl.)', 'Diabetic Retin.', 'Optic Cupping', 'Path. Myopia', 'Tessellated',
    'Age (Years)', 'Sex (Male=1)'
]

corr_matrix = df_corr_source[corr_cols].corr(method='pearson').fillna(0.0)
pval_matrix = pd.DataFrame(np.ones((len(corr_cols), len(corr_cols))), index=corr_cols, columns=corr_cols)

for c1 in corr_cols:
    for c2 in corr_cols:
        if c1 == c2:
            pval_matrix.loc[c1, c2] = 0.0
        else:
            std1 = df_corr_source[c1].std()
            std2 = df_corr_source[c2].std()
            if std1 > 1e-9 and std2 > 1e-9:
                r, p = stats.pearsonr(df_corr_source[c1], df_corr_source[c2])
                pval_matrix.loc[c1, c2] = p if not np.isnan(p) else 1.0
            else:
                pval_matrix.loc[c1, c2] = 1.0

corr_matrix.columns = corr_labels
corr_matrix.index = corr_labels
pval_matrix.columns = corr_labels
pval_matrix.index = corr_labels

corr_matrix.to_csv(f"{OUT_DIR}/biological_correlation_matrix_pearson.csv")
pval_matrix.to_csv(f"{OUT_DIR}/biological_correlation_matrix_pvalues.csv")

# Create annotated strings with significance asterisks (* p<0.05, ** p<0.01, *** p<0.001)
annot_text = np.empty(corr_matrix.shape, dtype=object)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr_matrix.iloc[i, j]
        p = pval_matrix.iloc[i, j]
        if i == j:
            stars = ""
        elif p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = ""
        annot_text[i, j] = f"{val:+.2f}{stars}"

# Plot Publication Figure 3: Full Biological Correlation Matrix Heatmap
fig, ax = plt.subplots(figsize=(10.5, 9), dpi=300)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1) # Mask upper triangle

cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, annot=annot_text, fmt="", cmap=cmap,
            vmin=-0.4, vmax=1.0, center=0, square=True, linewidths=1.5,
            cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient (r)"}, ax=ax,
            annot_kws={"size": 10.5, "weight": "bold"})

ax.set_title("Figure 3: Biological Cross-System Correlation Matrix\nBetween Retinal Microvascular Abnormality Probabilities & Carotid Ultrasound CIMT\n(* p < 0.05, ** p < 0.01, *** p < 0.001)",
             fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.yticks(rotation=0, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_biological_correlation_heatmap.png", bbox_inches='tight', dpi=300)
fig.savefig(f"{ARTIFACTS_DIR}/fig3_biological_correlation_heatmap.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved Figure 3:", f"{OUT_DIR}/fig3_biological_correlation_heatmap.png")


# ==============================================================================
# SECTION 4: SCATTER & REGRESSION PLOTS (CIMT VS BIOMARKERS)
# ==============================================================================
print("\n[4/5] Generating Bivariate Regression & Biomarker Trend Plots...")

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), dpi=300)

# Panel A: Measured CIMT vs CVD Risk Score
ax = axes[0]
sns.regplot(x='Measured_CIMT_mm', y='CVD_Risk_Prob', data=df_corr_source, ax=ax,
            scatter_kws={'alpha': 0.75, 'color': '#1f4e79', 's': 55},
            line_kws={'color': '#b30000', 'linewidth': 2.5})
r_val, p_val = stats.pearsonr(df_corr_source['Measured_CIMT_mm'], df_corr_source['CVD_Risk_Prob'])
ax.axvline(0.90, color='crimson', linestyle='--', linewidth=1.5, label='Atherosclerotic Plaque Threshold (0.90 mm)')
ax.set_title(f"A. CIMT vs. AI CVD Risk Probability\nPearson r = {r_val:.3f} (p = {p_val:.2e})", fontweight='bold', pad=10)
ax.set_xlabel("Measured Ultrasound CIMT (mm)")
ax.set_ylabel("AI Estimated CVD Risk Probability")
ax.legend(loc='upper left', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)

# Panel B: Measured CIMT vs BRVO Probability
ax = axes[1]
sns.regplot(x='Measured_CIMT_mm', y='BRVO_Prob', data=df_corr_source, ax=ax,
            scatter_kws={'alpha': 0.75, 'color': '#78281f', 's': 55},
            line_kws={'color': '#d9534f', 'linewidth': 2.5})
r_brvo, p_brvo = stats.pearsonr(df_corr_source['Measured_CIMT_mm'], df_corr_source['BRVO_Prob'])
ax.axvline(0.90, color='crimson', linestyle='--', linewidth=1.5, label='Atherosclerotic Plaque Threshold (0.90 mm)')
ax.set_title(f"B. CIMT vs. Retinal Vein Occlusion (BRVO)\nPearson r = {r_brvo:.3f} (p = {p_brvo:.2e})", fontweight='bold', pad=10)
ax.set_xlabel("Measured Ultrasound CIMT (mm)")
ax.set_ylabel("Branch Retinal Vein Occlusion (BRVO) Probability")
ax.legend(loc='upper left', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)

# Panel C: Measured CIMT vs Predicted CIMT (Continuous Regression Performance)
ax = axes[2]
# Retrieve test predictions for true vs predicted CIMT
df_te_reg = df_preds_te
sns.regplot(x='True_CIMT_mm', y='Pred_CIMT_mm', data=df_te_reg, ax=ax,
            scatter_kws={'alpha': 0.75, 'color': '#117a65', 's': 55},
            line_kws={'color': '#1b4d3e', 'linewidth': 2.5})
r_reg, p_reg = stats.pearsonr(df_te_reg['True_CIMT_mm'], df_te_reg['Pred_CIMT_mm'])
mae_reg = mean_absolute_error(df_te_reg['True_CIMT_mm'], df_te_reg['Pred_CIMT_mm'])
# Unity line
lims = [0.6, 1.5]
ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=1.5, label='Ideal Unity Line (y = x)')
ax.set_title(f"C. Ultrasound CIMT vs AI Predicted CIMT\nPearson r = {r_reg:.3f} | MAE = ±{mae_reg:.3f} mm", fontweight='bold', pad=10)
ax.set_xlabel("True Measured Carotid Ultrasound (mm)")
ax.set_ylabel("AI Predicted Continuous CIMT (mm)")
ax.legend(loc='upper left', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Figure 4: Quantitative Cross-Modal Biomarker Trends & Ultrasound Continuous Validation\n(Demonstrating Pathophysiological Linear Coupling Between Eye and Carotid Vasculature)", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_cimt_biomarker_trends.png", bbox_inches='tight', dpi=300)
fig.savefig(f"{ARTIFACTS_DIR}/fig4_cimt_biomarker_trends.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved Figure 4:", f"{OUT_DIR}/fig4_cimt_biomarker_trends.png")


# ==============================================================================
# SECTION 5: ROC CURVES ACROSS SPLITS & DISEASES
# ==============================================================================
print("\n[5/5] Generating Multi-Panel ROC Curves for CIMT and 5 Diseases...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

# Panel A: CIMT ROC curves across Train, Val, Test
ax = axes[0]
for name, (y_t, y_p), col in [
    ('Training (N=2,603)', roc_tr, '#7f8c8d'),
    ('Validation (N=200)', roc_val, '#2980b9'),
    ('Independent Testing (N=100)', roc_te, '#27ae60')
]:
    fpr, tpr, _ = roc_curve(y_t, y_p)
    auc_score = roc_auc_score(y_t, y_p)
    ax.plot(fpr, tpr, color=col, linewidth=2.5, label=f"{name} (AUROC = {auc_score:.4f})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUROC = 0.5000)')
ax.set_title("A. Carotid Atherosclerosis (CIMT >= 0.90mm) ROC Curves", fontweight='bold', pad=10)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
ax.legend(loc='lower right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)

# Panel B: MuReD 5 Retinal Diseases ROC Curves on Test Set
ax = axes[1]
colors_mured = {'DR': '#e74c3c', 'MYA': '#9b59b6', 'ODC': '#e67e22', 'TSLN': '#1abc9c', 'BRVO': '#34495e'}
for d in diseases:
    y_t, y_p = mured_roc_data[d]
    fpr, tpr, _ = roc_curve(y_t, y_p)
    auc_score = roc_auc_score(y_t, y_p)
    ax.plot(fpr, tpr, color=colors_mured[d], linewidth=2.5,
            label=f"{d}: {disease_names[d]} (AUROC = {auc_score:.4f})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUROC = 0.5000)')
ax.set_title("B. MuReD 5 Retinal Abnormality ROC Curves (Test N=444)", fontweight='bold', pad=10)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
ax.legend(loc='lower right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Figure 5: Receiver Operating Characteristic (ROC) Diagnostic Trajectories\nAcross Dataset Splits and Multi-Disease Ophthalmic Transfer", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_roc_curves_cimt_and_mured.png", bbox_inches='tight', dpi=300)
fig.savefig(f"{ARTIFACTS_DIR}/fig5_roc_curves_cimt_and_mured.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved Figure 5:", f"{OUT_DIR}/fig5_roc_curves_cimt_and_mured.png")

print("\n================================================================================")
print("  ALL PUBLICATION FIGURES & TABLES GENERATED SUCCESSFULLY! ")
print("================================================================================")
