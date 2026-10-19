import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# Set random seed for 100% reproducibility
np.random.seed(42)

def load_data(filepath):
    d = np.load(filepath, allow_pickle=True)
    return {
        'pids': d['pids'],
        'left': d['left_embeds'].astype(np.float32),
        'right': d['right_embeds'].astype(np.float32),
        'retina': np.concatenate([d['left_embeds'], d['right_embeds']], axis=1).astype(np.float32),
        'demo': np.column_stack([d['age'], d['gender']]).astype(np.float32),
        'label': d['label'].astype(np.float32),
        'thickness': d['thickness'].astype(np.float32)
    }

print("Loading CIMT Cohorts (Train: 2,603, Val: 200, Test: 100)...")
train_d = load_data('D:/retfound anti/cimt_train_embeds.npz')
val_d = load_data('D:/retfound anti/cimt_val_embeds.npz')
test_d = load_data('D:/retfound anti/cimt_test_embeds.npz')

# 1. Feature Preprocessing (Fit scaler on Train only)
scaler_demo = StandardScaler()
X_demo_tr = scaler_demo.fit_transform(train_d['demo'])
X_demo_va = scaler_demo.transform(val_d['demo'])
X_demo_te = scaler_demo.transform(test_d['demo'])

scaler_ret = StandardScaler()
X_ret_tr = scaler_ret.fit_transform(train_d['retina'])
X_ret_va = scaler_ret.transform(val_d['retina'])
X_ret_te = scaler_ret.transform(test_d['retina'])

# 2. Train Branch Models
print("Training Demographic Branch (Age + Gender)...")
clf_demo = LogisticRegression(C=1.0, random_state=42)
clf_demo.fit(X_demo_tr, train_d['label'])

print("Training Ocular Branch (RETFound Bilateral 2,048 embeddings)...")
clf_ret = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
clf_ret.fit(X_ret_tr, train_d['label'])

# Probabilities
p_demo_va = clf_demo.predict_proba(X_demo_va)[:, 1]
p_demo_te = clf_demo.predict_proba(X_demo_te)[:, 1]

p_ret_va = clf_ret.predict_proba(X_ret_va)[:, 1]
p_ret_te = clf_ret.predict_proba(X_ret_te)[:, 1]

# 3. Model 1 (Demographics Only Evaluation)
fpr_d, tpr_d, thresh_d = roc_curve(val_d['label'], p_demo_va)
tau_demo = thresh_d[np.argmax(tpr_d - fpr_d)]
bin_demo_te = (p_demo_te >= tau_demo).astype(int)

# 4. Model 2 (Retina Only Evaluation)
fpr_r, tpr_r, thresh_r = roc_curve(val_d['label'], p_ret_va)
tau_ret = thresh_r[np.argmax(tpr_r - fpr_r)]
bin_ret_te = (p_ret_te >= tau_ret).astype(int)

# 5. Model 3 (Hierarchical Late Fusion - Tuned STRICTLY on Validation Split)
print("\nOptimizing Hierarchical Fusion on Validation Split (N=200)...")
best_w = 0.5
best_val_auc = 0.0

for w in np.linspace(0.50, 0.98, 49):
    p_comb_va = w * p_demo_va + (1 - w) * p_ret_va
    auc_v = roc_auc_score(val_d['label'], p_comb_va)
    if auc_v > best_val_auc:
        best_val_auc = auc_v
        best_w = w

# Determine optimal threshold on validation set
p_fused_va = best_w * p_demo_va + (1 - best_w) * p_ret_va
fpr_f, tpr_f, thresh_f = roc_curve(val_d['label'], p_fused_va)
tau_fused = thresh_f[np.argmax(tpr_f - fpr_f)]

# 6. Apply to Unseen Test Cohort (N=100)
p_fused_te = best_w * p_demo_te + (1 - best_w) * p_ret_te
bin_fused_te = (p_fused_te >= tau_fused).astype(int)

print("\n--- ABLATION BENCHMARK (ON 100 TEST PATIENTS) ---")

def report_model(name, y_true, p_pred, b_pred, tau):
    acc = accuracy_score(y_true, b_pred)
    bacc = balanced_accuracy_score(y_true, b_pred)
    auc = roc_auc_score(y_true, p_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, b_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print(f"[{name}] tau={tau:.4f} | AUROC={auc:.4f} | Acc={acc*100:.2f}% | BalAcc={bacc*100:.2f}% | Sens={sens*100:.2f}% | Spec={spec*100:.2f}% | (TP={tp}, FP={fp}, TN={tn}, FN={fn})")

report_model("Demographics Only (Age+Gender)", test_d['label'], p_demo_te, bin_demo_te, tau_demo)
report_model("Retina Only (RETFound Bilateral)", test_d['label'], p_ret_te, bin_ret_te, tau_ret)
report_model(f"Hierarchical Fused (w_demo={best_w:.2f})", test_d['label'], p_fused_te, bin_fused_te, tau_fused)

# 7. 95% Bootstrap Confidence Intervals for the Hierarchical Fused Model
n_boot = 1000
boot_aucs, boot_accs, boot_sens, boot_specs = [], [], [], []
y_te = test_d['label']

for i in range(n_boot):
    idx = np.random.choice(len(y_te), size=len(y_te), replace=True)
    if len(np.unique(y_te[idx])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_te[idx], p_fused_te[idx]))
    boot_accs.append(accuracy_score(y_te[idx], bin_fused_te[idx]))
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_te[idx], bin_fused_te[idx], labels=[0, 1]).ravel()
    boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)
    boot_specs.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)

print("\n--- 95% BOOTSTRAP CONFIDENCE INTERVALS (HIERARCHICAL FUSED MODEL) ---")
print(f"AUROC:       {np.mean(boot_aucs):.4f} (95% CI: {np.percentile(boot_aucs, 2.5):.4f} - {np.percentile(boot_aucs, 97.5):.4f})")
print(f"Accuracy:    {np.mean(boot_accs)*100:.2f}% (95% CI: {np.percentile(boot_accs, 2.5)*100:.2f}% - {np.percentile(boot_accs, 97.5)*100:.2f}%)")
print(f"Sensitivity: {np.mean(boot_sens)*100:.2f}% (95% CI: {np.percentile(boot_sens, 2.5)*100:.2f}% - {np.percentile(boot_sens, 97.5)*100:.2f}%)")
print(f"Specificity: {np.mean(boot_specs)*100:.2f}% (95% CI: {np.percentile(boot_specs, 2.5)*100:.2f}% - {np.percentile(boot_specs, 97.5)*100:.2f}%)")

# 8. CIMT Thickness Continuous Regression
# We fit Ridge on Demographic features + Retinal PCA/features
ridge = Ridge(alpha=10.0, random_state=42)
X_fused_tr = np.concatenate([X_demo_tr, X_ret_tr], axis=1)
X_fused_te = np.concatenate([X_demo_te, X_ret_te], axis=1)
ridge.fit(X_fused_tr, train_d['thickness'])
pred_thick = ridge.predict(X_fused_te)

mae = mean_absolute_error(test_d['thickness'], pred_thick)
rmse = np.sqrt(mean_squared_error(test_d['thickness'], pred_thick))
r_val, p_val = pearsonr(test_d['thickness'], pred_thick)

print("\n--- CIMT THICKNESS REGRESSION METRICS ---")
print(f"MAE:       {mae:.4f} mm")
print(f"RMSE:      {rmse:.4f} mm")
print(f"Pearson r: {r_val:.4f} (p-value: {p_val:.2e})")

# 9. Save Predictions CSV for Independent Verification
audit_df = pd.DataFrame({
    'Patient_ID': test_d['pids'].astype(int),
    'Age': test_d['demo'][:, 0].astype(int),
    'Gender': test_d['demo'][:, 1].astype(int),
    'Actual_CIMT_mm': test_d['thickness'],
    'Predicted_CIMT_mm': pred_thick,
    'Actual_CVD_Label': test_d['label'].astype(int),
    'Prob_Demographics_Only': p_demo_te,
    'Class_Demographics_Only': bin_demo_te,
    'Prob_Retina_Only': p_ret_te,
    'Class_Retina_Only': bin_ret_te,
    'Prob_Hierarchical_Fused': p_fused_te,
    'Class_Hierarchical_Fused': bin_fused_te
})
audit_df.to_csv('D:/new pro/test_predictions_audit.csv', index=False)
print("\nSaved test_predictions_audit.csv successfully with 100 rows containing all 3 models.")
