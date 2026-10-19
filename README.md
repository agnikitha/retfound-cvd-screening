# RETFound-CVD: Non-Invasive Cardiovascular Risk & 5-Disease Ophthalmic AI Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20CUDA-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Foundation Model](https://img.shields.io/badge/Model-RETFound%20ViT--Large%20(304M)-06B6D4.svg)](https://www.nature.com/articles/s41586-023-06555-x)
[![GPU Cloud Benchmark](https://img.shields.io/badge/Kaggle%20GPU-Tesla%20T4%20Verified-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/code/xtygvh/retfound-cvd-clinical-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Clinical Summary:**  
> An enterprise-grade, explainable artificial intelligence system that transforms a routine 30-second color fundus photograph into an automated screening clinic for:
> 1. **Cardiovascular Disease (CVD) Risk:** Detecting Carotid Intima-Media Thickness ($\text{CIMT} \ge 0.90\text{ mm}$).
> 2. **Continuous CIMT Wall Thickness:** Estimating millimeter arterial thickening ($0.60\text{ mm} - 1.60\text{ mm}$).
> 3. **Retinal Biological Age & Sex:** Calculating Retinal Age Gap as an index of systemic vascular aging.
> 4. **5 Retinal Diseases (MuReD):** Diabetic Retinopathy (DR), Glaucoma/Optic Disc Cupping (ODC), Pathological Myopia (MYA), Tessellated Fundus (TSLN), and Retinal Vein Occlusion (BRVO).
> 5. **Explainable AI (XAI):** Generating layer-level Grad-CAM heatmaps hooked to Layer 20 of RETFound's 24-layer Vision Transformer.

---

## 🔬 Multi-Tier Clinical Validation (Verified on Kaggle Tesla T4 GPU)

| Clinical Tier | Definition | Cohort Size ($N$) | Accuracy (%) | AUROC | Clinical Sensitivity | False Positives | False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Strict Unfiltered** | Raw test cohort. Zero exclusions. | **$N = 100$** | **89.00%** | **0.9472** | 90.00% | 6 | 5 |
| **Tier 2: Paper Benchmark** | Excludes $0.90\text{ mm}$ acoustic probe noise (`thickness != 0.9`). | **$N = 92$** | **92.39%** | **0.9714** | **97.62%** | 6 | **1** |
| **Tier 3: Clinical Triage** | Decisive cohort ($p \le 0.78$ or $p \ge 0.85$). | **$N = 80$** | **95.00%** | **0.9855** | **97.22%** | **3** | **1** |

*Note:* In Tier 2, out of 42 sick patients with arterial plaque, the AI correctly identifies **41 of them** (only 1 missed), achieving a **97.62% Clinical Sensitivity**.

---

## 🚀 Key Features

* **Dual Demographics Mode:**
  * **Auto-Predict Mode:** Infers biological vascular age and biological sex directly from microvascular remodeling.
  * **Manual Override Mode:** Allows entering chronological age and sex manually to calculate the patient's biological *Retinal Age Gap*.
* **Layer 20/24 ViT Grad-CAM:** Targets superior and inferior temporal retinal arteriolar arcades without black-border edge leakage.
* **Full Split Audit Trail:** Trained deterministically (`seed=42`) across 2,603 training patients, 200 validation patients, and 100 locked test patients.
* **Modern Web Interface:** Dark glassmorphism UI with real-time drag-and-drop ingestion, progress bars, and automated medical diagnostic reporting.

---

## 🛠️ Quick Start & Local Web App

### 1. Installation
```bash
git clone https://github.com/agnikitha/retfound-cvd-screening.git
cd retfound-cvd-screening
pip install -r requirements.txt
```

### 2. Launch the Clinical Screening GUI
```bash
python unified_web_gui.py
```
Open your browser and navigate to:  
👉 **`http://127.0.0.1:5050`**

---

## 📁 Repository Structure

```
├── unified_web_gui.py                     # Single-file interactive web application
├── models_cache.pkl                       # Serialized L2 linear probes, regressors & disease heads
├── retfound_gradcam_cvd.py                # Standalone ViT Block -4 Grad-CAM engine
├── create_audit_verified_dashboard.py     # 3-panel verified clinical dashboard generator
├── audit_comprehensive_evaluation.py     # All-splits evaluation script
├── eval_regression_baseline.py            # CIMT continuous mm regression baseline
├── test_predictions_audit.csv             # 100 test patient audited predictions
├── final_audit_cimt_retinal_risks.csv     # 100 test patient 5-disease probabilities
├── retfound_cvd_comprehensive_audit_report.md # Full academic audit report
├── kaggle_kernel/
│   └── retfound_cvd_kaggle.ipynb          # Cloud GPU benchmark notebook
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 📜 Citation & Acknowledgements
* **RETFound Foundation Model:** Zhou et al., *"A foundation model for generalizable disease detection from retinal images."* Nature 622, 156–163 (2023).
* **China Fundus-CIMT Cohort:** High-resolution fundus images paired with B-mode carotid ultrasound.
* **MuReD Ophthalmic Dataset:** Multi-label retinal disease benchmark.

---
*Developed by Clinical AI Engineering Pair (October 2026).*
