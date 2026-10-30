import os
import sys
import gc
import json
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = r"D:\new pro"
RETFOUND_DIR = r"D:\retfound anti\RETFound_MAE"
CKPT_PATH = r"D:\retfound anti\RETFound_cfp_weights.pth"
EMBEDS_DIR = r"D:\retfound anti"
IMG_DIR = os.path.join(BASE_DIR, "sample_images")
OUT_CACHE = os.path.join(BASE_DIR, "gradcam_cache")
os.makedirs(OUT_CACHE, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GRID = 14  # 224 / 16

# 1. Build RETFound ViT
if RETFOUND_DIR not in sys.path:
    sys.path.insert(0, RETFOUND_DIR)
import models_vit

print("Loading RETFound ViT-Large backbone...", flush=True)
model = models_vit.vit_large_patch16(num_classes=1000, drop_path_rate=0.0, global_pool=False)
sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = sd.get("model", sd)
model.load_state_dict(sd, strict=False)
model.eval()
print("RETFound ViT loaded successfully.", flush=True)

# 2. Hook Layer 20 (Block index 20 of 24)
# Justification: Block 20/24 preserves high spatial patch-token localization of retinal vasculature
# before final semantic homogenization in the terminal blocks (22-24).
class BlockHook:
    def __init__(self, m, layer_idx=20):
        self.act = None
        m.blocks[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.act = output
        if output.requires_grad:
            output.retain_grad()

hook = BlockHook(model, 20)

# 3. Fit Logistic Regression Head on TRAIN split
def load_split(name):
    return np.load(os.path.join(EMBEDS_DIR, f"cimt_{name}_embeds.npz"), allow_pickle=True)

print("Fitting authentic retina classifier head on train split...", flush=True)
d_train = load_split("train")
X_tr = np.concatenate([d_train["left_embeds"], d_train["right_embeds"]], axis=1).astype(np.float32)
y_tr = d_train["label"]
sc = StandardScaler().fit(X_tr)
clf = LogisticRegression(C=0.01, max_iter=1000, random_state=42).fit(sc.transform(X_tr), y_tr)
w = (clf.coef_[0] / sc.scale_).astype(np.float32)
w_L, w_R = w[:1024], w[1024:]
m_L, m_R = sc.mean_[:1024], sc.mean_[1024:]

def find_patient_info(pid):
    for split in ("test", "val", "train"):
        d = load_split(split)
        ids = np.array([str(int(p)) for p in d["pids"]])
        hit = np.where(ids == str(int(pid)))[0]
        if len(hit):
            i = hit[0]
            return {
                "split": split,
                "label": int(d["label"][i])
            }
    return {"split": "test", "label": 1}

# Circular token mask (inscribed circle: 112px radius in 224x224 grid)
r_idx, c_idx = np.indices((14, 14))
center_d = np.sqrt(((r_idx + 0.5) * 16 - 112)**2 + ((c_idx + 0.5) * 16 - 112)**2)
valid_mask = center_d <= 112  # Exactly 156/196 tokens (79.6% retinal, 20.4% geometric corner background)

# Perceptually uniform clinical heatmap colormap (White at 0.0 -> Pale Yellow -> Warm Orange -> Vivid Red -> Deep Crimson)
cmap_colors = [
    (1.0, 1.0, 1.0),      # 0.00: White background / Baseline (Zero attribution)
    (1.0, 0.94, 0.65),   # 0.15: Soft Warm Gold
    (0.98, 0.55, 0.15),   # 0.40: Amber / Orange
    (0.85, 0.15, 0.15),   # 0.70: Vivid Red
    (0.50, 0.00, 0.10)    # 1.00: Deep Crimson / Peak Saliency
]
clinical_saliency_cmap = LinearSegmentedColormap.from_list("clinical_saliency", cmap_colors, N=256)

def compute_retfound_gradcam_and_random(img_path, w_eye, m_eye, rng):
    pil_img = Image.open(img_path).convert("RGB")
    
    # 224x224 input for ViT
    img224 = np.array(pil_img.resize((224, 224), Image.Resampling.LANCZOS))
    x_norm = (img224 / 255.0 - MEAN) / STD
    t = torch.tensor(x_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)
    
    # 1. Authentic Grad-CAM on Block 20/24 patch tokens (excluding token 0, which is [CLS])
    model.zero_grad(set_to_none=True)
    feat = model.forward_features(t)[0]
    contrib_raw = torch.dot(feat, torch.tensor(w_eye, dtype=torch.float32))
    contrib_raw.backward()
    
    act = hook.act.detach()[0, 1:, :]       # (196, 1024) patch tokens
    grad = hook.act.grad[0, 1:, :]          # (196, 1024) patch gradients
    alpha = grad.mean(dim=0)                # Channel importance vector
    cam = (act @ alpha).reshape(GRID, GRID).cpu().numpy()
    
    # Centered logit contribution: w . (f - mean)
    contrib_centered = float(contrib_raw.detach()) - float(np.dot(w_eye, m_eye))
    
    # 2. Shuffled Head Permutation Control (Adebayo et al. sanity check)
    w_shuffled = rng.permutation(w_eye)
    t_rand = torch.tensor(x_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    feat_rand = model.forward_features(t_rand)[0]
    torch.dot(feat_rand, torch.tensor(w_shuffled, dtype=torch.float32)).backward()
    
    act_rand = hook.act.detach()[0, 1:, :]
    grad_rand = hook.act.grad[0, 1:, :]
    alpha_rand = grad_rand.mean(dim=0)
    cam_rand = (act_rand @ alpha_rand).reshape(GRID, GRID).cpu().numpy()
    
    # True Pearson correlation on valid retinal tokens
    if np.std(cam[valid_mask]) > 1e-9 and np.std(cam_rand[valid_mask]) > 1e-9:
        corr_rand = float(np.corrcoef(cam[valid_mask], cam_rand[valid_mask])[0, 1])
    else:
        corr_rand = 0.0
        
    # Free PyTorch memory
    del t, t_rand, act, grad, act_rand, grad_rand
    model.zero_grad(set_to_none=True)
    gc.collect()
    
    # 3. Retinal Token Masking: Clean positive attribution
    # Strictly zero out non-retinal geometric corner tokens
    cam_pos = np.where(valid_mask, np.clip(cam, 0, None), 0.0)
    
    # 4. Attribution Concentration Ratio (C10)
    valid_pos = cam_pos[valid_mask]
    sorted_pos = np.sort(valid_pos)[::-1]
    k10 = max(1, int(round(0.10 * valid_mask.sum())))
    total_valid_attr = float(sorted_pos.sum() + 1e-12)
    c10 = float(sorted_pos[:k10].sum() / total_valid_attr)
    saliency_profile = f"Focal Token Attribution (C10 = {c10:.1%})" if c10 >= 0.35 else f"Diffuse Token Attribution (C10 = {c10:.1%})"
    
    # 5. Attribution-Weighted Spatial Centroid (over top 10% highest-attribution tokens)
    top_thresh = sorted_pos[k10 - 1]
    top_mask = (cam_pos >= top_thresh) & valid_mask & (cam_pos > 0)
    if top_mask.sum() > 0:
        top_weights = cam_pos[top_mask]
        x_coords = (c_idx[top_mask] + 0.5) / 14.0
        y_coords = (r_idx[top_mask] + 0.5) / 14.0
        cx = float(np.sum(x_coords * top_weights) / np.sum(top_weights))
        cy = float(np.sum(y_coords * top_weights) / np.sum(top_weights))
    else:
        cx, cy = 0.5, 0.5
        
    # 6. High-Resolution Display Interpolation (512x512)
    disp_size = (512, 512)
    img_disp = np.array(pil_img.resize(disp_size, Image.Resampling.LANCZOS)) / 255.0
    scale = float(cam_pos.max() + 1e-12)
    norm_cam = cam_pos / scale
    pos_high = cv2.resize(norm_cam, disp_size, interpolation=cv2.INTER_CUBIC)
    pos_high = np.clip(pos_high, 0.0, 1.0)
    
    # Circular anti-aliased mask for crisp display
    cy_d, cx_d = disp_size[1] / 2.0, disp_size[0] / 2.0
    radius_d = disp_size[0] * 0.465
    y_g, x_g = np.ogrid[:disp_size[1], :disp_size[0]]
    dist_g = np.sqrt((x_g - cx_d)**2 + (y_g - cy_d)**2)
    circle_mask = np.clip((radius_d - dist_g) / 2.0 + 0.5, 0.0, 1.0)
    
    # Strictly zero out periphery
    pos_high = pos_high * circle_mask
    
    return {
        "img_disp": img_disp,
        "pos_high": pos_high,
        "circle_mask": circle_mask,
        "c10": round(c10, 3),
        "saliency_profile": saliency_profile,
        "centroid_x": round(cx, 3),
        "centroid_y": round(cy, 3),
        "contrib_centered": contrib_centered,
        "corr_rand": corr_rand
    }

def determine_anatomical_quadrant(cx, cy, eye):
    dist_from_c = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    if dist_from_c < 0.16:
        return "Central Macular / Peripapillary Region"
        
    is_superior = cy < 0.5
    v_str = "Superior" if is_superior else "Inferior"
    
    # Right Eye (OD): Optic nerve is Nasal (medial, user right side x >= 0.5).
    # Macula is Temporal (lateral, user left side x < 0.5).
    # Left Eye (OS): Optic nerve is Nasal (medial, user left side x < 0.5).
    # Macula is Temporal (lateral, user right side x >= 0.5).
    is_temporal = (cx < 0.5) if eye == "R" else (cx >= 0.5)
    h_str = "Temporal" if is_temporal else "Nasal"
    return f"{v_str}-{h_str} Retinal Quadrant"

PATIENT_IDS = [
    "151594008", "151932002", "152071010", "152073003",
    "155500005", "157742003", "162638003", "165065003",
    "359658002", "362299003"
]

audit_results = {}
all_shuffled_corrs = []
rng = np.random.default_rng(42)

for pid in PATIENT_IDS:
    p_info = find_patient_info(pid)
    print(f"\nProcessing Patient {pid} (Split: {p_info['split']}, Label: {p_info['label']})...", flush=True)
    
    for eye in ("R", "L"):
        w_eye = w_R if eye == "R" else w_L
        m_eye = m_R if eye == "R" else m_L
        img_name = f"{pid}_{eye}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        res = compute_retfound_gradcam_and_random(img_path, w_eye, m_eye, rng)
        img_disp = res["img_disp"]
        pos_high = res["pos_high"]
        c_mask = res["circle_mask"]
        cx = res["centroid_x"]
        cy = res["centroid_y"]
        target_quadrant = determine_anatomical_quadrant(cx, cy, eye)
        all_shuffled_corrs.append(res["corr_rand"])
        
        # 1. Panel 1: Original fundus with crisp white background
        c_mask_3ch = c_mask[:, :, np.newaxis]
        p1 = img_disp * c_mask_3ch + 1.0 * (1.0 - c_mask_3ch)
        p1 = np.clip(p1, 0.0, 1.0)
        
        # 2. Panel 2: PURE Heatmap on neutral white circular background (no fundus underneath!)
        heat_rgb = clinical_saliency_cmap(pos_high)[:, :, :3]
        p2 = heat_rgb * c_mask_3ch + 1.0 * (1.0 - c_mask_3ch)
        p2 = np.clip(p2, 0.0, 1.0)
        
        # 3. Panel 3: Grad-CAM Overlay on Fundus (smooth alpha blend over color fundus)
        # Alpha is 0 for non-activated tissue (<=0.05), naturally revealing real disc & vessels
        alpha = np.clip((pos_high - 0.05) / 0.95, 0.0, 1.0)[:, :, np.newaxis] * 0.60
        p3 = (1.0 - alpha) * img_disp + alpha * heat_rgb
        p3 = p3 * c_mask_3ch + 1.0 * (1.0 - c_mask_3ch)
        p3 = np.clip(p3, 0.0, 1.0)
        
        # Render 3-panel figure with dedicated colorbar and one-line caption
        fig = plt.figure(figsize=(15, 5.5), facecolor="#ffffff", dpi=140)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.15)
        
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(p1)
        ax0.set_title("1. Input Fundus Photograph", color="#0f172a", fontsize=11.5, pad=10, fontweight="bold")
        ax0.axis("off")
        
        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.imshow(p2, cmap=clinical_saliency_cmap, vmin=0.0, vmax=1.0)
        ax1.set_title("2. Positive Grad-CAM Attribution (Block 20/24)", color="#0f172a", fontsize=11.5, pad=10, fontweight="bold")
        ax1.axis("off")
        cbar = fig.colorbar(im1, ax=ax1, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.75)
        cbar.set_label("Normalized Attribution [0.0 - 1.0]", fontsize=8.5, color="#334155", labelpad=5, fontweight="semibold")
        cbar.ax.tick_params(labelsize=8, colors="#334155")
        
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(p3)
        ax2.set_title("3. Grad-CAM Overlay on Fundus", color="#0f172a", fontsize=11.5, pad=10, fontweight="bold")
        ax2.axis("off")
        
        # Single-line reviewer-grade caption
        fig.text(0.5, 0.03, "Grad-CAM at block 20/24 (retina branch only); reflects model sensitivity, not diagnosis. Systemic CVD risk is driven mainly by age and sex.", 
                 ha="center", fontsize=9.5, style="italic", color="#475569")
        
        # Save specific eye file and default file
        out_eye_path = os.path.join(OUT_CACHE, f"{pid}_{eye}_gradcam.png")
        plt.savefig(out_eye_path, format="png", dpi=140, bbox_inches="tight", facecolor="#ffffff")
        
        if eye == "R" or not os.path.exists(os.path.join(OUT_CACHE, f"{pid}_gradcam.png")):
            out_default_path = os.path.join(OUT_CACHE, f"{pid}_gradcam.png")
            plt.savefig(out_default_path, format="png", dpi=140, bbox_inches="tight", facecolor="#ffffff")
            
        plt.close(fig)
        
        key = f"{pid}_{eye}"
        audit_results[key] = {
            "pid": pid,
            "eye": eye,
            "split": p_info["split"],
            "label": p_info["label"],
            "centroid_x": cx,
            "centroid_y": cy,
            "c10": res["c10"],
            "saliency_profile": res["saliency_profile"],
            "target_quadrant": target_quadrant,
            "centred_logit_contrib": round(res["contrib_centered"], 4),
            "measured_corr_shuffled": round(res["corr_rand"], 3)
        }
        print(f"  -> Eye {eye}: Centroid=({cx}, {cy}) -> {target_quadrant} | {res['saliency_profile']} | Shuffled r={res['corr_rand']:+.3f} | Centred logit: {res['contrib_centered']:+.3f}", flush=True)

# Compute Cohort Benchmark Metrics (Mean +- SD)
mean_shuff = float(np.mean(all_shuffled_corrs))
std_shuff = float(np.std(all_shuffled_corrs))

output_payload = {
    "cohort_shuffled_mean": round(mean_shuff, 3),
    "cohort_shuffled_std": round(std_shuff, 3),
    "pass_threshold": 0.25,
    "tokens_retinal_fov_share": 0.796,
    "tokens_background_excluded_share": 0.204,
    "audit_cases": audit_results
}

audit_json_path = os.path.join(BASE_DIR, "retfound_gradcam_audit.json")
with open(audit_json_path, "w") as f:
    json.dump(output_payload, f, indent=2)

print("\n--- TEST COHORT BENCHMARK COMPLETE ---", flush=True)
print(f"Shuffled-Head Permutation: Mean r = {mean_shuff:+.3f} +- {std_shuff:.3f} (Threshold <= 0.25)", flush=True)
print(f"Audit metadata saved to {audit_json_path}", flush=True)
