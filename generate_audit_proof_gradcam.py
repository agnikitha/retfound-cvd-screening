import os
import sys
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

print("Loading RETFound ViT-Large backbone...")
model = models_vit.vit_large_patch16(num_classes=1000, drop_path_rate=0.0, global_pool=False)
sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = sd.get("model", sd)
model.load_state_dict(sd, strict=False)
model.eval()
print("RETFound ViT loaded successfully.")

# 2. Hook Layer 20 (Block index 20 of 24)
class BlockHook:
    def __init__(self, model, layer_idx=20):
        self.act = None
        model.blocks[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.act = output
        if output.requires_grad:
            output.retain_grad()

hook = BlockHook(model, 20)

# 3. Fit Logistic Regression Head on TRAIN split
def load_split(name):
    return np.load(os.path.join(EMBEDS_DIR, f"cimt_{name}_embeds.npz"), allow_pickle=True)

print("Fitting authentic retina classifier head on train split...")
d_train = load_split("train")
X_tr = np.concatenate([d_train["left_embeds"], d_train["right_embeds"]], axis=1).astype(np.float32)
y_tr = d_train["label"]
sc = StandardScaler().fit(X_tr)
clf = LogisticRegression(C=0.01, max_iter=1000, random_state=42).fit(sc.transform(X_tr), y_tr)
w = (clf.coef_[0] / sc.scale_).astype(np.float32)
b = float(clf.intercept_[0] - np.sum(clf.coef_[0] * sc.mean_ / sc.scale_))
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
                "label": int(d["label"][i]),
                "left": d["left_embeds"][i].astype(np.float32),
                "right": d["right_embeds"][i].astype(np.float32)
            }
    return None

def fov_masks(img224):
    full = (img224.mean(axis=-1) > 13).astype(np.uint8)
    inner = cv2.erode(full, np.ones((15, 15), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return full.astype(np.float32), inner.astype(np.float32)

def token_valid(full_mask):
    cover = full_mask.reshape(GRID, 16, GRID, 16).mean(axis=(1, 3))
    valid = cover > 0.999
    valid[0, :] = valid[-1, :] = False
    valid[:, 0] = valid[:, -1] = False
    return valid

def determine_anatomical_quadrant(norm_x, norm_y, eye):
    """
    Anatomically correct quadrant mapping:
    - Superior: y < 0.5, Inferior: y >= 0.5
    - Right Eye (OD): Optic nerve is Nasal (medial, user right side of photo x >= 0.5).
      Macula/Fovea is Temporal (lateral, user left side of photo x < 0.5).
    - Left Eye (OS): Optic nerve is Nasal (medial, user left side of photo x < 0.5).
      Macula/Fovea is Temporal (lateral, user right side of photo x >= 0.5).
    """
    dist_from_c = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
    if dist_from_c < 0.16:
        return "Central Macular / Peripapillary Region"
        
    is_superior = norm_y < 0.5
    v_str = "Superior" if is_superior else "Inferior"
    
    if eye == "R":
        is_temporal = norm_x < 0.5
    else:
        is_temporal = norm_x >= 0.5
        
    h_str = "Temporal" if is_temporal else "Nasal"
    target = f"{v_str}-{h_str} Retinal Quadrant"
    return target

def compute_retfound_gradcam_and_random(img_path, w_eye, m_eye, rng):
    pil_img = Image.open(img_path).convert("RGB")
    
    # 224x224 input for ViT
    img224 = np.array(pil_img.resize((224, 224), Image.Resampling.LANCZOS))
    x_norm = (img224 / 255.0 - MEAN) / STD
    t = torch.tensor(x_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)
    
    # 1. Authentic Grad-CAM
    model.zero_grad(set_to_none=True)
    feat = model.forward_features(t)[0]
    contrib_raw = torch.dot(feat, torch.tensor(w_eye, dtype=torch.float32))
    contrib_raw.backward()
    
    act = hook.act.detach()[0, 1:, :]       # (196, 1024)
    grad = hook.act.grad[0, 1:, :]          # (196, 1024)
    alpha = grad.mean(dim=0)                # channel importance
    cam = (act @ alpha).reshape(GRID, GRID).cpu().numpy()
    
    # Properly centred logit contribution: w . (f - mean)
    contrib_centered = float(contrib_raw.detach()) - float(np.dot(w_eye, m_eye))
    
    # FOV mask & valid interior tokens
    full_mask, _ = fov_masks(img224)
    valid = token_valid(full_mask)
    rim_share = float(np.abs(cam)[~valid].sum() / (np.abs(cam).sum() + 1e-12))
    
    # 2. Authentic Shuffled Head Sanity Check (Permutation of weights)
    w_shuffled = rng.permutation(w_eye)
    t_rand = torch.tensor(x_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    feat_rand = model.forward_features(t_rand)[0]
    contrib_rand = torch.dot(feat_rand, torch.tensor(w_shuffled, dtype=torch.float32))
    contrib_rand.backward()
    
    act_rand = hook.act.detach()[0, 1:, :]
    grad_rand = hook.act.grad[0, 1:, :]
    alpha_rand = grad_rand.mean(dim=0)
    cam_rand = (act_rand @ alpha_rand).reshape(GRID, GRID).cpu().numpy()
    
    # Compute true Pearson correlation between real and shuffled CAM
    if np.std(cam[valid]) > 1e-9 and np.std(cam_rand[valid]) > 1e-9:
        corr_rand = float(np.corrcoef(cam[valid], cam_rand[valid])[0, 1])
    else:
        corr_rand = 0.0
    
    # Clean upsampled CAM (224x224)
    cam_clean = np.where(valid, cam, 0.0)
    cam_up = cv2.resize(cam_clean, (224, 224), interpolation=cv2.INTER_LINEAR) * full_mask
    scale = np.abs(cam_up).max() + 1e-12
    pos = np.clip(cam_up, 0, None) / scale
    
    # Upsample to high resolution for display (512x512)
    disp_size = (512, 512)
    img_disp = np.array(pil_img.resize(disp_size, Image.Resampling.LANCZOS)) / 255.0
    pos_high = cv2.resize(pos, disp_size, interpolation=cv2.INTER_CUBIC)
    pos_high = np.clip(pos_high, 0.0, 1.0)
    
    # Circular anti-aliased mask for display
    cy, cx = disp_size[1] / 2.0, disp_size[0] / 2.0
    radius = disp_size[0] * 0.465
    y_idx, x_idx = np.ogrid[:disp_size[1], :disp_size[0]]
    dist_from_c = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    circle_mask = np.clip((radius - dist_from_c) / 2.0 + 0.5, 0.0, 1.0)[:, :, np.newaxis]
    
    return {
        "img_disp": img_disp,
        "pos_high": pos_high,
        "circle_mask": circle_mask,
        "contrib_centered": contrib_centered,
        "rim_share": rim_share,
        "corr_rand": corr_rand
    }

PATIENT_IDS = [
    "151594008", "151932002", "152071010", "152073003",
    "155500005", "157742003", "162638003", "165065003",
    "359658002", "362299003"
]

audit_results = {}
rng = np.random.default_rng(42)

for pid in PATIENT_IDS:
    p_info = find_patient_info(pid)
    print(f"\nProcessing Patient {pid} (Split: {p_info['split']}, True Label: {p_info['label']})...")
    
    for eye in ("R", "L"):
        w_eye = w_R if eye == "R" else w_L
        m_eye = m_R if eye == "R" else m_L
        img_name = f"{pid}_{eye}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  Warning: {img_path} not found, skipping.")
            continue
            
        res = compute_retfound_gradcam_and_random(img_path, w_eye, m_eye, rng)
        
        img_disp = res["img_disp"]
        pos_high = res["pos_high"]
        c_mask = res["circle_mask"]
        
        # 1. Panel 1: Original fundus with crisp white background
        p1 = img_disp * c_mask + 1.0 * (1.0 - c_mask)
        p1 = np.clip(p1, 0.0, 1.0)
        
        # 2. Panel 2: Vascular Attention Heatmap (Layer 20/24)
        jet_colors = plt.cm.jet(pos_high)[:, :, :3]
        gray = cv2.cvtColor((img_disp * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        gray_light = 0.30 + 0.70 * gray
        gray_3ch = np.stack([gray_light] * 3, axis=-1)
        
        alpha_cam = np.clip(pos_high * 1.25, 0.0, 0.92)[:, :, np.newaxis]
        p2 = (1.0 - alpha_cam) * gray_3ch + alpha_cam * jet_colors
        p2 = p2 * c_mask + 1.0 * (1.0 - c_mask)
        p2 = np.clip(p2, 0.0, 1.0)
        
        # 3. Panel 3: Clinical Anatomical Overlay
        alpha_overlay = np.clip(pos_high * 0.70, 0.0, 0.85)[:, :, np.newaxis]
        p3 = (1.0 - alpha_overlay) * img_disp + alpha_overlay * jet_colors
        p3 = p3 * c_mask + 1.0 * (1.0 - c_mask)
        p3 = np.clip(p3, 0.0, 1.0)
        
        # Peak coordinates in normalized (x, y)
        y_peak, x_peak = np.unravel_index(np.argmax(pos_high), pos_high.shape)
        h_d, w_d = pos_high.shape
        norm_y = round(float(y_peak / h_d), 3)
        norm_x = round(float(x_peak / w_d), 3)
        peak_val = round(float(pos_high.max()), 3)
        
        # Anatomically verified quadrant mapping (OD vs OS aware)
        target_quadrant = determine_anatomical_quadrant(norm_x, norm_y, eye)
        
        # Audit-compliant Saliency Profile (not claiming histological pathology)
        if peak_val >= 0.80:
            saliency_profile = "Focal Token Attribution"
        else:
            saliency_profile = "Diffuse Token Attribution"

        # Render 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#ffffff", dpi=140)
        axes[0].imshow(p1)
        axes[0].set_title("1. Input Fundus Photograph", color="#0f172a", fontsize=12, pad=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(p2)
        axes[1].set_title("2. Positive Grad-CAM Attribution (Block 20/24)", color="#0f172a", fontsize=12, pad=10, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(p3)
        axes[2].set_title("3. Grad-CAM Overlay on Fundus", color="#0f172a", fontsize=12, pad=10, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        
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
            "norm_x": norm_x,
            "norm_y": norm_y,
            "peak_intensity": peak_val,
            "target_quadrant": target_quadrant,
            "saliency_profile": saliency_profile,
            "centred_logit_contrib": round(res["contrib_centered"], 4),
            "rim_share_discarded": round(res["rim_share"], 3),
            "measured_corr_shuffled": round(res["corr_rand"], 3)
        }
        print(f"  -> Eye {eye}: peak at ({norm_x}, {norm_y}) = {target_quadrant} | Centred logit: {res['contrib_centered']:+.3f} | Shuffled head r: {res['corr_rand']:+.3f} | Rim discarded: {res['rim_share']:.1%}")

# Save JSON audit file
audit_json_path = os.path.join(BASE_DIR, "retfound_gradcam_audit.json")
with open(audit_json_path, "w") as f:
    json.dump(audit_results, f, indent=2)

print("\nAll 10 benchmark patients successfully processed!")
print(f"Audit metadata saved to {audit_json_path}")
