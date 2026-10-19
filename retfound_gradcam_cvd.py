#!/usr/bin/env python
"""
Grad-CAM for the RETFound retinal branch of the CIMT / CVD-risk pipeline.

WHAT THIS EXPLAINS
  The retina-only logistic-regression branch (StandardScaler + LogisticRegression,
  C=0.01, 2,048 inputs = left 1024 + right 1024), exactly as trained in
  audit_comprehensive_evaluation.py. It does NOT explain the final decision of the
  hierarchical model, which is ~95% age/gender. Label every figure accordingly.

WHAT IT FIXES (vs. the previous Grad-CAM script)
  1. Explains the CVD retinal head trained on the CIMT train split (not a MuReD DR head).
  2. Verifies, before any CAM is drawn, that the image preprocessing + pooling reproduce
     your STORED embeddings (cosine similarity). If they do not, it stops.
  3. Hooks the output of a transformer block and keeps its gradient directly
     (no register_full_backward_hook on LayerNorm).
  4. Normalises by max |CAM| inside the field of view, so a flat map stays flat
     (no min-max stretching that forces a red hot spot onto any image).
  5. Shows positive AND negative attribution (sign is relative to a baseline; not "protective").
  6. Discards the outermost token ring and any token not fully inside the retina (ViT border
     artifacts) and reports rim_share_of_attribution = share of attribution that was discarded.
  7. Head-randomisation sanity check: if the map barely changes when the classifier
     weights are shuffled, it is not explaining the classifier.

USAGE (Windows paths shown)
  python retfound_gradcam_cvd.py ^
      --retfound_dir "D:/retfound anti/RETFound_MAE" ^
      --ckpt "D:/retfound anti/RETFound_cfp_weights.pth" ^
      --embeds_dir "D:/retfound anti" ^
      --img_dir "D:/new pro/dataset ttv/cimt_images" ^
      --pids 151594008 151932002 ^
      --out_dir "D:/new pro/gradcam_out"

  retfound_dir must contain models_vit.py (the file you uploaded).
  Images are expected as <pid>_R.png and <pid>_L.png (change with --img_pattern).
"""
import argparse
import os
import sys

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

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
GRID = 14  # 224 / 16


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def fov_crop(rgb, thr=15, margin=0.02):
    """Crop to the circular field of view using a green-channel threshold."""
    ys, xs = np.where(rgb[..., 1] > thr)
    if len(ys) == 0:
        return rgb
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my, mx = int(margin * (y1 - y0)), int(margin * (x1 - x0))
    y0, x0 = max(0, y0 - my), max(0, x0 - mx)
    y1, x1 = min(rgb.shape[0] - 1, y1 + my), min(rgb.shape[1] - 1, x1 + mx)
    return rgb[y0:y1 + 1, x0:x1 + 1]


def preprocess(path, mode):
    """Returns (normalised tensor 1x3x224x224, uint8 224x224x3 display image)."""
    rgb = np.array(Image.open(path).convert("RGB"))
    if mode == "fovcrop":
        rgb = fov_crop(rgb)
    img224 = np.array(Image.fromarray(rgb).resize((224, 224)))  # PIL default resample
    x = (img224 / 255.0 - MEAN) / STD
    t = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return t, img224


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(retfound_dir, ckpt, pool_mean):
    """pool_mean=True  -> mean of patch tokens + fc_norm (global_pool=True)
       pool_mean=False -> final norm + CLS token       (global_pool=False)"""
    if retfound_dir not in sys.path:
        sys.path.insert(0, retfound_dir)
    import models_vit
    model = models_vit.vit_large_patch16(
        num_classes=1000, drop_path_rate=0.0, global_pool=pool_mean)
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    msg = model.load_state_dict(sd, strict=False)
    skip = ("head", "decoder", "mask_token")  # irrelevant MAE / classifier-head keys
    miss = [k for k in msg.missing_keys if not k.startswith(skip)]
    unexp = [k for k in msg.unexpected_keys if not k.startswith(skip)]
    print(f"  [pool={'mean' if pool_mean else 'cls'}] missing keys: {miss}  unexpected keys: {unexp}")
    model.eval()
    return model


class BlockHook:
    """Captures the output of one transformer block and keeps its gradient."""

    def __init__(self, model, layer_idx):
        self.act = None
        model.blocks[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.act = output
        if output.requires_grad:
            output.retain_grad()


def embed(model, x):
    with torch.no_grad():
        return model.forward_features(x)[0].cpu().numpy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ----------------------------------------------------------------------------
# Data / classifier head
# ----------------------------------------------------------------------------
def load_split(embeds_dir, name):
    return np.load(os.path.join(embeds_dir, f"cimt_{name}_embeds.npz"), allow_pickle=True)


def find_patient(embeds_dir, pid):
    for split in ("test", "val", "train"):
        d = load_split(embeds_dir, split)
        ids = np.array([str(int(p)) for p in d["pids"]])
        hit = np.where(ids == str(int(pid)))[0]
        if len(hit):
            i = hit[0]
            return dict(split=split, label=int(d["label"][i]),
                        left=d["left_embeds"][i].astype(np.float32),
                        right=d["right_embeds"][i].astype(np.float32))
    raise KeyError(f"patient {pid} not found in train/val/test npz files")


def fit_retina_head(embeds_dir):
    """Same recipe as audit_comprehensive_evaluation.py (retina branch)."""
    d = load_split(embeds_dir, "train")
    X = np.concatenate([d["left_embeds"], d["right_embeds"]], axis=1).astype(np.float32)
    y = d["label"]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=1000, random_state=42).fit(sc.transform(X), y)
    w = (clf.coef_[0] / sc.scale_).astype(np.float32)
    b = float(clf.intercept_[0] - np.sum(clf.coef_[0] * sc.mean_ / sc.scale_))
    return w, b, sc, clf


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
def gradcam(model, hook, x, w_eye):
    """Signed Grad-CAM (14x14) for this eye's contribution w_eye . f(x) to the logit."""
    x = x.clone().requires_grad_(True)  # needed so the graph exists with frozen weights
    model.zero_grad(set_to_none=True)
    feat = model.forward_features(x)[0]
    contrib = torch.dot(feat, torch.tensor(w_eye, dtype=torch.float32))
    contrib.backward()
    act = hook.act.detach()[0, 1:, :]       # (196, 1024) patch tokens
    grad = hook.act.grad[0, 1:, :]          # (196, 1024)
    alpha = grad.mean(dim=0)                # channel weights
    cam = (act @ alpha).reshape(GRID, GRID).cpu().numpy()
    return cam, float(contrib.detach())


def fov_masks(img224):
    """(full field of view, eroded interior). Erosion treats the image frame as outside (borderValue=0),
    otherwise the mask leaks where the retinal circle touches the frame."""
    full = (img224.mean(axis=-1) > 13).astype(np.uint8)
    inner = cv2.erode(full, np.ones((15, 15), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return full.astype(np.float32), inner.astype(np.float32)


def token_valid(full_mask):
    """14x14 bool grid: token is valid only if its whole 16x16 patch lies inside the field of view
    AND it is not in the outermost token ring (where ViT border artifacts concentrate)."""
    cover = full_mask.reshape(GRID, 16, GRID, 16).mean(axis=(1, 3))
    valid = cover > 0.999
    valid[0, :] = valid[-1, :] = False
    valid[:, 0] = valid[:, -1] = False
    return valid


def upsample(cam):
    return cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)


def overlay(img224, heat01):
    """Blend jet colours into the image in proportion to heat (zero heat = untouched image)."""
    heat01 = np.clip(heat01, 0, 1)
    h = matplotlib.colormaps["jet"](heat01)[..., :3]
    a = 0.6 * heat01[..., None]
    return np.clip((1 - a) * img224 / 255.0 + a * h, 0, 1)


# ----------------------------------------------------------------------------
# Auto-selection of preprocessing / pooling by reproducing stored embeddings
# ----------------------------------------------------------------------------
def choose_config(args, check_pids):
    pools = [True, False] if args.pool == "auto" else [args.pool == "mean"]
    preps = ["resize", "fovcrop"] if args.prep == "auto" else [args.prep]
    results = []
    for pm in pools:
        print(f"Building model pool={'mean' if pm else 'cls'} ...")
        model = build_model(args.retfound_dir, args.ckpt, pm)
        for prep in preps:
            cs = []
            for pid in check_pids:
                p = find_patient(args.embeds_dir, pid)
                for eye, key in (("L", "left"), ("R", "right")):
                    path = os.path.join(args.img_dir, args.img_pattern.format(pid=pid, eye=eye))
                    x, _ = preprocess(path, prep)
                    cs.append(cosine(embed(model, x), p[key]))
            results.append((float(np.mean(cs)), pm, prep, model))
            print(f"  pool={'mean' if pm else 'cls'}  prep={prep:8s}  mean cosine vs stored embeddings = {np.mean(cs):.4f}")
    results.sort(key=lambda r: r[0], reverse=True)
    best = results[0]
    if best[0] < args.min_cosine and not args.force:
        sys.exit(
            f"\nSTOP: best cosine {best[0]:.4f} < {args.min_cosine}. The embedding extraction used a different "
            "preprocessing/pooling/checkpoint than this script, so a Grad-CAM through this model would not "
            "explain your stored-embedding classifier. Check your extraction script (resize method, FOV crop, "
            "pooling), then re-run. (--force overrides, results would then be unreliable.)")
    return best[1], best[2], best[3], best[0]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retfound_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embeds_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--pids", nargs="+", type=int, required=True)
    ap.add_argument("--out_dir", default="gradcam_out")
    ap.add_argument("--img_pattern", default="{pid}_{eye}.png")
    ap.add_argument("--layer", type=int, default=-4, help="block index to explain (-4 = block 20/24; avoid -1 when using CLS pooling due to zero patch gradients post-block 23)")
    ap.add_argument("--pool", choices=["auto", "mean", "cls"], default="auto")
    ap.add_argument("--prep", choices=["auto", "resize", "fovcrop"], default="auto")
    ap.add_argument("--min_cosine", type=float, default=0.98)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    print("== Step 1: reproduce stored embeddings to pick preprocessing / pooling ==")
    pool_mean, prep, model, cos_best = choose_config(args, args.pids[:3])
    print(f"-> using pool={'mean' if pool_mean else 'cls'}, prep={prep}, cosine={cos_best:.4f}\n")

    print("== Step 2: fit the retina-only head on the TRAIN split ==")
    w, b, sc, clf = fit_retina_head(args.embeds_dir)
    w_L, w_R = w[:1024], w[1024:]
    hook = BlockHook(model, args.layer)

    rows = []
    for pid in args.pids:
        p = find_patient(args.embeds_dir, pid)
        # check that the folded head equals sklearn on the stored embeddings
        stored = np.concatenate([p["left"], p["right"]])[None]
        logit_sk = float(clf.decision_function(sc.transform(stored))[0])
        logit_fold = float(w @ stored[0] + b)
        assert abs(logit_sk - logit_fold) < 1e-3, "folded head != sklearn head"

        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        for r, (eye, w_eye, key) in enumerate((("R", w_R, "right"), ("L", w_L, "left"))):
            path = os.path.join(args.img_dir, args.img_pattern.format(pid=pid, eye=eye))
            x, img224 = preprocess(path, prep)
            cam, contrib_raw = gradcam(model, hook, x, w_eye)
            # centre the contribution: w.(f - mean) so that  L + R + intercept = retina logit
            m_eye = sc.mean_[1024:] if eye == "R" else sc.mean_[:1024]
            contrib = contrib_raw - float(np.dot(w_eye, m_eye))

            full_mask, _ = fov_masks(img224)
            mask = full_mask
            valid = token_valid(full_mask)
            # share of |attribution| that sat on border / partial tokens (discarded below)
            rim_share = float(np.abs(cam)[~valid].sum() / (np.abs(cam).sum() + 1e-12))
            cam_up = upsample(np.where(valid, cam, 0.0)) * mask
            scale = np.abs(cam_up).max() + 1e-12      # shared scale, no min-max stretching
            pos = np.clip(cam_up, 0, None) / scale
            neg = np.clip(-cam_up, 0, None) / scale

            # sanity: randomise the classifier weights
            cam_rand, _ = gradcam(model, hook, x, rng.permutation(w_eye))
            corr_rand = float(np.corrcoef(cam[valid], cam_rand[valid])[0, 1]) if (np.std(cam[valid]) > 1e-9 and np.std(cam_rand[valid]) > 1e-9) else 0.0  # valid tokens only

            # summary statistics of the positive-evidence map
            ys, xs = np.mgrid[0:224, 0:224]
            m_idx = mask > 0.5
            cy, cx = ys[m_idx].mean(), xs[m_idx].mean()
            rad = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / (np.sqrt(m_idx.sum() / np.pi) + 1e-9)
            pos_mass = pos.sum() + 1e-12
            peak = np.unravel_index(np.argmax(pos), pos.shape)
            rows.append(dict(
                patient_id=pid, eye=eye, split=p["split"], label=p["label"],
                eye_logit_contribution=round(contrib, 4),
                total_retina_logit=round(logit_fold, 4),
                pos_peak_x=round(peak[1] / 224, 3), pos_peak_y=round(peak[0] / 224, 3),
                pos_mass_central_half=round(float(pos[rad < 0.5].sum() / pos_mass), 3),
                rim_share_of_attribution=round(rim_share, 3),
                corr_with_shuffled_head=round(corr_rand, 3)))

            axes[r, 0].imshow(img224); axes[r, 0].set_title(f"{pid}_{eye} (model input)")
            axes[r, 1].imshow(overlay(img224, pos))
            axes[r, 1].set_title(f"Positive attribution (eye contribution {contrib:+.2f})\nborder tokens discarded: {rim_share:.0%} of attribution")
            axes[r, 2].imshow(overlay(img224, neg))
            axes[r, 2].set_title("Negative attribution\n(sign is relative to a baseline, not 'protective')")
            for a in axes[r]:
                a.axis("off")
            axes[r, 1].text(2, 218, f"corr(shuffled head)={corr_rand:.2f}", color="white", fontsize=9)

        fig.suptitle(
            f"Patient {pid}  |  true label={p['label']}  |  Grad-CAM of the RETINA-ONLY branch "
            f"(block {args.layer}, 14x14 patch grid)\n"
            "Exploratory: explains the retinal model only; final model is dominated by age/gender. "
            "No anatomical validation.", fontsize=11)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"gradcam_cvd_{pid}.png")
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

    df = pd.DataFrame(rows)
    csv = os.path.join(args.out_dir, "gradcam_summary.csv")
    df.to_csv(csv, index=False)
    print(f"\nsaved {csv}\n")
    print(df.to_string(index=False))
    print("\nHow to read corr_with_shuffled_head: values near 1 mean the map barely depends on the "
          "classifier weights (it is NOT explaining the classifier); values near 0 or below mean it does.")


if __name__ == "__main__":
    main()