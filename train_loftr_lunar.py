"""
Lunar LoFTR Fine-Tuning Entry Point (SIH26166).
Fine-tunes LoFTR on real and calibrated lunar data pairs with coarse/fine supervision.
Saves checkpoints to models/loftr/lunar_finetuned/best.ckpt.
"""
import sys
import os
import time
import argparse
import json
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import kornia.feature as kf

from src.training.dataset import (
    LunarDatasetScanner,
    LunarPairGenerator,
    PseudoGroundTruthGenerator,
    LunarLoFTRDataset
)
from src.training.loss import LoFTRLoss

def main():
    parser = argparse.ArgumentParser(description="Train / Fine-Tune LoFTR on Lunar Imagery")
    parser.add_argument("--data-root", type=str, default="./data/raw", help="Path to data directory")
    parser.add_argument("--config", type=str, default="./configs/loftr_lunar.yaml", help="Path to training config")
    parser.add_argument("--output", type=str, default="./models/loftr/lunar_finetuned", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "training": {"learning_rate": 1e-4, "num_epochs": 3, "batch_size": 1, "image_size": 512},
            "data": {"min_overlap_pct": 20.0}
        }

    t_cfg = config.get("training", {})
    num_epochs = args.epochs or t_cfg.get("num_epochs", 3)
    lr = args.lr or float(t_cfg.get("learning_rate", 1e-4))
    img_size = int(t_cfg.get("image_size", 512))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("       CHANDRAYAAN-2 / LUNAR LOFTR DOMAIN ADAPTATION & FINE-TUNING")
    print("=" * 80)
    print(f"[*] Execution Device : {device}")
    print(f"[*] Output Directory : {out_dir.resolve()}")
    print(f"[*] Epochs: {num_epochs} | Learning Rate: {lr}")

    # 1. Scan and Prepare Dataset
    data_root = Path(args.data_root)
    scanner = LunarDatasetScanner(data_root)
    items = scanner.scan()

    if not items and (Path("data/samples")).exists():
        print("[!] No raw data found in data/raw. Falling back to calibrated data/samples for training verification...")
        items = scanner.scan(Path("data/samples"))

    if not items:
        print("[!] ERROR: No lunar imagery found in data/raw or data/samples.")
        print("[!] Place real Chandrayaan-2 PDS4/GeoTIFF products into:")
        print("      - data/raw/ch2/ohrc/")
        print("      - data/raw/ch2/tmc2/")
        print("      - data/raw/ch2/iirs/")
        print("      - data/raw/lroc/")
        print("[*] Exiting cleanly as specified in SIH Section 37.")
        return 1

    pair_gen = LunarPairGenerator(min_overlap_pct=config.get("data", {}).get("min_overlap_pct", 20.0))
    pairs = pair_gen.generate_pairs(items, max_pairs=30)

    gt_gen = PseudoGroundTruthGenerator(min_inliers=4)
    supervised_pairs = []
    print(f"[*] Generating geometric pseudo-ground-truth across {len(pairs)} candidate scene pairs...")
    for p in pairs:
        res = gt_gen.derive_ground_truth(p)
        if res.get("ground_truth_status") == "geometry-derived" and len(res.get("correspondences", [])) >= 4:
            supervised_pairs.append(res)

    print(f"\n--- DATASET COMPOSITION ---")
    print(f"Total Available Pairs       : {len(pairs)}")
    print(f"Geometry-Supervised Pairs   : {len(supervised_pairs)}")

    if len(supervised_pairs) == 0:
        print("[!] Notice: No pairs met the minimum inlier consensus threshold.")
        print("[!] Training requires at least 1 verified pair. Exiting gracefully.")
        return 0

    # Train / Val split (80/20)
    n_total = len(supervised_pairs)
    n_train = max(1, int(n_total * 0.8))
    train_pairs = supervised_pairs[:n_train]
    val_pairs = supervised_pairs[n_train:] if n_train < n_total else supervised_pairs[:1]

    print(f"Train Pairs: {len(train_pairs)} | Validation Pairs: {len(val_pairs)}")

    train_ds = LunarLoFTRDataset(train_pairs, image_size=img_size, augment=True, is_training=True)
    val_ds = LunarLoFTRDataset(val_pairs, image_size=img_size, augment=False, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 2. Load Pretrained LoFTR
    print("\n[*] Initializing LoFTR with pretrained weights...")
    try:
        model = kf.LoFTR(pretrained="outdoor").to(device)
    except Exception:
        model = kf.LoFTR(pretrained=None).to(device)

    if args.resume and Path(args.resume).exists():
        print(f"[*] Resuming from checkpoint: {args.resume}")
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk.get("state_dict", chk))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = LoFTRLoss(coarse_weight=1.0, fine_weight=0.25)

    history = []
    best_val_loss = float("inf")

    print("\n[*] Beginning fine-tuning loop...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            s_img = batch["image0"].to(device)
            r_img = batch["image1"].to(device)
            kpts0 = batch["keypoints0"].squeeze(0).to(device)
            kpts1 = batch["keypoints1"].squeeze(0).to(device)

            loss, l_dict = loss_fn.forward_train_step(model, s_img, r_img, kpts0, kpts1)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                s_img = batch["image0"].to(device)
                r_img = batch["image1"].to(device)
                kpts0 = batch["keypoints0"].squeeze(0).to(device)
                kpts1 = batch["keypoints1"].squeeze(0).to(device)
                v_loss, _ = loss_fn.forward_train_step(model, s_img, r_img, kpts0, kpts1)
                val_loss += v_loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Save Checkpoint
        chk_payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "model_type": "LOFTR_LUNAR_FINETUNED",
            "device": str(device)
        }

        # Save last
        torch.save(chk_payload, out_dir / "last.ckpt")

        # Save best
        if avg_val_loss < best_val_loss or epoch == 1:
            best_val_loss = avg_val_loss
            torch.save(chk_payload, out_dir / "best.ckpt")
            print(f"  [+] Saved new best checkpoint to {out_dir / 'best.ckpt'}")

    # Save training log and metadata
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Fine-tuning complete. Checkpoints saved to: {out_dir.resolve()}")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
