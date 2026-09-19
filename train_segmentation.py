"""Train Attention U-Net++ tumor segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bus_pipeline.config import load_config
from bus_pipeline.datasets import SegmentationDataset, filter_records, load_metadata
from bus_pipeline.evaluation import evaluate_segmentation_model
from bus_pipeline.losses import BCEDiceLoss
from bus_pipeline.segmentation_models import build_segmentation_model
from bus_pipeline.utils import ensure_dir, get_device, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "cuda"))

    paths = config.get("paths", {})
    seg_cfg = config.get("segmentation", {})
    train_cfg = seg_cfg.get("training", {})
    data_cfg = config.get("data", {})
    checkpoint_dir = ensure_dir(paths.get("checkpoint_dir", "outputs/checkpoints"))
    log_dir = ensure_dir(paths.get("tensorboard_dir", "outputs/tensorboard") + "/segmentation")

    records = load_metadata(paths.get("metadata_csv", "data/metadata.csv"), root_dir=data_cfg.get("root_dir"))
    train_records = filter_records(records, "train")
    val_records = filter_records(records, "val")
    if not train_records or not val_records:
        raise ValueError("Metadata must contain train and val splits for segmentation training.")

    train_dataset = SegmentationDataset(
        train_records,
        image_size=seg_cfg.get("image_size", 256),
        preprocess_config=config.get("preprocessing", {}),
        augment=True,
        crop_rois=seg_cfg.get("crop_rois", True),
        roi_margin=config.get("localization", {}).get("roi_margin", 0.12),
    )
    val_dataset = SegmentationDataset(
        val_records,
        image_size=seg_cfg.get("image_size", 256),
        preprocess_config=config.get("preprocessing", {}),
        augment=False,
        crop_rois=seg_cfg.get("crop_rois", True),
        roi_margin=config.get("localization", {}).get("roi_margin", 0.12),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    model = build_segmentation_model(seg_cfg.get("model", {})).to(device)
    criterion = BCEDiceLoss(
        bce_weight=train_cfg.get("bce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(log_dir=str(log_dir))
    best_dice = -1.0

    for epoch in range(1, train_cfg.get("epochs", 80) + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"seg epoch {epoch}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)

        train_loss = running_loss / len(train_dataset)
        val_metrics = evaluate_segmentation_model(model, val_loader, device, threshold=seg_cfg.get("threshold", 0.5))
        scheduler.step(val_metrics["dice"])
        writer.add_scalar("loss/train", train_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        save_checkpoint(
            checkpoint_dir / "segmentation_last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics,
            extra={"config": config},
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(
                checkpoint_dir / "segmentation_best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                extra={"config": config},
            )
            save_json(val_metrics, checkpoint_dir / "segmentation_best_metrics.json")
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_dice={val_metrics['dice']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
