"""Train EfficientNet-B3 CBAM lesion classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bus_pipeline.classification_models import build_classifier
from bus_pipeline.config import load_config
from bus_pipeline.constants import CLASS_NAMES
from bus_pipeline.datasets import ClassificationDataset, filter_records, load_metadata
from bus_pipeline.evaluation import evaluate_classifier_model
from bus_pipeline.utils import ensure_dir, get_device, save_checkpoint, save_json, set_seed
from bus_pipeline.visualization import plot_confusion_matrix


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
    cls_cfg = config.get("classification", {})
    train_cfg = cls_cfg.get("training", {})
    data_cfg = config.get("data", {})
    checkpoint_dir = ensure_dir(paths.get("checkpoint_dir", "outputs/checkpoints"))
    log_dir = ensure_dir(paths.get("tensorboard_dir", "outputs/tensorboard") + "/classification")

    records = load_metadata(paths.get("metadata_csv", "data/metadata.csv"), root_dir=data_cfg.get("root_dir"))
    train_records = filter_records(records, "train")
    val_records = filter_records(records, "val")
    if not train_records or not val_records:
        raise ValueError("Metadata must contain train and val splits for classifier training.")

    train_dataset = ClassificationDataset(
        train_records,
        image_size=cls_cfg.get("image_size", 300),
        preprocess_config=config.get("preprocessing", {}),
        augment=True,
        crop_rois=cls_cfg.get("crop_rois", True),
        roi_margin=config.get("localization", {}).get("roi_margin", 0.12),
    )
    val_dataset = ClassificationDataset(
        val_records,
        image_size=cls_cfg.get("image_size", 300),
        preprocess_config=config.get("preprocessing", {}),
        augment=False,
        crop_rois=cls_cfg.get("crop_rois", True),
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

    model = build_classifier(cls_cfg.get("model", {})).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(log_dir=str(log_dir))
    best_f1 = -1.0

    for epoch in range(1, train_cfg.get("epochs", 60) + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"cls epoch {epoch}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)

        train_loss = running_loss / len(train_dataset)
        metrics = evaluate_classifier_model(model, val_loader, device, class_names=CLASS_NAMES)
        scheduler.step(metrics["macro_f1"])
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch)

        save_checkpoint(
            checkpoint_dir / "classifier_last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            extra={"config": config, "class_names": CLASS_NAMES},
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_checkpoint(
                checkpoint_dir / "classifier_best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                extra={"config": config, "class_names": CLASS_NAMES},
            )
            save_json(metrics, checkpoint_dir / "classifier_best_metrics.json")
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                CLASS_NAMES,
                checkpoint_dir / "classifier_best_confusion_matrix.png",
            )
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_f1={metrics['macro_f1']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
