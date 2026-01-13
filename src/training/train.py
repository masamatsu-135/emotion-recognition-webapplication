import argparse
from pathlib import Path
import time
import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

#from .dataset import FER2013ResNetDataset, get_resnet_transforms
#from .model import get_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

try:
    from .dataset import FER2013ResNetDataset, get_resnet_transforms
    from .model import get_model
except ImportError:
    from dataset import FER2013ResNetDataset, get_resnet_transforms
    from model import get_model

def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(model, dataloader, criterion,
                    optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for imgs, labels in tqdm(dataloader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = (device.type == "cuda")
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(outputs, labels) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            use_amp = (device.type == "cuda")
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(outputs, labels) * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def save_checkpoint(model, output_dir, model_type, filename=None):

    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = model_type.lower()
    if filename is None:
        filename = f"best_{model_type}_fer2013.pth"
    
    path = output_dir / filename
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved best model to: {path}")


def get_class_names_from_dataset(dataset):
    names = None
    if (hasattr(dataset, 'classes')
        and isinstance(dataset.classes, (list, tuple))):
        names = list(dataset.classes)
    elif (hasattr(dataset, 'class_to_idx')
          and isinstance(dataset.class_to_idx, dict)):
        names = [
            k for k, _ in sorted(
                dataset.class_to_idx.items(),
                key=lambda kv: kv[1]
                )
                ]
    if not names:
        names = [
            'Angry', 'Disgust', 'Fear', 'Happy',
            'Sad', 'Surprise', 'Neutral'
            ]
    return names


def plot_learning_curves(history, out_png, out_csv):

    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Loss
    axs[0].plot(epochs, history['train_loss'], label='Train')
    axs[0].plot(epochs, history['val_loss'], label='Val')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Loss')
    # Accuracy
    axs[1].plot(epochs, history['train_acc'], label='Train')
    axs[1].plot(epochs, history['val_acc'], label='Val')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].set_title('Accuracy')
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    if out_csv is not None:
        df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc'],
        })
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)


def compute_confusion_matrix(model, dataloader, device):

    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='ConfMat', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            use_amp = (device.type == "cuda")
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(labels_all)
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(cm, class_names, out_png, out_csv):

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
        )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    if out_csv is not None:
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="data/raw/fer2013.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="models/checkpoints")

    # ResNet-only
    p.add_argument("--pretrained", action="store_true")
    return p.parse_args()


def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
    return device


def build_dataloaders(csv_path, batch_size, device, num_workers):
    train_ds = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="Training",
        transform=get_resnet_transforms(train=True),
    )
    val_ds = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="PublicTest",
        transform=get_resnet_transforms(train=False),
    )

    is_cuda = (device.type == "cuda")
    is_windows = (os.name == "nt")
    
    effective_workers = 0 if is_windows else (num_workers if is_cuda else 0)

    pin_memory = is_cuda
    persistent_workers = (effective_workers > 0)  # workerがいる時だけ True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader, train_ds, val_ds


def build_model_optimizer_criterion(args, device):
    model = get_model(num_classes=7, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    return model, criterion, optimizer


def run_train_loop(
        model, train_loader, val_loader, criterion,
        optimizer, device, epochs, output_dir
        ):
    best_val_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
            )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} "
            f"({time.time()-start:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

            save_checkpoint(model, output_dir, model_type="resnet")

    model.load_state_dict(best_wts)
    print(f"[INFO] Best Val Acc: {best_val_acc:.4f}")
    return model, history


def write_reports(
        model, history, val_ds, val_loader, device, output_dir
        ):
    reports_dir = (output_dir.parent / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(
        history,
        reports_dir / "learning_curve_resnet.png",
        reports_dir / "training_history_resnet.csv",
    )

    class_names = get_class_names_from_dataset(val_ds)
    cm = compute_confusion_matrix(model, val_loader, device)
    plot_confusion_matrix(
        cm,
        class_names,
        reports_dir / "confusion_matrix_resnet.png",
        reports_dir / "confusion_matrix_resnet.csv",
    )


def run_training(args):
    device = setup_device()

    project_root = Path(__file__).resolve().parents[2]
    csv_path = (project_root / args.csv_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        csv_path, args.batch_size, device, args.num_workers
    )
    model, criterion, optimizer = build_model_optimizer_criterion(
        args, device
        )

    model, history = run_train_loop(
        model, train_loader, val_loader, criterion,
        optimizer, device, args.epochs, output_dir
    )
    write_reports(model, history, val_ds, val_loader, device, output_dir)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()