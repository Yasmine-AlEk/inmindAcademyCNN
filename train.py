import os  #filesystem utilities (paths, mkdir)
import yaml  #read config.yaml into a python dict
from tqdm import tqdm  #progress bar for batches/epochs
import random  #python rng for reproducibility
import numpy as np  #numpy rng for reproducibility

import torch  #tensors, autograd, cuda
import torch.nn as nn  #losses + module base classes
import torch.optim as optim  #optimizers (sgd, adamw)
from torch.utils.data import DataLoader, random_split  #batching + reproducible splitting
from torchvision import datasets, transforms  #cifar10 dataset + image transforms

from model import build_model  #model factory (keeps architecture separate)


with open("config.yaml", "r") as f:  #open the config file
    config = yaml.safe_load(f)  #parse yaml into dict

HP = config["hyperparameters"]  #short alias for hyperparameters


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)  #per-channel mean for normalization
CIFAR_STD  = (0.2023, 0.1994, 0.2010)  #per-channel std for normalization


class Cutout:  #augmentation: mask a random square patch on the tensor image
    """zero a random square region of a tensor image (c,h,w)."""
    def __init__(self, size=8, p=0.5):  #size is patch width/height; p is apply probability
        self.size = int(size)  #ensure indices are integers
        self.p = float(p)  #ensure probability is float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  #makes this usable inside transforms.compose
        if self.p <= 0:  #disabled case
            return x
        if torch.rand(1).item() > self.p:  #randomly skip cutout with probability (1-p)
            return x

        _, h, w = x.shape  #x is (c,h,w); keep h,w for bounds
        s = self.size
        if s <= 0:  #invalid size
            return x

        cy = torch.randint(0, h, (1,)).item()  #random center y
        cx = torch.randint(0, w, (1,)).item()  #random center x

        y1 = max(0, cy - s // 2)  #top boundary, clamped
        y2 = min(h, cy + s // 2)  #bottom boundary, clamped
        x1 = max(0, cx - s // 2)  #left boundary, clamped
        x2 = min(w, cx + s // 2)  #right boundary, clamped

        x[:, y1:y2, x1:x2] = 0.0  #set patch to zero (mask)
        return x


def build_transforms():  #build train and test/val preprocessing pipelines
    use_autoaugment = bool(HP.get("use_autoaugment", True))  #toggle strong augmentation
    cutout = Cutout(size=HP.get("cutout_size", 8), p=HP.get("cutout_prob", 0.5))  #cutout strength

    aug_list = [
        transforms.RandomCrop(32, padding=4),  #keep 32x32 but shift via padded crop
        transforms.RandomHorizontalFlip(),  #random left-right flip
    ]

    if use_autoaugment:
        transforms_policy = transforms.AutoAugmentPolicy.CIFAR10  #predefined cifar10 policy
        aug_list.append(transforms.AutoAugment(policy=transforms_policy))  #apply policy

    train_transform = transforms.Compose([
        *aug_list,  #apply geometric/color aug before tensor conversion
        transforms.ToTensor(),  #pil -> tensor float in [0,1] with shape (c,h,w)
        cutout,  #mask after to_tensor so we can index pixels precisely
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  #standardize inputs for stable optimization
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),  #no augmentation for fair metrics
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, test_transform


class EMA:  #exponential moving average of trainable parameters for evaluation
    """maintains a moving-average copy of parameters for smoother eval weights."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):  #decay close to 1 means slow, smooth average
        self.decay = float(decay)
        self.shadow = {}  #name -> ema tensor
        self.backup = {}  #name -> original tensor copy (for restore)

        for name, p in model.named_parameters():  #iterate named parameters for consistent mapping
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()  #start ema equal to initial weights

    @torch.no_grad()  #no gradients needed for ema math
    def update(self, model: nn.Module):  #call after optimizer step
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))  #ema = d*ema + (1-d)*w

    def store(self, model: nn.Module):  #save current weights before swapping in ema weights
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()

    def copy_to(self, model: nn.Module):  #overwrite model weights with ema weights
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name].data)  #copy into existing parameter storage

    def restore(self, model: nn.Module):  #restore original weights after eval
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name].data)
        self.backup = {}


def set_seed(seed: int):  #make randomness repeatable
    random.seed(seed)  #python rng
    np.random.seed(seed)  #numpy rng
    torch.manual_seed(seed)  #torch cpu rng
    torch.cuda.manual_seed_all(seed)  #torch gpu rng (all devices)


def get_loaders(train_transform, test_transform):  #create dataloaders for train/val/test
    data_root = config["paths"]["data_dir"]  #dataset path from config
    os.makedirs(data_root, exist_ok=True)  #ensure directory exists

    dataset_train_full = datasets.CIFAR10(
        root=data_root,
        train=True,  #use the 50k training images
        download=True,
        transform=train_transform,  #augmentation + normalize
    )

    dataset_test = datasets.CIFAR10(
        root=data_root,
        train=False,  #use the 10k test images
        download=True,
        transform=test_transform,  #no augmentation
    )

    val_split = float(HP.get("val_split", 0.1))  #fraction for validation
    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    dataset_train, dataset_val_idx = random_split(
        dataset_train_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(HP.get("seed", 42))),  #reproducible split
    )

    dataset_val_full = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=test_transform,  #clean validation (no augmentation)
    )
    dataset_val = torch.utils.data.Subset(dataset_val_full, dataset_val_idx.indices)  #same indices, different transform

    pin = bool(HP.get("pin_memory", True))  #faster cpu->gpu copies when using cuda
    nw = int(HP.get("num_workers", 0))  #loader worker processes

    train_loader = DataLoader(
        dataset_train,
        batch_size=int(HP["batch_size"]),
        shuffle=True,  #shuffle for better training
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),  #avoid worker restart cost
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=int(HP["batch_size"]),
        shuffle=False,  #deterministic eval
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=int(HP["batch_size"]),
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )

    return train_loader, val_loader, test_loader


@torch.no_grad()  #disable autograd for eval (saves memory/compute)
def evaluate(model, dataloader, criterion, device, tta: bool = False):  #compute average loss + accuracy
    model.eval()  #bn uses running stats; dropout off
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)  #non_blocking works with pinned memory
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)  #logits
        if tta:
            outputs_flip = model(torch.flip(images, dims=[3]))  #dims=[3] flips width -> horizontal flip
            outputs = 0.5 * (outputs + outputs_flip)  #average logits

        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)  #sum loss over samples
        preds = outputs.argmax(dim=1)  #predicted class index
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def make_optimizer(model):  #create optimizer from config
    opt_name = HP.get("optimizer", "sgd").lower()
    if opt_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=float(HP["lr"]),
            momentum=float(HP.get("momentum", 0.9)),
            weight_decay=float(HP.get("weight_decay", 5e-4)),
            nesterov=bool(HP.get("nesterov", True)),
        )
    elif opt_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=float(HP["lr"]),
            weight_decay=float(HP.get("weight_decay", 1e-2)),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def train(model, train_loader, val_loader, criterion, optimizer, device, ema: EMA | None):  #training loop + checkpointing
    epochs = int(HP["epochs"])
    best_val_acc = -1.0

    warmup_epochs = int(HP.get("warmup_epochs", 0))
    base_lr = float(HP["lr"])

    cosine_T = max(1, epochs - warmup_epochs)  #cosine length after warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_T)

    use_amp = bool(HP.get("amp", True)) and device.type == "cuda"  #amp only makes sense on cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  #prevents fp16 gradient underflow

    for epoch in range(epochs):
        model.train()  #bn updates stats; dropout on (if used)
        running_loss = 0.0

        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = base_lr * float(epoch + 1) / float(warmup_epochs)  #linear warmup
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  #set_to_none reduces memory writes

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if ema is not None:
                    ema.update(model)  #ema after optimizer update

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (i + 1), lr=optimizer.param_groups[0]["lr"])

        if epoch >= warmup_epochs:
            scheduler.step()  #cosine update per epoch after warmup

        train_loss = running_loss / len(train_loader)

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, tta=False)

        if ema is not None:
            ema.restore(model)

        print(f"Epoch {epoch+1}: Train loss {train_loss:.3f} | Val loss {val_loss:.3f} | Val acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(config["paths"]["best_model_path"]), exist_ok=True)

            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
                torch.save(model.state_dict(), config["paths"]["best_model_path"])
                ema.restore(model)
            else:
                torch.save(model.state_dict(), config["paths"]["best_model_path"])

            print(f"Saved best model (val acc {best_val_acc:.2f}%) -> {config['paths']['best_model_path']}")

    print("Finished Training")
    return best_val_acc


def main():  #orchestrates config, data, model, training, and testing
    set_seed(int(HP.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  #fastest conv algorithms for fixed sizes
        torch.backends.cuda.matmul.allow_tf32 = True  #allow tf32 for speed on rtx
        torch.backends.cudnn.allow_tf32 = True  #allow tf32 in cudnn convs
        try:
            torch.set_float32_matmul_precision("high")  #matmul precision policy (api depends on torch version)
        except Exception:
            pass

    train_transform, test_transform = build_transforms()
    train_loader, val_loader, test_loader = get_loaders(train_transform, test_transform)

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(HP.get("label_smoothing", 0.0)))
    optimizer = make_optimizer(model)

    ema = None
    if bool(HP.get("use_ema", True)):
        ema = EMA(model, decay=float(HP.get("ema_decay", 0.9999)))

    best_val = train(model, train_loader, val_loader, criterion, optimizer, device, ema)

    if os.path.exists(config["paths"]["best_model_path"]):
        model.load_state_dict(torch.load(config["paths"]["best_model_path"], map_location=device))
        print("Loaded best checkpoint for testing.")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, tta=bool(HP.get("tta", False)))
    print(f"Best Val acc: {best_val:.2f}% | Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%")

    os.makedirs(os.path.dirname(config["paths"]["model_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["paths"]["model_path"])
    print(f"Saved last model -> {config['paths']['model_path']}")

if __name__ == "__main__":
    main()