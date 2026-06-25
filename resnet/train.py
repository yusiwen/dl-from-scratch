import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from resnet.data import CelebADataset, ATTRIBUTES
from resnet.model import resnet18


def train():
    # --- Device setup ---
    # Apple MPS (Metal Performance Shaders) accelerates training on Mac Silicon.
    # It uses the unified memory architecture (CPU+GPU share 48GB on M4 Max),
    # which means less data copying overhead compared to discrete GPUs.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "mps":
        print(f"  MPS is available on M4 (48GB, capped at ~24GB usage)")

    # Limit CPU threads to 4 to avoid resource contention on M4's efficiency cores.
    torch.set_num_threads(4)

    # --- Image preprocessing pipeline ---
    # Why Resize(256) + CenterCrop(224)?
    #   ResNet was designed for ImageNet which has 224x224 inputs. Resizing to
    #   256 first, then center-cropping to 224, adds small translation variance
    #   as a weak data augmentation (the crop center may shift slightly relative
    #   to the subject).
    # Why Normalize with these specific mean/std values?
    #   These are the ImageNet dataset statistics. Normalizing to zero mean and
    #   unit variance helps gradient descent converge faster by keeping activations
    #   in a well-behaved range. Using ImageNet stats is a common transfer learning
    #   practice even for CelebA.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load 1000 CelebA samples with the above transforms.
    full_dataset = CelebADataset(ATTRIBUTES, num_samples=1000, transform=transform)

    # Split 1000 samples into 800 train / 200 validation.
    # Why 800/200 instead of standard 80/20? With only 1000 total samples, 200
    # gives us a reasonably-sized validation set (~60 per attribute positive class).
    # The random seed (42) ensures reproducible splits across runs.
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [800, 200], generator=torch.Generator().manual_seed(42)
    )

    # DataLoader creates minibatches for training.
    # Why batch_size=128? For a 11M-parameter model on 224x224 images, 128 images
    # consume roughly 2-3GB of GPU memory for activations. M4's 48GB unified memory
    # can comfortably handle this.
    # Why num_workers=0? On macOS with HF datasets, multiprocessing workers
    # have pickling issues with datasets library objects. Single-process loading
    # is simpler and fast enough for batch_size=128.
    # Why pin_memory=True? Speeds up CPU->GPU transfer on CUDA. On MPS it's
    # silently unsupported (raises a warning but doesn't hurt).
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    model = resnet18(num_classes=len(ATTRIBUTES)).to(device)

    # BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy in one operation.
    # Why combine them? It's numerically more stable than applying sigmoid
    # manually then BCELoss, because the log-sum-exp trick avoids floating-point
    # overflow when logits have extreme values.
    # This is the standard loss for multi-label binary classification (each
    # of the 15 attributes is an independent binary prediction).
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer: adaptive learning rate with momentum.
    # Why Adam instead of SGD? Adam converges faster out-of-the-box, requires
    # less learning rate tuning, and works well with noisy gradients.
    # For a small 1000-sample dataset, the faster convergence is more important
    # than the slight generalization benefit of SGD + momentum.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Mixed precision training (AMP) ---
    # AMP (Automatic Mixed Precision) uses float16 for most computations while
    # keeping critical ops in float32. This can nearly double throughput on
    # supported hardware. On MPS, float16 is supported via Apple's Metal API.
    # GradScaler prevents underflow in float16 gradients by scaling the loss
    # before backward and unscaling the gradients before optimizer.step().
    scaler = None
    try:
        from torch.amp import autocast, GradScaler
        if device.type == "mps":
            scaler = GradScaler("mps")
        else:
            scaler = GradScaler()
        use_amp = True
    except (ImportError, RuntimeError):
        # Fallback to full float32 if AMP is not available.
        use_amp = False

    num_epochs = 50
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")
    print(f"Attributes: {ATTRIBUTES}")
    print(f"Using AMP: {use_amp}")
    print()

    # --- Training loop ---
    for epoch in range(1, num_epochs + 1):
        # ---- Training phase ----
        # model.train() enables dropout and batch norm's running statistics updates.
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients from the previous step.
            # Why zero_grad()? PyTorch accumulates gradients by default (useful
            # for RNNs and gradient accumulation), so we must reset them each step.
            optimizer.zero_grad()

            if use_amp:
                # autocast enables automatic mixed precision for this block.
                # Operations inside are computed in float16 when beneficial,
                # and float32 when numerical precision is critical (e.g. reductions).
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                # Scale loss up before backward() to prevent float16 underflow.
                scaler.scale(loss).backward()
                # Step optimizer with unscaled gradients.
                scaler.step(optimizer)
                # Update the scale factor for the next iteration.
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        # ---- Validation phase ----
        # model.eval() disables dropout and fixes batch norm statistics.
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        # torch.no_grad() disables gradient computation entirely.
        # Why? We don't need gradients for validation, and disabling them
        # saves memory and computation (no need to store intermediate activations).
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if use_amp:
                    with autocast(device_type=device.type):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Sigmoid -> threshold at 0.5 for binary prediction.
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total * 100

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"Val Acc: {accuracy:.2f}%")

        # MPS has a known issue with memory fragmentation in long training runs.
        # Emptying the cache periodically helps keep memory usage bounded.
        if device.type == "mps":
            torch.mps.empty_cache()

    # Save the full model (architecture + weights) for inference.
    # torch.save(model, ...) saves the entire module, so you can later
    # load it with torch.load() without needing to instantiate the class first.
    print("\nTraining complete!")
    torch.save(model, "resnet/resnet18_celeba.pt")
    print("Model saved to resnet/resnet18_celeba.pt")


if __name__ == "__main__":
    train()
