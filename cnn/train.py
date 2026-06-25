import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from cnn.data import load_cifar10, CIFAR10_CLASSES
from cnn.model import SimpleCNN


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    torch.set_num_threads(4)

    train_loader, test_loader = load_cifar10(batch_size=128, num_workers=2)

    model = SimpleCNN(num_classes=len(CIFAR10_CLASSES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # CrossEntropyLoss combines LogSoftmax + Negative Log-Likelihood Loss.
    # Why not use softmax + NLLLoss separately? CrossEntropyLoss integrates
    # softmax numerically: it uses the log-sum-exp trick internally to avoid
    # floating-point underflow/overflow when the input logits have large magnitudes.
    criterion = nn.CrossEntropyLoss()

    # Adam: adaptive learning rate method that keeps a running average of
    # both gradients and squared gradients. It works well out-of-the-box
    # with minimal tuning compared to SGD.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # CosineAnnealingLR: smoothly decreases the learning rate from the initial
    # value to near-zero following a half-cosine curve over T_max epochs.
    # Why cosine decay? Early in training, a higher LR helps escape poor local minima;
    # later, a lower LR allows fine-grained convergence. Cosine annealing provides
    # this transition without manual step boundaries.
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    num_epochs = 30

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)               # raw logits: (B, 10)
            loss = criterion(outputs, labels)      # CrossEntropyLoss applies softmax internally
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)   # index of the max logit = predicted class
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # Step the LR scheduler after each epoch
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # --- Evaluation (every epoch for monitoring) ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100

        print(
            f"Epoch [{epoch:2d}/{num_epochs}]  "
            f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
            f"Test Loss: {avg_test_loss:.4f}  Acc: {test_acc:.2f}%  |  "
            f"LR: {lr:.2e}"
        )

    print("\nTraining complete!")
    torch.save(model, "cnn/simple_cnn_cifar10.pt")
    print("Model saved to cnn/simple_cnn_cifar10.pt")


if __name__ == "__main__":
    train()
