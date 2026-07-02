import numpy as np

from ml.mlp.data import load_mnist, MNIST_CLASSES
from ml.mlp.model import MLP, SGD
from utils.config import load_config, save_config
from utils.seed import set_seed


def train():
    cfg = load_config("ml/mlp/config.yaml")
    set_seed(cfg["seed"])

    train_loader, test_loader = load_mnist(batch_size=cfg["batch_size"])

    model = MLP(cfg["layer_dims"])
    optimizer = SGD(model, lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    lr_decay_epochs = cfg["lr_decay_epochs"]
    lr_decay_factor = cfg["lr_decay_factor"]
    total_params = sum(p.size for p, _ in model.params())
    print(f"Model parameters: {total_params:,}")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels, labels_one_hot in train_loader():
            logits = model.forward(images)
            loss = model.compute_loss(logits, labels_one_hot)
            epoch_loss += loss
            model.backward(labels_one_hot)
            optimizer.step()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        correct = 0
        total = 0
        for images, labels, _ in test_loader():
            logits = model.forward(images)
            predictions = np.argmax(logits, axis=1)
            correct += (predictions == labels).sum()
            total += len(labels)

        accuracy = correct / total * 100
        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_loss:.4f}  Test Acc: {accuracy:.2f}%  "
              f"LR: {optimizer.lr:.2e}")

        if epoch in lr_decay_epochs:
            optimizer.set_lr(optimizer.lr / lr_decay_factor)

    print("\nTraining complete!")

    save_path = "ml/mlp/mlp_mnist.npz"
    weights = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            weights[f"layer_{i}_W"] = layer.W
            weights[f"layer_{i}_b"] = layer.b
    np.savez(save_path, **weights)
    save_config(cfg, save_path.replace(".npz", "_config.yaml"))
    print(f"Weights saved to {save_path}")


if __name__ == "__main__":
    train()
